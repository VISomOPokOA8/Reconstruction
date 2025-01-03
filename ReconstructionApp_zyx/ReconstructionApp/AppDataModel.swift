//
//  AppDataModel.swift
//  ReconstructionApp
//
//  Created by Yuxuan Zhang on 29/11/24.
//

import RealityKit
import SwiftUI
import os
import AppKit

private let logger = Logger(subsystem: ReconstructionApp.subsystem, category: "AppDataModel")

@MainActor @Observable
class AppDataModel: NSObject, ObservableObject {
    // Models ////////////////////////////////////////////////////////////////
    @Published var progress: Double = 0.0
    @Published var isCalibrationComplete: Bool = false
    
    @Published var distortionCoefficients: [Double]?
    @Published var distortionImages: [NSImage]?

    private(set) var session: PhotogrammetrySession?

    private var pointcloud: PhotogrammetrySession.PointCloud?

    // URLs ////////////////////////////////////////////////////////////////
    @Published var calibrationImageFolder: URL?
    @Published var reconstructionImageFolder: URL?
    @Published var reconstructedModelURL: URL?

    @Published var dataFolder: URL? {
        didSet {
            if let dataFolder = dataFolder {
                logger.log("Data folder is set to \(dataFolder.path())")
                calibrationImageFolder = dataFolder.appendingPathComponent("calibration_images")
                reconstructionImageFolder = dataFolder.appendingPathComponent("reconstruction_images")
                reconstructedModelURL = dataFolder.appendingPathComponent("reconstruction.usdz")
            } else {
                logger.log("Data folder is set to nil")
                calibrationImageFolder = nil
                reconstructionImageFolder = nil
                reconstructedModelURL = nil
            }
        }
    }

    // System Message ////////////////////////////////////////////////////////////////
    var alertMessage: String = ""

    // States ////////////////////////////////////////////////////////////////
    enum State: Equatable {
        case ready
        case calibrating
        case calibration_viewing
        case quick_reconstructing
        case quick_reconstruction_viewing
        case gs_reconstructing
        case gs_reconstruction_viewing
        case error
    }

    @Published var state: State = .ready {
        didSet {
            logger.log("State is switched to \(String(describing: self.state))")
            if state == .ready {
                session = nil
            }
        }
    }

    // Request Number ////////////////////////////////////////////////////////////////
    var requestNum = 0

    // First-Level Functions ////////////////////////////////////////////////////////////////
    func startCalibration() {
        logger.log("Starting calibration...")
        
        guard let folderPath = calibrationImageFolder?.path else {
            DispatchQueue.main.async {
                self.alertMessage = "Calibration image folder is not set"
                self.state = .error
            }
            return
        }

        state = .calibrating
        progress = 0.0

        let wrapper = OpenCVWrapper()
        wrapper.delegate = self

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                // 调用校准方法
                let result = try OpenCVWrapper.calibrateCameraAndGetDistortion(folderPath)

                // 解析结果
                if let coefficients = result["coefficients"] as? [String: NSNumber],
                   let calibratedImages = result["calibratedImages"] as? [String] {
                    DispatchQueue.main.async {
                        // 更新标定系数
                        self.distortionCoefficients = coefficients.values.map { $0.doubleValue }
                        // 加载标定后的图像
                        self.distortionImages = calibratedImages.compactMap { NSImage(contentsOfFile: $0) }

                        self.isCalibrationComplete = true
                        self.state = .calibration_viewing
                        print("Calibration successful. Coefficients: \(coefficients)")
                    }
                } else {
                    DispatchQueue.main.async {
                        self.alertMessage = "Unexpected result format from calibration."
                        self.state = .error
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    self.alertMessage = "Calibration failed: \(error.localizedDescription)"
                    self.state = .error
                }
            }
        }
    }

    func startQuickReconstruction() async {
        logger.log("Starting quick reconstruction...")

        guard let dataFolder = dataFolder else {
            alertMessage = "Data folder is not selected"
            state = .error
            logger.info("\(self.alertMessage)")
            return
        }

        guard let reconstructionImageFolder = reconstructionImageFolder else {
            alertMessage = "Image folder is not selected"
            state = .error
            logger.info("\(self.alertMessage)")
            return
        }

        logger.log("Creating requests...")
        let requests = createRequests()
        guard !requests.isEmpty else {
            logger.log("Requests are not set")
            return
        }
        requestNum = requests.count

        state = .quick_reconstructing

        do {
            logger.log("Creating session...")
            session = try await createSession(imageFolder: reconstructionImageFolder)

            logger.log("Processing requests...")
            try session?.process(requests: requests)
        } catch {
            logger.warning("Creating session or processing requests failed")
            alertMessage = "\(error)"
            state = .error
        }
    }

    func startGSReconstruction() async {
        logger.log("Starting gaussian splatting reconstruction...")
    }

    func updateModelURL(newURL: URL) {
        reconstructedModelURL = newURL
    }

    // Second-Level Functions ////////////////////////////////////////////////////////////////
    func createRequests() -> [PhotogrammetrySession.Request] {
        guard let usdzModelURL = reconstructedModelURL else {
            logger.error("Reconstructed model URL is nil")
            return []
        }

        let detailLevel: PhotogrammetrySession.Request.Detail = .medium

        var requests: [PhotogrammetrySession.Request] = []
        requests.append(PhotogrammetrySession.Request.modelFile(url: usdzModelURL, detail: detailLevel))
        requests.append(PhotogrammetrySession.Request.pointCloud)
        requests.append(PhotogrammetrySession.Request.poses)

        return requests
    }

    private nonisolated func createSession(imageFolder: URL) async throws -> PhotogrammetrySession {
        var configuration: PhotogrammetrySession.Configuration
        configuration.meshPrimitive = .quad
        configuration.isObjectMaskingEnabled = false
        logger.log("Creating PhotogrammetrySession with \(String(describing: configuration))")
        return try PhotogrammetrySession(input: imageFolder, configuration: configuration)
    }
}

extension AppDataModel: OpenCVWrapperDelegate {
    nonisolated func calibrationProgress(_ progress: Double) {
        DispatchQueue.main.async {
            self.progress = progress
            print("Calibration progress: \(progress * 100)%")
        }
    }
}

extension PhotogrammetrySession.Error: @retroactive CustomStringConvertible {
    public var description: String {
        switch self {
        case .invalidImages:
            return "No valid images found in selected folder"
        case .invalidOutput:
            return "Cannot save to selected folder"
        case .insufficientStorage:
            return "Not enough disk space available to begin processing."
        @unknown default:
            logger.warning("Unknown Error case: \(self)")
            return "\(self)"
        }
    }
}

extension PhotogrammetrySession.Configuration.CustomDetailSpecification.TextureFormat: @retroactive Hashable {
    public func hash(into hasher: inout Hasher) {
        switch self {
        case .png: break
        case .jpeg(let compressionQuality):
            hasher.combine(compressionQuality)
        @unknown default:
            fatalError("Unknown texture format: \(self)")
        }
    }
}
