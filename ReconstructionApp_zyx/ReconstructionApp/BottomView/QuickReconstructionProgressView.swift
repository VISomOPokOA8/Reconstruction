//
//  QuickReconstructionProgressView.swift
//  ReconstructionApp
//
//  Created by Yuxuan Zhang on 7/12/24.
//

import RealityKit
import SwiftUI
import os

private let logger = Logger(subsystem: ReconstructionApp.subsystem, category: "QuickReconstructionProgressView")

struct QuickReconstructionProgressView: View {
    @State private var firstModelFileURL: URL? = nil
    @State private var detailLevel: PhotogrammetrySession.Request.Detail = .medium
    @State private var processingComplete = false

    @Environment(AppDataModel.self) private var appDataModel: AppDataModel
    @State private var progress = 0.0
    @State private var estimatedRemainingTime: TimeInterval?
    @State private var currentRequestDetailLevel: PhotogrammetrySession.Request.Detail?
    @State private var isCancelling = false
    @State private var numProcessedModels = 1

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(statusMessage)

            HStack {
                if appDataModel.session != nil {
                    ProgressView(value: progress)
                        .task {
                            await getSessionOutput()
                        }
                } else {
                    ProgressView(value: 0)
                }

                Button {
                    logger.log("Cancel button is clicked!")
                    isCancelling = true
                    if let session = appDataModel.session {
                        logger.log("Cancelling the session...")
                        session.cancel()
                    } else {
                        logger.log("Canceled the session before processing has started. Going back to the settings view...")
                        appDataModel.state = .ready
                    }
                } label: {
                    Image(systemName: "xmark.circle.fill")
                }
                .buttonStyle(PlainButtonStyle())
            }

            Text(formattedEstimatedRemainingTime)
                .foregroundStyle(.secondary)
                .font(.caption)
        }
    }

    private var statusMessage: String {
        guard appDataModel.session != nil, let currentRequestDetailLevel = currentRequestDetailLevel else {
            return "Preparing images..."
        }
        let detailLevel = "\(String(describing: currentRequestDetailLevel))".capitalized
        return "Creating \(detailLevel) 3D Model..."
    }

    private var formattedEstimatedRemainingTime: String {
        let calculating = "Calculating..."
        guard let estimatedRemainingTime = estimatedRemainingTime else { return calculating }

        let formatter = DateComponentsFormatter()
        formatter.unitsStyle = .full
        formatter.allowedUnits = [.minute, .second]

        if let estimatedRemainingTime = formatter.string(from: estimatedRemainingTime) {
            return "About " + estimatedRemainingTime + " remaining"
        } else {
            return calculating
        }
    }

    func getSessionOutput() async {
        do {
            guard let session = appDataModel.session else { return }
            for try await output in session.outputs {
                switch output {
                case .requestProgress(let request, let fractionComplete):
                    requestProgress(request: request, fractionComplete: fractionComplete)

                case .requestProgressInfo(_, let progressInfo):
                    estimatedRemainingTime = progressInfo.estimatedRemainingTime

                case .requestError(_, let error):
                    logger.log("requestError received: \(error)")

                    appDataModel.state = isCancelling ? .ready : .error
                    appDataModel.alertMessage = isCancelling ? "" : "Reconstruction Failed"

                case .requestComplete(let request, _):
                    logger.log("requestComplete received: \(String(describing: output))")
                    requestComplete(request: request)

                case .processingComplete:
                    logger.log("processingComplete received")

                    if appDataModel.state != .error {
                        processingComplete = true
                    }

                default:
                    continue
                }
            }
        } catch {
            logger.error("Getting output failed with \(error)")
        }
    }

    // MARK: - helper functions
    func requestProgress(request: PhotogrammetrySession.Request, fractionComplete: Double) {
        progress = fractionComplete

        if case .modelFile(_, let detail, _) = request, currentRequestDetailLevel != detail {
            currentRequestDetailLevel = detail
        }
    }

    func requestComplete(request: PhotogrammetrySession.Request) {
        if case .modelFile(let url, _, _) = request {
            // 从 AppDataModel 获取 reconstructionModelURL
            firstModelFileURL = appDataModel.reconstructedModelURL
            numProcessedModels += 1

            // Update the app's state to show the model after completing the first request.
            appDataModel.state = .quick_reconstruction_viewing
        }
    }
}
