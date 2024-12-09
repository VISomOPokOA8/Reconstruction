//
//  BottomView.swift
//  ReconstructionApp
//
//  Created by Yuxuan Zhang on 7/12/24.
//

import SwiftUI
import os

private let logger = Logger.init(subsystem: ReconstructionApp.subsystem, category: "BottomView")

struct BottomView: View {
    @State private var appDataModel = AppDataModel()
    
    var body: some View {
        VStack {
            HStack {
                switch appDataModel.state {
                case .ready, .error:
                    EmptyView()
                case .calibrating:
                    CalibrationProgressView()
                case .calibration_viewing:
                    Text("Calibration completed. ")
                case .quick_reconstructing:
                    QuickReconstructionProgressView()
                case .quick_reconstruction_viewing:
                    Text("Quick reconstruction completed. ")
                case .gs_reconstructing:
                    GSReconstructionProgressView()
                case .gs_reconstruction_viewing:
                    Text("GS reconstruction completed. ")
                }
                
                Spacer()
                
                switch appDataModel.state {
                case .ready:
                    Button("Start Calibrating") {
                        logger.log("Calibration button clicked")
                        Task {
                            await appDataModel.startCalibration()
                        }
                    }
                    
                case .calibrating:
                    Button {
                        logger.log("Cancel button clicked")
                        logger.log("Going back to the settings view...")
                        appDataModel.state = .ready
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                    }
                    .buttonStyle(PlainButtonStyle())
                    
                case .calibration_viewing:
                    Button("Start Quick Reconstructing") {
                        logger.log("Quick reconstruction button clicked")
                        Task {
                            await appDataModel.startCalibration()
                        }
                    }
                    
                case .quick_reconstructing:
                    Button {
                        logger.log("Cancel button clicked")
                        logger.log("Going back to the calibration view...")
                        appDataModel.state = .calibration_viewing
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                    }
                    .buttonStyle(PlainButtonStyle())
                    
                case .quick_reconstruction_viewing:
                    Button("Start GS Reconstructing") {
                        logger.log("GS reconstruction button clicked")
                        Task {
                            await appDataModel.startCalibration()
                        }
                    }
                    
                case .gs_reconstructing:
                    Button {
                        logger.log("Cancel button clicked")
                        logger.log("Going back to the quick reconstruction view...")
                        appDataModel.state = .quick_reconstruction_viewing
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                    }
                    .buttonStyle(PlainButtonStyle())
                    
                case .gs_reconstruction_viewing, .error:
                    Button("New Reconstruction") {
                        appDataModel.state = .ready
                    }
                }
            }
            .padding()
        }
    }
}
