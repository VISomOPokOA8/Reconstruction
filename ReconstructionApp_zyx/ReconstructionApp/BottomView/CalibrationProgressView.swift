//
//  CalibrationProgressView.swift
//  ReconstructionApp
//
//  Created by Yuxuan Zhang on 7/12/24.
//

import SwiftUI

struct CalibrationProgressView: View {
    @ObservedObject var appDataModel: AppDataModel

    var body: some View {
        VStack {
            if appDataModel.state == .calibrating {
                ProgressView(value: appDataModel.progress)
                    .progressViewStyle(LinearProgressViewStyle())
                    .padding()
                Text("Calibration Progress: \(Int(appDataModel.progress * 100))%")
                    .font(.headline)
            } else if appDataModel.state == .calibration_viewing {
                Text("Calibration Complete!")
                    .font(.title)
                    .foregroundColor(.green)
            } else if appDataModel.state == .error {
                Text("Error: \(appDataModel.alertMessage)")
                    .foregroundColor(.red)
            }
        }
    }
}
