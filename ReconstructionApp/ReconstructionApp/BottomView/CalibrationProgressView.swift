//
//  CalibrationProgressView.swift
//  ReconstructionApp
//
//  Created by Yuxuan Zhang on 7/12/24.
//

import SwiftUI

struct CalibrationProgressView: View {
    @ObservedObject var dataModel: AppDataModel

    var body: some View {
        VStack {
            if dataModel.state == .calibration_viewing {
                Text("Calibration Complete!")
                    .font(.title)
                    .foregroundColor(.green)
            } else if dataModel.state == .calibrating {
                ProgressView(value: dataModel.calibrationProgress)
                    .progressViewStyle(LinearProgressViewStyle())
                    .padding()
                Text("Calibration Progress: \(Int(dataModel.calibrationProgress * 100))%")
                    .font(.headline)
            }
        }
    }
}
