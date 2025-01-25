//
//  CalibrationView.swift
//  ReconstructionApp
//
//  Created by Yuxuan Zhang on 5/12/24.
//

import SwiftUI
import AppKit
import os

private let logger = Logger(subsystem: ReconstructionApp.subsystem, category: "CalibrationView")

struct CalibrationView: View {
    @ObservedObject var appDataModel: AppDataModel

    var body: some View {
        VStack {
            if let distortionCoefficients = appDataModel.distortionCoefficients {
                Text("Distortion Coefficients:")
                    .font(.headline)
                VStack(alignment: .leading, spacing: 5) {
                    Text("k1: \(distortionCoefficients[0])")
                    Text("k2: \(distortionCoefficients[1])")
                    Text("p1: \(distortionCoefficients[2])")
                    Text("p2: \(distortionCoefficients[3])")
                    Text("k3: \(distortionCoefficients[4])")
                }
                .padding()
            } else {
                Text("No Distortion Coefficients Available")
                    .foregroundColor(.red)
            }

            // Display thumbnails for calibrated images
            if let distortionImages = appDataModel.distortionImages {
                ScrollView(.horizontal) {
                    HStack {
                        ForEach(distortionImages, id: \ .self) { image in
                            Image(nsImage: image)
                                .resizable()
                                .scaledToFit()
                                .frame(width: 100, height: 100)
                                .padding(5)
                        }
                    }
                }
            } else {
                Text("No Calibrated Images Available")
                    .foregroundColor(.red)
            }
        }
        .navigationTitle("Calibration")
        .padding()
    }
}
