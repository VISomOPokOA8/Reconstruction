//
//  QuickReconstructionView.swift
//  ReconstructionApp
//
//  Created by Yuxuan Zhang on 6/12/24.
//

import RealityKit
import SwiftUI
import os

private let logger = Logger(subsystem: ReconstructionApp.subsystem, category: "ReconstructionView")

struct ReconstructionView: View {
    @Environment(AppDataModel.self) private var appDataModel: AppDataModel
    @State private var modelURL: URL?
        
    var body: some View {
        if let url = modelURL {
            RealityView { content in
                if let entity = try? await ModelEntity(contentsOf: url) {
                    content.add(entity)
                    content.cameraTarget = entity
                } else {
                    logger.warning("Couldn't load the model")
                }
            }
            .realityViewCameraControls(.orbit)
            .background(Color(red: 0.12, green: 0.12, blue: 0.12))
            .onAppear {
                modelURL = appDataModel.reconstructedModelURL
            }
            .onChange(of: appDataModel.reconstructedModelURL) { oldURL, newURL in
                modelURL = newURL
            }
        } else {
            EmptyView()
        }
    }
}
