//
//  CalibrationView.swift
//  ReconstructionApp
//
//  Created by Yuxuan Zhang on 5/12/24.
//

import SwiftUI
import AppKit

struct CalibrationView: View {
    @Environment(AppDataModel.self) private var appDataModel: AppDataModel
    @State private var index: Int = 0
    private let image: NSImage?
    
    init() {}

    var body: some View {
        VStack {
            if let images = appDataModel.distortionImages {
                if let image {
                    Image(nsImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxWidth: .infinity, maxHeight: 300)
                        .padding()
                } else {
                    Text("No image selected")
                        .frame(maxWidth: .infinity, maxHeight: 300)
                        .background(Color.gray.opacity(0.2))
                        .padding()
                }
                
                Divider()
                
                CalibrationThumbnailsView(
                    images: images,
                    selectedIndex: index
                ) { selectedIndex in
                    index = selectedIndex
                }
            } else {
                EmptyView()
            }
        }
    }
}
