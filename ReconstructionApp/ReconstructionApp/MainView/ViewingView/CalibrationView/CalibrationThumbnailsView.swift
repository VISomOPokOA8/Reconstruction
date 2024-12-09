//
//  CalibrationThumbnailsView.swift
//  ReconstructionApp
//
//  Created by Yuxuan Zhang on 5/12/24.
//

import SwiftUI
import AppKit

struct CalibrationThumbnailsView: View {
    let images: [NSImage]
    let selectedIndex: Int
    let onThumbnailTap: (Int) -> Void // Callback to notify the parent view

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack {
                ForEach(images.indices, id: \.self) { index in
                    ThumbnailItemView(
                        image: images[index],
                        isSelected: index == selectedIndex
                    )
                    .onTapGesture {
                        onThumbnailTap(index) // Notify parent view when tapped
                    }
                }
            }
            .padding(.horizontal)
        }
    }
}

struct ThumbnailItemView: View {
    let image: NSImage
    let isSelected: Bool

    var body: some View {
        Image(nsImage: image)
            .resizable()
            .aspectRatio(contentMode: .fit)
            .frame(width: 80, height: 80)
            .border(isSelected ? Color.blue : Color.clear, width: 2)
            .cornerRadius(8)
            .padding(4)
            .background(isSelected ? Color.blue.opacity(0.2) : Color.clear)
            .cornerRadius(10)
    }
}
