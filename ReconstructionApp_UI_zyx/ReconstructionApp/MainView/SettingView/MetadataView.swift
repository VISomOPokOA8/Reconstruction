//
//  MetadataView.swift
//  ReconstructionApp
//
//  Created by Yuxuan Zhang on 29/11/24.
//

import SwiftUI

struct MetadataView: View {
    @State private var showInfo = false

    var body: some View {
        Button {
            showInfo = true
        } label: {
            Image(systemName: "photo.badge.checkmark")
                .foregroundColor(.green)
                .frame(height: 15)
        }
        .buttonStyle(.plain)
        .popover(isPresented: $showInfo) {
            VStack(alignment: .leading) {
                Text("Image Metadata Found")
                    .foregroundStyle(.secondary)
                    .padding(.horizontal)
                    .padding(.top, 7)

                Divider()

                Text("Depth and Gravity Vector included in dataset.")
                    .padding([.horizontal, .bottom])
            }
            .font(.callout)
            .frame(width: 250)
        }
    }
}
