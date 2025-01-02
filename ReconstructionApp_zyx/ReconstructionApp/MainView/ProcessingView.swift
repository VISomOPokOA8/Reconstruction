//
//  ProcessingView.swift
//  ReconstructionApp
//
//  Created by Yuxuan Zhang on 29/11/24.
//

import SwiftUI

struct ProcessingView: View {
    var body: some View {
        Image(systemName: "cube.transparent")
            .resizable()
            .aspectRatio(contentMode: .fit)
            .frame(width: 120)
            .foregroundStyle(.tertiary)
            .fontWeight(.ultraLight)
    }
}
