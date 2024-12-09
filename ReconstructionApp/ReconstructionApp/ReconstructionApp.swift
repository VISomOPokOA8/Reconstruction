//
//  main.swift
//  ReconstructionApp
//
//  Created by Yuxuan Zhang on 29/11/24.
//

import SwiftUI

struct ReconstructionApp: App {
    static let subsystem: String = "com.app.zyx.Reconstruction"

    var body: some Scene {
        Window("ObjectCaptureReconstruction", id: "main") {
            ContentView()
                .frame(width: 400, height: 360)
        }
        .windowResizability(.contentSize)
    }
}
