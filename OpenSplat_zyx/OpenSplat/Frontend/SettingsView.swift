//
//  SettingsView.swift
//  OpenSplat
//
//  Created by Yuxuan Zhang on 1/10/25.
//

import SwiftUI

struct SettingsView: View {
    @State private var projectRoot: String
    @State private var outputScene: String = "splat.ply"
    @State private var saveEvery: Int = -1
    @State private var validate: Bool
    @State private var valImage: String = "random"
    @State private var valRender: String = ""
    @State private var keepCrs: Bool
    
    @State private var downScaleFactor: Float = 1.0
    @State private var numIters: Int = 30000
    @State private var numDownscales: Int = 2
    @State private var resolutionSchedule: Int = 3000
    @State private var shDegree: Int = 3
    @State private var shDegreeInterval: Int = 1000
    @State private var ssimWeight: Float = 0.2
    @State private var refineEvery: Int = 100
    @State private var warmupLength: Int = 500
    @State private var resetAlphaEvery: Int = 30
    @State private var densifyGradThresh: Float = 0.0002
    @State private var densifySizeThresh: Float = 0.01
    @State private var stopScreenSizeAt: Int = 4000
    @State private var splitScreenSize: Float = 0.05
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("General")) {
                    HStack {
                        Text("Input Path")
                        Spacer()
                        TextField("Input Path", value: $projectRoot, formatter: )
                    }
                }
                
                Section(header: Text("Advanced")) {
                    
                }
            }
        }
        .navigationTitle("OpenSplat Settings")
    }
}
