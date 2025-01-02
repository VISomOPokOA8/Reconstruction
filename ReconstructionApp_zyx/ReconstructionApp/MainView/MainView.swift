//
//  MainView.swift
//  ReconstructionApp
//
//  Created by Yuxuan Zhang on 29/11/24.
//

import SwiftUI

struct MainView: View {
    @State private var appDataModel = AppDataModel()
    @State private var showAlert = false
    
    var body: some View {
        VStack {
            switch appDataModel.state {
            case .ready:
                SettingView()
            
            case .calibrating, .quick_reconstructing, .gs_reconstructing: 
                ProcessingView()
                
            case .calibration_viewing:
                CalibrationView()
                
            case .quick_reconstruction_viewing, .gs_reconstruction_viewing:
                ReconstructionView()
                
            case .error:
                EmptyView()
            }
        }
        .environment(appDataModel)
        .onChange(of: appDataModel.state) {
            if appDataModel.state == .error {
                showAlert = true
            }
        }
        .alert(appDataModel.alertMessage, isPresented: $showAlert) {
            Button("OK") {
                appDataModel.state = .ready
            }
        }
    }
}
