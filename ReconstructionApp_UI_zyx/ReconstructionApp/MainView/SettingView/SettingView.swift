//
//  SettingView.swift
//  ReconstructionApp
//
//  Created by Yuxuan Zhang on 29/11/24.
//

import RealityKit
import SwiftUI
import os

struct SettingView: View {
    @Environment(AppDataModel.self) private var appDataModel: AppDataModel
    
    @State private var calibrationFolder: URL?
    @State private var calibrationImageURLs: [URL] = []
    @State private var calibrationImageNum: Int?
    
    @State private var reconstructionFolder: URL?
    @State private var reconstructionImageURLs: [URL] = []
    @State private var reconstructionImageNum: Int?
    @State private var metadataAvailability = ImageHelper.MetadataAvailability()
    
    @State private var showFileImporter = false
    
    init() {}
    
    var body: some View {
        VStack {
            // Calibration Images ////////////////////////////////////////////////////////////////
            LabeledContent("Calibration Images:") {
                VStack(spacing: 6) {
                    HStack {
                        Text(calibrationTitle).foregroundStyle(.secondary).font(.caption)
                        
                        Spacer()
                        
                        if calibrationFolder != nil {
                            Button {
                                calibrationFolder = nil
                            } label: {
                                Image(systemName: "xmark.circle.fill").frame(height: 15)
                            }
                            .buttonStyle(.plain)
                            .foregroundStyle(.secondary)
                        }
                    }
                    .padding([.leading, .trailing], 6)
                    .padding(.top, 3)
                    .frame(height: 20)
                    
                    Divider().padding(.top, -4).padding(.horizontal, 6)
                    
                    ThumbnailsView(imageURLs: $calibrationImageURLs)
                }
                .background(Color.gray.opacity(0.1))
                .cornerRadius(10)
                .onAppear {
                    calibrationFolder = appDataModel.calibrationImageFolder
                }
            }
            .frame(height: 110)
            .dropDestination(for: URL.self) { items, location in
                guard !items.isEmpty else {
                    return false
                }
                var isDirectory: ObjCBool = false
                if let url = items.first, FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue == true {
                    calibrationFolder = url
                    return true
                }
            }
            .task(id: calibrationFolder) {
                guard let reconstructionFolder else {
                    calibrationImageNum = nil
                    calibrationImageURLs = []
                    return
                }
                
                reconstructionImageNum = reconstructionImageURLs.count
            }
            
            Spacer()
            
            // Reconstruction Images ////////////////////////////////////////////////////////////////
            LabeledContent("Reconstruction Images:") {
                VStack(spacing: 6) {
                    HStack {
                        Text(reconstructionTitle).foregroundStyle(.secondary).font(.caption)
                        
                        Spacer()
                        
                        if metadataAvailability.gravity && metadataAvailability.depth {
                            MetadataView()
                        }
                        
                        if reconstructionFolder != nil {
                            Button {
                                reconstructionFolder = nil
                            } label: {
                                Image(systemName: "xmark.circle.fill").frame(height: 15)
                            }
                            .buttonStyle(.plain)
                            .foregroundStyle(.secondary)
                        }
                    }
                    .padding([.leading, .trailing], 6)
                    .padding(.top, 3)
                    .frame(height: 20)
                    
                    Divider().padding(.top, -4).padding(.horizontal, 6)
                    
                    ThumbnailsView(imageURLs: $reconstructionImageURLs)
                }
                .background(Color.gray.opacity(0.1))
                .cornerRadius(10)
                .onAppear {
                    reconstructionFolder = appDataModel.reconstructionImageFolder
                }
            }
            .frame(height: 110)
            .onChange(of: reconstructionFolder) {
                metadataAvailability = ImageHelper.MetadataAvailability()
            }
            .dropDestination(for: URL.self) { items, location in
                guard !items.isEmpty else {
                    return false
                }
                var isDirectory: ObjCBool = false
                if let url = items.first, FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue == true {
                    reconstructionFolder = url
                    return true
                }
            }
            .task(id: reconstructionFolder) {
                guard let reconstructionFolder else {
                    reconstructionImageNum = nil
                    reconstructionImageURLs = []
                    metadataAvailability = ImageHelper.MetadataAvailability()
                    return
                }
                
                reconstructionImageURLs = ImageHelper.getListOfURLs(from: reconstructionFolder)
                if reconstructionImageURLs.isEmpty {
                    appDataModel.state = .error
                    appDataModel.alertMessage = "\(String(describing: PhotogrammetrySession.Error.invalidImages(reconstructionFolder)))"
                    self.reconstructionFolder = nil
                    return
                }
                
                reconstructionImageNum = reconstructionImageURLs.count
                
                metadataAvailability = await ImageHelper.loadMetadataAvailability(from: reconstructionImageURLs)
            }
            
            Spacer()
            
            // Data Folder Selection ////////////////////////////////////////////////////////////////
            LabeledContent("Data Folder:") {
                Button {
                    showFileImporter.toggle()
                } label: {
                    HStack {
                        
                    }
                }
                .fileImporter(isPresented: $showFileImporter, allowedContentTypes: [.folder]) { result in
                    switch result {
                    case .success(let directory):
                        let gotAccess = directory.startAccessingSecurityScopedResource()
                        if !gotAccess {
                            return
                        }
                        appDataModel.dataFolder = directory
                        
                    case .failure(let error):
                        appDataModel.alertMessage = "\(error)"
                        appDataModel.state = .error
                    }
                }
            }
        }
    }
    
    private var calibrationTitle: String {
        if let calibrationImageNum = calibrationImageNum {
            return "\(calibrationImageNum) images"
        } else {
            return "Drag in a folder of images"
        }
    }
    
    private var reconstructionTitle: String {
        if let reconstructionImageNum = reconstructionImageNum {
            return "\(reconstructionImageNum) images"
        } else {
            return "Drag in a folder of images"
        }
    }
}
