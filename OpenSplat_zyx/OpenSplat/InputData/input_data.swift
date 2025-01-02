//
//  input_data.swift
//  OpenSplat
//
//  Created by Yuxuan Zhang on 12/28/24.
//

import Foundation
import simd
import Metal

struct Camera {
    var id: Int = -1
    var width: Int = 0
    var height: Int = 0
    
    var fx: Float = 0
    var fy: Float = 0
    var cx: Float = 0
    var cy: Float = 0
    
    var k1: Float = 0
    var k2: Float = 0
    var p1: Float = 0
    var p2: Float = 0
    var k3: Float = 0
    
    var camToWorld: simd_float4x4
    var filePath: String = ""
    
    var K: simd_float3x3
    var image: MTLTexture?
    
    var imagePyramids: [Int: MTLTexture]
    
    func getIntrinsicsMatrix() -> simd_float3x3 {
        return simd_float3x3(
            SIMD3<Float>(fx, 0.0, cx),
            SIMD3<Float>(0.0, fy, cy),
            SIMD3<Float>(0.0, 0.0, 1.0)
        )
    }
    
    func hasDistortionParameters() -> Bool {
        return k1 != 0.0 || k2 != 0.0 || p1 != 0.0 || p2 != 0.0 || k3 != 0.0
    }
    
    func undistortionParameters() -> [Float] {
        return [Float](arrayLiteral: k1, k2, p1, p2, k3, 0.0, 0.0, 0.0)
    }
    
    mutating func getImage(downscaleFactor: Int, device: MTLDevice, commandQueue: MTLCommandQueue) throws -> MTLTexture {
        guard let baseImage = image else {
            throw NSError(domain: "Image is empty.", code: 1)
        }
        
        if downscaleFactor <= 1 {
            return baseImage
        } else {
            if let cachedImage = imagePyramids[downscaleFactor] {
                return cachedImage
            } else {
                guard let resizedImage = resizeTexture(texture: baseImage, downscaleFactor: downscaleFactor, device: device, commandQueue: commandQueue) else {
                    throw NSError(domain: "Failed to resize image.", code: 2)
                }
                imagePyramids[downscaleFactor] = resizedImage
                return resizedImage
            }
        }
    }
    
    mutating func loadImage(downscaleFactor: Float, device: MTLDevice, commandQueue: MTLCommandQueue) throws {
        if image != nil {
            throw NSError(domain: "loadImage already called", code: 3)
        }
        print("Loading" + filePath)
        
        guard let img = imreadRGB(filename: filePath, device: device) else {
            throw NSError(domain: "Failed to read image. ", code: 2)
        }
        
        var rescaleF: Float = 1.0
        if img.height != height || img.width != width {
            rescaleF = Float(img.height) / Float(img.width)
        }
        fx *= rescaleF
        fy *= rescaleF
        cx *= rescaleF
        cy *= rescaleF
        
        if downscaleFactor > 1.0 {
            let scaleFactor = 1.0 / downscaleFactor
            guard let img = resizeTexture(texture: img, downscaleFactor: Int(downscaleFactor), device: device, commandQueue: commandQueue) else {
                throw NSError(domain: "Failed to resize image.", code: 2)
            }
            fx *= scaleFactor
            fy *= scaleFactor
            cx *= scaleFactor
            cy *= scaleFactor
        }
        
        var K = getIntrinsicsMatrix()
        var roi = MTLRegionMake2D(0, 0, 0, 0)
        
        if hasDistortionParameters() {
            let distCoeffs = undistortionParameters()
            let (newK, roi) = getOptimalNewCameraMatrix(cameraMatrix: K, distCoeffs: distCoeffs, imageSize: (img.width, img.height), alpha: 0)
            
            let undistortedDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: img.pixelFormat, width: img.width, height: img.height, mipmapped: false)
            undistortedDescriptor.usage = [.shaderRead, .shaderWrite]
            guard let undistorted = device.makeTexture(descriptor: undistortedDescriptor) else {
                throw NSError(domain: "Failed to create undistorted texture.", code: 2)
            }
            image = undistorted
            K = newK
        } else {
            roi = MTLRegionMake2D(0, 0, img.width, img.height)
            image = img
        }
        
        guard let image = cropTexture(texture: image, region: roi, device: device) else {
            throw NSError(domain: "Failed to crop texture. ", code: 2)
        }
        
        height = image.height
        width = image.width
        fx = K[0][0]
        fy = K[1][1]
        cx = K[0][2]
        cy = K[1][2]
    }
}

struct Points {
    var xyz: [SIMD3<Float>]
    var rgb: [SIMD3<UInt8>]
}

struct InputData {
    var cameras: [Camera]
    var scale: Float
    var translation: SIMD3<Float>
    var points: Points
    
    func getCameras(validata: Bool, valImage: String = "random") throws -> ([Camera], Camera?) {
        if !validata {
            return (cameras, nil)
        } else {
            var valIdx: size_t = -1
            
            if valImage == "random" {
                valIdx = Int.random(in: 0..<cameras.count)
            } else {
                for i in 0..<cameras.count {
                    if cameras[i].filePath == valImage {
                        valIdx = i
                        break
                    }
                }
                if valIdx == -1 {
                    throw NSError(domain: valImage + "not in the list of cameras. ", code: 4)
                }
            }
            
            var cams: [Camera] = []
            var valCam: Camera? = nil
            
            for i in 0..<cameras.count {
                if i != valIdx {
                    cams.append(cameras[i])
                } else {
                    valCam = cameras[i]
                }
            }
            
            return (cams, valCam)
        }
    }
    
    func saveCameras(filename: String, keepCrs: Bool) throws {
        var j: [[String: Any]] = []
        
        for i in 0..<cameras.count {
            let cam = cameras[i]
            var camera: [String: Any] = [:]
            
            camera["id"] = i
            camera["img_name"] = URL(fileURLWithPath: cam.filePath).lastPathComponent
            camera["width"] = cam.width
            camera["height"] = cam.height
            camera["fx"] = cam.fx
            camera["fy"] = cam.fy
            
            var R = simd_float3x3(
                SIMD3<Float>(cam.camToWorld.columns.0.x, cam.camToWorld.columns.0.y, cam.camToWorld.columns.0.z),
                SIMD3<Float>(cam.camToWorld.columns.1.x, cam.camToWorld.columns.1.y, cam.camToWorld.columns.1.z),
                SIMD3<Float>(cam.camToWorld.columns.2.x, cam.camToWorld.columns.2.y, cam.camToWorld.columns.2.z)
                )
            var T = SIMD3<Float>(
                cam.camToWorld.columns.3.x,
                cam.camToWorld.columns.3.y,
                cam.camToWorld.columns.3.z
            )
            
            R = simd_mul(R, simd_float3x3(diagonal: SIMD3<Float>(1.0, -1.0, -1.0)))
            
            if keepCrs {
                T = T * (1 + 1 / scale)
            }
            
            let position = SIMD3<Float>(T.x, T.y, T.z)
            let rotation = simd_float3x3(
                SIMD3<Float>(R[0][0], R[0][1], R[0][2]),
                SIMD3<Float>(R[1][0], R[1][1], R[1][2]),
                SIMD3<Float>(R[2][0], R[2][1], R[2][2])
            )
            
            camera["position"] = position
            camera["rotation"] = rotation
            j.append(camera)
        }
        
        if let jsonData = try? JSONSerialization.data(withJSONObject: j, options: .prettyPrinted) {
            let url = URL(fileURLWithPath: filename)
            try? jsonData.write(to: url)
            print("Wrote " + filename)
        } else {
            throw NSError(domain: "Failed to save cameras. ", code: 2)
        }
    }
}

func inputDataFromX(projectRoot: String) throws -> InputData {
    let root = URL(fileURLWithPath: projectRoot)
    
    let fileManager = FileManager.default
    if fileManager.fileExists(atPath: root.appendingPathComponent("transforms.json").path()) {
        return try inputDataFromNerfStudio(projectRoot: projectRoot)
    } else {
        throw NSError(domain: "Invalid project folder", code: 5)
    }
}

//
//  Customized Functions
//

func resizeTexture(texture: MTLTexture, downscaleFactor: Int, device: MTLDevice, commandQueue: MTLCommandQueue) -> MTLTexture? {
    let targetWidth = texture.width / downscaleFactor
    let targetHeight = texture.height / downscaleFactor

    let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: texture.pixelFormat,
                                                              width: targetWidth,
                                                              height: targetHeight,
                                                              mipmapped: false)
    descriptor.usage = [.shaderRead, .shaderWrite]
    guard let outputTexture = device.makeTexture(descriptor: descriptor) else {
        print("Failed to create output texture.")
        return nil
    }

    guard let library = device.makeDefaultLibrary(),
          let function = library.makeFunction(name: "resizeTexture"),
          let pipeline = try? device.makeComputePipelineState(function: function) else {
        print("Failed to create compute pipeline.")
        return nil
    }

    guard let commandBuffer = commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
        print("Failed to create command buffer or encoder.")
        return nil
    }

    encoder.setComputePipelineState(pipeline)
    encoder.setTexture(texture, index: 0)
    encoder.setTexture(outputTexture, index: 1)

    let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
    let threadGroups = MTLSize(width: (targetWidth + 15) / 16,
                               height: (targetHeight + 15) / 16,
                               depth: 1)
    encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    return outputTexture
}

func getOptimalNewCameraMatrix(cameraMatrix: simd_float3x3, distCoeffs: [Float], imageSize: (Int, Int), alpha: Float) -> (simd_float3x3, MTLRegion) {
    let width = Float(imageSize.0)
    let height = Float(imageSize.1)
    
    let newWidth = width * (1.0 - alpha)
    let newHeight = height * (1.0 - alpha)
    
    var newCameraMatrix = cameraMatrix
    newCameraMatrix[0][0] *= newWidth / width
    newCameraMatrix[1][1] *= newHeight / height
    newCameraMatrix[0][2] = newWidth / 2.0
    newCameraMatrix[1][2] = newHeight / 2.0
    
    let roiX = (width - newWidth) / 2.0
    let roiY = (height - newHeight) / 2.0
    let roi = MTLRegionMake2D(Int(roiX), Int(roiY), Int(newWidth), Int(newHeight))
    
    return (newCameraMatrix, roi)
}

func cropTexture(texture: MTLTexture?, region: MTLRegion, device: MTLDevice) -> MTLTexture? {
    guard let texture = texture else { return nil }
    
    let descriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: texture.pixelFormat,
        width: region.size.width,
        height: region.size.height,
        mipmapped: false
    )
    descriptor.usage = [.shaderRead, .shaderWrite]
    
    guard let croppedTexture = device.makeTexture(descriptor: descriptor) else {
        print("Failed to create cropped texture.")
        return nil
    }
    
    let bytesPerRow = texture.width * 4
    var buffer = [UInt8](repeating: 0, count: region.size.width * region.size.height * 4)
    texture.getBytes(&buffer, bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0)
    croppedTexture.replace(region: MTLRegionMake2D(0, 0, region.size.width, region.size.height), mipmapLevel: 0, withBytes: buffer, bytesPerRow: region.size.width * 4)
    
    return croppedTexture
}

