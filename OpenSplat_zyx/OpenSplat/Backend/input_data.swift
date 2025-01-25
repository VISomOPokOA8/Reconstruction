//
//  input_data.swift
//  OpenSplat
//
//  Created by Yuxuan Zhang on 12/28/24.
//

import Foundation
import simd

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
    var image: [[[Float]]]
    
    var imagePyramids: [Int: [[[Float]]]]
    
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
    
    func undistortionParameters() -> simd_float8 {
        return simd_float8(k1, k2, p1, p2, k3, 0.0, 0.0, 0.0)
    }
    
    mutating func getImage(downscaleFactor: Int) -> [[[Float]]] {
        if downscaleFactor <= 1 {
            return image
        } else {
            if imagePyramids[downscaleFactor] != nil {
                return imagePyramids[downscaleFactor]!
            }
            
            image = resizeImage(matrix: image, downscaleFactor: downscaleFactor)
            imagePyramids[downscaleFactor] = image
            return image
        }
    }
    
    mutating func loadImage(downscaleFactor: Float) throws {
        if !image.isEmpty {
            throw NSError(domain: "loadImage already called", code: 1)
        }
        print("Loading" + filePath)
        
        
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
//
//

func resizeImage(matrix: [[[Float]]], downscaleFactor: Int) -> [[[Float]]] {
    let oldHeight = matrix.count
    let oldWidth = matrix[0].count
    let channels = matrix[0][0].count

    guard downscaleFactor > 1 else {
        fatalError("downscaleFactor should be greater than 1")
    }

    let newHeight = oldHeight / downscaleFactor
    let newWidth = oldWidth / downscaleFactor

    // 创建缩放后的空矩阵
    var resizedMatrix = [[[Float]]](
        repeating: [[Float]](
            repeating: [Float](repeating: 0, count: channels),
            count: newWidth
        ),
        count: newHeight
    )

    // 遍历新矩阵
    for newRow in 0..<newHeight {
        for newCol in 0..<newWidth {
            for channel in 0..<channels {
                // 计算对应的原始区域
                let startRow = newRow * downscaleFactor
                let endRow = min((newRow + 1) * downscaleFactor, oldHeight)
                let startCol = newCol * downscaleFactor
                let endCol = min((newCol + 1) * downscaleFactor, oldWidth)

                // 计算区域平均值
                var sum: Float = 0.0
                var count: Float = 0.0

                for row in startRow..<endRow {
                    for col in startCol..<endCol {
                        sum += matrix[row][col][channel]
                        count += 1.0
                    }
                }

                resizedMatrix[newRow][newCol][channel] = sum / count
            }
        }
    }

    return resizedMatrix
}

