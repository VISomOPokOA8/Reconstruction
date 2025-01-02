//
//  nerfstudio.swift
//  OpenSplat
//
//  Created by Yuxuan Zhang on 12/31/24.
//

import Foundation
import simd

typealias Mat4 = [[Float]]

struct Frame {
    var filePath = ""
    var width = 0
    var height = 0
    var fx = 0.0
    var fy = 0.0
    var cx = 0.0
    var cy = 0.0
    var k1 = 0.0
    var k2 = 0.0
    var p1 = 0.0
    var p2 = 0.0
    var k3 = 0.0
    var transformMatrix: Mat4 = []
}

func to_json(f: Frame) -> [String: Any] {
    return [
        "file_path": f.filePath,
        "w": f.width,
        "h": f.height,
        "fl_x": f.fx,
        "fl_y": f.fy,
        "cx": f.cx,
        "cy": f.cy,
        "k1": f.k1,
        "k2": f.k2,
        "p1": f.p1,
        "p2": f.p2,
        "k3": f.k3,
        "transform_matrix": f.transformMatrix
    ]
}

func from_json(j: [String: Any]) -> Frame {
    let filePath = j["file_path"] as? String ?? ""
    let width = j["w"] as? Int ?? 0
    let height = j["h"] as? Int ?? 0
    let fx = j["fl_x"] as? Double ?? 0.0
    let fy = j["fl_y"] as? Double ?? 0.0
    let cx = j["cx"] as? Double ?? 0.0
    let cy = j["cy"] as? Double ?? 0.0
    let k1 = j["k1"] as? Double ?? 0.0
    let k2 = j["k2"] as? Double ?? 0.0
    let p1 = j["p1"] as? Double ?? 0.0
    let p2 = j["p2"] as? Double ?? 0.0
    let k3 = j["k3"] as? Double ?? 0.0
    let transformMatrix = j["transform_matrix"] as? Mat4 ?? []
    
    return Frame(
        filePath: filePath,
        width: width,
        height: height,
        fx: fx,
        fy: fy,
        cx: cx,
        cy: cy,
        k1: k1,
        k2: k2,
        p1: p1,
        p2: p2,
        k3: k3,
        transformMatrix: transformMatrix
    )
}

struct Transforms {
    var cameraModel: String
    var frames: [Frame]
    var plyFilePath: String
}

func to_json(t: Transforms) -> [String: Any] {
    return [
        "camera_model": t.cameraModel,
        "frames": t.frames,
        "ply_file_path": t.plyFilePath
    ]
}

func from_json(j: [String: Any]) -> Transforms {
    let cameraModel = j["camera_model"] as? String ?? ""
    let frames = j["frames"] as? [Frame] ?? []
    let plyFilePath = j["ply_file_path"] as? String ?? ""
    
    return Transforms(cameraModel: cameraModel, frames: frames, plyFilePath: plyFilePath)
}

func readTransforms(filename: String) -> Transforms {
    guard let fileURL = URL(string: filename),
          let data = try? Data(contentsOf: fileURL),
          let jsonObject = try? JSONSerialization.jsonObject(with: data, options: []),
          let jsonDict = jsonObject as? [String: Any] else {
        fatalError("Failed to read or parse file at \(filename)")
    }
    return from_json(j: jsonDict)
}

func posesFromTransforms(t: Transforms) -> [simd_float4x4] {
    var poses = Array(repeating: simd_float4x4(0), count: t.frames.count)
    for c: size_t in 0..<t.frames.count {
        for i: size_t in 0..<4 {
            for j: size_t in 0..<4 {
                poses[c][i][j] = t.frames[c].transformMatrix[i][j]
            }
        }
    }
    return poses
}

func inputDataFromNerfStudio(projectRoot: String) throws -> InputData {
    var ret = InputData(cameras: [], scale: 0.0, translation: SIMD3<Float>(), points: Points(xyz: [], rgb: []))
    let nsRoot = URL(string: projectRoot)
    guard let transformsPath = nsRoot?.appendingPathComponent("transforms.json") else {
        throw NSError(domain: "Failed to create transforms path. ", code: 1)
    }
    
    guard FileManager.default.fileExists(atPath: transformsPath.path()) else {
        throw NSError(domain: transformsPath.path() + "does not exist. ", code: 2)
    }
    let t = readTransforms(filename: transformsPath.path())
    if t.plyFilePath.isEmpty {
        throw NSError(domain: "ply_file_path is empty", code: 3)
    }
    guard let pointSetPath = nsRoot?.appendingPathComponent(t.plyFilePath) else {
        throw NSError(domain: "Failed to create point set path. ", code: 1)
    }
    let pSet = try readPointSet(filename: pointSetPath.path())
    
    let unorientedPoses = posesFromTransforms(t: t)
    
    let r = autoScaleAndCenterPoses(poses: unorientedPoses)
    let poses = r.0
    ret.translation = r.1
    ret.scale = r.2
    
    for i: size_t in 0..<t.frames.count {
        let f = t.frames[i]
        
        let camera = Camera(
            id: i,
            width: f.width,
            height: f.height,
            fx: Float(f.fx),
            fy: Float(f.fy),
            cx: Float(f.cx),
            cy: Float(f.cy),
            k1: Float(f.k1),
            k2: Float(f.k2),
            p1: Float(f.p1),
            p2: Float(f.p2),
            k3: Float(f.k3),
            camToWorld: poses[i],
            filePath: (nsRoot?.appendingPathComponent(f.filePath).path ?? ""),
            K: simd_float3x3(0),
            image: nil,
            imagePyramids: [:]
        )
        
        ret.cameras.append(camera)
    }
    
    guard let points = pSet?.points else {
        throw NSError(domain: "Failed to clone points. ", code: 1)
    }
    
    ret.points.xyz = points.map { point in
        return (point - ret.translation) * ret.scale
    }
    
    guard let rgb = pSet?.colors else {
        throw NSError(domain: "Failed to clone colors. ", code: 1)
    }
    ret.points.rgb = rgb
    
    return ret
}
