//
//  SphericalHarmonics.swift
//  OpenSplat
//
//  Created by Yuxuan Zhang on 23/12/24.
//

import Foundation

func degFromSh(numBases: Int) -> Int {
    switch numBases {
    case 1:
        return 1
    case 2:
        return 4
    case 3:
        return 9
    case 4:
        return 16
    default:
        return 25
    }
}

let C0: Float = 0.28209479177387814

func rgb2sh(rgb: [SIMD3<Float>]) -> [SIMD3<Float>] {
    return rgb.map { SIMD3<Float>(($0.x - 0.5) / C0, ($0.y - 0.5) / C0, ($0.z - 0.5) / C0) }
}

func sh2rgb(sh: [SIMD3<Float>]) -> [SIMD3<Float>] {
    return sh.map { SIMD3<Float>(
            max(0.0, min(1.0, $0.x * Float(C0) + 0.5)),
            max(0.0, min(1.0, $0.y * Float(C0) + 0.5)),
            max(0.0, min(1.0, $0.z * Float(C0) + 0.5))
    )}
}

class SphericalHarmonics {
    func apply(degreesToUse: Int,
               viewDirs: [SIMD3<Float>],
               coeffs: [[SIMD3<Float>]]) -> [SIMD3<Float>] {
        let numPoints = coeffs.count
        let degree = degFromSh(numBases: coeffs[0].count)
        
        return compute_sh_forward_tensor(num_points: numPoints, degree: degree, degrees_to_use: degreesToUse, viewdirs: viewDirs, coeffs: coeffs)
    }
    
    func forward(ctx: inout [String: Any],
                 degreesToUse: Int,
                 viewDirs: [SIMD3<Float>],
                 coeffs: [[SIMD3<Float>]]) -> [SIMD3<Float>] {
        let numPoints = coeffs.count
        let degree = degFromSh(numBases: coeffs[0].count)
        
        ctx["degreesToUse"] = degreesToUse
        ctx["degree"] = degree
        ctx["save_for_backward"] = viewDirs
        
        return compute_sh_forward_tensor(num_points: numPoints, degree: degree, degrees_to_use: degreesToUse, viewdirs: viewDirs, coeffs: coeffs)
    }
    
    func backward(ctx: inout [String: Any], grad_output: ) {
        
    }
}
