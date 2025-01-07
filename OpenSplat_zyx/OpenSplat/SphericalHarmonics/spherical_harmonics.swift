//
//  SphericalHarmonics.swift
//  OpenSplat
//
//  Created by Yuxuan Zhang on 23/12/24.
//

import Foundation
import Metal

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

let C0: Double = 0.28209479177387814

func rgb2sh(rgb: [SIMD3<Float>]) -> [SIMD3<Float>] {
    return rgb.map { color in
        return SIMD3<Float>(
            (color.x - 0.5) * Float(C0),
            (color.y - 0.5) * Float(C0),
            (color.z - 0.5) * Float(C0))
    }
}

func sh2rgb(sh: [SIMD3<Float>]) -> [SIMD3<Float>] {
    return sh.map { coeffs in
        return SIMD3<Float>(
            max(0.0, min(1.0, coeffs.x * Float(C0) + 0.5)),
            max(0.0, min(1.0, coeffs.y * Float(C0) + 0.5)),
            max(0.0, min(1.0, coeffs.z * Float(C0) + 0.5))
        )
    }
}

class SphericalHarmonics {
    func forward(ctx: inout [String: Any], degreesToUse: Int, viewDirs: MTLBuffer, coeffs: MTLBuffer, mtlctx: MetalContext) -> MTLBuffer? {
        let numPoints = coeffs.length / MemoryLayout<Float>.size
        let degree = degFromSh(numBases: coeffs.length / (numPoints * MemoryLayout<Float>.size))

        ctx["degreesToUse"] = degreesToUse
        ctx["degree"] = degree
        ctx["viewDirs"] = viewDirs

        return compute_sh_forward_tensor(num_points: numPoints, degree: degree, degrees_to_use: degreesToUse,
                                         viewdirs: viewDirs, coeffs: coeffs, context: mtlctx)
        }
    
    func backward(ctx: [String: Any], grad_outputs: [MTLBuffer], mtlctx: MetalContext) -> [MTLBuffer]? {
        guard let v_colors = grad_outputs.first else {
            print("grad_outputs must contain at least one buffer.")
            return nil
        }
        
        guard let degreesToUse = ctx["degreesToUse"] as? Int,
              let degree = ctx["degree"] as? Int,
              let viewDirs = ctx["viewDirs"] as? MTLBuffer else {
            print("Missing context data.")
            return nil
        }
        
        let numPoints = v_colors.length / MemoryLayout<Float>.size / 3
        
        guard let v_coeffs = compute_sh_backward_tensor(num_points: numPoints, degree: degree, degrees_to_use: degreesToUse, viewdirs: viewDirs, v_colors: v_colors, context: mtlctx) else {
            print("Failed to compute backward tensor.")
            return nil
        }
        
        return [MTLBuffer?](arrayLiteral: nil, nil, v_coeffs).compactMap { $0 }
    }
}
