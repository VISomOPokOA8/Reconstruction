//
//  tensor_math.swift
//  OpenSplat
//
//  Created by Yuxuan Zhang on 12/28/24.
//

import Foundation
import simd

func quatToRotMat(quat: SIMD4<Float>) -> simd_float3x3 {
    let u = simd_normalize(quat)
    
    return simd_float3x3(
        SIMD3<Float>(1.0 - 2.0 * (u.y * u.y + u.z * u.z), 2.0 * (u.x * u.y - u.w * u.z), 2.0 * (u.x * u.z + u.w * u.y)),
        SIMD3<Float>(2.0 * (u.x * u.y + u.w * u.z), 1.0 - 2.0 * (u.x * u.x + u.z * u.z), 2.0 * (u.y * u.z - u.w * u.x)),
        SIMD3<Float>(2.0 * (u.x * u.z - u.w * u.y), 2.0 * (u.y * u.z + u.w * u.x), 1.0 - 2.0 * (u.x * u.x + u.y * u.y))
    )
}

func autoScaleAndCenterPoses(poses: [simd_float4x4]) -> ([simd_float4x4], SIMD3<Float>, Float) {
    var origins = poses.map { SIMD3<Float>($0[3].x, $0[3].y, $0[3].z) }
    let center = origins.reduce(SIMD3<Float>(repeating: 0), +) / Float(origins.count)
    origins = origins.map { $0 - center }
    
    let f = 1.0 / (origins.flatMap { [$0.x, $0.y, $0.z] }.map { abs($0) }.max() ?? 1.0)
    origins = origins.map { $0 * f }
    
    let transformedPoses = zip(poses, origins).map { pose, origin in
        var updatedPose = pose
        updatedPose[3] = SIMD4<Float>(origin, 1.0)
        return updatedPose
    }
    
    return (transformedPoses, center, f)
}

func rotationMatrix(a: SIMD3<Float>, b: SIMD3<Float>) -> simd_float3x3 {
    let a1 = a / simd_normalize(a)
    let b1 = b / simd_normalize(b)
    let v = simd_cross(a1, b1)
    let c = simd_dot(a1, b1)
    let EPS: Float = 1e-8
    if c < -1 + EPS {
        let eps = SIMD3<Float>(Float.random(in: -0.005...0.005), Float.random(in: -0.005...0.005), Float.random(in: -0.005...0.005))
        return rotationMatrix(a: a1 + eps, b: b1)
    } else {
        let s = simd_normalize(v)
        let skew = simd_float3x3(
            SIMD3<Float>(0.0, -v.z, -v.y),
            SIMD3<Float>(v.z, 0.0, -v.x),
            SIMD3<Float>(-v.y, v.x, 0.0)
        )
        
        let skewSquared = simd_mul(skew, skew)
        let factor = (1 - c) / (simd_dot(s, s) + EPS)

        return simd_float3x3(1) + skew + skewSquared * factor
    }
}

