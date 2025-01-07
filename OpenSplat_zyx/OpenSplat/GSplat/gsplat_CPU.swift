//
//  gsplat_cpu.swift
//  OpenSplat
//
//  Created by Yuxuan Zhang on 1/5/25.
//

import Foundation
import simd

func quatToRot(quats: [SIMD4<Float>]) -> [simd_float3x3] {
    return quats.map { quat in
        // Normalize the quaternion
        let normalizedQuat = simd_normalize(quat)
        let w = normalizedQuat.w
        let x = normalizedQuat.x
        let y = normalizedQuat.y
        let z = normalizedQuat.z

        // Compute rotation matrix
        return simd_float3x3(
            SIMD3<Float>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)),
            SIMD3<Float>(2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)),
            SIMD3<Float>(2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y))
        )
    }
}

func project_gaussians_forward_tensor_cpu(num_points: Int,
                                          means3d: [SIMD3<Float>],
                                          scales: [SIMD3<Float>],
                                          glob_scale: Float,
                                          quats: [SIMD4<Float>],
                                          viewmat: simd_float4x4,
                                          projmat: simd_float4x4,
                                          fx: Float,
                                          fy: Float,
                                          cx: Float,
                                          cy: Float,
                                          img_height: Int,
                                          img_width: Int,
                                          clip_thresh: Float) -> ([SIMD2<Float>],
                                                                  [Int],
                                                                  [SIMD3<Float>],
                                                                  [simd_float2x2],
                                                                  [Float]) {
    let fovx = 0.5 * Float(img_width) / fx
    let fovy = 0.5 * Float(img_height) / fy
    
    let Rclip = simd_float3x3(SIMD3<Float>(viewmat.columns.0.x, viewmat.columns.0.y, viewmat.columns.0.z),
                              SIMD3<Float>(viewmat.columns.1.x, viewmat.columns.1.y, viewmat.columns.1.z),
                              SIMD3<Float>(viewmat.columns.2.x, viewmat.columns.2.y, viewmat.columns.2.z))
    let Tclip = SIMD3<Float>(viewmat.columns.3.x, viewmat.columns.3.y, viewmat.columns.3.z)
    let pView = means3d.map { Rclip * $0 + Tclip }
    
    let R = quatToRot(quats: quats)
    let M = zip(R, scales).map { simd_mul($0, $1 * glob_scale) }
    let cov3d = M.map { simd_float3x3(SIMD3<Float>($0.x * $0.x, $0.x * $0.y, $0.x * $0.z),
                                      SIMD3<Float>($0.y * $0.x, $0.y * $0.y, $0.y * $0.z),
                                      SIMD3<Float>($0.z * $0.x, $0.z * $0.y, $0.z * $0.z))}
    
    let limX = 1.3 * fovx
    let limY = 1.3 * fovy
    
    let minLimX = pView.map { $0.z * min(limX, max(-limX, $0.x / $0.z)) }
    let minLimY = pView.map { $0.z * min(limY, max(-limY, $0.y / $0.z)) }
    
    let t = zip(zip(minLimX, minLimY), pView).map { SIMD3<Float>($0.0, $0.1, $1.z) }
    let rz = t.map { 1.0 / $0.z }
    let rz2 = rz.map { pow($0, 2) }
    
    let J: [simd_float3x2] = zip(zip(rz, rz2), t).map { (rzPair, t) in
        let (rz, rz2) = rzPair
        return simd_float3x2(
            SIMD2<Float>(fx * rz, 0.0),
            SIMD2<Float>(0.0, fy * rz),
            SIMD2<Float>(-fx * t.x * rz2, -fy * t.y * rz2)
        )
    }
    
    let T = J.map { simd_mul($0, Rclip) }
    var cov2d = zip(T, cov3d).map { (T, cov3d) in
        var result = simd_mul(T, simd_mul(cov3d, simd_transpose(T)))
        result[0, 0] += 0.3
        result[1, 1] += 0.3
        return result
    }
    
    let eps: Float = 1e-6
    let det = cov2d.map { min($0[0, 0] * $0[1, 1] - pow($0[0, 1], 2), eps) }
    let conics = zip(cov2d, det).map { SIMD3<Float>($0[1, 1] / $1, -$0[0, 1] / $1, $0[0, 0] / $1) }
    
    let b = cov2d.map{ ($0[0, 0] + $0[1, 1]) / 2.0 }
    let sq = b.map { sqrt(min(pow($0, 2), 0.1)) }
    let v1 = zip(b, sq).map { $0 + $1 }
    let v2 = zip(b, sq).map { $0 - $1 }
    let radius = zip(v1, v2).map { ceil(3.0 * sqrt(max($0, $1))) }
    
    let pHom = means3d.map { simd_mul(projmat, SIMD4<Float>($0.x, $0.y, $0.z, 1.0)) }
    let rw = pHom.map { 1.0 / min($0.w, eps) }
    let pProj = zip(pHom, rw).map { SIMD3<Float>($0.x * $1, $0.y * $1, $0.z * $1) }
    let u = pProj.map { 0.5 * (($0.x + 1.0) * Float(img_width) - 1.0) }
    let v = pProj.map { 0.5 * (($0.y + 1.0) * Float(img_height) - 1.0) }
    let xys = zip(u, v).map { SIMD2<Float>($0, $1) }
    
    let radii = radius.map { Int($0) }
    let camDepths = pProj.map { $0.z }
    
    return (xys, radii, conics, cov2d, camDepths)
}
