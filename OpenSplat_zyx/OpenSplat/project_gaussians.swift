//
//  project_gaussians.swift
//  OpenSplat
//
//  Created by Yuxuan Zhang on 25/12/24.
//

import Foundation
import simd
import Metal

class ProjectGaussians {
    func apply(means: [SIMD3<Float>],
               scales: [SIMD3<Float>],
               globScale: Float,
               quats: [SIMD4<Float>],
               viewMat: simd_float4x4,
               projMat: simd_float4x4,
               fx: Float,
               fy: Float,
               cx: Float,
               cy: Float,
               imgHeight: Int,
               imgWidth: Int,
               tileBounds: TileBounds,
               clipThresh: Float = 0.01) -> ([SIMD2<Float>],
                                             [Float],
                                             [Int],
                                             [SIMD3<Float>],
                                             [Int],
                                             [[Float]]) {
        let numPoints = means.count
        
        let t = project_gaussians_forward_tensor(num_points: numPoints, means3d: means, scales: scales, glob_scale: globScale, quats: quats, viewmat: viewMat, projmat: projMat, fx: fx, fy: fy, cx: cx, cy: cy, img_height: imgHeight, img_width: imgWidth, tile_bounds: tileBounds, clip_thresh: clipThresh)
        let cov3d = t.0
        let xys = t.1
        let depths = t.2
        let radii = t.3
        let conics = t.4
        let numTilesHit = t.5
        
        return (xys, depths, radii, conics, numTilesHit, cov3d)
    }
    
    func forward(ctx: inout [String: Any],
                 means: [SIMD3<Float>],
                 scales: [SIMD3<Float>],
                 globScale: Float,
                 quats: [SIMD4<Float>],
                 viewMat: simd_float4x4,
                 projMat: simd_float4x4,
                 fx: Float,
                 fy: Float,
                 cx: Float,
                 cy: Float,
                 imgHeight: Int,
                 imgWidth: Int,
                 tileBounds: TileBounds,
                 clipThresh: Float = 0.01) -> ([SIMD2<Float>],
                                               [Float],
                                               [Int],
                                               [SIMD3<Float>],
                                               [Int],
                                               [[Float]]) {
        let numPoints = means.count
        
        let t = project_gaussians_forward_tensor(num_points: numPoints, means3d: means, scales: scales, glob_scale: globScale, quats: quats, viewmat: viewMat, projmat: projMat, fx: fx, fy: fy, cx: cx, cy: cy, img_height: imgHeight, img_width: imgWidth, tile_bounds: tileBounds, clip_thresh: clipThresh)
        let cov3d = t.0
        let xys = t.1
        let depths = t.2
        let radii = t.3
        let conics = t.4
        let numTilesHit = t.5
        
        ctx["imgHeight"] = imgWidth
        ctx["imgWidth"] = imgWidth
        ctx["numPoints"] = numPoints
        ctx["globScale"] = globScale
        ctx["fx"] = fx
        ctx["fy"] = fy
        ctx["cx"] = cx
        ctx["cy"] = cy
        ctx["save_for_backward"] = (means, scales, quats, viewMat, projMat, cov3d, radii, conics)
        
        return (xys, depths, radii, conics, numTilesHit, cov3d)
    }
    
    func backward(ctx: inout [String: Any], grad_outputs: ) ->  {
        
    }
}
