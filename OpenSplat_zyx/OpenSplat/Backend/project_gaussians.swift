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
        var means = means
        var scales = scales
        var quats = quats
        var viewMat = viewMat
        var projMat = projMat
        
        let t = project_gaussians_forward_tensor(num_points: numPoints,
                                                 means3d: &means,
                                                 scales: &scales,
                                                 glob_scale: globScale,
                                                 quats: &quats,
                                                 viewmat: &viewMat,
                                                 projmat: &projMat,
                                                 fx: fx,
                                                 fy: fy,
                                                 cx: cx,
                                                 cy: cy,
                                                 img_height: imgHeight,
                                                 img_width: imgWidth,
                                                 tile_bounds: tileBounds,
                                                 clip_thresh: clipThresh)
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
        var means = means
        var scales = scales
        var quats = quats
        var viewMat = viewMat
        var projMat = projMat
         
        let t = project_gaussians_forward_tensor(num_points: numPoints,
                                                 means3d: &means,
                                                 scales: &scales,
                                                 glob_scale: globScale,
                                                 quats: &quats,
                                                 viewmat: &viewMat,
                                                 projmat: &projMat,
                                                 fx: fx,
                                                 fy: fy,
                                                 cx: cx,
                                                 cy: cy,
                                                 img_height: imgHeight,
                                                 img_width: imgWidth,
                                                 tile_bounds: tileBounds,
                                                 clip_thresh: clipThresh)
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
    
    func backward(ctx: inout [String: Any],
                  grad_outputs: ([SIMD2<Float>],
                                 [Float],
                                 [Int],
                                 [SIMD3<Float>],
                                 [Int],
                                 [[Float]])) -> ([SIMD3<Float>],
                                                 [SIMD3<Float>],
                                                 Float?,
                                                 [SIMD4<Float>],
                                                 simd_float4x4?,
                                                 simd_float4x4?,
                                                 Float?,
                                                 Float?,
                                                 Float?,
                                                 Float?,
                                                 Int?,
                                                 Int?,
                                                 TileBounds?,
                                                 Float?) {
        var v_xys = grad_outputs.0
        var v_depths = grad_outputs.1
        let v_radii = grad_outputs.2
        var v_conics = grad_outputs.3
        let v_numTiles = grad_outputs.4
        let v_cov3d = grad_outputs.5
        
        guard let numPoints = ctx["numPoints"] as? Int,
              let globScale = ctx["globScale"] as? Float,
              let fx = ctx["fx"] as? Float,
              let fy = ctx["fy"] as? Float,
              let cx = ctx["cx"] as? Float,
              let cy = ctx["cy"] as? Float,
              let imgHeight = ctx["imgHeight"] as? Int,
              let imgWidth = ctx["imgWidth"] as? Int,
              let saved = ctx["save_for_backward"] as? ([SIMD3<Float>], [SIMD3<Float>], [SIMD4<Float>], simd_float4x4, simd_float4x4, [[Float]], [Int], [SIMD3<Float>]) else {
            fatalError("Context does not contain expected data.")
        }
        var means = saved.0
        var scales = saved.1
        var quats = saved.2
        var viewMat = saved.3
        var projMat = saved.4
        var cov3d = saved.5
        var radii = saved.6
        var conics = saved.7
        
        let t = project_gaussians_backward_tensor(num_points: numPoints,
                                                  means3d: &means,
                                                  scales: &scales,
                                                  glob_scale: globScale,
                                                  quats: &quats,
                                                  viewmat: &viewMat,
                                                  projmat: &projMat,
                                                  fx: fx,
                                                  fy: fy,
                                                  cx: cx,
                                                  cy: cy,
                                                  img_height: imgHeight,
                                                  img_width: imgWidth,
                                                  cov3d: &cov3d,
                                                  radii: &radii,
                                                  conics: &conics,
                                                  v_xy: &v_xys,
                                                  v_depth: &v_depths,
                                                  v_conic: &v_conics)
        
        return (t.2, t.3, nil, t.4, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil)
    }
}
