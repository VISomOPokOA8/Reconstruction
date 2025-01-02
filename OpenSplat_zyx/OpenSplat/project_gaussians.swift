//
//  project_gaussians.swift
//  OpenSplat
//
//  Created by Yuxuan Zhang on 25/12/24.
//

import Foundation
import Metal

#if METAL

class ProjectGaussians {
    func forward(ctx: inout [String: Any], means: MTLBuffer, scales: MTLBuffer, globScale: Float, quats: MTLBuffer, viewMat: MTLBuffer, projMat: MTLBuffer, fx: Float, fy: Float, cx: Float, cy: Float, imgHeight: Int, imgWidth: Int, tileBounds: SIMD3<Int>, clipThresh: Float = 0.01, mtlctx: MetalContext) -> [MTLBuffer]? {
        let numPoints = means.length / (3 * MemoryLayout<Float>.stride)
        
        let t = project_gaussians_forward_tensor(num_points: numPoints, means3d: means, scales: scales, glob_scale: globScale, quats: quats, viewmat: viewMat, projmat: projMat, fx: fx, fy: fy, cx: cx, cy: cy, img_height: imgHeight, img_width: imgWidth, tile_bounds: tileBounds, clip_thresh: clipThresh, context: mtlctx)
        
        guard let cov3d = t.0, let xys = t.1, let depths = t.2, let radii = t.3, let conics = t.4, let numTilesHit = t.5 else {
            print("Failed to project forward tensor.")
            return nil
        }
        
        ctx["imgHeight"] = imgHeight
        ctx["imgWidth"] = imgWidth
        ctx["numPoints"] = numPoints
        ctx["globScale"] = globScale
        ctx["fx"] = fx
        ctx["fy"] = fy
        ctx["cx"] = cx
        ctx["cy"] = cy
        ctx["projectBuffers"] = [MTLBuffer](arrayLiteral: means, scales, quats, viewMat, projMat, cov3d, radii, conics)
        
        return [MTLBuffer?](arrayLiteral: xys, depths, radii, conics, numTilesHit, cov3d).compactMap { $0 }
    }
    
    func backward(ctx: inout [String: Any], grad_outputs: [MTLBuffer], mtlctx: MetalContext) -> [MTLBuffer]? {
        let v_xys = grad_outputs[0]
        let v_depths = grad_outputs[1]
        let v_radii = grad_outputs[2]
        let v_conics = grad_outputs[3]
        let v_numTiles = grad_outputs[4]
        let v_cov3d = grad_outputs[5]
        
        guard let imgHeight = ctx["imgHeight"] as? Int,
              let imgWidth = ctx["imgWidth"] as? Int,
              let numPoints = ctx["numPoints"] as? Int,
              let globScale = ctx["globScale"] as? Float,
              let fx = ctx["fx"] as? Float,
              let fy = ctx["fy"] as? Float,
              let cx = ctx["cx"] as? Float,
              let cy = ctx["cy"] as? Float,
              let projectBuffers = ctx["projectBuffers"] as? [MTLBuffer] else {
            print("Missing context data.")
            return nil
        }
        
        let means = projectBuffers[0]
        let scales = projectBuffers[1]
        let quats = projectBuffers[2]
        let viewMat = projectBuffers[3]
        let projMat = projectBuffers[4]
        let cov3d = projectBuffers[5]
        let radii = projectBuffers[6]
        let conics = projectBuffers[7]
        
        let t = project_gaussians_backward_tensor(num_points: numPoints, means3d: means, scales: scales, glob_scale: globScale, quats: quats, viewmat: viewMat, projmat: projMat, fx: fx, fy: fy, cx: cx, cy: cy, img_height: imgHeight, img_width: imgWidth, cov3d: cov3d, radii: radii, conics: conics, v_xy: v_xys, v_depth: v_depths, v_conic: v_conics, context: mtlctx)
        
        let v_means = t.2
        let v_scales = t.3
        let v_quats = t.4
        
        return [MTLBuffer?](arrayLiteral: v_means, v_scales, nil, v_quats, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil).compactMap { $0 }
    }
}

#endif

class ProjectGaussiansCPU {
    func apply(means: [[Float]], scales: [[Float]], globScale: Float, quats: [[Float]], viewMat: [[Float]], projMat: [[Float]], fx: Float, fy: Float, cx: Float, cy: Float, imgHeight: Int, imgWidth: Int, clipThresh: Float) -> ([[Float]], [Int], [[Float]], [[[Float]]], [Float]) {
        let numPoints = means.count
        
        let t = project_gaussians_forward_tensor_cpu(num_points: numPoints, means3d: means, scales: scales, glob_scale: globScale, quats: quats, viewmat: viewMat, projmat: projMat, fx: fx, fy: fy, cx: cx, cy: cy, img_height: imgHeight, img_width: imgWidth, clip_thresh: clipThresh)
        
        let xys = t.0
        let radii = t.1
        let conics = t.2
        let cov2d = t.3
        let camDepths = t.4
        
        return (xys, radii, conics, cov2d, camDepths)
    }
}
