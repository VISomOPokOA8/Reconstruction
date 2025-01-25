//
//  rasterize_gaussians.swift
//  OpenSplat
//
//  Created by Yuxuan Zhang on 25/12/24.
//

import Foundation
import simd

func binAndSortGaussians(numPoints: Int,
                         numIntersects: Int,
                         xys: [SIMD2<Float>],
                         depths: [Float],
                         radii: [Int],
                         cumTilesHit: [Int],
                         tileBounds: TileBounds) -> ([Int64],
                                                     [Int],
                                                     [Int64],
                                                     [Int],
                                                     [SIMD2<Int>]) {
    var xys = xys
    var depths = depths
    var radii = radii
    var cumTilesHit = cumTilesHit
    
    let t = map_gaussian_to_intersects_tensor(num_points: numPoints,
                                              num_intersects: numIntersects,
                                              xys: &xys,
                                              depths: &depths,
                                              radii: &radii,
                                              num_tile_hit: &cumTilesHit,
                                              tile_bounds: tileBounds)
    
    let isectIds = t.0
    let gaussianIds = t.1
    
    var isectIdsSorted = isectIds.sorted()
    let sortedIndices = isectIds.enumerated().sorted { $0.element < $1.element }.map { $0.offset }
    let gaussianIdsSorted = sortedIndices.map { gaussianIds[$0] }
    
    let tileBins = get_tile_bin_edges_tensor(num_intersects: numIntersects, isect_ids_sorted: &isectIdsSorted)
    return (isectIds, gaussianIds, isectIdsSorted, gaussianIdsSorted, tileBins)
}

class RasterizeGaussians {
    func apply(xys: [SIMD2<Float>],
               depths: [Float],
               radii: [Int],
               conics: [SIMD3<Float>],
               numTilesHit: [Int],
               colors: [SIMD3<Float>],
               opacity: [Float],
               imgHeight: Int,
               imgWidth: Int,
               background: SIMD3<Float>) -> [[SIMD3<Float>]] {
        let numPoints = xys.count
        var xys = xys
        var depths = depths
        var radii = radii
        var conics = conics
        var numTilesHit = numTilesHit
        var colors = colors
        var opacity = opacity
        var background = background
        
        let tileBounds: TileBounds = ((imgWidth + BLOCK_X - 1) / BLOCK_X,
                                      (imgHeight + BLOCK_Y - 1) / BLOCK_Y,
                                      1)
        let block = SIMD3<Int>(BLOCK_X, BLOCK_Y, 1)
        let imgSize = SIMD3<Int>(imgWidth, imgHeight, 1)
        
        let cumTilesHit = cumsum(array: numTilesHit)
        let numIntersects = cumTilesHit[cumTilesHit.count - 1]
        
        let b = binAndSortGaussians(numPoints: numPoints,
                                    numIntersects: numIntersects,
                                    xys: xys,
                                    depths: depths,
                                    radii: radii,
                                    cumTilesHit: cumTilesHit,
                                    tileBounds: tileBounds)
        var gaussianIdsSorted = b.3
        var tileBins = b.4
        
        let t = rasterize_forward_tensor(tile_bounds: tileBounds,
                                         block: block,
                                         img_size: imgSize,
                                         gaussian_ids_sorted: &gaussianIdsSorted,
                                         tile_bins: &tileBins,
                                         xys: &xys,
                                         conics: &conics,
                                         colors: &colors,
                                         opacities: &opacity,
                                         background: &background)
        let outImg = t.0
        let finalTs = t.1
        let finalIdx = t.2
        
        return outImg
    }
    
    func forward(ctx: inout [String: Any],
                 xys: [SIMD2<Float>],
                 depths: [Float],
                 radii: [Int],
                 conics: [SIMD3<Float>],
                 numTilesHit: [Int],
                 colors: [SIMD3<Float>],
                 opacity: [Float],
                 imgHeight: Int,
                 imgWidth: Int,
                 background: SIMD3<Float>) -> [[SIMD3<Float>]] {
        let numPoints = xys.count
        var xys = xys
        var depths = depths
        var radii = radii
        var conics = conics
        var numTilesHit = numTilesHit
        var colors = colors
        var opacity = opacity
        var background = background
        
        let tileBounds: TileBounds = ((imgWidth + BLOCK_X - 1) / BLOCK_X,
                                      (imgHeight + BLOCK_Y - 1) / BLOCK_Y,
                                      1)
        let block = SIMD3<Int>(BLOCK_X, BLOCK_Y, 1)
        let imgSize = SIMD3<Int>(imgWidth, imgHeight, 1)
        
        let cumTilesHit = cumsum(array: numTilesHit)
        let numIntersects = cumTilesHit[cumTilesHit.count - 1]
        
        let b = binAndSortGaussians(numPoints: numPoints,
                                    numIntersects: numIntersects,
                                    xys: xys,
                                    depths: depths,
                                    radii: radii,
                                    cumTilesHit: cumTilesHit,
                                    tileBounds: tileBounds)
        var gaussianIdsSorted = b.3
        var tileBins = b.4
        
        let t = rasterize_forward_tensor(tile_bounds: tileBounds,
                                         block: block,
                                         img_size: imgSize,
                                         gaussian_ids_sorted: &gaussianIdsSorted,
                                         tile_bins: &tileBins,
                                         xys: &xys,
                                         conics: &conics,
                                         colors: &colors,
                                         opacities: &opacity,
                                         background: &background)
        let outImg = t.0
        let finalTs = t.1
        let finalIdx = t.2
        
        ctx["imgWidth"] = imgWidth
        ctx["imgHeight"] = imgHeight
        ctx["save_for_backward"] = (gaussianIdsSorted, tileBins, xys, conics, colors, opacity, background, finalTs, finalIdx)
        
        return outImg
    }
    
    func backward(ctx: inout [String: Any], grad_outputs: ([[SIMD3<Float>]])) {
        var v_outImg = grad_outputs
        
        guard let imgHeight = ctx["imgHeight"] as? Int,
              let imgWidth = ctx["imgWidth"] as? Int,
              let saved = ctx["save_for_backward"] as? ([Int], [SIMD2<Int>], [SIMD2<Float>], [SIMD3<Float>], [SIMD3<Float>], [Float], SIMD3<Float>, [[Float]], [[Int]]) else {
            fatalError("Context does not contain expected data.")
        }
        
        var gaussianIdsSorted = saved.0
        var tileBins = saved.1
        var xys = saved.2
        var conics = saved.3
        var colors = saved.4
        var opacity = saved.5
        var background = saved.6
        var finalTs = saved.7
        var finalIdx = saved.8
        
        var v_outAlpha = v_outImg.map { $0.map { $0.x * 0 } }
        
        let t = rasterize_backward_tensor(img_height: imgHeight,
                                          img_width: imgWidth,
                                          gaussian_ids_sorted: &gaussianIdsSorted,
                                          tile_bins: &tileBins,
                                          xys: &xys,
                                          conics: &conics,
                                          colors: &colors,
                                          opacities: &opacity,
                                          background: &background,
                                          final_Ts: &finalTs,
                                          final_idx: &finalIdx,
                                          v_output: &v_outImg,
                                          v_output_alpha: &v_outAlpha)
    }
}

//
// Customized Functions
//

func cumsum(array: [Int]) -> [Int] {
    var result = [Int]()
    var runningSum = 0
    
    for value in array {
        runningSum += value
        result.append(runningSum)
    }
    return result
}

