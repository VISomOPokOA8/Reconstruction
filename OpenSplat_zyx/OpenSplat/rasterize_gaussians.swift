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
    let t = map_gaussian_to_intersects_tensor(num_points: numPoints,
                                              num_intersects: numIntersects,
                                              xys: xys,
                                              depths: depths,
                                              radii: radii,
                                              num_tile_hit: cumTilesHit,
                                              tile_bounds: tileBounds)
    
    let isectIds = t.0
    let gaussianIds = t.1
    
    let isectIdsSorted = isectIds.sorted()
    let sortedIndices = isectIds.enumerated().sorted { $0.element < $1.element }.map { $0.offset }
    let gaussianIdsSorted = sortedIndices.map { gaussianIds[$0] }
    
    let tileBins = get_tile_bin_edges_tensor(num_intersects: numIntersects, isect_ids_sorted: isectIdsSorted)
    return (isectIds, gaussianIds, isectIdsSorted, gaussianIdsSorted, tileBins)
}

class RasterizeGaussians {
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
                 background: SIMD3<Float>) {
        let numPoints = xys.count
        
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
        let gaussianIdsSorted = b.3
        let tileBins = b.4
        
        let t = 
    }
    
    func backward(ctx: inout [String: Any], grad_outputs) {
        
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

