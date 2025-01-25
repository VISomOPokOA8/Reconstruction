//
//  kdtree_tensor.swift
//  OpenSplat
//
//  Created by Yuxuan Zhang on 1/4/25.
//

import Foundation
import simd

class PointsTensor {
    var tensor: [SIMD3<Float>]
    private var kdTree: KdTree?

    init(tensor: [SIMD3<Float>]) {
        self.tensor = tensor
        self.kdTree = nil
    }
    
    deinit {
        
    }

    func scales() -> [Float] {
        if kdTree == nil {
            kdTree = KdTree(dimension: 3, points: tensor, maxLeaf: KDTREE_MAX_LEAF)
        }
        guard let kdTree = kdTree else {
            fatalError("KDTree initialization failed.")
        }
        var scales = [Float](repeating: 0.0, count: tensor.count)
        let count = 4

        var indices = [Int](repeating: 0, count: count)
        var sqrDists = [Float](repeating: 0.0, count: count)
        
        for i in 0..<tensor.count {
            kdTree.knnSearch(point: tensor[i], count: count, indices: &indices, sqrDists: &sqrDists)

            var sum: Float = 0.0
            for j in 1..<count {
                sum += sqrt(sqrDists[j])
            }
            scales[i] = sum / Float(count - 1)
        }

        return scales
    }

    func freeIndex() {
        kdTree = nil
    }
}

extension PointsTensor {
    func getIndex() -> KdTree? {
        if kdTree == nil {
            kdTree = buildIndex()
        }
        return kdTree
    }

    func buildIndex() -> KdTree {
        let tree = KdTree(dimension: 3, points: tensor, maxLeaf: KDTREE_MAX_LEAF)
        kdTree = tree
        return tree
    }

    func kdtree_get_point_count() -> size_t {
        return tensor.count
    }

    func kdtree_get_pt(idx: Int, dim: Int) -> Float {
        guard idx < tensor.count else {
            fatalError("Index out of bounds")
        }
        return tensor[idx][dim]
    }

    func kdtree_get_bbox<BBOX>(bb: inout BBOX) -> Bool {
        // KDTree 的边界盒功能可以视具体需求决定是否实现。
        // 这里返回 false，表示未定义边界盒。
        return false
    }
}

class KdTree {
    private var dimension: Int
    private var points: [SIMD3<Float>]
    private var maxLeaf: Int

    init(dimension: Int, points: [SIMD3<Float>], maxLeaf: Int) {
        self.dimension = dimension
        self.points = points
        self.maxLeaf = maxLeaf
    }

    func knnSearch(point: SIMD3<Float>, count: Int, indices: inout [Int], sqrDists: inout [Float]) {
        var distances = points.enumerated().map { (index, otherPoint) -> (Int, Float) in
            let dist = simd_distance_squared(point, otherPoint)
            return (index, dist)
        }
        distances.sort { $0.1 < $1.1 }

        for i in 0..<min(count, distances.count) {
            indices[i] = distances[i].0
            sqrDists[i] = distances[i].1
        }
    }
}
