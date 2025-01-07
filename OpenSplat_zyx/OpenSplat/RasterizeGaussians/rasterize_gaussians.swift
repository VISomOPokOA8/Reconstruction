//
//  rasterize_gaussians.swift
//  OpenSplat
//
//  Created by Yuxuan Zhang on 25/12/24.
//

import Foundation
import Metal

func binAndSortGaussians(numpoints: Int, numIntersects: Int, xys: MTLBuffer, depths: MTLBuffer, radii: MTLBuffer, numTilesHit: MTLBuffer, tileBounds: SIMD3<Int>, mtlctx: MetalContext) -> (MTLBuffer?, MTLBuffer?, MTLBuffer?, MTLBuffer?, MTLBuffer?) {
    let t = map_gaussian_to_intersects_tensor(num_points: numpoints, num_intersects: numIntersects, xys: xys, depths: depths, radii: radii, num_tiles_hit: numTilesHit, tile_bounds: tileBounds, context: mtlctx)
    guard let isectIds = t.0, let gaussianIds = t.1 else {
        print("Failed to map gaussian to intersects tensor.")
        return (nil, nil, nil, nil, nil)
    }
    
    let sorted = tensor_sort(input: isectIds, device: mtlctx.device, queue: mtlctx.queue)
    guard let isectIdsSorted = sorted.0, let sortedIndices = sorted.1 else {
        print("Failed to sort intersection IDs.")
        return (nil, nil, nil, nil, nil)
    }

    let gaussianIdsSorted = tensor_gather(input: gaussianIds, indices: sortedIndices, device: mtlctx.device, queue: mtlctx.queue)
    let tileBins = get_tile_bin_edges_tensor(num_intersects: numIntersects, isect_ids_sorted: isectIdsSorted, context: mtlctx)
    
    return (isectIds, gaussianIds, isectIdsSorted, gaussianIdsSorted, tileBins)
}

class RasterizeGaussians {
    func forward(ctx: inout [String: Any], xys: MTLBuffer, depths: MTLBuffer, radii: MTLBuffer, conics: MTLBuffer, numTilesHit: MTLBuffer, colors: MTLBuffer, opacity: MTLBuffer, imgHeight: Int, imgWidth: Int, background: MTLBuffer, mtlctx: MetalContext) -> MTLBuffer? {
        let numPoints = xys.length / (2 * MemoryLayout<Float>.stride)

        let tileBounds = SIMD3<Int>(
            (imgWidth + 15) / 16,
            (imgHeight + 15) / 16,
            1
        )
        let block = SIMD3<Int>(16, 16, 1)
        let imgSize = SIMD3<Int>(imgWidth, imgHeight, 1)

        guard let numTileBounds = tensor_cumsum(input: numTilesHit, device: mtlctx.device, queue: mtlctx.queue) else {
            print("Failed to compute cumulative sum of tiles hit.")
            return nil
        }

        let numIntersects = numTileBounds.contents().assumingMemoryBound(to: Int32.self)[numTileBounds.length / MemoryLayout<Int32>.stride - 1]

        let b = binAndSortGaussians(numpoints: numPoints, numIntersects: Int(numIntersects), xys: xys, depths: depths, radii: radii, numTilesHit: numTileBounds, tileBounds: tileBounds, mtlctx: mtlctx)
        guard let gaussianIdsSorted = b.3, let tileBins = b.4 else {
            print("Failed to bin and sort Gaussians.")
            return nil
        }

        let t = rasterize_forward_tensor(tile_bounds: tileBounds, block: block, img_size: imgSize, gaussian_ids_sorted: gaussianIdsSorted, tile_bins: tileBins, xys: xys, conics: conics, colors: colors, opacities: opacity, background: background, context: mtlctx)

        guard let outImg = t.0, let finalTs = t.1, let finalIdx = t.2 else {
            print("Failed to rasterize forward tensor.")
            return nil
        }

        ctx["imgHeight"] = imgHeight
        ctx["imgWidth"] = imgWidth
        ctx["rasterizeBuffers"] = [MTLBuffer](arrayLiteral: gaussianIdsSorted, tileBins, xys, conics, colors, opacity, background, finalTs, finalIdx)

        return outImg
    }
    
    func backward(ctx: inout [String: Any], grad_outputs: [MTLBuffer], mtlctx: MetalContext) -> [MTLBuffer]? {
        guard let v_outImg = grad_outputs.first else {
            print("grad_outputs must contain at least one buffer.")
            return nil
        }
        
        guard let imgHeight = ctx["imgHeight"] as? Int,
              let imgWidth = ctx["imgWidth"] as? Int,
              let rasterizeBuffers = ctx["rasterizeBuffers"] as? [MTLBuffer] else {
            print("Missing context data.")
            return nil
        }
        
        let gaussianIdsSorted = rasterizeBuffers[0]
        let tileBins = rasterizeBuffers[1]
        let xys = rasterizeBuffers[2]
        let conics = rasterizeBuffers[3]
        let colors = rasterizeBuffers[4]
        let opacity = rasterizeBuffers[5]
        let background = rasterizeBuffers[6]
        let finalTs = rasterizeBuffers[7]
        let finalIdx = rasterizeBuffers[8]
        
        let v_outAlpha = mtlctx.device.makeBuffer(length: v_outImg.length * MemoryLayout<Float>.size, options: .storageModeShared)
        
        guard let v_outAlpha = v_outAlpha else {
            print("Failed to allocate v_outAlpha buffer.")
            return nil
        }

        let t = rasterize_backward_tensor(img_height: imgHeight, img_width: imgWidth, gaussian_ids_sorted: gaussianIdsSorted, tile_bins: tileBins, xys: xys, conics: conics, colors: colors, opacities: opacity, background: background, final_Ts: finalTs, final_idx: finalIdx, v_output: v_outImg, v_output_alpha: v_outAlpha, context: mtlctx)
        
        let v_xy = t.0
        let v_conic = t.1
        let v_colors = t.2
        let v_opacity = t.3

        return [MTLBuffer?](arrayLiteral: v_xy, nil, nil, v_conic, nil, v_colors, v_opacity, nil, nil, nil).compactMap { $0 }
    }
}

//
//  Customized Functions
//

func tensor_sort(input: MTLBuffer, device: MTLDevice, queue: MTLCommandQueue) -> (MTLBuffer?, MTLBuffer?) {
    guard let library = device.makeDefaultLibrary(),
          let function = library.makeFunction(name: "tensor_sort"),
          let pipeline = try? device.makeComputePipelineState(function: function),
          let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
        print("Failed to set up Metal compute pipeline for tensor_sort.")
        return (nil, nil)
    }

    let length = input.length / MemoryLayout<Float>.size
    guard let sortedBuffer = device.makeBuffer(length: input.length, options: .storageModeShared),
          let indicesBuffer = device.makeBuffer(length: length * MemoryLayout<UInt32>.size, options: .storageModeShared) else {
        print("Failed to create buffers for tensor_sort outputs.")
        return (nil, nil)
    }

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(input, offset: 0, index: 0)
    encoder.setBuffer(sortedBuffer, offset: 0, index: 1)
    encoder.setBuffer(indicesBuffer, offset: 0, index: 2)

    let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let threadGroups = MTLSize(width: (length + 255) / 256, height: 1, depth: 1)
    encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    return (sortedBuffer, indicesBuffer)
}

func tensor_gather(input: MTLBuffer, indices: MTLBuffer, device: MTLDevice, queue: MTLCommandQueue) -> MTLBuffer? {
    guard let library = device.makeDefaultLibrary(),
          let function = library.makeFunction(name: "tensor_gather"),
          let pipeline = try? device.makeComputePipelineState(function: function),
          let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
        print("Failed to set up Metal compute pipeline for tensor_gather.")
        return nil
    }

    let length = indices.length / MemoryLayout<UInt32>.size
    guard let outputBuffer = device.makeBuffer(length: length * MemoryLayout<Float>.size, options: .storageModeShared) else {
        print("Failed to create buffer for tensor_gather output.")
        return nil
    }

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(input, offset: 0, index: 0)
    encoder.setBuffer(indices, offset: 0, index: 1)
    encoder.setBuffer(outputBuffer, offset: 0, index: 2)

    let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let threadGroups = MTLSize(width: (length + 255) / 256, height: 1, depth: 1)
    encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    return outputBuffer
}

func tensor_cumsum(input: MTLBuffer, device: MTLDevice, queue: MTLCommandQueue) -> MTLBuffer? {
    guard let library = device.makeDefaultLibrary(),
          let function = library.makeFunction(name: "tensor_cumsum"),
          let pipeline = try? device.makeComputePipelineState(function: function),
          let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
        print("Failed to set up Metal compute pipeline for tensor_cumsum.")
        return nil
    }

    let length = input.length / MemoryLayout<Float>.size
    guard let outputBuffer = device.makeBuffer(length: input.length, options: .storageModeShared) else {
        print("Failed to create buffer for tensor_cumsum output.")
        return nil
    }

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(input, offset: 0, index: 0)
    encoder.setBuffer(outputBuffer, offset: 0, index: 1)

    let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let threadGroups = MTLSize(width: (length + 255) / 256, height: 1, depth: 1)
    encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    return outputBuffer
}
