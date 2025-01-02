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

func rgb2sh(rgb: MTLBuffer, device: MTLDevice, queue: MTLCommandQueue) -> MTLBuffer? {
    guard let library = device.makeDefaultLibrary(),
          let function = library.makeFunction(name: "rgb2sh"),
          let pipeline = try? device.makeComputePipelineState(function: function),
          let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
        print("Failed to set up Metal compute pipeline.")
        return nil
    }
    
    let length = rgb.length
    guard let resultBuffer = device.makeBuffer(length: length, options: .storageModeShared) else {
        print("Failed to create result buffer.")
        return nil
    }
    
    var c0: Float = Float(C0)
    guard let c0Buffer = device.makeBuffer(bytes: &c0, length: MemoryLayout<Float>.size, options: .storageModeShared) else {
        print("Failed to create constant buffer.")
        return nil
    }
    
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(rgb, offset: 0, index: 0)
    encoder.setBuffer(resultBuffer, offset: 0, index: 1)
    encoder.setBuffer(c0Buffer, offset: 0, index: 2)
    
    let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let threadGroups = MTLSize(width: (length + 255) / 256, height: 1, depth: 1)
    encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
    encoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    return resultBuffer
}

func sh2rgb(sh: MTLBuffer, device: MTLDevice, queue: MTLCommandQueue) -> MTLBuffer? {
    guard let library = device.makeDefaultLibrary(),
          let function = library.makeFunction(name: "sh2rgb"),
          let pipeline = try? device.makeComputePipelineState(function: function),
          let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
        print("Failed to set up Metal compute pipeline.")
        return nil
    }

    let length = sh.length
    guard let resultBuffer = device.makeBuffer(length: length, options: .storageModeShared) else {
        print("Failed to create result buffer.")
        return nil
    }

    var c0: Float = Float(C0)
    guard let c0Buffer = device.makeBuffer(bytes: &c0, length: MemoryLayout<Float>.size, options: .storageModeShared) else {
        print("Failed to create constant buffer.")
        return nil
    }

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(sh, offset: 0, index: 0)
    encoder.setBuffer(resultBuffer, offset: 0, index: 1)
    encoder.setBuffer(c0Buffer, offset: 0, index: 2)

    let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
    let threadGroups = MTLSize(width: (length + 255) / 256, height: 1, depth: 1)
    encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    return resultBuffer
}

#if METAL

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

#endif

class SphericalHarmonicsCPU {
    func apply(degreesToUse: Int, viewdirs: [[Float]], coeffs: [[Float]]) -> [[Float]] {
        let numPoints = coeffs.count
        let degree = degFromSh(numBases: coeffs[0].count)
        
        return compute_sh_forward_tensor_cpu(num_points: numPoints, degree: degree, degrees_to_use: degreesToUse, viewdirs: viewdirs, coeffs: coeffs)
    }
}
