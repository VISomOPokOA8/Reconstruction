//
//  gsplat_metal.swift
//  OpenSplat
//
//  Created by Yuxuan Zhang on 24/12/24.
//

import Foundation
import simd
import Metal

struct MetalContext {
    var device: MTLDevice
    var queue: MTLCommandQueue
    var d_queue: DispatchQueue
    
    var nd_rasterize_backward_kernel_cpso: MTLComputePipelineState?
    var nd_rasterize_forward_kernel_cpso: MTLComputePipelineState?
    var rasterize_backward_kernel_cpso: MTLComputePipelineState?
    var project_gaussians_forward_kernel_cpso: MTLComputePipelineState?
    var project_gaussians_backward_kernel_cpso: MTLComputePipelineState?
    var compute_sh_forward_kernel_cpso: MTLComputePipelineState?
    var compute_sh_backward_kernel_cpso: MTLComputePipelineState?
    var compute_cov2d_bounds_kernel_cpso: MTLComputePipelineState?
    var map_gaussian_to_intersects_kernel_cpso: MTLComputePipelineState?
    var get_tile_bin_edges_kernel_cpso: MTLComputePipelineState?
}

func num_sh_bases(degree: Int) -> Int {
    switch degree {
    case 0:
        return 1
    case 1:
        return 4
    case 2:
        return 9
    case 3:
        return 16
    default:
        return 25
    }
}

func init_gsplat_metal_context() -> MetalContext? {
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else {
        print("Metal device or command queue initialization failed.")
        return nil
    }

    let d_queue = DispatchQueue(label: "com.opensplat.metal.dispatch", attributes: .concurrent)
    var context = MetalContext(device: device, queue: commandQueue, d_queue: d_queue)

    var metal_library: MTLLibrary?
    if let path = Bundle.main.path(forResource: "default", ofType: "metallib"),
       let library = try? device.makeLibrary(filepath: path) {
        print("Loaded Metal library from precompiled metallib.")
        metal_library = library
    } else {
        print("Precompiled Metal library not found. Attempting to load source.")
        if let source_path = Bundle.main.path(forResource: "gsplat_metal", ofType: "metal"),
           let source_code = try? String(contentsOfFile: source_path) {
            metal_library = try? device.makeLibrary(source: source_code, options: nil)
        } else {
            print("Failed to load Metal library from source.")
            return nil
        }
    }

    guard let library = metal_library else {
        print("Failed to load Metal library.")
        return nil
    }

    context.nd_rasterize_backward_kernel_cpso = load_pipeline(device: device, library: library, functionName: "nd_rasterize_backward_kernel")
    context.nd_rasterize_forward_kernel_cpso = load_pipeline(device: device, library: library, functionName: "nd_rasterize_forward_kernel")
    context.rasterize_backward_kernel_cpso = load_pipeline(device: device, library: library, functionName: "rasterize_backward_kernel")
    context.project_gaussians_forward_kernel_cpso = load_pipeline(device: device, library: library, functionName: "project_gaussians_forward_kernel")
    context.project_gaussians_backward_kernel_cpso = load_pipeline(device: device, library: library, functionName: "project_gaussians_backward_kernel")
    context.compute_sh_forward_kernel_cpso = load_pipeline(device: device, library: library, functionName: "compute_sh_forward_kernel")
    context.compute_sh_backward_kernel_cpso = load_pipeline(device: device, library: library, functionName: "compute_sh_backward_kernel")
    context.compute_cov2d_bounds_kernel_cpso = load_pipeline(device: device, library: library, functionName: "compute_cov2d_bounds_kernel")
    context.map_gaussian_to_intersects_kernel_cpso = load_pipeline(device: device, library: library, functionName: "map_gaussian_to_intersects_kernel")
    context.get_tile_bin_edges_kernel_cpso = load_pipeline(device: device, library: library, functionName: "get_tile_bin_edges_kernel")

    return context
}

//
//  GSPLAT_METAL_ADD_KERNEL
//

func load_pipeline(device: MTLDevice, library: MTLLibrary, functionName: String) -> MTLComputePipelineState? {
    do {
        guard let function = library.makeFunction(name: functionName) else {
            print("Failed to create function \(functionName) from Metal library.")
            return nil
        }
        return try device.makeComputePipelineState(function: function)
    } catch {
        print("Failed to create pipeline state for \(functionName): \(error)")
        return nil
    }
}

//
//  END
//

func get_global_context() -> MetalContext? {
    var context: MetalContext? = nil
    if context == nil {
        context = init_gsplat_metal_context()
    }
    return context!
}

enum EncodeType {
    case float
    case int
    case uint
    case array
    case tensor
}

struct EncodeArg {
    var type: EncodeType
    var fScalar: Float?
    var i32Scalar: Int32?
    var u32Scalar: UInt32?
    var array: UnsafeRawPointer?
    var arrayNumBytes: Int?
    var tensor: Any? // Replace with a specific tensor type if available

    static func scalar(_ value: Float) -> EncodeArg {
        return EncodeArg(type: .float, fScalar: value, i32Scalar: nil, u32Scalar: nil, array: nil, arrayNumBytes: nil, tensor: nil)
    }

    static func scalar(_ value: Int32) -> EncodeArg {
        return EncodeArg(type: .int, fScalar: nil, i32Scalar: value, u32Scalar: nil, array: nil, arrayNumBytes: nil, tensor: nil)
    }

    static func scalar(_ value: UInt32) -> EncodeArg {
        return EncodeArg(type: .uint, fScalar: nil, i32Scalar: nil, u32Scalar: value, array: nil, arrayNumBytes: nil, tensor: nil)
    }

    static func array(_ value: UnsafeRawPointer, numBytes: Int) -> EncodeArg {
        return EncodeArg(type: .array, fScalar: nil, i32Scalar: nil, u32Scalar: nil, array: value, arrayNumBytes: numBytes, tensor: nil)
    }

    static func tensor(_ value: Any) -> EncodeArg {
        return EncodeArg(type: .tensor, fScalar: nil, i32Scalar: nil, u32Scalar: nil, array: nil, arrayNumBytes: nil, tensor: value)
    }
}

func dispatchPipeline(context: MetalContext, cpso: MTLComputePipelineState, gridSize: MTLSize, threadGroupSize: MTLSize, args: [EncodeArg]) {
    guard let commandBuffer = context.queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
        print("Failed to create command buffer or encoder.")
        return
    }

    encoder.setComputePipelineState(cpso)

    for (index, arg) in args.enumerated() {
        switch arg.type {
        case .float:
            if let value = arg.fScalar {
                var mutableValue = value
                encoder.setBytes(&mutableValue, length: MemoryLayout<Float>.size, index: index)
            }
        case .int:
            if let value = arg.i32Scalar {
                var mutableValue = value
                encoder.setBytes(&mutableValue, length: MemoryLayout<Int32>.size, index: index)
            }
        case .uint:
            if let value = arg.u32Scalar {
                var mutableValue = value
                encoder.setBytes(&mutableValue, length: MemoryLayout<UInt32>.size, index: index)
            }
        case .array:
            if let pointer = arg.array, let length = arg.arrayNumBytes {
                encoder.setBytes(pointer, length: length, index: index)
            }
        case .tensor:
            if let buffer = arg.tensor as? MTLBuffer {
                encoder.setBuffer(buffer, offset: 0, index: index)
            }
        }
    }

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
}

func compute_cov2d_bounds_tensor(num_pts: Int, covs2d: MTLBuffer, context: MetalContext) -> (MTLBuffer, MTLBuffer)? {
    guard let computePipeline = context.compute_cov2d_bounds_kernel_cpso else {
        print("Compute pipeline for compute_cov2d_bounds_kernel is not initialized.")
        return nil
    }

    let device = context.device

    let conicsBuffer = device.makeBuffer(length: num_pts * MemoryLayout<Float>.size * 2, options: .storageModeShared)
    let radiiBuffer = device.makeBuffer(length: num_pts * MemoryLayout<Float>.size, options: .storageModeShared)

    guard let conicsBuffer = conicsBuffer, let radiiBuffer = radiiBuffer else {
        print("Failed to create buffers for output.")
        return nil
    }

    let gridSize = MTLSize(width: num_pts, height: 1, depth: 1)
    let threadGroupSize = MTLSize(width: min(computePipeline.maxTotalThreadsPerThreadgroup, num_pts), height: 1, depth: 1)

    dispatchPipeline(context: context, cpso: computePipeline, gridSize: gridSize, threadGroupSize: threadGroupSize, args: [
        EncodeArg.scalar(Float(num_pts)),
        EncodeArg.tensor(covs2d),
        EncodeArg.tensor(conicsBuffer),
        EncodeArg.tensor(radiiBuffer)
    ])

    return (conicsBuffer, radiiBuffer)
}

func compute_sh_forward_tensor(num_points: Int, degree: Int, degrees_to_use: Int, viewdirs: MTLBuffer, coeffs: MTLBuffer, context: MetalContext) -> MTLBuffer? {
    guard let computePipeline = context.compute_sh_forward_kernel_cpso else {
        print("Compute pipeline for compute_sh_forward_kernel is not initialized.")
        return nil
    }

    let device = context.device
    let num_bases = num_sh_bases(degree: degree)

    let colors = device.makeBuffer(length: num_points * MemoryLayout<Float>.size * 3, options: .storageModeShared)

    guard let colors = colors else {
        print("Failed to create buffer for output colors.")
        return nil
    }

    let gridSize = MTLSize(width: num_points, height: 1, depth: 1)
    let threadGroupSize = MTLSize(width: min(computePipeline.maxTotalThreadsPerThreadgroup, num_points), height: 1, depth: 1)

    dispatchPipeline(context: context, cpso: computePipeline, gridSize: gridSize, threadGroupSize: threadGroupSize, args: [
        EncodeArg.scalar(Float(num_points)),
        EncodeArg.scalar(Float(degree)),
        EncodeArg.scalar(Float(degrees_to_use)),
        EncodeArg.tensor(viewdirs),
        EncodeArg.tensor(coeffs),
        EncodeArg.tensor(colors)
    ])

    return colors
}

func compute_sh_backward_tensor(num_points: Int, degree: Int, degrees_to_use: Int, viewdirs: MTLBuffer, v_colors: MTLBuffer, context: MetalContext) -> MTLBuffer? {
    guard let computePipeline = context.compute_sh_backward_kernel_cpso else {
        print("Compute pipeline for compute_sh_backward_kernel is not initialized.")
        return nil
    }

    let device = context.device
    let num_bases = num_sh_bases(degree: degree)

    let v_coeffs = device.makeBuffer(length: num_points * num_bases * MemoryLayout<Float>.size * 3, options: .storageModeShared)

    guard let v_coeffs = v_coeffs else {
        print("Failed to create buffer for output v_coeffs.")
        return nil
    }

    let gridSize = MTLSize(width: num_points, height: 1, depth: 1)
    let threadGroupSize = MTLSize(width: min(computePipeline.maxTotalThreadsPerThreadgroup, num_points), height: 1, depth: 1)

    dispatchPipeline(context: context, cpso: computePipeline, gridSize: gridSize, threadGroupSize: threadGroupSize, args: [
        EncodeArg.scalar(Float(num_points)),
        EncodeArg.scalar(Float(degree)),
        EncodeArg.scalar(Float(degrees_to_use)),
        EncodeArg.tensor(viewdirs),
        EncodeArg.tensor(v_colors),
        EncodeArg.tensor(v_coeffs)
    ])

    return v_coeffs
}

func project_gaussians_forward_tensor(num_points: Int, means3d: MTLBuffer, scales: MTLBuffer, glob_scale: Float, quats: MTLBuffer, viewmat: MTLBuffer, projmat: MTLBuffer, fx: Float, fy: Float, cx: Float, cy: Float, img_height: Int, img_width: Int, tile_bounds: SIMD3<Int>, clip_thresh: Float, context: MetalContext) -> (MTLBuffer?, MTLBuffer?, MTLBuffer?, MTLBuffer?, MTLBuffer?, MTLBuffer?) {
    guard let computePipeline = context.project_gaussians_forward_kernel_cpso else {
        print("Compute pipeline for project_gaussians_forward_kernel is not initialized.")
        return (nil, nil, nil, nil, nil, nil)
    }

    let device = context.device

    let cov3d_d = device.makeBuffer(length: num_points * 6 * MemoryLayout<Float>.size, options: .storageModeShared)
    let xys_d = device.makeBuffer(length: num_points * 2 * MemoryLayout<Float>.size, options: .storageModeShared)
    let depths_d = device.makeBuffer(length: num_points * MemoryLayout<Float>.size, options: .storageModeShared)
    let radii_d = device.makeBuffer(length: num_points * MemoryLayout<Int32>.size, options: .storageModeShared)
    let conics_d = device.makeBuffer(length: num_points * 3 * MemoryLayout<Float>.size, options: .storageModeShared)
    let num_tiles_hit_d = device.makeBuffer(length: num_points * MemoryLayout<Int32>.size, options: .storageModeShared)

    guard let cov3d_d = cov3d_d, let xys_d = xys_d, let depths_d = depths_d, let radii_d = radii_d, let conics_d = conics_d, let num_tiles_hit_d = num_tiles_hit_d else {
        print("Failed to create buffers for outputs.")
        return (nil, nil, nil, nil, nil, nil)
    }

    var intrins: [Float] = [fx, fy, cx, cy]
    var img_size: [UInt32] = [UInt32(img_width), UInt32(img_height)]
    var tile_bounds_arr: [UInt32] = [UInt32(tile_bounds.x), UInt32(tile_bounds.y), UInt32(tile_bounds.z), 0xDEAD]

    let gridSize = MTLSize(width: num_points, height: 1, depth: 1)
    let threadGroupSize = MTLSize(width: min(computePipeline.maxTotalThreadsPerThreadgroup, num_points), height: 1, depth: 1)

    dispatchPipeline(context: context, cpso: computePipeline, gridSize: gridSize, threadGroupSize: threadGroupSize, args: [
        EncodeArg.scalar(Float(num_points)),
        EncodeArg.tensor(means3d),
        EncodeArg.tensor(scales),
        EncodeArg.scalar(glob_scale),
        EncodeArg.tensor(quats),
        EncodeArg.tensor(viewmat),
        EncodeArg.tensor(projmat),
        EncodeArg.array(&intrins, numBytes: intrins.count * MemoryLayout<Float>.size),
        EncodeArg.array(&img_size, numBytes: img_size.count * MemoryLayout<UInt32>.size),
        EncodeArg.array(&tile_bounds_arr, numBytes: tile_bounds_arr.count * MemoryLayout<UInt32>.size),
        EncodeArg.scalar(clip_thresh),
        EncodeArg.tensor(cov3d_d),
        EncodeArg.tensor(xys_d),
        EncodeArg.tensor(depths_d),
        EncodeArg.tensor(radii_d),
        EncodeArg.tensor(conics_d),
        EncodeArg.tensor(num_tiles_hit_d)
    ])

    return (cov3d_d, xys_d, depths_d, radii_d, conics_d, num_tiles_hit_d)
}

func project_gaussians_backward_tensor(num_points: Int, means3d: MTLBuffer, scales: MTLBuffer, glob_scale: Float, quats: MTLBuffer, viewmat: MTLBuffer, projmat: MTLBuffer, fx: Float, fy: Float, cx: Float, cy: Float, img_height: Int, img_width: Int, cov3d: MTLBuffer, radii: MTLBuffer, conics: MTLBuffer, v_xy: MTLBuffer, v_depth: MTLBuffer, v_conic: MTLBuffer, context: MetalContext) -> (MTLBuffer?, MTLBuffer?, MTLBuffer?, MTLBuffer?, MTLBuffer?) {
    guard let computePipeline = context.project_gaussians_backward_kernel_cpso else {
        print("Compute pipeline for project_gaussians_backward_kernel is not initialized.")
        return (nil, nil, nil, nil, nil)
    }

    let device = context.device

    let v_cov2d = device.makeBuffer(length: num_points * 3 * MemoryLayout<Float>.size, options: .storageModeShared)
    let v_cov3d = device.makeBuffer(length: num_points * 6 * MemoryLayout<Float>.size, options: .storageModeShared)
    let v_mean3d = device.makeBuffer(length: num_points * 3 * MemoryLayout<Float>.size, options: .storageModeShared)
    let v_scale = device.makeBuffer(length: num_points * 3 * MemoryLayout<Float>.size, options: .storageModeShared)
    let v_quat = device.makeBuffer(length: num_points * 4 * MemoryLayout<Float>.size, options: .storageModeShared)

    guard let v_cov2d = v_cov2d, let v_cov3d = v_cov3d, let v_mean3d = v_mean3d, let v_scale = v_scale, let v_quat = v_quat else {
        print("Failed to create buffers for outputs.")
        return (nil, nil, nil, nil, nil)
    }

    var intrins: [Float] = [fx, fy, cx, cy]
    var img_size: [UInt32] = [UInt32(img_width), UInt32(img_height)]

    let gridSize = MTLSize(width: num_points, height: 1, depth: 1)
    let threadGroupSize = MTLSize(width: min(computePipeline.maxTotalThreadsPerThreadgroup, num_points), height: 1, depth: 1)

    dispatchPipeline(context: context, cpso: computePipeline, gridSize: gridSize, threadGroupSize: threadGroupSize, args: [
        EncodeArg.scalar(Float(num_points)),
        EncodeArg.tensor(means3d),
        EncodeArg.tensor(scales),
        EncodeArg.scalar(glob_scale),
        EncodeArg.tensor(quats),
        EncodeArg.tensor(viewmat),
        EncodeArg.tensor(projmat),
        EncodeArg.array(&intrins, numBytes: intrins.count * MemoryLayout<Float>.size),
        EncodeArg.array(&img_size, numBytes: img_size.count * MemoryLayout<UInt32>.size),
        EncodeArg.tensor(cov3d),
        EncodeArg.tensor(radii),
        EncodeArg.tensor(conics),
        EncodeArg.tensor(v_xy),
        EncodeArg.tensor(v_depth),
        EncodeArg.tensor(v_conic),
        EncodeArg.tensor(v_cov2d),
        EncodeArg.tensor(v_cov3d),
        EncodeArg.tensor(v_mean3d),
        EncodeArg.tensor(v_scale),
        EncodeArg.tensor(v_quat)
    ])

    return (v_cov2d, v_cov3d, v_mean3d, v_scale, v_quat)
}

func map_gaussian_to_intersects_tensor(num_points: Int, num_intersects: Int, xys: MTLBuffer, depths: MTLBuffer, radii: MTLBuffer, num_tiles_hit: MTLBuffer, tile_bounds: SIMD3<Int>, context: MetalContext) -> (MTLBuffer?, MTLBuffer?) {
    guard let computePipeline = context.map_gaussian_to_intersects_kernel_cpso else {
        print("Compute pipeline for map_gaussian_to_intersects_kernel is not initialized.")
        return (nil, nil)
    }

    let device = context.device

    let gaussian_ids_unsorted = device.makeBuffer(length: num_intersects * MemoryLayout<Int32>.size, options: .storageModeShared)
    let isect_ids_unsorted = device.makeBuffer(length: num_intersects * MemoryLayout<Int64>.size, options: .storageModeShared)

    guard let gaussian_ids_unsorted = gaussian_ids_unsorted, let isect_ids_unsorted = isect_ids_unsorted else {
        print("Failed to create buffers for outputs.")
        return (nil, nil)
    }

    var tile_bounds_arr: [UInt32] = [UInt32(tile_bounds.x), UInt32(tile_bounds.y), UInt32(tile_bounds.z), 0xDEAD]

    let gridSize = MTLSize(width: num_points, height: 1, depth: 1)
    let threadGroupSize = MTLSize(width: min(computePipeline.maxTotalThreadsPerThreadgroup, num_points), height: 1, depth: 1)

    dispatchPipeline(context: context, cpso: computePipeline, gridSize: gridSize, threadGroupSize: threadGroupSize, args: [
        EncodeArg.scalar(Float(num_points)),
        EncodeArg.tensor(xys),
        EncodeArg.tensor(depths),
        EncodeArg.tensor(radii),
        EncodeArg.tensor(num_tiles_hit),
        EncodeArg.array(&tile_bounds_arr, numBytes: tile_bounds_arr.count * MemoryLayout<UInt32>.size),
        EncodeArg.tensor(isect_ids_unsorted),
        EncodeArg.tensor(gaussian_ids_unsorted)
    ])

    return (isect_ids_unsorted, gaussian_ids_unsorted)
}

func get_tile_bin_edges_tensor(num_intersects: Int, isect_ids_sorted: MTLBuffer, context: MetalContext) -> MTLBuffer? {
    guard let computePipeline = context.get_tile_bin_edges_kernel_cpso else {
        print("Compute pipeline for get_tile_bin_edges_kernel is not initialized.")
        return nil
    }

    let device = context.device

    let tile_bins = device.makeBuffer(length: num_intersects * 2 * MemoryLayout<Int32>.size, options: .storageModeShared)

    guard let tile_bins = tile_bins else {
        print("Failed to create buffer for output tile_bins.")
        return nil
    }

    let gridSize = MTLSize(width: num_intersects, height: 1, depth: 1)
    let threadGroupSize = MTLSize(width: min(computePipeline.maxTotalThreadsPerThreadgroup, num_intersects), height: 1, depth: 1)

    dispatchPipeline(context: context, cpso: computePipeline, gridSize: gridSize, threadGroupSize: threadGroupSize, args: [
        EncodeArg.scalar(Float(num_intersects)),
        EncodeArg.tensor(isect_ids_sorted),
        EncodeArg.tensor(tile_bins)
    ])

    return tile_bins
}

func rasterize_forward_tensor(tile_bounds: SIMD3<Int>, block: SIMD3<Int>, img_size: SIMD3<Int>, gaussian_ids_sorted: MTLBuffer, tile_bins: MTLBuffer, xys: MTLBuffer, conics: MTLBuffer, colors: MTLBuffer, opacities: MTLBuffer, background: MTLBuffer, context: MetalContext) -> (MTLBuffer?, MTLBuffer?, MTLBuffer?) {
    guard let computePipeline = context.nd_rasterize_forward_kernel_cpso else {
        print("Compute pipeline for rasterize_forward_tensor is not initialized.")
        return (nil, nil, nil)
    }

    let device = context.device

    let img_width = img_size.x
    let img_height = img_size.y
    let channels = colors.length / (img_width * img_height * MemoryLayout<Float>.stride)

    let out_img = device.makeBuffer(length: img_width * img_height * channels * MemoryLayout<Float>.stride, options: .storageModeShared)
    let final_Ts = device.makeBuffer(length: img_width * img_height * MemoryLayout<Float>.stride, options: .storageModeShared)
    let final_idx = device.makeBuffer(length: img_width * img_height * MemoryLayout<Int32>.stride, options: .storageModeShared)

    guard let out_img = out_img, let final_Ts = final_Ts, let final_idx = final_idx else {
        print("Failed to create buffers for outputs.")
        return (nil, nil, nil)
    }

    var img_size_dim3: [UInt32] = [UInt32(img_size.x), UInt32(img_size.y), UInt32(img_size.z), 0xDEAD]
    var tile_bounds_arr: [UInt32] = [UInt32(tile_bounds.x), UInt32(tile_bounds.y), UInt32(tile_bounds.z), 0xDEAD]
    var block_size_dim2: [Int32] = [Int32(block.x), Int32(block.y)]

    let gridSize = MTLSize(width: img_width, height: img_height, depth: 1)
    let threadGroupSize = MTLSize(width: block.x, height: block.y, depth: 1)

    dispatchPipeline(context: context, cpso: computePipeline, gridSize: gridSize, threadGroupSize: threadGroupSize, args: [
        EncodeArg.array(&tile_bounds_arr, numBytes: tile_bounds_arr.count * MemoryLayout<UInt32>.size),
        EncodeArg.array(&img_size_dim3, numBytes: img_size_dim3.count * MemoryLayout<UInt32>.size),
        EncodeArg.scalar(Float(channels)),
        EncodeArg.tensor(gaussian_ids_sorted),
        EncodeArg.tensor(tile_bins),
        EncodeArg.tensor(xys),
        EncodeArg.tensor(conics),
        EncodeArg.tensor(colors),
        EncodeArg.tensor(opacities),
        EncodeArg.tensor(final_Ts),
        EncodeArg.tensor(final_idx),
        EncodeArg.tensor(out_img),
        EncodeArg.tensor(background),
        EncodeArg.array(&block_size_dim2, numBytes: block_size_dim2.count * MemoryLayout<Int32>.size)
    ])

    return (out_img, final_Ts, final_idx)
}

func nd_rasterize_forward_tensor(tile_bounds: SIMD3<Int>, block: SIMD3<Int>, img_size: SIMD3<Int>, gaussian_ids_sorted: MTLBuffer, tile_bins: MTLBuffer, xys: MTLBuffer, conics: MTLBuffer, colors: MTLBuffer, opacities: MTLBuffer, background: MTLBuffer, context: MetalContext) -> (MTLBuffer?, MTLBuffer?, MTLBuffer?) {
    guard let computePipeline = context.nd_rasterize_forward_kernel_cpso else {
        print("Compute pipeline for nd_rasterize_forward_tensor is not initialized.")
        return (nil, nil, nil)
    }

    let device = context.device

    let img_width = img_size.x
    let img_height = img_size.y
    let channels = colors.length / (img_width * img_height * MemoryLayout<Float>.stride)

    let out_img = device.makeBuffer(length: img_width * img_height * channels * MemoryLayout<Float>.stride, options: .storageModeShared)
    let final_Ts = device.makeBuffer(length: img_width * img_height * MemoryLayout<Float>.stride, options: .storageModeShared)
    let final_idx = device.makeBuffer(length: img_width * img_height * MemoryLayout<Int32>.stride, options: .storageModeShared)

    guard let out_img = out_img, let final_Ts = final_Ts, let final_idx = final_idx else {
        print("Failed to create buffers for outputs.")
        return (nil, nil, nil)
    }

    var img_size_dim3: [UInt32] = [UInt32(img_size.x), UInt32(img_size.y), UInt32(img_size.z), 0xDEAD]
    var tile_bounds_arr: [UInt32] = [UInt32(tile_bounds.x), UInt32(tile_bounds.y), UInt32(tile_bounds.z), 0xDEAD]
    var block_size_dim2: [Int32] = [Int32(block.x), Int32(block.y)]

    let gridSize = MTLSize(width: img_width, height: img_height, depth: 1)
    let threadGroupSize = MTLSize(width: block.x, height: block.y, depth: 1)

    dispatchPipeline(context: context, cpso: computePipeline, gridSize: gridSize, threadGroupSize: threadGroupSize, args: [
        EncodeArg.array(&tile_bounds_arr, numBytes: tile_bounds_arr.count * MemoryLayout<UInt32>.size),
        EncodeArg.array(&img_size_dim3, numBytes: img_size_dim3.count * MemoryLayout<UInt32>.size),
        EncodeArg.scalar(Float(channels)),
        EncodeArg.tensor(gaussian_ids_sorted),
        EncodeArg.tensor(tile_bins),
        EncodeArg.tensor(xys),
        EncodeArg.tensor(conics),
        EncodeArg.tensor(colors),
        EncodeArg.tensor(opacities),
        EncodeArg.tensor(final_Ts),
        EncodeArg.tensor(final_idx),
        EncodeArg.tensor(out_img),
        EncodeArg.tensor(background),
        EncodeArg.array(&block_size_dim2, numBytes: block_size_dim2.count * MemoryLayout<Int32>.size)
    ])

    return (out_img, final_Ts, final_idx)
}

func rasterize_backward_tensor(img_height: Int, img_width: Int, gaussian_ids_sorted: MTLBuffer, tile_bins: MTLBuffer, xys: MTLBuffer, conics: MTLBuffer, colors: MTLBuffer, opacities: MTLBuffer, background: MTLBuffer, final_Ts: MTLBuffer, final_idx: MTLBuffer, v_output: MTLBuffer, v_output_alpha: MTLBuffer, context: MetalContext) -> (MTLBuffer?, MTLBuffer?, MTLBuffer?, MTLBuffer?) {
    guard let computePipeline = context.rasterize_backward_kernel_cpso else {
        print("Compute pipeline for rasterize_backward_kernel is not initialized.")
        return (nil, nil, nil, nil)
    }

    let device = context.device

    // Allocate output buffers
    let num_points = xys.length / (2 * MemoryLayout<Float>.stride) // Assuming xys stores (x, y) for each point
    let channels = colors.length / (num_points * MemoryLayout<Float>.stride)

    let v_xy = device.makeBuffer(length: num_points * 2 * MemoryLayout<Float>.stride, options: .storageModeShared)
    let v_conic = device.makeBuffer(length: num_points * 3 * MemoryLayout<Float>.stride, options: .storageModeShared)
    let v_colors = device.makeBuffer(length: num_points * channels * MemoryLayout<Float>.stride, options: .storageModeShared)
    let v_opacity = device.makeBuffer(length: num_points * MemoryLayout<Float>.stride, options: .storageModeShared)

    guard let v_xy = v_xy, let v_conic = v_conic, let v_colors = v_colors, let v_opacity = v_opacity else {
        print("Failed to allocate output buffers.")
        return (nil, nil, nil, nil)
    }

    var img_size: [UInt32] = [UInt32(img_width), UInt32(img_height)]
    var tile_bounds_arr: [UInt32] = [
        (UInt32(img_width) + 15) / 16, // Assuming BLOCK_X = 16
        (UInt32(img_height) + 15) / 16, // Assuming BLOCK_Y = 16
        1,
        0xDEAD
    ]

    let gridSize = MTLSize(width: img_width, height: img_height, depth: 1)
    let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1) // Assuming BLOCK_X = 16 and BLOCK_Y = 16

    // Dispatch the kernel
    dispatchPipeline(context: context, cpso: computePipeline, gridSize: gridSize, threadGroupSize: threadGroupSize, args: [
        EncodeArg.array(&tile_bounds_arr, numBytes: tile_bounds_arr.count * MemoryLayout<UInt32>.size),
        EncodeArg.array(&img_size, numBytes: img_size.count * MemoryLayout<UInt32>.size),
        EncodeArg.tensor(gaussian_ids_sorted),
        EncodeArg.tensor(tile_bins),
        EncodeArg.tensor(xys),
        EncodeArg.tensor(conics),
        EncodeArg.tensor(colors),
        EncodeArg.tensor(opacities),
        EncodeArg.tensor(background),
        EncodeArg.tensor(final_Ts),
        EncodeArg.tensor(final_idx),
        EncodeArg.tensor(v_output),
        EncodeArg.tensor(v_output_alpha),
        EncodeArg.tensor(v_xy),
        EncodeArg.tensor(v_conic),
        EncodeArg.tensor(v_colors),
        EncodeArg.tensor(v_opacity)
    ])

    return (v_xy, v_conic, v_colors, v_opacity)
}

func nd_rasterize_backward_tensor(img_height: Int, img_width: Int, gaussian_ids_sorted: MTLBuffer, tile_bins: MTLBuffer, xys: MTLBuffer, conics: MTLBuffer, colors: MTLBuffer, opacities: MTLBuffer, background: MTLBuffer, final_Ts: MTLBuffer, final_idx: MTLBuffer, v_output: MTLBuffer, v_output_alpha: MTLBuffer, context: MetalContext) -> (MTLBuffer?, MTLBuffer?, MTLBuffer?, MTLBuffer?) {
    guard let computePipeline = context.nd_rasterize_backward_kernel_cpso else {
        print("Compute pipeline for nd_rasterize_backward_tensor is not initialized.")
        return (nil, nil, nil, nil)
    }

    let device = context.device

    let num_points = xys.length / (2 * MemoryLayout<Float>.stride) // Assuming xys stores (x, y) for each point
    let channels = colors.length / (num_points * MemoryLayout<Float>.stride)

    let v_xy = device.makeBuffer(length: num_points * 2 * MemoryLayout<Float>.stride, options: .storageModeShared)
    let v_conic = device.makeBuffer(length: num_points * 3 * MemoryLayout<Float>.stride, options: .storageModeShared)
    let v_colors = device.makeBuffer(length: num_points * channels * MemoryLayout<Float>.stride, options: .storageModeShared)
    let v_opacity = device.makeBuffer(length: num_points * MemoryLayout<Float>.stride, options: .storageModeShared)

    guard let v_xy = v_xy, let v_conic = v_conic, let v_colors = v_colors, let v_opacity = v_opacity else {
        print("Failed to allocate output buffers.")
        return (nil, nil, nil, nil)
    }

    var img_size: [UInt32] = [UInt32(img_width), UInt32(img_height)]
    var tile_bounds_arr: [UInt32] = [
        (UInt32(img_width) + 15) / 16, // Assuming BLOCK_X = 16
        (UInt32(img_height) + 15) / 16, // Assuming BLOCK_Y = 16
        1,
        0xDEAD
    ]

    let gridSize = MTLSize(width: img_width, height: img_height, depth: 1)
    let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1) // Assuming BLOCK_X = 16 and BLOCK_Y = 16

    dispatchPipeline(context: context, cpso: computePipeline, gridSize: gridSize, threadGroupSize: threadGroupSize, args: [
        EncodeArg.array(&tile_bounds_arr, numBytes: tile_bounds_arr.count * MemoryLayout<UInt32>.size),
        EncodeArg.array(&img_size, numBytes: img_size.count * MemoryLayout<UInt32>.size),
        EncodeArg.scalar(Float(channels)),
        EncodeArg.tensor(gaussian_ids_sorted),
        EncodeArg.tensor(tile_bins),
        EncodeArg.tensor(xys),
        EncodeArg.tensor(conics),
        EncodeArg.tensor(colors),
        EncodeArg.tensor(opacities),
        EncodeArg.tensor(background),
        EncodeArg.tensor(final_Ts),
        EncodeArg.tensor(final_idx),
        EncodeArg.tensor(v_output),
        EncodeArg.tensor(v_output_alpha),
        EncodeArg.tensor(v_xy),
        EncodeArg.tensor(v_conic),
        EncodeArg.tensor(v_colors),
        EncodeArg.tensor(v_opacity)
    ])

    return (v_xy, v_conic, v_colors, v_opacity)
}
