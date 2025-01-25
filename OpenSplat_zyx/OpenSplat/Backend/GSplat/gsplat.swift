//
//  gsplat.swift
//  OpenSplat
//
//  Created by Yuxuan Zhang on 1/4/25.
//

import Foundation
import simd
import Metal

struct MetalContext {
    var device: MTLDevice
    var queue: MTLCommandQueue
    var d_queue: DispatchQueue
    
    let nd_rasterize_forward_kernel_cpso: MTLComputePipelineState
    let nd_rasterize_backward_kernel_cpso: MTLComputePipelineState
    let rasterize_backward_kernel_cpso: MTLComputePipelineState
    let project_gaussians_forward_kernel_cpso: MTLComputePipelineState
    let project_gaussians_backward_kernel_cpso: MTLComputePipelineState
    let compute_sh_forward_kernel_cpso: MTLComputePipelineState
    let compute_sh_backward_kernel_cpso: MTLComputePipelineState
    let compute_cov2d_bounds_kernel_cpso: MTLComputePipelineState
    let map_gaussian_to_intersects_kernel_cpso: MTLComputePipelineState
    let get_tile_bin_edges_kernel_cspo: MTLComputePipelineState
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
    guard let device = MTLCreateSystemDefaultDevice() else {
        print("Failed to create Metal device.")
        return nil
    }

    guard let queue = device.makeCommandQueue() else {
        print("Failed to create Metal command queue.")
        return nil
    }

    let dQueue = DispatchQueue(label: "com.opensplat.dispatch")

    var metalLibrary: MTLLibrary?
    if let path = Bundle.main.path(forResource: "default", ofType: "metallib"),
       let libraryURL = URL(string: path) {
        do {
            metalLibrary = try device.makeLibrary(URL: libraryURL)
            print("Loaded metallib from precompiled library.")
        } catch {
            print("Failed to load metallib: \(error)")
            return nil
        }
    } else {
        print("Precompiled metallib not found. Attempting to compile from source.")

        if let sourcePath = Bundle.main.path(forResource: "gsplat_metal", ofType: "metal"),
           let source = try? String(contentsOfFile: sourcePath) {
            do {
                metalLibrary = try device.makeLibrary(source: source, options: nil)
                print("Successfully compiled metal source.")
            } catch {
                print("Failed to compile metal source: \(error)")
                return nil
            }
        } else {
            print("Metal source not found.")
            return nil
        }
    }

    func loadKernel(_ name: String) -> MTLComputePipelineState? {
        guard let function = metalLibrary?.makeFunction(name: name) else {
            print("Failed to load function: \(name)")
            return nil
        }
        do {
            return try device.makeComputePipelineState(function: function)
        } catch {
            print("Failed to create pipeline state for \(name): \(error)")
            return nil
        }
    }

    let ndRasterizeBackwardKernel = loadKernel("nd_rasterize_backward_kernel")!
    let ndRasterizeForwardKernel = loadKernel("nd_rasterize_forward_kernel")!
    let rasterizeBackwardKernel = loadKernel("rasterize_backward_kernel")!
    let projectGaussiansForwardKernel = loadKernel("project_gaussians_forward_kernel")!
    let projectGaussiansBackwardKernel = loadKernel("project_gaussians_backward_kernel")!
    let computeShForwardKernel = loadKernel("compute_sh_forward_kernel")!
    let computeShBackwardKernel = loadKernel("compute_sh_backward_kernel")!
    let computeCov2dBoundsKernel = loadKernel("compute_cov2d_bounds_kernel")!
    let mapGaussianToIntersectsKernel = loadKernel("map_gaussian_to_intersects_kernel")!
    let getTileBinEdgesKernel = loadKernel("get_tile_bin_edges_kernel")!

    return MetalContext(device: device,
                        queue: queue,
                        d_queue: dQueue,
                        nd_rasterize_forward_kernel_cpso: ndRasterizeForwardKernel,
                        nd_rasterize_backward_kernel_cpso: ndRasterizeBackwardKernel,
                        rasterize_backward_kernel_cpso: rasterizeBackwardKernel,
                        project_gaussians_forward_kernel_cpso: projectGaussiansForwardKernel,
                        project_gaussians_backward_kernel_cpso: projectGaussiansBackwardKernel,
                        compute_sh_forward_kernel_cpso: computeShForwardKernel,
                        compute_sh_backward_kernel_cpso: computeShBackwardKernel,
                        compute_cov2d_bounds_kernel_cpso: computeCov2dBoundsKernel,
                        map_gaussian_to_intersects_kernel_cpso: mapGaussianToIntersectsKernel,
                        get_tile_bin_edges_kernel_cspo: getTileBinEdgesKernel)
}

func get_global_context() -> MetalContext {
    struct StaticContext {
        static var shared: MetalContext? = {
            return init_gsplat_metal_context()
        }()
    }
    return StaticContext.shared!
}

func dispatchKernel(context: MetalContext,
                    pipelineState: MTLComputePipelineState,
                    gridSize: MTLSize,
                    threadGroupSize: MTLSize,
                    args: [MTLBuffer]) {
    guard let commandBuffer = context.queue.makeCommandBuffer(),
          let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
        print("Failed to create command buffer or compute encoder.")
        return
    }

    computeEncoder.setComputePipelineState(pipelineState)

    for (index, buffer) in args.enumerated() {
        computeEncoder.setBuffer(buffer, offset: 0, index: index)
    }

    computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
    computeEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
}

func compute_cov2d_bounds_tensor(num_pts: Int,
                                 covs2d: inout [SIMD3<Float>]) -> ([SIMD3<Float>],
                                                                   [Float]) {
    var num_pts = num_pts
    
    var conics = Array(repeating: SIMD3<Float>(0.0, 0.0, 0.0), count: num_pts)
    var radii = Array(repeating: Float(0.0), count: num_pts)
    
    let ctx = get_global_context()
    let grid_size = MTLSizeMake(num_pts, 1, 1)
    let num_threads_per_group = min(ctx.compute_cov2d_bounds_kernel_cpso.maxTotalThreadsPerThreadgroup, num_pts)
    let thread_group_size = MTLSizeMake(num_threads_per_group, 1, 1)
    dispatchKernel(context: ctx,
                   pipelineState: ctx.compute_cov2d_bounds_kernel_cpso,
                   gridSize: grid_size,
                   threadGroupSize: thread_group_size,
                   args: [ctx.device.makeBuffer(bytes: &num_pts, length: MemoryLayout<Int>.size, options: [])!,
                          ctx.device.makeBuffer(bytes: &covs2d, length: covs2d.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &conics, length: conics.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &radii, length: radii.count * MemoryLayout<Float>.stride, options: [])!
                   ]
    )
    
    return (conics, radii)
}

func compute_sh_forward_tensor(num_points: Int,
                               degree: Int,
                               degrees_to_use: Int,
                               viewdirs: inout [SIMD3<Float>],
                               coeffs: inout [[SIMD3<Float>]]) -> [SIMD3<Float>] {
    var num_points = num_points
    var degree = degree
    var degrees_to_use = degrees_to_use
    var coeffs_flat = coeffs.flatMap { $0 }
    
    let num_bases = num_sh_bases(degree: degree)
    if coeffs.count != num_points || coeffs.first?.count != num_bases {
        fatalError("coeffs must have dimensions (N, D, 3)")
    }
    var colors = Array(repeating: SIMD3<Float>(), count: num_points)
    
    let ctx = get_global_context()
    let grid_size = MTLSizeMake(num_points, 1, 1)
    let num_threads_per_group = min(ctx.compute_sh_forward_kernel_cpso.maxTotalThreadsPerThreadgroup, num_points)
    let thread_group_size = MTLSizeMake(num_threads_per_group, 1, 1)
    dispatchKernel(context: ctx,
                   pipelineState: ctx.compute_sh_forward_kernel_cpso,
                   gridSize: grid_size,
                   threadGroupSize: thread_group_size,
                   args: [ctx.device.makeBuffer(bytes: &num_points, length: MemoryLayout<Int>.size, options: [])!,
                          ctx.device.makeBuffer(bytes: &degree, length: MemoryLayout<Int>.size, options: [])!,
                          ctx.device.makeBuffer(bytes: &degrees_to_use, length: MemoryLayout<Int>.size, options: [])!,
                          ctx.device.makeBuffer(bytes: &viewdirs, length: viewdirs.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &coeffs_flat, length: coeffs_flat.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &colors, length: colors.count * MemoryLayout<Float>.stride, options: [])!
                         ]
    )
    
    coeffs = reshapeTo2DArray(flatArray: coeffs_flat, numColumns: num_bases)
    return colors
}

func compute_sh_backward_tensor(num_points: Int,
                                degree: Int,
                                degrees_to_use: Int,
                                viewdirs: inout [SIMD3<Float>],
                                v_color: inout [SIMD3<Float>]) -> [[SIMD3<Float>]] {
    var num_points = num_points
    var degree = degree
    var degrees_to_use = degrees_to_use
    
    if viewdirs.count != num_points {
        fatalError("viewdirs must have dimensions (N, 3)")
    }
    if v_color.count != num_points {
        fatalError("v_color must have dimensions (N, 3)")
    }
    let num_bases = num_sh_bases(degree: degree)
    var v_coeffs = Array(repeating: Array(repeating: SIMD3<Float>(0.0, 0.0, 0.0), count: num_bases), count: num_points)
    let v_coeffs_flat = v_coeffs.flatMap { $0 }
    
    let ctx = get_global_context()
    let grid_size = MTLSizeMake(num_points, 1, 1)
    let num_threads_per_group = min(ctx.compute_sh_backward_kernel_cpso.maxTotalThreadsPerThreadgroup, num_points)
    let thread_group_size = MTLSizeMake(num_threads_per_group, 1, 1)
    dispatchKernel(context: ctx,
                   pipelineState: ctx.compute_sh_backward_kernel_cpso,
                   gridSize: grid_size,
                   threadGroupSize: thread_group_size,
                   args: [ctx.device.makeBuffer(bytes: &num_points, length: MemoryLayout<Int>.size, options: [])!,
                          ctx.device.makeBuffer(bytes: &degree, length: MemoryLayout<Int>.size, options: [])!,
                          ctx.device.makeBuffer(bytes: &degrees_to_use, length: MemoryLayout<Int>.size, options: [])!,
                          ctx.device.makeBuffer(bytes: viewdirs, length: MemoryLayout<simd_float4x4>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: v_color, length: v_color.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: v_coeffs_flat, length: v_coeffs_flat.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!
                   ]
    )
    
    v_coeffs = reshapeTo2DArray(flatArray: v_coeffs_flat, numColumns: num_bases)
    return v_coeffs
}

func project_gaussians_forward_tensor(num_points: Int,
                                      means3d: inout [SIMD3<Float>],
                                      scales: inout [SIMD3<Float>],
                                      glob_scale: Float,
                                      quats: inout [SIMD4<Float>],
                                      viewmat: inout simd_float4x4,
                                      projmat: inout simd_float4x4,
                                      fx: Float,
                                      fy: Float,
                                      cx: Float,
                                      cy: Float,
                                      img_height: Int,
                                      img_width: Int,
                                      tile_bounds: TileBounds,
                                      clip_thresh: Float) -> ([[Float]],
                                                              [SIMD2<Float>],
                                                              [Float],
                                                              [Int],
                                                              [SIMD3<Float>],
                                                              [Int]) {
    var num_points = num_points
    var glob_scale = glob_scale
    var clip_thresh = clip_thresh

    var cov3d_d = Array(repeating: Array(repeating: Float(0.0), count: 6), count: num_points)
    var cov3d_d_flat = cov3d_d.flatMap { $0 }
    var xys_d = Array(repeating: SIMD2<Float>(0.0, 0.0), count: num_points)
    var depths_d = Array(repeating: Float(0.0), count: num_points)
    var radii_d = Array(repeating: 0, count: num_points)
    var conics_d = Array(repeating: SIMD3<Float>(0.0, 0.0, 0.0), count: num_points)
    var num_tiles_hit_d = Array(repeating: 0, count: num_points)
    
    var intrins = SIMD4<Float>(fx, fy, cx, cy)
    var img_size = SIMD2<Int>(img_width, img_height)
    var tile_bounds_arr = SIMD4<Int>(tile_bounds.0, tile_bounds.1, tile_bounds.2, 0xDEAD)
    
    let ctx = get_global_context()
    let grid_size = MTLSizeMake(num_points, 1, 1)
    let num_threads_per_group = min(ctx.project_gaussians_forward_kernel_cpso.maxTotalThreadsPerThreadgroup, num_points)
    let thread_group_size = MTLSizeMake(num_threads_per_group, 1, 1)
    dispatchKernel(context: ctx,
                   pipelineState: ctx.project_gaussians_forward_kernel_cpso,
                   gridSize: grid_size,
                   threadGroupSize: thread_group_size,
                   args: [ctx.device.makeBuffer(bytes: &num_points, length: MemoryLayout<Int>.size, options: [])!,
                          ctx.device.makeBuffer(bytes: &means3d, length: means3d.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &scales, length: scales.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &glob_scale, length: MemoryLayout<Float>.size, options: [])!,
                          ctx.device.makeBuffer(bytes: &quats, length: quats.count * MemoryLayout<SIMD4<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &viewmat, length: MemoryLayout<simd_float4x4>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &projmat, length: MemoryLayout<simd_float4x4>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &intrins, length: MemoryLayout<SIMD4<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &img_size, length: MemoryLayout<SIMD2<Int>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &tile_bounds_arr, length: MemoryLayout<SIMD4<Int>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &clip_thresh, length: MemoryLayout<Float>.size, options: [])!,
                          ctx.device.makeBuffer(bytes: &cov3d_d_flat, length: cov3d_d_flat.count * MemoryLayout<Float>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &xys_d, length: xys_d.count * MemoryLayout<SIMD2<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &depths_d, length: depths_d.count * MemoryLayout<Float>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &radii_d, length: radii_d.count * MemoryLayout<Int>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &conics_d, length: conics_d.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &num_tiles_hit_d, length: num_tiles_hit_d.count * MemoryLayout<Int>.stride, options: [])!
                         ]
    )
    
    cov3d_d = reshapeTo2DArray(flatArray: cov3d_d_flat, numColumns: 6)
    return (cov3d_d, xys_d, depths_d, radii_d, conics_d, num_tiles_hit_d)
}

func project_gaussians_backward_tensor(num_points: Int,
                                       means3d: inout [SIMD3<Float>],
                                       scales: inout [SIMD3<Float>],
                                       glob_scale: Float,
                                       quats: inout [SIMD4<Float>],
                                       viewmat: inout simd_float4x4,
                                       projmat: inout simd_float4x4,
                                       fx: Float,
                                       fy: Float,
                                       cx: Float,
                                       cy: Float,
                                       img_height: Int,
                                       img_width: Int,
                                       cov3d: inout [[Float]],
                                       radii: inout [Int],
                                       conics: inout [SIMD3<Float>],
                                       v_xy: inout [SIMD2<Float>],
                                       v_depth: inout [Float],
                                       v_conic: inout [SIMD3<Float>]) -> ([SIMD3<Float>],
                                                                          [[Float]],
                                                                          [SIMD3<Float>],
                                                                          [SIMD3<Float>],
                                                                          [SIMD4<Float>]) {
    var num_points = num_points
    var glob_scale = glob_scale
    
    var v_cov2d = Array(repeating: SIMD3<Float>(0.0, 0.0, 0.0), count: num_points)
    var v_cov3d = Array(repeating: Array(repeating: Float(0.0), count: 6), count: num_points)
    var v_cov3d_flat = v_cov3d.flatMap { $0 }
    var v_mean3d = Array(repeating: SIMD3<Float>(0.0, 0.0, 0.0), count: num_points)
    var v_scale = Array(repeating: SIMD3<Float>(0.0, 0.0, 0.0), count: num_points)
    var v_quat = Array(repeating: SIMD4<Float>(0.0, 0.0, 0.0, 0.0), count: num_points)
    
    var intrins = SIMD4<Float>(fx, fy, cx, cy)
    var img_size = SIMD2<Int>(img_width, img_height)
    
    var cov3d_flat = cov3d.flatMap { $0 }
    
    let ctx = get_global_context()
    let grid_size = MTLSizeMake(num_points, 1, 1)
    let num_threads_per_group = min(ctx.project_gaussians_backward_kernel_cpso.maxTotalThreadsPerThreadgroup, num_points)
    let thread_group_size = MTLSizeMake(num_threads_per_group, 1, 1)
    dispatchKernel(context: ctx,
                   pipelineState: ctx.project_gaussians_backward_kernel_cpso,
                   gridSize: grid_size,
                   threadGroupSize: thread_group_size,
                   args: [ctx.device.makeBuffer(bytes: &num_points, length: MemoryLayout<Int>.size, options: [])!,
                          ctx.device.makeBuffer(bytes: &means3d, length: means3d.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &scales, length: scales.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &glob_scale, length: MemoryLayout<Float>.size, options: [])!,
                          ctx.device.makeBuffer(bytes: &quats, length: quats.count * MemoryLayout<SIMD4<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &viewmat, length: MemoryLayout<simd_float4x4>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &projmat, length: MemoryLayout<simd_float4x4>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &intrins, length: MemoryLayout<SIMD4<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &img_size, length: MemoryLayout<SIMD2<Int>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &cov3d_flat, length: cov3d_flat.count * MemoryLayout<Float>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &radii, length: radii.count * MemoryLayout<Int>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &conics, length: conics.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &v_xy, length: v_xy.count * MemoryLayout<SIMD2<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &v_depth, length: v_depth.count * MemoryLayout<Float>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &v_conic, length: v_conic.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &v_cov2d, length: v_cov2d.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &v_cov3d_flat, length: v_cov3d_flat.count * MemoryLayout<Float>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &v_mean3d, length: v_mean3d.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &v_scale, length: v_scale.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &v_quat, length: v_quat.count * MemoryLayout<SIMD4<Float>>.stride, options: [])!
                         ]
    )
    
    v_cov3d = reshapeTo2DArray(flatArray: v_cov3d_flat, numColumns: 6)
    cov3d = reshapeTo2DArray(flatArray: cov3d_flat, numColumns: cov3d.first!.count)
    return (v_cov2d, v_cov3d, v_mean3d, v_scale, v_quat)
}

func map_gaussian_to_intersects_tensor(num_points: Int,
                                       num_intersects: Int,
                                       xys: inout [SIMD2<Float>],
                                       depths: inout [Float],
                                       radii: inout [Int],
                                       num_tile_hit: inout [Int],
                                       tile_bounds: TileBounds) -> ([Int64], [Int]) {
    var num_points = num_points
    
    var gaussian_ids_unsorted = Array(repeating: 0, count: num_intersects)
    var isect_ids_unsorted = Array(repeating: Int64(0), count: num_intersects)
    
    var tile_bounds_arr = SIMD4<Int>(tile_bounds.0, tile_bounds.1, tile_bounds.2, 0xDEAD)
    
    let ctx = get_global_context()
    let grid_size = MTLSizeMake(num_points, 1, 1)
    let num_threads_per_group = min(ctx.map_gaussian_to_intersects_kernel_cpso.maxTotalThreadsPerThreadgroup, num_points)
    let thread_group_size = MTLSizeMake(num_threads_per_group, 1, 1)
    dispatchKernel(context: ctx,
                   pipelineState: ctx.map_gaussian_to_intersects_kernel_cpso,
                   gridSize: grid_size,
                   threadGroupSize: thread_group_size,
                   args: [ctx.device.makeBuffer(bytes: &num_points, length: MemoryLayout<Int>.size, options: [])!,
                          ctx.device.makeBuffer(bytes: &xys, length: xys.count * MemoryLayout<SIMD2<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &depths, length: depths.count * MemoryLayout<Float>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &radii, length: radii.count * MemoryLayout<Int>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &num_tile_hit, length: num_tile_hit.count * MemoryLayout<Int>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &tile_bounds_arr, length: MemoryLayout<SIMD4<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &isect_ids_unsorted, length: isect_ids_unsorted.count * MemoryLayout<Int>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &gaussian_ids_unsorted, length: gaussian_ids_unsorted.count * MemoryLayout<Int64>.stride, options: [])!
                   ]
    )
    
    return (isect_ids_unsorted, gaussian_ids_unsorted)
}

func get_tile_bin_edges_tensor(num_intersects: Int,
                               isect_ids_sorted: inout [Int64]) -> [SIMD2<Int>] {
    var num_intersects = num_intersects
    let tile_bins = Array(repeating: SIMD2<Int>(0, 0), count: num_intersects)
    
    let ctx = get_global_context()
    let grid_size = MTLSizeMake(num_intersects, 1, 1)
    let num_threads_per_group = min(ctx.get_tile_bin_edges_kernel_cspo.maxTotalThreadsPerThreadgroup, num_intersects)
    let thread_group_size = MTLSizeMake(num_threads_per_group, 1, 1)
    dispatchKernel(context: ctx,
                   pipelineState: ctx.get_tile_bin_edges_kernel_cspo,
                   gridSize: grid_size,
                   threadGroupSize: thread_group_size,
                   args: [ctx.device.makeBuffer(bytes: &num_intersects, length: MemoryLayout<Int>.size, options: [])!,
                          ctx.device.makeBuffer(bytes: isect_ids_sorted, length: isect_ids_sorted.count * MemoryLayout<Int64>.size, options: [])!,
                          ctx.device.makeBuffer(bytes: tile_bins, length: tile_bins.count * MemoryLayout<SIMD2<Int>>.size, options: [])!
                         ]
    )
    
    return tile_bins
}

func rasterize_forward_tensor(tile_bounds: TileBounds,
                              block: SIMD3<Int>,
                              img_size: SIMD3<Int>,
                              gaussian_ids_sorted: inout [Int],
                              tile_bins: inout [SIMD2<Int>],
                              xys: inout [SIMD2<Float>],
                              conics: inout [SIMD3<Float>],
                              colors: inout [SIMD3<Float>],
                              opacities: inout [Float],
                              background: inout SIMD3<Float>) -> ([[SIMD3<Float>]],
                                                                  [[Float]],
                                                                  [[Int]]) {
    var channels = 3
    let img_width = img_size.x
    let img_height = img_size.y
    
    var out_img = Array(repeating: Array(repeating: SIMD3<Float>(0.0, 0.0, 0.0), count: img_width), count: img_height)
    var out_img_flat = out_img.flatMap { $0 }
    var final_Ts = Array(repeating: Array(repeating: Float(0.0), count: img_width), count: img_height)
    var final_Ts_flat = final_Ts.flatMap { $0 }
    var final_idx = Array(repeating: Array(repeating: 0, count: img_width), count: img_height)
    var final_idx_flat = final_idx.flatMap { $0 }
    
    var img_size_dim3 = SIMD4<Int>(img_size.x, img_size.y, img_size.z, 0xDEAD)
    var tile_bounds_arr = SIMD4<Int>(tile_bounds.0, tile_bounds.1, tile_bounds.2, 0xDEAD)
    var block_size_dim2 = SIMD2<Int>(block.x, block.y)
    
    let ctx = get_global_context()
    let grid_size = MTLSizeMake(img_width, img_height, 1)
    let thread_group_size = MTLSizeMake(block_size_dim2.x, block_size_dim2.y, 1)
    dispatchKernel(context: ctx,
                   pipelineState: ctx.nd_rasterize_forward_kernel_cpso,
                   gridSize: grid_size,
                   threadGroupSize: thread_group_size,
                   args: [ctx.device.makeBuffer(bytes: &tile_bounds_arr, length: MemoryLayout<SIMD4<Int>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &img_size_dim3, length: MemoryLayout<SIMD4<Int>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &channels, length: MemoryLayout<Int>.size, options: [])!,
                          ctx.device.makeBuffer(bytes: &gaussian_ids_sorted, length: gaussian_ids_sorted.count * MemoryLayout<Int>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &tile_bins, length: tile_bins.count * MemoryLayout<SIMD2<Int>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &xys, length: xys.count * MemoryLayout<SIMD2<Int>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &conics, length: conics.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &colors, length: colors.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &opacities, length: opacities.count * MemoryLayout<Float>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &final_Ts_flat, length: final_Ts_flat.count * MemoryLayout<Float>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &final_idx_flat, length: final_idx_flat.count * MemoryLayout<Int>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &out_img_flat, length: out_img_flat.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &background, length: MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &block_size_dim2, length: MemoryLayout<SIMD2<Int>>.stride, options: [])!
                         ]
    )
    
    out_img = reshapeTo2DArray(flatArray: out_img_flat, numColumns: img_width)
    final_Ts = reshapeTo2DArray(flatArray: final_Ts_flat, numColumns: img_width)
    final_idx = reshapeTo2DArray(flatArray: final_idx_flat, numColumns: img_width)
    return (out_img, final_Ts, final_idx)
}

func rasterize_backward_tensor(img_height: Int,
                               img_width: Int,
                               gaussian_ids_sorted: inout [Int],
                               tile_bins: inout [SIMD2<Int>],
                               xys: inout [SIMD2<Float>],
                               conics: inout [SIMD3<Float>],
                               colors: inout [SIMD3<Float>],
                               opacities: inout [Float],
                               background: inout SIMD3<Float>,
                               final_Ts: inout [[Float]],
                               final_idx: inout [[Int]],
                               v_output: inout [[SIMD3<Float>]],
                               v_output_alpha: inout [[Float]]) -> ([SIMD2<Float>],
                                                                    [SIMD3<Float>],
                                                                    [SIMD3<Float>],
                                                                    [Float]) {
    let num_points = xys.count
    let channels = 3
    var final_Ts_flat = final_Ts.flatMap { $0 }
    var final_idx_flat = final_idx.flatMap { $0 }
    var v_output_flat = v_output.flatMap { $0 }
    var v_output_alpha_flat = v_output_alpha.flatMap { $0 }
    
    var v_xy = Array(repeating: SIMD2<Float>(0.0, 0.0), count: num_points)
    var v_conics = Array(repeating: SIMD3<Float>(0.0, 0.0, 0.0), count: num_points)
    var v_colors = Array(repeating: SIMD3<Float>(0.0, 0.0, 0.0), count: num_points)
    var v_opacity = Array(repeating: Float(0.0), count: num_points)
    
    var img_size = SIMD2<Int>(img_width, img_height)
    var tile_bounds_arr = SIMD4<Int>((img_width + BLOCK_X - 1) / BLOCK_X,
                                     (img_height + BLOCK_Y - 1) / BLOCK_Y,
                                     1,
                                     0xDEAD)
    
    let ctx = get_global_context()
    let grid_size = MTLSizeMake(img_width, img_height, 1)
    let thread_group_size = MTLSizeMake(BLOCK_X, BLOCK_Y, 1)
    dispatchKernel(context: ctx,
                   pipelineState: ctx.rasterize_backward_kernel_cpso,
                   gridSize: grid_size,
                   threadGroupSize: thread_group_size,
                   args: [ctx.device.makeBuffer(bytes: &tile_bounds_arr, length: MemoryLayout<SIMD4<Int>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &img_size, length: MemoryLayout<SIMD2<Int>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &gaussian_ids_sorted, length: gaussian_ids_sorted.count * MemoryLayout<Int>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &tile_bins, length: tile_bins.count * MemoryLayout<SIMD2<Int>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &xys, length: xys.count * MemoryLayout<SIMD2<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &conics, length: conics.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &colors, length: colors.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &opacities, length: opacities.count * MemoryLayout<Float>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &background, length: MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &final_Ts_flat, length: final_Ts_flat.count * MemoryLayout<Float>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &final_idx_flat, length: final_idx_flat.count * MemoryLayout<Int>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &v_output_flat, length: v_output_flat.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &v_output_alpha_flat, length: v_output_alpha_flat.count * MemoryLayout<Float>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &v_xy, length: v_xy.count * MemoryLayout<SIMD2<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &v_conics, length: v_conics.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &v_colors, length: v_colors.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &v_opacity, length: v_opacity.count * MemoryLayout<Float>.stride, options: [])!
                   ]
    )
    
    final_Ts = reshapeTo2DArray(flatArray: final_Ts_flat, numColumns: final_Ts.first!.count)
    final_idx = reshapeTo2DArray(flatArray: final_idx_flat, numColumns: final_idx.first!.count)
    v_output = reshapeTo2DArray(flatArray: v_output_flat, numColumns: v_output.first!.count)
    v_output_alpha = reshapeTo2DArray(flatArray: v_output_alpha_flat, numColumns: v_output_alpha.first!.count)
    return (v_xy, v_conics, v_colors, v_opacity)
}

func nd_rasterize_forward_tensor(tile_bounds: TileBounds,
                                 block: (Int, Int, Int),
                                 img_size: (Int, Int, Int),
                                 gaussian_ids_sorted: inout [Int],
                                 tile_bins: inout [SIMD2<Int>],
                                 xys: inout [SIMD2<Float>],
                                 conics: inout [SIMD3<Float>],
                                 colors: inout [SIMD3<Float>],
                                 opacities: inout [Float],
                                 background: inout SIMD3<Float>) -> ([[SIMD3<Float>]],
                                                                     [[Float]],
                                                                     [[Int]]) {
    var channels = 3
    let img_width = img_size.0
    let img_height = img_size.1
    
    var out_img = Array(repeating: Array(repeating: SIMD3<Float>(0.0, 0.0, 0.0), count: img_width), count: img_height)
    var out_img_flat = out_img.flatMap { $0 }
    var final_Ts = Array(repeating: Array(repeating: Float(0.0), count: img_width), count: img_height)
    var final_Ts_flat = final_Ts.flatMap { $0 }
    var final_idx = Array(repeating: Array(repeating: 0, count: img_width), count: img_height)
    var final_idx_flat = final_idx.flatMap { $0 }
    
    var img_size_dim3 = SIMD4<Int>(img_size.0, img_size.1, img_size.2, 0xDEAD)
    var tile_bounds_arr = SIMD4<Int>(tile_bounds.0, tile_bounds.1, tile_bounds.2, 0xDEAD)
    var block_size_dim2 = SIMD2<Int>(block.0, block.1)
    
    let ctx = get_global_context()
    let grid_size = MTLSizeMake(img_width, img_height, 1)
    let thread_group_size = MTLSizeMake(block_size_dim2.x, block_size_dim2.y, 1)
    dispatchKernel(context: ctx,
                   pipelineState: ctx.nd_rasterize_forward_kernel_cpso,
                   gridSize: grid_size,
                   threadGroupSize: thread_group_size,
                   args: [ctx.device.makeBuffer(bytes: &tile_bounds_arr, length: MemoryLayout<SIMD4<Int>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &img_size_dim3, length: MemoryLayout<SIMD4<Int>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &channels, length: MemoryLayout<Int>.size, options: [])!,
                          ctx.device.makeBuffer(bytes: &gaussian_ids_sorted, length: gaussian_ids_sorted.count * MemoryLayout<Int>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &tile_bins, length: tile_bins.count * MemoryLayout<SIMD2<Int>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &xys, length: xys.count * MemoryLayout<SIMD2<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &conics, length: conics.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &colors, length: colors.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &opacities, length: opacities.count * MemoryLayout<Float>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &final_Ts_flat, length: final_Ts_flat.count * MemoryLayout<Float>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &final_idx_flat, length: final_idx_flat.count * MemoryLayout<Int>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &out_img_flat, length: out_img_flat.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &background, length: MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &block_size_dim2, length: MemoryLayout<SIMD2<Int>>.stride, options: [])!
                         ]
    )
    
    out_img = reshapeTo2DArray(flatArray: out_img_flat, numColumns: img_width)
    final_Ts = reshapeTo2DArray(flatArray: final_Ts_flat, numColumns: img_width)
    final_idx = reshapeTo2DArray(flatArray: final_idx_flat, numColumns: img_width)
    return (out_img, final_Ts, final_idx)
}

func nd_rasterize_backward_tensor(img_height: Int,
                                  img_width: Int,
                                  gaussian_ids_sorted: inout [Int],
                                  tile_bins: inout [SIMD2<Int>],
                                  xys: inout [SIMD2<Float>],
                                  conics: inout [SIMD3<Float>],
                                  colors: inout [SIMD3<Float>],
                                  opacities: inout [Float],
                                  background: inout SIMD3<Float>,
                                  final_Ts: inout [[Float]],
                                  final_idx: inout [[Int]],
                                  v_output: inout [[SIMD3<Float>]],
                                  v_output_alpha: inout [[Float]]) -> ([SIMD2<Float>],
                                                                       [SIMD3<Float>],
                                                                       [SIMD3<Float>],
                                                                       [Float]) {
    let num_points = xys.count
    var channels = 3
    var final_Ts_flat = final_Ts.flatMap { $0 }
    var final_idx_flat = final_idx.flatMap { $0 }
    var v_output_flat = v_output.flatMap { $0 }
    var v_output_alpha_flat = v_output_alpha.flatMap { $0 }
    
    var v_xy = Array(repeating: SIMD2<Float>(0.0, 0.0), count: num_points)
    var v_conic = Array(repeating: SIMD3<Float>(0.0, 0.0, 0.0), count: num_points)
    var v_colors = Array(repeating: SIMD3<Float>(0.0, 0.0, 0.0), count: num_points)
    var v_opacity = Array(repeating: Float(0.0), count: num_points)
    let workspace = Array(repeating: Array(repeating: SIMD3<Float>(0.0, 0.0, 0.0), count: img_width), count: img_height)
    var workspace_flat = workspace.flatMap { $0 }
    
    var img_size = SIMD2<Int>(img_width, img_height)
    var tile_bounds_arr = SIMD4<Int>((img_width + BLOCK_X - 1) / BLOCK_X,
                                     (img_height + BLOCK_Y - 1) / BLOCK_Y,
                                     1,
                                     0xDEAD)
    
    let ctx = get_global_context()
    let grid_size = MTLSizeMake(img_width, img_height, 1)
    let thread_group_size = MTLSizeMake(BLOCK_X, BLOCK_Y, 1)
    dispatchKernel(context: ctx,
                   pipelineState: ctx.nd_rasterize_backward_kernel_cpso,
                   gridSize: grid_size,
                   threadGroupSize: thread_group_size,
                   args: [ctx.device.makeBuffer(bytes: &tile_bounds_arr, length: MemoryLayout<SIMD4<Int>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &img_size, length: MemoryLayout<SIMD2<Int>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &channels, length: MemoryLayout<Int>.size, options: [])!,
                          ctx.device.makeBuffer(bytes: &gaussian_ids_sorted, length: gaussian_ids_sorted.count * MemoryLayout<Int>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &tile_bins, length: tile_bins.count * MemoryLayout<SIMD2<Int>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &xys, length: xys.count * MemoryLayout<SIMD2<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &conics, length: conics.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &colors, length: colors.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &opacities, length: opacities.count * MemoryLayout<Float>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &background, length: MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &final_Ts_flat, length: final_Ts_flat.count * MemoryLayout<Float>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &final_idx_flat, length: final_idx_flat.count * MemoryLayout<Int>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &v_output_flat, length: v_output_flat.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &v_output_alpha_flat, length: v_output_alpha_flat.count * MemoryLayout<Float>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &v_xy, length: v_xy.count * MemoryLayout<SIMD2<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &v_conic, length: v_conic.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &v_colors, length: v_colors.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &v_opacity, length: v_opacity.count * MemoryLayout<Float>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &workspace_flat, length: workspace_flat.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!
                         ]
    )
    
    final_Ts = reshapeTo2DArray(flatArray: final_Ts_flat, numColumns: img_width)
    final_idx = reshapeTo2DArray(flatArray: final_idx_flat, numColumns: img_width)
    v_output = reshapeTo2DArray(flatArray: v_output_flat, numColumns: img_width)
    v_output_alpha = reshapeTo2DArray(flatArray: v_output_alpha_flat, numColumns: img_width)
    return (v_xy, v_conic, v_colors, v_opacity)
}

//
// Customized Functions
//

func reshapeTo2DArray<T>(flatArray: [T], numColumns: Int) -> [[T]] {
    guard flatArray.count % numColumns == 0 else {
        fatalError("Flat array size is not divisible by the number of columns.")
    }

    let numRows = flatArray.count / numColumns
    return stride(from: 0, to: flatArray.count, by: numColumns).map { startIndex in
        Array(flatArray[startIndex..<startIndex + numColumns])
    }
}
