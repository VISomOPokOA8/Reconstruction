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

func compute_cov2d_bounds_tensor(num_pts: Int, A) {
    
}

func compute_sh_forward_tensor(num_points: Int, degree: Int, degrees_to_use: Int, viewdirs, coeffs) {
    
}

func compute_sh_backward_tensor(num_points: Int, degree: Int, degrees_to_use: Int, viewdirs, v_colors) {
    
}

func project_gaussians_forward_tensor(num_points: Int,
                                      means3d: [SIMD3<Float>],
                                      scales: [SIMD3<Float>],
                                      glob_scale: Float,
                                      quats: [SIMD4<Float>],
                                      viewmat: simd_float4x4,
                                      projmat: simd_float4x4,
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

    let cov3d_d = [[Float]](repeating: [Float](repeating: 0.0, count: 6), count: num_points)
    let xys_d = [SIMD2<Float>](repeating: SIMD2<Float>(0.0, 0.0), count: num_points)
    let depths_d = [Float](repeating: 0.0, count: num_points)
    let radii_d = [Int](repeating: 0, count: num_points)
    let conics_d = [SIMD3<Float>](repeating: SIMD3<Float>(0.0, 0.0, 0.0), count: num_points)
    let num_tiles_hit_d = [Int](repeating: 0, count: num_points)
    
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
                          ctx.device.makeBuffer(bytes: means3d, length: means3d.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: scales, length: scales.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &glob_scale, length: MemoryLayout<Float>.size, options: [])!,
                          ctx.device.makeBuffer(bytes: quats, length: quats.count * MemoryLayout<SIMD4<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: [viewmat], length: MemoryLayout<simd_float4x4>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: [projmat], length: MemoryLayout<simd_float4x4>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &intrins, length: MemoryLayout<SIMD4<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &img_size, length: MemoryLayout<SIMD2<Int>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &tile_bounds_arr, length: MemoryLayout<SIMD4<Int>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: &clip_thresh, length: MemoryLayout<Float>.size, options: [])!,
                          ctx.device.makeBuffer(bytes: cov3d_d, length: cov3d_d.count * MemoryLayout<[Float]>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: xys_d, length: xys_d.count * MemoryLayout<SIMD2<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: depths_d, length: depths_d.count * MemoryLayout<Float>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: radii_d, length: radii_d.count * MemoryLayout<Int>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: conics_d, length: conics_d.count * MemoryLayout<SIMD3<Float>>.stride, options: [])!,
                          ctx.device.makeBuffer(bytes: num_tiles_hit_d, length: num_tiles_hit_d.count * MemoryLayout<Int>.stride, options: [])!
                         ]
    )
    
    return (cov3d_d, xys_d, depths_d, radii_d, conics_d, num_tiles_hit_d)
}

func project_gaussians_backward_tensor(num_points: Int,
                                       means3d: [SIMD3<Float>],
                                       scales: [SIMD3<Float>],
                                       glob_scale: Float,
                                       quats: [SIMD4<Float>],
                                       viewmat: simd_float4x4,
                                       projmat: simd_float4x4,
                                       fx: Float,
                                       fy: Float,
                                       cx: Float,
                                       cy: Float,
                                       img_height: Int,
                                       img_width: Int,
                                       cov3d, radii, conics, v_xy, v_depth, v_conic) {
    
}

func map_gaussian_to_intersects_tensor(num_points: Int,
                                       num_intersects: Int,
                                       xys,
                                       depths,
                                       radii,
                                       num_tile_hit,
                                       tile_bounds: TileBounds) {
    
}

func get_tile_bin_edges_tensor(num_intersects: Int,
                               isect_ids_sorted) {
    
}

func rasterize_forward_tensor(tile_bounds: TileBounds,
                              block: (Int, Int, Int),
                              img_size: (Int, Int, Int),
                              gaussian_ids_sorted,
                              tile_bins,
                              xys,
                              conics,
                              colors: [[SIMD3<Float>]],
                              opacities: [[Float]],
                              background: SIMD3<Float>) {
    
}

func rasterize_backward_tensor(img_height: Int,
                               img_width: Int,
                               gaussian_ids_sorted,
                               tile_bins,
                               xys,
                               conics,
                               colors: [[SIMD3<Float>]],
                               opacities: [[Float]],
                               background: SIMD3<Float>,
                               final_Ts,
                               final_idx,
                               v_output,
                               v_output_alpha) {
    
}

func nd_rasterize_forward_tensor(tile_bounds: TileBounds,
                                 block: (Int, Int, Int),
                                 img_size: (Int, Int, Int),
                                 gaussian_ids_sorted,
                                 tile_bins,
                                 xys,
                                 conics,
                                 colors: [[SIMD3<Float>]],
                                 opacities: [[Float]],
                                 background: SIMD3<Float>) {
    
}

func nd_rasterize_backward_tensor(img_height: Int,
                                  img_width: Int,
                                  gaussian_ids_sorted,
                                  tile_bins,
                                  xys,
                                  conics,
                                  colors: [[SIMD3<Float>]],
                                  opacities: [[Float]],
                                  background: SIMD3<Float>,
                                  final_Ts,
                                  final_idx,
                                  v_output,
                                  v_output_alpha) {
    
}
