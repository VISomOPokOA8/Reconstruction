//
//  rasterize_gaussians.metal
//  OpenSplat
//
//  Created by Yuxuan Zhang on 25/12/24.
//

#include <metal_stdlib>
using namespace metal;

// Kernel for tensor_sort using Bitonic Sort
kernel void tensor_sort(const device float* input [[ buffer(0) ]],
                        device float* sorted_output [[ buffer(1) ]],
                        device uint* sorted_indices [[ buffer(2) ]],
                        constant uint& length [[ buffer(3) ]],
                        uint id [[ thread_position_in_grid ]]) {
    threadgroup float shared_data[256];
    threadgroup uint shared_indices[256];

    uint local_id = id % 256;
    uint group_id = id / 256;

    if (id < length) {
        shared_data[local_id] = input[id];
        shared_indices[local_id] = id;
    } else {
        shared_data[local_id] = FLT_MAX;
        shared_indices[local_id] = uint(-1);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bitonic Sort
    for (uint size = 2; size <= 256; size *= 2) {
        for (uint stride = size / 2; stride > 0; stride /= 2) {
            uint swap_idx = local_id ^ stride;

            if (swap_idx > local_id) {
                bool ascending = ((local_id & size) == 0);

                if ((ascending && shared_data[local_id] > shared_data[swap_idx]) ||
                    (!ascending && shared_data[local_id] < shared_data[swap_idx])) {
                    float temp = shared_data[local_id];
                    shared_data[local_id] = shared_data[swap_idx];
                    shared_data[swap_idx] = temp;

                    uint temp_idx = shared_indices[local_id];
                    shared_indices[local_id] = shared_indices[swap_idx];
                    shared_indices[swap_idx] = temp_idx;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (id < length) {
        sorted_output[id] = shared_data[local_id];
        sorted_indices[id] = shared_indices[local_id];
    }
}

// Kernel for tensor_gather
kernel void tensor_gather(const device float* input [[ buffer(0) ]],
                          const device uint* indices [[ buffer(1) ]],
                          device float* output [[ buffer(2) ]],
                          constant uint& length [[ buffer(3) ]],
                          uint id [[ thread_position_in_grid ]]) {
    if (id >= length) return;

    // Gather operation
    output[id] = input[indices[id]];
}

// Kernel for tensor_cumsum
kernel void tensor_cumsum(const device float* input [[ buffer(0) ]],
                           device float* output [[ buffer(1) ]],
                           constant uint& length [[ buffer(3) ]],
                           uint id [[ thread_position_in_grid ]]) {
    if (id >= length) return;

    // Simple exclusive prefix sum (inefficient for large arrays, replace with parallel prefix sum)
    float sum = 0.0;
    for (uint i = 0; i <= id; i++) {
        sum += input[i];
    }
    output[id] = sum;
}
