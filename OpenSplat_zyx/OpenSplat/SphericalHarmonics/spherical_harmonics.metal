//
//  spherical_harmonics.metal
//  OpenSplat
//
//  Created by Yuxuan Zhang on 25/12/24.
//

#include <metal_stdlib>
using namespace metal;

kernel void rgb2sh(device float *rgb [[ buffer(0) ]],
                  device float *result [[ buffer(1) ]],
                  constant float &c0 [[ buffer(2) ]],
                  uint id [[ thread_position_in_grid ]]) {
    result[id] = (rgb[id] - 0.5) / c0;
}

kernel void sh2rgb(device float *sh [[ buffer(0) ]],
                   device float *result [[ buffer(1) ]],
                   constant float &c0 [[ buffer(2) ]],
                   uint id [[ thread_position_in_grid ]]) {
    result[id] = clamp((sh[id] * c0) + 0.5, 0.0, 1.0);
}
