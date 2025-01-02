//
//  optim_scheduler.metal
//  OpenSplat
//
//  Created by Yuxuan Zhang on 12/28/24.
//

#include <metal_stdlib>
using namespace metal;

kernel void adam_update_kernel(const device float *params [[ buffer(0) ]],
                               const device float *grads [[ buffer(1) ]],
                               device float *m [[ buffer(2) ]],
                               device float *v [[ buffer(3) ]],
                               constant float &t [[ buffer(4) ]],
                               constant float &lr [[ buffer(5) ]],
                               constant float &b1 [[ buffer(6) ]],
                               constant float &b2 [[ buffer(7) ]],
                               constant float &eps [[ buffer(8) ]]) {
    uint id = get_global_id(0);
    float grad = grads[id];
    m[id] = b1 * m[id] + (1.0 - b1) * grad;
    v[id] = b2 * v[id] + (1.0 - b2) * grad * grad;
    float mHat = m[id] / (1.0 - pow(b1, t));
    float vHat = v[id] / (1.0 - pow(b2, t));
    params[id] -= lr * mHat / (sqrt(vHat) + eps);
}

