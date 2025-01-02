//
//  input_data.metal
//  OpenSplat
//
//  Created by Yuxuan Zhang on 12/31/24.
//

#include <metal_stdlib>
using namespace metal;

kernel void resizeTexture(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]) {

    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }

    float2 uv = float2(gid.x, gid.y) / float2(outputTexture.get_width(), outputTexture.get_height());

    float2 inputCoord = uv * float2(inputTexture.get_width(), inputTexture.get_height());
    float4 color = inputTexture.read(uint2(inputCoord));

    outputTexture.write(color, gid);
}
