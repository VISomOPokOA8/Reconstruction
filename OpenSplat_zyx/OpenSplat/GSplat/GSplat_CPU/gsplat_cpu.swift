//
//  gsplat_cpu.swift
//  OpenSplat
//
//  Created by Yuxuan Zhang on 25/12/24.
//

import Foundation

func quatToRot(quat: [Float]) -> [[Float]] {
    // Normalize quaternion
    let length = sqrt(quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3])
    let w = quat[0] / length
    let x = quat[1] / length
    let y = quat[2] / length
    let z = quat[3] / length

    // Create rotation matrix
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)]
    ]
}

func project_gaussians_forward_tensor_cpu(num_points: Int, means3d: [[Float]], scales: [[Float]], glob_scale: Float, quats: [[Float]], viewmat: [[Float]], projmat: [[Float]], fx: Float, fy: Float, cx: Float, cy: Float, img_height: Int, img_width: Int, clip_thresh: Float) -> ([[Float]], [Int], [[Float]], [[[Float]]], [Float]) {
    let fovx = 0.5 * Float(img_width) / fx
    let fovy = 0.5 * Float(img_height) / fy
    
    var Rclip = [[Float]](repeating: [0.0, 0.0, 0.0], count: 3)
    for i in 0..<3 {
        Rclip[i] = Array(viewmat[i][0..<3])
    }

    var Tclip = [Float](repeating: 0.0, count: 3)
    for i in 0..<3 {
        Tclip[i] = viewmat[i][3]
    }

    var pView = [[Float]](repeating: [0.0, 0.0, 0.0], count: num_points)
    for i in 0..<num_points {
        for j in 0..<3 {
            pView[i][j] = Rclip[j][0] * means3d[i][0] + Rclip[j][1] * means3d[i][1] + Rclip[j][2] * means3d[i][2] + Tclip[j]
        }
    }

    var R = [[[Float]]](repeating: [[Float]](repeating: [0.0, 0.0, 0.0], count: 3), count: num_points)
    for i in 0..<num_points {
        R[i] = quatToRot(quat: quats[i])
    }

    var M = [[[Float]]](repeating: [[Float]](repeating: [0.0, 0.0, 0.0], count: 3), count: num_points)
    for i in 0..<num_points {
        for j in 0..<3 {
            for k in 0..<3 {
                M[i][j][k] = R[i][j][k] * glob_scale * scales[i][j]
            }
        }
    }

    var cov3d = [[[Float]]](repeating: [[Float]](repeating: [0.0, 0.0, 0.0], count: 3), count: num_points)
    for i in 0..<num_points {
        for j in 0..<3 {
            for k in 0..<3 {
                cov3d[i][j][k] = 0.0
                for l in 0..<3 {
                    cov3d[i][j][k] += M[i][j][l] * M[i][k][l]
                }
            }
        }
    }
    
    let limX = 1.3 * fovx
    let limY = 1.3 * fovy
    var minLimX = [Float](repeating: 0.0, count: num_points)
    var minLimY = [Float](repeating: 0.0, count: num_points)
    
    var t = [[Float]](repeating: [0.0, 0.0, 0.0], count: num_points)
    for i in 0..<num_points {
        minLimX[i] = pView[i][2] * min(limX, max(-limX, pView[i][0] / pView[i][2]))
        minLimY[i] = pView[i][2] * min(limY, max(-limY, pView[i][1] / pView[i][2]))
        t[i] = [minLimX[i], minLimY[i], pView[i][2]]
    }
    
    var rz = [Float](repeating: 0.0, count: num_points)
    var rz2 = [Float](repeating: 0.0, count: num_points)
    for i in 0..<num_points {
        rz[i] = 1.0 / t[i][2]
        rz2[i] = rz[i] * rz[i]
    }

    var J = [[[Float]]](repeating: [[Float]](repeating: [0.0, 0.0, 0.0], count: 2), count: num_points)
    for i in 0..<num_points {
        J[i][0] = [fx * rz[i], 0.0, -fx * t[i][0] * rz2[i]]
        J[i][1] = [0.0, fy * rz[i], -fy * t[i][1] * rz2[i]]
    }

    var T = [[[Float]]](repeating: [[Float]](repeating: [0.0, 0.0, 0.0], count: 2), count: num_points)
    for i in 0..<num_points {
        for j in 0..<2 {
            for k in 0..<3 {
                T[i][j][k] = 0.0
                for l in 0..<3 {
                    T[i][j][k] += J[i][j][l] * Rclip[l][k]
                }
            }
        }
    }

    var cov2d = [[[Float]]](repeating: [[Float]](repeating: [0.0, 0.0], count: 2), count: num_points)
    for i in 0..<num_points {
        for j in 0..<2 {
            for k in 0..<2 {
                cov2d[i][j][k] = 0.0
                for l in 0..<3 {
                    for m in 0..<3 {
                        cov2d[i][j][k] += T[i][j][l] * cov3d[i][l][m] * T[i][k][m]
                    }
                }
            }
        }
    }
    for i in 0..<num_points {
        cov2d[i][0][0] += 0.3
        cov2d[i][1][1] += 0.3
    }
    
    // Compute determinant and conics
    let eps: Float = 1e-6
    var det = [Float](repeating: 0.0, count: num_points)
    var conics = [[Float]](repeating: [0.0, 0.0, 0.0], count: num_points)
    for i in 0..<num_points {
        det[i] = cov2d[i][0][0] * cov2d[i][1][1] - pow(cov2d[i][0][1], 2)
        det[i] = max(det[i], eps)
        conics[i] = [
            cov2d[i][1][1] / det[i],
            -cov2d[i][0][1] / det[i],
            cov2d[i][0][0] / det[i]
        ]
    }

    // Compute radii
    var b = [Float](repeating: 0.0, count: num_points)
    var sq = [Float](repeating: 0.0, count: num_points)
    var radii = [Int](repeating: 0, count: num_points)
    for i in 0..<num_points {
        b[i] = (cov2d[i][0][0] + cov2d[i][1][1]) / 2.0
        sq[i] = sqrt(max(b[i] * b[i] - det[i], 0.1))
        let v1 = b[i] + sq[i]
        let v2 = b[i] - sq[i]
        radii[i] = Int(ceil(3.0 * sqrt(max(v1, v2))))
    }

    // Project pixels
    var pHom = [[Float]](repeating: [0.0, 0.0, 0.0, 1.0], count: num_points)
    for i in 0..<num_points {
        pHom[i][0] = means3d[i][0]
        pHom[i][1] = means3d[i][1]
        pHom[i][2] = means3d[i][2]
    }

    var pProj = [[Float]](repeating: [0.0, 0.0, 0.0], count: num_points)
    for i in 0..<num_points {
        for j in 0..<3 {
            for k in 0..<4 {
                pProj[i][j] += projmat[j][k] * pHom[i][k]
            }
        }
    }

    var xys = [[Float]](repeating: [0.0, 0.0], count: num_points)
    var camDepths = [Float](repeating: 0.0, count: num_points)
    for i in 0..<num_points {
        let rw = 1.0 / max(pProj[i][2], eps)
        xys[i][0] = 0.5 * ((pProj[i][0] * rw + 1.0) * Float(img_width) - 1.0)
        xys[i][1] = 0.5 * ((pProj[i][1] * rw + 1.0) * Float(img_height) - 1.0)
        camDepths[i] = pProj[i][2]
    }

    return (xys, radii, conics, cov2d, camDepths)
}

func rasterize_forward_tensor_cpu(height: Int, width: Int, xys: [[Float]], conics: [[Float]], colors: [[Float]], opacities: [Float], background: [Float], cov2d: [[[Float]]], camDepths: [Float]) -> ([[[Float]]], [[Float]], [[[Int]]]) {
    let channels = colors[0].count
    let numPoints = xys.count
    let pDepths = camDepths
    var px2gid = [[[Int]]](repeating: [[Int]](repeating: [], count: width), count: height)
    
    var gIndices = Array(0..<numPoints)
    gIndices.sort { pDepths[$0] < pDepths[$1] }
    
    var outImg = [[[Float]]](repeating: [[Float]](repeating: Array(repeating: 0.0, count: channels), count: width), count: height)
    var finalTs = [[Float]](repeating: Array(repeating: 1.0, count: width), count: height)
    var done = [[Bool]](repeating: Array(repeating: false, count: width), count: height)

    var sqCov2dX = [Float](repeating: 0.0, count: numPoints)
    var sqCov2dY = [Float](repeating: 0.0, count: numPoints)
    for i in 0..<numPoints {
        sqCov2dX[i] = 3.0 * sqrt(cov2d[i][0][0])
        sqCov2dY[i] = 3.0 * sqrt(cov2d[i][1][1])
    }

    let pConics = conics
    let pCenters = xys
    let pOpacities = opacities
    let pColors = colors

    let bgX = background[0]
    let bgY = background[1]
    let bgZ = background[2]

    let alphaThresh: Float = 1.0 / 255.0

    for idx in gIndices {
        let gaussianId = idx

        let A = pConics[gaussianId][0]
        let B = pConics[gaussianId][1]
        let C = pConics[gaussianId][2]

        let gX = pCenters[gaussianId][0]
        let gY = pCenters[gaussianId][1]

        let sqx = sqCov2dX[gaussianId]
        let sqy = sqCov2dY[gaussianId]

        let minx = max(0, Int(floor(gY - sqy)) - 2)
        let maxx = min(height, Int(ceil(gY + sqy)) + 2)
        let miny = max(0, Int(floor(gX - sqx)) - 2)
        let maxy = min(width, Int(ceil(gX + sqx)) + 2)

        for i in minx..<maxx {
            for j in miny..<maxy {
                if done[i][j] { continue }

                let xCam = gX - Float(j)
                let yCam = gY - Float(i)
                let sigma = 0.5 * (A * xCam * xCam + C * yCam * yCam) + B * xCam * yCam

                if sigma < 0.0 { continue }
                let alpha = min(0.999, pOpacities[gaussianId] * exp(-sigma))
                if alpha < alphaThresh { continue }

                let T = finalTs[i][j]
                let nextT = T * (1.0 - alpha)
                if nextT <= 1e-4 {
                    done[i][j] = true
                    continue
                }

                let vis = alpha * T

                outImg[i][j][0] += vis * pColors[gaussianId][0]
                outImg[i][j][1] += vis * pColors[gaussianId][1]
                outImg[i][j][2] += vis * pColors[gaussianId][2]

                finalTs[i][j] = nextT
                px2gid[i][j].append(gaussianId)
            }
        }
    }

    // Background
    for i in 0..<height {
        for j in 0..<width {
            let T = finalTs[i][j]
            outImg[i][j][0] += T * bgX
            outImg[i][j][1] += T * bgY
            outImg[i][j][2] += T * bgZ

            px2gid[i][j].reverse()
        }
    }

    return (outImg, finalTs, px2gid)
}

func rasterize_backward_tensor_cpu(height: Int, width: Int, xys: [[Float]], conics: [[Float]], colors: [[Float]], opacities: [Float], background: [Float], cov2d: [[Float]], camDepths: [Float], finalTs: [[Float]], px2gid: [[Int]], v_output: [[[Float]]], v_output_alpha: [[Float]]) -> ([[Float]], [[Float]], [[Float]], [Float]) {
    let num_points = xys.count
    let channels = colors[0].count
    
    var v_xy = [[Float]](repeating: [0.0, 0.0], count: num_points)
    var v_conic = [[Float]](repeating: [0.0, 0.0, 0.0], count: num_points)
    var v_colors = [[Float]](repeating: [Float](repeating: 0.0, count: channels), count: num_points)
    var v_opacity = [Float](repeating: 0.0, count: num_points)

    let bgX = background[0]
    let bgY = background[1]
    let bgZ = background[2]
    
    let alphaThresh: Float = 1.0 / 255.0
    
    for i in 0..<height {
        for j in 0..<width {
            let pixIdx = i * width + j
            let Tfinal = finalTs[i][j]
            var T = Tfinal
            var buffer = [Float](repeating: 0.0, count: channels)

            for gaussianId in px2gid[pixIdx] {
                let A = conics[gaussianId][0]
                let B = conics[gaussianId][1]
                let C = conics[gaussianId][2]

                let gX = xys[gaussianId][0]
                let gY = xys[gaussianId][1]

                let xCam = gX - Float(j)
                let yCam = gY - Float(i)
                let sigma = 0.5 * (A * xCam * xCam + C * yCam * yCam) + B * xCam * yCam

                if sigma < 0.0 { continue }
                let vis = exp(-sigma)
                let alpha = min(0.99, opacities[gaussianId] * vis)
                if alpha < alphaThresh { continue }

                let ra = 1.0 / (1.0 - alpha)
                T *= ra
                let fac = alpha * T

                for c in 0..<channels {
                    v_colors[gaussianId][c] += fac * v_output[i][j][c]
                }

                let v_alpha = ((colors[gaussianId][0] * T - buffer[0] * ra) * v_output[i][j][0]) +
                              ((colors[gaussianId][1] * T - buffer[1] * ra) * v_output[i][j][1]) +
                              ((colors[gaussianId][2] * T - buffer[2] * ra) * v_output[i][j][2]) +
                              (Tfinal * ra * v_output_alpha[i][j]) +
                              (-Tfinal * ra * bgX * v_output[i][j][0]) +
                              (-Tfinal * ra * bgY * v_output[i][j][1]) +
                              (-Tfinal * ra * bgZ * v_output[i][j][2])

                for c in 0..<channels {
                    buffer[c] += colors[gaussianId][c] * fac
                }

                let v_sigma = -opacities[gaussianId] * vis * v_alpha
                v_conic[gaussianId][0] += 0.5 * v_sigma * xCam * xCam
                v_conic[gaussianId][1] += 0.5 * v_sigma * xCam * yCam
                v_conic[gaussianId][2] += 0.5 * v_sigma * yCam * yCam

                v_xy[gaussianId][0] += v_sigma * (A * xCam + B * yCam)
                v_xy[gaussianId][1] += v_sigma * (B * xCam + C * yCam)

                v_opacity[gaussianId] += vis * v_alpha
            }
        }
    }

    return (v_xy, v_conic, v_colors, v_opacity)
}

let SH_C0: Float = 0.28209479177387814
let SH_C1: Float = 0.4886025119029199
let SH_C2: [Float] = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
let SH_C3: [Float] = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435]
let SH_C4: [Float] = [2.5033429417967046, -1.7701307697799304, 0.9461746957575601, -0.6690465435572892, 0.10578554691520431, -0.6690465435572892, 0.47308734787878004, -1.7701307697799304, 0.6258357354491761]

func compute_sh_forward_tensor_cpu(num_points: Int, degree: Int, degrees_to_use: Int, viewdirs: [[Float]], coeffs: [[Float]]) -> [[Float]] {
    let numChannels = 3
    let numBases = num_sh_bases(degree: degrees_to_use)
    var result = [[Float]](repeating: [Float](repeating: 0.0, count: num_sh_bases(degree: degree)), count: viewdirs.count)

    for i in 0..<num_points {
        result[i][0] = SH_C0
        if numBases > 1 {
            let x = viewdirs[i][0]
            let y = viewdirs[i][1]
            let z = viewdirs[i][2]
            result[i][1] = SH_C1 * -y
            result[i][2] = SH_C1 * z
            result[i][3] = SH_C1 * -x

            if numBases > 4 {
                let xx = x * x
                let yy = y * y
                let zz = z * z
                let xy = x * y
                let yz = y * z
                let xz = x * z

                result[i][4] = SH_C2[0] * xy
                result[i][5] = SH_C2[1] * yz
                result[i][6] = SH_C2[2] * (2.0 * zz - xx - yy)
                result[i][7] = SH_C2[3] * xz
                result[i][8] = SH_C2[4] * (xx - yy)

                if numBases > 9 {
                    result[i][9] = SH_C3[0] * y * (3 * xx - yy)
                    result[i][10] = SH_C3[1] * xy * z
                    result[i][11] = SH_C3[2] * y * (4 * zz - xx - yy)
                    result[i][12] = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                    result[i][13] = SH_C3[4] * x * (4 * zz - xx - yy)
                    result[i][14] = SH_C3[5] * z * (xx - yy)
                    result[i][15] = SH_C3[6] * x * (xx - 3 * yy)

                    if numBases > 16 {
                        result[i][16] = SH_C4[0] * xy * (xx - yy)
                        result[i][17] = SH_C4[1] * yz * (3 * xx - yy)
                        result[i][18] = SH_C4[2] * xy * (7 * zz - 1)
                        result[i][19] = SH_C4[3] * yz * (7 * zz - 3)
                        result[i][20] = SH_C4[4] * (zz * (35 * zz - 30) + 3)
                        result[i][21] = SH_C4[5] * xz * (7 * zz - 3)
                        result[i][22] = SH_C4[6] * (xx - yy) * (7 * zz - 1)
                        result[i][23] = SH_C4[7] * xz * (xx - 3 * yy)
                        result[i][24] = SH_C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
                    }
                }
            }
        }
    }

    return result.map { row in
        zip(row, coeffs).map { $0 * $1.reduce(0, +) }
    }
}

