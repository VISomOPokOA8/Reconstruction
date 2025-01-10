//
//  model.swift
//  OpenSplat
//
//  Created by Yuxuan Zhang on 1/4/25.
//

import Foundation
import Metal
import simd

func randomQuatTensor(n: Int) -> [SIMD4<Float>] {
    let u = (0..<n).map { _ in Float.random(in: 0..<1) }
    let v = (0..<n).map { _ in Float.random(in: 0..<1) }
    let w = (0..<n).map { _ in Float.random(in: 0..<1) }
    return zip(zip(u, v), w).map { (uv, w) in
        let (u, v) = uv
        return [
            sqrt(1 - u) * sin(2 * PI * v),
            sqrt(1 - u) * cos(2 * PI * v),
            sqrt(u) * sin(2 * PI * w),
            sqrt(u) * cos(2 * PI * w)
        ]
    }
}

func projectMatrix(zNear: Float, zFar: Float, fovX: Float, fovY: Float) -> simd_float4x4 {
    let t = zNear * tan(0.5 * fovY)
    let b = -t
    let r = zNear * tan(0.5 * fovX)
    let l = -r
    return simd_float4x4(
        [2.0 * zNear / (r - l), 0.0, (r + l) / (r - l), 0.0],
        [0.0, 2.0 * zNear / (t - b), (t + b) / (t - b), 0.0],
        [0.0, 0.0, (zFar + zNear) / (zFar - zNear), -1.0 * zFar * zNear / (zFar - zNear)],
        [0.0, 0.0, 1.0, 0.0]
    )
}

func psnr() {
    
}

func l1() {
    
}

class Model {
    var means: [SIMD3<Float>]
    var scales: [SIMD3<Float>]
    var quats: [SIMD4<Float>]
    var featuresDc: [SIMD3<Float>]
    var featuresRest: [[SIMD3<Float>]]
    var opacities: [Float]
    
    var meansOpt: AdamOptimizer?
    var scalesOpt: AdamOptimizer?
    var quatsOpt: AdamOptimizer?
    var featuresDcOpt: AdamOptimizer?
    var featuresRestOpt: AdamOptimizer?
    var opacitiesOpt: AdamOptimizer?
    
    var meansOptSchedular: OptimScheduler?
    
    var radii: [Int]
    var xys: [SIMD2<Float>]
    var lastHeight: Int
    var lastWidth: Int
    
    var xysGradNorm
    var visCounts
    var max2DSize
    
    var backgroundColor: SIMD3<Float>
    var device: MTLDevice
    var ssim: SSIM
    
    var numCameras: Int
    var numDownscales: Int
    var resolutionSchedule: Int
    var shDegree: Int
    var shDegreeInterval: Int
    var refineEvery: Int
    var warmupLength: Int
    var resetAlphaEvery: Int
    var stopSplitAt: Int
    var densifyGradThresh: Float
    var densifySizeThresh: Float
    var stopScreenSizeAt: Int
    var splitScreenSize: Float
    var maxSteps: Int
    var keepCrs: Bool
    
    var scale: Float
    var translation: SIMD3<Float>
    
    //
    // customized properties
    //
    
    var meansGrad: [SIMD3<Float>]
    var scalesGrad: [SIMD3<Float>]
    var quatsGrad: [SIMD4<Float>]
    var featuresDcGrad: [SIMD3<Float>]
    var featuresRestGrad: [[SIMD3<Float>]]
    var opacitiesGrad: [Float]
    
    var xysGrad: [SIMD2<Float>]
    
    // end
    
    init(inputData: InputData, numCameras: Int, numDownscales: Int, resolutionSchedule: Int, shDegree: Int, shDegreeInterval: Int, refineEvery: Int, warmupLength: Int, resetAlphaEvery: Int, desifyGradThresh: Float, desifySizeThresh: Float, stopScreenSizeAt: Int, splitScreenSize: Float, maxSteps: Int, keepCrs: Bool, device: MTLDevice) {
        self.numCameras = numCameras
        self.numDownscales = numDownscales
        self.resolutionSchedule = resolutionSchedule
        self.shDegree = shDegree
        self.shDegreeInterval = shDegreeInterval
        self.refineEvery = refineEvery
        self.warmupLength = warmupLength
        self.resetAlphaEvery = resetAlphaEvery
        self.stopSplitAt = maxSteps / 2
        self.densifyGradThresh = desifyGradThresh
        self.densifySizeThresh = desifySizeThresh
        self.stopScreenSizeAt = stopScreenSizeAt
        self.splitScreenSize = splitScreenSize
        self.maxSteps = maxSteps
        self.keepCrs = keepCrs
        self.device = device
        self.ssim = SSIM(windowSize: 11, channel: 3)
        
        let numPoints = inputData.points.xyz.count
        self.scale = inputData.scale
        self.translation = inputData.translation
        
        self.means = inputData.points.xyz
        self.meansGrad = Array(repeating: SIMD3<Float>(), count: means.count)
        self.scales = PointsTensor(tensor: inputData.points.xyz).scales().map { SIMD3<Float>(log($0), log($0), log($0)) }
        self.scalesGrad = Array(repeating: SIMD3<Float>(), count: scales.count)
        self.quats = randomQuatTensor(n: numPoints)
        self.quatsGrad = Array(repeating: SIMD4<Float>(), count: quats.count)
        
        let dimSh = num_sh_bases(degree: shDegree)
        var shs = Array(repeating: Array(repeating: SIMD3<Float>(0.0, 0.0, 0.0), count: dimSh), count: numPoints)
        
        let shs_0 = rgb2sh(rgb: inputData.points.rgb.map { SIMD3<Float>($0) / 255.0 } )
        for i in 0..<numPoints {
            shs[i][0] = shs_0[i]
        }
        
        self.featuresDc = shs.map { $0[0] }
        self.featuresDcGrad = Array(repeating: SIMD3<Float>(), count: featuresDc.count)
        self.featuresRest = shs.map { Array($0[1..<dimSh]) }
        self.featuresRestGrad = Array(repeating: Array(repeating: SIMD3<Float>(), count: dimSh), count: featuresRest.count)
        self.opacities = Array(repeating: log(0.1 / (1.0 - 0.1)), count: numPoints)
        self.opacitiesGrad = Array(repeating: Float(), count: opacities.count)
        
        self.backgroundColor = SIMD3<Float>(0.613, 0.0101, 0.3984)
        
        self.meansOpt =
        self.scalesOpt =
        self.quatsOpt =
        self.featuresDcOpt =
        self.featuresRestOpt =
        self.opacitiesOpt =
        
        self.meansOptSchedular =
    }
                                                                                             
    deinit {
        self.meansOpt = nil
        self.scalesOpt = nil
        self.quatsOpt = nil
        self.featuresDcOpt = nil
        self.featuresRestOpt = nil
        self.opacitiesOpt = nil
        
        self.meansOptSchedular = nil
    }
    
    func forward(cam: Camera, step: Int) -> [[SIMD3<Float>]] {
        let scaleFactor = Float(getDownscaleFactor(step: step))
        let fx = cam.fx / scaleFactor
        let fy = cam.fy / scaleFactor
        let cx = cam.cx / scaleFactor
        let cy = cam.cy / scaleFactor
        let height = Int(Float(cam.height) / scaleFactor)
        let width = Int(Float(cam.width) / scaleFactor)
        
        var R = simd_float3x3(
            SIMD3<Float>(cam.camToWorld.columns.0.x, cam.camToWorld.columns.0.y, cam.camToWorld.columns.0.z),
            SIMD3<Float>(cam.camToWorld.columns.1.x, cam.camToWorld.columns.1.y, cam.camToWorld.columns.1.z),
            SIMD3<Float>(cam.camToWorld.columns.2.x, cam.camToWorld.columns.2.y, cam.camToWorld.columns.2.z)
        )
        let T = SIMD3<Float>(cam.camToWorld.columns.3.x, cam.camToWorld.columns.3.y, cam.camToWorld.columns.3.z)
        
        let R_diag = simd_float3x3(
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0]
        )
        R = simd_mul(R, R_diag)
        
        let Rinv = simd_transpose(R)
        let Tinv = simd_mul(-Rinv, T)
        
        self.lastHeight = height
        self.lastWidth = width
        
        let viewMat = simd_float4x4(
            SIMD4<Float>(Rinv.columns.0, 0.0),
            SIMD4<Float>(Rinv.columns.1, 0.0),
            SIMD4<Float>(Rinv.columns.2, 0.0),
            SIMD4<Float>(Tinv, 1.0)
        )
        
        let fovX = 2.0 * atan(Float(width) / (2.0 * fx))
        let fovY = 2.0 * atan(Float(width) / (2.0 * fy))
        
        let projMat = projectMatrix(zNear: 0.001, zFar: 1000.0, fovX: fovX, fovY: fovY)
        let colors = zip(featuresDc.map { [$0] }, featuresRest).map { $0 + $1 }
        
        let tileBounds: TileBounds = ((width + BLOCK_X - 1) / BLOCK_X,
                                      (height + BLOCK_Y - 1) / BLOCK_Y,
                                      1)
        let p = ProjectGaussians().apply(means: means,
                                         scales: scales.map { exp($0) },
                                         globScale: 1.0,
                                         quats: quats.map { $0 / simd_normalize($0) },
                                         viewMat: viewMat,
                                         projMat: simd_mul(projMat, viewMat),
                                         fx: fx, fy: fy,
                                         cx: cx, cy: cy,
                                         imgHeight: height, imgWidth: width,
                                         tileBounds: tileBounds)
        self.xys = p.0
        let depths = p.1
        self.radii = p.2
        let conics = p.3
        let numTilesHit = p.4
        
        if radii.reduce(0, +) == 0 {
            return Array(repeating: Array(repeating: backgroundColor, count: width), count: height)
        }
        self.xysGrad = Array(repeating: SIMD2<Float>(), count: xys.count)
        
        var viewDirs = means.map { $0 - T }
        viewDirs = viewDirs.map { $0 / simd_normalize($0) }
        let degreesToUse = min(step / shDegreeInterval, shDegree)
        var rgbs = SphericalHarmonics().apply(degreesToUse: degreesToUse, viewDirs: viewDirs, coeffs: colors)
        
        rgbs = rgbs.map { SIMD3<Float>(min($0.x + 0.5, 0.0), min($0.y + 0.5, 0.0), min($0.z + 0.5, 0.0)) }
        
        let rgb = 
    }
    
    func optimizersZeroGrad() {
        
    }
    
    func optimizersStep() {
        
    }
    
    func optimizersStep(step: Int) {
        
    }
    
    func getDownscaleFactor(step: Int) -> Int {
        
    }
    
    func afterTrain(step: Int) {
        
    }
    
    func save(filename: String) {
        
    }
    
    func savePly(filename: String) {
        
    }
    
    func saveSplat(filename: String) {
        
    }
    
    func saveDebugPly(filename: String) {
        
    }
    
    func mainLoss() {
        
    }
    
    func addToOptimizer(optimizer: AdamOptimizer, newParam, idcs, nSamples: Int) {
        
    }
    
    func removeFromOptimizer(optimizer: AdamOptimizer, newParam, deletedMask) {
        
    }
}
