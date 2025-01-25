//
//  optim_scheduler.swift
//  OpenSplat
//
//  Created by Yuxuan Zhang on 12/28/24.
//

import Foundation
import simd

class OptimScheduler {
    var opt: AdamOptimizer
    var lrInit: Float
    var lrFinal: Float
    var maxSteps: Int
    
    init(opt: AdamOptimizer, lrFinal: Float, maxSteps: Int) {
        self.opt = opt
        self.lrInit = opt.learningRate
        self.lrFinal = lrFinal
        self.maxSteps = maxSteps
    }
    
    func step(step: Int) {
        let lr = getLearningRate(step: step)
        opt.learningRate = lr
    }
    
    func getLearningRate(step: Int) -> Float {
        let t = max(min(Float(step) / Float(maxSteps), 1.0), 0.0)
        return exp(log(lrInit) * (1.0 - t) + log(lrFinal) * t)
    }
}

//
//  Customized Functions
//

class AdamOptimizer {
    var learningRate: Float
    var beta1: Float
    var beta2: Float
    var epsilon: Float
    var t: Int
    var m: Any
    var v: Any

    init<T>(learningRate: Float = 0.001,
            beta1: Float = 0.9,
            beta2: Float = 0.999,
            epsilon: Float = 1e-8,
            paramSize: Int,
            paramType: T.Type = SIMD3<Float>.self) {
        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

        if paramType == SIMD3<Float>.self {
            self.m = Array(repeating: SIMD3<Float>(0, 0, 0), count: paramSize)
            self.v = Array(repeating: SIMD3<Float>(0, 0, 0), count: paramSize)
        } else if paramType == SIMD4<Float>.self {
            self.m = Array(repeating: SIMD4<Float>(0, 0, 0, 0), count: paramSize)
            self.v = Array(repeating: SIMD4<Float>(0, 0, 0, 0), count: paramSize)
        } else if paramType == [SIMD3<Float>].self {
            self.m = Array(repeating: Array(repeating: SIMD3<Float>(0, 0, 0), count: paramSize), count: paramSize)
            self.v = Array(repeating: Array(repeating: SIMD3<Float>(0, 0, 0), count: paramSize), count: paramSize)
        } else {
            fatalError("Unsupported parameter type")
        }
    }

    func zeroGrad() {
        if let mSIMD3 = m as? [SIMD3<Float>], let vSIMD3 = v as? [SIMD3<Float>] {
            m = Array(repeating: SIMD3<Float>(0, 0, 0), count: mSIMD3.count)
            v = Array(repeating: SIMD3<Float>(0, 0, 0), count: vSIMD3.count)
        } else if let mSIMD4 = m as? [SIMD4<Float>], let vSIMD4 = v as? [SIMD4<Float>] {
            m = Array(repeating: SIMD4<Float>(0, 0, 0, 0), count: mSIMD4.count)
            v = Array(repeating: SIMD4<Float>(0, 0, 0, 0), count: vSIMD4.count)
        } else if let mNestedSIMD3 = m as? [[SIMD3<Float>]], let vNestedSIMD3 = v as? [[SIMD3<Float>]] {
            m = Array(repeating: Array(repeating: SIMD3<Float>(0, 0, 0), count: mNestedSIMD3[0].count), count: mNestedSIMD3.count)
            v = Array(repeating: Array(repeating: SIMD3<Float>(0, 0, 0), count: vNestedSIMD3[0].count), count: vNestedSIMD3.count)
        }
    }

    func step<T>(params: inout T, grads: T) {
        t += 1
        let biasCorrection1 = 1 - pow(beta1, Float(t))
        let biasCorrection2 = 1 - pow(beta2, Float(t))
        let lr = learningRate * sqrt(biasCorrection2) / biasCorrection1

        if var paramsSIMD3 = params as? [SIMD3<Float>], let gradsSIMD3 = grads as? [SIMD3<Float>],
           var mSIMD3 = m as? [SIMD3<Float>], var vSIMD3 = v as? [SIMD3<Float>] {
            for i in 0..<paramsSIMD3.count {
                mSIMD3[i] = beta1 * mSIMD3[i] + (1 - beta1) * gradsSIMD3[i]
                vSIMD3[i] = beta2 * vSIMD3[i] + (1 - beta2) * simd_mul(gradsSIMD3[i], gradsSIMD3[i])

                let mHat = mSIMD3[i] / biasCorrection1
                let vHat = vSIMD3[i] / biasCorrection2
                let update = mHat / (sqrt(vHat) + epsilon)

                paramsSIMD3[i] -= lr * update
            }
            params = paramsSIMD3 as! T
            m = mSIMD3
            v = vSIMD3
        } else if var paramsSIMD4 = params as? [SIMD4<Float>], let gradsSIMD4 = grads as? [SIMD4<Float>],
                  var mSIMD4 = m as? [SIMD4<Float>], var vSIMD4 = v as? [SIMD4<Float>] {
            for i in 0..<paramsSIMD4.count {
                mSIMD4[i] = beta1 * mSIMD4[i] + (1 - beta1) * gradsSIMD4[i]
                vSIMD4[i] = beta2 * vSIMD4[i] + (1 - beta2) * simd_mul(gradsSIMD4[i], gradsSIMD4[i])

                let mHat = mSIMD4[i] / biasCorrection1
                let vHat = vSIMD4[i] / biasCorrection2
                let update = mHat / (sqrt(vHat) + epsilon)

                paramsSIMD4[i] -= lr * update
            }
            params = paramsSIMD4 as! T
            m = mSIMD4
            v = vSIMD4
        } else if var paramsNestedSIMD3 = params as? [[SIMD3<Float>]], let gradsNestedSIMD3 = grads as? [[SIMD3<Float>]],
                  var mNestedSIMD3 = m as? [[SIMD3<Float>]], var vNestedSIMD3 = v as? [[SIMD3<Float>]] {
            for i in 0..<paramsNestedSIMD3.count {
                for j in 0..<paramsNestedSIMD3[i].count {
                    mNestedSIMD3[i][j] = beta1 * mNestedSIMD3[i][j] + (1 - beta1) * gradsNestedSIMD3[i][j]
                    vNestedSIMD3[i][j] = beta2 * vNestedSIMD3[i][j] + (1 - beta2) * simd_mul(gradsNestedSIMD3[i][j], gradsNestedSIMD3[i][j])

                    let mHat = mNestedSIMD3[i][j] / biasCorrection1
                    let vHat = vNestedSIMD3[i][j] / biasCorrection2
                    let update = mHat / (sqrt(vHat) + epsilon)

                    paramsNestedSIMD3[i][j] -= lr * update
                }
            }
            params = paramsNestedSIMD3 as! T
            m = mNestedSIMD3
            v = vNestedSIMD3
        } else {
            fatalError("Unsupported parameter or gradient type")
        }
    }
}
