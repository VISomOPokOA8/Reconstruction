//
//  optim_scheduler.swift
//  OpenSplat
//
//  Created by Yuxuan Zhang on 12/28/24.
//

import Foundation

class OptimScheduler {
    var opt: AdamOptimizer
    var lrInit: Float
    var lrFinal: Float
    var maxSteps: Int
    
    init(opt: AdamOptimizer, lrFinal: Float, maxSteps: Int) {
        self.opt = opt
        self.lrInit = opt.learningRate // Dynamically fetch the optimizer's initial learning rate
        self.lrFinal = lrFinal
        self.maxSteps = maxSteps
    }
    
    func step(step: Int) {
        let lr = getLearningRate(step: step)
        opt.learningRate = lr
    }
    
    func getLearningRate(step: Int) -> Float {
        let t: Float = max(min(Float(step) / Float(maxSteps), 1.0), 0.0)
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
    var m: [Float]
    var v: [Float]

    init(learningRate: Float = 0.001, beta1: Float = 0.9, beta2: Float = 0.999, epsilon: Float = 1e-8, paramSize: Int) {
        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = Array(repeating: 0.0, count: paramSize)
        self.v = Array(repeating: 0.0, count: paramSize)
    }

    func step(params: inout [Float], grads: [Float]) {
        t += 1
        let biasCorrection1 = 1 - pow(beta1, Float(t))
        let biasCorrection2 = 1 - pow(beta2, Float(t))
        let lr = learningRate * sqrt(biasCorrection2) / biasCorrection1

        for i in 0..<params.count {
            // 更新一阶矩和二阶矩
            m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
            v[i] = beta2 * v[i] + (1 - beta2) * (grads[i] * grads[i])

            // 计算更新值
            let mHat = m[i] / biasCorrection1
            let vHat = v[i] / biasCorrection2
            let update = mHat / (sqrt(vHat) + epsilon)

            // 更新参数
            params[i] -= lr * update
        }
    }
}
