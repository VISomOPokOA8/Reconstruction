//
//  optim_scheduler.swift
//  OpenSplat
//
//  Created by Yuxuan Zhang on 12/28/24.
//

import Foundation
import Metal

class OptimScheduler {
    var adam: AdamOptimizer
    var lrInit: Float
    var lrFinal: Float
    var maxSteps: Int
    
    init(adam: AdamOptimizer, lrFinal: Float, maxSteps: Int) {
        self.adam = adam
        
        self.lrFinal = lrFinal
        self.maxSteps = maxSteps
    }
    
    func step(step: Int) {
        let lr = getLearningRate(step: step)
        adam.learningRate = lr
    }
    
    func getLearningRate(step: Int) -> Float {
        let t: Float = max(min(Float(step) / Float(maxSteps), 1.0), 1.0)
        return exp(log(lrInit) * (1.0 - t) + log(lrFinal) * t)
    }
}

//
//  Customized Functions
//

struct AdamOptimizer {
    var learningRate: Float
    let beta1: Float
    let beta2: Float
    let epsilon: Float
    var m: [MTLBuffer] // 一阶矩
    var v: [MTLBuffer] // 二阶矩

    init(device: MTLDevice, parameterCount: Int, learningRate: Float = 0.001, beta1: Float = 0.9, beta2: Float = 0.999, epsilon: Float = 1e-8) {
        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = []
        self.v = []
        for _ in 0..<parameterCount {
            let mBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)!
            let vBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)!
            self.m.append(mBuffer)
            self.v.append(vBuffer)
        }
    }

    func update(parameters: [MTLBuffer], gradients: [MTLBuffer], timestep: Int, commandQueue: MTLCommandQueue) {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        let device = commandQueue.device

        for i in 0..<parameters.count {
            // Load kernel for Adam optimizer update
            guard let library = device.makeDefaultLibrary(),
                  let function = library.makeFunction(name: "adam_update_kernel"),
                  let pipelineState = try? device.makeComputePipelineState(function: function),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(pipelineState)
            encoder.setBuffer(parameters[i], offset: 0, index: 0)
            encoder.setBuffer(gradients[i], offset: 0, index: 1)
            encoder.setBuffer(m[i], offset: 0, index: 2)
            encoder.setBuffer(v[i], offset: 0, index: 3)

            var step = Float(timestep)
            encoder.setBytes(&step, length: MemoryLayout<Float>.stride, index: 4)

            var lr = learningRate
            var b1 = beta1
            var b2 = beta2
            var eps = epsilon
            encoder.setBytes(&lr, length: MemoryLayout<Float>.stride, index: 5)
            encoder.setBytes(&b1, length: MemoryLayout<Float>.stride, index: 6)
            encoder.setBytes(&b2, length: MemoryLayout<Float>.stride, index: 7)
            encoder.setBytes(&eps, length: MemoryLayout<Float>.stride, index: 8)

            let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
            let gridSize = MTLSize(width: (parameters[i].length + 255) / 256, height: 1, depth: 1)
            encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
}

