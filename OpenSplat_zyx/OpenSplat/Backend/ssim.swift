import Accelerate
import Foundation

class SSIM {
    var windowSize: Int
    var channel: Int
    var window: [[[Float]]]
    
    init(windowSize: Int, channel: Int) {
        self.windowSize = windowSize
        self.channel = channel
        self.window = createWindow()
    }
    
    func eval(rendered: [[SIMD3<Float>]], gt: [[SIMD3<Float>]]) -> Float {
        let img1 = permute(input: gt)
        let img2 = permute(input: rendered)
        
        let mu1 = conv2d(input: img1, kernel: window, padding: windowSize / 2, groups: channel)
        let mu2 = conv2d(input: img2, kernel: window, padding: windowSize / 2, groups: channel)
        
        let mu1Sq = mu1.map { $0.map { $0.map { pow($0, 2.0) } } }
        let mu2Sq = mu1.map { $0.map { $0.map { pow($0, 2.0) } } }
        let mu1mu2 = zip(mu1, mu2).map { zip($0, $1).map { zip($0, $1).map { $0 * $1 } } }
        
        var sigma1Sq = conv2d(input: img1.map { $0.map { $0.map { $0 * $0 } } }, kernel: window, padding: windowSize / 2, groups: channel)
        sigma1Sq = zip(sigma1Sq, mu1Sq).map { zip($0, $1).map { zip($0, $1).map { $0 - $1 } } }
        var sigma2Sq = conv2d(input: img2.map { $0.map { $0.map { $0 * $0 } } }, kernel: window, padding: windowSize / 2, groups: channel)
        sigma2Sq = zip(sigma2Sq, mu2Sq).map { zip($0, $1).map { zip($0, $1).map { $0 - $1 } } }
        var sigma12 = conv2d(input: zip(img1, img2).map { zip($0, $1).map { zip($0, $1).map { $0 * $1 } } }, kernel: window, padding: windowSize / 2, groups: channel)
        sigma12 = zip(sigma12, mu1mu2).map { zip($0, $1).map { zip($0, $1).map { $0 - $1 } } }
        
        let C1: Float = 0.01 * 0.01
        let C2: Float = 0.03 * 0.03
        
        let numerator = zip(mu1mu2, sigma12).map { zip($0, $1).map { zip($0, $1).map { (2.0 * $0 + C1) * (2.0 * $1 + C2) } } }
        let denominator = zip(zip(mu1Sq, mu2Sq).map { zip($0, $1).map { zip($0, $1).map { $0 + $1 + C1 } } }, zip(sigma1Sq, sigma2Sq).map { zip($0, $1).map { zip($0, $1).map {$0 + $1 + C1 } } }).map { zip($0, $1).map { zip($0, $1).map { $0 * $1 } } }
        let ssimMap = zip(numerator, denominator).map { zip($0, $1).map { zip($0, $1).map { $0 / $1 } } }
        
        return mean(of: ssimMap)
    }
    
    private func createWindow() -> [[[Float]]] {
        let _1DWindow = gaussian(sigma: 1.5).map { [$0] }
        let _2DWindow = mm(matrixA: _1DWindow, matrixB: transpose(matrix: _1DWindow))
        return Array(repeating: _2DWindow, count: channel)
    }
    
    private func gaussian(sigma: Float) -> [Float] {
        var gauss = Array(repeating: Float(0.0), count: windowSize)
        for i in 0..<windowSize {
            gauss[i] = exp(-(pow(floorf(Float(i - windowSize) / 2.0), 2)) / (2.0 * sigma * sigma))
        }
        let sum = gauss.reduce(0, +)
        return gauss.map { $0 / sum }
    }
}

//
//  Customized Functions
//

func transpose(matrix: [[Float]]) -> [[Float]] {
    guard let firstRow = matrix.first else {
        return []
    }
    let row = matrix.count
    let column = firstRow.count

    var transposedMatrix = Array(repeating: Array(repeating: Float(0.0), count: row), count: column)
    for i in 0..<row {
        for j in 0..<column {
            transposedMatrix[j][i] = matrix[i][j]
        }
    }

    return transposedMatrix
}

func mm(matrixA: [[Float]], matrixB: [[Float]]) -> [[Float]] {
    guard let aColumns = matrixA.first?.count,
          matrixA.count > 0,
          matrixB.count > 0,
          matrixB.first?.count ?? 0 > 0,
          aColumns == matrixB.count else {
        fatalError("Invalid matrices dimensions for multiplication")
    }

    let aRows = matrixA.count
    let bColumns = matrixB.first!.count

    var result = Array(repeating: Array(repeating: Float(0.0), count: bColumns), count: aRows)

    for i in 0..<aRows {
        for j in 0..<bColumns {
            for k in 0..<aColumns {
                result[i][j] += matrixA[i][k] * matrixB[k][j]
            }
        }
    }

    return result
}

func permute(input: [[SIMD3<Float>]]) -> [[[Float]]] {
    let rows = input.count
    guard let cols = input.first?.count else {
        return []
    }

    var output = Array(repeating: Array(repeating: Array(repeating: Float(0.0), count: cols), count: rows), count: 3)
    for i in 0..<rows {
        for j in 0..<cols {
            let simdValue = input[i][j]
            output[j][i][0] = simdValue.x
            output[j][i][1] = simdValue.y
            output[j][i][2] = simdValue.z
        }
    }

    return output
}

func conv2d(input: [[[Float]]], kernel: [[[Float]]], padding: Int = 0, groups: Int = 1) -> [[[Float]]] {
    let inputChannels = input.count
    let inputHeight = input[0].count
    let inputWidth = input[0][0].count
    
    let kernelChannels = kernel.count
    let kernelHeight = kernel[0].count
    let kernelWidth = kernel[0][0].count
    
    guard inputChannels % groups == 0, kernelChannels % groups == 0 else {
        fatalError("Input and kernel channels must be divisible by the number of groups.")
    }
    
    let outputChannels = kernelChannels / groups
    let outputHeight = inputHeight + 2 * padding - kernelHeight + 1
    let outputWidth = inputWidth + 2 * padding - kernelWidth + 1
    
    var output = Array(repeating: Array(repeating: Array(repeating: Float(0.0), count: outputWidth), count: outputHeight), count: outputChannels * groups)
    
    var paddedInput = input
    if padding > 0 {
        paddedInput = addPadding(input: input, padding: padding)
    }
    
    for g in 0..<groups {
        for oc in 0..<outputChannels {
            let outputIndex = g * outputChannels + oc
            for ic in 0..<(inputChannels / groups) {
                let inputIndex = g * (inputChannels / groups) + ic
                for i in 0..<outputHeight {
                    for j in 0..<outputWidth {
                        var sum: Float = 0
                        for ki in 0..<kernelHeight {
                            for kj in 0..<kernelWidth {
                                let row = i + ki
                                let col = j + kj
                                sum += paddedInput[inputIndex][row][col] * kernel[outputIndex][ki][kj]
                            }
                        }
                        output[outputIndex][i][j] += sum
                    }
                }
            }
        }
    }
    
    return output
}

func addPadding(input: [[[Float]]], padding: Int) -> [[[Float]]] {
    let inputChannels = input.count
    let inputHeight = input[0].count
    let inputWidth = input[0][0].count
    
    let paddedHeight = inputHeight + 2 * padding
    let paddedWidth = inputWidth + 2 * padding
    
    var paddedInput = [[[Float]]](
        repeating: [[Float]](repeating: [Float](repeating: 0, count: paddedWidth), count: paddedHeight),
        count: inputChannels
    )
    
    for c in 0..<inputChannels {
        for i in 0..<inputHeight {
            for j in 0..<inputWidth {
                paddedInput[c][i + padding][j + padding] = input[c][i][j]
            }
        }
    }
    
    return paddedInput
}

func mean(of tensor: [[[Float]]]) -> Float {
    let flat = tensor.flatMap { $0.flatMap { $0 } }
    var mean: Float = 0
    vDSP_meanv(flat, 1, &mean, vDSP_Length(flat.count))
    return mean
}
