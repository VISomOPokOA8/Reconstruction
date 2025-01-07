import Accelerate
import Foundation

class SSIM {
    var windowSize: Int
    var channel: Int
    var window: [[[[Float]]]]
    
    init(windowSize: Int, channel: Int) {
        self.windowSize = windowSize
        self.channel = channel
        self.window = createWindow()
    }
    
    func eval() {
        
    }
    
    private func createWindow() -> [[[[Float]]]] {
        let gaussian = gaussian(sigma: 1.5)
        let _1DWindow = gaussian.map { [$0] }
        let _1DWindow_T = transpose(matrix: _1DWindow)
        let _2DWindow = [[matrixMultiply(_1DWindow, _1DWindow_T)]]
        let _2DWindow_E = expand(tensor: _2DWindow, toShape: (channel, 1, windowSize, windowSize))
        return _2DWindow_E
    }
    
    private func gaussian(sigma: Float) -> [Float] {
        var gauss = [Float](repeating: 0.0, count: windowSize)
        for i in 0..<windowSize {
            gauss[i] = exp(-(pow(floorf(Float(i - windowSize) / 2.0), 2)) / (2.0 * sigma * sigma))
        }
        let sum = gauss.reduce(0, +)
        return gauss.map { $0 / sum }
    }
}

//
//  Customized Function
//

func transpose(matrix: [[Float]]) -> [[Float]] {
    let rowCount = matrix.count
    guard rowCount > 0 else { return [] }
    let colCount = matrix[0].count

    var transposed = [[Float]](repeating: [Float](repeating: 0, count: rowCount), count: colCount)
    for i in 0..<rowCount {
        for j in 0..<colCount {
            transposed[j][i] = matrix[i][j]
        }
    }
    return transposed
}

func matrixMultiply(_ a: [[Float]], _ b: [[Float]]) -> [[Float]] {
    let aRows = a.count
    guard aRows > 0 else { return [] }
    let aCols = a[0].count
    let bRows = b.count
    let bCols = b[0].count
    guard aCols == bRows else {
        fatalError("矩阵维度不匹配，无法相乘")
    }

    var result = [[Float]](repeating: [Float](repeating: 0, count: bCols), count: aRows)
    for i in 0..<aRows {
        for j in 0..<bCols {
            for k in 0..<aCols {
                result[i][j] += a[i][k] * b[k][j]
            }
        }
    }
    return result
}

func expand(tensor: [[[[Float]]]], toShape shape: (Int, Int, Int, Int)) -> [[[[Float]]]] {
    let (newChannels, newDepth, newHeight, newWidth) = shape
    let oldChannels = tensor.count
    let oldDepth = tensor[0].count
    let oldHeight = tensor[0][0].count
    let oldWidth = tensor[0][0][0].count

    // 检查输入张量的尺寸是否兼容
    guard oldChannels == 1 || oldChannels == newChannels,
          oldDepth == 1 || oldDepth == newDepth,
          oldHeight == 1 || oldHeight == newHeight,
          oldWidth == 1 || oldWidth == newWidth else {
        fatalError("张量形状不兼容，无法扩展")
    }

    // 初始化新张量
    var expandedTensor = [[[[Float]]]](
        repeating: [[[Float]]](
            repeating: [[Float]](
                repeating: [Float](repeating: 0, count: newWidth),
                count: newHeight
            ),
            count: newDepth
        ),
        count: newChannels
    )

    // 填充扩展张量
    for c in 0..<newChannels {
        for d in 0..<newDepth {
            for h in 0..<newHeight {
                for w in 0..<newWidth {
                    let value = tensor[c % oldChannels][d % oldDepth][h % oldHeight][w % oldWidth]
                    expandedTensor[c][d][h][w] = value
                }
            }
        }
    }

    return expandedTensor
}
