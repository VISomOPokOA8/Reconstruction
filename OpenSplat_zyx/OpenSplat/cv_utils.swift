//
//  cv_utils.swift
//  OpenSplat
//
//  Created by Yuxuan Zhang on 26/12/24.
//

import Foundation
import Metal
import MetalKit

func imreadRGB(filename: String, device: MTLDevice) -> [[[Float]]] {
    guard let nsImage = NSImage(contentsOfFile: filename),
          let tiffData = nsImage.tiffRepresentation,
          let bitmapImage = NSBitmapImageRep(data: tiffData),
          let cgImage = bitmapImage.cgImage else {
        print("Failed to load image from \(filename)")
        return nil
    }

    let textureLoader = MTKTextureLoader(device: device)
    do {
        let texture = try textureLoader.newTexture(cgImage: cgImage, options: [MTKTextureLoader.Option.SRGB : false])
        return texture
    } catch {
        print("Failed to create MTLTexture: \(error)")
        return nil
    }
}

func imwriteRGB(filename: String, texture: MTLTexture, device: MTLDevice) {
    guard let ciImage = CIImage(mtlTexture: texture, options: nil) else {
        print("Failed to create CIImage from MTLTexture.")
        return
    }

    let context = CIContext(mtlDevice: device)
    guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
        print("Failed to create CGImage.")
        return
    }

    let destinationURL = URL(fileURLWithPath: filename)
    let bitmapRep = NSBitmapImageRep(cgImage: cgImage)
    let pngData = bitmapRep.representation(using: .png, properties: [:])
    do {
        try pngData?.write(to: destinationURL)
        print("Image saved to \(filename)")
    } catch {
        print("Failed to save image to \(filename): \(error)")
    }
}

func tensorToMat(t: [[Float]], device: MTLDevice) -> MTLTexture? {
    let height = t.count
    let width = t[0].count

    let textureDescriptor = MTLTextureDescriptor()
    textureDescriptor.pixelFormat = .r32Float
    textureDescriptor.width = width
    textureDescriptor.height = height
    textureDescriptor.usage = [.shaderRead, .shaderWrite]

    guard let texture = device.makeTexture(descriptor: textureDescriptor) else {
        print("Failed to create texture.")
        return nil
    }

    var flattenedArray: [Float] = t.flatMap { $0 }
    let bytesPerRow = width * MemoryLayout<Float>.size
    texture.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0, withBytes: &flattenedArray, bytesPerRow: bytesPerRow)

    return texture
}

func matToTensor(m: MTLTexture) -> [[Float]] {
    let width = m.width
    let height = m.height

    var output: [[Float]] = Array(repeating: Array(repeating: 0.0, count: width), count: height)
    let bytesPerRow = width * MemoryLayout<Float>.size
    var flattenedArray = [Float](repeating: 0, count: width * height)

    m.getBytes(&flattenedArray, bytesPerRow: bytesPerRow, from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)

    for row in 0..<height {
        for col in 0..<width {
            output[row][col] = flattenedArray[row * width + col]
        }
    }

    return output
}

func tensorToImage(tensor: [[Float]], device: MTLDevice) -> MTLTexture? {
    let height = tensor.count
    let width = tensor[0].count / 3  // 每行包含 R、G、B 三个值

    let textureDescriptor = MTLTextureDescriptor()
    textureDescriptor.pixelFormat = .rgba8Unorm
    textureDescriptor.width = width
    textureDescriptor.height = height
    textureDescriptor.usage = [.shaderRead, .shaderWrite]

    guard let texture = device.makeTexture(descriptor: textureDescriptor) else {
        print("Failed to create texture.")
        return nil
    }

    // 将浮点值缩放并转换为 uint8 格式
    var flattenedData: [UInt8] = tensor.flatMap { $0.map { UInt8($0 * 255.0) } }
    let bytesPerRow = width * 4 // 每行 4 通道（RGBA）
    texture.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0, withBytes: &flattenedData, bytesPerRow: bytesPerRow)

    return texture
}

func imageToTensor(image: MTLTexture) -> [[Float]] {
    let width = image.width
    let height = image.height

    var output: [[Float]] = Array(repeating: Array(repeating: 0.0, count: width * 4), count: height)
    let bytesPerRow = width * 4
    var flattenedArray = [UInt8](repeating: 0, count: width * height * 4)

    image.getBytes(&flattenedArray, bytesPerRow: bytesPerRow, from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)

    for row in 0..<height {
        for col in 0..<width * 4 {
            output[row][col] = Float(flattenedArray[row * bytesPerRow + col]) / 255.0
        }
    }

    return output
}

