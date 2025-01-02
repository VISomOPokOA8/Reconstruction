//
//  point_io.swift
//  OpenSplat
//
//  Created by Yuxuan Zhang on 12/29/24.
//

import Foundation

struct XYZ {
    var x: Float
    var y: Float
    var z: Float
}

let KDTREE_MAX_LEAF = 10

struct PointSet {
    var points: [SIMD3<Float>] = []
    var colors: [SIMD3<UInt8>] = []
    var normals: [SIMD3<Float>] = []
    var views: [UInt8] = []

    var kdTree: UnsafeMutableRawPointer? = nil
    var m_spacing: Double = -1.0

    @inlinable mutating func getIndex() -> UnsafeMutablePointer<MyKDTree>? {
        if let kdTree = kdTree {
            return kdTree.assumingMemoryBound(to: MyKDTree.self)
        } else {
            return buildIndex()
        }
    }

    @inlinable mutating func buildIndex() -> UnsafeMutablePointer<MyKDTree>? {
        if kdTree == nil {
            let instance = UnsafeMutablePointer<MyKDTree>.allocate(capacity: 1)
            instance.initialize(to: MyKDTree(3, self, KDTREE_MAX_LEAF))
            kdTree = UnsafeMutableRawPointer(instance)
        }
        return kdTree?.assumingMemoryBound(to: MyKDTree.self)
    }

    @inlinable func count() -> size_t {
        return points.count
    }

    @inlinable func kdtree_get_point_count() -> size_t {
        return points.count
    }

    @inlinable func kdtree_get_pt(idx: size_t, dim: size_t) -> Float {
        return points[idx][dim]
    }

    func kdtree_get_bbox<BBOX>(_ bb: inout BBOX) -> Bool {
        return false
    }

    mutating func appendPoint(src: PointSet, idx: size_t) {
        points.append(src.points[idx])
        colors.append(src.colors[idx])
    }

    func hasNormals() -> Bool {
        return !normals.isEmpty
    }

    func hasColors() -> Bool {
        return !colors.isEmpty
    }

    func hasViews() -> Bool {
        return !views.isEmpty
    }

    mutating func spacing(kNeighbors: Int = 3) -> Double {
        if m_spacing != -1 {
            return m_spacing
        }

        guard let index = getIndex() else {
            return m_spacing
        }

        let np = count()
        let SAMPLES = min(np, 10000)
        let count = kNeighbors + 1

        var distMap: [size_t: size_t] = [:]
        var indices = [size_t](repeating: 0, count: count)
        var sqrDists = [Float](repeating: 0.0, count: count)

        for _ in 0..<SAMPLES {
            let idx = Int.random(in: 0..<np)
            index.pointee.knnSearch(points[idx], count, &indices, &sqrDists)

            let sum = (1..<kNeighbors).reduce(0.0) { $0 + sqrt(sqrDists[$1]) } / Float(kNeighbors)

            let k = size_t(ceil(sum * 100))
            distMap[k, default: 0] += 1
        }

        var max_val = size_t.min
        var d: size_t = 0
        for (key, value) in distMap {
            if value > max_val {
                d = key
                max_val = value
            }
        }

        m_spacing = max(0.01, Double(d) / 100.0)
        return m_spacing
    }

    mutating func freeIndex() {
        if let kdTree = kdTree {
            kdTree.deallocate()
            self.kdTree = nil
        }
    }

    @inlinable func colorsTensor() -> [SIMD3<UInt8>] {
        return colors
    }

    @inlinable func pointsTensor() -> [SIMD3<Float>] {
        return points
    }
}


func getVertexLine(reader: inout InputStream) throws -> String? {
    repeat {
        guard let line = readLine() else {
            print("Cannot read from input stream. ")
            return nil
        }
        
        if line.hasPrefix("element") {
            return line
        } else if line.hasPrefix("comment") || line.hasPrefix("obj_info") {
            continue
        } else {
            throw NSError(domain: "Invalid PLY file", code: 1)
        }
    } while true
}

func getVertexCount(from line: String) throws -> size_t {
    let tokens = line.split(separator: " ").map { String($0) }
    
    guard tokens.count >= 3 else {
        throw NSError(domain: "Invalid PLY file", code: 1)
    }
    
    guard !(tokens[0] != "element" && tokens[1] != "vertex") else {
        throw NSError(domain: "Invalid PLY file", code: 1)
    }
    
    guard let vertexCount = Int(tokens[2]) else {
        throw NSError(domain: "Invalid vertex count", code: 2)
    }
    
    return vertexCount
}

func checkHeader(reader: inout InputStream, prop: String) throws {
    guard var line = readLine() else {
        print("Cannot read from input stream. ")
        return
    }
    line = line.filter { $0 != "\r" }
    
    guard line.hasSuffix(prop) else {
        throw NSError(domain: "Invalid PLY file", code: 1)
    }
}

func hasHeader(line: String, prop: String) -> Bool {
    return line.hasPrefix("property") && line.hasSuffix(prop)
}

func fastPlyReadPointSet(filename: String) throws -> PointSet? {
    guard var reader = InputStream(fileAtPath: filename) else {
        print("Cannot read from file.")
        return nil
    }
    reader.open()
    
    var r = PointSet(
        points: [],
        colors: [],
        normals: [],
        views: []
    )

    var line: String
    line = readLine()!
    line = line.filter { $0 != "\r" }
    
    if line != "ply" {
        print("Invalid PLY file (header does not start with ply). ")
        return nil
    }
    
    line = readLine()!
    line = line.filter { $0 != "\r" }
    
    let ascii = line == "format ascii 1.0"
    
    guard let vertexLine = try getVertexLine(reader: &reader) else {
        print("Cannot get vertex line from reader. ")
        return nil
    }
    let count = try getVertexCount(from: vertexLine)
    print("Reading " + String(count) + "points. ")
    
    try checkHeader(reader: &reader, prop: "x")
    try checkHeader(reader: &reader, prop: "y")
    try checkHeader(reader: &reader, prop: "z")
    
    var c = 0
    var hasViews = false
    var hasNormals = false
    var hasColors = false
    
    var redIdx: size_t = 0, greenIdx: size_t = 1, blueIdx: size_t = 2
    
    line = readLine()!
    line = line.filter { $0 != "\r" }
    
    while line != "end_header" {
        if hasHeader(line: line, prop: "nx") || hasHeader(line: line, prop: "normal_x") || hasHeader(line: line, prop: "normalx") {
            hasNormals = true
        }
        if hasHeader(line: line, prop: "red") {
            hasColors = true
            redIdx = c
        }
        if hasHeader(line: line, prop: "green") {
            hasColors = true
            greenIdx = c
        }
        if hasHeader(line: line, prop: "blue") {
            hasColors = true
            blueIdx = c
        }
        if hasHeader(line: line, prop: "views") {
            hasViews = true
        }
        
        if c > 100 {
            break
        }
        c += 1
        line = readLine()!
        line = line.filter { $0 != "\r" }
    }
    
    let colorIdxMin = min(redIdx, greenIdx, blueIdx)
    redIdx -= colorIdxMin
    greenIdx -= colorIdxMin
    blueIdx -= colorIdxMin
    if redIdx + greenIdx + blueIdx != 3 {
        throw NSError(domain: "red/green/blue properties need to be contiguous", code: 3)
    }
    
    r.points = Array(repeating: SIMD3<Float>(), count: count)
    if hasNormals {
        r.normals = Array(repeating: SIMD3<Float>(), count: count)
    }
    if hasColors {
        r.colors = Array(repeating: SIMD3<UInt8>(), count: count)
    }
    if hasViews {
        r.views = Array(repeating: UInt8(), count: count)
    }
    
    if ascii {
        for i in 0..<count {
            guard let line = readLine() else {
                throw NSError(domain: "Invalid PLY file (missing vertex data).", code: 1)
            }
            let components = line.split(separator: " ").map { String($0) }
            guard components.count >= 3 else {
                throw NSError(domain: "Invalid PLY file (incomplete vertex data).", code: 1)
            }
            
            r.points[i] = SIMD3<Float>(
                Float(components[0]) ?? 0,
                Float(components[1]) ?? 0,
                Float(components[2]) ?? 0
            )
            
            if hasNormals {
                r.normals[i] = SIMD3<Float>(
                    Float(components[3]) ?? 0,
                    Float(components[4]) ?? 0,
                    Float(components[5]) ?? 0
                )
            }
            
            if hasColors {
                r.colors[i] = SIMD3<UInt8>(
                    UInt8(components[colorIdxMin]) ?? 0,
                    UInt8(components[colorIdxMin + 1]) ?? 0,
                    UInt8(components[colorIdxMin + 2]) ?? 0
                )
            }
            
            if hasViews {
                r.views[i] = UInt8(components.last!) ?? 0
            }
        }
    } else {
        var float3Buffer = [Float](repeating: 0, count: 3)
        var int3Buffer = [UInt8](repeating: 0, count: 3)
        var intBuffer = [UInt8](repeating: 0, count: 1)

        for i in 0..<count {
            let bytesRead = reader.read(&float3Buffer, maxLength: MemoryLayout<Float>.size * 3)
            guard bytesRead == MemoryLayout<Float>.size * 3 else {
                throw NSError(domain: "Failed to read binary vertex data.", code: 4)
            }
            r.points[i] = SIMD3<Float>(float3Buffer[0], float3Buffer[1], float3Buffer[2])

            if hasNormals {
                let bytesRead = reader.read(&float3Buffer, maxLength: MemoryLayout<Float>.size * 3)
                guard bytesRead == MemoryLayout<Float>.size * 3 else {
                    throw NSError(domain: "Failed to read binary normal data.", code: 4)
                }
                r.normals[i] = SIMD3<Float>(float3Buffer[0], float3Buffer[1], float3Buffer[2])
            }

            if hasColors {
                let bytesRead = reader.read(&int3Buffer, maxLength: 3)
                guard bytesRead == 3 else {
                    throw NSError(domain: "Failed to read binary color data.", code: 4)
                }
                r.colors[i] = SIMD3<UInt8>(int3Buffer[0], int3Buffer[1], int3Buffer[2])
            }

            if hasViews {
                let bytesRead = reader.read(&intBuffer, maxLength: 1)
                guard bytesRead == 1 else {
                    throw NSError(domain: "Failed to read binary view count data.", code: 4)
                }
                r.views[i] = intBuffer[0]
            }
        }
    }
    
    reader.close()
    
    return r
}

func readPointSet(filename: String) throws -> PointSet? {
    let path = URL(fileURLWithPath: filename)
    let fileExtension = path.pathExtension.lowercased()

    if fileExtension == "ply" {
        return try fastPlyReadPointSet(filename: filename)
    } else {
        return nil
    }
}

func fileExists(path: String) -> Bool {
    return FileManager.default.fileExists(atPath: path)
}

//
//  Customized Functions
//

protocol KDTree {
    init(_ param1: Int, _ pointSet: PointSet, _ param2: size_t)
    func knnSearch(_ point: SIMD3<Float>, _ count: Int, _ indices: inout [size_t], _ sqrDists: inout [Float])
}

struct MyKDTree: KDTree {
    var param1: Int
    var pointSet: PointSet
    var param2: size_t

    init(_ param1: Int, _ pointSet: PointSet, _ param2: size_t) {
        self.param1 = param1
        self.pointSet = pointSet
        self.param2 = param2
    }

    func knnSearch(_ point: SIMD3<Float>, _ count: Int, _ indices: inout [size_t], _ sqrDists: inout [Float]) {
        // Implement nearest neighbor search logic here
        // For demonstration, we just set dummy data
        for i in 0..<count {
            indices[i] = size_t(i)
            sqrDists[i] = Float(i) * 0.5
        }
    }
}
