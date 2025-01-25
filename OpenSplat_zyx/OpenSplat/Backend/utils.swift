//
//  utils.swift
//  OpenSplat
//
//  Created by Yuxuan Zhang on 1/2/25.
//

import Foundation

class InfiniteRandomIterator<T> {
    typealias VecType = [T]
    private var v: VecType
    private var i: Int
    private var engine: RandomNumberGenerator

    init(_ v: VecType) {
        self.v = v
        self.i = 0
        self.engine = SystemRandomNumberGenerator()
        shuffleV()
    }

    func shuffleV() {
        v.shuffle(using: &engine)
        i = 0
    }

    func next() -> T {
        let ret = v[i]
        i += 1
        if i >= v.count {
            shuffleV()
        }
        return ret
    }
}

func parallel_for<IndexType: BinaryInteger>(begin: IndexType, end: IndexType, funcBody: @escaping (IndexType) -> Void) {
    let range = Int(end - begin)
    guard range > 0 else { return }

    let numThreads = min(ProcessInfo.processInfo.activeProcessorCount, range)
    let chunkSize = (range + numThreads - 1) / numThreads
    let dispatchGroup = DispatchGroup()

    for i in 0..<numThreads {
        let chunkBegin = begin + IndexType(i * chunkSize)
        let chunkEnd = min(chunkBegin + IndexType(chunkSize), end)

        dispatchGroup.enter()
        DispatchQueue.global().async {
            for item in stride(from: Int(chunkBegin), to: Int(chunkEnd), by: 1) {
                funcBody(IndexType(item))
            }
            dispatchGroup.leave()
        }
    }

    dispatchGroup.wait()
}
