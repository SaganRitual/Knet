// We are a way for the cosmos to know itself. -- C. Sagan

import Accelerate
import Foundation

class Knet {
    var layerStack = [KnetLayerProtocol]()

    var cExternalInputs = 0
    var cExternalOutputs = 0
    var cInternalIOputs = 0
    var cIOData: Int { cExternalInputs + cExternalOutputs + cInternalIOputs }

    var cBiases = 0
    var cWeights = 0
    var cStatics: Int { cBiases + cWeights }

    var pEverything: UnsafeMutableBufferPointer<Float>!
    var cEverything: Int { cIOData + cBiases + cWeights }

    var sBiases = 0
    var sInputs = 0
    var sOutputs = 0
    var sWeights = 0

    var biasesBuffer: UnsafeMutableBufferPointer<Float>!
    var inputBuffer: UnsafeMutableBufferPointer<Float>!
    var outputBuffer: UnsafeMutableBufferPointer<Float>!
    var staticsBuffer: UnsafeMutableBufferPointer<Float>!
    var weightsBuffer: UnsafeMutableBufferPointer<Float>!

    init(_ netStructure: KnetStructure) {
        setupCounts(netStructure)
        setupBuffers(netStructure)
        launchNet()
    }

    deinit { pEverything.deallocate() }

    func activate() { layerStack.forEach { $0.activate() } }
}

extension Knet {
    enum Activation: String, Codable { case identity, tanh }
    enum LayerLevel: String, Codable, CaseIterable { case top, hidden, bottom }
    enum PoolingFunction: String, Codable { case average, max }

    static func bnnsActivation(_ kActivation: Activation) -> BNNSActivation {
        switch kActivation {
        case .identity: return BNNSActivation(function: .identity)
        case .tanh:     return BNNSActivation(function: .tanh)
        }
    }

    static func bnnsPoolingFunction(
        _ kPoolingFunction: PoolingFunction
    ) -> BNNSPoolingFunction {
        switch kPoolingFunction {
        case .average: return BNNSPoolingFunction.average
        case .max:     return BNNSPoolingFunction.max
        }
    }

    static var filterParameters = BNNSFilterParameters(
        flags: BNNSFlags.useClientPointer.rawValue, n_threads: 0,
        alloc_memory: nil, free_memory: nil
    )
}
