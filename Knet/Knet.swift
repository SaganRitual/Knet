// We are a way for the cosmos to know itself. -- C. Sagan

import Accelerate
import Foundation

protocol HasWeightsProtocol {
    var cWeights: Int { get }
}

protocol HasOrderProtocol {
    var order: Int { get }
}

struct KnetSpec: Codable {
    let fullyConnectedLayers: [KFCSpec]
    let poolingLayers: [KPLSpec]
}

class Knet {
    func orderSort(_ lhs: String, _ rhs: String) -> Bool {
        Knet.orderSort(fcLookup[lhs]!, fcLookup[rhs]!)
    }

    static func orderSort(_ lhs: HasOrderProtocol, _ rhs: HasOrderProtocol) -> Bool {
        lhs.order < rhs.order
    }

    var fcLookup = [String: KFullyConnected]()
    var fcStack: [KFullyConnected]!

    var cExternalInputs = 0
    var cExternalOutputs = 0
    var cInternalIOputs = 0
    var cIOData: Int { cExternalInputs + cExternalOutputs + cInternalIOputs }

    var cBiases = 0
    var cWeights = 0
    var cStatics: Int { cBiases + cWeights }

    var layerSpecs: [KFCSpec]!

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

    init(json netStructure: String) {
        setupCounts(json: netStructure)
        resetBufferIndexes()
        setupBuffers(layerSpecs)
        launchNet()
    }

    deinit { pEverything.deallocate() }

    func activate() { fcStack.forEach { $0.activate() } }
}

extension Knet {
    enum Activation: String, Codable { case identity, tanh }
    enum LayerLevel: String, Codable { case top, hidden, bottom }
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
