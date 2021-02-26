// We are a way for the cosmos to know itself. -- C. Sagan

import Accelerate
import Foundation

protocol HasWeightsProtocol: KnetLayerSpecProtocol {
    var cWeights: Int { get }
}

extension HasWeightsProtocol {
    var cWeights: Int { cInputs * cOutputs }
}

protocol KnetLayerSpecProtocol: class {
    var activation: Knet.Activation { get }
    var cBiases: Int { get }
    var cInputs: Int { get }
    var cOutputs: Int { get }

    var aggregateOutputBuffer: Bool { get }
    var aggregateInputBuffer: Bool { get }

    var inputSpecs: [KnetLayerSpecProtocol] { get set }
    var outputSpecs: [KnetLayerSpecProtocol] { get set }

    func makeLayer(
        pBiases: UnsafeBufferPointer<Float>,
        pInputs: UnsafeBufferPointer<Float>,
        pOutputs: UnsafeBufferPointer<Float>,
        pWeights: UnsafeBufferPointer<Float>?
    ) -> KnetLayerProtocol
}

extension KnetLayerSpecProtocol {
    var cBiases: Int { cOutputs }
}

protocol KnetLayerProtocol: class {
    var filter: BNNSFilter { get }
    var pInputs: UnsafeMutableRawPointer { get }
    var pOutputs: UnsafeMutableRawPointer { get }

    var layerInputBuffer: UnsafeBufferPointer<Float>! { get set }
    var layerOutputBuffer: UnsafeBufferPointer<Float>! { get set }
}

extension KnetLayerProtocol {
    func activate() {
        print("inputs  \(layerInputBuffer!.map { $0 })")
        BNNSFilterApply(filter, pInputs, pOutputs)
        print("outputs \(layerOutputBuffer!.map { $0 })")
    }
}

class KnetLayer: KnetLayerProtocol {
    let filter: BNNSFilter
    let pInputs: UnsafeMutableRawPointer
    let pOutputs: UnsafeMutableRawPointer

    var layerInputBuffer: UnsafeBufferPointer<Float>!
    var layerOutputBuffer: UnsafeBufferPointer<Float>!

    init(
        pInputs: UnsafeBufferPointer<Float>,
        pOutputs: UnsafeBufferPointer<Float>,
        filter: BNNSFilter
    ) {
        self.layerInputBuffer = pInputs
        self.layerOutputBuffer = pOutputs

        self.filter = filter
        self.pInputs = UnsafeMutableRawPointer(mutating: pInputs.baseAddress!)
        self.pOutputs = UnsafeMutableRawPointer(mutating: pOutputs.baseAddress!)
    }
}
