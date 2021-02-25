// We are a way for the cosmos to know itself. -- C. Sagan

import Accelerate
import Foundation

protocol KnetLayerSpecProtocol: HasOrderProtocol, HasWeightsProtocol {
    var activation: Knet.Activation { get }
    var cInputs: Int { get }
    var cOutputs: Int { get }
    var layerLevel: Knet.LayerLevel { get }
    var name: String { get }
    var order: Int { get }
    var inputConnections: [String]? { get }
    var outputConnection: String? { get }
}

extension KnetLayerSpecProtocol {
    var cWeights: Int { cInputs * cOutputs }
}

protocol KnetLayerProtocol: class, HasOrderProtocol {
    var order: Int { get }

    var filter: BNNSFilter { get }
    var pInputs: UnsafeMutableRawPointer { get }
    var pOutputs: UnsafeMutableRawPointer { get }

    var layerInputBuffer: UnsafeBufferPointer<Float>! { get set }
    var layerOutputBuffer: UnsafeBufferPointer<Float>! { get set }
}

extension KnetLayerProtocol {
    func activate() { BNNSFilterApply(filter, pInputs, pOutputs) }
}

class KnetLayer: KnetLayerProtocol {

    let order: Int

    let filter: BNNSFilter
    let pInputs: UnsafeMutableRawPointer
    let pOutputs: UnsafeMutableRawPointer

    var layerInputBuffer: UnsafeBufferPointer<Float>!
    var layerOutputBuffer: UnsafeBufferPointer<Float>!

    init(
        order: Int, cInputs: Int, cOutputs: Int,
        activation: BNNSActivation,
        pBiases: UnsafeBufferPointer<Float>,
        pInputs: UnsafeBufferPointer<Float>,
        pOutputs: UnsafeBufferPointer<Float>,
        pWeights: UnsafeBufferPointer<Float>?,
        filter: BNNSFilter
    ) {
        self.order = order
        self.layerInputBuffer = pInputs
        self.layerOutputBuffer = pOutputs

        self.filter = filter
        self.pInputs = UnsafeMutableRawPointer(mutating: pInputs.baseAddress!)
        self.pOutputs = UnsafeMutableRawPointer(mutating: pOutputs.baseAddress!)
    }
}
