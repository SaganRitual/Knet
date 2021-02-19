// We are a way for the cosmos to know itself. -- C. Sagan

import Accelerate
import Foundation

struct KFCSpec: Codable, HasOrderProtocol {
    enum Activation: String, Codable { case identity, tanh }

    let activation: Activation
    let cInputs: Int
    let cOutputs: Int
    let name: String
    let order: Int
    let inputConnections: [String]?
    let outputConnection: String?
}

class KFullyConnected: HasOrderProtocol {
    enum Level { case top, hidden, bottom }

    static func bnnsActivation(_ kActivation: KFCSpec.Activation) -> BNNSActivation {
        switch kActivation {
        case .identity: return BNNSActivation(function: .identity)
        case .tanh:     return BNNSActivation(function: .tanh)
        }
    }

    static var filterParameters = BNNSFilterParameters(
        flags: BNNSFlags.useClientPointer.rawValue, n_threads: 0,
        alloc_memory: nil, free_memory: nil
    )

    let order: Int

    let filter: BNNSFilter
    let pInputs: UnsafeMutableRawPointer
    let pOutputs: UnsafeMutableRawPointer

    init(
        order: Int, cInputs: Int, cOutputs: Int,
        activation: BNNSActivation,
        pBiases: UnsafeBufferPointer<Float>,
        pInputs: UnsafeBufferPointer<Float>,
        pOutputs: UnsafeBufferPointer<Float>,
        pWeights: UnsafeBufferPointer<Float>
    ) {
        var layerParameters = KFullyConnected.makeLayerParameters(
            cInputs: cInputs, cOutputs: cOutputs, activation: activation,
            pBiases: pBiases, pWeights: pWeights
        )

        guard let f = BNNSFilterCreateLayerFullyConnected(
            &layerParameters, &KFullyConnected.filterParameters
        ) else { fatalError("What is it this time!") }

        self.order = order

        self.filter = f
        self.pInputs = UnsafeMutableRawPointer(mutating: pInputs.baseAddress!)
        self.pOutputs = UnsafeMutableRawPointer(mutating: pOutputs.baseAddress!)
    }

    func activate() { BNNSFilterApply(filter, pInputs, pOutputs) }
}

private extension KFullyConnected {
    // swiftlint:disable function_body_length
    static func makeLayerParameters(
        cInputs: Int, cOutputs: Int,
        activation: BNNSActivation,
        pBiases: UnsafeBufferPointer<Float>,
        pWeights: UnsafeBufferPointer<Float>
    ) -> BNNSLayerParametersFullyConnected {
        let rpBiases = UnsafeMutableRawPointer(mutating: pBiases.baseAddress)
        let rpWeights = UnsafeMutableRawPointer(mutating: pWeights.baseAddress)

        let i_desc = BNNSNDArrayDescriptor(
            flags: BNNSNDArrayFlags(0),
            layout: BNNSDataLayoutVector,
            size: (cInputs, 1, 1, 0, 0, 0, 0, 0),
            stride: (0, 0, 0, 0, 0, 0, 0, 0),
            data: nil,
            data_type: .float,
            table_data: nil,
            table_data_type: .float,
            data_scale: 0,
            data_bias: 0
        )

        let o_desc = BNNSNDArrayDescriptor(
            flags: BNNSNDArrayFlags(0),
            layout: BNNSDataLayoutVector,
            size: (cOutputs, 1, 1, 0, 0, 0, 0, 0),
            stride: (0, 0, 0, 0, 0, 0, 0, 0),
            data: nil,
            data_type: .float,
            table_data: nil,
            table_data_type: .float,
            data_scale: 0,
            data_bias: 0
        )

        let w_desc = BNNSNDArrayDescriptor(
            flags: BNNSNDArrayFlags(0),
            layout: BNNSDataLayoutRowMajorMatrix,
            size: (cInputs, cOutputs, 1, 0, 0, 0, 0, 0),
            stride: (0, 0, 0, 0, 0, 0, 0, 0),
            data: rpWeights,
            data_type: .float,
            table_data: nil,
            table_data_type: .float,
            data_scale: 0,
            data_bias: 0
        )

        let bias = BNNSNDArrayDescriptor(
            flags: BNNSNDArrayFlags(0),
            layout: BNNSDataLayoutVector,
            size: (cOutputs, 1, 1, 0, 0, 0, 0, 0),
            stride: (0, 0, 0, 0, 0, 0, 0, 0),
            data: rpBiases,
            data_type: .float,
            table_data: nil,
            table_data_type: .float,
            data_scale: 0,
            data_bias: 0
        )

        return BNNSLayerParametersFullyConnected(
            i_desc: i_desc, w_desc: w_desc, o_desc: o_desc, bias: bias,
            activation: activation
        )
    }
    // swiftlint:enable function_body_length
}

struct KDescriptorSet {
    let i_desc: BNNSNDArrayDescriptor
    let o_desc: BNNSNDArrayDescriptor
    let w_desc: BNNSNDArrayDescriptor
    let bias: BNNSNDArrayDescriptor
}
