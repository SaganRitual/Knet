// We are a way for the cosmos to know itself. -- C. Sagan

import Accelerate
import Foundation

struct KFCSpec: Codable, KnetLayerSpecProtocol {
    let activation: Knet.Activation
    let cInputs: Int
    let cOutputs: Int
    var layerLevel: Knet.LayerLevel
    let name: String
    let order: Int
    let inputConnections: [String]?
    let outputConnection: String?
}

class KFullyConnected: KnetLayer {
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

        guard let filter = BNNSFilterCreateLayerFullyConnected(
            &layerParameters, &Knet.filterParameters
        ) else { fatalError("What is it this time!") }

        super.init(
            order: order, cInputs: cInputs, cOutputs: cOutputs,
            activation: activation, pBiases: pBiases, pInputs: pInputs,
            pOutputs: pOutputs, pWeights: pWeights, filter: filter
        )
    }
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
