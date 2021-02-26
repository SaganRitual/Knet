// We are a way for the cosmos to know itself. -- C. Sagan

import Accelerate
import Foundation

class KFCSpec: KnetLayerSpecProtocol, HasWeightsProtocol {
    var activation: Knet.Activation

    var cInputs = 0
    var cOutputs = 0
    var cWeights: Int { cInputs * cOutputs }

    var aggregateOutputBuffer = false
    var aggregateInputBuffer = false

    var startInputs = 0
    var startOutputs: Int { startInputs + cInputs }
    var startBiases: Int { startOutputs + cOutputs }
    var startWeights: Int { startBiases + cOutputs }

    var inputSpecs = [KnetLayerSpecProtocol]()
    var outputSpecs = [KnetLayerSpecProtocol]()

    init(
        activation: Knet.Activation, cInputs: Int, cOutputs: Int,
        aggregateInputBuffer: Bool = false, aggregateOutputBuffer: Bool = false
    ) {
        self.activation = activation
        self.cInputs = cInputs
        self.cOutputs = cOutputs
        self.aggregateOutputBuffer = aggregateOutputBuffer
        self.aggregateInputBuffer = aggregateInputBuffer
    }

    func makeLayer(
        pBiases: UnsafeBufferPointer<Float>,
        pInputs: UnsafeBufferPointer<Float>,
        pOutputs: UnsafeBufferPointer<Float>,
        pWeights: UnsafeBufferPointer<Float>?
    ) -> KnetLayerProtocol {
        KFullyConnected(
            activation: Knet.bnnsActivation(activation),
            cInputs: cInputs, cOutputs: cOutputs,
            pBiases: pBiases, pInputs: pInputs,
            pOutputs: pOutputs, pWeights: pWeights!
        )
    }
}

class KFullyConnected: KnetLayer {
    let cInputs: Int
    let cOutputs: Int

    init(
        activation: BNNSActivation, cInputs: Int, cOutputs: Int,
        pBiases: UnsafeBufferPointer<Float>,
        pInputs: UnsafeBufferPointer<Float>,
        pOutputs: UnsafeBufferPointer<Float>,
        pWeights: UnsafeBufferPointer<Float>
    ) {
        var layerParameters = KFullyConnected.makeLayerParameters(
            cInputs: cInputs, cOutputs: cOutputs, activation: activation,
            pBiases: pBiases, pWeights: pWeights
        )

        self.cInputs = cInputs
        self.cOutputs = cOutputs

        guard let filter = BNNSFilterCreateLayerFullyConnected(
            &layerParameters, &Knet.filterParameters
        ) else { fatalError("What is it this time!") }

        super.init(pInputs: pInputs, pOutputs: pOutputs, filter: filter)
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
