// We are a way for the cosmos to know itself. -- C. Sagan

import Accelerate
import Foundation

struct KPLSpec: Codable, KnetLayerSpecProtocol {
    let activation: Knet.Activation
    let inputWidth: Int
    let inputHeight: Int
    let cOutputs: Int
    let kernelSide: Int
    let layerLevel: Knet.LayerLevel
    let name: String
    let order: Int
    let poolingFunction: Knet.PoolingFunction
    let inputConnections: [String]?
    let outputConnection: String?

    var cInputs: Int { inputWidth * inputHeight }
}

class KPoolingLayer: KnetLayer {

    init(
        order: Int, inputWidth: Int, inputHeight: Int,
        kernelSide: Int, cOutputs: Int, activation: BNNSActivation,
        poolingFunction: BNNSPoolingFunction,
        pBiases: UnsafeBufferPointer<Float>,
        pInputs: UnsafeBufferPointer<Float>,
        pOutputs: UnsafeBufferPointer<Float>,
        filter: BNNSFilter
    ) {
        var layerParameters = KPoolingLayer.makeLayerParameters(
            inputWidth: inputWidth, inputHeight: inputHeight,
            kernelSide: kernelSide, cOutputs: cOutputs, activation: activation,
            poolingFunction: poolingFunction, pBiases: pBiases
        )

        guard let filter = BNNSFilterCreateLayerPooling(
            &layerParameters, &Knet.filterParameters
        ) else { fatalError("What is it this time!") }

        super.init(
            order: order, cInputs: inputWidth * inputHeight, cOutputs: cOutputs,
            activation: activation, pBiases: pBiases, pInputs: pInputs,
            pOutputs: pOutputs, pWeights: nil, filter: filter
        )
    }
}

private extension KPoolingLayer {
    // swiftlint:disable function_body_length
    // swiftlint:disable function_parameter_count
    static func makeLayerParameters(
        inputWidth: Int, inputHeight: Int, kernelSide: Int, cOutputs: Int,
        activation: BNNSActivation, poolingFunction: BNNSPoolingFunction,
        pBiases: UnsafeBufferPointer<Float>
    ) -> BNNSLayerParametersPooling {
        let rpBiases = UnsafeMutableRawPointer(mutating: pBiases.baseAddress)

        let i_desc = BNNSNDArrayDescriptor(
            flags: BNNSNDArrayFlags(0),
            layout: BNNSDataLayoutImageCHW,
            size: (inputWidth, inputHeight, 1, 0, 0, 0, 0, 0),
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
            layout: BNNSDataLayoutImageCHW,
            size: (inputWidth, inputHeight, 1, 0, 0, 0, 0, 0),
            stride: (0, 0, 0, 0, 0, 0, 0, 0),
            data: nil,
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

        return BNNSLayerParametersPooling(
            i_desc: i_desc, o_desc: o_desc, bias: bias,
            activation: activation, pooling_function: poolingFunction,
            k_width: kernelSide, k_height: kernelSide,
            x_stride: 1, y_stride: 1,
            x_dilation_stride: 0, y_dilation_stride: 0,
            x_padding: kernelSide / 2, y_padding: kernelSide / 2,
            pad: (0, 0, 0, 0)
        )
    }
    // swiftlint:enable function_body_length
    // swiftline:enable function_parameter_count
}
