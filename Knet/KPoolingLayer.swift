// We are a way for the cosmos to know itself. -- C. Sagan

import Accelerate
import Foundation

class KPLSpec: KnetLayerSpecProtocol {
    let activation: Knet.Activation
    let poolingFunction: Knet.PoolingFunction

    let imageWidth: Int
    let imageHeight: Int
    let kernelWidth: Int
    let kernelHeight: Int

    var imageArea: Int { imageWidth * imageHeight }
    var cInputs: Int { imageArea }
    var cOutputs: Int { imageArea }

    var startInputs = 0
    var startOutputs: Int { startInputs + cInputs }
    var startBiases: Int { startOutputs + cOutputs }

    var inputSpecs = [KnetLayerSpecProtocol]()
    var outputSpecs = [KnetLayerSpecProtocol]()

    init(
        activation: Knet.Activation,
        poolingFunction: Knet.PoolingFunction,
        imageWidth: Int, imageHeight: Int,
        kernelWidth: Int, kernelHeight: Int
    ) {
        self.activation = activation
        self.poolingFunction = poolingFunction
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.kernelWidth = kernelWidth
        self.kernelHeight = kernelHeight
    }

    func makeLayer(
        pBiases: UnsafeBufferPointer<Float>,
        pInputs: UnsafeBufferPointer<Float>,
        pOutputs: UnsafeBufferPointer<Float>,
        pWeights: UnsafeBufferPointer<Float>?
    ) -> KnetLayerProtocol {
        KPoolingLayer(
            layerSpec: self, pBiases: pBiases,
            pInputs: pInputs, pOutputs: pOutputs
        )
    }
}

class KPoolingLayer: KnetLayer {
    init(
        layerSpec: KPLSpec,
        pBiases: UnsafeBufferPointer<Float>,
        pInputs: UnsafeBufferPointer<Float>,
        pOutputs: UnsafeBufferPointer<Float>
    ) {
        var layerParameters = KPoolingLayer.makeLayerParameters(
            layerSpec: layerSpec,
            pBiases: pBiases, pInputs: pInputs, pOutputs: pOutputs
        )

        guard let filter = BNNSFilterCreateLayerPooling(
            &layerParameters, &Knet.filterParameters
        ) else { fatalError("What is it this time!") }

        super.init(pInputs: pInputs, pOutputs: pOutputs, filter: filter)
    }
}

private extension KPoolingLayer {
    // swiftlint:disable function_body_length
    static func makeLayerParameters(
        layerSpec: KPLSpec,
        pBiases: UnsafeBufferPointer<Float>,
        pInputs: UnsafeBufferPointer<Float>,
        pOutputs: UnsafeBufferPointer<Float>
    ) -> BNNSLayerParametersPooling {
        let rpBiases = UnsafeMutableRawPointer(mutating: pBiases.baseAddress)

        let i_desc = BNNSNDArrayDescriptor(
            flags: BNNSNDArrayFlags(0),
            layout: BNNSDataLayoutImageCHW,
            size: (layerSpec.imageWidth, layerSpec.imageHeight, 1, 0, 0, 0, 0, 0),
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
            size: (layerSpec.imageWidth, layerSpec.imageHeight, 1, 0, 0, 0, 0, 0),
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
            size: (layerSpec.cBiases, 1, 1, 0, 0, 0, 0, 0),
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
            activation: Knet.bnnsActivation(layerSpec.activation),
            pooling_function: Knet.bnnsPoolingFunction(layerSpec.poolingFunction),
            k_width: layerSpec.kernelWidth, k_height: layerSpec.kernelHeight,
            x_stride: 1, y_stride: 1,
            x_dilation_stride: 0, y_dilation_stride: 0,
            x_padding: layerSpec.kernelWidth / 2,
            y_padding: layerSpec.kernelHeight / 2,
            pad: (0, 0, 0, 0)
        )
    }
    // swiftlint:enable function_body_length
}
