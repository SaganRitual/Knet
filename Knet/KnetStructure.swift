// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation

enum Config {
    static let mutationProbability = 0.85
    static let randomerCSamplesToGenerate = 50_000
    static let activation = Knet.Activation.identity
    static let poolingFunction = Knet.PoolingFunction.average
    static let cInputsFunnel = 2 * imageWidth * imageHeight
    static let cOutputsFunnel = imageWidth
    static let cOutputsPlayerFc = imageWidth * imageHeight
    static let cOutputsAggregator = imageWidth
    static let imageHeight = 7
    static let imageWidth = 7
    static let kernelHeight = 4
    static let kernelWidth = 4
}

class KnetStructure {
    var startBiases = 0
    var startInputs = 0
    var startOutputs = 0
    var startWeights = 0

    var hiddenLayers = [KnetLayerSpecProtocol]()
    var motorLayer = [KnetLayerSpecProtocol]()
    var sensorLayer = [KnetLayerSpecProtocol]()

    var allLayers: [KnetLayerSpecProtocol] { upperLayers + motorLayer }
    var upperLayers: [KnetLayerSpecProtocol] { sensorLayer + hiddenLayers }

    init() {
        let blackPoolSpec = KPLSpec(
            activation: Config.activation, poolingFunction: Config.poolingFunction,
            imageWidth: Config.imageWidth, imageHeight: Config.imageHeight,
            kernelWidth: Config.kernelWidth, kernelHeight: Config.kernelHeight
        )

        let redPoolSpec = KPLSpec(
            activation: Config.activation, poolingFunction: Config.poolingFunction,
            imageWidth: Config.imageWidth, imageHeight: Config.imageHeight,
            kernelWidth: Config.kernelWidth, kernelHeight: Config.kernelHeight
        )

        sensorLayer.append(contentsOf: [blackPoolSpec, redPoolSpec])

        let blackPoolFc = KFCSpec(
            activation: Config.activation,
            cInputs: blackPoolSpec.imageArea, cOutputs: blackPoolSpec.imageArea,
            aggregateOutputBuffer: true
        )

        let redPoolFc = KFCSpec(
            activation: Config.activation,
            cInputs: redPoolSpec.imageArea, cOutputs: redPoolSpec.imageArea,
            aggregateOutputBuffer: true
        )

        hiddenLayers.append(contentsOf: [blackPoolFc, redPoolFc])

        connect(blackPoolSpec, to: blackPoolFc)
        connect(redPoolSpec, to: redPoolFc)

        let aggregator = KFCSpec(
            activation: Config.activation,
            cInputs: blackPoolSpec.imageArea + redPoolSpec.imageArea,
            cOutputs: Config.cOutputsAggregator,
            aggregateInputBuffer: true
        )

        connect(blackPoolFc, to: aggregator)
        connect(redPoolFc, to: aggregator)

        hiddenLayers.append(aggregator)

        let funnel = KFCSpec(
            activation: Config.activation,
            cInputs: Config.cOutputsFunnel,
            cOutputs: Config.cOutputsFunnel
        )

        connect(aggregator, to: funnel)

        motorLayer = [funnel]
    }

    func connect(
        _ upperLayer: KnetLayerSpecProtocol,
        to lowerLayer: KnetLayerSpecProtocol
    ) {
        upperLayer.outputSpecs.append(lowerLayer)
        lowerLayer.inputSpecs.append(upperLayer)
    }
}
