// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation

enum Config {
    static let activation = Knet.Activation.identity
    static let poolingFunction = Knet.PoolingFunction.average
    static let cInputsFunnel = 3
    static let cOutputsFunnel = 3
    static let cOutputsPlayerFc = 3
    static let cOutputsAggregator = 3
    static let imageHeight = 3
    static let imageWidth = 3
    static let kernelHeight = 3
    static let kernelWidth = 3
}

struct KnetStructure {
    var startBiases = 0
    var startInputs = 0
    var startOutputs = 0
    var startWeights = 0

    var hiddenLayers = [KnetLayerSpecProtocol]()
    var motorLayer = [KnetLayerSpecProtocol]()
    var sensorLayer = [KnetLayerSpecProtocol]()

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
            cInputs: blackPoolSpec.imageArea, cOutputs: blackPoolSpec.imageArea
        )

        let redPoolFc = KFCSpec(
            activation: Config.activation,
            cInputs: redPoolSpec.imageArea, cOutputs: redPoolSpec.imageArea
        )

        hiddenLayers.append(contentsOf: [blackPoolFc, redPoolFc])

        connect(blackPoolSpec, to: blackPoolFc)
        connect(redPoolSpec, to: redPoolFc)

        let aggregator = KFCSpec(
            activation: Config.activation,
            cInputs: blackPoolSpec.imageArea + redPoolSpec.imageArea,
            cOutputs: Config.cOutputsAggregator
        )

        connect(blackPoolFc, to: aggregator)
        connect(redPoolFc, to: aggregator)

        hiddenLayers.append(aggregator)

        let funnel = KFCSpec(
            activation: Config.activation,
            cInputs: Config.cInputsFunnel,
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
