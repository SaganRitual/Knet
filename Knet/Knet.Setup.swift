// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation

extension Knet {
    func advancePointers(_ layerSpec: KnetLayerSpecProtocol) {
        sBiases += layerSpec.cBiases
        sInputs += layerSpec.cInputs
        sOutputs += layerSpec.cOutputs
    }

    func setupAggregateBuffers(
        _ toAggregate: [KnetLayerSpecProtocol]
    ) {
        let sink = toAggregate.last!

        for layerSpec in toAggregate.dropLast() {
            setupBuffers(layerSpec) { advancePointers(layerSpec) }
        }

        setupBuffers(sink) {
            sBiases += sink.cBiases
            sOutputs += sink.cOutputs
            sInputs += sink.cInputs
        }
    }

    func setupBuffers(_ netStructure: KnetStructure) {
        pEverything = .allocate(capacity: cEverything)

        var toAggregate = [KnetLayerSpecProtocol]()

        for layerSpec in netStructure.allLayers {

            if layerSpec.aggregateOutputBuffer {
                toAggregate.append(layerSpec)
            } else if layerSpec.aggregateInputBuffer {
                toAggregate.append(layerSpec)
                setupAggregateBuffers(toAggregate)
            } else {
                setupBuffers(layerSpec) { advancePointers(layerSpec) }
            }
        }
    }

    func setupBuffers(
        _ layerSpec: KnetLayerSpecProtocol,
        _ advancePointers: () -> Void
    ) {
        let pBiases = UnsafeBufferPointer(
            rebasing: pEverything[sBiases..<(sBiases + layerSpec.cBiases)]
        )

        let pInputs = UnsafeBufferPointer(
            rebasing: pEverything[sInputs..<(sInputs + layerSpec.cInputs)]
        )

        let pOutputs = UnsafeBufferPointer(
            rebasing: pEverything[sOutputs..<(sOutputs + layerSpec.cOutputs)]
        )

        let pWeights: UnsafeBufferPointer<Float>?
        if let ls = layerSpec as? HasWeightsProtocol, ls.cWeights > 0 {
            pWeights = UnsafeBufferPointer(
                rebasing: pEverything[sWeights..<(sWeights + ls.cWeights)]
            )
        } else { pWeights = nil }

        let layer = layerSpec.makeLayer(
            pBiases: pBiases, pInputs: pInputs,
            pOutputs: pOutputs, pWeights: pWeights
        )

        layerStack.append(layer)

        advancePointers()
    }

    func setBufferStarts() {
        sInputs = 0
        sOutputs = cExternalInputs
        sBiases = cIOData
        sWeights = sBiases + cBiases
    }

    func setupCounts(_ netStructure: KnetStructure) {
        let counts = KnetCounts().setupCounts(netStructure)

        self.cExternalInputs = counts.cExternalInputs
        self.cExternalOutputs = counts.cExternalOutputs
        self.cInternalIOputs = counts.cInternalIOputs

        self.cBiases = counts.cBiases
        self.cWeights = counts.cWeights
    }
}

extension Knet {
    func launchNet() {
        setBufferStarts()

        self.biasesBuffer = UnsafeMutableBufferPointer(
            rebasing: pEverything[sBiases..<(sBiases + cBiases)]
        )

        self.inputBuffer = UnsafeMutableBufferPointer(
            rebasing: pEverything[sInputs..<(sInputs + cExternalInputs)]
        )

        self.outputBuffer = UnsafeMutableBufferPointer(
            rebasing: pEverything[(cIOData - cExternalOutputs)..<cIOData]
        )

        self.staticsBuffer = UnsafeMutableBufferPointer(
            rebasing: pEverything[sBiases..<cEverything]
        )

        self.weightsBuffer = UnsafeMutableBufferPointer(
            rebasing: pEverything[sWeights..<(sWeights + cWeights)]
        )
    }
}
