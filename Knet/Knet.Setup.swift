// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation

extension Knet {

    func resetBufferIndexes() {
        sBiases = cIOData // Biases come after all the i/o buffers
        sInputs = 0
        sOutputs = cExternalInputs
        sWeights = sBiases + cBiases
    }

    func setupBuffers(_ layerSpecs: [KFCSpec]) {
        pEverything = .allocate(capacity: cEverything)

        layerSpecs.forEach {
            setupBuffers(layerSpec: $0, isInputLayer: $0.inputConnections == nil)
        }
    }

    func setupBuffers(layerSpec: KFCSpec, isInputLayer: Bool) {
        let pBiases = UnsafeBufferPointer(
            rebasing: pEverything[sBiases..<(sBiases + layerSpec.cOutputs)]
        )

        let pInputs = UnsafeBufferPointer(
            rebasing: pEverything[sInputs..<(sInputs + layerSpec.cInputs)]
        )

        let pOutputs = UnsafeBufferPointer(
            rebasing: pEverything[sOutputs..<(sOutputs + layerSpec.cOutputs)]
        )

        let cLayerWeights = layerSpec.cInputs * layerSpec.cOutputs

        let pWeights = UnsafeBufferPointer(
            rebasing: pEverything[sWeights..<(sWeights + cLayerWeights)]
        )

        let fc = KFullyConnected(
            order: layerSpec.order,
            cInputs: layerSpec.cInputs, cOutputs: layerSpec.cOutputs,
            activation: Knet.bnnsActivation(layerSpec.activation),
            pBiases: pBiases, pInputs: pInputs,
            pOutputs: pOutputs, pWeights: pWeights
        )

        fcLookup[layerSpec.name] = fc

        sBiases += pBiases.count
        sInputs += pInputs.count
        sOutputs += pOutputs.count
        sWeights += pWeights.count
    }

    func setupCounts(json netStructure: String) {
        let counts = KnetCounts.setupCounts(json: netStructure)

        self.cExternalInputs = counts.cExternalInputs
        self.cExternalOutputs = counts.cExternalOutputs
        self.cInternalIOputs = counts.cInternalIOputs

        self.cBiases = counts.cBiases
        self.cWeights = counts.cWeights
    }
}

extension Knet {
    func launchNet() {
        resetBufferIndexes()

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

        self.fcStack = fcLookup.values.sorted(by: Knet.orderSort)
    }
}
