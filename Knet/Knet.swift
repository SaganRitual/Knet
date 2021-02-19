// We are a way for the cosmos to know itself. -- C. Sagan

import Accelerate
import Foundation

protocol HasOrderProtocol {
    var order: Int { get }
}

class Knet {
    func orderSort(_ lhs: String, _ rhs: String) -> Bool {
        Knet.orderSort(fcLookup[lhs]!, fcLookup[rhs]!)
    }

    static func orderSort(_ lhs: HasOrderProtocol, _ rhs: HasOrderProtocol) -> Bool {
        lhs.order < rhs.order
    }

    var fcLookup = [String: KFullyConnected]()
    var fcStack: [KFullyConnected]!

    var cExternalInputs = 0
    var cExternalOutputs = 0
    var cInternalIOputs = 0
    var cIOData: Int { cExternalInputs + cExternalOutputs + cInternalIOputs }

    var cBiases = 0
    var cWeights = 0
    var cStatics: Int { cBiases + cWeights }

    var layerSpecs: [KFCSpec]!

    var pEverything: UnsafeMutableBufferPointer<Float>!
    var cEverything: Int { cIOData + cBiases + cWeights }

    var sBiases = 0
    var sInputs = 0
    var sOutputs = 0
    var sWeights = 0

    var biasesBuffer: UnsafeMutableBufferPointer<Float>!
    var inputBuffer: UnsafeMutableBufferPointer<Float>!
    var outputBuffer: UnsafeMutableBufferPointer<Float>!
    var staticsBuffer: UnsafeMutableBufferPointer<Float>!
    var weightsBuffer: UnsafeMutableBufferPointer<Float>!

    init(json netStructure: String) {
        setupCounts(json: netStructure)
        resetBufferIndexes()
        setupBuffers(layerSpecs)
        launchNet()
    }

    func activate() { fcStack.forEach { $0.activate() } }
}

private extension Knet {

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

        let nextSInputs = isInputLayer ? cExternalInputs : layerSpec.cInputs

        let pInputs = UnsafeBufferPointer(
            rebasing: pEverything[sInputs..<(sInputs + nextSInputs)]
        )

        let pOutputs = UnsafeBufferPointer(
            rebasing: pEverything[sOutputs..<(sOutputs + layerSpec.cOutputs)]
        )

        let cWeights = layerSpec.cInputs * layerSpec.cOutputs

        let pWeights = UnsafeBufferPointer(
            rebasing: pEverything[sWeights..<(sWeights + cWeights)]
        )

        let fc = KFullyConnected(
            order: layerSpec.order,
            cInputs: layerSpec.cInputs, cOutputs: layerSpec.cOutputs,
            activation: KFullyConnected.bnnsActivation(layerSpec.activation),
            pBiases: pBiases, pInputs: pInputs,
            pOutputs: pOutputs, pWeights: pWeights
        )

        fcLookup[layerSpec.name] = fc

        sBiases += pBiases.count
        sInputs = sOutputs
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

        self.layerSpecs = counts.layerSpecs
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
