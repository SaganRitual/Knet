// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation

class KnetCounts: CustomDebugStringConvertible {
    var debugDescription: String {
        "cExtIn: \(cExternalInputs)"
        + " cExtOut: \(cExternalOutputs)"
        + " cIntIOP: \(cInternalIOputs)"
        + " cIOData: \(cIOData)"
        + " cBiases: \(cBiases) cWeights: \(cWeights) cStatics: \(cStatics)"
    }

    var cExternalInputs: Int = 0
    var cExternalOutputs: Int = 0
    var cInternalIOputs: Int = 0
    var cIOData: Int { cExternalInputs + cInternalIOputs + cExternalOutputs }

    var cBiases: Int = 0
    var cWeights: Int = 0
    var cStatics: Int { cBiases + cWeights }

    func setupCounts(_ netStructure: KnetStructure) -> KnetCounts {
        cExternalInputs = netStructure.sensorLayer.reduce(0) { $0 + $1.cInputs }
        cExternalOutputs = netStructure.motorLayer.reduce(0) { $0 + $1.cOutputs }

        cInternalIOputs = netStructure.hiddenLayers.reduce(0) {
            $0 + $1.cInputs + $1.cOutputs
        }

        cBiases = netStructure.upperLayers.reduce(0) { $0 + $1.cBiases }

        cWeights = netStructure.allLayers
            .compactMap { $0 as? HasWeightsProtocol }
            .reduce(0) { $0 + $1.cWeights }

        return self // For chaining
    }
}
