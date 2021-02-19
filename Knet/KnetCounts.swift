// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation

struct KnetCounts: CustomDebugStringConvertible {
    var debugDescription: String {
        "cExtIn: \(cExternalInputs)"
        + " cExtOut: \(cExternalOutputs)"
        + " cIntIOP: \(cInternalIOputs)"
        + " cIOData: \(cIOData)"
        + " cBiases: \(cBiases) cWeights: \(cWeights) cStatics: \(cStatics)"
    }

    let layerSpecs: [KFCSpec]

    let cExternalInputs: Int
    let cExternalOutputs: Int
    let cInternalIOputs: Int
    let cIOData: Int

    let cBiases: Int
    let cWeights: Int
    let cStatics: Int
}

extension KnetCounts {
    static func setupCounts(json netStructure: String) -> KnetCounts {
        let decoder = JSONDecoder()

        guard let layerSpecs = try? decoder.decode(
            [KFCSpec].self, from: netStructure.data(using: .utf8)!
        )
        .sorted(by: Knet.orderSort) else {
            fatalError("Couldn't decode your crappy json, loser")
        }

        var cExternalInputs = 0
        var cExternalOutputs = 0
        var cInternalIOputs = 0
        var cBiases = 0
        var cWeights = 0

        let sensorSpecs = layerSpecs.filter { $0.inputConnections == nil }

        for sensorSpec in sensorSpecs {
            cExternalInputs += sensorSpec.cInputs
            cInternalIOputs += sensorSpec.cOutputs
            cBiases += sensorSpec.cOutputs
            cWeights += sensorSpec.cOutputs * sensorSpec.cInputs
        }

        let hiddenSpecs = layerSpecs.filter {
            $0.inputConnections != nil && $0.outputConnection != nil
        }

        for layerSpec in hiddenSpecs {
            cInternalIOputs += layerSpec.cOutputs
            cBiases += layerSpec.cOutputs
            cWeights += layerSpec.cOutputs * layerSpec.cInputs
        }

        let outputSpec = layerSpecs.last!

        precondition(
            outputSpec.inputConnections != nil &&
            outputSpec.inputConnections!.isEmpty == false &&
            outputSpec.outputConnection == nil,
            "No proper spec found for an output filter"
        )

        cExternalOutputs = outputSpec.cOutputs
        cBiases += outputSpec.cOutputs
        cWeights += outputSpec.cOutputs * outputSpec.cInputs

        let cIOData = cExternalInputs + cExternalOutputs + cInternalIOputs
        let cStatics = cBiases + cWeights

        return KnetCounts(
            layerSpecs: layerSpecs,
            cExternalInputs: cExternalInputs,
            cExternalOutputs: cExternalOutputs,
            cInternalIOputs: cInternalIOputs, cIOData: cIOData,
            cBiases: cBiases, cWeights: cWeights, cStatics: cStatics
        )
    }

}
