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

    fileprivate init(_ scratch: ScratchCounts) {
        self.layerSpecs = scratch.layerSpecs
        self.cExternalInputs = scratch.cExternalInputs
        self.cExternalOutputs = scratch.cExternalOutputs
        self.cInternalIOputs = scratch.cInternalIOputs
        self.cIOData = scratch.cIOData
        self.cBiases = scratch.cBiases
        self.cWeights = scratch.cWeights
        self.cStatics = scratch.cStatics
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
        let scratch = ScratchCounts()

        do {
            scratch.layerSpecs = try decoder.decode(
                [KFCSpec].self, from: netStructure.data(using: .utf8)!
            ).sorted(by: Knet.orderSort)
        } catch {
            fatalError(
                "\n\nCan't decode your crappy JSON, loser."
                + "\nAlso, here's a useless error message for you: "
                + "\n\(error.localizedDescription)\n\n"
            )
        }

        scratch.setupCounts()

        return KnetCounts(scratch)
    }
}

// Same as KnetCounts, but with vars so I can use them for
// counting. I can't stand to make the real KnetCounts writable
private class ScratchCounts {
    var layerSpecs: [KFCSpec]!

    var cExternalInputs: Int = 0
    var cExternalOutputs: Int = 0
    var cInternalIOputs: Int = 0
    var cIOData: Int = 0

    var cBiases: Int = 0
    var cWeights: Int = 0
    var cStatics: Int = 0

    func setupCounts() {
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
    }
}
