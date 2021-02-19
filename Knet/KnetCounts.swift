// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation

struct KnetCounts: CustomDebugStringConvertible {
    var debugDescription: String {
        "cExtIn: \(cExternalInputs)"
        + " cExtOut: \(cExternalOutputs)"
        + " cIntIOP: \(cInternalIOPuts)"
        + " cIOData: \(cIOData)"
        + " cBiases: \(cBiases) cWeights: \(cWeights) cStatics: \(cStatics)"
    }

    internal init(
        cExternalInputs: Int, cExternalOutputs: Int,
        cInternalIOPuts: Int, cIOData: Int,
        cBiases: Int, cWeights: Int, cStatics: Int
    ) {
        self.cExternalInputs = cExternalInputs
        self.cExternalOutputs = cExternalOutputs
        self.cInternalIOPuts = cInternalIOPuts
        self.cIOData = cIOData
        self.cBiases = cBiases
        self.cWeights = cWeights
        self.cStatics = cStatics
    }

    init(_ net: Knet) {
        self.cExternalInputs = net.cExternalInputs
        self.cExternalOutputs = net.cExternalOutputs
        self.cInternalIOPuts = net.cInternalIOputs
        self.cIOData = net.cIOData
        self.cBiases = net.cBiases
        self.cWeights = net.cWeights
        self.cStatics = net.cStatics
    }

    let cExternalInputs: Int
    let cExternalOutputs: Int
    let cInternalIOPuts: Int
    let cIOData: Int

    let cBiases: Int
    let cWeights: Int
    let cStatics: Int
}
