// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation

print("Hello, World!")

let mBiases = UnsafeMutableBufferPointer<Float>.allocate(capacity: 2)
let mInputs = UnsafeMutableBufferPointer<Float>.allocate(capacity: 2)
let mOutputs = UnsafeMutableBufferPointer<Float>.allocate(capacity: 2)
let mWeights = UnsafeMutableBufferPointer<Float>.allocate(capacity: 4)

mBiases.initialize(repeating: 0)
mInputs.initialize(repeating: 1)
mOutputs.initialize(repeating: 1)
mWeights.initialize(repeating: 1)

let pBiases = UnsafeBufferPointer(mBiases)
let pInputs = UnsafeBufferPointer(mInputs)
let pOutputs = UnsafeBufferPointer(mOutputs)
let pWeights = UnsafeBufferPointer(mWeights)

let aggregator = KFCSpec(
    activation: .identity,
    cInputs: 49, cOutputs: 49,
    layerLevel: .hidden, name: "aggregator", order: 2,
    inputConnections: ["blackpool", "redpool"], outputConnection: "decider"
)

let motorOutputs = KFCSpec(
    activation: .identity,
    cInputs: 49, cOutputs: 7,
    layerLevel: .bottom, name: "aggregator", order: 3,
    inputConnections: ["aggregator"], outputConnection: nil
)

let blackpool = KPLSpec(
    activation: .identity, inputWidth: 7, inputHeight: 7,
    cOutputs: 49, kernelSide: 3, layerLevel: .top, name: "blackpool", order: 0,
    poolingFunction: .average,
    inputConnections: nil, outputConnection: "aggregator"
)

let redpool = KPLSpec(
    activation: .identity, inputWidth: 7, inputHeight: 7,
    cOutputs: 49, kernelSide: 3, layerLevel: .top, name: "redpool", order: 1,
    poolingFunction: .average,
    inputConnections: nil, outputConnection: "aggregator"
)

let netspec = KnetSpec(
    fullyConnectedLayers: [
        aggregator, motorOutputs
    ],
    poolingLayers: [
        blackpool, redpool
    ]
)

let encoder = JSONEncoder()
encoder.outputFormatting = .prettyPrinted
let string = try encoder.encode(netspec)
print(String(data: string, encoding: .utf8)!)
