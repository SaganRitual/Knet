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

let lock = KFullyConnected(
    cInputs: 2, cOutputs: 2, activation: .init(function: .identity),
    pBiases: pBiases, pInputs: pInputs,
    pOutputs: pOutputs, pWeights: pWeights
)

let arkonPresent = KFullyConnected(
    cInputs: 2, cOutputs: 2, activation: .init(function: .identity),
    pBiases: pBiases, pInputs: pInputs,
    pOutputs: pOutputs, pWeights: pWeights
)

let arkonEnergy = KFullyConnected(
    cInputs: 2, cOutputs: 2, activation: .init(function: .identity),
    pBiases: pBiases, pInputs: pInputs,
    pOutputs: pOutputs, pWeights: pWeights
)

let arkonFitness = KFullyConnected(
    cInputs: 2, cOutputs: 2, activation: .init(function: .identity),
    pBiases: pBiases, pInputs: pInputs,
    pOutputs: pOutputs, pWeights: pWeights
)

let mannaPresent = KFullyConnected(
    cInputs: 2, cOutputs: 2, activation: .init(function: .identity),
    pBiases: pBiases, pInputs: pInputs,
    pOutputs: pOutputs, pWeights: pWeights
)

let mannaEnergy = KFullyConnected(
    cInputs: 2, cOutputs: 2, activation: .init(function: .identity),
    pBiases: pBiases, pInputs: pInputs,
    pOutputs: pOutputs, pWeights: pWeights
)

let aggregator = KFullyConnected(
    cInputs: 2, cOutputs: 2, activation: .init(function: .identity),
    pBiases: pBiases, pInputs: pInputs,
    pOutputs: pOutputs, pWeights: pWeights
)

fc.activate()

print(pInputs.map { $0 }, pOutputs.map { $0 })
