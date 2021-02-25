// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation

/*
 {
   "inputHeight" : \(innerGridSize.height),
   "layerLevel" : "top",
   "order" : 1,
   "kernelSide" : \(kernelSide),
   "inputWidth" : \(innerGridSize.width),
   "poolingFunction" : "average",
   "cOutputs" : \(innerGridSize.area()),
   "outputConnection" : "red-decider",
   "name" : "redpool",
   "activation" : "tanh"
 }

 {
   "activation" : "tanh",
   "layerLevel" : "hidden",
   "order" : 2,
   "outputConnection" : "good-measure",
   "cOutputs" : \(innerGridSize.area()),
   "inputConnections" : [
     "blackpool"
   ],
   "cInputs" : \(innerGridSize.area()),
   "name" : "black-decider"
 },
 {
   "activation" : "tanh",
   "layerLevel" : "hidden",
   "order" : 3,
   "outputConnection" : "good-measure",
   "cOutputs" : \(innerGridSize.area()),
   "inputConnections" : [
     "redpool"
   ],
   "cInputs" : \(innerGridSize.area()),
   "name" : "red-decider"
 },
 {
   "activation" : "tanh",
   "layerLevel" : "bottom",
   "order" : 4,
   "cOutputs" : \(innerGridSize.width),
   "inputConnections" : [
     "blackpool"
   ],
   "cInputs" : \(innerGridSize.area()),
   "name" : "good-measure"
 }

 */

let kernelSide = 3

struct IGS {
    let height: Int = 3
    let width: Int = 3
    func area() -> Int { height * width }
}

let innerGridSize = IGS()

let netStructure = """
{
  "poolingLayers" : [
    {
      "inputHeight" : \(innerGridSize.height),
      "layerLevel" : "top",
      "order" : 0,
      "kernelSide" : \(kernelSide),
      "inputWidth" : \(innerGridSize.width),
      "poolingFunction" : "average",
      "cOutputs" : \(innerGridSize.area()),
      "outputConnection" : "good-measure",
      "name" : "blackpool",
      "activation" : "tanh"
    }
  ],
  "fullyConnectedLayers" : [
  ]
}
"""
