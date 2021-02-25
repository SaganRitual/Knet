// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation

let inputHeight = 7
let inputWidth = 7
let area = inputWidth * inputHeight
let kernelSide = 3

/*
 {
   "inputHeight" : 7,
   "order" : 7,
   "kernelSide" : 3,
   "inputWidth" : 7,
   "cOutputs" : 49,
   "inputConnections" : [
     "mannaEnergy",
     "aggregator"
   ],
   "name" : "pool",
   "activation" : "identity"
 }

 */

let netStructure = """
{
  "poolingLayers" : [
    {
      "inputHeight" : 7,
      "layerLevel" : "top",
      "order" : 0,
      "kernelSide" : 3,
      "inputWidth" : 7,
      "poolingFunction" : "average",
      "cOutputs" : 49,
      "outputConnection" : "aggregator",
      "name" : "blackpool",
      "activation" : "identity"
    },
    {
      "inputHeight" : 7,
      "layerLevel" : "top",
      "order" : 1,
      "kernelSide" : 3,
      "inputWidth" : 7,
      "poolingFunction" : "average",
      "cOutputs" : 49,
      "outputConnection" : "aggregator",
      "name" : "redpool",
      "activation" : "identity"
    }
  ],
  "fullyConnectedLayers" : [
    {
      "activation" : "identity",
      "layerLevel" : "hidden",
      "order" : 2,
      "outputConnection" : "decider",
      "cOutputs" : 49,
      "inputConnections" : [
        "blackpool",
        "redpool"
      ],
      "cInputs" : 49,
      "name" : "aggregator"
    },
    {
      "activation" : "identity",
      "layerLevel" : "bottom",
      "order" : 3,
      "cOutputs" : 7,
      "inputConnections" : [
        "aggregator"
      ],
      "cInputs" : 49,
      "name" : "aggregator"
    }
  ]
}
"""
