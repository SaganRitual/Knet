// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation

let area = 1

let netStructure = """
[
  {
    "activation" : "identity",
    "cInputs" : \(area),
    "cOutputs" : \(area),
    "name" : "lock",
    "order" : 0,
    "outputConnection" : "aggregator",
  },
  {
    "activation" : "identity",
    "cInputs" : \(area),
    "cOutputs" : \(area),
    "name" : "arkon",
    "order" : 1,
    "outputConnection" : "aggregator",
  },
  {
    "activation" : "identity",
    "cInputs" : \(area),
    "cOutputs" : \(area),
    "name" : "arkonEnergy",
    "order" : 2,
    "outputConnection" : "aggregator",
  },
  {
    "activation" : "identity",
    "cInputs" : \(area),
    "cOutputs" : \(area),
    "name" : "arkonFitness",
    "order" : 3,
    "outputConnection" : "aggregator",
  },
  {
    "activation" : "identity",
    "cInputs" : \(area),
    "cOutputs" : \(area),
    "name" : "manna",
    "order" : 4,
    "outputConnection" : "aggregator",
  },
  {
    "activation" : "identity",
    "cInputs" : \(area),
    "cOutputs" : \(area),
    "name" : "mannaEnergy",
    "order" : 5,
    "outputConnection" : "aggregator",
  },
  {
    "activation" : "identity",
    "cInputs" : \(area),
    "cOutputs" : \(area),
    "name" : "aggregator",
    "order" : 6,
    "inputConnections" : [
      "lock",
      "arkon",
      "arkonEnergy",
      "arkonFitness",
      "manna",
      "mannaEnergy",
    ]
  }
]
"""
