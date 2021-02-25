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
    "cOutputs" : \(area),
    "name" : "aggregator",
    "order" : 6,
    "inputConnections" : [
      "lock"
    ]
  }
]
"""
