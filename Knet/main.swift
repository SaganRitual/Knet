// We are a way for the cosmos to know itself. -- C. Sagan

import Accelerate
import Foundation

let net = Knet(json: netStructure)

let pEverything = net.pEverything

net.pEverything.assign(repeating: -42)
net.inputBuffer.assign(repeating: 1)
net.biasesBuffer.assign(repeating: 0)
net.weightsBuffer.assign(repeating: 1)

net.activate()

print(net.outputBuffer!.map { $0 })
