// We are a way for the cosmos to know itself. -- C. Sagan

import Accelerate
import Foundation

let net = Knet(json: netStructure)

net.inputBuffer.assign(repeating: 1)
net.biasesBuffer.assign(repeating: 0)
net.weightsBuffer.assign(repeating: 1)

net.activate()

print(net.outputBuffer!.map { $0 })
