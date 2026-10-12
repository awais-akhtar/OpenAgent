import Metal
import Foundation
guard let device = MTLCreateSystemDefaultDevice() else {
    print("OPENAGENT_RUNTIME_UNAVAILABLE {\"reason\":\"Native Metal device unavailable\"}")
    exit(2)
}
guard let queue = device.makeCommandQueue() else { fatalError("Metal command queue unavailable") }
let library = try device.makeLibrary(source: """
#include <metal_stdlib>
using namespace metal;
kernel void certify(device int *output [[buffer(0)]]) { output[0] = 734; }
""", options: nil)
guard let function = library.makeFunction(name: "certify"),
      let buffer = device.makeBuffer(length: 4, options: .storageModeShared),
      let command = queue.makeCommandBuffer(),
      let encoder = command.makeComputeCommandEncoder() else { fatalError("allocation failed") }
let pipeline = try device.makeComputePipelineState(function: function)
encoder.setComputePipelineState(pipeline)
encoder.setBuffer(buffer, offset: 0, index: 0)
encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
encoder.endEncoding()
command.commit()
command.waitUntilCompleted()
guard command.status == .completed, command.error == nil,
      buffer.contents().load(as: Int32.self) == 734 else { fatalError("execution failed") }
print("OPENAGENT_NATIVE_EXECUTED")
