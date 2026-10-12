"""Caller-owned collective transport for original V4.1 publisher sessions.

All ranks create and use the same session schedule. Workers and process groups
are provided by the application; this module never initializes or destroys them.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class V41DistributedOptions:
    group: object
    device: object
    torch_version: str
    checkpoint_identity: str
    max_collective_bytes: int
    bf16: bool = True
    fp8: bool = True
    fp4: bool = True

    def __post_init__(self):
        if type(self.torch_version) is not str or not self.torch_version:
            raise ValueError('explicit installed transport version required')
        if type(self.checkpoint_identity) is not str or not self.checkpoint_identity:
            raise ValueError('caller-pinned complete checkpoint identity required')
        if type(self.max_collective_bytes) is not int or self.max_collective_bytes<=0:
            raise ValueError('positive collective allocation admission required')
        if any(type(value) is not bool for value in (self.bf16,self.fp8,self.fp4)) or not (self.bf16 or self.fp8 or self.fp4):
            raise ValueError('at least one explicit distributed projection format required')


def make_projection_backend(options,*,checkpoint,payload,bf16_provider,fp8_library,fp4_library,
                            max_arithmetic_bytes,owned,arithmetic_device=None):
    """Register the complete admitted bank before freezing arithmetic identity."""
    from .bf16_tensor_parallel import BF16TensorParallelProjections
    from .packed_tensor_parallel import PackedFP8TensorParallelProjections,PackedMXFP4TensorParallelProjections
    from .projection_dispatch import ProjectionDispatch
    from .publisher_checkpoint.v41_adapter import _shard_rule
    from .v41_execution.fp8_projection import NativeFP8Projection
    if type(options) is not V41DistributedOptions:
        raise TypeError('explicit distributed publisher options required')
    execution_device=None
    if arithmetic_device is not None:
        import torch
        from .k3_execution.device_policy import admit_device
        execution_device=admit_device(torch,arithmetic_device)
        if execution_device!=admit_device(torch,options.device):
            raise ValueError('V4.1 collective and neural arithmetic devices must match')
    common=dict(group=options.group,device=options.device,torch_version=options.torch_version,
        checkpoint_identity=options.checkpoint_identity,max_bytes=options.max_collective_bytes)
    assignments={}
    if options.bf16:
        if bf16_provider is None:raise ValueError('distributed BF16 requires a streamed row budget')
        bank=BF16TensorParallelProjections(provider=bf16_provider,**common)
        owned.callback(bank.close)
        bank._agree(('v41-projection-inventory-v1',options.bf16,options.fp8,options.fp4,
            options.checkpoint_identity,options.max_collective_bytes,
            tuple(sorted(bf16_provider.sources.items()))))
        for source in sorted(bf16_provider.sources):
            bank.register_source(source);assignments[source]=bank
    elif bf16_provider is not None:
        assignments.update({source:bf16_provider for source in bf16_provider.sources})
    if options.fp8:
        provider=payload.fp8_provider
        if provider is None:raise ValueError('distributed FP8 requires the original packed provider')
        if arithmetic_device is None:
            kernel=NativeFP8Projection(library=fp8_library,max_bytes=max_arithmetic_bytes)
        else:
            from .v41_execution.device_projection import DevicePublisherPackedProjection
            kernel=DevicePublisherPackedProjection(device=arithmetic_device,torch_version=options.torch_version,
                max_bytes=max_arithmetic_bytes,encoding='fp8-e4m3-g32x32')
        owned.callback(kernel.close)
        bank=PackedFP8TensorParallelProjections(provider=provider,kernel=kernel,**common)
        owned.callback(bank.close)
        rules=tuple(rule for rule in checkpoint._select(None) if _shard_rule(rule)
            and rule.encoding=='fp8-e4m3-g32x32' and rule.sources[0] in provider.tensor_shapes
            # wo_a stores FP8 bytes but executes through the publisher's
            # decoded BF16 group-local projection boundary. Its existing BF16
            # bank assignment must not be replaced by an FP32 packed dot.
            and not rule.sources[0].endswith('.wo_a.weight'))
        bank._agree(('v41-packed-inventory-v1',options.bf16,options.fp8,options.fp4,
            options.checkpoint_identity,options.max_collective_bytes,
            tuple(sorted((rule.sources[0],rule.shape) for rule in rules))))
        for rule in sorted(rules,key=lambda item:item.sources[0]):
            source=rule.sources[0];bank.register_source(source);assignments[source]=bank
    if options.fp4:
        from .native_mxfp4 import NativeMXFP4
        from .publisher_checkpoint.v41_provider import V41FP4ProjectionProvider
        provider=V41FP4ProjectionProvider(checkpoint,max_projection_bytes=max_arithmetic_bytes)
        owned.callback(provider.close)
        if arithmetic_device is None:
            kernel=NativeMXFP4(library=fp4_library,max_bytes=max_arithmetic_bytes)
        else:
            from .v41_execution.device_projection import DevicePublisherPackedProjection
            kernel=DevicePublisherPackedProjection(device=arithmetic_device,torch_version=options.torch_version,
                max_bytes=max_arithmetic_bytes,encoding='v41-fp4-g32')
        owned.callback(kernel.close)
        bank=PackedMXFP4TensorParallelProjections(provider=provider,kernel=kernel,**common)
        owned.callback(bank.close)
        rules=tuple(rule for rule in checkpoint._select(None) if rule.encoding=='v41-fp4-g32')
        bank._agree(('v41-fp4-inventory-v1',options.bf16,options.fp8,options.fp4,
            options.checkpoint_identity,options.max_collective_bytes,
            tuple(sorted((rule.sources[0],rule.shape) for rule in rules))))
        for rule in sorted(rules,key=lambda item:item.sources[0]):
            source=rule.sources[0];bank.register_source(source);assignments[source]=bank
    dispatch=ProjectionDispatch(assignments)
    owned.callback(dispatch.close)
    if arithmetic_device is not None:
        dispatch.device=execution_device
    return dispatch


__all__=['V41DistributedOptions','make_projection_backend']
