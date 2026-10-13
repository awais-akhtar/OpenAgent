"""Caller-owned collective transport for original V4.1 publisher sessions.

All ranks create and use the same session schedule. Workers and process groups
are provided by the application; this module never initializes or destroys them.
"""
from dataclasses import dataclass


def _validate_mtp_projection_inventory(checkpoint, provider, *, bank_format):
    """Fail closed if a V4.1 bank provider omits an eligible MTP matrix.

    Rank agreement proves that ranks selected the same inventory, not that
    the inventory is complete. MTP executes through the borrowed arithmetic
    owner's projection dispatch, so every eligible source for an enabled
    format must be registered there rather than silently falling back to a
    rank-local reader.
    """
    from .publisher_checkpoint.v41_adapter import _shard_rule

    rules = tuple(rule for rule in checkpoint._select(None)
                  if _shard_rule(rule) and rule.target.startswith('v41.mtp.'))
    source_universe = {rule.sources[0] for rule in rules}
    if bank_format == 'bf16':
        actual_inventory = getattr(provider, 'sources', None)
        if not hasattr(actual_inventory, 'keys'):
            raise ValueError('V4.1 MTP BF16 provider source inventory required')
        expected = set()
        for rule in rules:
            source = rule.sources[0]
            is_bf16 = (len(rule.sources) == 1 and
                       checkpoint._metadata.tensors[source].dtype == 'BF16')
            is_grouped_wo_a = (len(rule.sources) == 2 and
                               source.endswith('.wo_a.weight') and
                               rule.encoding == 'fp8-e4m3-g32x32')
            if is_bf16 or is_grouped_wo_a:
                expected.add(source)
    elif bank_format == 'fp8':
        actual_inventory = getattr(provider, 'tensor_shapes', None)
        if not hasattr(actual_inventory, 'keys'):
            raise ValueError('V4.1 MTP FP8 provider source inventory required')
        expected = {rule.sources[0] for rule in rules
                    if rule.encoding == 'fp8-e4m3-g32x32' and
                    not rule.sources[0].endswith('.wo_a.weight')}
    elif bank_format == 'fp4':
        actual_inventory = getattr(provider, 'tensor_shapes', None)
        if not hasattr(actual_inventory, 'keys'):
            raise ValueError('V4.1 MTP FP4 provider source inventory required')
        expected = {rule.sources[0] for rule in rules
                    if rule.encoding == 'v41-fp4-g32'}
    else:
        raise ValueError('supported V4.1 MTP bank format required')
    actual = set(actual_inventory).intersection(source_universe)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError('incomplete V4.1 MTP '+bank_format+
                         ' projection inventory; missing='+repr(missing)+
                         ', unexpected='+repr(unexpected))
    return tuple(sorted(expected))


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
    import torch.distributed as distributed
    from .bf16_tensor_parallel import BF16TensorParallelProjections
    from .packed_tensor_parallel import PackedFP8TensorParallelProjections,PackedMXFP4TensorParallelProjections
    from .projection_dispatch import ProjectionDispatch
    from .publisher_checkpoint.v41_adapter import _shard_rule
    from .v41_execution.fp8_projection import NativeFP8Projection
    if type(options) is not V41DistributedOptions:
        raise TypeError('explicit distributed publisher options required')
    if not distributed.is_available() or not distributed.is_initialized():
        raise RuntimeError('caller-initialized distributed group required before bank admission')
    world_size=distributed.get_world_size(options.group)
    if type(world_size) is not int or world_size<=0:
        raise ValueError('nonempty distributed projection group required')
    policy_error=None
    local_policy=None
    execution_device=None
    try:
        report=getattr(checkpoint,'report',None)
        fp8_provider=getattr(payload,'fp8_provider',None)
        if (getattr(report,'family',None)!='deepseek_v41' or
                getattr(payload,'family',None)!='deepseek_v41' or
                type(getattr(report,'config_sha256',None)) is not str or
                report.config_sha256!=getattr(payload,'config_sha256',None)):
            raise ValueError('matching V4.1 checkpoint and payload identities required')
        if arithmetic_device is not None:
            import torch
            from .k3_execution.device_policy import admit_device
            execution_device=admit_device(torch,arithmetic_device)
            collective_device=admit_device(torch,options.device)
            if execution_device!=collective_device:
                raise ValueError('V4.1 collective and neural arithmetic devices must match')
        projection_inventory=tuple(sorted(
            (rule.encoding,tuple(rule.sources),tuple(rule.shape))
            for rule in checkpoint._select(None) if _shard_rule(rule)))
        mtp_inventory = []
        if options.bf16:
            mtp_inventory.append(('bf16', _validate_mtp_projection_inventory(
                checkpoint, bf16_provider, bank_format='bf16')))
        if options.fp8:
            mtp_inventory.append(('fp8', _validate_mtp_projection_inventory(
                checkpoint, fp8_provider, bank_format='fp8')))
        local_policy=(
            options.bf16,options.fp8,options.fp4,options.torch_version,
            options.checkpoint_identity,str(options.device),options.max_collective_bytes,
            max_arithmetic_bytes,
            None if execution_device is None else str(execution_device),
            report.family,report.config_sha256,payload.family,payload.config_sha256,
            projection_inventory,
            None if bf16_provider is None else tuple(sorted(bf16_provider.sources.items())),
            None if fp8_provider is None else tuple(sorted(fp8_provider.tensor_shapes.items())),
            tuple(mtp_inventory),
        )
    except Exception as failure:
        policy_error=(str(failure) or type(failure).__name__)[:2048]
    policies=[None]*world_size
    # Agree on every branch selector and source inventory before constructing
    # format-specific banks. Without this common first collective, divergent
    # per-rank flags or provider inventories can send ranks into different
    # collective sequences (or let one rank reject while peers block).
    distributed.all_gather_object(policies,(local_policy,policy_error),group=options.group)
    policy_errors=[f'rank {rank}: {record[1]}' for rank,record in enumerate(policies)
                   if type(record) is tuple and len(record)==2 and record[1] is not None]
    if policy_errors:
        raise ValueError('V4.1 distributed bank preflight failed: '+'; '.join(policy_errors))
    if any(type(record) is not tuple or len(record)!=2 for record in policies):
        raise ValueError('malformed V4.1 distributed bank preflight record')
    if any(record[0]!=policies[0][0] for record in policies):
        raise ValueError('V4.1 distributed bank policy or source inventory differs across ranks')
    def agreed_stage(name, construct):
        """Make rank-local setup failures unanimous before the next collective.

        Kernel/provider construction may load native libraries or reserve
        memory and can fail on only one rank. Peers must not enter a bank's
        first registration collective while that rank unwinds locally.
        """
        value=None
        failure=None
        try:
            value=construct()
        except BaseException as error:
            failure=error
        records=[None]*world_size
        try:
            distributed.all_gather_object(records,
                None if failure is None else (type(failure).__name__,str(failure)[:2048]),
                group=options.group)
        except BaseException:
            values=value if type(value) is tuple else (value,)
            for resource in reversed(values):
                close=getattr(resource,'close',None)
                if callable(close):
                    try:close()
                    except BaseException:pass
            raise
        failed=[f'rank {rank}: {record[0]}: {record[1]}'
                for rank,record in enumerate(records) if record is not None]
        if failed:
            cleanup_errors=[]
            values=value if type(value) is tuple else (value,)
            for resource in reversed(values):
                close=getattr(resource,'close',None)
                if callable(close):
                    try:close()
                    except BaseException as cleanup:cleanup_errors.append(str(cleanup))
            if failure is not None and cleanup_errors:
                failure.add_note('local distributed setup cleanup failed: '+'; '.join(cleanup_errors))
            detail=f'V4.1 distributed {name} setup failed: '+'; '.join(failed)
            if failure is not None:
                raise RuntimeError(detail) from failure
            raise RuntimeError(detail)
        return value
    common=dict(group=options.group,device=options.device,torch_version=options.torch_version,
        checkpoint_identity=options.checkpoint_identity,max_bytes=options.max_collective_bytes)
    assignments={}
    if options.bf16:
        if bf16_provider is None:raise ValueError('distributed BF16 requires a streamed row budget')
        bank=agreed_stage('BF16 bank',lambda:BF16TensorParallelProjections(provider=bf16_provider,**common))
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
        def make_fp8_bank():
            if arithmetic_device is None:
                kernel=NativeFP8Projection(library=fp8_library,max_bytes=max_arithmetic_bytes)
            else:
                from .v41_execution.device_projection import DevicePublisherPackedProjection
                kernel=DevicePublisherPackedProjection(device=arithmetic_device,torch_version=options.torch_version,
                    max_bytes=max_arithmetic_bytes,encoding='fp8-e4m3-g32x32')
            try:return kernel,PackedFP8TensorParallelProjections(provider=provider,kernel=kernel,**common)
            except BaseException:
                kernel.close()
                raise
        kernel,bank=agreed_stage('FP8 kernel and bank',make_fp8_bank)
        owned.callback(kernel.close)
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
        def make_fp4_provider():
            provider=V41FP4ProjectionProvider(checkpoint,max_projection_bytes=max_arithmetic_bytes)
            try:
                _validate_mtp_projection_inventory(checkpoint,provider,bank_format='fp4')
            except BaseException:
                provider.close()
                raise
            return provider
        provider=agreed_stage('FP4 provider',make_fp4_provider)
        owned.callback(provider.close)
        def make_fp4_bank():
            if arithmetic_device is None:
                kernel=NativeMXFP4(library=fp4_library,max_bytes=max_arithmetic_bytes)
            else:
                from .v41_execution.device_projection import DevicePublisherPackedProjection
                kernel=DevicePublisherPackedProjection(device=arithmetic_device,torch_version=options.torch_version,
                    max_bytes=max_arithmetic_bytes,encoding='v41-fp4-g32')
            try:return kernel,PackedMXFP4TensorParallelProjections(provider=provider,kernel=kernel,**common)
            except BaseException:
                kernel.close()
                raise
        kernel,bank=agreed_stage('FP4 kernel and bank',make_fp4_bank)
        owned.callback(kernel.close)
        owned.callback(bank.close)
        rules=tuple(rule for rule in checkpoint._select(None) if rule.encoding=='v41-fp4-g32')
        bank._agree(('v41-fp4-inventory-v1',options.bf16,options.fp8,options.fp4,
            options.checkpoint_identity,options.max_collective_bytes,
            tuple(sorted((rule.sources[0],rule.shape) for rule in rules))))
        for rule in sorted(rules,key=lambda item:item.sources[0]):
            source=rule.sources[0];bank.register_source(source);assignments[source]=bank
    dispatch=agreed_stage('projection dispatch',lambda:ProjectionDispatch(assignments))
    owned.callback(dispatch.close)
    if arithmetic_device is not None:
        dispatch.device=execution_device
    return dispatch


__all__=['V41DistributedOptions','make_projection_backend']
