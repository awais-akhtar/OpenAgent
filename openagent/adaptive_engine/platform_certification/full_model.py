"""Local command orchestration using the existing whole-model comparison runner."""
import copy
import json
import uuid


def certify_full_model(args, report, unavailable):
    report['stage'] = 'decoder_prerequisites'
    manifest_path = getattr(args, 'decoder_manifest', None)
    if manifest_path is None:
        raise unavailable('full-model requires --decoder-manifest with candidate and oracle commands')
    try:
        from ..wholemodel_validation.runner import run, digest_file
    except ImportError as exc:
        raise unavailable('whole-model comparison dependency unavailable: ' + str(exc)) from exc
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not isinstance(manifest, dict):
        raise ValueError('decoder manifest must be an object')
    if manifest.get('decoder') != args.decoder or manifest.get('backend') != args.backend:
        raise ValueError('decoder manifest family/backend mismatch')
    commands = manifest.get('commands')
    if not isinstance(commands, dict) or set(commands) != {'candidate', 'oracle'}:
        raise ValueError('explicit candidate and oracle commands required')
    for argv in commands.values():
        if (not isinstance(argv, list) or not argv or
                any(not isinstance(x, str) or not x or '\0' in x for x in argv)):
            raise ValueError('commands must be nonempty explicit argv lists')
    if commands['candidate'] == commands['oracle']:
        raise ValueError('candidate and oracle commands must be independently supplied')
    comparison = copy.deepcopy(manifest['comparison'])
    if comparison.get('schema') != 'openagent.wholemodel-validation-input.v1':
        raise ValueError('existing wholemodel comparison input schema required')
    tokens = comparison.get('tokens')
    schedules = comparison.get('chunk_schedules')
    if (not isinstance(tokens, list) or len(tokens) < 3 or
            not isinstance(schedules, list) or not any(
                isinstance(s, list) and len(s) > 1 and any(type(n) is int and n > 1 for n in s)
                for s in schedules)):
        raise ValueError('at least three prompt tokens and a mixed incremental chunk schedule required')
    output = args.report.resolve().parent / ('decoder-artifacts-' + uuid.uuid4().hex)
    output.mkdir(parents=True, exist_ok=False)
    report['full_model_evidence'] = dict(executed=False, certified=False,
        decoder=args.decoder, backend=args.backend, artifact_directory=str(output),
        manifest_sha256=digest_file(manifest_path),
        trust='operator-supplied command evidence; not signed hardware attestation')
    for role in ('candidate', 'oracle'):
        spec = comparison[role]
        if not isinstance(spec, dict) or not spec.get('artifacts'):
            raise ValueError('complete pinned runner/checkpoint artifacts required for ' + role)
        spec['factory'] = 'openagent.adaptive_engine.platform_certification.command_adapter:create_command_adapter'
        spec['options'] = dict(command=commands[role], backend=args.backend, decoder=args.decoder,
            device=args.device, device_name=args.device_name,
            min_memory_bytes=args.min_memory_bytes, timeout=args.timeout,
            audit_directory=str(output / role))
    report['stage'] = 'full_model_comparison'
    result = run(comparison, base_directory=manifest_path.parent, output_directory=output)
    report['full_model_evidence']['comparison_report'] = dict(
        path=str(output / 'report.json'), sha256=digest_file(output / 'report.json'))
    report['full_model_evidence']['observation_artifact'] = result.get('observation_artifact')
    if result.get('status') != 'passed' or result.get('errors'):
        errors = result.get('errors', [])
        if errors and all(error.get('type') == 'DecoderUnavailable' for error in errors):
            raise unavailable('; '.join(error['message'] for error in errors))
        raise RuntimeError('full-model comparison failed; inspect captured comparison report')
    runtimes = result.get('adapter_runtime', {})
    models = result.get('models', {})
    if (set(runtimes) != {'candidate', 'oracle'} or
            models['candidate']['implementation'] == models['oracle']['implementation']):
        raise ValueError('independent implementations and both runtime evidence inventories required')
    for role in ('candidate', 'oracle'):
        runtime = runtimes[role]
        if not runtime.get('forward_evidence') or not runtime.get('commands'):
            raise ValueError('actual decoder command/placement evidence missing for ' + role)
        if any(record.get('returncode') != 0 or record.get('error') for record in runtime['commands']):
            raise ValueError('unsuccessful decoder command recorded')
    artifact = result.get('observation_artifact')
    if not artifact or not result.get('observations') or not result.get('generated_tokens'):
        raise ValueError('full/incremental/cache/token observation artifacts absent')
    if digest_file(artifact['path']) != artifact['sha256']:
        raise ValueError('captured observation artifact hash changed')
    checks = result.get('checks')
    if not checks or any(check.get('passed') is not True for check in checks):
        raise ValueError('missing or unsuccessful mandatory full-model comparisons')
    report['full_model_evidence'].update(executed=True, certified=True,
        coverage='exact pinned text-model case only', comparisons=len(checks),
        generated_tokens=result['generated_tokens'])
    row = next(row for row in report['decoder_backend_matrix'] if row['decoder'] == args.decoder)
    row.update(status='passed', executed=True, certified=True, gaps=[],
               coverage='exact pinned text-model case only')
    report['full_model_certified'] = True

