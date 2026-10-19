"""Explicit local operator entry point for leased native OLMoE execution."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from .native_leased_olmoe_segments import NativeLeasedOlmoeSegmentEngine
from .native_olmoe_segments import NativeOlmoeEdge
from .olmoe import OlmoeConfig


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_directory", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--public-runtime-library", type=Path,
                        help="Explicit Segment v1/Edge v2 dispatcher library; default uses the existing native path")
    parser.add_argument("--checkpoint-identity",
                        help="Admitted checkpoint manifest SHA-256 identity for the public runtime path")
    parser.add_argument("--snapshot-mib", type=positive, default=1024)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="Raw text, encoded by the checkpoint's local tokenizer")
    source.add_argument("--token-ids", help="Nonempty JSON array of token IDs")
    source.add_argument("--oracle", type=Path, help="JSON with prompt_ids and full_ids")
    parser.add_argument("--max-tokens", type=positive, default=32)
    parser.add_argument("--context-tokens", type=positive, default=512)
    parser.add_argument("--cache-mib", type=positive, default=256)
    parser.add_argument("--resident-mib", type=positive, default=2048)
    parser.add_argument("--transfer-mib", type=positive, default=2048)
    parser.add_argument("--staging-mib", type=positive, default=256)
    return parser


def _ids(value, vocab):
    if type(value) is not list or not value:
        raise ValueError("token IDs must be a nonempty JSON array")
    if any(type(token) is not int or not 0 <= token < vocab for token in value):
        raise ValueError("token IDs must be integers in the checkpoint vocabulary")
    return value


def run(args):
    config = OlmoeConfig.read(args.model_directory)
    if args.context_tokens > config.max_position_embeddings:
        raise ValueError("context capacity exceeds the checkpoint limit")
    expected = None
    tokenizer = None
    if args.oracle is not None:
        raw = json.loads(args.oracle.read_text(encoding="utf-8"))
        if type(raw) is not dict:
            raise ValueError("oracle must be a JSON object")
        prompt = _ids(raw["prompt_ids"], config.vocab_size)
        full = _ids(raw["full_ids"], config.vocab_size)
        if full[:len(prompt)] != prompt or len(full) == len(prompt):
            raise ValueError("oracle must contain the exact prompt prefix and a continuation")
        expected = full[len(prompt):]
        count = len(expected)
    elif args.token_ids is not None:
        prompt = _ids(json.loads(args.token_ids), config.vocab_size)
        count = args.max_tokens
    else:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(args.model_directory / "tokenizer.json"))
        prompt = _ids(tokenizer.encode(args.prompt).ids, config.vocab_size)
        count = args.max_tokens
    if len(prompt) + count - 1 > args.context_tokens:
        raise ValueError("prompt and generation exceed the selected context capacity")
    library = args.library.resolve(strict=True)
    mib = 1024**2
    start = time.perf_counter()
    if getattr(args, "public_runtime_library", None) is not None:
        return _run_public(args, config, prompt, count, expected, tokenizer, library, start)
    with ExitStack() as stack:
        engine = stack.enter_context(NativeLeasedOlmoeSegmentEngine(
            args.model_directory, args.manifest, 0, config.num_hidden_layers, library,
            max_resident_bytes=args.resident_mib*mib,
            expert_cache_bytes=args.cache_mib*mib, context_tokens=args.context_tokens,
            max_bytes=args.transfer_mib*mib, max_staging_bytes=args.staging_mib*mib))
        edge = stack.enter_context(NativeOlmoeEdge(args.model_directory, library,
            max_resident_bytes=args.resident_mib*mib, max_bytes=args.transfer_mib*mib))
        session = stack.enter_context(engine.create_decoder_session(edge))
        loaded = time.perf_counter()
        generated, pending = [], prompt
        first_token_ms = None
        for _ in range(count):
            logits = session.forward(pending)
            token = int(np.argmax(logits[-1]))
            if first_token_ms is None:
                first_token_ms = (time.perf_counter()-loaded)*1000
            generated.append(token)
            pending = [token]
            if expected is None and token == config.eos_token_id:
                break
        result = dict(schema="openagent.native-olmoe-generation.v1",
            prompt_ids=prompt, generated_ids=generated,
            implementation=engine.numeric_class, status=engine.status(),
            load_ms=(loaded-start)*1000, first_token_ms=first_token_ms,
            total_ms=(time.perf_counter()-start)*1000,
            upstream_abi_compatible=False)
        if tokenizer is not None:
            result["text"] = tokenizer.decode(generated)
        if expected is not None:
            result.update(expected_ids=expected, matched=generated == expected)
        return result


def _run_public(args, config, prompt, count, expected, tokenizer, library, started):
    from .segment_edge_adapters import NativeOlmoePublicAdapters
    from .segment_edge_runtime import SegmentEdgeExecution, SegmentRange
    from .expert_runtime import SparseExpertManifest

    if not getattr(args, "checkpoint_identity", None):
        raise ValueError('--checkpoint-identity is required with --public-runtime-library')
    manifest = SparseExpertManifest.read(args.manifest)
    canonical_manifest = json.dumps(manifest.to_dict(), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')
    manifest_identity = hashlib.sha256(canonical_manifest).hexdigest()
    if args.checkpoint_identity != manifest_identity:
        raise ValueError('--checkpoint-identity must be the SHA-256 of the canonical parsed expert manifest')
    mib = 1024**2
    adapters = NativeOlmoePublicAdapters(args.public_runtime_library,
        kernel_library=library, model_directory=args.model_directory, manifest=manifest,
        expert_cache_bytes=args.cache_mib*mib, max_staging_bytes=args.staging_mib*mib,
        max_snapshot_bytes=args.snapshot_mib*mib,
        max_batch_rows=min(32, args.context_tokens)).register()
    execution = None
    primary = None
    try:
        execution = SegmentEdgeExecution(args.public_runtime_library,
            edge_adapter=adapters.edge_id.decode(), model_directory=str(args.model_directory),
            checkpoint_identity=args.checkpoint_identity,
            ranges=(SegmentRange(adapters.segment_id.decode(), str(args.model_directory),
                                 0, config.num_hidden_layers),),
            context_tokens=args.context_tokens, native_memory_bytes=args.resident_mib*mib,
            max_transfer_bytes=args.transfer_mib*mib, max_snapshot_bytes=args.snapshot_mib*mib)
        loaded = time.perf_counter()
        generated, pending = [], prompt
        first_token_ms = None
        for _ in range(count):
            for offset in range(0, len(pending), execution.max_prefill_tokens):
                logits = execution.forward(pending[offset:offset+execution.max_prefill_tokens])
            token = int(np.argmax(logits[-1]))
            if first_token_ms is None:
                first_token_ms = (time.perf_counter()-loaded)*1000
            generated.append(token)
            pending = [token]
            if expected is None and token == config.eos_token_id:
                break
        result = dict(schema='openagent.native-olmoe-public-generation.v1',
            prompt_ids=prompt, generated_ids=generated,
            implementation='openagent-native-olmoe-public-adapters',
            status=execution.status(), adapter_status=adapters.status(),
            load_ms=(loaded-started)*1000, first_token_ms=first_token_ms,
            total_ms=(time.perf_counter()-started)*1000,
            abi_contract={'segment': 1, 'edge': 2, 'certified': False})
        if tokenizer is not None:
            result['text'] = tokenizer.decode(generated)
        if expected is not None:
            result.update(expected_ids=expected, matched=generated == expected)
        return result
    except BaseException as error:
        primary = error
        raise
    finally:
        cleanup_errors = []
        if execution is not None:
            try:
                execution.close()
            except BaseException as cleanup:
                cleanup_errors.append(cleanup)
        try:
            adapters.retry_cleanup()
        except BaseException as cleanup:
            cleanup_errors.append(cleanup)
        if cleanup_errors:
            message = '; '.join(str(error) for error in cleanup_errors)
            if primary is not None:
                primary.add_note('public native runtime cleanup failed: '+message)
            else:
                first = cleanup_errors[0]
                for secondary in cleanup_errors[1:]:
                    first.add_note('additional public native runtime cleanup failure: '+str(secondary))
                raise first


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result.get("matched", True) else 1
    except KeyboardInterrupt:
        print("Native generation cancelled", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, KeyError, ImportError, MemoryError, ArithmeticError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
