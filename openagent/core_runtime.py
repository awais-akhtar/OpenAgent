"""Resource-aware core runtime planning for chat, voice, research, and local models."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
import platform
from typing import Any


CPU_MODEL_STACK = {
    "chat": "phi4-mini",
    "agent": "phi4-mini",
    "research": "phi4-mini",
    "code": "qwen2.5-coder:3b",
    "speed": "gemma3:2b",
    "extraction": "llama3.2:3b",
    "low_code": "deepseek-coder:1.3b",
    "high_research": "qwen2.5:7b",
    "voice": "faster-whisper:base.en",
}


@dataclass(frozen=True)
class SmoothRuntimeProfile:
    tier: str
    ram_gb: float
    cpu_count: int
    gpu_available: bool
    quantization_bits: int
    effective_bits: float
    quantization_method: str
    quantization_mode: str
    memory_budget_gb: float
    context_tokens: int
    output_tokens: int
    worker_count: int
    recommended_chat_model: str
    recommended_code_model: str
    recommended_research_model: str
    recommended_voice_engine: str
    recommended_agent_model: str
    recommended_speed_model: str
    recommended_low_resource_code_model: str
    recommended_extraction_model: str
    recommended_high_context_research_model: str
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "ram_gb": self.ram_gb,
            "cpu_count": self.cpu_count,
            "gpu_available": self.gpu_available,
            "quantization_bits": self.quantization_bits,
            "effective_bits": self.effective_bits,
            "quantization_method": self.quantization_method,
            "quantization_mode": self.quantization_mode,
            "memory_budget_gb": self.memory_budget_gb,
            "context_tokens": self.context_tokens,
            "output_tokens": self.output_tokens,
            "worker_count": self.worker_count,
            "recommended_chat_model": self.recommended_chat_model,
            "recommended_code_model": self.recommended_code_model,
            "recommended_research_model": self.recommended_research_model,
            "recommended_voice_engine": self.recommended_voice_engine,
            "recommended_agent_model": self.recommended_agent_model,
            "recommended_speed_model": self.recommended_speed_model,
            "recommended_low_resource_code_model": self.recommended_low_resource_code_model,
            "recommended_extraction_model": self.recommended_extraction_model,
            "recommended_high_context_research_model": self.recommended_high_context_research_model,
            "notes": list(self.notes),
            "privacy_boundary": "All planning is based on local hardware signals and local configuration only.",
        }


def detect_total_memory_gb() -> float:
    override = os.environ.get("OPENAGENT_TOTAL_RAM_GB", "").strip()
    if override:
        try:
            return round(max(0.5, float(override)), 2)
        except ValueError:
            pass
    system = platform.system()
    if system == "Windows":
        return windows_total_memory_gb()
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return round((pages * page_size) / (1024**3), 2)
        except (ValueError, OSError, AttributeError):
            pass
    return 8.0


def windows_total_memory_gb() -> float:
    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
        return round(status.ullTotalPhys / (1024**3), 2)
    return 8.0


def smooth_runtime_profile(*, task: str = "chat", hardware: dict[str, Any] | None = None) -> SmoothRuntimeProfile:
    hardware = hardware or {}
    ram_gb = float(hardware.get("ram_gb") or hardware.get("memory_gb") or detect_total_memory_gb())
    cpu_count = int(hardware.get("cpu_count") or os.cpu_count() or 1)
    gpu = hardware.get("gpu") if isinstance(hardware.get("gpu"), dict) else {}
    gpu_available = bool(gpu.get("available"))
    quantization_mode = str(
        hardware.get("quantization_mode")
        or os.environ.get("OPENAGENT_QUANTIZATION_MODE", "auto")
    ).strip().lower() or "auto"
    if gpu_available and ram_gb >= 16:
        tier = "balanced-gpu"
        bits = 4
        effective_bits = 4.0
        ctx = 6144
        out = 768
        workers = min(4, max(1, cpu_count // 2))
        chat_model = CPU_MODEL_STACK["chat"]
        code_model = CPU_MODEL_STACK["code"]
        research_model = CPU_MODEL_STACK["high_research"]
        notes = (
            "GPU detected: keep compact CPU-safe defaults but allow Qwen2.5 7B for deeper research.",
            "Use Phi-4 Mini for general reasoning and Qwen2.5 Coder 3B for code.",
        )
    elif ram_gb <= 4:
        tier = "survival-cpu"
        bits = 1
        effective_bits = 1.58
        ctx = 1024
        out = 192
        workers = 1
        chat_model = "gemma3:1b"
        code_model = CPU_MODEL_STACK["low_code"]
        research_model = "gemma3:1b"
        notes = (
            "Survival CPU mode: prefer Gemma speed profiles and DeepSeek Coder for the lowest coding footprint.",
            "Use this for old or memory-constrained machines where responsiveness matters more than quality.",
        )
    elif ram_gb <= 6:
        tier = "micro-cpu"
        bits = 2
        effective_bits = 2.0
        ctx = 1536
        out = 256
        workers = 1
        chat_model = CPU_MODEL_STACK["speed"]
        code_model = CPU_MODEL_STACK["low_code"]
        research_model = "llama3.2:1b"
        notes = (
            "Micro CPU mode: Gemma 3 2B is the speed-first chat model and DeepSeek Coder 1.3B handles basic code.",
            "Prefer short chats, focused code questions, and research summaries over long context dumps.",
        )
    elif ram_gb <= 8:
        tier = "ultra-light-8gb"
        bits = 3
        effective_bits = 3.0
        ctx = 2048
        out = 384
        workers = 1
        chat_model = CPU_MODEL_STACK["speed"]
        code_model = CPU_MODEL_STACK["code"]
        research_model = CPU_MODEL_STACK["extraction"]
        notes = (
            "8 GB friendly mode: Gemma 3 2B keeps chat instant, Qwen2.5 Coder 3B handles code, and Llama 3.2 3B handles RAG extraction.",
            "Use Phi-4 Mini when quality matters more than speed and local benchmark evidence is acceptable.",
        )
    elif ram_gb <= 12:
        tier = "light-cpu"
        bits = 4
        effective_bits = 4.0
        ctx = 3072
        out = 512
        workers = min(2, max(1, cpu_count // 2))
        chat_model = CPU_MODEL_STACK["chat"]
        code_model = CPU_MODEL_STACK["code"]
        research_model = CPU_MODEL_STACK["research"]
        notes = (
            "CPU light mode: Phi-4 Mini becomes the default reasoning model and Qwen2.5 Coder 3B handles coding.",
            "Research uses staged evidence windows to keep the CPU responsive.",
        )
    else:
        tier = "standard-cpu"
        bits = 4
        effective_bits = 4.0
        ctx = 4096
        out = 768
        workers = min(3, max(1, cpu_count // 2))
        chat_model = CPU_MODEL_STACK["chat"]
        code_model = CPU_MODEL_STACK["code"]
        research_model = CPU_MODEL_STACK["high_research"] if ram_gb >= 16 else CPU_MODEL_STACK["research"]
        notes = (
            "Standard CPU mode: Phi-4 Mini is the default chat/agent model and Qwen2.5 Coder 3B is the code model.",
            "Use Qwen2.5 7B for deeper research only when RAM is sufficient and benchmarks pass.",
        )

    if quantization_mode in {"maximum-compression", "max-compression", "max"}:
        bits = max(1, bits - 1)
        effective_bits = min(effective_bits, float(bits))
        ctx = min(ctx, 2048)
        out = min(out, 384)
        workers = 1
        notes = (*notes, "Maximum compression mode reduces bit depth and runtime budget for smoother CPU use.")
    elif quantization_mode in {"below-2bit", "sub2", "sub-2", "1.58"}:
        bits = 1
        effective_bits = 1.58
        ctx = min(ctx, 1536)
        out = min(out, 256)
        workers = 1
        notes = (*notes, "Sub-2-bit mode records 1.58-bit quantization planning for extreme CPU memory pressure.")
    elif quantization_mode in {"quality", "high-quality"}:
        bits = max(bits, 4)
        effective_bits = max(effective_bits, 4.0)
        notes = (*notes, "Quality mode keeps at least 4-bit quantization planning.")

    if task == "code" and ram_gb <= 8:
        out = min(out, 320)
    if task == "research" and ram_gb <= 8:
        ctx = min(ctx, 1792)
        out = min(out, 360)

    return SmoothRuntimeProfile(
        tier=tier,
        ram_gb=round(ram_gb, 2),
        cpu_count=cpu_count,
        gpu_available=gpu_available,
        quantization_bits=bits,
        effective_bits=effective_bits,
        quantization_method=quantization_method(bits, effective_bits),
        quantization_mode=quantization_mode,
        memory_budget_gb=round(max(0.5, ram_gb * 0.42), 2),
        context_tokens=ctx,
        output_tokens=out,
        worker_count=workers,
        recommended_chat_model=os.environ.get("OPENAGENT_SMOOTH_CHAT_MODEL", chat_model),
        recommended_code_model=os.environ.get("OPENAGENT_SMOOTH_CODE_MODEL", code_model),
        recommended_research_model=os.environ.get("OPENAGENT_SMOOTH_RESEARCH_MODEL", research_model),
        recommended_voice_engine=os.environ.get("OPENAGENT_SMOOTH_VOICE_ENGINE", CPU_MODEL_STACK["voice"]),
        recommended_agent_model=os.environ.get("OPENAGENT_SMOOTH_AGENT_MODEL", CPU_MODEL_STACK["agent"]),
        recommended_speed_model=os.environ.get("OPENAGENT_SMOOTH_SPEED_MODEL", CPU_MODEL_STACK["speed"]),
        recommended_low_resource_code_model=os.environ.get("OPENAGENT_SMOOTH_LOW_CODE_MODEL", CPU_MODEL_STACK["low_code"]),
        recommended_extraction_model=os.environ.get("OPENAGENT_SMOOTH_EXTRACTION_MODEL", CPU_MODEL_STACK["extraction"]),
        recommended_high_context_research_model=os.environ.get("OPENAGENT_SMOOTH_HIGH_RESEARCH_MODEL", CPU_MODEL_STACK["high_research"]),
        notes=notes,
    )


def model_request_options(
    *,
    task: str = "chat",
    hardware: dict[str, Any] | None = None,
    prompt_tokens: int | None = None,
    output_tokens: int | None = None,
    low_latency: bool = False,
) -> dict[str, Any]:
    profile = smooth_runtime_profile(task=task, hardware=hardware)
    interactive_caps = {
        "chat": 192,
        "agent": 224,
        "research": 240,
        "code": 384,
        "voice": 96,
    }
    cap_env = os.environ.get(f"OPENAGENT_{task.upper()}_MAX_OUTPUT_TOKENS", "").strip()
    if not cap_env:
        cap_env = os.environ.get("OPENAGENT_MAX_OUTPUT_TOKENS", "").strip()
    interactive_cap = interactive_caps.get(task, 224)
    try:
        configured_cap = int(cap_env) if cap_env else None
    except ValueError:
        configured_cap = None
    if output_tokens is None:
        desired_output = interactive_cap
        output_cap = configured_cap or interactive_cap
        profile_cap = profile.output_tokens
    else:
        desired_output = int(output_tokens)
        output_cap = configured_cap or max(profile.output_tokens, desired_output)
        # Long-form document jobs may deliberately trade latency for a complete
        # result. Keep the request bounded by half of the hardware-safe context.
        profile_cap = max(profile.output_tokens, profile.context_tokens // 2)
    output_tokens = max(32, min(profile_cap, output_cap, int(desired_output)))
    if prompt_tokens is None:
        context_tokens = profile.context_tokens
    else:
        required = max(512, int(prompt_tokens) + output_tokens + 96)
        context_tokens = min(profile.context_tokens, ((required + 255) // 256) * 256)
    default_threads = min(12, max(profile.worker_count, max(1, profile.cpu_count // 2)))
    try:
        thread_count = int(os.environ.get("OPENAGENT_CPU_THREADS", str(default_threads)))
    except ValueError:
        thread_count = default_threads
    thread_count = max(1, min(profile.cpu_count, thread_count))
    batch_size = 128 if profile.ram_gb <= 8 else 256
    if low_latency and prompt_tokens is not None and prompt_tokens <= 512:
        batch_size = min(batch_size, 128)
    return {
        "num_ctx": context_tokens,
        "num_predict": output_tokens,
        "num_thread": thread_count,
        "num_batch": batch_size,
        "num_keep": min(256, max(32, (prompt_tokens or 128) // 3)),
        "temperature": 0.2 if task in {"research", "code"} else 0.35,
        "top_p": 0.9,
        "num_gpu": 0 if not profile.gpu_available else 999,
        "use_mmap": True,
        "use_mlock": False,
    }


def quantization_method(bits: int, effective_bits: float) -> str:
    if effective_bits < 2:
        return "q1_58_K_S"
    if bits <= 2:
        return "q2_K"
    if bits == 3:
        return "q3_K_S"
    return f"q{bits}_K_M"


def adaptive_quantization_plan(
    *,
    task: str = "chat",
    target_ram_gb: float | None = None,
    quantization_mode: str = "auto",
    parameter_count_b: float | None = None,
) -> dict[str, Any]:
    hardware: dict[str, Any] = {"quantization_mode": quantization_mode}
    if target_ram_gb is not None:
        hardware["ram_gb"] = target_ram_gb
    profile = smooth_runtime_profile(task=task, hardware=hardware)
    params_b = parameter_count_b or default_parameter_count_b(profile, task)
    estimated_weights_gb = (params_b * 1_000_000_000 * profile.effective_bits / 8) / (1024**3)
    runtime_overhead_gb = min(1.8, max(0.35, profile.context_tokens / 4096))
    estimated_total_gb = round(estimated_weights_gb + runtime_overhead_gb, 2)
    return {
        "task": task,
        "profile": profile.to_dict(),
        "parameter_count_b": params_b,
        "estimated_weights_gb": round(estimated_weights_gb, 2),
        "estimated_total_runtime_gb": estimated_total_gb,
        "fits_memory_budget": estimated_total_gb <= profile.memory_budget_gb,
        "cpu_strategy": {
            "threads": profile.worker_count,
            "mmap": True,
            "gpu_layers": 0 if not profile.gpu_available else "auto",
            "chunked_prefill": True,
            "staged_prompts": task in {"research", "code"},
        },
        "privacy_boundary": "Quantization planning uses local hardware values only. It does not upload prompts or model files.",
    }


def default_parameter_count_b(profile: SmoothRuntimeProfile, task: str) -> float:
    if profile.tier == "survival-cpu":
        return 1.0 if task != "code" else 1.3
    if profile.tier == "micro-cpu":
        return 2.0 if task == "chat" else 1.3
    if profile.tier == "ultra-light-8gb":
        return 3.0 if task in {"code", "research"} else 2.0
    if task == "research" and profile.recommended_research_model == CPU_MODEL_STACK["high_research"]:
        return 7.0
    if task == "chat":
        return 3.8
    return 3.0


def model_profiles() -> list[dict[str, Any]]:
    return [
        {
            "id": "chat-survival-cpu",
            "task": "chat",
            "model": "gemma3:1b-it-q1_58",
            "ram_gb": 3,
            "quantization": "Q1.58",
            "notes": "Emergency CPU profile for very small machines.",
        },
        {
            "id": "chat-speed-cpu",
            "task": "chat",
            "model": CPU_MODEL_STACK["speed"],
            "ram_gb": 6,
            "quantization": "Q4_K_M",
            "notes": "Speed-first real-time chat profile for CPU-only machines.",
        },
        {
            "id": "chat-agent-quality",
            "task": "chat",
            "model": CPU_MODEL_STACK["chat"],
            "ram_gb": 8,
            "quantization": "Q4_K_M",
            "notes": "Best default quality profile for chat, agents, and general reasoning.",
        },
        {
            "id": "code-low-resource",
            "task": "code",
            "model": CPU_MODEL_STACK["low_code"],
            "ram_gb": 4,
            "quantization": "Q4_K_M",
            "notes": "Lowest-resource coding profile for simple scripts and older CPUs.",
        },
        {
            "id": "code-default-cpu",
            "task": "code",
            "model": CPU_MODEL_STACK["code"],
            "ram_gb": 8,
            "quantization": "Q4_K_M",
            "notes": "Default coding model for multi-file logic, debugging, SQL, and Python.",
        },
        {
            "id": "research-extraction",
            "task": "research",
            "model": CPU_MODEL_STACK["extraction"],
            "ram_gb": 8,
            "quantization": "Q4_K_M",
            "notes": "Fast RAG extraction and summarization profile.",
        },
        {
            "id": "research-agent-quality",
            "task": "research",
            "model": CPU_MODEL_STACK["research"],
            "ram_gb": 8,
            "quantization": "Q4_K_M",
            "notes": "Reasoning-first research and agent planning profile.",
        },
        {
            "id": "research-high-context",
            "task": "research",
            "model": CPU_MODEL_STACK["high_research"],
            "ram_gb": 16,
            "quantization": "Q4_K_M",
            "notes": "Deeper structured research profile for 16 GB+ systems.",
        },
        {
            "id": "voice-local",
            "task": "voice",
            "model": "faster-whisper:base.en",
            "ram_gb": 2,
            "quantization": "int8",
            "notes": "High-accuracy local STT on CPU after model cache.",
        },
    ]
