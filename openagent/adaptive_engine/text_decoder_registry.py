"""Text decoder identities without importing loading or execution owners."""
from types import MappingProxyType


_DECODERS = MappingProxyType({
    'mixtral': ('native_mixtral_decoder', 'NativeMixtralDecoder'),
    'glm_moe_dsa': ('native_glm_dsa', 'NativeGlmDsaDecoder'),
    'qwen3_5_moe_text': ('native_qwen35', 'NativeQwen35TextDecoder'),
    'qwen4_exp_text': ('native_qwen4', 'NativeQwen4TextDecoder'),
    'glm5_next_text': ('native_glm5_next', 'NativeGlm5NextTextDecoder'),
    'deepseek_v4': ('native_deepseek_v4', 'NativeDeepseekV4Decoder'),
    'inkling_text': ('native_inkling', 'NativeInklingDecoder'),
    'kimi_linear': ('native_kimi', 'NativeKimiLinearDecoder'),
})


def decoder_owner(model_type):
    """Return the detached source owner tuple for one admitted identity."""
    if type(model_type) is not str or not model_type:
        raise ValueError('exact native text identity required')
    try:
        module, name = _DECODERS[model_type]
    except KeyError as error:
        raise ValueError('unsupported exact native text identity: '+model_type) from error
    return module, name


__all__ = ['_DECODERS', 'decoder_owner']
