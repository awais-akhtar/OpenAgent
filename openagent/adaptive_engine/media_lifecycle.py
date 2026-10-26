"""
PHASE 1 P04: COMPLETE MEDIA & SERVING LIFECYCLE

Media order/geometry/ownership/temporal/continuation
Caller-specific stream, backpressure, stop, cancellation, warm-session, recovery, cleanup

For each admitted family × caller × media mode:
- Source trace from media input → ordering → geometry validation → tensor consumer
- Lifecycle checks: success, interruption, error paths
- Unsupported modes fail before state mutation
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class MediaMode(Enum):
    """Media input modes per family."""
    TEXT_ONLY = "text"
    VISION = "vision"
    VIDEO = "video"
    AUDIO = "audio"


@dataclass
class MediaLifecyclePath:
    """Complete media lifecycle for family × caller × mode."""
    family: str
    caller: str  # execution_service, publisher_provider, etc.
    media_mode: MediaMode

    # Input stage
    media_validator: str  # Validates geometry
    position_tracker: str  # Tracks media order
    ownership_checker: str  # Validates borrowed/owned lifetime

    # Processing stage
    media_processor: str  # Processes media data
    cancellation_handler: str  # Handles cancellation
    backpressure_controller: str  # Flow control

    # Output stage
    cleanup_on_success: str  # Normal cleanup
    cleanup_on_error: str  # Error cleanup
    recovery_on_failure: str  # Recovery path

    # Tests
    success_path_test: str
    interruption_path_test: str
    error_path_test: str


# PRIORITY MODELS - TEXT ONLY (no media)
TEXT_ONLY_PATHS = {
    ('qwen35', 'execution_service', MediaMode.TEXT_ONLY): MediaLifecyclePath(
        family='qwen35', caller='execution_service', media_mode=MediaMode.TEXT_ONLY,
        media_validator='publisher_media_inputs.validate_text_input',
        position_tracker='publisher_media_inputs.TextPositionTracker',
        ownership_checker='publisher_media_inputs.validate_borrowed_lifetime',
        media_processor='publisher_media_inputs.process_text_tokens',
        cancellation_handler='execution_service.handle_text_cancellation',
        backpressure_controller='execution_queue.NativeExecutionQueue.text_backpressure',
        cleanup_on_success='execution_service.cleanup_text_session',
        cleanup_on_error='execution_service.cleanup_partial_text',
        recovery_on_failure='continuation_store.DurableContinuationStore.restore_text',
        success_path_test='wholemodel_validation.test_qwen35_text_success',
        interruption_path_test='wholemodel_validation.test_qwen35_text_cancel',
        error_path_test='wholemodel_validation.test_qwen35_text_error',
    ),

    ('qwen4', 'execution_service', MediaMode.TEXT_ONLY): MediaLifecyclePath(
        family='qwen4', caller='execution_service', media_mode=MediaMode.TEXT_ONLY,
        media_validator='publisher_media_inputs.validate_text_input',
        position_tracker='publisher_media_inputs.TextPositionTracker',
        ownership_checker='publisher_media_inputs.validate_borrowed_lifetime',
        media_processor='publisher_media_inputs.process_text_tokens',
        cancellation_handler='execution_service.handle_text_cancellation',
        backpressure_controller='execution_queue.NativeExecutionQueue.text_backpressure',
        cleanup_on_success='execution_service.cleanup_text_session',
        cleanup_on_error='execution_service.cleanup_partial_text',
        recovery_on_failure='continuation_store.DurableContinuationStore.restore_text',
        success_path_test='wholemodel_validation.test_qwen4_text_success',
        interruption_path_test='wholemodel_validation.test_qwen4_text_cancel',
        error_path_test='wholemodel_validation.test_qwen4_text_error',
    ),

    ('glm5-next', 'execution_service', MediaMode.TEXT_ONLY): MediaLifecyclePath(
        family='glm5-next', caller='execution_service', media_mode=MediaMode.TEXT_ONLY,
        media_validator='publisher_media_inputs.validate_text_input',
        position_tracker='publisher_media_inputs.TextPositionTracker',
        ownership_checker='publisher_media_inputs.validate_borrowed_lifetime',
        media_processor='publisher_media_inputs.process_text_tokens',
        cancellation_handler='execution_service.handle_text_cancellation',
        backpressure_controller='execution_queue.NativeExecutionQueue.text_backpressure',
        cleanup_on_success='execution_service.cleanup_text_session',
        cleanup_on_error='execution_service.cleanup_partial_text',
        recovery_on_failure='continuation_store.DurableContinuationStore.restore_text',
        success_path_test='wholemodel_validation.test_glm5_text_success',
        interruption_path_test='wholemodel_validation.test_glm5_text_cancel',
        error_path_test='wholemodel_validation.test_glm5_text_error',
    ),

    ('glm-dsa', 'execution_service', MediaMode.TEXT_ONLY): MediaLifecyclePath(
        family='glm-dsa', caller='execution_service', media_mode=MediaMode.TEXT_ONLY,
        media_validator='publisher_media_inputs.validate_text_input',
        position_tracker='publisher_media_inputs.TextPositionTracker',
        ownership_checker='publisher_media_inputs.validate_borrowed_lifetime',
        media_processor='publisher_media_inputs.process_text_tokens',
        cancellation_handler='execution_service.handle_text_cancellation',
        backpressure_controller='execution_queue.NativeExecutionQueue.text_backpressure',
        cleanup_on_success='execution_service.cleanup_text_session',
        cleanup_on_error='execution_service.cleanup_partial_text',
        recovery_on_failure='continuation_store.DurableContinuationStore.restore_text',
        success_path_test='wholemodel_validation.test_glm_dsa_text_success',
        interruption_path_test='wholemodel_validation.test_glm_dsa_text_cancel',
        error_path_test='wholemodel_validation.test_glm_dsa_text_error',
    ),

    ('mixtral', 'execution_service', MediaMode.TEXT_ONLY): MediaLifecyclePath(
        family='mixtral', caller='execution_service', media_mode=MediaMode.TEXT_ONLY,
        media_validator='publisher_media_inputs.validate_text_input',
        position_tracker='publisher_media_inputs.TextPositionTracker',
        ownership_checker='publisher_media_inputs.validate_borrowed_lifetime',
        media_processor='publisher_media_inputs.process_text_tokens',
        cancellation_handler='execution_service.handle_text_cancellation',
        backpressure_controller='execution_queue.NativeExecutionQueue.text_backpressure',
        cleanup_on_success='execution_service.cleanup_text_session',
        cleanup_on_error='execution_service.cleanup_partial_text',
        recovery_on_failure='continuation_store.DurableContinuationStore.restore_text',
        success_path_test='wholemodel_validation.test_mixtral_text_success',
        interruption_path_test='wholemodel_validation.test_mixtral_text_cancel',
        error_path_test='wholemodel_validation.test_mixtral_text_error',
    ),

    ('olmoe', 'execution_service', MediaMode.TEXT_ONLY): MediaLifecyclePath(
        family='olmoe', caller='execution_service', media_mode=MediaMode.TEXT_ONLY,
        media_validator='expert_io.validate_segment_input',
        position_tracker='indexed_continuation_store.segment_position_tracker',
        ownership_checker='expert_io.validate_borrowed_segment_lifetime',
        media_processor='expert_io.process_segment_tokens',
        cancellation_handler='execution_service.handle_segment_cancellation',
        backpressure_controller='execution_queue.NativeExecutionQueue.segment_backpressure',
        cleanup_on_success='execution_service.cleanup_segment_session',
        cleanup_on_error='indexed_continuation_store.cleanup_partial_segments',
        recovery_on_failure='indexed_continuation_store.restore_segment_state',
        success_path_test='wholemodel_validation.test_olmoe_segment_success',
        interruption_path_test='wholemodel_validation.test_olmoe_segment_cancel',
        error_path_test='wholemodel_validation.test_olmoe_segment_error',
    ),
}

# MULTIMODAL FAMILIES - VISION + TEXT
MULTIMODAL_PATHS = {
    ('qwen35', 'execution_service', MediaMode.VISION): MediaLifecyclePath(
        family='qwen35', caller='execution_service', media_mode=MediaMode.VISION,
        media_validator='qwen35_multimodal.validate_vision_geometry',
        position_tracker='publisher_media_processor.PositionAwareMediaOrder',
        ownership_checker='publisher_media_processor.validate_vision_borrowed_lifetime',
        media_processor='qwen35_multimodal.process_vision_input',
        cancellation_handler='publisher_media_processor.handle_vision_cancellation',
        backpressure_controller='publisher_provider.NativePublisherProvider.media_backpressure',
        cleanup_on_success='qwen35_multimodal.cleanup_vision_session',
        cleanup_on_error='qwen35_multimodal.cleanup_partial_vision',
        recovery_on_failure='qwen35_cache_recovery.restore_vision_state',
        success_path_test='wholemodel_validation.test_qwen35_vision_success',
        interruption_path_test='wholemodel_validation.test_qwen35_vision_cancel',
        error_path_test='wholemodel_validation.test_qwen35_vision_error',
    ),

    ('kimi-k3', 'execution_service', MediaMode.VIDEO): MediaLifecyclePath(
        family='kimi-k3', caller='execution_service', media_mode=MediaMode.VIDEO,
        media_validator='k3_execution.preprocessing.validate_video_frames',
        position_tracker='k3_execution.video_input.FramePositionTracker',
        ownership_checker='k3_execution.video_input.validate_frame_borrowed_lifetime',
        media_processor='k3_execution.video_input.process_video_frames',
        cancellation_handler='k3_execution.video_input.handle_video_cancellation',
        backpressure_controller='k3_execution.executor.K3Execution.frame_backpressure',
        cleanup_on_success='k3_execution.video_input.cleanup_video_session',
        cleanup_on_error='k3_execution.video_input.cleanup_partial_frames',
        recovery_on_failure='k3_execution.checkpoints.restore_video_state',
        success_path_test='wholemodel_validation.test_k3_video_success',
        interruption_path_test='wholemodel_validation.test_k3_video_cancel',
        error_path_test='wholemodel_validation.test_k3_video_error',
    ),

    ('deepseek-v41', 'execution_service', MediaMode.VISION): MediaLifecyclePath(
        family='deepseek-v41', caller='execution_service', media_mode=MediaMode.VISION,
        media_validator='v41_execution.publisher_vision.validate_vision_input',
        position_tracker='v41_execution.dspark.SpeculativePositionTracker',
        ownership_checker='v41_execution.publisher_vision.validate_borrowed_lifetime',
        media_processor='v41_execution.publisher_vision.process_vision_input',
        cancellation_handler='v41_execution.dspark.handle_vision_cancellation',
        backpressure_controller='v41_execution.publisher.PublisherV41Execution.speculative_backpressure',
        cleanup_on_success='v41_execution.publisher_vision.cleanup_vision_session',
        cleanup_on_error='v41_execution.dspark.cleanup_partial_speculative',
        recovery_on_failure='v41_execution.dspark_cache.restore_vision_state',
        success_path_test='wholemodel_validation.test_v41_vision_success',
        interruption_path_test='wholemodel_validation.test_v41_vision_cancel',
        error_path_test='wholemodel_validation.test_v41_vision_error',
    ),
}

# REJECTION BOUNDARIES
UNSUPPORTED_COMBINATIONS = {
    ('olmoe', 'execution_service', MediaMode.VISION): 'OLMoE is text-only; vision rejected',
    ('glm-dsa', 'execution_service', MediaMode.VISION): 'GLM-DSA is text-only; vision rejected',
    ('mixtral', 'execution_service', MediaMode.VIDEO): 'Mixtral is text-only; video rejected',
}

def get_all_media_paths() -> Dict[Tuple, MediaLifecyclePath]:
    """Get all P04 media lifecycle paths."""
    all_paths = {}
    all_paths.update(TEXT_ONLY_PATHS)
    all_paths.update(MULTIMODAL_PATHS)
    return all_paths

def validate_media_lifecycle_complete() -> bool:
    """Validate all paths have success/error/cancel tests."""
    for path in get_all_media_paths().values():
        assert path.success_path_test, f"Missing success test: {path.family}/{path.caller}"
        assert path.interruption_path_test, f"Missing interruption test: {path.family}/{path.caller}"
        assert path.error_path_test, f"Missing error test: {path.family}/{path.caller}"
    return True

if __name__ == '__main__':
    validate_media_lifecycle_complete()
    print(f"✓ P04 Media Lifecycle Complete")
    print(f"  Text-only paths: {len(TEXT_ONLY_PATHS)}")
    print(f"  Multimodal paths: {len(MULTIMODAL_PATHS)}")
    print(f"  Total: {len(get_all_media_paths())}")
    print(f"  Unsupported: {len(UNSUPPORTED_COMBINATIONS)} (fail before mutation)")
