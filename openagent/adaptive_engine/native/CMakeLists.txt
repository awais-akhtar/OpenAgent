cmake_minimum_required(VERSION 3.21)
project(OpenAgentSegmentRuntime LANGUAGES C)

include(GNUInstallDirs)
option(OPENAGENT_BUILD_DECODERS "Build resident CPU decoder libraries" ON)
option(OPENAGENT_BUILD_KERNELS "Build CPU arithmetic and packed kernels" ON)
option(OPENAGENT_BUILD_URING "Build the Linux liburing reader" OFF)
option(OPENAGENT_BUILD_PLATFORM_PROBES "Build explicitly selected platform probes" OFF)
option(OPENAGENT_REGISTER_PLATFORM_TESTS "Register opt-in next-phase platform checks" OFF)
set(OPENAGENT_PLATFORM_BACKEND "linux" CACHE STRING "linux, cuda, hip or metal")
set_property(CACHE OPENAGENT_PLATFORM_BACKEND PROPERTY STRINGS linux cuda hip metal)
set(OPENAGENT_PLATFORM_DEVICE "0" CACHE STRING "Explicit accelerator device index")
set(OPENAGENT_DECODER_MANIFEST "" CACHE FILEPATH "Pinned complete decoder comparison input")
set(OPENAGENT_DECODER_FAMILY "" CACHE STRING "Decoder identity in the pinned comparison input")

function(openagent_native_library target)
    add_library(${target} SHARED ${ARGN})
    target_compile_features(${target} PRIVATE c_std_11)
    set_target_properties(${target} PROPERTIES C_EXTENSIONS OFF
        POSITION_INDEPENDENT_CODE ON)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
        if(NOT WIN32)
            target_link_libraries(${target} PRIVATE m)
        endif()
    endif()
    install(TARGETS ${target}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endfunction()

set(ENGINE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
if(OPENAGENT_BUILD_DECODERS)
    foreach(family IN ITEMS mixtral_decoder glm_dsa glm5_next inkling kimi
            qwen35 qwen4 qwen35_vision deepseek_v4)
        openagent_native_library(openagent_${family}
            "${ENGINE_SOURCE_DIR}/native_${family}.c")
    endforeach()
    openagent_native_library(openagent_kimi_k3_decoder
        "${ENGINE_SOURCE_DIR}/native_kimi_k3_decoder.c"
        "${ENGINE_SOURCE_DIR}/native_kimi_k3_attnres.c")
endif()

if(OPENAGENT_BUILD_KERNELS)
    foreach(kernel IN ITEMS tensor_ops media_ops mxfp4 mixtral_delta qwen4_hyper
            kimi_k3_kda kimi_k3_moe kimi_k3_dense kimi_k3_blocks kimi_k3_attnres
            deepseek_v41_ced deepseek_v41_csa2 deepseek_v41_engram
            deepseek_v41_index deepseek_v41_mhc deepseek_v41_quant)
        openagent_native_library(openagent_${kernel}
            "${ENGINE_SOURCE_DIR}/native_${kernel}.c")
    endforeach()
    openagent_native_library(openagent_k3_attention
        "${ENGINE_SOURCE_DIR}/k3_execution/attention.c")
    openagent_native_library(openagent_v41_kernels
        "${ENGINE_SOURCE_DIR}/v41_execution/kernels.c")
    openagent_native_library(openagent_v41_fp8_projection
        "${ENGINE_SOURCE_DIR}/v41_execution/fp8_projection.c")
endif()

if(OPENAGENT_BUILD_URING)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR "The liburing reader requires Linux")
    endif()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LIBURING REQUIRED IMPORTED_TARGET liburing)
    openagent_native_library(openagent_uring_reader uring_reader.c)
    target_link_libraries(openagent_uring_reader PRIVATE PkgConfig::LIBURING)
endif()

add_library(openagent_segment_runtime SHARED
    segment_edge_compat.c
    segment_family_models.c
    qwen4_public_model.c
    qwen4_public_session.c
    qwen4_public_adapters.c
    original_experts.c
    segment_executor.c
    segment_bridge.c
    segment_math.c
)
target_compile_features(openagent_segment_runtime PRIVATE c_std_11)
target_include_directories(openagent_segment_runtime PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/openagent>
)
set_target_properties(openagent_segment_runtime PROPERTIES
    C_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    C_VISIBILITY_PRESET hidden
)
if(MSVC)
    # The public adapter registry and resource ledgers use C11 atomic flags
    # and reference counters. MSVC keeps these optional atomics experimental
    # unless explicitly enabled, even with /std:c11.
    target_compile_options(openagent_segment_runtime PRIVATE /W4 /experimental:c11atomics)
else()
    target_compile_options(openagent_segment_runtime PRIVATE -Wall -Wextra -Wpedantic)
    if(NOT WIN32)
        target_link_libraries(openagent_segment_runtime PRIVATE m)
    endif()
endif()

install(TARGETS openagent_segment_runtime
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
)
install(FILES
    segment_edge_compat.h
    segment_family_models.h
    qwen4_composition_abi.h
    qwen4_public_model.h
    qwen4_public_session.h
    original_experts.h
    native_exports.h
    segment_executor.h
    segment_bridge.h
    segment_math.h
    DESTINATION include/openagent
)

# These targets and CTest entries are opt-in. Configuring CPU libraries does
# not discover an accelerator or execute any Python import or platform probe.
if(OPENAGENT_BUILD_PLATFORM_PROBES OR OPENAGENT_REGISTER_PLATFORM_TESTS)
    if(NOT OPENAGENT_PLATFORM_BACKEND MATCHES "^(linux|cuda|hip|metal)$")
        message(FATAL_ERROR "Explicit supported platform backend required")
    endif()
endif()
if(OPENAGENT_BUILD_PLATFORM_PROBES)
    set(PLATFORM_SOURCE_DIR "${ENGINE_SOURCE_DIR}/platform_certification")
    if(OPENAGENT_PLATFORM_BACKEND STREQUAL "cuda")
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            message(FATAL_ERROR "Set CMAKE_CUDA_ARCHITECTURES for the requested hardware")
        endif()
        enable_language(CUDA)
        add_executable(openagent_platform_probe "${PLATFORM_SOURCE_DIR}/probe.cu")
        set_target_properties(openagent_platform_probe PROPERTIES CUDA_STANDARD 14
            CUDA_STANDARD_REQUIRED ON)
    elseif(OPENAGENT_PLATFORM_BACKEND STREQUAL "hip")
        if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
            message(FATAL_ERROR "Set CMAKE_HIP_ARCHITECTURES for the requested hardware")
        endif()
        enable_language(HIP)
        set_source_files_properties("${PLATFORM_SOURCE_DIR}/probe.cu" PROPERTIES LANGUAGE HIP)
        add_executable(openagent_platform_probe "${PLATFORM_SOURCE_DIR}/probe.cu")
        target_compile_definitions(openagent_platform_probe PRIVATE OA_HIP)
        set_target_properties(openagent_platform_probe PROPERTIES HIP_STANDARD 14
            HIP_STANDARD_REQUIRED ON)
    elseif(OPENAGENT_PLATFORM_BACKEND STREQUAL "metal")
        if(NOT APPLE)
            message(FATAL_ERROR "Metal probe targets require an Apple host")
        endif()
        find_program(OPENAGENT_XCRUN xcrun REQUIRED)
        set(METAL_PROBE "${CMAKE_CURRENT_BINARY_DIR}/openagent_platform_probe")
        add_custom_command(OUTPUT "${METAL_PROBE}"
            COMMAND "${OPENAGENT_XCRUN}" swiftc "${PLATFORM_SOURCE_DIR}/probe.swift"
                -framework Metal -o "${METAL_PROBE}"
            DEPENDS "${PLATFORM_SOURCE_DIR}/probe.swift" VERBATIM)
        add_custom_target(openagent_platform_probe ALL DEPENDS "${METAL_PROBE}")
        install(PROGRAMS "${METAL_PROBE}" DESTINATION ${CMAKE_INSTALL_BINDIR})
    elseif(NOT OPENAGENT_BUILD_URING)
        message(FATAL_ERROR "Linux platform probe selection requires OPENAGENT_BUILD_URING")
    endif()
    if(TARGET openagent_platform_probe AND NOT OPENAGENT_PLATFORM_BACKEND STREQUAL "metal")
        install(TARGETS openagent_platform_probe RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
endif()

if(OPENAGENT_REGISTER_PLATFORM_TESTS)
    include(CTest)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    get_filename_component(OPENAGENT_REPOSITORY_ROOT "${ENGINE_SOURCE_DIR}/../.." ABSOLUTE)
    add_test(NAME openagent_backend_engine
        COMMAND "${Python3_EXECUTABLE}" -m openagent.adaptive_engine.platform_certification
            --backend "${OPENAGENT_PLATFORM_BACKEND}" --scope engine
            --device "${OPENAGENT_PLATFORM_DEVICE}"
            --report "${CMAKE_CURRENT_BINARY_DIR}/evidence/engine.json")
    set_tests_properties(openagent_backend_engine PROPERTIES
        WORKING_DIRECTORY "${OPENAGENT_REPOSITORY_ROOT}" TIMEOUT 3600
        LABELS "engine;${OPENAGENT_PLATFORM_BACKEND};operator-opt-in")
    if(OPENAGENT_DECODER_MANIFEST OR OPENAGENT_DECODER_FAMILY)
        if(NOT OPENAGENT_DECODER_MANIFEST OR NOT OPENAGENT_DECODER_FAMILY)
            message(FATAL_ERROR "Both complete decoder manifest and family are required")
        endif()
        add_test(NAME openagent_whole_decoder
            COMMAND "${Python3_EXECUTABLE}" -m openagent.adaptive_engine.platform_certification
                --backend "${OPENAGENT_PLATFORM_BACKEND}" --scope full-model
                --decoder "${OPENAGENT_DECODER_FAMILY}"
                --decoder-manifest "${OPENAGENT_DECODER_MANIFEST}"
                --device "${OPENAGENT_PLATFORM_DEVICE}"
                --report "${CMAKE_CURRENT_BINARY_DIR}/evidence/decoder.json")
        set_tests_properties(openagent_whole_decoder PROPERTIES
            WORKING_DIRECTORY "${OPENAGENT_REPOSITORY_ROOT}" TIMEOUT 7200
            LABELS "whole-model;${OPENAGENT_PLATFORM_BACKEND};operator-opt-in")
    endif()
endif()
