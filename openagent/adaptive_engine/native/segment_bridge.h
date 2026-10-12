#ifndef OA_SEGMENT_BRIDGE_H
#define OA_SEGMENT_BRIDGE_H
#include <stdint.h>
#include "native_exports.h"
#ifdef __cplusplus
extern "C" {
#endif
#define OA_SEGMENT_ABI_VERSION 1u
#define OA_SEGMENT_MAX_HANDLES 65536u
typedef uint64_t oa_handle;
typedef int32_t oa_status;
/* Serialize every API call per loaded library, across threads and adapters.
 * Handles are local to that library instance, never persistent or pointers.
 * The registrant retains callback/context until successful release.
 * Same-thread synchronous callback reentry may use other non-busy handles;
 * caller serialization must exclude concurrent threads, not forbid reentry.
 * Storage grows in fixed-address pages up to OA_SEGMENT_MAX_HANDLES live
 * registrations. Empty slots are reused, but handle IDs never recycle.
 * Pages are reclaimed when the last handle is successfully released.
 * Allocation/capacity/ID exhaustion returns OA_LIMIT without registration. */
enum { OA_OK=0, OA_INVALID=1, OA_VERSION=2, OA_HANDLE=3, OA_BUSY=4,
       OA_LIMIT=5, OA_CALLBACK=6, OA_CANCELLED=7 };
enum { OA_ENGINE=1, OA_SESSION=2, OA_EDGE=3 };
enum { OA_CLOSE=1, OA_FORWARD=2, OA_EMBED=3, OA_HEAD=4,
       OA_RESET=5, OA_SNAPSHOT=6, OA_RESTORE=7 };
/* Borrowed buffers and context: valid only during a synchronous call.
 * Callback must never unwind across C; written <= capacity, even on failure.
 * Nonzero-length pointers must address valid caller-owned storage. The ABI
 * checks lengths, not pointer validity; it is not an untrusted-code sandbox. */
typedef oa_status (OA_CALL *oa_callback)(void *context, uint32_t operation,
    uint64_t rows, const void *input, uint64_t input_bytes,
    void *output, uint64_t capacity, uint64_t *written);
OA_API uint32_t OA_CALL oa_segment_version(void);
OA_API oa_status OA_CALL oa_segment_register(uint32_t version, uint32_t kind,
    oa_handle parent, uint64_t max_bytes, oa_callback callback,
    void *context, oa_handle *handle);
/* Tensor calls require positive rows, input length and output capacity.
 * RESET has no rows/buffers; SNAPSHOT has no rows/input; RESTORE requires
 * input, with no rows/output. Invalid envelopes never invoke the callback. */
OA_API oa_status OA_CALL oa_segment_call(oa_handle handle, uint32_t operation,
    uint64_t rows, const void *input, uint64_t input_bytes,
    void *output, uint64_t capacity, uint64_t *written);
OA_API oa_status OA_CALL oa_segment_release(oa_handle handle);
#ifdef __cplusplus
}
#endif
#endif
