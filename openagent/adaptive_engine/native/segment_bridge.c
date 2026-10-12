#include "segment_bridge.h"
#include <stddef.h>
#include <string.h>
#include <stdlib.h>

/* Caller serializes ALL calls into a library instance, including registration.
 * IDs never recycle; stale IDs cannot acquire a new object in a reused slot. */
#define OA_PAGE_SLOTS 256u
typedef struct entry_page entry_page;
typedef struct {
    oa_handle id, parent;
    uint64_t limit;
    uint32_t kind, children, busy;
    oa_callback callback;
    void *context;
    entry_page *owner;
} entry;
struct entry_page {
    struct entry_page *next;
    uint32_t used;
    entry slots[OA_PAGE_SLOTS];
};
static entry_page *pages = NULL;
static uint32_t page_count = 0, live_count = 0;
static oa_handle next_id = 1;
static entry *lookup(oa_handle id) {
    entry_page *page;
    unsigned i;
    if (!id) return NULL;
    for (page=pages; page; page=page->next) if (page->used) {
        for (i=0; i<OA_PAGE_SLOTS; ++i)
            if (page->slots[i].id == id) return &page->slots[i];
    }
    return NULL;
}
static void reclaim_empty_registry(void) {
    entry_page *page;
    /* A callback's busy entry remains live until it returns, so no active
     * callback can retain a pointer when the last successful release gets here. */
    if (live_count) return;
    while (pages) {
        page=pages; pages=page->next; free(page);
    }
    page_count=0;
}
uint32_t OA_CALL oa_segment_version(void) { return OA_SEGMENT_ABI_VERSION; }
oa_status OA_CALL oa_segment_register(uint32_t version, uint32_t kind,
    oa_handle parent, uint64_t max_bytes, oa_callback callback,
    void *context, oa_handle *handle) {
    entry *p = NULL;
    entry *e = NULL;
    entry_page *page;
    unsigned i;
    if (!handle) return OA_INVALID;
    *handle = 0;
    if (version != OA_SEGMENT_ABI_VERSION) return OA_VERSION;
    if (!callback || !max_bytes || max_bytes > SIZE_MAX ||
        kind < OA_ENGINE || kind > OA_EDGE) return OA_INVALID;
    if (kind == OA_SESSION) {
        p = lookup(parent);
        if (!p || p->kind != OA_ENGINE) return OA_HANDLE;
        if (p->busy) return OA_BUSY;
    } else if (parent) return OA_INVALID;
    if (!next_id || live_count>=OA_SEGMENT_MAX_HANDLES ||
        (p && p->children==UINT32_MAX)) return OA_LIMIT;
    for (page=pages; page; page=page->next) if (page->used<OA_PAGE_SLOTS) {
        for (i=0; i<OA_PAGE_SLOTS; ++i) if (!page->slots[i].id) {
            e=&page->slots[i]; break;
        }
        if (e) break;
    }
    if (!e) {
        if (page_count>=OA_SEGMENT_MAX_HANDLES/OA_PAGE_SLOTS ||
            sizeof(entry_page)>SIZE_MAX/(size_t)(page_count+1)) return OA_LIMIT;
        page=(entry_page*)calloc(1,sizeof(*page));
        if (!page) return OA_LIMIT;
        /* Pages never move. Publish only after allocation succeeds; no further
         * failure point follows, and retained parent/callback pointers stay valid. */
        page->next=pages; pages=page; ++page_count;
        e=&page->slots[0];
    }
    e->id = next_id++;
    e->parent = parent; e->limit = max_bytes; e->kind = kind;
    e->callback = callback; e->context = context;
    e->owner = page;
    ++page->used; ++live_count;
    if (p) ++p->children;
    *handle = e->id;
    return OA_OK;
}
oa_status OA_CALL oa_segment_call(oa_handle handle, uint32_t operation,
    uint64_t rows, const void *input, uint64_t input_bytes,
    void *output, uint64_t capacity, uint64_t *written) {
    entry *e = lookup(handle);
    oa_status status;
    uint64_t count = 0;
    if (!written) return OA_INVALID;
    *written = 0;
    if (!e) return OA_HANDLE;
    if (e->busy) return OA_BUSY;
    if (!((e->kind == OA_SESSION && (operation == OA_FORWARD ||
          operation == OA_RESET || operation == OA_SNAPSHOT || operation == OA_RESTORE)) ||
          (e->kind == OA_EDGE && (operation == OA_EMBED || operation == OA_HEAD))))
        return OA_INVALID;
    if ((input_bytes && !input) || (capacity && !output)) return OA_INVALID;
    if (input_bytes > e->limit || capacity > e->limit ||
        input_bytes > SIZE_MAX || capacity > SIZE_MAX) return OA_LIMIT;
    /* Validate operation envelopes before handing borrowed buffers to callbacks.
     * Tensor widths and snapshot contents remain backend responsibilities. */
    if (operation == OA_FORWARD || operation == OA_EMBED || operation == OA_HEAD) {
        if (!rows || !input_bytes || !capacity) return OA_INVALID;
    } else if (operation == OA_RESET) {
        if (rows || input_bytes || capacity) return OA_INVALID;
    } else if (operation == OA_SNAPSHOT) {
        if (rows || input_bytes) return OA_INVALID;
    } else if (operation == OA_RESTORE) {
        if (rows || !input_bytes || capacity) return OA_INVALID;
    }
    e->busy = 1;
    status = e->callback(e->context, operation, rows, input, input_bytes,
                         output, capacity, &count);
    e->busy = 0;
    if (count > capacity || status < OA_OK || status > OA_CANCELLED) return OA_CALLBACK;
    if (status == OA_OK) *written = count;
    return status;
}
oa_status OA_CALL oa_segment_release(oa_handle handle) {
    entry *e = lookup(handle), *p;
    oa_status status;
    uint64_t count = 0;
    if (!e) return OA_HANDLE;
    if (e->busy || e->children) return OA_BUSY;
    e->busy = 1;
    status = e->callback(e->context, OA_CLOSE, 0, NULL, 0, NULL, 0, &count);
    e->busy = 0;
    if (count || status < OA_OK || status > OA_CANCELLED) return OA_CALLBACK;
    if (status != OA_OK) return status;
    p = lookup(e->parent);
    if (p) --p->children;
    --e->owner->used;
    memset(e, 0, sizeof(*e));
    --live_count;
    reclaim_empty_registry();
    return OA_OK;
}
