#define _GNU_SOURCE
#define _FILE_OFFSET_BITS 64
#include <liburing.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int oa_uring_probe(void) {
    struct io_uring ring;
    int rc = io_uring_queue_init(2, &ring, 0);
    if (!rc) io_uring_queue_exit(&ring);
    return rc;
}

/* ABI: zero on success, negative errno on failure, including CQE errors. */
int oa_uring_read(const char *path, uint64_t offset, uint64_t size,
                  void *output, uint64_t *received) {
    struct io_uring ring;
    struct stat st;
    void *buffer = NULL;
    int fd, initialized = 0, rc = 0;
    uint64_t wanted, done = 0;
    size_t alignment = 4096, capacity;
    if (!path || !received || (size && !output)) return -EINVAL;
    *received = 0;
    if (offset > INT64_MAX || size > INT64_MAX - offset || size > SIZE_MAX)
        return -EOVERFLOW;
    fd = open(path, O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (fd < 0) return -errno;
    if (fstat(fd, &st) < 0) { rc = -errno; goto cleanup; }
    if (!S_ISREG(st.st_mode) || st.st_size < 0) { rc = -EINVAL; goto cleanup; }
    wanted = offset >= (uint64_t)st.st_size ? 0 : (uint64_t)st.st_size - offset;
    if (wanted > size) wanted = size;
    if (!wanted) goto cleanup;
    if (st.st_blksize > 0 && (uint64_t)st.st_blksize > alignment)
        alignment = (size_t)st.st_blksize;
    /* posix_memalign requires power-of-two alignment. */
    if ((alignment & (alignment - 1)) || alignment > 1048576) {
        rc = -EOPNOTSUPP; goto cleanup;
    }
    capacity = 1048576 / alignment * alignment;
    rc = posix_memalign(&buffer, alignment, capacity);
    if (rc) { rc = -rc; goto cleanup; }
    rc = io_uring_queue_init(2, &ring, 0);
    if (rc < 0) goto cleanup;
    initialized = 1;
    while (done < wanted) {
        uint64_t position = offset + done;
        uint64_t start = position / alignment * alignment;
        size_t lower = (size_t)(position - start), count, copied;
        uint64_t needed = wanted - done;
        struct io_uring_sqe *sqe;
        struct io_uring_cqe *cqe;
        if (needed > capacity - lower) needed = capacity - lower;
        count = (lower + (size_t)needed + alignment - 1) / alignment * alignment;
        if (start > INT64_MAX - count) { rc = -EOVERFLOW; goto cleanup; }
        sqe = io_uring_get_sqe(&ring);
        if (!sqe) { rc = -EIO; goto cleanup; }
        io_uring_prep_read(sqe, fd, buffer, (unsigned)count, start);
        rc = io_uring_submit(&ring);
        if (rc < 0) goto cleanup;
        if (rc != 1) { rc = -EIO; goto cleanup; }
        do { rc = io_uring_wait_cqe(&ring, &cqe); } while (rc == -EINTR);
        if (rc < 0) goto cleanup;
        rc = cqe->res;
        io_uring_cqe_seen(&ring, cqe);
        if (rc < 0) goto cleanup;
        if ((size_t)rc > count || (size_t)rc <= lower) { rc = -EIO; goto cleanup; }
        copied = (size_t)rc - lower;
        if (copied > needed) copied = (size_t)needed;
        memcpy((char *)output + (size_t)done, (char *)buffer + lower, copied);
        done += copied;
    }
    *received = done;
    rc = 0;
cleanup:
    /* Drain kernel ownership before releasing buffers and the descriptor. */
    if (initialized) io_uring_queue_exit(&ring);
    free(buffer);
    close(fd);
    return rc;
}
