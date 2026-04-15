#include "plat.h"

SHARED_EXPORT
void *hostbuf_allocate(uint64_t size) {
    void *ptr = NULL;
    size_t alloc_size = (size_t)size;

    if (!size) {
        return NULL;
    }

    if (!CHECK_CU(cuMemAllocHost(&ptr, alloc_size))) {
        log(DEBUG, "%s: CUDA host allocation failed (%zuk)\n", __func__, alloc_size / K);
        return NULL;
    }

    return ptr;
}

SHARED_EXPORT
void hostbuf_free(void *arg) {
    if (!arg) {
        return;
    }
    CHECK_CU(cuMemFreeHost(arg));
}
