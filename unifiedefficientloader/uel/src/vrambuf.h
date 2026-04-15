#pragma once

#include "plat.h"

typedef struct VramBuffer {
    CUdeviceptr base_ptr;
    size_t max_size;
    size_t allocated;
    size_t handle_count;
    int device;
    struct VramBuffer *next;
    CUmemGenericAllocationHandle handles[1];
} VramBuffer;

SHARED_EXPORT
void *vrambuf_create(int device, size_t max_size);

SHARED_EXPORT
bool vrambuf_grow(void *arg, size_t required_size);

SHARED_EXPORT
void vrambuf_destroy(void *arg);

SHARED_EXPORT
CUdeviceptr vrambuf_get(void *arg);
