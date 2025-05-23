#pragma once

// Tensor core fragments

template <typename T, int n>
struct Vec
{
    T elems[n];
    __device__ T& operator[](int i) { return elems[i]; }
};

using FragA = Vec<half2, 4>;
using FragB = Vec<half2, 2>;
using FragC = Vec<float, 4>;
using FragC_h = Vec<half2, 2>;

// m8n8k4 tensor core matmul (emulated on Ampere and later), don't use
//
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m8n8k4-with-f16-floating-point-type

__device__ inline void ptx_mma_m8n8k4
(
    const Vec<half2, 2>& frag_a,
    const Vec<half2, 2>& frag_b,
    Vec<float, 8>& frag_c
)
{
    const uint32_t* a = reinterpret_cast<const uint32_t*>(&frag_a);
    const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
    float* c = reinterpret_cast<float*>(&frag_c);
    const float* d = reinterpret_cast<const float*>(&frag_c);

    asm volatile
    (
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"

        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3]),"=f"(c[4]), "=f"(c[5]), "=f"(c[6]), "=f"(c[7])

        :  "r"(a[0]), "r"(a[1]),
           "r"(b[0]), "r"(b[1]),
           "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]), "f"(d[4]), "f"(d[5]), "f"(d[6]), "f"(d[7])
    );
}

// m16n8k16 tensor core matmul
//
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type

// FP16 @ FP16 + FP32 -> FP32
__device__ inline void ptx_mma_m16n8k16
(
    const FragA& frag_a,
    const FragB& frag_b,
    FragC& frag_c
)
{
    const uint32_t* a = reinterpret_cast<const uint32_t*>(&frag_a);
    const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
    float* c = reinterpret_cast<float*>(&frag_c);
    const float* d = reinterpret_cast<const float*>(&frag_c);

    asm volatile
    (
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"

        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        :  "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
           "r"(b[0]), "r"(b[1]),
           "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3])
    );
}

// FP16 @ FP16 + FP16 -> FP16
__device__ inline void ptx_mma_m16n8k16
(
    const FragA& frag_a,
    const FragB& frag_b,
    FragC_h& frag_c
)
{
    const uint32_t* a = reinterpret_cast<const uint32_t*>(&frag_a);
    const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
    uint32_t* c = reinterpret_cast<uint32_t*>(&frag_c);
    const uint32_t* d = reinterpret_cast<const uint32_t*>(&frag_c);

    asm volatile
    (
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"

        : "=r"(c[0]), "=r"(c[1])
        :  "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
           "r"(b[0]), "r"(b[1]),
           "r"(d[0]), "r"(d[1])
    );
}

// Global barrier

__device__ inline void barrier_acquire
(
    int* lock,
    int stage
)
{
    if (threadIdx.x == 0)
    {
        volatile int state = -1;
        do
        {
            asm volatile ("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
        }
        while (state != stage);
    }
    __syncthreads();
}

__device__ inline void barrier_release
(
    int* lock,
    int val,
    bool reset
)
{
    __syncthreads();
    if (threadIdx.x == 0)
    {
        if (reset)
        {
            *lock = 0;
            return;
        }
        asm volatile ("fence.acq_rel.gpu;\n");
        asm volatile ("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(lock), "r"(val));
    }
}

// Load global to shared memory, predicated. Seems to produce incorrect code when compiling for Blackwell, but
// `if (...) cp_async(...)` compiles to a predicated instruction anyway

__device__ inline void cp_async_pred(void* smem_ptr, const void* glob_ptr, bool pred = true)
{
    const int bytes = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   .reg .pred p;\n"
        "   setp.ne.b32 p, %0, 0;\n"
        "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
        "}\n" :: "r"((int) pred), "r"(smem), "l"(glob_ptr), "n"(bytes)
    );
}

// Load global to shared memory

__device__ inline void cp_async(void* smem_ptr, const void* glob_ptr)
{
    const int bytes = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   cp.async.cg.shared.global [%0], [%1], %2;\n"
        "}\n" :: "r"(smem), "l"(glob_ptr), "n"(bytes)
    );
}

// Load global to shared memory with cache hint to evict data from L2 ASAP

__device__ inline void cp_async_stream(void* smem_ptr, const void* glob_ptr)
{
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    const int bytes = 16;
    asm volatile
    (
        "{\n"
        "   .reg .b64 p;\n"
        "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;\n"
        "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
        "}\n" :: "r"(smem), "l"(glob_ptr), "n"(bytes)
    );
}

// Async copy fence, commit all pending async copies

__device__ inline void cp_async_fence()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

// Wait until at most n async groups are still pending.

template <int n>
__device__ inline void cp_async_wait()
{
    asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
}

// Load 16x16 matrix fragment from shared memory, directly in tensor core layout

__device__ inline void ldsm4(FragA& frag_a, const void* smem_ptr)
{
    uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile
    (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(smem)
    );
}

__device__ inline uint32_t mul_lo_u32(uint32_t x, uint32_t y)
{
    uint32_t w;
    asm volatile
    (
        "mul.lo.u32 %0, %1, %2;"
        : "=r"(w)
        :  "r"(x), "r"(y)
    );
    return w;
}

__device__ inline uint32_t mul_hi_u32(uint32_t x, uint32_t y)
{
    uint32_t w;
    asm volatile
    (
        "mul.hi.u32 %0, %1, %2;"
        : "=r"(w)
        :  "r"(x), "r"(y)
    );
    return w;
}

static __forceinline__ __device__ uint32_t bfe64(uint32_t lo, uint32_t hi, int offset, int length)
{
    uint64_t value = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
    uint64_t result64;
    asm volatile ("bfe.u64 %0, %1, %2, %3;"
                  : "=l"(result64)
                  : "l"(value), "r"(offset), "r"(length));

    return static_cast<uint32_t>(result64);
}