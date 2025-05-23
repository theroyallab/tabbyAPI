#pragma once

typedef struct __align__(8) half4
{
    half2 x;
    half2 y;
    __host__ __device__ half4() = default;
    __host__ __device__ half4(half2 x_, half2 y_) : x(x_), y(y_) {}
    __host__ __device__ half4(half h0, half h1, half h2, half h3) :
         x(__halves2half2(h0, h1)),
         y(__halves2half2(h2, h3)) {}
}
half4;

typedef struct __align__(16) half8
{
    half2 x;
    half2 y;
    half2 z;
    half2 w;
    __host__ __device__ half8() = default;
    __host__ __device__ half8(half2 x_, half2 y_, half2 z_, half2 w_) : x(x_), y(y_), z(z_), w(w_) {}
    __host__ __device__ half8(half h0, half h1, half h2, half h3, half h4, half h5, half h6, half h7) :
         x(__halves2half2(h0, h1)),
         y(__halves2half2(h2, h3)),
         z(__halves2half2(h4, h5)),
         w(__halves2half2(h6, h7)) {}
}
half8;

struct Dim3
{
    int m;
    int k;
    int n;
    inline __device__ int numel_a() { return m * k; }
    inline __device__ int numel_b() { return k * n; }
    inline __device__ int numel_c() { return m * n; }
};

#define READ128(__x, __y) ((uint4*)&__x)[0] = ((uint4*)(__y))[0];
#define WRITE128(__x, __y) ((uint4*)__x)[0] = ((uint4*)(&__y))[0];
#define READ64(__x, __y) ((uint2*)&__x)[0] = ((uint2*)(__y))[0];
#define WRITE64(__x, __y) ((uint2*)__x)[0] = ((uint2*)(&__y))[0];

#define LOW_TO_FLOAT(__x) __half2float(__low2half(__x))
#define HIGH_TO_FLOAT(__x) __half2float(__high2half(__x))

#define CLAMP(__x, __min, __max) fmaxf(__min, fminf(__x, __max))
#define CLAMP_FP16(__x) CLAMP(__x, -65504.0f, 65504.0f)

#define SWAP16(__x) __byte_perm(__x, 0, 0x1032)

union half2_uint32
{
    uint32_t as_uint32;
    half2 as_half2;
    __device__ half2_uint32(uint32_t val) : as_uint32(val) {}
    __device__ half2_uint32(half2 val) : as_half2(val) {}
    __device__ half2_uint32() : as_uint32(0) {}
};

union half_uint16
{
    uint16_t as_uint16;
    half as_half;
    __device__ half_uint16(uint16_t val) : as_uint16(val) {}
    __device__ half_uint16(half val) : as_half(val) {}
    __device__ half_uint16() : as_uint16(0) {}
};

#define cuda_check(ans) { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ inline float fxor(float v, uint32_t mask)
{
    uint32_t* vi = reinterpret_cast<uint32_t*>(&v);
    *vi ^= mask;
    return v;
}

__device__ inline half2 h2xor(half2 v, uint32_t mask)
{
    uint32_t* vi = reinterpret_cast<uint32_t*>(&v);
    *vi ^= mask;
    return v;
}

#define NEG_INF_F16 __ushort_as_half(0xFC00)
#define POS_INF_F16 __ushort_as_half(0x7C00)
