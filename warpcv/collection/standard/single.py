import cupy as cp
from ...utils import options, backend

# Float32 version
bilinear_kernel_1c = cp.RawKernel(r'''
extern "C" __global__
void cutlass_resize_bilinear(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int in_h, int in_w,
    int out_h, int out_w
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= out_w || y >= out_h) return;
     
    const float scale_x = __fdividef((float)(in_w - 1), (float)(out_w - 1));
    const float scale_y = __fdividef((float)(in_h - 1), (float)(out_h - 1));
     
    const float src_x = x * scale_x;
    const float src_y = y * scale_y;
    
    const int x0 = __float2int_rd(src_x);
    const int y0 = __float2int_rd(src_y);
    const int x1 = min(x0 + 1, in_w - 1);
    const int y1 = min(y0 + 1, in_h - 1);
     
    const float dx = src_x - x0;
    const float dy = src_y - y0;
    const float dx1 = 1.0f - dx;
    const float dy1 = 1.0f - dy;
    
    const float v00 = src[y0 * in_w + x0];
    const float v01 = src[y0 * in_w + x1];
    const float v10 = src[y1 * in_w + x0];
    const float v11 = src[y1 * in_w + x1];
     
    const float top = __fmaf_rn(v01, dx, __fmaf_rn(v00, dx1, 0.0f));
    const float bottom = __fmaf_rn(v11, dx, __fmaf_rn(v10, dx1, 0.0f));
    
    dst[y * out_w + x] = __fmaf_rn(bottom, dy, __fmaf_rn(top, dy1, 0.0f));
}
''', 'cutlass_resize_bilinear', options=options, backend=backend)

# Float16 version
bilinear_kernel_1c_fp16 = cp.RawKernel(r'''
#include <cuda_fp16.h>

extern "C" __global__
void cutlass_resize_bilinear_fp16(
    const half* __restrict__ src,
    half* __restrict__ dst,
    int in_h, int in_w,
    int out_h, int out_w
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= out_w || y >= out_h) return;
     
    const float scale_x = __fdividef((float)(in_w - 1), (float)(out_w - 1));
    const float scale_y = __fdividef((float)(in_h - 1), (float)(out_h - 1));
     
    const float src_x = x * scale_x;
    const float src_y = y * scale_y;
    
    const int x0 = __float2int_rd(src_x);
    const int y0 = __float2int_rd(src_y);
    const int x1 = min(x0 + 1, in_w - 1);
    const int y1 = min(y0 + 1, in_h - 1);
     
    const half dx = __float2half(src_x - x0);
    const half dy = __float2half(src_y - y0);
    const half dx1 = __hsub(__float2half(1.0f), dx);
    const half dy1 = __hsub(__float2half(1.0f), dy);
    
    const half v00 = src[y0 * in_w + x0];
    const half v01 = src[y0 * in_w + x1];
    const half v10 = src[y1 * in_w + x0];
    const half v11 = src[y1 * in_w + x1];
     
    const half top = __hfma(v01, dx, __hfma(v00, dx1, __float2half(0.0f)));
    const half bottom = __hfma(v11, dx, __hfma(v10, dx1, __float2half(0.0f)));
    
    dst[y * out_w + x] = __hfma(bottom, dy, __hfma(top, dy1, __float2half(0.0f)));
}
''', 'cutlass_resize_bilinear_fp16', options=options, backend=backend)

def cupy_resize_1c(img_cp, out_h, out_w, dtype, block_size):
    '''
    Comparable to cv2.resize with INTER_LINEAR for 1 channel image.
    Supports both float32 and float16 (half) precision.
    '''
    in_h, in_w = img_cp.shape
    assert len(img_cp.shape) == 2, "Only single channel images supported"

    img_cp = cp.ascontiguousarray(img_cp).astype(dtype, copy=False)
    out_cp = cp.empty((out_h, out_w), dtype=dtype)
     
    grid = (
        (out_w + block_size[0] - 1) // block_size[0],
        (out_h + block_size[1] - 1) // block_size[1]
    )
     
    if dtype == cp.float16:
        bilinear_kernel_1c_fp16(
            grid, block_size, (img_cp, out_cp, in_h, in_w, out_h, out_w)
        )
    else:
        bilinear_kernel_1c(
            grid, block_size, (img_cp, out_cp, in_h, in_w, out_h, out_w)
        )
     
    return out_cp

# Float32 version
bilinear_kernel_3c = cp.RawKernel(r'''
extern "C" __global__
void cutlass_resize_bilinear_3c(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int in_h, int in_w,
    int out_h, int out_w,
    int c
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int ch = blockIdx.z;
     
    if (x >= out_w || y >= out_h) return;

    const float scale_x = __fdividef((float)in_w, (float)out_w);
    const float scale_y = __fdividef((float)in_h, (float)out_h);
    
    float src_x = __fmaf_rn(x + 0.5f, scale_x, -0.5f);
    float src_y = __fmaf_rn(y + 0.5f, scale_y, -0.5f);
    
    src_x = fminf(fmaxf(src_x, 0.0f), (float)(in_w - 1.0001f));
    src_y = fminf(fmaxf(src_y, 0.0f), (float)(in_h - 1.0001f));

    const int x0 = __float2int_rd(src_x);
    const int y0 = __float2int_rd(src_y);
    const int x1 = min(x0 + 1, in_w - 1);
    const int y1 = min(y0 + 1, in_h - 1);
    
    const float dx = src_x - x0;
    const float dy = src_y - y0;
    const float dx1 = 1.0f - dx;
    const float dy1 = 1.0f - dy;
    
    const int src_idx_y0 = y0 * in_w * c;
    const int src_idx_y1 = y1 * in_w * c;
    const int src_idx_x0 = x0 * c;
    const int src_idx_x1 = x1 * c;
    
    const float v00 = src[src_idx_y0 + src_idx_x0 + ch];
    const float v01 = src[src_idx_y0 + src_idx_x1 + ch];
    const float v10 = src[src_idx_y1 + src_idx_x0 + ch];
    const float v11 = src[src_idx_y1 + src_idx_x1 + ch];
    
    const float top = __fmaf_rn(v01, dx, __fmaf_rn(v00, dx1, 0.0f));
    const float bottom = __fmaf_rn(v11, dx, __fmaf_rn(v10, dx1, 0.0f));
    
    const int dst_idx = (y * out_w + x) * c + ch;
    dst[dst_idx] = __fmaf_rn(bottom, dy, __fmaf_rn(top, dy1, 0.0f));
}
''', 'cutlass_resize_bilinear_3c', options=options, backend=backend)

# Float16 version
bilinear_kernel_3c_fp16 = cp.RawKernel(r'''
#include <cuda_fp16.h>

extern "C" __global__
void cutlass_resize_bilinear_3c_fp16(
    const half* __restrict__ src,
    half* __restrict__ dst,
    int in_h, int in_w,
    int out_h, int out_w,
    int c
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int ch = blockIdx.z;
     
    if (x >= out_w || y >= out_h) return;

    const float scale_x = __fdividef((float)in_w, (float)out_w);
    const float scale_y = __fdividef((float)in_h, (float)out_h);
    
    float src_x = __fmaf_rn(x + 0.5f, scale_x, -0.5f);
    float src_y = __fmaf_rn(y + 0.5f, scale_y, -0.5f);
    
    src_x = fminf(fmaxf(src_x, 0.0f), (float)(in_w - 1.0001f));
    src_y = fminf(fmaxf(src_y, 0.0f), (float)(in_h - 1.0001f));

    const int x0 = __float2int_rd(src_x);
    const int y0 = __float2int_rd(src_y);
    const int x1 = min(x0 + 1, in_w - 1);
    const int y1 = min(y0 + 1, in_h - 1);
    
    const half dx = __float2half(src_x - x0);
    const half dy = __float2half(src_y - y0);
    const half dx1 = __hsub(__float2half(1.0f), dx);
    const half dy1 = __hsub(__float2half(1.0f), dy);
    
    const int src_idx_y0 = y0 * in_w * c;
    const int src_idx_y1 = y1 * in_w * c;
    const int src_idx_x0 = x0 * c;
    const int src_idx_x1 = x1 * c;
    
    const half v00 = src[src_idx_y0 + src_idx_x0 + ch];
    const half v01 = src[src_idx_y0 + src_idx_x1 + ch];
    const half v10 = src[src_idx_y1 + src_idx_x0 + ch];
    const half v11 = src[src_idx_y1 + src_idx_x1 + ch];
    
    const half top = __hfma(v01, dx, __hfma(v00, dx1, __float2half(0.0f)));
    const half bottom = __hfma(v11, dx, __hfma(v10, dx1, __float2half(0.0f)));
    
    const int dst_idx = (y * out_w + x) * c + ch;
    dst[dst_idx] = __hfma(bottom, dy, __hfma(top, dy1, __float2half(0.0f)));
}
''', 'cutlass_resize_bilinear_3c_fp16', options=options, backend=backend)

def cupy_resize_3c(img_cp, out_h, out_w, dtype, block_size):
    '''
    Comparable to cv2.resize with INTER_LINEAR for 3 channels image.
    Supports both float32 and float16 (half) precision.
    '''
    h, w, c = img_cp.shape
    assert c == 3, "Only 3-channel images supported"

    img_cp = cp.ascontiguousarray(img_cp).astype(dtype, copy=False)
    out = cp.empty((out_h, out_w, 3), dtype=dtype)
    
    grid = (
        (out_w + block_size[0] - 1) // block_size[0],
        (out_h + block_size[1] - 1) // block_size[1],
        c
    )
    
    if dtype == cp.float16:
        bilinear_kernel_3c_fp16(
            grid, block_size, (img_cp, out, h, w, out_h, out_w, c)
        )
    else:
        bilinear_kernel_3c(
            grid, block_size, (img_cp, out, h, w, out_h, out_w, c)
        )
    
    return out

# Float32 version
bgr2rgb_float_kernel = cp.RawKernel(r'''
extern "C" __global__
void cutlass_bgr2rgb_float(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int h, int w
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;
    
    if (x >= w || y >= h) return;
                                    
    int idx = (y * w + x) * 3 + c;
    
    // Map BGR to RGB: channel 0<->2, channel 1 stays
    int src_c = (c == 0) ? 2 : (c == 2) ? 0 : 1;
    int src_idx = (y * w + x) * 3 + src_c;
    
    dst[idx] = src[src_idx];
}
''', 'cutlass_bgr2rgb_float', options=options, backend=backend)

# Float16 version
bgr2rgb_float_kernel_fp16 = cp.RawKernel(r'''
#include <cuda_fp16.h>

extern "C" __global__
void cutlass_bgr2rgb_float_fp16(
    const half* __restrict__ src,
    half* __restrict__ dst,
    int h, int w
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;
    
    if (x >= w || y >= h) return;
                                    
    int idx = (y * w + x) * 3 + c;
    
    // Map BGR to RGB: channel 0<->2, channel 1 stays
    int src_c = (c == 0) ? 2 : (c == 2) ? 0 : 1;
    int src_idx = (y * w + x) * 3 + src_c;
    
    dst[idx] = src[src_idx];
}
''', 'cutlass_bgr2rgb_float_fp16', options=options, backend=backend)

def cupy_cvt_bgr2rgb_float(img_cp, dtype, block_size):
    '''
    Convert BGR image to RGB format for float images.
    Comparable to cv2.cvtColor(img, cv2.COLOR_BGR2RGB).
    Supports both float32 and float16 (half) precision.
    '''
    h, w, c = img_cp.shape
    assert c == 3, "Only 3-channel images supported"

    img_cp = cp.ascontiguousarray(img_cp).astype(dtype, copy=False)
    out = cp.empty_like(img_cp, dtype=dtype)
    
    grid = (
        (w + block_size[0] - 1) // block_size[0],
        (h + block_size[1] - 1) // block_size[1],
        c
    )
    
    if dtype == cp.float16:
        bgr2rgb_float_kernel_fp16(
            grid, block_size, (img_cp, out, h, w)
        )
    else:
        bgr2rgb_float_kernel(
            grid, block_size, (img_cp, out, h, w)
        )
    
    return out