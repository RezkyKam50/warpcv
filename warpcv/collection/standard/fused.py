import cupy as cp
from ...utils import options, backend

# Float32 version
fused_bgr2rgb_resize_kernel = cp.RawKernel(r'''
extern "C" __global__
void cutlass_fused_bgr2rgb_resize_3c(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int in_h, int in_w,
    int out_h, int out_w
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= out_w || y >= out_h || c >= 3 || 
        out_w <= 0 || out_h <= 0 || in_w <= 0 || in_h <= 0) return;
    
    int src_c, dst_c;
    if (c == 0) {
        src_c = 2;  // R in source
        dst_c = 0;  // B in destination
    } else if (c == 1) {
        src_c = 1;  // G in source
        dst_c = 1;  // G in destination
    } else { // c == 2
        src_c = 0;  // B in source
        dst_c = 2;  // R in destination
    }
    
    const float scale_x = __fdividef((float)in_w, (float)out_w);  
    const float scale_y = __fdividef((float)in_h, (float)out_h);
     
    float src_x = __fmaf_rn(x + 0.5f, scale_x, -0.5f);
    float src_y = __fmaf_rn(y + 0.5f, scale_y, -0.5f);
                                   
    src_x = fminf(fmaxf(src_x, 0.0f), (float)(in_w - 1));
    src_y = fminf(fmaxf(src_y, 0.0f), (float)(in_h - 1));                                       

    int x0 = __float2int_rd(src_x);
    int y0 = __float2int_rd(src_y);                                      
    int x1 = min(x0 + 1, in_w - 1);
    int y1 = min(y0 + 1, in_h - 1);
                                           
    float dx = src_x - (float)x0;
    float dy = src_y - (float)y0;
    float dx1 = 1.0f - dx;
    float dy1 = 1.0f - dy;
     
    int idx00 = (y0 * in_w + x0) * 3 + src_c;
    int idx01 = (y0 * in_w + x1) * 3 + src_c;
    int idx10 = (y1 * in_w + x0) * 3 + src_c;
    int idx11 = (y1 * in_w + x1) * 3 + src_c;
    int dst_idx = (y * out_w + x) * 3 + dst_c;
       
    float v00 = src[idx00];
    float v01 = src[idx01];
    float v10 = src[idx10];
    float v11 = src[idx11];
                                                                 
    float top = __fmaf_rn(v00, dx1, v01 * dx);
    float bottom = __fmaf_rn(v10, dx1, v11 * dx);
    dst[dst_idx] = __fmaf_rn(top, dy1, bottom * dy);
}
''', 'cutlass_fused_bgr2rgb_resize_3c', options=options, backend=backend)

# Float16 version
fused_bgr2rgb_resize_kernel_fp16 = cp.RawKernel(r'''
#include <cuda_fp16.h>

extern "C" __global__
void cutlass_fused_bgr2rgb_resize_3c_fp16(
    const half* __restrict__ src,
    half* __restrict__ dst,
    int in_h, int in_w,
    int out_h, int out_w
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= out_w || y >= out_h || c >= 3 || 
        out_w <= 0 || out_h <= 0 || in_w <= 0 || in_h <= 0) return;
    
    int src_c, dst_c;
    if (c == 0) {
        src_c = 2;
        dst_c = 0;
    } else if (c == 1) {
        src_c = 1;
        dst_c = 1;
    } else {
        src_c = 0;
        dst_c = 2;
    }
    
    const float scale_x = __fdividef((float)in_w, (float)out_w);  
    const float scale_y = __fdividef((float)in_h, (float)out_h);
     
    float src_x = __fmaf_rn(x + 0.5f, scale_x, -0.5f);
    float src_y = __fmaf_rn(y + 0.5f, scale_y, -0.5f);
                                   
    src_x = fminf(fmaxf(src_x, 0.0f), (float)(in_w - 1));
    src_y = fminf(fmaxf(src_y, 0.0f), (float)(in_h - 1));                                       

    int x0 = __float2int_rd(src_x);
    int y0 = __float2int_rd(src_y);                                      
    int x1 = min(x0 + 1, in_w - 1);
    int y1 = min(y0 + 1, in_h - 1);
                                           
    half dx = __float2half(src_x - (float)x0);
    half dy = __float2half(src_y - (float)y0);
    half dx1 = __hsub(__float2half(1.0f), dx);
    half dy1 = __hsub(__float2half(1.0f), dy);
     
    int idx00 = (y0 * in_w + x0) * 3 + src_c;
    int idx01 = (y0 * in_w + x1) * 3 + src_c;
    int idx10 = (y1 * in_w + x0) * 3 + src_c;
    int idx11 = (y1 * in_w + x1) * 3 + src_c;
    int dst_idx = (y * out_w + x) * 3 + dst_c;
       
    half v00 = src[idx00];
    half v01 = src[idx01];
    half v10 = src[idx10];
    half v11 = src[idx11];
                                                                 
    half top = __hfma(v00, dx1, __hmul(v01, dx));
    half bottom = __hfma(v10, dx1, __hmul(v11, dx));
    dst[dst_idx] = __hfma(top, dy1, __hmul(bottom, dy));
}
''', 'cutlass_fused_bgr2rgb_resize_3c_fp16', options=options, backend=backend)

def fused_bgr2rgb_resize_3c(img_cp, out_h, out_w, dtype, block_size):
    '''
    Fused BGR to RGB conversion and resizing for 3 channels image.
    Comparable to cv2.cvtColor + cv2.resize with INTER_LINEAR.
    Supports both float32 and float16 (half) precision.
    '''
    h, w, c = img_cp.shape
    assert c == 3, "Only 3-channel images supported"

    img_cp = cp.ascontiguousarray(img_cp).astype(dtype, copy=False)
    out = cp.empty((out_h, out_w, 3), dtype=dtype) 
     
    grid = (
        (out_w + block_size[0] - 1) // block_size[0],
        (out_h + block_size[1] - 1) // block_size[1],
        (c + block_size[2] - 1) // block_size[2]
    )
    
    if dtype == cp.float16:
        fused_bgr2rgb_resize_kernel_fp16(
            grid, block_size, (img_cp, out, h, w, out_h, out_w)
        )
    else:
        fused_bgr2rgb_resize_kernel(
            grid, block_size, (img_cp, out, h, w, out_h, out_w)
        )
    return out

# Float32 version
fused_resize_normalize_transpose_kernel_high = cp.RawKernel(r'''
extern "C" __global__
void cutlass_fused_resize_normalize_transpose_3c(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int in_h, int in_w,
    int out_h, int out_w,
    const float* __restrict__ mean,
    const float* __restrict__ std
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= out_w || y >= out_h) return;
    
    const float scale_x = __fdividef((float)in_w, (float)out_w);  
    const float scale_y = __fdividef((float)in_h, (float)out_h);
    
    float src_x = __fmaf_rn(x + 0.5f, scale_x, -0.5f);
    float src_y = __fmaf_rn(y + 0.5f, scale_y, -0.5f);
    
    src_x = fminf(fmaxf(src_x, 0.0f), (float)(in_w - 1));
    src_y = fminf(fmaxf(src_y, 0.0f), (float)(in_h - 1));
    
    int x0 = __float2int_rd(src_x);
    int y0 = __float2int_rd(src_y);
    int x1 = min(x0 + 1, in_w - 1);
    int y1 = min(y0 + 1, in_h - 1);
    
    float dx = src_x - (float)x0;
    float dy = src_y - (float)y0;
    float dx1 = 1.0f - dx;
    float dy1 = 1.0f - dy;
    
    int base00 = (y0 * in_w + x0) * 3;
    int base01 = (y0 * in_w + x1) * 3;
    int base10 = (y1 * in_w + x0) * 3;
    int base11 = (y1 * in_w + x1) * 3;
    
    #pragma unroll 3
    for(int c = 0; c < 3; c++) {
        float v00 = src[base00 + c];
        float v01 = src[base01 + c];
        float v10 = src[base10 + c];
        float v11 = src[base11 + c];
        
        float top = __fmaf_rn(v00, dx1, v01 * dx);
        float bottom = __fmaf_rn(v10, dx1, v11 * dx);
        float interpolated = __fmaf_rn(top, dy1, bottom * dy);
        
        float normalized = __fdividef(interpolated - mean[c], std[c]);
        
        dst[c * (out_h * out_w) + y * out_w + x] = normalized;
    }
}
''', 'cutlass_fused_resize_normalize_transpose_3c', options=options, backend=backend)

# Float16 version
fused_resize_normalize_transpose_kernel_high_fp16 = cp.RawKernel(r'''
#include <cuda_fp16.h>

extern "C" __global__
void cutlass_fused_resize_normalize_transpose_3c_fp16(
    const half* __restrict__ src,
    half* __restrict__ dst,
    int in_h, int in_w,
    int out_h, int out_w,
    const half* __restrict__ mean,
    const half* __restrict__ std
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= out_w || y >= out_h) return;
    
    const float scale_x = __fdividef((float)in_w, (float)out_w);  
    const float scale_y = __fdividef((float)in_h, (float)out_h);
    
    float src_x = __fmaf_rn(x + 0.5f, scale_x, -0.5f);
    float src_y = __fmaf_rn(y + 0.5f, scale_y, -0.5f);
    
    src_x = fminf(fmaxf(src_x, 0.0f), (float)(in_w - 1));
    src_y = fminf(fmaxf(src_y, 0.0f), (float)(in_h - 1));
    
    int x0 = __float2int_rd(src_x);
    int y0 = __float2int_rd(src_y);
    int x1 = min(x0 + 1, in_w - 1);
    int y1 = min(y0 + 1, in_h - 1);
    
    half dx = __float2half(src_x - (float)x0);
    half dy = __float2half(src_y - (float)y0);
    half dx1 = __hsub(__float2half(1.0f), dx);
    half dy1 = __hsub(__float2half(1.0f), dy);
    
    int base00 = (y0 * in_w + x0) * 3;
    int base01 = (y0 * in_w + x1) * 3;
    int base10 = (y1 * in_w + x0) * 3;
    int base11 = (y1 * in_w + x1) * 3;
    
    #pragma unroll 3
    for(int c = 0; c < 3; c++) {
        half v00 = src[base00 + c];
        half v01 = src[base01 + c];
        half v10 = src[base10 + c];
        half v11 = src[base11 + c];
        
        half top = __hfma(v00, dx1, __hmul(v01, dx));
        half bottom = __hfma(v10, dx1, __hmul(v11, dx));
        half interpolated = __hfma(top, dy1, __hmul(bottom, dy));
        
        half normalized = __hdiv(__hsub(interpolated, mean[c]), std[c]);
        
        dst[c * (out_h * out_w) + y * out_w + x] = normalized;
    }
}
''', 'cutlass_fused_resize_normalize_transpose_3c_fp16', options=options, backend=backend)

def fused_resize_normalize_transpose_3c_4k(img_cp, out_h, out_w, mean, std, dtype, block_size):
    '''
    Fused resize + normalize + HWC→CHW transpose for 3-channel 4k resolution images.
    Supports both float32 and float16 (half) precision.
    '''
    h, w, c = img_cp.shape
    assert c == 3, "Only 3-channel images supported"

    img_cp = cp.ascontiguousarray(img_cp).astype(dtype, copy=False)
    out = cp.empty((3, out_h, out_w), dtype=dtype)
    
    grid = (
        (out_w + block_size[0] - 1) // block_size[0],
        (out_h + block_size[1] - 1) // block_size[1]
    )

    if dtype == cp.float16:
        fused_resize_normalize_transpose_kernel_high_fp16(
            grid, block_size, (img_cp, out, h, w, out_h, out_w, mean, std)
        )
    else:
        fused_resize_normalize_transpose_kernel_high(
            grid, block_size, (img_cp, out, h, w, out_h, out_w, mean, std)
        )
    
    return out

# Float32 version
fused_resize_normalize_transpose_kernel_low = cp.RawKernel(r'''
extern "C" __global__
void cutlass_fused_resize_normalize_transpose_3c(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int in_h, int in_w,
    int out_h, int out_w,
    const float* __restrict__ mean,
    const float* __restrict__ std
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= out_w || y >= out_h || c >= 3 || 
        out_w <= 0 || out_h <= 0 || in_w <= 0 || in_h <= 0) return;
    
    const float scale_x = __fdividef((float)in_w, (float)out_w);  
    const float scale_y = __fdividef((float)in_h, (float)out_h);
    
    float src_x = __fmaf_rn(x + 0.5f, scale_x, -0.5f);
    float src_y = __fmaf_rn(y + 0.5f, scale_y, -0.5f);
    
    src_x = fminf(fmaxf(src_x, 0.0f), (float)(in_w - 1));
    src_y = fminf(fmaxf(src_y, 0.0f), (float)(in_h - 1));
    
    int x0 = __float2int_rd(src_x);
    int y0 = __float2int_rd(src_y);
    int x1 = min(x0 + 1, in_w - 1);
    int y1 = min(y0 + 1, in_h - 1);
    
    float dx = src_x - (float)x0;
    float dy = src_y - (float)y0;
    float dx1 = 1.0f - dx;
    float dy1 = 1.0f - dy;
    
    int idx00 = (y0 * in_w + x0) * 3 + c;
    int idx01 = (y0 * in_w + x1) * 3 + c;
    int idx10 = (y1 * in_w + x0) * 3 + c;
    int idx11 = (y1 * in_w + x1) * 3 + c;
    
    float v00 = src[idx00];
    float v01 = src[idx01];
    float v10 = src[idx10];
    float v11 = src[idx11];
    
    float top = __fmaf_rn(v00, dx1, v01 * dx);
    float bottom = __fmaf_rn(v10, dx1, v11 * dx);
    float interpolated = __fmaf_rn(top, dy1, bottom * dy);
     
    float normalized = __fdividef(interpolated - mean[c], std[c]);
     
    int dst_idx = c * (out_h * out_w) + y * out_w + x;
    dst[dst_idx] = normalized;
}
''', 'cutlass_fused_resize_normalize_transpose_3c', options=options, backend=backend)

# Float16 version
fused_resize_normalize_transpose_kernel_low_fp16 = cp.RawKernel(r'''
#include <cuda_fp16.h>

extern "C" __global__
void cutlass_fused_resize_normalize_transpose_3c_fp16(
    const half* __restrict__ src,
    half* __restrict__ dst,
    int in_h, int in_w,
    int out_h, int out_w,
    const half* __restrict__ mean,
    const half* __restrict__ std
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= out_w || y >= out_h || c >= 3 || 
        out_w <= 0 || out_h <= 0 || in_w <= 0 || in_h <= 0) return;
    
    const float scale_x = __fdividef((float)in_w, (float)out_w);  
    const float scale_y = __fdividef((float)in_h, (float)out_h);
    
    float src_x = __fmaf_rn(x + 0.5f, scale_x, -0.5f);
    float src_y = __fmaf_rn(y + 0.5f, scale_y, -0.5f);
    
    src_x = fminf(fmaxf(src_x, 0.0f), (float)(in_w - 1));
    src_y = fminf(fmaxf(src_y, 0.0f), (float)(in_h - 1));
    
    int x0 = __float2int_rd(src_x);
    int y0 = __float2int_rd(src_y);
    int x1 = min(x0 + 1, in_w - 1);
    int y1 = min(y0 + 1, in_h - 1);
    
    half dx = __float2half(src_x - (float)x0);
    half dy = __float2half(src_y - (float)y0);
    half dx1 = __hsub(__float2half(1.0f), dx);
    half dy1 = __hsub(__float2half(1.0f), dy);
    
    int idx00 = (y0 * in_w + x0) * 3 + c;
    int idx01 = (y0 * in_w + x1) * 3 + c;
    int idx10 = (y1 * in_w + x0) * 3 + c;
    int idx11 = (y1 * in_w + x1) * 3 + c;
    
    half v00 = src[idx00];
    half v01 = src[idx01];
    half v10 = src[idx10];
    half v11 = src[idx11];
    
    half top = __hfma(v00, dx1, __hmul(v01, dx));
    half bottom = __hfma(v10, dx1, __hmul(v11, dx));
    half interpolated = __hfma(top, dy1, __hmul(bottom, dy));
     
    half normalized = __hdiv(__hsub(interpolated, mean[c]), std[c]);
     
    int dst_idx = c * (out_h * out_w) + y * out_w + x;
    dst[dst_idx] = normalized;
}
''', 'cutlass_fused_resize_normalize_transpose_3c_fp16', options=options, backend=backend)

def fused_resize_normalize_transpose_3c(img_cp, out_h, out_w, mean, std, dtype, block_size):
    '''
    Fused resize + normalize + HWC→CHW transpose for 3-channel images.
    Supports both float32 and float16 (half) precision.
    '''
    h, w, c = img_cp.shape
    assert c == 3, "Only 3-channel images supported"

    img_cp = cp.ascontiguousarray(img_cp).astype(dtype, copy=False)
    out = cp.empty((3, out_h, out_w), dtype=dtype)
    
    grid = (
        (out_w + block_size[0] - 1) // block_size[0],
        (out_h + block_size[1] - 1) // block_size[1],
        (c + block_size[2] - 1) // block_size[2]
    )

    if dtype == cp.float16:
        fused_resize_normalize_transpose_kernel_low_fp16(
            grid, block_size, (img_cp, out, h, w, out_h, out_w, mean, std)
        )
    else:
        fused_resize_normalize_transpose_kernel_low(
            grid, block_size, (img_cp, out, h, w, out_h, out_w, mean, std)
        )
    
    return out