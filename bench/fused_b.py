import time
import numpy as np
import cupy as cp
import cv2
import sys
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir)) 

import os
import json
os.environ['WCV_COMPILE_OPTIONS'] = json.dumps(["-O3", "--use_fast_math"])
os.environ['WCV_BACKEND'] = 'nvcc'
from warpcv.collection.standard.fused import *
 
INPUT_SIZES = [(480, 640), (720, 1280), (1080, 1920), (2160, 3840)]
OUTPUT_SIZES = [(224, 224), (256, 256), (512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]
BLOCK_SIZE = (32, 32, 1)
BLOCK_SIZE_4K = (32, 32, 1)
WARMUP_ITERS = 5
BENCH_ITERS = 20
CUDA_AVAILABLE = cv2.cuda.getCudaEnabledDeviceCount() > 0 if hasattr(cv2, 'cuda') else False

def get_gpu_memory_mb():
    try:
        meminfo = cp.cuda.runtime.memGetInfo()
        free_mem, total_mem = meminfo
        return (total_mem - free_mem) / (1024 ** 2)  
    except:
        return 0

def Time(func, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    for _ in range(warmup):
        func()
        if CUDA_AVAILABLE: 
            cp.cuda.Stream.null.synchronize()
     
    if CUDA_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        cp.cuda.Stream.null.synchronize()
    
    times = []
    vram_usages = []
    
    for _ in range(iters):
        if CUDA_AVAILABLE: 
            cp.cuda.Stream.null.synchronize()
            baseline_vram = get_gpu_memory_mb()
        
        start = time.perf_counter()
        result = func()
        
        if CUDA_AVAILABLE: 
            cp.cuda.Stream.null.synchronize()
            peak_vram = get_gpu_memory_mb()
            vram_usages.append(peak_vram - baseline_vram)
        
        times.append((time.perf_counter() - start) * 1000)
         
        del result
        if CUDA_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
            cp.cuda.Stream.null.synchronize()
    
    return np.mean(times), np.max(vram_usages) if vram_usages else 0
 
def benchmark(img, out_size, pipelines):
    img_cp = cp.array(img)
    out_h, out_w = out_size
    results = {}
    vram_results = {}

    for name, func in pipelines.items():
        avg_time, peak_vram = Time(lambda: func(img, img_cp, out_h, out_w))
        results[name] = avg_time
        if name != 'cv2_cpu':
            vram_results[name] = peak_vram
    
    return results, vram_results
 
def bgr2rgb_cpu(img, img_cp, out_h, out_w):
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (out_w, out_h))

def bgr2rgb_cuda(img, img_cp, out_h, out_w):
    gpu = cv2.cuda_GpuMat()
    gpu.upload(img)
    gpu = cv2.cuda.cvtColor(gpu, cv2.COLOR_BGR2RGB)
    gpu = cv2.cuda.resize(gpu, (out_w, out_h))
    return gpu.download()

def bgr2rgb_cupy(img, img_cp, out_h, out_w):
    rgb = img_cp[:, :, ::-1]
    y_idx = cp.linspace(0, img_cp.shape[0]-1, out_h).astype(cp.int32)
    x_idx = cp.linspace(0, img_cp.shape[1]-1, out_w).astype(cp.int32)
    y_grid, x_grid = cp.meshgrid(y_idx, x_idx, indexing='ij')
    return rgb[y_grid, x_grid]

def bgr2rgb_custom(img, img_cp, out_h, out_w, block_size=BLOCK_SIZE):
    return fused_bgr2rgb_resize_3c(img_cp, out_h, out_w, cp.float32, block_size)

def rnt_cpu(img, img_cp, out_h, out_w):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255
    resized = cv2.resize(img, (out_w, out_h))
    normalized = (resized - mean) / std
    return np.transpose(normalized, (2, 0, 1))

def rnt_cuda(img, img_cp, out_h, out_w):
    mean = cp.array([0.485, 0.456, 0.406], dtype=cp.float32) * 255
    std = cp.array([0.229, 0.224, 0.225], dtype=cp.float32) * 255
    gpu = cv2.cuda_GpuMat()
    gpu.upload(img)
    gpu = cv2.cuda.resize(gpu, (out_w, out_h))
    resized = cp.asarray(gpu.download())
    normalized = (resized - mean) / std
    return cp.transpose(normalized, (2, 0, 1))

def rnt_cupy(img, img_cp, out_h, out_w):
    mean = cp.array([0.485, 0.456, 0.406], dtype=cp.float32) * 255
    std = cp.array([0.229, 0.224, 0.225], dtype=cp.float32) * 255
    y_idx = cp.linspace(0, img_cp.shape[0]-1, out_h).astype(cp.int32)
    x_idx = cp.linspace(0, img_cp.shape[1]-1, out_w).astype(cp.int32)
    y_grid, x_grid = cp.meshgrid(y_idx, x_idx, indexing='ij')
    resized = img_cp[y_grid, x_grid]
    normalized = (resized - mean) / std
    return cp.transpose(normalized, (2, 0, 1))

def rnt_custom(img, img_cp, out_h, out_w, block_size=BLOCK_SIZE):
    mean = cp.array([0.485, 0.456, 0.406], dtype=cp.float32) * 255
    std = cp.array([0.229, 0.224, 0.225], dtype=cp.float32) * 255
    return fused_resize_normalize_transpose_3c(img_cp, out_h, out_w, mean, std, cp.float32, block_size)
 
def plot(all_results_bgr2rgb, all_results_rnt, all_vram_bgr2rgb, all_vram_rnt, title):
    methods = list(all_results_bgr2rgb[0][2].keys())
    labels = [f"{in_h}x{in_w}→{out_h}x{out_w}" for (in_h, in_w), (out_h, out_w), _, _ in all_results_bgr2rgb]
     
    color_map = {
        'warpcv': '#EF4444',   
        'cv2_cpu': '#3B82F6',    
        'cv2_cuda': '#10B981',    
        'cupy': '#F59E0B'         
    }
    marker_map = {
        'warpcv': 'circle',
        'cv2_cpu': 'square',
        'cv2_cuda': 'diamond',
        'cupy': 'triangle-up'
    }
    
    fig = go.Figure()
    speedups_text = []
    all_times = []
    
    for m in methods:
        times = []
        for bgr_res, rnt_res in zip(all_results_bgr2rgb, all_results_rnt):
            bgr_time = bgr_res[2].get(m, 0)
            rnt_time = rnt_res[2].get(m, 0)
            avg_time = (bgr_time + rnt_time) / 2
            times.append(avg_time)
        
        all_times.extend(times)
         
        line_width = 3 if m == 'warpcv' else 2
        line_dash = 'solid' if m == 'warpcv' else 'dash'
        
        fig.add_trace(go.Scatter(
            x=list(range(len(labels))),
            y=times,
            name=m,
            mode='lines+markers',
            line=dict(
                color=color_map.get(m, '#6B7280'),
                width=line_width,
                dash=line_dash
            ),
            marker=dict(
                symbol=marker_map.get(m, 'circle'),
                size=10,
                color=color_map.get(m, '#6B7280'),
                line=dict(width=1, color='white')
            )
        ))
        
        if m != 'warpcv':
            warpcv_times = []
            for bgr_res, rnt_res in zip(all_results_bgr2rgb, all_results_rnt):
                warpcv_time = (bgr_res[2]['warpcv'] + rnt_res[2]['warpcv']) / 2
                warpcv_times.append(warpcv_time)
            avg_improvement_ms = np.mean(np.array(times) - np.array(warpcv_times))
            avg_improvement_us = avg_improvement_ms * 1000  
            speedups_text.append(f"<b>warpcv</b> vs <b>{m}</b>: <b>({avg_improvement_us:.0f} µs less avg </b> latency)")
     
    y_min = min(all_times)
    y_max = max(all_times)
    y_middle = (y_min + y_max) / 2
     
    fig.update_layout(
        title={
            'text': title,
            'font': {'size': 24, 'color': '#1F2937', 'family': 'Arial, sans-serif'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(labels))),
            ticktext=labels,
            tickangle=-45,
            title='Resolution (Input → Output)',
            title_font={'size': 14, 'color': '#374151'},
            gridcolor='#E5E7EB',
            showgrid=True
        ),
        yaxis=dict(
            title='Average Time (ms)',
            title_font={'size': 14, 'color': '#374151'},
            gridcolor='#E5E7EB',
            showgrid=True
        ),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#F9FAFB',
        font={'family': 'Arial, sans-serif', 'color': '#374151'},
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='#D1D5DB',
            borderwidth=1,
            font={'size': 12}
        ),
        hovermode='x unified',
        width=1400,
        height=700,
        margin=dict(l=80, r=80, t=100, b=120)
    )
     
    speedup_str = "<br>".join(speedups_text)
    fig.add_annotation(
        text=speedup_str,
        xref='paper',
        yref='y',
        x=0.02,
        y=y_middle,
        xanchor='left',
        yanchor='middle',
        showarrow=False,
        align='left',
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='#D1D5DB',
        borderwidth=1,
        borderpad=10,
        font={'size': 14, 'color': '#374151'}
    )
    
    fig.show()

def plot_vram_usage(all_vram_bgr2rgb, all_vram_rnt, title):
    """Plot VRAM usage for GPU-accelerated methods"""
    if not all_vram_bgr2rgb:
        return
     
    gpu_methods = [m for m in all_vram_bgr2rgb[0][3].keys() if m != 'cv2_cpu']
    if not gpu_methods:
        print("No GPU methods with VRAM data to plot")
        return
        
    labels = [f"{in_h}x{in_w}→{out_h}x{out_w}" for (in_h, in_w), (out_h, out_w), _, _ in all_vram_bgr2rgb]
    
    color_map = {
        'warpcv': '#EF4444',
        'cv2_cuda': '#10B981',
        'cupy': '#F59E0B'
    }
    marker_map = {
        'warpcv': 'circle',
        'cv2_cuda': 'diamond',
        'cupy': 'triangle-up'
    }
    
    fig = go.Figure()
    
    for m in gpu_methods:
        vram_usages = []
        for bgr_res, rnt_res in zip(all_vram_bgr2rgb, all_vram_rnt):
            bgr_vram = bgr_res[3].get(m, 0)
            rnt_vram = rnt_res[3].get(m, 0)
            avg_vram = (bgr_vram + rnt_vram) / 2
            vram_usages.append(avg_vram)
        
        line_width = 3 if m == 'warpcv' else 2
        line_dash = 'solid' if m == 'warpcv' else 'dash'
        
        fig.add_trace(go.Scatter(
            x=list(range(len(labels))),
            y=vram_usages,
            name=m,
            mode='lines+markers',
            line=dict(
                color=color_map.get(m, '#6B7280'),
                width=line_width,
                dash=line_dash
            ),
            marker=dict(
                symbol=marker_map.get(m, 'circle'),
                size=10,
                color=color_map.get(m, '#6B7280'),
                line=dict(width=1, color='white')
            ),
            hovertemplate='<b>%{fullData.name}</b><br>VRAM: %{y:.2f}MB<extra></extra>'
        ))
    
    fig.update_layout(
        title={
            'text': title,
            'font': {'size': 24, 'color': '#1F2937', 'family': 'Arial, sans-serif'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(labels))),
            ticktext=labels,
            tickangle=-45,
            title='Resolution (Input → Output)',
            title_font={'size': 14, 'color': '#374151'},
            gridcolor='#E5E7EB',
            showgrid=True
        ),
        yaxis=dict(
            title='VRAM Usage (MB)',
            title_font={'size': 14, 'color': '#374151'},
            gridcolor='#E5E7EB',
            showgrid=True
        ),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#F9FAFB',
        font={'family': 'Arial, sans-serif', 'color': '#374151'},
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='#D1D5DB',
            borderwidth=1,
            font={'size': 12}
        ),
        hovermode='x unified',
        width=1400,
        height=700,
        margin=dict(l=80, r=80, t=100, b=120)
    )
    
    fig.show()

def plot_avg_latency_barplot(all_results_bgr2rgb, all_results_rnt, title):
    methods = list(all_results_bgr2rgb[0][2].keys())
     
    avg_latencies = {}
    for m in methods:
        times = []
        for bgr_res, rnt_res in zip(all_results_bgr2rgb, all_results_rnt):
            bgr_time = bgr_res[2].get(m, 0)
            rnt_time = rnt_res[2].get(m, 0)
            avg_time = (bgr_time + rnt_time) / 2
            times.append(avg_time)
        avg_latencies[m] = np.mean(times)
     
    sorted_methods = sorted(avg_latencies.items(), key=lambda x: x[1], reverse=True)
    sorted_names = [m[0] for m in sorted_methods]
    sorted_latencies = [m[1] for m in sorted_methods]
     
    color_map = {
        'warpcv': '#EF4444',       
        'cv2_cpu': '#3B82F6',    
        'cv2_cuda': '#10B981',    
        'cupy': '#F59E0B'
    }
    
    colors = [color_map.get(name, '#6B7280') for name in sorted_names]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=sorted_names,
        y=sorted_latencies,
        marker=dict(
            color=colors,
            line=dict(color='white', width=1)
        ),
        text=[f'{lat:.2f}ms' for lat in sorted_latencies],
        textposition='outside',
        textfont=dict(size=14, color='#374151'),
        hovertemplate='<b>%{x}</b><br>Avg Latency: %{y:.2f}ms<extra></extra>'
    ))
    
    if 'warpcv' in avg_latencies:
        warpcv_latency = avg_latencies['warpcv']
        annotations = []
        for i, (name, latency) in enumerate(sorted_methods):
            if name != 'warpcv':
                speedup = latency / warpcv_latency
                annotations.append(
                    dict(
                        x=name,
                        y=latency / 2,
                        text=f'{speedup:.2f}x<br>slower',
                        showarrow=False,
                        font=dict(size=12, color='white', family='Arial, sans-serif'),
                        bgcolor='rgba(0, 0, 0, 0.6)',
                        borderpad=4
                    )
                )
        fig.update_layout(annotations=annotations)
    
    fig.update_layout(
        title={
            'text': title,
            'font': {'size': 24, 'color': '#1F2937', 'family': 'Arial, sans-serif'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(
            title='_',
            title_font={'size': 14, 'color': '#374151'},
            tickfont={'size': 13, 'color': '#374151'}
        ),
        yaxis=dict(
            title='Average Latency (ms)',
            title_font={'size': 14, 'color': '#374151'},
            gridcolor='#E5E7EB',
            showgrid=True
        ),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#F9FAFB',
        font={'family': 'Arial, sans-serif', 'color': '#374151'},
        width=1000,
        height=600,
        margin=dict(l=80, r=80, t=100, b=80),
        showlegend=False
    )
    
    fig.show()

def plot_avg_vram_barplot(all_vram_bgr2rgb, all_vram_rnt, title):
    if not all_vram_bgr2rgb:
        return
    
    gpu_methods = [m for m in all_vram_bgr2rgb[0][3].keys() if m != 'cv2_cpu']
    if not gpu_methods:
        print("No GPU methods with VRAM data to plot")
        return
    
    avg_vram = {}
    for m in gpu_methods:
        vram_usages = []
        for bgr_res, rnt_res in zip(all_vram_bgr2rgb, all_vram_rnt):
            bgr_vram = bgr_res[3].get(m, 0)
            rnt_vram = rnt_res[3].get(m, 0)
            avg_vram_usage = (bgr_vram + rnt_vram) / 2
            vram_usages.append(avg_vram_usage)
        avg_vram[m] = np.mean(vram_usages)
    
    sorted_methods = sorted(avg_vram.items(), key=lambda x: x[1], reverse=True)
    sorted_names = [m[0] for m in sorted_methods]
    sorted_vram = [m[1] for m in sorted_methods]
    
    color_map = {
        'warpcv': '#EF4444',
        'cv2_cuda': '#10B981',
        'cupy': '#F59E0B'
    }
    
    colors = [color_map.get(name, '#6B7280') for name in sorted_names]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=sorted_names,
        y=sorted_vram,
        marker=dict(
            color=colors,
            line=dict(color='white', width=1)
        ),
        text=[f'{vram:.2f}MB' for vram in sorted_vram],
        textposition='outside',
        textfont=dict(size=14, color='#374151'),
        hovertemplate='<b>%{x}</b><br>Avg VRAM: %{y:.2f}MB<extra></extra>'
    ))
    
    if 'warpcv' in avg_vram:
        warpcv_vram = avg_vram['warpcv']
        annotations = []
        for _, (name, vram) in enumerate(sorted_methods):
            if name != 'warpcv':
                ratio = vram / warpcv_vram if warpcv_vram > 0 else 0
                annotations.append(
                    dict(
                        x=name,
                        y=vram / 2,
                        text=f'{ratio:.2f}x<br>more',
                        showarrow=False,
                        font=dict(size=12, color='white', family='Arial, sans-serif'),
                        bgcolor='rgba(0, 0, 0, 0.6)',
                        borderpad=4
                    )
                )
        fig.update_layout(annotations=annotations)
    
    fig.update_layout(
        title={
            'text': title,
            'font': {'size': 24, 'color': '#1F2937', 'family': 'Arial, sans-serif'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(
            title='_',
            title_font={'size': 14, 'color': '#374151'},
            tickfont={'size': 13, 'color': '#374151'}
        ),
        yaxis=dict(
            title='Average VRAM Usage (MB)',
            title_font={'size': 14, 'color': '#374151'},
            gridcolor='#E5E7EB',
            showgrid=True
        ),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#F9FAFB',
        font={'family': 'Arial, sans-serif', 'color': '#374151'},
        width=1000,
        height=600,
        margin=dict(l=80, r=80, t=100, b=80),
        showlegend=False
    )
    
    fig.show()

def main():    
    all_results_bgr2rgb = []
    all_results_rnt = []
    all_vram_bgr2rgb = []
    all_vram_rnt = []

    bgr2rgb_pipelines = {'cv2_cpu': bgr2rgb_cpu, 'cupy': bgr2rgb_cupy, 'warpcv': bgr2rgb_custom}
    if CUDA_AVAILABLE: bgr2rgb_pipelines['cv2_cuda'] = bgr2rgb_cuda

    rnt_pipelines = {'cv2_cpu': rnt_cpu, 'cupy': rnt_cupy, 'warpcv': rnt_custom}
    if CUDA_AVAILABLE: rnt_pipelines['cv2_cuda'] = rnt_cuda

    for in_h, in_w in INPUT_SIZES:
        for out_h, out_w in OUTPUT_SIZES:            
            img = np.random.rand(in_h, in_w, 3).astype(np.float32) * 255
 
            res_bgr2rgb, vram_bgr2rgb = benchmark(img, (out_h, out_w), bgr2rgb_pipelines)
            all_results_bgr2rgb.append(((in_h, in_w), (out_h, out_w), res_bgr2rgb, vram_bgr2rgb))
            all_vram_bgr2rgb.append(((in_h, in_w), (out_h, out_w), res_bgr2rgb, vram_bgr2rgb))
 
            res_rnt, vram_rnt = benchmark(img, (out_h, out_w), rnt_pipelines)
            all_results_rnt.append(((in_h, in_w), (out_h, out_w), res_rnt, vram_rnt))
            all_vram_rnt.append(((in_h, in_w), (out_h, out_w), res_rnt, vram_rnt))
    
     
    plot(all_results_bgr2rgb, all_results_rnt, all_vram_bgr2rgb, all_vram_rnt, "BGR2RGB + RNT Benchmark ")
    plot_avg_latency_barplot(all_results_bgr2rgb, all_results_rnt, "Avg. Latency")
     
    if CUDA_AVAILABLE:
        plot_vram_usage(all_vram_bgr2rgb, all_vram_rnt, "Intermediate (peak cumulative allocation) VRAM Usage Over Resolutions")
        plot_avg_vram_barplot(all_vram_bgr2rgb, all_vram_rnt, "Intermediate (per-iteration allocation) Avg. VRAM")

if __name__ == "__main__":
    main()