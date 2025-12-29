import cupy as cp
from warpcv.collection.standard.single import *
from warpcv.collection.standard.fused import *

import tensorrt as trt
from nvtx import push_range, pop_range

import pycuda.driver as cuda
import pycuda.autoinit

from cupy.cuda import Device
Device(0).use() # this avoids conflict with PyCUDA context

class TRT_MDE:
    def __init__(self, trt_path):
        self.logger = trt.Logger(trt.Logger.WARNING) 
        with open(trt_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.gpu_block2c = (32, 16)
        self.gpu_block3c = (32, 16, 1)  

        self.mean = cp.array([0.485, 0.456, 0.406], dtype=cp.float32)
        self.std = cp.array([0.229, 0.224, 0.225], dtype=cp.float32)
         
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.output_bindings = []
 
        self.stream = cuda.Stream()
        self.stream_ptr = self.stream.handle

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            shape = self.engine.get_tensor_shape(tensor_name)
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype, mem_flags=cuda.host_alloc_flags.WRITECOMBINED)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append({
                    'host': host_mem,
                    'device': device_mem,
                    'name': tensor_name,
                    'shape': shape,
                    'dtype': dtype
                })
            else:
                self.outputs.append({
                    'host': host_mem,
                    'device': device_mem,
                    'name': tensor_name,
                    'shape': shape,
                    'dtype': dtype
                })
                self.output_bindings.append(int(device_mem))
         
        self.input_shape = self.inputs[0]['shape']
        self.input_h = self.input_shape[2] if self.input_shape[2] > 0 else 518
        self.input_w = self.input_shape[3] if self.input_shape[3] > 0 else 518

        self.cupy_stream = cp.cuda.ExternalStream(self.stream_ptr) # operate CuPy with PyCUDA stream
        
        # Graph objects
        self.preprocess_graph = None
        self.normalize_graph = None
        self.postprocess_graph = None

        # Persistent buffers for resize
        self.resize_input_buffer = None
        self.resize_output_buffer = None
        
        # Persistent buffers for preprocess
        self.preprocess_input_buffer = None
        self.preprocess_output_buffer = None
        
        # Persistent buffers for postprocess
        self.postprocess_input_buffer = None
        self.postprocess_output_buffer = None
        self.original_h = None
        self.original_w = None

    def _resize(self, img_cp, fused):
        push_range("Resize Kernel Func.")
        
        with self.cupy_stream:
            if self.resize_input_buffer is None:
                self.resize_input_buffer = cp.empty_like(img_cp)
                self.resize_output_buffer = cp.empty(
                    (self.input_h, self.input_w, 3), 
                    dtype=cp.float32
                )
            cp.copyto(self.resize_input_buffer, img_cp)
            
            if self.preprocess_graph is None:
                # print("Capturing resize graph...")
                self.cupy_stream.begin_capture()
                if fused:
                    self.resize_output_buffer = fused_bgr2rgb_resize_3c(
                        self.resize_input_buffer, 
                        self.input_h, 
                        self.input_w, 
                        cp.float32,
                        self.gpu_block3c
                    )
                else:
                    resized_cp = cupy_resize_3c(
                        self.resize_input_buffer, 
                        self.input_w, 
                        self.input_h, 
                        cp.float32,
                        self.gpu_block3c
                    )
                    self.resize_output_buffer = cupy_cvt_bgr2rgb_float(
                        resized_cp, 
                        cp.float32,
                        self.gpu_block3c
                    )
                self.preprocess_graph = self.cupy_stream.end_capture()
            else:
                # print("Replaying resize graph...")
                self.preprocess_graph.launch(self.cupy_stream)
        
        pop_range()
        return self.resize_output_buffer

    def _preprocess(self, img_cp, fused):
        push_range("Preprocess Kernel Func.")
        
        with self.cupy_stream:
            if self.preprocess_input_buffer is None:
                self.preprocess_input_buffer = cp.empty_like(img_cp)
                self.preprocess_output_buffer = cp.empty(
                    (1, 3, self.input_h, self.input_w), 
                    dtype=cp.float32
                )

            cp.copyto(self.preprocess_input_buffer, img_cp)
            
            if self.normalize_graph is None:
                # print("Capturing normalize graph...")
                self.cupy_stream.begin_capture()
                if fused:
                    self.preprocess_output_buffer = cust_mde_nhwc_nchw(
                        self.preprocess_input_buffer,
                        self.mean,
                        self.std,
                        self.gpu_block3c
                    )
                else:
                    temp = self.preprocess_input_buffer / 255.0
                    temp = (temp - self.mean) / self.std
                    temp = cp.transpose(temp, (2, 0, 1))
                    self.preprocess_output_buffer = cp.expand_dims(temp, axis=0)
                self.normalize_graph = self.cupy_stream.end_capture()
            else:
                # print("Replaying normalize graph...")
                self.normalize_graph.launch(self.cupy_stream)
        
        pop_range()
        self.cp_wait()
        return self.preprocess_output_buffer
    
    def _trt2cp2trt(self, output_shape):
        # this function implements zero copy operations between CuPy and TensorRT
        # the original operations uses memcpy between GPU and CPU which is slow
        push_range("Cupy Wrapper for TRT CTX Func.")
        output_mem = cp.cuda.UnownedMemory(
            int(self.outputs[0]['device']),
            self.outputs[0]['host'].nbytes,
            owner=None
        )
        output_ptr = cp.cuda.MemoryPointer(output_mem, 0)
        depth_cp = cp.ndarray(
            output_shape,
            dtype=self.outputs[0]['dtype'],
            memptr=output_ptr
        )
        pop_range()
        return depth_cp

    def _postprocess(self, input_image, depth_cp, batched=False):
        if not batched:
            depth_cp = depth_cp[0]   

        push_range("Postprocess Func.")
        
        original_h, original_w = input_image.shape[:2]
        
        with self.cupy_stream:
            # allocate persistent buffers on first call or if dimensions change
            if (self.postprocess_input_buffer is None or 
                self.original_h != original_h or 
                self.original_w != original_w):
                
                self.original_h = original_h
                self.original_w = original_w
                self.postprocess_input_buffer = cp.empty_like(depth_cp)
                self.postprocess_output_buffer = cp.empty(
                    (original_h, original_w), 
                    dtype=cp.float32
                )
                # reset graph since dimensions changed
                self.postprocess_graph = None
        
            depth_contiguous = cp.ascontiguousarray(depth_cp.astype(cp.float32))
            cp.copyto(self.postprocess_input_buffer, depth_contiguous)
            
            if self.postprocess_graph is None:
                # print("Capturing postprocess graph...")
                self.cupy_stream.begin_capture()
                self.postprocess_output_buffer = cupy_resize_2c(
                    self.postprocess_input_buffer, 
                    self.original_h, 
                    self.original_w, 
                    cp.float32,
                    self.gpu_block2c
                )
                self.postprocess_graph = self.cupy_stream.end_capture()
            else:
                # print("Replaying postprocess graph...")
                self.postprocess_graph.launch(self.cupy_stream)
        
        pop_range()
        self.cp_wait()
        return cp.asnumpy(self.postprocess_output_buffer)
    
    def cp_wait(self):
        self.cupy_stream.synchronize()

    def infer(self, input_image: cp.asnumpy):
        push_range("MDE Inference", color="red")

        if isinstance(input_image, cp.ndarray):
            img_cp = input_image
        else:
            img_cp = cp.asarray(input_image)

        rgb_cp = self._resize(img_cp, fused=True)
        img_cp = self._preprocess(rgb_cp, fused=True)

        img_flat = img_cp.ravel()
            
        # set input output bindings
        self.context.set_tensor_address(self.inputs[0]['name'], img_flat.data.ptr)
        for i, out in enumerate(self.outputs):
            self.context.set_tensor_address(out['name'], self.output_bindings[i])
        
        # inference
        self.context.execute_async_v3(stream_handle=self.stream_ptr)
        output_shape = self.context.get_tensor_shape(self.outputs[0]['name'])

        depth_cp = self._trt2cp2trt(output_shape)
        depth_result = self._postprocess(input_image, depth_cp)

        pop_range()
        return depth_result