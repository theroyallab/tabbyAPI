from __future__ import annotations

from dataclasses import dataclass

import torch
import os, glob
import numpy as np
import json
import mmap
from ..util import Timer, cuda_sync_active
from ..ext import exllamav3_ext as ext
from functools import lru_cache

MAX_DEFERRED_LOAD_CHUNK = 2*1024**2

def convert_dtype(dt: str):
    if dt == "I32": return torch.int, np.int32, 4
    elif dt == "I16": return torch.short, np.int16, 2
    elif dt == "F16": return torch.float16, np.float16, 2
    elif dt == "BF16": return torch.bfloat16, np.float16, 2
    elif dt == "F32": return torch.float, np.float32, 4
    else:
        raise ValueError(f"Unknown dtype {dt}")


def read_header(filename: str) -> dict:
    with open(filename, "rb") as fp:
        header_size = np.fromfile(fp, dtype = np.int64, count = 1).item()
        header_json = fp.read(header_size)
        header = json.loads(header_json.decode("utf-8"))
        header["_header_offset"] = fp.tell()
        return header


@dataclass
class STCMetrics:
    bytes_loaded: int = 0
    time_elapsed: float = 0.0
    deferred_tensors: int = 0
    deferred_passes: int = 0
    direct_tensors: int = 0
    total_chunks: int = 0
    def bandwidth(self):
        return self.bytes_loaded / (1024 ** 3) / self.time_elapsed
    def print(self):
        print(f" -- Total size: {self.bytes_loaded:,} bytes, {self.bytes_loaded / 1024**3:.2f} GB")
        print(f" -- Load time: {self.time_elapsed:.3f} seconds")
        print(f" -- Bandwidth: {self.bandwidth():.3f} GB / s")
        print(f" -- Deferred: {self.deferred_tensors:,} tensors in {self.deferred_passes:,} passes, {self.total_chunks:,} chunks")
        print(f" -- Direct: {self.direct_tensors:,} tensors")


class SafetensorsCollection:

    def __init__(
        self,
        directory: str,
        load_method: str | None = None
    ):
        """
        Scan directory for .safetensors files and build collection, preparing to load tensors indexed by key.

        :param directory:
            Directory to scan.

        :param load_method:
            - "mt_fread": multithreaded C++ loader using fread
            - "python": use fp.seek() and fp.read() to load tensor data via bytearray and torch.frombuffer
        """

        self.directory = directory
        self.tensor_file_map = {}
        self.file_headers = {}
        self.handles: dict[str, list | None] = {}
        self.load_method = load_method or "mt_fread"

        self.metrics = STCMetrics()

        self.tensor_files = []
        self.add_tensor_files(directory)

        self.new_tensors = None
        self.deferred_mode = False
        self.deferred_loads = []


    def add_tensor_files(
        self,
        directory: str,
        warn_if_override: bool = True
    ):
        st_pattern = os.path.join(directory, "*.safetensors")
        new_tensor_files = glob.glob(st_pattern)
        self.tensor_files += new_tensor_files

        overrides = 0
        for st_file in new_tensor_files:
            self.handles[st_file] = None
            header = read_header(st_file)
            self.file_headers[st_file] = header
            for key in header.keys():
                if key in ["__metadata__", "_header_offset"]:
                    continue
                if key in self.tensor_file_map and warn_if_override:
                    # print(f" !! Overriding {key} from {self.tensor_file_map[key]} with f{st_file}")
                    overrides += 1
                self.tensor_file_map[key] = st_file
        if overrides:
            print(f" !! Replaced {overrides} tensors from {directory}")


    def has_tensor(
        self,
        key: str,
    ):
        if self.new_tensors and key in self.new_tensors:
            return True
        return key in self.tensor_file_map


    def has_tensor_group(
        self,
        key: str,
        subkeys: list,
    ):
        sources = [self.tensor_file_map]
        if self.new_tensors:
            sources += [self.new_tensors]
        return any(
            all(
                (
                    f"{key}.{subkey}" in source if isinstance(subkey, str) else
                    any(f"{key}.{sk}" in source for sk in subkey)
                ) for subkey in subkeys
            ) for source in sources
        )


    def get_tensor_sizes(
        self,
        prefix: str,
    ):
        assert self.new_tensors is None  # TODO
        keys = [
            key for key in self.tensor_file_map.keys()
            if key == prefix or key.startswith(prefix + ".")
        ]
        sizes = [self.get_tensor_size(key) for key in keys]
        return sizes


    def get_tensor_size(
        self,
        key: str,
        optional: bool = False
    ):
        assert self.new_tensors is None  # TODO
        if not key in self.tensor_file_map:
            if not optional:
                raise ValueError(f"Required tensor {key} not found in any *.safetensors file in {self.directory}")
            else:
                return 0

        filename = self.tensor_file_map[key]
        header = self.file_headers[filename]
        h = header[key]
        # _, _, esize = convert_dtype(h["dtype"])
        # bytesize = np.prod(h["shape"]) * esize
        beg, end = h["data_offsets"]
        bytesize = end - beg
        return bytesize


    def list_tensors(
        self,
        prefix: str,
    ) -> dict:
        assert self.new_tensors is None  # TODO
        keys = [
            key for key in self.tensor_file_map.keys()
            if key == prefix or key.startswith(prefix + ".")
        ]
        results = {}
        for key in keys:
            filename = self.tensor_file_map[key]
            header = self.file_headers[filename]
            h = header[key]
            dtype, np_dtype, esize = convert_dtype(h["dtype"])
            beg, end = h["data_offsets"]
            results[key] = {
                "shape": h["shape"],
                "n_bytes": end - beg,
                "dtype": str(dtype),
            }
        return results


    # TODO: deferred load
    def get_tensors(
        self,
        prefix: str,
        device: torch.device | None = None,
        allow_bf16: bool = False,
    ) -> dict:
        assert self.new_tensors is None  # TODO
        keys = [
            key for key in self.tensor_file_map.keys()
            if key == prefix or key.startswith(prefix + ".")
        ]
        result = {key: self.get_tensor(key, device, allow_bf16 = allow_bf16) for key in keys}
        return result


    # TODO: deferred load
    def get_tensor(
        self,
        key: str,
        device: torch.device | None = None,
        optional: bool = False,
        allow_bf16: bool = False,
        float2half: bool = False,
        no_defer: bool = False,
        transpose: bool = False,
        pad_to: tuple = None,
    ) -> torch.Tensor | None:

        if device is None:
            device = torch.device("cpu")

        if self.new_tensors and key in self.new_tensors:
            tensor = self.new_tensors[key].to(device)
            if transpose:
                tensor = tensor.T.contiguous()
            return tensor

        if not key in self.tensor_file_map:
            if not optional:
                raise ValueError(f"Required tensor {key} not found in any *.safetensors file in {self.directory}")
            else:
                return None

        filename = self.tensor_file_map[key]
        header = self.file_headers[filename]
        h = header[key]
        offset = header["_header_offset"]

        dtype, np_dtype, esize = convert_dtype(h["dtype"])
        beg, end = h["data_offsets"]
        bytesize = end - beg
        shape = h["shape"]
        numel = np.prod(shape)
        assert numel * esize == bytesize, \
            f"Incorrect size of {key} in {filename}"

        load_method = self.load_method
        if load_method == "mt_fread" and self.deferred_mode and not no_defer:
            load_method = "defer"

        with (Timer() as timer):
            match load_method:
                case "defer":
                    h = self.handles[filename]
                    if not h:
                        try:
                            h = ext.stloader_open_file(filename)
                            self.handles[filename] = h
                        except RuntimeError as e:
                            print(f" ## Error opening {filename}")
                            raise e
                    bf16_to_fp16 = (dtype == torch.bfloat16 and not allow_bf16)
                    fp32_to_fp16 = (dtype == torch.float and float2half)
                    load_shape = tuple(shape)
                    load_shape_t = load_shape if not transpose else (shape[1], shape[0])
                    load_dtype = dtype
                    if bf16_to_fp16 and load_dtype == torch.bfloat16:
                        load_dtype = torch.half
                    final_shape = pad_to if pad_to is not None else load_shape_t
                    final_dtype = dtype if not (bf16_to_fp16 or fp32_to_fp16) else torch.float16
                    if final_shape == load_shape_t:
                        tensor = torch.empty(final_shape, dtype = final_dtype, device = device)
                    else:
                        tensor = torch.zeros(final_shape, dtype = final_dtype, device = device)
                    if transpose or fp32_to_fp16 or final_shape != load_shape_t:
                        temp_tensor = torch.empty(load_shape, dtype = load_dtype, device = device)
                    else:
                        temp_tensor = None
                    self.deferred_loads.append({
                        "filename": filename,
                        "file_offset": offset + beg,
                        "bytesize": bytesize,
                        "temp_tensor": temp_tensor,
                        "dest_tensor": tensor,
                        "bf16_to_fp16": bf16_to_fp16,
                        "fp32_to_fp16": fp32_to_fp16,
                        "cuda": tensor.is_cuda,
                        "device_id": tensor.device.index if tensor.is_cuda else -1,
                        "transpose": transpose,
                    })
                    self.metrics.deferred_tensors += 1

                case "mt_fread":
                    h = self.handles[filename]
                    if not h:
                        try:
                            h = ext.stloader_open_file(filename)
                            self.handles[filename] = h
                        except RuntimeError as e:
                            print(f" ## Error opening {filename}")
                            raise e
                    tensor = torch.empty(shape, dtype = dtype, device = device)
                    assert tensor.is_contiguous()
                    if device != "cpu":
                        cuda_sync_active()
                    ext.stloader_read(
                        h,
                        offset + beg,
                        bytesize,
                        tensor,
                    )
                    if tensor.dtype == torch.bfloat16 and not allow_bf16:
                        tensor = tensor.to(torch.float16)
                    if tensor.dtype == torch.float and float2half:
                        tensor = tensor.to(torch.float16)
                    if transpose:
                        tensor = tensor.T
                    if pad_to is not None:
                        padded = torch.zeros(pad_to, dtype = tensor.dtype, device = tensor.device)
                        padded[tuple(slice(0, s) for s in tensor.shape)].copy_(tensor)
                        tensor = padded
                    tensor = tensor.contiguous()
                    self.metrics.direct_tensors += 1


                case "python":
                    with open(filename, "rb") as fp:
                        fp.seek(offset + beg)
                        buffer = bytearray(fp.read(bytesize))
                        tensor = torch.frombuffer(buffer, dtype = dtype, count = numel).reshape(shape)
                        if tensor.dtype == torch.bfloat16 and not allow_bf16:
                            tensor = tensor.to(torch.float16)
                        if tensor.dtype == torch.float and float2half:
                            tensor = tensor.to(torch.float16)
                        if transpose:
                            tensor = tensor.T
                        if pad_to is not None:
                            padded = torch.zeros(pad_to, dtype = tensor.dtype, device = tensor.device)
                            padded[tuple(slice(0, s) for s in tensor.shape)].copy_(tensor)
                            tensor = padded
                        tensor = tensor.to(device).contiguous()
                    self.metrics.direct_tensors += 1

                case _:
                    raise ValueError(f"Invalid load_method: {load_method}")

        self.metrics.bytes_loaded += bytesize
        self.metrics.time_elapsed += timer.interval

        return tensor


    def close(self):
        assert self.new_tensors is None
        for filename, h in self.handles.items():
            if h:
                ext.stloader_close_file(h)
                self.handles[filename] = None


    @lru_cache
    def max_key_len(self):
        l = max(len(k) for k in self.tensor_file_map.keys())
        return l


    def set_new_tensors(self, new_tensors):
        self.new_tensors = new_tensors


    def begin_deferred_load(self):
        assert not self.deferred_mode
        self.deferred_mode = True


    def end_deferred_load(self):
        assert self.deferred_mode

        with (Timer() as timer):

            cpu_loads = {}
            cuda_loads = {}
            for load in self.deferred_loads:
                filenmame = load["filename"]
                cuda = load["cuda"]
                if cuda:
                    if not filenmame in cuda_loads:
                        cuda_loads[filenmame] = []
                    cuda_loads[filenmame].append(load)
                else:
                    if not filenmame in cpu_loads:
                        cpu_loads[filenmame] = []
                    cpu_loads[filenmame].append(load)

            def make_workload(l):
                wl = []
                for w in l:
                    if w["temp_tensor"] is not None:
                        dst = w["temp_tensor"].data_ptr()
                    else:
                        # Not transposing, padding or converting fp32->fp16, load directly
                        dst = w["dest_tensor"].data_ptr()
                    bytesize = w["bytesize"]
                    src = w["file_offset"]
                    while bytesize > 0:
                        j = ext.TensorLoadJob(
                            self.handles[w["filename"]],
                            src,
                            min(bytesize, MAX_DEFERRED_LOAD_CHUNK),
                            dst,
                            w["bf16_to_fp16"],
                            w["fp32_to_fp16"],
                            w["cuda"],
                            w["device_id"]
                        )
                        src += MAX_DEFERRED_LOAD_CHUNK
                        dst += MAX_DEFERRED_LOAD_CHUNK
                        bytesize -= MAX_DEFERRED_LOAD_CHUNK
                        wl.append(j)
                return wl

            for filename, loads in cpu_loads.items():
                loads = sorted(loads, key = lambda c: -c["bytesize"])
                workload = make_workload(loads)
                self.metrics.total_chunks += len(workload)
                ext.stloader_deferred_cpu(workload)
                for w in loads:
                    if w["temp_tensor"] is not None:
                        src = w["temp_tensor"]
                        if w["transpose"]:
                            src = src.T
                        unpadded_idx = tuple(slice(0, s) for s in src.shape)
                        w["dest_tensor"][unpadded_idx].copy_(src)

            for filename, loads in cuda_loads.items():
                loads = sorted(loads, key = lambda c: -c["bytesize"])
                workload = make_workload(loads)
                self.metrics.total_chunks += len(workload)
                ext.stloader_deferred_cuda(workload, MAX_DEFERRED_LOAD_CHUNK)
                for w in loads:
                    if w["temp_tensor"] is not None:
                        src = w["temp_tensor"]
                        if w["transpose"]:
                            src = src.T
                        unpadded_idx = tuple(slice(0, s) for s in src.shape)
                        w["dest_tensor"][unpadded_idx].copy_(src)

        self.metrics.time_elapsed += timer.interval
        self.metrics.deferred_passes += 1

        self.deferred_mode = False
        self.deferred_loads = []


    def abort_deferred_load(self):
        self.deferred_mode = False
        self.deferred_loads = []


class VariantSafetensorsCollection:

    def __init__(
        self,
        tensor_map: dict[str, str],
        **kwargs
    ):
        self.tensor_map = None
        self.tensor_map_sort = None
        self.all_dirs = None
        self.stcs = {}
        self.kwargs = kwargs
        self.update_map(tensor_map)


    def update_map(
        self,
        tensor_map: dict[str, str]
    ):
        self.tensor_map = tensor_map
        self.tensor_map_sort = sorted(tensor_map.items(), key = lambda kv: len(kv[0]), reverse = True)
        all_dirs = list(set(tensor_map.values()))

        for d in all_dirs:
            if d not in self.stcs:
                self.stcs[d] = SafetensorsCollection(directory = d, **self.kwargs)


    def has_tensor(
        self,
        key: str,
    ):
        return any(key in stc.tensor_file_map for stc in self.stcs.values())


    def has_tensor_group(
        self,
        key: str,
        subkeys: list[str],
    ):
        return all(
            any(f"{key}.{subkey}" in stc.tensor_file_map for stc in self.stcs.values())
            for subkey in subkeys
        )


    def get_tensor(
        self,
        key: str,
        device: torch.device | None = None,
        optional: bool = False,
        allow_bf16: bool = False
    ) -> torch.Tensor | None:

        file = None
        for k, v in self.tensor_map_sort:
            if key.startswith(k):
                file = v
                break
        if file is None:
            if not optional:
                raise ValueError(f"No prefix found in variants map with the matching key: {key}")
            else:
                return None

        return self.stcs[file].get_tensor(key, device, optional, allow_bf16)


    def close(self):
        for stc in self.stcs.values():
            stc.close()


    def get_metrics(self):
        res = [stc.get_metrics() for stc in self.stcs.values()]
        bytes_loaded = sum(r[0] for r in res)
        time_elapsed = sum(r[1] for r in res)
        bandwidth = bytes_loaded / (1024**3) / time_elapsed
        return bytes_loaded, time_elapsed, bandwidth
