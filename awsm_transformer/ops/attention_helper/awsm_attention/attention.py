"""
Python wrapper around libattentionwrapper.so for flash attention forward/backward.

Usage:
    helper = FlashAttentionHelper()              # loads lib, detects arch
    helper.forward(q, k, v, out, softmax_lse,
                   q_seq_offsets, k_seq_offsets,
                   q_seq_lens, k_seq_lens,
                   max_seqlen_q, max_seqlen_k,
                   causal=True)
"""

from __future__ import annotations

import ctypes
import os
import pathlib
from ctypes import (
    POINTER,
    c_int,
    c_uint64,
    c_void_p,
    byref,
)
from typing import Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Locate the shared library
# ---------------------------------------------------------------------------

_LIB_NAME = "libattentionwrapper.so"
_LIB_SUBPACKAGE = "awsm_attention.lib"


def _find_lib() -> str:
    """Locate libattentionwrapper.so shipped inside the installed package.

    Resolution order:
        1. ``AWSM_ATTENTION_LIB`` environment variable (explicit override).
        2. ``awsm_attention/lib/`` via :mod:`importlib.resources` (Python ≥ 3.9)
           or :mod:`importlib_resources` backport / ``pkg_resources`` fallback.

    Raises
    ------
    FileNotFoundError
        If the library cannot be found through any of the above methods.
    """
    # 1. Explicit env-var override ─ useful for custom builds / testing.
    env = os.environ.get("AWSM_ATTENTION_LIB")
    if env:
        p = pathlib.Path(env)
        if p.is_file():
            return str(p)
        raise FileNotFoundError(
            f"AWSM_ATTENTION_LIB is set to '{env}' but the file does not exist."
        )

    # 2. importlib.resources (preferred, works with zipped wheels too)
    try:
        # Python ≥ 3.9 API
        from importlib.resources import files  # type: ignore[attr-defined]

        lib_path = files(_LIB_SUBPACKAGE).joinpath(_LIB_NAME)
        # as_posix works for both MultiplexedPath and PosixPath
        resolved = str(lib_path)
        if os.path.isfile(resolved):
            return resolved
    except (ImportError, ModuleNotFoundError, TypeError):
        pass

    raise FileNotFoundError(
        f"Cannot find {_LIB_NAME}. Ensure the package was installed correctly "
        f"(pip install .) or set the AWSM_ATTENTION_LIB environment variable "
        f"to the full path of the shared library."
    )


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------

# Must match the C enum FlashDtype { BF16=0, FP16=1, FP32=2 }
_TORCH_DTYPE_TO_FLASH = {
    torch.bfloat16: 0,
    torch.float16: 1,
    torch.float32: 2,
}


def _flash_dtype(t: torch.Tensor) -> int:
    d = _TORCH_DTYPE_TO_FLASH.get(t.dtype)
    if d is None:
        raise ValueError(
            f"Unsupported dtype {t.dtype}. Expected one of bf16, fp16, fp32."
        )
    return d


# ---------------------------------------------------------------------------
# Pointer / tensor helpers
# ---------------------------------------------------------------------------

def _data_ptr(t: torch.Tensor) -> c_void_p:
    """Return a ctypes void pointer to the tensor's data."""
    return c_void_p(t.data_ptr())


def _int_ptr(t: torch.Tensor) -> POINTER(c_int):
    """Return a ctypes int* pointer for an int32 tensor."""
    assert t.dtype == torch.int32, f"Expected int32 tensor, got {t.dtype}"
    assert t.is_contiguous(), "Tensor must be contiguous"
    return ctypes.cast(c_void_p(t.data_ptr()), POINTER(c_int))


def _float_ptr(t: torch.Tensor) -> POINTER(ctypes.c_float):
    """Return a ctypes float* pointer for a float32 tensor."""
    assert t.dtype == torch.float32, f"Expected float32 tensor, got {t.dtype}"
    assert t.is_contiguous(), "Tensor must be contiguous"
    return ctypes.cast(c_void_p(t.data_ptr()), POINTER(ctypes.c_float))


# ---------------------------------------------------------------------------
# GPU arch / SM count detection
# ---------------------------------------------------------------------------

def _get_gpu_info(device: Optional[torch.device] = None) -> Tuple[int, int]:
    """Return (arch, sm_count) for the given CUDA device.

    arch is encoded as major*10 + minor (e.g. 90 for SM90 / H100).
    """
    if device is None:
        device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    arch = props.major * 10 + props.minor
    sm_count = props.multi_processor_count
    return arch, sm_count


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FlashAttentionHelper:
    """Thin Python wrapper around the C flash‑attention helper library.

    On construction, the shared library is loaded and GPU arch / SM count are
    detected.  Forward and backward calls allocate a workspace on the fly,
    invoke the C function, and then free the workspace.
    """

    def __init__(
        self,
        lib_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        resolved_path = lib_path or _find_lib()
        self._load_lib(resolved_path)
        self._setup_signatures()
        self.arch, self.sm_count = _get_gpu_info(device)

    def _load_lib(self, wrapper_lib_path: str):
        """Load the C library and all its transitive dependencies.

        ``libattentionwrapper.so`` is linked with relative rpaths like
        ``flash2/lib/`` and ``flash3/lib/`` that only resolve from the
        original build directory (the project root).  We handle this by:

          1. Pre-loading ``libcuda.so`` globally (CUDA driver symbols).
          2. Explicitly pre-loading ``libflash2.so`` / ``libflash3.so``
             by absolute path with ``RTLD_GLOBAL``.
          3. Temporarily ``chdir``-ing to the project root so any
             ``DT_NEEDED`` entries in the wrapper can still be resolved
             by the dynamic linker via the baked relative rpaths.
        """
        wrapper_path = pathlib.Path(wrapper_lib_path).resolve()
        lib_dir = wrapper_path.parent          # .../awsm_attention/lib/
        pkg_dir = lib_dir.parent               # .../awsm_attention/
        project_root = pkg_dir.parent          # .../attention_helper/

        # 1. CUDA driver
        for lib_name in ("libcuda.so.1", "libcuda.so"):
            try:
                ctypes.CDLL(lib_name, mode=ctypes.RTLD_GLOBAL)
                break
            except OSError:
                continue

        # 2. Explicit pre-load from all candidate locations
        search_dirs = [
            lib_dir,                           # installed: awsm_attention/lib/
            project_root / "flash2" / "lib",   # build tree
            project_root / "flash3" / "lib",   # build tree
            project_root / "lib",              # build tree top-level
        ]
        for so_name in ("libflash2.so", "libflash3.so"):
            for d in search_dirs:
                p = d / so_name
                if p.is_file():
                    try:
                        ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        continue
                    break

        # 3. chdir to project root so relative rpaths baked into
        #    libattentionwrapper.so (e.g. "flash3/lib/") resolve.
        orig_cwd = os.getcwd()
        try:
            if project_root.is_dir():
                os.chdir(str(project_root))
            self._lib = ctypes.CDLL(str(wrapper_path), mode=ctypes.RTLD_GLOBAL)
        finally:
            os.chdir(orig_cwd)

    # ------------------------------------------------------------------
    # ctypes function signatures
    # ------------------------------------------------------------------
    def _setup_signatures(self):
        lib = self._lib

        # int flash_attention_get_workspace_size(
        #     int arch, int sm_count, FlashDtype flash_dtype, int is_training,
        #     int num_q_heads, int num_kv_heads, int head_dim,
        #     int max_chunk_size, int max_seq_len, int max_seqs_in_chunk,
        #     int is_causal, uint64_t *ret_workspace_size);
        lib.flash_attention_get_workspace_size.restype = c_int
        lib.flash_attention_get_workspace_size.argtypes = [
            c_int, c_int, c_int, c_int,          # arch, sm_count, dtype, is_training
            c_int, c_int, c_int,                  # num_q_heads, num_kv_heads, head_dim
            c_int, c_int, c_int,                  # max_chunk_size, max_seq_len, max_seqs_in_chunk
            c_int,                                # is_causal
            POINTER(c_uint64),                    # ret_workspace_size
        ]

        # int flash_attention_fwd(CUstream stream, int arch, int sm_count,
        #     FlashDtype flash_dtype,
        #     int num_seqs, int total_q, int total_k,
        #     int *q_seq_offsets, int *q_seq_lens, int max_seqlen_q,
        #     int *k_seq_offsets, int *k_seq_lens, int max_seqlen_k,
        #     int num_q_heads, int num_kv_heads, int head_dim,
        #     void *x_q, void *x_k, void *x_v,
        #     void *x_attn_out, float *softmax_lse,
        #     int is_causal,
        #     uint64_t workspaceBytes, void *workspace);
        lib.flash_attention_fwd.restype = c_int
        lib.flash_attention_fwd.argtypes = [
            c_void_p,                             # CUstream
            c_int, c_int, c_int,                  # arch, sm_count, dtype
            c_int,                                # num_seqs
            c_int, c_int,                         # total_q, total_k
            POINTER(c_int), POINTER(c_int), c_int,  # q_seq_offsets, q_seq_lens, max_seqlen_q
            POINTER(c_int), POINTER(c_int), c_int,  # k_seq_offsets, k_seq_lens, max_seqlen_k
            c_int, c_int, c_int,                  # num_q_heads, num_kv_heads, head_dim
            c_void_p, c_void_p, c_void_p,         # x_q, x_k, x_v
            c_void_p, POINTER(ctypes.c_float),     # x_attn_out, softmax_lse
            c_int,                                # is_causal
            c_uint64, c_void_p,                   # workspaceBytes, workspace
        ]

        # int flash_attention_bwd(CUstream stream, int arch, int sm_count,
        #     FlashDtype flash_dtype,
        #     int num_seqs, int total_q, int total_k,
        #     int *q_seq_offsets, int *q_seq_lens, int max_seqlen_q,
        #     int *k_seq_offsets, int *k_seq_lens, int max_seqlen_k,
        #     int num_q_heads, int num_kv_heads, int head_dim,
        #     void *x_q, void *x_k, void *x_v,
        #     void *x_attn_out, float *softmax_lse,
        #     void *dx_out,
        #     void *dx_q, void *dx_k, void *dx_v,
        #     int is_causal,
        #     uint64_t workspaceBytes, void *workspace);
        lib.flash_attention_bwd.restype = c_int
        lib.flash_attention_bwd.argtypes = [
            c_void_p,                             # CUstream
            c_int, c_int, c_int,                  # arch, sm_count, dtype
            c_int,                                # num_seqs
            c_int, c_int,                         # total_q, total_k
            POINTER(c_int), POINTER(c_int), c_int,  # q_seq_offsets, q_seq_lens, max_seqlen_q
            POINTER(c_int), POINTER(c_int), c_int,  # k_seq_offsets, k_seq_lens, max_seqlen_k
            c_int, c_int, c_int,                  # num_q_heads, num_kv_heads, head_dim
            c_void_p, c_void_p, c_void_p,         # x_q, x_k, x_v
            c_void_p, POINTER(ctypes.c_float),     # x_attn_out, softmax_lse
            c_void_p,                             # dx_out
            c_void_p, c_void_p, c_void_p,         # dx_q, dx_k, dx_v
            c_int,                                # is_causal
            c_uint64, c_void_p,                   # workspaceBytes, workspace
        ]

    # ------------------------------------------------------------------
    # Workspace helpers
    # ------------------------------------------------------------------
    def get_workspace_size(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_chunk_size: int,
        max_seq_len: int,
        max_seqs_in_chunk: int,
        is_causal: bool,
        is_training: bool,
        dtype: torch.dtype = torch.bfloat16,
        sm_count: Optional[int] = None,
    ) -> int:
        """Query the C library for the required workspace size in bytes."""
        effective_sm_count = sm_count if sm_count is not None else self.sm_count
        ws = c_uint64(0)
        ret = self._lib.flash_attention_get_workspace_size(
            c_int(self.arch),
            c_int(effective_sm_count),
            c_int(_TORCH_DTYPE_TO_FLASH[dtype]),
            c_int(int(is_training)),
            c_int(num_q_heads),
            c_int(num_kv_heads),
            c_int(head_dim),
            c_int(max_chunk_size),
            c_int(max_seq_len),
            c_int(max_seqs_in_chunk),
            c_int(int(is_causal)),
            byref(ws),
        )
        if ret != 0:
            raise RuntimeError("flash_attention_get_workspace_size failed")
        return ws.value

    def _allocate_workspace(self, size_bytes: int, device: torch.device) -> torch.Tensor:
        """Allocate a uint8 workspace tensor on the given device."""
        if size_bytes == 0:
            return torch.empty(0, dtype=torch.uint8, device=device)
        return torch.empty(size_bytes, dtype=torch.uint8, device=device)

    @staticmethod
    def _get_cuda_stream() -> int:
        """Get the raw CUstream pointer for the current CUDA stream."""
        stream = torch.cuda.current_stream()
        return stream.cuda_stream  # int handle

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        q: torch.Tensor,            # (total_q, n_q_heads, head_dim)
        k: torch.Tensor,            # (total_k, n_kv_heads, head_dim)
        v: torch.Tensor,            # (total_k, n_kv_heads, head_dim)
        out: torch.Tensor,          # (total_q, n_q_heads, head_dim)
        softmax_lse: torch.Tensor,  # (n_q_heads, total_q)
        q_seq_offsets: torch.Tensor, # (num_seqs + 1,) int32
        k_seq_offsets: torch.Tensor, # (num_seqs + 1,) int32
        q_seq_lens: torch.Tensor,    # (num_seqs,) int32
        k_seq_lens: torch.Tensor,    # (num_seqs,) int32
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool = True,
        sm_count: Optional[int] = None,
    ):
        """Run flash attention forward pass.

        Allocates workspace, calls C library, frees workspace on return.

        Parameters
        ----------
        sm_count : int, optional
            Override the auto-detected SM count. Useful for limiting the number
            of SMs used by the kernel (e.g. to reserve SMs for overlapping work).
        """
        # Validate
        assert q.is_cuda and k.is_cuda and v.is_cuda
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        assert out.is_contiguous()
        device = q.device

        # softmax_lse may be a non-contiguous slice of a larger buffer.
        # The kernel writes contiguously, so we need a contiguous tensor.
        lse_contiguous = softmax_lse.contiguous()
        lse_is_alias = lse_contiguous.data_ptr() == softmax_lse.data_ptr()

        total_q = q.shape[0]
        total_k = k.shape[0]
        num_q_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        head_dim = q.shape[2]
        num_seqs = q_seq_offsets.shape[0] - 1

        ### should probably assert that total_k lines up with sum of k_seq_lens...

        flash_dtype = _flash_dtype(q)
        effective_sm_count = sm_count if sm_count is not None else self.sm_count

        # Get workspace size and allocate.
        ws_bytes = self.get_workspace_size(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_chunk_size=total_q,
            max_seq_len=max(total_q, total_k),
            max_seqs_in_chunk=num_seqs,
            is_causal=causal,
            is_training=False,
            dtype=q.dtype,
            sm_count=effective_sm_count,
        )
        workspace = self._allocate_workspace(ws_bytes, device)

        stream_ptr = self._get_cuda_stream()

        ret = self._lib.flash_attention_fwd(
            c_void_p(stream_ptr),
            c_int(self.arch),
            c_int(effective_sm_count),
            c_int(flash_dtype),
            c_int(num_seqs),
            c_int(total_q),
            c_int(total_k),
            _int_ptr(q_seq_offsets),
            _int_ptr(q_seq_lens),
            c_int(max_seqlen_q),
            _int_ptr(k_seq_offsets),
            _int_ptr(k_seq_lens),
            c_int(max_seqlen_k),
            c_int(num_q_heads),
            c_int(num_kv_heads),
            c_int(head_dim),
            _data_ptr(q),
            _data_ptr(k),
            _data_ptr(v),
            _data_ptr(out),
            _float_ptr(lse_contiguous),
            c_int(int(causal)),
            c_uint64(ws_bytes),
            _data_ptr(workspace) if ws_bytes > 0 else c_void_p(0),
        )

        # workspace freed when tensor goes out of scope
        del workspace

        if ret != 0:
            raise RuntimeError(f"flash_attention_fwd failed with error code {ret}")

        # Copy back if softmax_lse was non-contiguous (a view of a larger buffer)
        if not lse_is_alias:
            softmax_lse.copy_(lse_contiguous)

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------
    def backward(
        self,
        dout: torch.Tensor,          # (total_q, n_q_heads, head_dim)
        q: torch.Tensor,             # (total_q, n_q_heads, head_dim)
        k: torch.Tensor,             # (total_k, n_kv_heads, head_dim)
        v: torch.Tensor,             # (total_k, n_kv_heads, head_dim)
        out: torch.Tensor,           # (total_q, n_q_heads, head_dim)
        softmax_lse: torch.Tensor,   # (n_q_heads, total_q)
        dq: torch.Tensor,            # (total_q, n_q_heads, head_dim)
        dk: torch.Tensor,            # (total_k, n_kv_heads, head_dim)
        dv: torch.Tensor,            # (total_k, n_kv_heads, head_dim)
        q_seq_offsets: torch.Tensor,  # (num_seqs + 1,) int32
        k_seq_offsets: torch.Tensor,  # (num_seqs + 1,) int32
        q_seq_lens: torch.Tensor,     # (num_seqs,) int32
        k_seq_lens: torch.Tensor,     # (num_seqs,) int32
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool = True,
        sm_count: Optional[int] = None,
    ):
        """Run flash attention backward pass.

        Allocates workspace, calls C library, frees workspace on return.

        Parameters
        ----------
        sm_count : int, optional
            Override the auto-detected SM count. Useful for limiting the number
            of SMs used by the kernel (e.g. to reserve SMs for overlapping work).
        """
        assert q.is_cuda and k.is_cuda and v.is_cuda
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        assert dq.is_contiguous() and dk.is_contiguous() and dv.is_contiguous()
        assert dout.is_contiguous() and out.is_contiguous()
        device = q.device

        # softmax_lse may be a non-contiguous slice of a larger buffer.
        lse_contiguous = softmax_lse.contiguous()

        total_q = q.shape[0]
        total_k = k.shape[0]
        num_q_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        head_dim = q.shape[2]
        num_seqs = q_seq_offsets.shape[0] - 1

        flash_dtype = _flash_dtype(q)
        effective_sm_count = sm_count if sm_count is not None else self.sm_count


        ws_bytes = self.get_workspace_size(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_chunk_size=total_q,
            max_seq_len=max(total_q, total_k),
            max_seqs_in_chunk=num_seqs,
            is_causal=causal,
            is_training=True,
            dtype=q.dtype,
            sm_count=effective_sm_count,
        )
        workspace = self._allocate_workspace(ws_bytes, device)

        stream_ptr = self._get_cuda_stream()

        ret = self._lib.flash_attention_bwd(
            c_void_p(stream_ptr),
            c_int(self.arch),
            c_int(effective_sm_count),
            c_int(flash_dtype),
            c_int(num_seqs),
            c_int(total_q),
            c_int(total_k),
            _int_ptr(q_seq_offsets),
            _int_ptr(q_seq_lens),
            c_int(max_seqlen_q),
            _int_ptr(k_seq_offsets),
            _int_ptr(k_seq_lens),
            c_int(max_seqlen_k),
            c_int(num_q_heads),
            c_int(num_kv_heads),
            c_int(head_dim),
            _data_ptr(q),
            _data_ptr(k),
            _data_ptr(v),
            _data_ptr(out),
            _float_ptr(lse_contiguous),
            _data_ptr(dout),
            _data_ptr(dq),
            _data_ptr(dk),
            _data_ptr(dv),
            c_int(int(causal)),
            c_uint64(ws_bytes),
            _data_ptr(workspace) if ws_bytes > 0 else c_void_p(0),
        )

        del workspace

        if ret != 0:
            raise RuntimeError(f"flash_attention_bwd failed with error code {ret}")