# SPDX-License-Identifier: Apache-2.0
"""Weight Cache Daemon — a persistent process that holds post-quantized,
TP-sharded model weights in GPU memory and serves them via CUDA IPC handles.

Each GPU runs one daemon process for its TP rank. The daemon:
1. Loads model weights from disk (full pipeline: disk → TP shard → quantize)
2. Exports every parameter/buffer as a CUDA IPC handle
3. Serves handles over a Unix socket to requesting engine processes
4. Validates CacheConfig compatibility before serving

Usage:
    # Single-node: launch all TP rank daemons with a single command:
    python -m sglang.srt.weight_cache.daemon \
        --model-path /path/to/model --tp-size 4 \
        --load-format auto --dtype auto --quantization fp8

    # Multi-node: run on each node with --nnodes and --node-rank:
    # Node 0:
    python -m sglang.srt.weight_cache.daemon \
        --model-path /path/to/model --tp-size 16 \
        --nnodes 2 --node-rank 0 \
        --dist-init-method tcp://node0-ip:29500

    # Node 1:
    python -m sglang.srt.weight_cache.daemon \
        --model-path /path/to/model --tp-size 16 \
        --nnodes 2 --node-rank 1 \
        --dist-init-method tcp://node0-ip:29500

    # Or launch a single daemon for a specific rank:
    python -m sglang.srt.weight_cache.daemon \
        --model-path /path/to/model \
        --gpu-id 0 --tp-size 4 --tp-rank 0 \
        --dist-init-method tcp://127.0.0.1:29500
"""

import logging
import os
import signal
import socket
import time
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import publish
from sglang.srt.utils import MultiprocessingSerializer

from .protocol import (
    CacheConfig,
    build_daemon_model_specs,
    check_ipc_quant_support,
    cleanup_stale_daemon_files,
    compute_env_stamp,
    compute_global_rank,
    compute_local_gpu_id,
    format_daemon_role,
    get_quant_method_name,
    get_ready_path,
    get_socket_path,
    hash_quant_config,
    recv_msg,
    send_msg,
)

logger = logging.getLogger(__name__)

# Per-connection timeout for the serial serve loop. A client exchange is tiny
# (a config dict + IPC handle metadata), so this generous bound never trips a
# healthy client, yet guarantees one hung/dead peer can't stall the other
# engine ranks indefinitely.
CLIENT_CONNECTION_TIMEOUT = 30.0

# CPU tensors cannot cross CUDA IPC, so their bytes ride inline in the socket
# message. Arbitrary; sized to pass K3's 16 KiB vision pos_emb while keeping a
# stray large CPU tensor from silently bloating every client handshake.
MAX_INLINE_TENSOR_BYTES = 1 << 20


def _describe_oversized(name: str, tensor: "torch.Tensor") -> str:
    """One line for the refusal listing when a CPU tensor cannot ride inline."""
    num_bytes = tensor.numel() * tensor.element_size()
    return f"{name} {tuple(tensor.shape)} {tensor.dtype} ({num_bytes // 1024} KiB)"


class WeightCacheDaemon:
    """Persistent GPU weight cache for a single TP rank.

    Holds the complete post-quantization state_dict in GPU memory and
    serves CUDA IPC handles to engine processes via Unix socket.
    """

    def __init__(
        self,
        model_path: str,
        gpu_id: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        pp_size: int = 1,
        pp_rank: int = 0,
        dp_size: int = 1,
        ep_size: int = 1,
        load_format: str = "auto",
        dtype: str = "auto",
        quantization: Optional[str] = None,
        moe_a2a_backend: str = "none",
        moe_runner_backend: str = "auto",
        model_loader_extra_config: str = "{}",
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        dist_init_method: Optional[str] = None,
        is_draft_model: bool = False,
        speculative_algorithm: Optional[str] = None,
    ):
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.pp_size = pp_size
        self.pp_rank = pp_rank
        self.dp_size = dp_size
        self.ep_size = ep_size
        self.load_format = load_format
        self.dtype = dtype
        self.quantization = quantization
        self.moe_a2a_backend = moe_a2a_backend
        self.moe_runner_backend = moe_runner_backend
        self.model_loader_extra_config = model_loader_extra_config
        self.trust_remote_code = trust_remote_code
        # handle_speculative_decoding stamps "main" on the engine as soon as a
        # draft path is set; the daemon loads the draft as its own model_path,
        # so that hook never fires here and the fingerprints would disagree.
        if is_draft_model and revision is None:
            revision = "main"
        self.revision = revision
        self.dist_init_method = dist_init_method
        # Draft daemon: holds the speculative draft model instead of the
        # target, on the "_draft" socket.
        self.is_draft_model = is_draft_model
        self.speculative_algorithm = speculative_algorithm

        global_rank = compute_global_rank(tp_size, pp_rank, tp_rank)
        self.socket_path = get_socket_path(global_rank, is_draft_model=is_draft_model)
        self.ready_path = get_ready_path(global_rank, is_draft_model=is_draft_model)

        self.model = None
        self.config: Optional[CacheConfig] = None
        # name -> {"handle": base64_str, "shape": list, "dtype": str, "is_param": bool}
        self.state_entries: Dict[str, Dict[str, Any]] = {}
        # Module names whose MXFP4 post-load built MegaMoE weights; see
        # _collect_mega_moe_modules.
        self.mega_moe_modules: List[str] = []
        self.preloaded_weights_bytes = 0
        # Resolved on the first _device_used_bytes() call; see it for why the
        # choice must be sticky.
        self._use_nvml_accounting: Optional[bool] = None

    def _init_distributed(self, server_args, model_config):
        """Initialize the distributed backend required for model loading.

        Uses the same world_size/rank formula as the engine:
            world_size = tp_size * pp_size
            rank = tp_size * pp_rank + tp_rank
        """
        from sglang.srt.distributed.parallel_state import (
            init_distributed_environment,
            initialize_model_parallel,
            model_parallel_is_initialized,
        )

        if model_parallel_is_initialized():
            logger.info(
                f"[WeightCacheDaemon gpu={self.gpu_id}] "
                f"Distributed already initialized, skipping"
            )
            return

        # Initialize distributed environment
        import torch.distributed as dist

        if not dist.is_initialized():
            if self.dist_init_method is None:
                # Fallback: auto-assign a port. This only works for single-process.
                import socket as sock_mod

                with sock_mod.socket(sock_mod.AF_INET, sock_mod.SOCK_STREAM) as s:
                    s.bind(("127.0.0.1", 0))
                    free_port = s.getsockname()[1]
                self.dist_init_method = f"tcp://127.0.0.1:{free_port}"

            init_distributed_environment(
                world_size=self.tp_size * self.pp_size,
                rank=compute_global_rank(self.tp_size, self.pp_rank, self.tp_rank),
                distributed_init_method=self.dist_init_method,
                local_rank=self.gpu_id,
                backend=current_platform.get_torch_distributed_backend_str(),
            )

        initialize_model_parallel(
            tensor_model_parallel_size=self.tp_size,
            pipeline_model_parallel_size=self.pp_size,
            expert_model_parallel_size=self.ep_size,
        )

        # Initialize DP attention state (required by some models like Qwen3 MoE)
        from sglang.srt.layers.dp_attention import initialize_dp_attention

        initialize_dp_attention(server_args, model_config)

        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id} tp_rank={self.tp_rank}] "
            f"Distributed backend initialized (tp_size={self.tp_size}, "
            f"pp_size={self.pp_size}, "
            f"world_size={self.tp_size * self.pp_size})"
        )

    def load(self):
        """Full loading pipeline: disk → TP shard → quantize → export IPC handles."""
        # CUDA IPC weight sharing relies on torch's _share_cuda_ handle export,
        # which only exists on CUDA-alike platforms (CUDA / ROCm). Fail loud here
        # instead of dying deep inside the export with an opaque error.
        if not current_platform.is_cuda_alike():
            raise RuntimeError(
                f"[WeightCacheDaemon] the weight cache daemon requires a CUDA-alike "
                f"platform (CUDA or ROCm) for CUDA IPC weight sharing, but the "
                f"active platform device type is {current_platform.device_type!r}. "
                f"Disable the weight cache (--weight-cache-mode off)."
            )
        # expandable_segments makes torch's caching allocator hand out memory
        # that cannot be exported via _share_cuda_, so the IPC export below would
        # die mid-way with an opaque CUDA error. Fail fast with an actionable
        # message before touching the device.
        self._assert_ipc_compatible_allocator()
        current_platform.set_device(current_platform.get_device(self.gpu_id))

        # Reduce thread contention during multi-process loading
        torch.set_num_threads(1)

        # Lazy imports to avoid circular dependencies and speed up startup
        from sglang.srt.configs.device_config import DeviceConfig
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.layers.moe import initialize_moe_config
        from sglang.srt.model_loader.loader import get_model_loader
        from sglang.srt.server_args import ServerArgs

        server_args = ServerArgs(
            model_path=self.model_path,
            dtype=self.dtype,
            quantization=self.quantization,
            trust_remote_code=self.trust_remote_code,
            tp_size=self.tp_size,
            pp_size=self.pp_size,
            dp_size=self.dp_size,
            ep_size=self.ep_size,
            moe_a2a_backend=self.moe_a2a_backend,
            moe_runner_backend=self.moe_runner_backend,
            load_format=self.load_format,
            model_loader_extra_config=self.model_loader_extra_config,
        )
        publish(server_args, role="weight_cache_daemon")
        # ServerArgs resolution rewrites the parallel sizes (megamoe spans ep
        # over tp). The engine fingerprints and shards on the resolved values,
        # so adopt them here too or the experts shard differently from the
        # engine that maps them.
        self.tp_size = server_args.tp_size
        self.pp_size = server_args.pp_size
        self.dp_size = server_args.dp_size
        self.ep_size = server_args.ep_size
        # The engine does this in Scheduler.init; without it the daemon's
        # get_moe_{a2a,runner}_backend() lazily report none/auto and both the
        # IPC quant gate and the env stamp disagree with the engine.
        initialize_moe_config(server_args)

        # Initialize distributed backend for model loading
        # (must be done after server_args and model_config are available)
        # Build model config first, then init distributed
        model_config = ModelConfig(
            model_path=self.model_path,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
            dtype=self.dtype,
            quantization=self.quantization,
            # Drives _config_draft_model's arch remap, so the daemon loads the
            # same nn.Module the draft worker will map.
            is_draft_model=self.is_draft_model,
            speculative_algorithm=self.speculative_algorithm,
        )

        # Build cache config fingerprint BEFORE loading the model.
        # Loading may mutate hf_config.quantization_config (e.g. via
        # process_weights_after_loading), which would produce a different
        # hash than what the engine computes from the original config.
        # ModelConfig always exposes hf_config/quantization directly;
        # quantization_config is the only genuinely-optional attribute.
        quant_config = getattr(model_config.hf_config, "quantization_config", None)
        quant_method = get_quant_method_name(
            self.quantization or model_config.quantization
        )
        if not quant_method and quant_config is not None:
            quant_method = get_quant_method_name(quant_config)

        self.config = CacheConfig(
            model_path=self.model_path,
            model_arch=(
                model_config.hf_config.architectures[0]
                if model_config.hf_config.architectures
                else ""
            ),
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            pp_size=self.pp_size,
            pp_rank=self.pp_rank,
            dp_size=self.dp_size,
            ep_size=self.ep_size,
            quant_method=quant_method,
            quant_config_hash=hash_quant_config(quant_config),
            dtype=str(model_config.dtype),
            revision=self.revision or "",
            is_draft_model=self.is_draft_model,
            **compute_env_stamp(),
        )

        # Refuse to serve quant methods not verified to round-trip through pure
        # IPC tensor export. Checked before loading so an unsupported model
        # fails fast instead of after minutes of disk I/O.
        check_ipc_quant_support(quant_method, quant_config, where="daemon")

        # Sampled before _init_distributed, so the post-teardown delta is
        # everything this daemon leaves resident.
        current_platform.empty_cache()
        memory_before_load = self._device_used_bytes()

        # Initialize distributed backend (requires server_args + model_config)
        self._init_distributed(server_args, model_config)

        # Build load config
        load_config = LoadConfig(
            load_format=self.load_format,
            model_loader_extra_config=self.model_loader_extra_config,
            tp_rank=self.tp_rank,
        )

        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id} tp_rank={self.tp_rank}] "
            f"Loading model from disk: {self.model_path}"
        )
        tic = time.perf_counter()

        # Load model using DefaultModelLoader (includes TP sharding + quant post-process)
        loader = get_model_loader(load_config=load_config, model_config=model_config)
        self.model = loader.load_model(
            model_config=model_config,
            device_config=DeviceConfig(current_platform.device_type, self.gpu_id),
        )

        elapsed = time.perf_counter() - tic
        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id} tp_rank={self.tp_rank}] "
            f"Model loaded from disk in {elapsed:.2f}s"
        )

        # Ensure every post-processing kernel has retired before we export the
        # memory: clients map these tensors read-only via IPC and would otherwise
        # risk observing half-written weights.
        current_platform.synchronize()
        current_platform.empty_cache()

        # Export all parameters and buffers as IPC handles
        self._export_state()

        # Nothing below this point collectives, and the group's NCCL comm
        # buffers sit on the device the engine sizes its KV pool against.
        # After the export: _share_cuda_ needs the allocation, not the group.
        self._teardown_distributed()

        # After the teardown, so NCCL buffers freed there are not charged to
        # the engine. An unreadable sample reports 0, like an older daemon:
        # the engine skips the correction rather than trusting a bad one.
        delta = self._memory_delta(memory_before_load, self._device_used_bytes())
        if delta is None:
            logger.warning(
                f"[WeightCacheDaemon gpu={self.gpu_id} tp_rank={self.tp_rank}] "
                f"Could not measure resident weight memory; reporting 0, so "
                f"the engine will size its KV pool without that correction."
            )
        self.preloaded_weights_bytes = max(0, delta or 0)

        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id} tp_rank={self.tp_rank}] "
            f"Exported {len(self.state_entries)} tensors as IPC handles, "
            f"preloaded_weights_bytes="
            f"{self.preloaded_weights_bytes / 1024**3:.2f} GB. Ready to serve."
        )

    def _teardown_distributed(self) -> None:
        """Release the loading-time process group and its NCCL buffers."""
        from sglang.srt.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        before = self._device_used_bytes()
        destroy_model_parallel()
        destroy_distributed_environment()
        current_platform.synchronize()
        current_platform.empty_cache()
        freed = self._memory_delta(self._device_used_bytes(), before)
        freed_str = "unknown" if freed is None else f"{freed / 1024**3:.2f} GB"
        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id} tp_rank={self.tp_rank}] "
            f"Tore down the loading process group, freeing {freed_str} "
            f"for the engine."
        )

    def _device_used_bytes(self) -> Optional[int]:
        """Bytes this process holds on its GPU, or None if it cannot be read.

        Per-process, not whole-device: under speculation a target and a draft
        daemon load the same GPU concurrently, and a device-wide delta charges
        each one the other's weights. Not memory_reserved either, since NCCL
        cudaMallocs its comm buffers outside torch's caching allocator.
        """
        # Chosen once and never mixed: the two sources are on different
        # scales, and a delta spanning both lands straight in the KV budget.
        # A source that stops answering abandons the measurement.
        if self._use_nvml_accounting is None:
            self._use_nvml_accounting = self._nvml_process_used_bytes() is not None
            if not self._use_nvml_accounting:
                logger.warning(
                    f"[WeightCacheDaemon gpu={self.gpu_id} "
                    f"tp_rank={self.tp_rank}] NVML per-process memory is "
                    f"unavailable; falling back to torch.memory_reserved, "
                    f"which does not see NCCL's buffers."
                )
        if self._use_nvml_accounting:
            return self._nvml_process_used_bytes()
        return torch.cuda.memory_reserved(self.gpu_id)

    @staticmethod
    def _memory_delta(before: Optional[int], after: Optional[int]) -> Optional[int]:
        """Difference of two samples, or None if either could not be read."""
        if before is None or after is None:
            return None
        return after - before

    def _nvml_process_used_bytes(self) -> Optional[int]:
        """This process's GPU memory per NVML, or None if NVML cannot say.

        None rather than 0 when the process is absent: 0 would read as
        "nothing allocated" and skew the delta.
        """
        try:
            import pynvml

            pynvml.nvmlInit()
            # NVML indexes physical GPUs, so map through CUDA_VISIBLE_DEVICES
            # the way cuda_vmm_utils does or a masked run reads another card.
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            device_ids = (
                list(map(int, visible.split(",")))
                if visible
                else list(range(torch.cuda.device_count()))
            )
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_ids[self.gpu_id])
            pid = os.getpid()
            for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                if proc.pid == pid:
                    return int(proc.usedGpuMemory or 0)
        except Exception:
            return None
        return None

    @staticmethod
    def _assert_ipc_compatible_allocator() -> None:
        """Reject allocator configs incompatible with CUDA IPC export.

        The expandable-segments allocator returns memory that cannot be shared
        through torch's _share_cuda_ handle, which would make the export fail
        partway with an opaque error. Detect it up front and fail loud.
        """
        for var in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
            conf = os.environ.get(var, "")
            for field in conf.split(","):
                key, _, value = field.partition(":")
                if (
                    key.strip() == "expandable_segments"
                    and value.strip().lower() == "true"
                ):
                    raise RuntimeError(
                        f"[WeightCacheDaemon] {var} sets expandable_segments:True, "
                        f"which is incompatible with CUDA IPC weight sharing: the "
                        f"expandable-segments allocator hands out memory that cannot "
                        f"be exported via _share_cuda_, so the IPC handle export "
                        f"would fail mid-way. Unset expandable_segments for the "
                        f"weight cache daemon process (it can stay enabled for the "
                        f"engine itself)."
                    )

    @staticmethod
    def _encode_tensor(tensor):
        """Encode one tensor as a socket entry, or None if it is too large.

        A CPU tensor cannot cross CUDA IPC: its handle rebuilds through an
        authkey-checked file descriptor that an unrelated engine process always
        fails. Small ones ship their bytes inline instead.
        """
        entry = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
        }
        if tensor.is_cuda:
            entry["handle"] = MultiprocessingSerializer.serialize(
                tensor.data, output_str=True
            )
            return entry

        if tensor.numel() * tensor.element_size() > MAX_INLINE_TENSOR_BYTES:
            return None
        # view(uint8), not numpy(): numpy has no bfloat16 or float8 dtype.
        # The client reads it back the same way.
        entry["data"] = (
            tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
        )
        return entry

    def _collect_mega_moe_modules(self):
        """Record which modules the MXFP4 post-load took down the mega branch.

        _mega_moe_weights_built and _mxfp4_backend are plain Python attributes
        that IPC drops, and they gate build-once short-circuits in every
        mxfp4_*_moe backend. The client cannot re-derive which modules had them
        -- structural matching would also hit experts that ran a different
        branch, silently forcing them onto the mega path -- so the daemon,
        which knows, names them explicitly.
        """
        self.mega_moe_modules = [
            name
            for name, module in self.model.named_modules()
            if getattr(module, "_mega_moe_weights_built", False)
        ]
        if self.mega_moe_modules:
            logger.info(
                f"[WeightCacheDaemon gpu={self.gpu_id}] "
                f"{len(self.mega_moe_modules)} modules built MegaMoE weights"
            )

    def _export_state(self):
        """Export model parameters and buffers as CUDA IPC handles.

        This includes both persistent buffers (in state_dict) and non-persistent
        buffers (e.g. rotary embedding cos_sin_cache) so the engine can fully
        reconstruct the model state via zero-copy IPC.
        """
        self.state_entries.clear()
        self._collect_mega_moe_modules()

        # remove_duplicate=False so tied weights are recognized as parameters
        # under every name. state_dict() below emits both tied keys, and with the
        # deduped set the duplicate would be mis-registered as a buffer, not a
        # parameter, on the client.
        param_names = set(
            name for name, _ in self.model.named_parameters(remove_duplicate=False)
        )
        state_dict_names = set(self.model.state_dict().keys())
        oversized_cpu = []
        inline_count = 0

        # Export all items from state_dict (parameters + persistent buffers)
        for name, tensor in self.model.state_dict().items():
            entry = self._encode_tensor(tensor)
            if entry is None:
                oversized_cpu.append(_describe_oversized(name, tensor))
                continue
            entry["is_param"] = name in param_names
            inline_count += "data" in entry
            self.state_entries[name] = entry

        # Also export non-persistent buffers (not in state_dict but needed
        # for inference, e.g. rotary embedding cos_sin_cache)
        non_persistent_count = 0
        # remove_duplicate=False for the same reason as the parameters above:
        # a tied buffer must be keyed under every name the client looks up.
        for name, buf in self.model.named_buffers(remove_duplicate=False):
            if name not in state_dict_names:
                entry = self._encode_tensor(buf)
                if entry is None:
                    oversized_cpu.append(_describe_oversized(name, buf))
                    continue
                entry["is_param"] = False
                inline_count += "data" in entry
                self.state_entries[name] = entry
                non_persistent_count += 1

        # Anything too large to ride inline is refused here rather than at the
        # client's rebuild, where it surfaces as an opaque authkey rejection.
        if oversized_cpu:
            listing = "\n  ".join(sorted(oversized_cpu))
            raise RuntimeError(
                f"[WeightCacheDaemon gpu={self.gpu_id}] {len(oversized_cpu)} model "
                f"tensors live on CPU and exceed the "
                f"{MAX_INLINE_TENSOR_BYTES // 1024} KiB inline limit, so they "
                f"cannot cross IPC:\n  {listing}\n"
                f"Disable the weight cache (--weight-cache-mode off) for this model."
            )

        # Log total size
        total_bytes = sum(
            len(entry.get("handle") or entry.get("data") or b"")
            for entry in self.state_entries.values()
        )
        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id}] "
            f"Exported {len(self.state_entries)} tensors "
            f"({non_persistent_count} non-persistent buffers, "
            f"{inline_count} CPU tensors sent inline), "
            f"serialized handle size ~{total_bytes / 1024 / 1024:.1f} MB"
        )

    def serve(self):
        """Block and serve IPC handles over Unix socket."""
        # Do NOT unlink an existing socket here: stale-file cleanup is the launch
        # path's job (cleanup_stale_daemon_files refuses to remove a socket whose
        # .ready still points at a live PID). A leftover live socket makes bind()
        # fail loudly instead of silently stealing another daemon's socket.
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old_umask = os.umask(0o177)
        try:
            sock.bind(self.socket_path)
        finally:
            os.umask(old_umask)
        sock.listen(8)
        sock.settimeout(1.0)  # Allow periodic shutdown check

        # Write ready file
        with open(self.ready_path, "w") as f:
            f.write(f"pid={os.getpid()}\n")
            f.write(f"config={self.config.to_dict()}\n")

        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id}] " f"Listening on {self.socket_path}"
        )

        self._running = True

        def _signal_handler(signum, frame):
            logger.info(
                f"[WeightCacheDaemon gpu={self.gpu_id}] Received signal {signum}, shutting down"
            )
            self._running = False

        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)

        try:
            while self._running:
                try:
                    conn, _ = sock.accept()
                    # The listen-socket timeout above only bounds accept(); the
                    # accepted connection is blocking by default. Since we serve
                    # connections serially, a client that connects but never sends
                    # (or dies mid-send) would block recv_msg forever and stall
                    # every other engine rank. Bound each exchange instead.
                    conn.settimeout(CLIENT_CONNECTION_TIMEOUT)
                    try:
                        self._handle_connection(conn)
                    except Exception as e:
                        logger.error(
                            f"[WeightCacheDaemon gpu={self.gpu_id}] "
                            f"Error handling connection: {e}",
                            exc_info=True,
                        )
                    finally:
                        conn.close()
                except socket.timeout:
                    continue
        finally:
            sock.close()
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
            if os.path.exists(self.ready_path):
                os.unlink(self.ready_path)
            logger.info(f"[WeightCacheDaemon gpu={self.gpu_id}] Shutdown complete")

    def _handle_connection(self, conn: socket.socket):
        """Handle a single client connection."""
        req = recv_msg(conn)

        if req.get("type") == "query_config":
            # Client asks for config without requesting handles
            send_msg(conn, {"status": "ok", "config": self.config.to_dict()})

        elif req.get("type") == "fetch_state":
            # Client requests full state with IPC handles
            engine_config = CacheConfig.from_dict(req["config"])
            if not self.config.matches(engine_config):
                # Log detailed mismatch info for debugging
                daemon_dict = self.config.to_dict()
                engine_dict = engine_config.to_dict()
                mismatches = {
                    k: (daemon_dict.get(k), engine_dict.get(k))
                    for k in daemon_dict
                    if daemon_dict.get(k) != engine_dict.get(k)
                }
                logger.warning(
                    f"[WeightCacheDaemon gpu={self.gpu_id}] "
                    f"Config mismatch: {mismatches}"
                )
                send_msg(
                    conn, {"status": "mismatch", "daemon_config": self.config.to_dict()}
                )
                return

            logger.info(
                f"[WeightCacheDaemon gpu={self.gpu_id}] "
                f"Serving {len(self.state_entries)} IPC handles to engine"
            )
            send_msg(
                conn,
                {
                    "status": "ok",
                    "config": self.config.to_dict(),
                    "entries": self.state_entries,
                    "preloaded_weights_bytes": self.preloaded_weights_bytes,
                    "mega_moe_modules": self.mega_moe_modules,
                    # PID so the client can watch daemon liveness: if this
                    # process dies while clients hold IPC mappings, their
                    # param.data (and any CUDA-graph-captured addresses) dangle.
                    "pid": os.getpid(),
                },
            )

        elif req.get("type") == "ping":
            send_msg(conn, {"status": "ok"})

        else:
            send_msg(
                conn,
                {
                    "status": "error",
                    "message": f"Unknown request type: {req.get('type')}",
                },
            )

    def shutdown(self):
        """Release GPU memory and clean up."""
        if dist.is_initialized():
            dist.destroy_process_group()
        if self.model is not None:
            del self.model
            self.model = None
        self.state_entries.clear()
        current_platform.empty_cache()
        self._running = False


def run_weight_cache_daemon(
    model_path: str,
    gpu_id: int,
    tp_size: int = 1,
    tp_rank: int = 0,
    pp_size: int = 1,
    pp_rank: int = 0,
    dp_size: int = 1,
    ep_size: int = 1,
    load_format: str = "auto",
    dtype: str = "auto",
    quantization: Optional[str] = None,
    moe_a2a_backend: str = "none",
    moe_runner_backend: str = "auto",
    model_loader_extra_config: str = "{}",
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
    dist_init_method: Optional[str] = None,
    is_draft_model: bool = False,
    speculative_algorithm: Optional[str] = None,
):
    """Entry point for running a weight cache daemon process."""
    role = format_daemon_role(is_draft_model)
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [Daemon gpu={gpu_id} tp_rank={tp_rank}{role}] %(levelname)s %(message)s",
    )

    # Die if our parent (the engine or the standalone launcher that spawned us)
    # dies, even on SIGKILL/OOM-kill. Without this an orphaned daemon keeps a
    # full weight copy pinned in GPU memory and its live-PID .ready file blocks
    # the next launch — the opposite of fast recovery.
    from sglang.srt.utils import kill_itself_when_parent_died

    kill_itself_when_parent_died()

    daemon = WeightCacheDaemon(
        model_path=model_path,
        gpu_id=gpu_id,
        tp_size=tp_size,
        tp_rank=tp_rank,
        pp_size=pp_size,
        pp_rank=pp_rank,
        dp_size=dp_size,
        ep_size=ep_size,
        load_format=load_format,
        dtype=dtype,
        quantization=quantization,
        moe_a2a_backend=moe_a2a_backend,
        moe_runner_backend=moe_runner_backend,
        model_loader_extra_config=model_loader_extra_config,
        trust_remote_code=trust_remote_code,
        revision=revision,
        dist_init_method=dist_init_method,
        is_draft_model=is_draft_model,
        speculative_algorithm=speculative_algorithm,
    )

    daemon.load()
    daemon.serve()


def build_daemon_command(
    *,
    python_executable: str,
    model_path: str,
    gpu_id: int,
    tp_size: int,
    tp_rank: int,
    pp_size: int,
    pp_rank: int,
    dp_size: int,
    ep_size: int,
    load_format: str,
    dtype: str,
    moe_a2a_backend: str,
    moe_runner_backend: str,
    dist_init_method: str,
    quantization: Optional[str] = None,
    model_loader_extra_config: str = "{}",
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
    is_draft_model: bool = False,
    speculative_algorithm: Optional[str] = None,
) -> List[str]:
    """Argv for one daemon rank.

    Single source of truth for both launchers. Every argument that reaches the
    CacheConfig fingerprint must be here: one the engine resolves but does not
    forward becomes a daemon that loads a different layout and either fails the
    handshake or, worse, serves weights the client re-derives differently.
    """
    cmd = [
        python_executable,
        "-m",
        "sglang.srt.weight_cache.daemon",
        "--model-path",
        model_path,
        "--gpu-id",
        str(gpu_id),
        "--tp-size",
        str(tp_size),
        "--tp-rank",
        str(tp_rank),
        "--pp-size",
        str(pp_size),
        "--pp-rank",
        str(pp_rank),
        "--dp-size",
        str(dp_size),
        "--ep-size",
        str(ep_size),
        "--load-format",
        load_format,
        "--dtype",
        dtype,
        "--moe-a2a-backend",
        moe_a2a_backend,
        "--moe-runner-backend",
        moe_runner_backend,
        "--dist-init-method",
        dist_init_method,
    ]
    if quantization:
        cmd += ["--quantization", quantization]
    if model_loader_extra_config and model_loader_extra_config != "{}":
        cmd += ["--model-loader-extra-config", model_loader_extra_config]
    if trust_remote_code:
        cmd += ["--trust-remote-code"]
    if revision:
        cmd += ["--revision", revision]
    if is_draft_model:
        cmd += ["--is-draft-model"]
    if speculative_algorithm:
        cmd += ["--speculative-algorithm", speculative_algorithm]
    return cmd


def _allocate_local_rendezvous() -> str:
    """Allocate a fresh loopback tcp:// endpoint for one daemon group.

    Loopback is correct only because an auto-allocated group never spans
    nodes; multi-node runs must pass an explicit rendezvous address.
    """
    import socket as sock_mod

    with sock_mod.socket(sock_mod.AF_INET, sock_mod.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return f"tcp://127.0.0.1:{s.getsockname()[1]}"


def launch_weight_cache_daemons(
    model_path: str,
    tp_size: int = 1,
    pp_size: int = 1,
    dp_size: int = 1,
    ep_size: int = 1,
    nnodes: int = 1,
    node_rank: int = 0,
    base_gpu_id: int = 0,
    gpu_id_step: int = 1,
    load_format: str = "auto",
    dtype: str = "auto",
    quantization: Optional[str] = None,
    moe_a2a_backend: str = "none",
    moe_runner_backend: str = "auto",
    model_loader_extra_config: str = "{}",
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
    dist_init_method: Optional[str] = None,
    timeout: int = 1800,
    force: bool = False,
    speculative_algorithm: Optional[str] = None,
    speculative_draft_model_path: Optional[str] = None,
    speculative_draft_model_quantization: Optional[str] = None,
    speculative_draft_model_revision: Optional[str] = None,
    draft_dist_init_method: Optional[str] = None,
):
    """Launch weight cache daemon processes for this node's PP×TP ranks.

    For single-node (nnodes=1): spawns pp_size * tp_size daemons.
    For multi-node (nnodes>1): spawns this node's share of PP×TP daemons,
    mapping local gpu_id to the correct global (pp_rank, tp_rank).

    Uses subprocess.Popen instead of multiprocessing.Process to avoid
    initializing CUDA in the parent process, which can degrade CUDA IPC
    performance in child processes.

    Usage (single-node):
        python -m sglang.srt.weight_cache.daemon \\
            --model-path /path/to/model --tp-size 4

    Usage (multi-node, run on each node):
        # Node 0:
        python -m sglang.srt.weight_cache.daemon \\
            --model-path /path/to/model --tp-size 16 \\
            --nnodes 2 --node-rank 0 \\
            --dist-init-method tcp://node0-ip:29500

        # Node 1:
        python -m sglang.srt.weight_cache.daemon \\
            --model-path /path/to/model --tp-size 16 \\
            --nnodes 2 --node-rank 1 \\
            --dist-init-method tcp://node0-ip:29500
    """
    import subprocess
    import sys

    # Replicate _calculate_rank_ranges logic from engine.py
    pp_size_per_node = max(pp_size // nnodes, 1)
    nnodes_per_pp_rank = max(nnodes // pp_size, 1)
    pp_rank_range = range(
        pp_size_per_node * (node_rank // nnodes_per_pp_rank),
        pp_size_per_node * (node_rank // nnodes_per_pp_rank + 1),
    )
    nnodes_per_tp_group = nnodes_per_pp_rank
    tp_size_per_node = tp_size // nnodes_per_tp_group
    tp_rank_range = range(
        tp_size_per_node * (node_rank % nnodes_per_tp_group),
        tp_size_per_node * (node_rank % nnodes_per_tp_group + 1),
    )

    if nnodes > 1 and dist_init_method is None:
        raise ValueError(
            "dist_init_method is required for multi-node weight cache daemons. "
            "Use --dist-init-method tcp://<node0-ip>:<port> to specify the "
            "rendezvous address accessible from all nodes."
        )

    # Auto-allocate a free port for the distributed init method
    if dist_init_method is None:
        dist_init_method = _allocate_local_rendezvous()

    # Each daemon set forms its own process group of tp_size * pp_size ranks,
    # so target and draft must not share a rendezvous or the two groups
    # deadlock against each other.
    if speculative_algorithm is not None and draft_dist_init_method is None:
        if nnodes > 1:
            raise ValueError(
                "draft_dist_init_method is required for multi-node weight "
                "cache daemons with speculative decoding: the draft daemons "
                "form their own process group and cannot share the target "
                "daemons' rendezvous address."
            )
        draft_dist_init_method = _allocate_local_rendezvous()

    specs = build_daemon_model_specs(
        model_path=model_path,
        quantization=quantization,
        revision=revision,
        target_dist_init_method=dist_init_method,
        speculative_algorithm=speculative_algorithm,
        speculative_draft_model_path=speculative_draft_model_path,
        speculative_draft_model_quantization=speculative_draft_model_quantization,
        speculative_draft_model_revision=speculative_draft_model_revision,
        draft_dist_init_method=draft_dist_init_method,
    )

    python_path = sys.executable

    # Validate and clean up stale .ready/.sock files from prior runs.
    for spec in specs:
        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                global_rank = compute_global_rank(tp_size, pp_rank, tp_rank)
                cleanup_stale_daemon_files(
                    global_rank, force=force, is_draft_model=spec.is_draft_model
                )

    procs = []
    for spec in specs:
        role = spec.role or "target"
        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                gpu_id = compute_local_gpu_id(
                    pp_rank,
                    tp_rank,
                    pp_size_per_node,
                    tp_size_per_node,
                    base_gpu_id=base_gpu_id,
                    gpu_id_step=gpu_id_step,
                )
                cmd = build_daemon_command(
                    python_executable=python_path,
                    model_path=spec.model_path,
                    gpu_id=gpu_id,
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                    pp_size=pp_size,
                    pp_rank=pp_rank,
                    dp_size=dp_size,
                    ep_size=ep_size,
                    load_format=load_format,
                    dtype=dtype,
                    moe_a2a_backend=moe_a2a_backend,
                    moe_runner_backend=moe_runner_backend,
                    dist_init_method=spec.dist_init_method,
                    quantization=spec.quantization,
                    model_loader_extra_config=model_loader_extra_config,
                    trust_remote_code=trust_remote_code,
                    revision=spec.revision,
                    is_draft_model=spec.is_draft_model,
                    speculative_algorithm=speculative_algorithm,
                )

                proc = subprocess.Popen(cmd)
                procs.append(proc)
                logger.info(
                    f"Launched {role} weight cache daemon gpu={gpu_id} "
                    f"pp_rank={pp_rank} tp_rank={tp_rank} pid={proc.pid}"
                )

    # Wait for all daemons on this node to become ready
    num_daemons = len(procs)
    check_interval = 2
    start_time = time.time()
    for spec in specs:
        role = spec.role or "target"
        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                global_rank = compute_global_rank(tp_size, pp_rank, tp_rank)
                ready_path = get_ready_path(
                    global_rank, is_draft_model=spec.is_draft_model
                )
                while not os.path.exists(ready_path):
                    time.sleep(check_interval)
                    if time.time() - start_time > timeout:
                        logger.error(
                            f"{role} weight cache daemon pp_rank={pp_rank} "
                            f"tp_rank={tp_rank} did not become ready "
                            f"within {timeout}s"
                        )
                        for p in procs:
                            p.terminate()
                        raise TimeoutError(
                            f"{role} weight cache daemon pp_rank={pp_rank} "
                            f"tp_rank={tp_rank} did not become ready "
                            f"within {timeout}s"
                        )
                    # Check if any daemon exited prematurely
                    for p in procs:
                        retcode = p.poll()
                        if retcode is not None:
                            logger.error(
                                f"Weight cache daemon exited prematurely "
                                f"with code {retcode}"
                            )
                            for other in procs:
                                if other.poll() is None:
                                    other.terminate()
                            raise RuntimeError(
                                f"Weight cache daemon exited prematurely "
                                f"with code {retcode}"
                            )
                logger.info(
                    f"{role} weight cache daemon pp_rank={pp_rank} "
                    f"tp_rank={tp_rank} is ready"
                )

    logger.info(
        f"All {num_daemons} weight cache daemons on node {node_rank} are ready "
        f"(pp_ranks={pp_rank_range.start}..{pp_rank_range.stop - 1}, "
        f"tp_ranks={tp_rank_range.start}..{tp_rank_range.stop - 1}, "
        f"dist_init_method={dist_init_method})"
    )

    # Monitor daemons — poll all of them and, the moment any one exits,
    # terminate the rest and raise. A serial proc.wait() would not notice a
    # mid-list death (e.g. procs[1] dying while procs[0] is still alive) until
    # the earlier proc happened to exit, and it never surfaced the failure.
    exited = None
    try:
        while exited is None:
            for proc in procs:
                if proc.poll() is not None:
                    exited = proc
                    break
            else:
                time.sleep(1)
                continue
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, shutting down daemons")
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        logger.info("All weight cache daemons have been terminated")

    if exited is not None:
        raise RuntimeError(
            f"Weight cache daemon (pid={exited.pid}) exited with code "
            f"{exited.returncode}; terminated the remaining daemons."
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SGLang Weight Cache Daemon")
    parser.add_argument("--model-path", required=True, help="Path to model weights")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="GPU device ID for a single daemon. "
        "If omitted, launches daemons for all TP ranks (0..tp_size-1).",
    )
    parser.add_argument(
        "--tp-rank",
        type=int,
        default=None,
        help="TP rank for a single daemon. "
        "If omitted, launches daemons for all TP ranks.",
    )
    parser.add_argument("--dp-size", type=int, default=1, help="Data parallel size")
    parser.add_argument("--ep-size", type=int, default=1, help="Expert parallel size")
    parser.add_argument("--pp-size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--pp-rank", type=int, default=0, help="Pipeline parallel rank")
    parser.add_argument("--nnodes", type=int, default=1, help="Total number of nodes")
    parser.add_argument(
        "--base-gpu-id",
        type=int,
        default=0,
        help="GPU id of this node's first rank (mirrors the engine's "
        "--base-gpu-id). Used to place daemons on the same GPUs the engine "
        "ranks will use.",
    )
    parser.add_argument(
        "--gpu-id-step",
        type=int,
        default=1,
        help="Stride between consecutive ranks' GPU ids (mirrors the engine's "
        "--gpu-id-step).",
    )
    parser.add_argument(
        "--node-rank",
        type=int,
        default=0,
        help="Rank of this node (0-indexed). Required for multi-node.",
    )
    parser.add_argument("--load-format", default="auto", help="Weight load format")
    parser.add_argument("--dtype", default="auto", help="Model dtype")
    parser.add_argument("--quantization", default=None, help="Quantization method")
    parser.add_argument(
        "--moe-a2a-backend",
        default="none",
        help="MoE all-to-all backend; must match the engine's, since it "
        "selects which process_weights_after_loading branch runs",
    )
    parser.add_argument(
        "--moe-runner-backend",
        default="auto",
        help="MoE runner backend; must match the engine's, since it selects "
        "which process_weights_after_loading branch runs",
    )
    parser.add_argument(
        "--model-loader-extra-config",
        default="{}",
        help="Extra config for model loader (JSON string)",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--revision", default=None, help="Model revision")
    parser.add_argument(
        "--dist-init-method",
        default=None,
        help="Distributed init method (e.g. tcp://node0-ip:29500). "
        "Auto-assigned for single-node when launching all ranks. "
        "Required for multi-node (nnodes > 1) and must be accessible "
        "from all nodes. Also required for tp_size > 1 when launching "
        "a single rank.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Timeout in seconds to wait for all daemons to become ready (default: 1800)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Take over a rank whose .ready file still points at a live PID by "
        "killing that daemon (use to reclaim a wedged/orphaned daemon).",
    )

    parser.add_argument(
        "--is-draft-model",
        action="store_true",
        help="Single-rank mode only: this daemon holds the speculative draft "
        "model instead of the target, and serves it on the '_draft' socket.",
    )
    parser.add_argument(
        "--speculative-algorithm",
        default=None,
        help="Speculative algorithm (e.g. DSPARK, EAGLE, NEXTN). In multi-rank "
        "mode this also launches a second daemon set for the draft model.",
    )
    parser.add_argument(
        "--speculative-draft-model-path",
        default=None,
        help="Draft model checkpoint. Defaults to --model-path.",
    )
    parser.add_argument(
        "--speculative-draft-model-quantization",
        default=None,
        help="Draft model quantization. Does NOT fall back to --quantization, "
        "mirroring ModelConfig.from_server_args.",
    )
    parser.add_argument(
        "--speculative-draft-model-revision",
        default=None,
        help="Draft model revision. Defaults to --revision.",
    )
    parser.add_argument(
        "--draft-dist-init-method",
        default=None,
        help="Rendezvous for the draft daemon group. Target and draft form "
        "separate process groups and must not share one. Auto-assigned for "
        "single-node; required for multi-node with speculative decoding.",
    )

    args = parser.parse_args()

    if args.gpu_id is not None or args.tp_rank is not None:
        # Single-rank mode: launch one daemon for the specified rank
        gpu_id = args.gpu_id if args.gpu_id is not None else args.tp_rank
        tp_rank = args.tp_rank if args.tp_rank is not None else args.gpu_id
        # Refuse to clobber a live daemon already holding this rank (mirrors the
        # multi-rank launcher and the engine path, which the multi-rank spawns
        # of this same entrypoint rely on). --force kills and takes over.
        cleanup_stale_daemon_files(
            compute_global_rank(args.tp_size, args.pp_rank, tp_rank),
            force=args.force,
            is_draft_model=args.is_draft_model,
        )
        run_weight_cache_daemon(
            model_path=args.model_path,
            gpu_id=gpu_id,
            tp_size=args.tp_size,
            tp_rank=tp_rank,
            pp_size=args.pp_size,
            pp_rank=args.pp_rank,
            dp_size=args.dp_size,
            ep_size=args.ep_size,
            load_format=args.load_format,
            dtype=args.dtype,
            quantization=args.quantization,
            moe_a2a_backend=args.moe_a2a_backend,
            moe_runner_backend=args.moe_runner_backend,
            model_loader_extra_config=args.model_loader_extra_config,
            trust_remote_code=args.trust_remote_code,
            revision=args.revision,
            dist_init_method=args.dist_init_method,
            is_draft_model=args.is_draft_model,
            speculative_algorithm=args.speculative_algorithm,
        )
    else:
        # Multi-rank mode: launch daemons for this node's TP ranks
        launch_weight_cache_daemons(
            model_path=args.model_path,
            tp_size=args.tp_size,
            pp_size=args.pp_size,
            dp_size=args.dp_size,
            ep_size=args.ep_size,
            nnodes=args.nnodes,
            node_rank=args.node_rank,
            base_gpu_id=args.base_gpu_id,
            gpu_id_step=args.gpu_id_step,
            load_format=args.load_format,
            dtype=args.dtype,
            quantization=args.quantization,
            moe_a2a_backend=args.moe_a2a_backend,
            moe_runner_backend=args.moe_runner_backend,
            model_loader_extra_config=args.model_loader_extra_config,
            trust_remote_code=args.trust_remote_code,
            revision=args.revision,
            dist_init_method=args.dist_init_method,
            timeout=args.timeout,
            force=args.force,
            speculative_algorithm=args.speculative_algorithm,
            speculative_draft_model_path=args.speculative_draft_model_path,
            speculative_draft_model_quantization=args.speculative_draft_model_quantization,
            speculative_draft_model_revision=args.speculative_draft_model_revision,
            draft_dist_init_method=args.draft_dist_init_method,
        )
