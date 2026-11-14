from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, TypeVar

import torch
from torch import nn
from torch.utils.data import DataLoader


T = TypeVar("T")


@dataclass
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    device: torch.device
    backend: str | None = None
    initialized: bool = False
    owns_process_group: bool = False

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def print(self, *args: Any, **kwargs: Any) -> None:
        if self.is_main_process:
            print(*args, **kwargs)

    def barrier(self) -> None:
        if self.enabled and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def broadcast_object(self, value: T, *, src: int = 0) -> T:
        if not self.enabled:
            return value
        payload: list[Any] = [value]
        torch.distributed.broadcast_object_list(payload, src=src)
        return payload[0]

    def reduce_weighted_average(self, weighted_sum: float, weight: float) -> float:
        totals = torch.tensor(
            [float(weighted_sum), float(weight)],
            dtype=torch.float64,
            device=self.device,
        )
        if self.enabled:
            torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
        total_weight = float(totals[1].detach().cpu())
        if total_weight <= 0:
            return 0.0
        return float((totals[0] / totals[1]).detach().cpu())

    def close(self) -> None:
        if self.owns_process_group and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            self.initialized = False
            self.owns_process_group = False


def initialize_distributed_context(device_name: str = "auto") -> DistributedContext:
    world_size = env_int("WORLD_SIZE", 1)
    rank = env_int("RANK", 0)
    local_rank = env_int("LOCAL_RANK", 0)
    local_world_size = env_int("LOCAL_WORLD_SIZE", world_size)
    device = resolve_training_device(device_name, local_rank=local_rank, world_size=world_size)
    backend: str | None = None
    initialized = False
    owns_process_group = False

    if world_size > 1:
        if not torch.distributed.is_available():
            raise RuntimeError("torch.distributed is not available in this PyTorch build")
        if device.type == "cuda":
            torch.cuda.set_device(device)
            backend = "nccl" if torch.distributed.is_nccl_available() else "gloo"
        else:
            backend = "gloo"
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=backend, init_method="env://")
            owns_process_group = True
        initialized = torch.distributed.is_initialized()

    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        local_world_size=local_world_size,
        device=device,
        backend=backend,
        initialized=initialized,
        owns_process_group=owns_process_group,
    )


def env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"environment variable {name} must be an integer, got {raw_value!r}") from exc


def resolve_training_device(device_name: str, *, local_rank: int = 0, world_size: int = 1) -> torch.device:
    normalized = device_name.strip().casefold()
    if world_size > 1 and normalized.startswith("cuda:"):
        raise ValueError("with DDP, pass --device auto or --device cuda so each rank uses LOCAL_RANK")
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", local_rank) if world_size > 1 else torch.device("cuda")
        return torch.device("cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return torch.device("cuda", local_rank) if world_size > 1 else torch.device("cuda")
    return torch.device(device_name)


def prepare_model_for_distributed(
    model: nn.Module,
    context: DistributedContext,
    *,
    find_unused_parameters: bool = True,
) -> nn.Module:
    if not context.enabled:
        return model
    from torch.nn.parallel import DistributedDataParallel

    # kwargs: dict[str, Any] = {"find_unused_parameters": find_unused_parameters}
    kwargs: dict[str, Any] = {}
    if context.device.type == "cuda":
        kwargs["device_ids"] = [context.device.index]
        kwargs["output_device"] = context.device.index
    return DistributedDataParallel(model, **kwargs)


def unwrap_distributed_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def distributed_summary(context: DistributedContext) -> dict[str, Any]:
    return {
        "enabled": context.enabled,
        "backend": context.backend,
        "rank": context.rank,
        "local_rank": context.local_rank,
        "world_size": context.world_size,
        "local_world_size": context.local_world_size,
    }


def set_dataloader_epoch(dataloader: DataLoader[Any], epoch: int) -> None:
    sampler = getattr(dataloader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
