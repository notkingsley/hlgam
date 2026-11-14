"""Small stdlib client for the Shadeform REST API."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


API_BASE = "https://api.shadeform.ai/v1"


@dataclass(frozen=True)
class AvailabilityOption:
    cloud: str
    region: str
    display_name: str
    shade_instance_type: str
    hourly_price_cents: int
    num_gpus: int
    gpu_type: str
    vram_per_gpu_gb: int
    memory_gb: int
    storage_gb: int
    vcpus: int

    @property
    def hourly_price_dollars(self) -> float:
        return self.hourly_price_cents / 100.0


@dataclass(frozen=True)
class InstanceInfo:
    id: str
    status: str
    ip: str | None
    ssh_user: str
    ssh_port: int
    cloud: str
    region: str
    shade_instance_type: str
    hourly_price_cents: int
    raw: Mapping[str, Any] = field(repr=False, default_factory=dict)


class ShadeformAPI:
    """Minimal Shadeform API client using only stdlib HTTP primitives."""

    def __init__(self, api_key: str, *, api_base: str = API_BASE) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        self._api_key = api_key
        self._api_base = api_base.rstrip("/")

    def list_available(self, gpu_types: str | Sequence[str], num_gpus: int | None = None) -> list[AvailabilityOption]:
        options = self._list_gpu_types(gpu_types, num_gpus=num_gpus, available=True)
        options.sort(key=lambda option: option.hourly_price_cents)
        return options

    def list_unavailable(
        self,
        gpu_types: str | Sequence[str],
        num_gpus: int | None = None,
    ) -> list[AvailabilityOption]:
        options = self._list_gpu_types(gpu_types, num_gpus=num_gpus, available=False)
        options.sort(key=lambda option: option.hourly_price_cents)
        return options

    def wait_for_availability(
        self,
        option: AvailabilityOption,
        *,
        poll_interval_seconds: int = 30,
        timeout_seconds: int = 3600,
    ) -> AvailabilityOption:
        start = time.time()
        while True:
            available = self.list_available(option.gpu_type, option.num_gpus)
            for candidate in available:
                if (
                    candidate.cloud == option.cloud
                    and candidate.region == option.region
                    and candidate.shade_instance_type == option.shade_instance_type
                ):
                    return candidate
            elapsed = int(time.time() - start)
            if elapsed > timeout_seconds:
                raise TimeoutError(
                    f"{option.shade_instance_type} on {option.cloud}/{option.region} "
                    f"did not become available after {timeout_seconds}s"
                )
            print(
                f"  [{elapsed:>4d}s] {option.shade_instance_type} on "
                f"{option.cloud}/{option.region} unavailable; polling again ...",
                file=sys.stderr,
            )
            time.sleep(poll_interval_seconds)

    def create_instance(
        self,
        *,
        cloud: str,
        region: str,
        shade_instance_type: str,
        name: str,
    ) -> str:
        payload = {
            "cloud": cloud,
            "region": region,
            "shade_instance_type": shade_instance_type,
            "shade_cloud": True,
            "name": name,
        }
        response = self._request("POST", "/instances/create", payload)
        instance_id = response.get("id")
        if not instance_id:
            raise RuntimeError(f"Shadeform create response did not include an instance id: {response}")
        return str(instance_id)

    def get_info(self, instance_id: str) -> InstanceInfo:
        payload = self._request("GET", f"/instances/{urllib.parse.quote(instance_id)}/info")
        return InstanceInfo(
            id=str(payload["id"]),
            status=str(payload["status"]),
            ip=payload.get("ip"),
            ssh_user=str(payload.get("ssh_user") or "shadeform"),
            ssh_port=int(payload.get("ssh_port") or 22),
            cloud=str(payload.get("cloud") or ""),
            region=str(payload.get("region") or ""),
            shade_instance_type=str(payload.get("shade_instance_type") or ""),
            hourly_price_cents=int(payload.get("hourly_price") or 0),
            raw=payload,
        )

    def wait_until_active(
        self,
        instance_id: str,
        *,
        poll_interval_seconds: int = 15,
        timeout_seconds: int = 900,
    ) -> InstanceInfo:
        start = time.time()
        while True:
            info = self.get_info(instance_id)
            elapsed = int(time.time() - start)
            ip_status = f" ip={info.ip}" if info.ip else ""
            print(f"  [{elapsed:>4d}s] status={info.status}{ip_status}", file=sys.stderr)
            if info.status == "active" and info.ip:
                return info
            if info.status in {"deleted", "error"}:
                raise RuntimeError(f"Instance {instance_id} entered {info.status!r}: {info.raw}")
            if elapsed > timeout_seconds:
                raise TimeoutError(f"Instance {instance_id} was not active after {timeout_seconds}s")
            time.sleep(poll_interval_seconds)

    def delete_instance(self, instance_id: str) -> None:
        self._request("POST", f"/instances/{urllib.parse.quote(instance_id)}/delete")

    def _list_gpu_types(
        self,
        gpu_types: str | Sequence[str],
        *,
        num_gpus: int | None,
        available: bool,
    ) -> list[AvailabilityOption]:
        requested_gpu_types = [gpu_types] if isinstance(gpu_types, str) else list(gpu_types)
        options: list[AvailabilityOption] = []
        for gpu_type in requested_gpu_types:
            options.extend(self._query_gpu_type(gpu_type, num_gpus=num_gpus, available=available))
        return options

    def _query_gpu_type(
        self,
        gpu_type: str,
        *,
        num_gpus: int | None,
        available: bool,
    ) -> list[AvailabilityOption]:
        query: dict[str, str | int] = {
            "gpu_type": gpu_type,
            "sort": "price",
            "available": "true" if available else "false",
        }
        if num_gpus is not None:
            query["num_gpus"] = num_gpus
        payload = self._request("GET", f"/instances/types?{urllib.parse.urlencode(query)}")
        parsed: list[AvailabilityOption] = []
        for item in payload.get("instance_types", []):
            config = item.get("configuration", {})
            for availability in item.get("availability", []):
                if bool(availability.get("available")) != available:
                    continue
                parsed.append(
                    AvailabilityOption(
                        cloud=str(item["cloud"]),
                        region=str(availability["region"]),
                        display_name=str(availability.get("display_name") or availability["region"]),
                        shade_instance_type=str(item["shade_instance_type"]),
                        hourly_price_cents=int(item.get("hourly_price") or 0),
                        num_gpus=int(config.get("num_gpus") or 1),
                        gpu_type=str(config.get("gpu_type") or gpu_type),
                        vram_per_gpu_gb=int(config.get("vram_per_gpu_in_gb") or 0),
                        memory_gb=int(config.get("memory_in_gb") or 0),
                        storage_gb=int(config.get("storage_in_gb") or 0),
                        vcpus=int(config.get("vcpus") or 0),
                    )
                )
        return parsed

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "X-API-KEY": self._api_key,
        }

    def _request(self, method: str, path: str, body: Mapping[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self._api_base}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(url, data=data, headers=self._headers(), method=method)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                response_body = response.read().decode("utf-8")
                return json.loads(response_body) if response_body else {}
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8") if exc.fp else ""
            raise RuntimeError(f"Shadeform API {method} {path} failed with {exc.code}: {error_body}") from exc
