"""Subprocess-based SSH/SCP helpers for ephemeral Shadeform instances."""

from __future__ import annotations

import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence


SSH_OPTIONS = (
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "LogLevel=ERROR",
)

DEFAULT_TAR_EXCLUDES = (
    "__pycache__",
    "*.pyc",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "runs",
    "logs",
)


def wait_for_ssh(
    *,
    key_path: Path,
    user: str,
    ip: str,
    port: int = 22,
    retries: int = 30,
    delay_seconds: int = 10,
) -> None:
    for attempt in range(1, retries + 1):
        result = subprocess.run(
            [*_ssh_base(key_path, user, ip, port), "echo ok"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            print(f"  SSH ready after attempt {attempt}", file=sys.stderr)
            return
        print(f"  SSH attempt {attempt}/{retries} failed; retrying in {delay_seconds}s ...", file=sys.stderr)
        time.sleep(delay_seconds)
    raise ConnectionError(f"Could not establish SSH to {user}@{ip}:{port}")


def run_remote(
    *,
    key_path: Path,
    user: str,
    ip: str,
    command: str,
    port: int = 22,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    command_line = [*_ssh_base(key_path, user, ip, port), command]
    completed = subprocess.run(command_line, text=True)
    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command_line)
    return completed


def upload(
    *,
    key_path: Path,
    user: str,
    ip: str,
    local_path: Path,
    remote_path: str,
    port: int = 22,
    exclude_patterns: Sequence[str] = DEFAULT_TAR_EXCLUDES,
) -> None:
    if local_path.is_dir():
        _upload_dir(
            key_path=key_path,
            user=user,
            ip=ip,
            local_dir=local_path,
            remote_path=remote_path,
            port=port,
            exclude_patterns=exclude_patterns,
        )
        return
    remote_parent = str(Path(remote_path).parent)
    run_remote(
        key_path=key_path,
        user=user,
        ip=ip,
        port=port,
        command=f"mkdir -p {shlex.quote(remote_parent)}",
    )
    target = f"{user}@{ip}:{remote_path}"
    subprocess.run([*_scp_base(key_path, port), str(local_path), target], check=True, text=True)


def download(
    *,
    key_path: Path,
    user: str,
    ip: str,
    remote_path: str,
    local_path: Path,
    port: int = 22,
) -> bool:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    source = f"{user}@{ip}:{remote_path}"
    completed = subprocess.run(
        [*_scp_base(key_path, port), "-r", source, str(local_path)],
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _upload_dir(
    *,
    key_path: Path,
    user: str,
    ip: str,
    local_dir: Path,
    remote_path: str,
    port: int,
    exclude_patterns: Sequence[str],
) -> None:
    local_dir = local_dir.resolve()
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as temp_file:
        archive_path = Path(temp_file.name)
    try:
        tar_command = ["tar", "czf", str(archive_path)]
        for pattern in exclude_patterns:
            tar_command.append(f"--exclude={pattern}")
        tar_command.extend(["-C", str(local_dir.parent), local_dir.name])
        subprocess.run(tar_command, check=True, text=True)

        remote_archive = f"/tmp/{archive_path.name}"
        subprocess.run(
            [*_scp_base(key_path, port), str(archive_path), f"{user}@{ip}:{remote_archive}"],
            check=True,
            text=True,
        )
        remote_parent = str(Path(remote_path).parent)
        extract_command = (
            f"mkdir -p {shlex.quote(remote_parent)} && "
            f"tar xzf {shlex.quote(remote_archive)} -C {shlex.quote(remote_parent)} && "
            f"rm -f {shlex.quote(remote_archive)}"
        )
        run_remote(key_path=key_path, user=user, ip=ip, port=port, command=extract_command)
    finally:
        archive_path.unlink(missing_ok=True)


def _ssh_base(key_path: Path, user: str, ip: str, port: int) -> list[str]:
    return ["ssh", "-i", str(key_path), "-p", str(port), *SSH_OPTIONS, f"{user}@{ip}"]


def _scp_base(key_path: Path, port: int) -> list[str]:
    return ["scp", "-i", str(key_path), "-P", str(port), *SSH_OPTIONS]
