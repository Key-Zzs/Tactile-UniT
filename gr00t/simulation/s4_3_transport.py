"""Short, ownership-audited AF_UNIX endpoints for S4.3 policy rollout workers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import socket
import stat
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MAX_ENDPOINT_BYTES = 80
MANIFEST_SCHEMA = "tactile3d-unit.s4-3-runtime-endpoint.v2"
LEASE_SCHEMA = "tactile3d-unit.s4-3-runtime-endpoint-lease.v1"
_ENDPOINT_NAME = re.compile(r"^tu3d_[0-9a-f]{12}_[0-9]+_[0-9a-f]{6}\.sock$")


class EndpointContractError(RuntimeError):
    """The requested endpoint does not satisfy the frozen transport contract."""


class EndpointOwnershipError(EndpointContractError):
    """An endpoint cannot be safely reclaimed or removed."""


@dataclass(frozen=True)
class RuntimeEndpoint:
    path: str
    runtime_root: str
    job_hash: str
    launcher_pid: int
    nonce: str
    encoded_length: int
    ceiling_bytes: int = MAX_ENDPOINT_BYTES

    @property
    def socket_path(self) -> Path:
        return Path(self.path)

    @property
    def lease_path(self) -> Path:
        return self.socket_path.with_suffix(self.socket_path.suffix + ".lease")


def _canonical_digest(identity: dict[str, Any]) -> str:
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _usable_runtime_root(candidate: Path) -> Path | None:
    try:
        candidate.mkdir(mode=0o700, parents=True, exist_ok=True)
        resolved = candidate.resolve(strict=True)
        metadata = resolved.stat()
    except OSError:
        return None
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
        return None
    try:
        if not os.access(resolved, os.W_OK | os.X_OK):
            return None
    except OSError:
        return None
    return resolved


def runtime_root() -> Path:
    """Return a short private runtime directory without repository-path coupling."""

    candidates = []
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        candidates.append(Path(xdg) / "tu3d")
    candidates.append(Path(tempfile.gettempdir()) / f"tu3d-{os.getuid()}")
    for candidate in candidates:
        resolved = _usable_runtime_root(candidate)
        if resolved is not None:
            return resolved
    raise EndpointContractError("no short writable private runtime root is available")


def build_runtime_endpoint(
    *,
    experiment_identity: str,
    task: str,
    variant: str,
    training_seed: int,
    worker_identity: str,
    root: Path | None = None,
    launcher_pid: int | None = None,
    nonce: str | None = None,
) -> RuntimeEndpoint:
    """Construct a collision-resistant endpoint containing no long provenance strings."""

    identity = {
        "experiment_identity": experiment_identity,
        "task": task,
        "variant": variant,
        "training_seed": int(training_seed),
        "worker_identity": worker_identity,
    }
    digest = _canonical_digest(identity)
    pid = os.getpid() if launcher_pid is None else int(launcher_pid)
    token = secrets.token_hex(3) if nonce is None else nonce
    if pid <= 0 or not re.fullmatch(r"[0-9a-f]{6}", token):
        raise EndpointContractError("invalid endpoint process identity or nonce")
    selected_root = runtime_root() if root is None else _usable_runtime_root(root)
    if selected_root is None:
        raise EndpointContractError("requested runtime root is not private and writable")
    name = f"tu3d_{digest}_{pid}_{token}.sock"
    if not _ENDPOINT_NAME.fullmatch(name):
        raise EndpointContractError("generated endpoint name violates the transport contract")
    path = selected_root / name
    encoded_length = len(os.fsencode(path))
    if encoded_length > MAX_ENDPOINT_BYTES:
        raise EndpointContractError(
            f"AF_UNIX endpoint is {encoded_length} encoded bytes; project ceiling is "
            f"{MAX_ENDPOINT_BYTES}"
        )
    return RuntimeEndpoint(
        path=str(path),
        runtime_root=str(selected_root),
        job_hash=digest,
        launcher_pid=pid,
        nonce=token,
        encoded_length=encoded_length,
    )


def _validate_endpoint(endpoint: RuntimeEndpoint) -> None:
    root = Path(endpoint.runtime_root).resolve(strict=True)
    path = endpoint.socket_path
    if path.parent.resolve(strict=True) != root or not _ENDPOINT_NAME.fullmatch(path.name):
        raise EndpointContractError("endpoint escaped the centralized runtime namespace")
    actual_length = len(os.fsencode(path))
    if actual_length != endpoint.encoded_length or actual_length > MAX_ENDPOINT_BYTES:
        raise EndpointContractError("endpoint encoded-length contract mismatch")


def write_endpoint_manifest(
    path: Path,
    endpoint: RuntimeEndpoint,
    provenance: dict[str, Any],
) -> None:
    _validate_endpoint(endpoint)
    value = {
        "schema": MANIFEST_SCHEMA,
        "endpoint": asdict(endpoint),
        "provenance": provenance,
        "rpc_payload_contract": "UNCHANGED_MULTIPROCESSING_CONNECTION_V1",
        "scientific_seed_influence": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_endpoint_manifest(path: Path) -> tuple[RuntimeEndpoint, dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != MANIFEST_SCHEMA:
        raise EndpointContractError("runtime endpoint manifest schema mismatch")
    endpoint = RuntimeEndpoint(**value["endpoint"])
    _validate_endpoint(endpoint)
    if value.get("scientific_seed_influence") is not False:
        raise EndpointContractError("transport identity may not influence scientific seeds")
    return endpoint, value


def _pid_is_active(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_lease(endpoint: RuntimeEndpoint) -> dict[str, Any] | None:
    if not endpoint.lease_path.exists():
        return None
    try:
        lease = json.loads(endpoint.lease_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EndpointOwnershipError("endpoint lease is unreadable") from error
    if (
        lease.get("schema") != LEASE_SCHEMA
        or lease.get("endpoint") != endpoint.path
        or lease.get("uid") != os.getuid()
    ):
        raise EndpointOwnershipError("endpoint lease ownership mismatch")
    return lease


def prepare_server_endpoint(endpoint: RuntimeEndpoint) -> bool:
    """Remove only a verified stale owned socket. Return whether one was reclaimed."""

    _validate_endpoint(endpoint)
    socket_exists = endpoint.socket_path.exists()
    lease = _read_lease(endpoint)
    if not socket_exists and lease is None:
        return False
    if lease is None:
        raise EndpointOwnershipError("refusing to unlink an unregistered endpoint")
    owner_pid = int(lease["pid"])
    if _pid_is_active(owner_pid):
        raise EndpointOwnershipError("endpoint is owned by an active registered server")
    if socket_exists:
        metadata = endpoint.socket_path.lstat()
        if not stat.S_ISSOCK(metadata.st_mode) or metadata.st_ino != int(lease["inode"]):
            raise EndpointOwnershipError("stale endpoint inode/type does not match its lease")
        endpoint.socket_path.unlink()
    endpoint.lease_path.unlink(missing_ok=True)
    return True


def register_server_endpoint(endpoint: RuntimeEndpoint, *, owner_pid: int | None = None) -> None:
    _validate_endpoint(endpoint)
    metadata = endpoint.socket_path.lstat()
    if not stat.S_ISSOCK(metadata.st_mode):
        raise EndpointOwnershipError("listener did not create an AF_UNIX socket")
    pid = os.getpid() if owner_pid is None else int(owner_pid)
    lease = {
        "schema": LEASE_SCHEMA,
        "endpoint": endpoint.path,
        "uid": os.getuid(),
        "pid": pid,
        "inode": metadata.st_ino,
    }
    descriptor = os.open(endpoint.lease_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(lease, output, sort_keys=True)
        output.write("\n")


def cleanup_server_endpoint(
    endpoint: RuntimeEndpoint,
    *,
    owner_pid: int | None = None,
    allow_unregistered_own_socket: bool = False,
) -> bool:
    """Remove this server's endpoint, never an arbitrary or active foreign socket."""

    _validate_endpoint(endpoint)
    pid = os.getpid() if owner_pid is None else int(owner_pid)
    lease = _read_lease(endpoint)
    if lease is None:
        if not endpoint.socket_path.exists():
            return False
        if not allow_unregistered_own_socket:
            raise EndpointOwnershipError("refusing unregistered socket cleanup")
        metadata = endpoint.socket_path.lstat()
        if not stat.S_ISSOCK(metadata.st_mode):
            raise EndpointOwnershipError("unregistered endpoint is not a socket")
        endpoint.socket_path.unlink()
        return True
    if int(lease["pid"]) != pid:
        raise EndpointOwnershipError("server may clean only its own registered endpoint")
    if endpoint.socket_path.exists():
        metadata = endpoint.socket_path.lstat()
        if not stat.S_ISSOCK(metadata.st_mode) or metadata.st_ino != int(lease["inode"]):
            raise EndpointOwnershipError("endpoint changed after server registration")
        endpoint.socket_path.unlink()
    endpoint.lease_path.unlink(missing_ok=True)
    return True


def platform_af_unix_payload_limit() -> int:
    """Measure the active runtime's pathname payload limit with disposable sockets."""

    root = Path(tempfile.gettempdir())
    prefix = f"tu3d-limit-{os.getpid()}-"
    last_success = 0
    for payload_length in range(len(os.fsencode(root / prefix)), 256):
        padding = "x" * max(1, payload_length - len(os.fsencode(root / prefix)))
        path = root / f"{prefix}{padding}"
        while len(os.fsencode(path)) > payload_length:
            padding = padding[:-1]
            path = root / f"{prefix}{padding}"
        endpoint = str(path)
        candidate = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            candidate.bind(endpoint)
        except OSError:
            candidate.close()
            break
        else:
            last_success = len(os.fsencode(path))
            candidate.close()
            path.unlink(missing_ok=True)
    return last_success
