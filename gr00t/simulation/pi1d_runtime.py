"""Causal online tactile and Contact-State transport for S4.3-PI1D."""

from __future__ import annotations

from dataclasses import dataclass
import socket
import struct
from pathlib import Path
from typing import Callable

import numpy as np

from gr00t.simulation.s4_3_pi1 import CONTACT_STATE_DIM, HISTORY_STEPS, TACTILE_DIM, OnlineTactileHistory


_HEADER = struct.Struct("!I")
_HISTORY_BYTES = HISTORY_STEPS * TACTILE_DIM * np.dtype(np.float32).itemsize
_CONTACT_STATE_BYTES = CONTACT_STATE_DIM * np.dtype(np.float32).itemsize


def recv_exact(connection: socket.socket, size: int) -> bytes:
    """Read exactly ``size`` bytes or fail on a truncated local transport."""

    chunks = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError(f"Contact-State sidecar closed with {remaining} bytes outstanding")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class ContactStateUnixClient:
    """Minimal read-only Unix-socket client for the frozen torch E_T sidecar."""

    def __init__(self, socket_path: Path, timeout_seconds: float = 30.0) -> None:
        self.socket_path = Path(socket_path)
        self.connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.connection.settimeout(timeout_seconds)
        self.connection.connect(str(self.socket_path))
        self.requests = 0
        self.errors: list[str] = []

    def encode(self, history: np.ndarray) -> np.ndarray:
        value = np.asarray(history, dtype=np.float32)
        if value.shape != (HISTORY_STEPS, TACTILE_DIM) or not np.isfinite(value).all():
            raise ValueError(f"Contact-State history must be finite [{HISTORY_STEPS},{TACTILE_DIM}]")
        payload = np.ascontiguousarray(value).tobytes()
        if len(payload) != _HISTORY_BYTES:
            raise RuntimeError("Contact-State request byte contract changed")
        try:
            self.connection.sendall(_HEADER.pack(len(payload)) + payload)
            response_size = _HEADER.unpack(recv_exact(self.connection, _HEADER.size))[0]
            if response_size != _CONTACT_STATE_BYTES:
                raise RuntimeError(f"Contact-State response has {response_size} bytes")
            result = np.frombuffer(recv_exact(self.connection, response_size), dtype=np.float32).copy()
            if result.shape != (CONTACT_STATE_DIM,) or not np.isfinite(result).all():
                raise RuntimeError("Contact-State response is not finite [256]")
            self.requests += 1
            return result
        except Exception as error:
            self.errors.append(f"{type(error).__name__}: {error}")
            raise

    def close(self) -> None:
        self.connection.close()


@dataclass
class CausalContactRuntime:
    """Maintain the exact 26-tick LEFT_REPEAT_FIRST online input to E_T."""

    encoder: ContactStateUnixClient | Callable[[np.ndarray], np.ndarray]

    def __post_init__(self) -> None:
        self.history = OnlineTactileHistory()
        self.control_tick = -1
        self._cached_tick = -2
        self._cached_state: np.ndarray | None = None

    def reset(self, tactile: np.ndarray) -> np.ndarray:
        history = self.history.reset(tactile)
        self.control_tick = 0
        self._cached_tick = -1
        self._cached_state = None
        return history

    def append(self, tactile: np.ndarray) -> np.ndarray:
        history = self.history.append(tactile)
        self.control_tick += 1
        self._cached_state = None
        return history

    def contact_state(self) -> np.ndarray:
        if self._cached_state is None or self._cached_tick != self.control_tick:
            history = self.history.value()
            if hasattr(self.encoder, "encode"):
                value = self.encoder.encode(history)  # type: ignore[union-attr]
            else:
                value = self.encoder(history)  # type: ignore[operator]
            value = np.asarray(value, dtype=np.float32)
            if value.shape != (CONTACT_STATE_DIM,) or not np.isfinite(value).all():
                raise ValueError("online E_T output must be finite [256]")
            self._cached_state = value.copy()
            self._cached_tick = self.control_tick
        return self._cached_state.copy()

    def current_history(self) -> np.ndarray:
        return self.history.value()
