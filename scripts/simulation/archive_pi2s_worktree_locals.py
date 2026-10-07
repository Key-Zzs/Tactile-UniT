#!/usr/bin/env python3
"""Archive PI2B worktree-local evidence without following aliases.

This utility is deliberately conservative:

* it reads, but never changes, the three source worktrees;
* symbolic links are recreated as links and their referents are never copied;
* every regular file is SHA-256 verified in a staging directory;
* a final name is claimed with ``mkdir`` and advances monotonically from
  ``INCOMPLETE`` to ``COMPLETE`` without hard links or final-name replacement; and
* an existing final archive is verified in place and is never overwritten.

The resulting copy lives on the shared experiment NAS.  It is a retirement
archive for worktree-local state, not an independent disaster-recovery backup.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import datetime as dt
import errno
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
SOURCE_ROOTS = (
    REPOSITORY_ROOT.with_name(f"{REPOSITORY_ROOT.name}-pi2b-policy"),
    REPOSITORY_ROOT.with_name(f"{REPOSITORY_ROOT.name}-contact-tokenizer"),
    REPOSITORY_ROOT.with_name(f"{REPOSITORY_ROOT.name}-pi2b-teacher"),
)
DEFAULT_DESTINATION = (
    REPOSITORY_ROOT / ".local" / "experiments" / "simulation" / "s4_3_pi2s" / "retirement_backups"
)
FORMAT_VERSION = 3
BUFFER_SIZE = 8 * 1024 * 1024
NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
DIRECTORY = getattr(os, "O_DIRECTORY", 0)
PUBLICATION_INCOMPLETE = "INCOMPLETE"
PUBLICATION_COMPLETE = "COMPLETE"
PUBLICATION_CONTENT = "CONTENT"
PUBLICATION_BUNDLE = "BUNDLE"
PUBLICATION_POLICY = (
    "nfs_mkdir_claim;private_content_slot;same_filesystem_bundle_rename;"
    "mkdir_complete_marker;no_hardlinks;no_final_name_replace"
)
AGGREGATE_PUBLICATION_NAME = "worktree_local_retirement_audit.publication"
AGGREGATE_AUDIT_NAME = "worktree_local_retirement_audit.json"


class ArchiveError(RuntimeError):
    """Raised when a source or archive fails a fail-closed check."""


class ProgressReporter:
    """Persist machine-readable progress and periodically emit a log line."""

    def __init__(self, path: Path, *, goal_uuid: str, producer: dict[str, str]) -> None:
        frozen_producer = copy_producer_identity(producer)
        self.path = path
        self._phase_started_monotonic: float | None = None
        self.state: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "pid": os.getpid(),
            "goal_uuid": goal_uuid,
            "source_commit": frozen_producer["source_commit"],
            "producer": frozen_producer,
            "status": "STARTING",
            "started_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "recovery": {
                "completed_final_archives_are_idempotently_verified": True,
                "incomplete_final_archives_are_verified_and_completed": True,
                "incomplete_staging_is_retained": True,
                "partial_staging_auto_resumed": False,
                "rerun_action": (
                    "verify COMPLETE publications; verify and complete an adopted bundle; "
                    "or create a new staging bundle for an empty INCOMPLETE claim"
                ),
            },
        }
        self._last_persist_time = 0.0
        self._last_persist_bytes = 0
        self.persist(force=True)

    def _atomic_write(self) -> None:
        temporary = self.path.with_name(f".{self.path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
        data = (json.dumps(self.state, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(descriptor)
        os.replace(temporary, self.path)
        fsync_directory(self.path.parent)

    def persist(self, *, force: bool = False) -> None:
        now = time.monotonic()
        completed = int(self.state.get("phase_bytes_completed", 0))
        total = int(self.state.get("phase_bytes_total", 0))
        if self._phase_started_monotonic is not None:
            elapsed = max(0.0, now - self._phase_started_monotonic)
            rate = completed / elapsed if elapsed > 0.0 else 0.0
            remaining = max(0, total - completed)
            self.state["phase_elapsed_seconds"] = elapsed
            self.state["phase_rate_bytes_per_second"] = rate
            self.state["phase_eta_seconds"] = remaining / rate if rate > 0.0 else None
        should_write = (
            force
            or now - self._last_persist_time >= 5.0
            or completed - self._last_persist_bytes >= 64 * 1024 * 1024
            or (total > 0 and completed == total)
        )
        if not should_write:
            return
        self.state["updated_at_utc"] = utc_now()
        self._atomic_write()
        self._last_persist_time = now
        self._last_persist_bytes = completed
        if self.state.get("phase"):
            percent = 100.0 if total == 0 else 100.0 * completed / total
            print(
                "PROGRESS "
                f"source={self.state.get('source_name')!a} "
                f"phase={self.state.get('phase')} "
                f"bytes={completed}/{total} ({percent:.1f}%) "
                f"file={self.state.get('current_path')!a}",
                flush=True,
            )

    def begin_phase(
        self,
        source_root: Path,
        phase: str,
        total_bytes: int,
        *,
        staging: Path | None = None,
    ) -> None:
        self.state.update(
            {
                "status": "RUNNING",
                "source_name": source_root.name,
                "source_worktree": str(source_root),
                "phase": phase,
                "phase_bytes_completed": 0,
                "phase_bytes_total": total_bytes,
                "current_path": None,
                "staging": str(staging) if staging is not None else None,
                "phase_started_at_utc": utc_now(),
            }
        )
        self._phase_started_monotonic = time.monotonic()
        self._last_persist_bytes = 0
        self.persist(force=True)

    def advance(self, byte_count: int, current_path: Path) -> None:
        self.state["phase_bytes_completed"] += byte_count
        self.state["current_path"] = str(current_path)
        self.persist()

    def note_archive_complete(self, final: Path, archive_status: str) -> None:
        completed = self.state.setdefault("completed_archives", [])
        completed.append({"archive": str(final), "archive_status": archive_status})
        self.persist(force=True)

    def finish(self, status: str, *, error: str | None = None) -> None:
        self.state["status"] = status
        self.state["error"] = error
        self.state["finished_at_utc"] = utc_now()
        self.persist(force=True)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(
    path: Path,
    progress: Callable[[int, Path], None] | None = None,
    *,
    no_follow: bool = False,
    expected_stat: os.stat_result | None = None,
) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | (NOFOLLOW if no_follow else 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ArchiveError(f"not a regular file while hashing: {path}")
        if expected_stat is not None:
            assert_same_file_identity(expected_stat, opened, path, "opened for hashing")
        stream = os.fdopen(descriptor, "rb", buffering=0, closefd=False)
        with stream:
            for block in iter(lambda: stream.read(BUFFER_SIZE), b""):
                digest.update(block)
                if progress is not None:
                    progress(len(block), path)
        closed_view = os.fstat(descriptor)
        if expected_stat is not None:
            assert_stable_regular_file(expected_stat, closed_view, path, "changed while hashing")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def copy_producer_identity(producer: dict[str, str]) -> dict[str, str]:
    """Validate and detach the producer identity embedded in archive evidence."""

    required = {"path", "repository_relative_path", "sha256", "source_commit"}
    if set(producer) != required:
        raise ArchiveError(f"producer identity fields must be exactly {sorted(required)!r}")
    if not all(isinstance(producer[key], str) for key in required):
        raise ArchiveError("producer identity values must be strings")
    if len(producer["sha256"]) != 64 or any(
        character not in "0123456789abcdef" for character in producer["sha256"]
    ):
        raise ArchiveError("producer sha256 must be 64 lowercase hexadecimal characters")
    if len(producer["source_commit"]) != 40 or any(
        character not in "0123456789abcdef" for character in producer["source_commit"]
    ):
        raise ArchiveError("producer source_commit must be 40 lowercase hexadecimal characters")
    relative = producer["repository_relative_path"]
    if (
        not relative
        or relative.startswith("/")
        or any(component in {"", ".", ".."} for component in relative.split("/"))
    ):
        raise ArchiveError("producer repository_relative_path is not canonical and relative")
    return dict(producer)


def verify_producer_source_commit(
    source_commit: str,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    script_path: Path = SCRIPT_PATH,
) -> dict[str, str]:
    """Bind this exact executable to its blob in the requested current HEAD."""

    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise ArchiveError("source commit must be 40 lowercase hexadecimal characters")
    repository_root = repository_root.absolute()
    script_path = script_path.absolute()
    try:
        relative_path = script_path.relative_to(repository_root).as_posix()
    except ValueError as exc:
        raise ArchiveError(f"producer script is outside repository root: {script_path}") from exc

    head_process = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD^{commit}"],
        cwd=repository_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if head_process.returncode != 0:
        raise ArchiveError(f"cannot resolve repository HEAD: {repository_root}")
    actual_head = head_process.stdout.strip()
    if source_commit != actual_head:
        raise ArchiveError(
            f"source commit mismatch: requested {source_commit}, actual {actual_head}"
        )

    blob_process = subprocess.run(
        ["git", "cat-file", "blob", f"{source_commit}:{relative_path}"],
        cwd=repository_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if blob_process.returncode != 0:
        raise ArchiveError(
            "producer script is absent from the requested source commit: " f"{relative_path}"
        )
    committed_sha256 = sha256_bytes(blob_process.stdout)
    worktree_sha256 = sha256_file(script_path, no_follow=True)
    if worktree_sha256 != committed_sha256:
        raise ArchiveError(
            "producer script differs from its source-commit blob: "
            f"{relative_path} ({worktree_sha256} != {committed_sha256})"
        )
    return copy_producer_identity(
        {
            "path": str(script_path),
            "repository_relative_path": relative_path,
            "sha256": worktree_sha256,
            "source_commit": source_commit,
        }
    )


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def write_json_fsync(path: Path, value: Any) -> None:
    data = (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def assert_same_file_identity(
    expected: os.stat_result,
    actual: os.stat_result,
    path: Path,
    action: str,
) -> None:
    fields = ("st_dev", "st_ino", "st_mode")
    if any(getattr(expected, field) != getattr(actual, field) for field in fields):
        raise ArchiveError(f"source identity changed when {action}: {path}")


def assert_stable_regular_file(
    expected: os.stat_result,
    actual: os.stat_result,
    path: Path,
    action: str,
) -> None:
    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_gid",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(expected, field) != getattr(actual, field) for field in fields):
        raise ArchiveError(f"{action}: {path}")


def assert_stable_entry(
    expected: os.stat_result,
    actual: os.stat_result,
    path: Path,
    action: str,
) -> None:
    """Reject replacement or metadata changes for any supported entry type."""

    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_gid",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(expected, field) != getattr(actual, field) for field in fields):
        raise ArchiveError(f"{action}: {path}")


def fsencoded_base64(value: str) -> str:
    """Encode a filesystem string losslessly, including surrogateescaped bytes."""

    return base64.b64encode(os.fsencode(value)).decode("ascii")


def entry_type(mode: int) -> str:
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISLNK(mode):
        return "symlink"
    raise ArchiveError(f"unsupported source entry type: mode={mode:o}")


def observed_entry_type(mode: int) -> str:
    """Describe an alias referent without restricting external entry types."""

    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISFIFO(mode):
        return "fifo"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISBLK(mode):
        return "block_device"
    if stat.S_ISCHR(mode):
        return "character_device"
    return "unknown"


def path_sort_key(path: str) -> bytes:
    return os.fsencode(path)


def relative_components(relative: str) -> tuple[str, ...]:
    """Parse one canonical, relative manifest path without normalising it."""

    if relative == ".":
        return ()
    if not relative or relative.startswith("/") or "\x00" in relative:
        raise ArchiveError(f"unsafe manifest path: {relative!r}")
    components = tuple(relative.split("/"))
    if any(component in {"", ".", ".."} for component in components):
        raise ArchiveError(f"unsafe manifest path: {relative!r}")
    return components


def display_path(root: Path, relative: str) -> Path:
    return root.joinpath(*relative_components(relative))


def open_absolute_directory_no_follow(path: Path) -> int:
    """Open an absolute directory path without following any component."""

    absolute = path.absolute()
    if absolute != path or not absolute.is_absolute():
        raise ArchiveError(f"tree root must be absolute: {path}")
    components = absolute.parts
    descriptor = os.open(components[0], os.O_RDONLY | DIRECTORY | NOFOLLOW)
    try:
        for component in components[1:]:
            if component in {"", ".", ".."}:
                raise ArchiveError(f"unsafe tree root component: {path}")
            next_descriptor = os.open(
                component,
                os.O_RDONLY | DIRECTORY | NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def open_root_directory(root: Path) -> tuple[int, os.stat_result]:
    """Open a tree root without following any path component."""

    if NOFOLLOW == 0:
        raise ArchiveError("this platform lacks O_NOFOLLOW; refusing unsafe traversal")
    before = root.lstat()
    if not stat.S_ISDIR(before.st_mode):
        raise ArchiveError(f"tree root is not a real directory: {root}")
    descriptor = open_absolute_directory_no_follow(root)
    try:
        opened = os.fstat(descriptor)
        assert_same_file_identity(before, opened, root, "opening directory")
        if not stat.S_ISDIR(opened.st_mode):
            raise ArchiveError(f"tree root is not a real directory: {root}")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, before


def open_parent_directory(
    root_descriptor: int, relative: str, display_root: Path
) -> tuple[int, str]:
    """Open every parent component relative to an anchored root, never aliases."""

    components = relative_components(relative)
    if not components:
        raise ArchiveError("the tree root has no parent-relative leaf")
    descriptor = os.dup(root_descriptor)
    try:
        for component in components[:-1]:
            next_descriptor = os.open(
                component,
                os.O_RDONLY | DIRECTORY | NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
            if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise ArchiveError(
                    f"source parent changed type: {display_path(display_root, relative)}"
                )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, components[-1]


def open_relative_directory(root_descriptor: int, relative: str, display_root: Path) -> int:
    if relative == ".":
        return os.dup(root_descriptor)
    parent_descriptor, leaf = open_parent_directory(root_descriptor, relative, display_root)
    try:
        descriptor = os.open(
            leaf,
            os.O_RDONLY | DIRECTORY | NOFOLLOW,
            dir_fd=parent_descriptor,
        )
    finally:
        os.close(parent_descriptor)
    try:
        is_directory = stat.S_ISDIR(os.fstat(descriptor).st_mode)
    except BaseException:
        os.close(descriptor)
        raise
    if not is_directory:
        os.close(descriptor)
        raise ArchiveError(f"source directory changed type: {display_path(display_root, relative)}")
    return descriptor


def hash_regular_at(
    directory_descriptor: int,
    name: str,
    expected_stat: os.stat_result,
    path: Path,
    reporter: ProgressReporter | None,
) -> str:
    """Hash a leaf opened relative to a trusted parent directory descriptor."""

    descriptor = os.open(name, os.O_RDONLY | NOFOLLOW, dir_fd=directory_descriptor)
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ArchiveError(f"not a regular file while hashing: {path}")
        assert_same_file_identity(expected_stat, opened, path, "opened for hashing")
        with os.fdopen(descriptor, "rb", buffering=0, closefd=False) as stream:
            for block in iter(lambda: stream.read(BUFFER_SIZE), b""):
                digest.update(block)
                if reporter is not None:
                    reporter.advance(len(block), path)
        assert_stable_regular_file(
            expected_stat,
            os.fstat(descriptor),
            path,
            "source changed while hashing",
        )
        path_after = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        assert_stable_regular_file(expected_stat, path_after, path, "source changed while hashing")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def record_from_stat(
    relative: str,
    info: os.stat_result,
    *,
    target: str | None,
    digest: str | None,
) -> dict[str, Any]:
    kind = entry_type(info.st_mode)
    return {
        "path": relative,
        "path_fsencoded_base64": fsencoded_base64(relative),
        "type": kind,
        "target": target,
        "target_fsencoded_base64": (fsencoded_base64(target) if target is not None else None),
        "mode": f"{stat.S_IMODE(info.st_mode):04o}",
        "uid": info.st_uid,
        "gid": info.st_gid,
        "mtime_ns": info.st_mtime_ns,
        # Directory inode sizes are filesystem-specific (Btrfs source versus
        # NFS destination), so the explicitly present value is null there.
        "size": None if kind == "directory" else info.st_size,
        "sha256": digest,
    }


def scan_tree(
    root: Path,
    *,
    include_hash: bool,
    reporter: ProgressReporter | None = None,
) -> list[dict[str, Any]]:
    root_descriptor, root_before = open_root_directory(root)
    records: list[dict[str, Any]] = [record_from_stat(".", root_before, target=None, digest=None)]
    directory_stats: dict[str, os.stat_result] = {".": root_before}
    pending = ["."]
    try:
        while pending:
            relative_directory = pending.pop()
            descriptor = open_relative_directory(root_descriptor, relative_directory, root)
            directory_path = display_path(root, relative_directory)
            try:
                assert_same_file_identity(
                    directory_stats[relative_directory],
                    os.fstat(descriptor),
                    directory_path,
                    "opening directory during scan",
                )
                children = list(os.scandir(descriptor))
                children.sort(key=lambda item: os.fsencode(item.name))
                child_directories: list[str] = []
                for child in children:
                    relative = (
                        child.name
                        if relative_directory == "."
                        else f"{relative_directory}/{child.name}"
                    )
                    path = display_path(root, relative)
                    info = child.stat(follow_symlinks=False)
                    kind = entry_type(info.st_mode)
                    target: str | None = None
                    digest: str | None = None
                    if kind == "directory":
                        child_descriptor = os.open(
                            child.name,
                            os.O_RDONLY | DIRECTORY | NOFOLLOW,
                            dir_fd=descriptor,
                        )
                        try:
                            assert_same_file_identity(
                                info,
                                os.fstat(child_descriptor),
                                path,
                                "opening directory during scan",
                            )
                        finally:
                            os.close(child_descriptor)
                        directory_stats[relative] = info
                        child_directories.append(relative)
                    elif kind == "file" and include_hash:
                        digest = hash_regular_at(descriptor, child.name, info, path, reporter)
                    elif kind == "symlink":
                        target = os.readlink(child.name, dir_fd=descriptor)
                        after = os.stat(
                            child.name,
                            dir_fd=descriptor,
                            follow_symlinks=False,
                        )
                        assert_stable_entry(info, after, path, "symlink changed while scanning")
                    records.append(record_from_stat(relative, info, target=target, digest=digest))
                pending.extend(reversed(child_directories))
            finally:
                os.close(descriptor)

        # Re-open every directory through the still-open root descriptor.  This
        # catches renamed/replaced parent components after their enumeration.
        for relative, expected in directory_stats.items():
            descriptor = open_relative_directory(root_descriptor, relative, root)
            try:
                assert_stable_entry(
                    expected,
                    os.fstat(descriptor),
                    display_path(root, relative),
                    "directory changed while scanning",
                )
            finally:
                os.close(descriptor)
        assert_stable_entry(
            root_before,
            root.lstat(),
            root,
            "tree root changed while scanning",
        )
    finally:
        os.close(root_descriptor)

    records.sort(key=lambda row: path_sort_key(row["path"]))
    paths = [row["path"] for row in records]
    if len(paths) != len(set(paths)):
        raise ArchiveError(f"duplicate relative path while scanning {root}")
    return records


def metadata_projection(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in row.items() if key != "sha256"} for row in records]


def payload_projection(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fields that must survive a cross-filesystem retirement copy.

    NFS identity mapping controls the destination uid/gid, so source ownership is
    preserved in the manifest rather than replayed.  Every requested scientific
    identity and filesystem field (path/type/link target/mode/mtime/size/hash)
    remains strict.
    """

    return [
        {key: value for key, value in row.items() if key not in {"uid", "gid"}} for row in records
    ]


def record_map(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["path"]: row for row in records}


def archive_name(source_root: Path) -> str:
    return f"{source_root.name}.local_archive"


def safe_payload_path(payload_root: Path, relative: str) -> Path:
    return payload_root.joinpath(*relative_components(relative))


def assert_stat_matches_record(
    info: os.stat_result,
    record: dict[str, Any],
    path: Path,
    action: str,
) -> None:
    actual_type = entry_type(info.st_mode)
    expected_values = {
        "type": actual_type,
        "mode": f"{stat.S_IMODE(info.st_mode):04o}",
        "uid": info.st_uid,
        "gid": info.st_gid,
        "mtime_ns": info.st_mtime_ns,
        "size": None if actual_type == "directory" else info.st_size,
    }
    for key, actual in expected_values.items():
        if record.get(key) != actual:
            raise ArchiveError(f"source metadata changed when {action} ({key}): {path}")


def read_stable_symlink_at(
    root_descriptor: int,
    source_root: Path,
    relative: str,
    expected: dict[str, Any],
) -> str:
    source = display_path(source_root, relative)
    parent_descriptor, name = open_parent_directory(root_descriptor, relative, source_root)
    try:
        before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if not stat.S_ISLNK(before.st_mode):
            raise ArchiveError(f"source changed type before symlink copy: {source}")
        assert_stat_matches_record(before, expected, source, "reading symlink")
        target = os.readlink(name, dir_fd=parent_descriptor)
        after = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        assert_stable_entry(before, after, source, "symlink changed while copying")
    finally:
        os.close(parent_descriptor)
    if target != expected["target"]:
        raise ArchiveError(f"symlink changed before copy: {source}")
    return target


def copy_regular_file(
    source_root_descriptor: int,
    source_root: Path,
    relative: str,
    destination: Path,
    expected: dict[str, Any],
    reporter: ProgressReporter | None,
) -> str:
    source = display_path(source_root, relative)
    parent_descriptor, name = open_parent_directory(source_root_descriptor, relative, source_root)
    source_descriptor: int | None = None
    destination_descriptor: int | None = None
    digest = hashlib.sha256()
    try:
        before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise ArchiveError(f"source changed type before copy: {source}")
        assert_stat_matches_record(before, expected, source, "opening for copy")
        source_descriptor = os.open(name, os.O_RDONLY | NOFOLLOW, dir_fd=parent_descriptor)
        opened = os.fstat(source_descriptor)
        assert_same_file_identity(before, opened, source, "opening for copy")
        destination_descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with (
            os.fdopen(source_descriptor, "rb", buffering=0, closefd=False) as input_stream,
            os.fdopen(destination_descriptor, "wb", closefd=False) as output_stream,
        ):
            for block in iter(lambda: input_stream.read(BUFFER_SIZE), b""):
                digest.update(block)
                output_stream.write(block)
                if reporter is not None:
                    reporter.advance(len(block), source)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        assert_stable_regular_file(
            before, os.fstat(source_descriptor), source, "source changed while copying"
        )
        after = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        assert_stable_regular_file(before, after, source, "source changed while copying")
    finally:
        if source_descriptor is not None:
            os.close(source_descriptor)
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        os.close(parent_descriptor)
    actual = digest.hexdigest()
    if actual != expected["sha256"]:
        raise ArchiveError(
            f"source digest changed while copying {source}: {actual} != {expected['sha256']}"
        )
    return actual


def fsync_regular_file(path: Path) -> None:
    if NOFOLLOW == 0:
        raise ArchiveError("this platform lacks O_NOFOLLOW; refusing unsafe fsync")
    descriptor = os.open(path, os.O_RDONLY | NOFOLLOW)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ArchiveError(f"not a regular file while fsyncing: {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def set_metadata(path: Path, record: dict[str, Any]) -> None:
    mode = int(record["mode"], 8)
    follow = record["type"] != "symlink"
    if follow:
        os.chmod(path, mode, follow_symlinks=True)
    # uid/gid are recorded in the source manifest but deliberately not replayed.
    # This NFSv3 export maps client ownership and rejects lchown on symlinks.
    try:
        os.utime(
            path,
            ns=(record["mtime_ns"], record["mtime_ns"]),
            follow_symlinks=follow,
        )
    except (NotImplementedError, OSError) as exc:
        if record["type"] != "symlink":
            raise ArchiveError(f"cannot preserve mtime for {path}: {exc}") from exc


def populate_payload(
    source_local: Path,
    payload_root: Path,
    records: list[dict[str, Any]],
    reporter: ProgressReporter | None = None,
) -> None:
    by_path = record_map(records)
    root_record = by_path["."]
    payload_root.mkdir(mode=0o700, parents=False, exist_ok=False)
    payload_root.chmod(0o700)

    directories = [row for row in records if row["type"] == "directory" and row["path"] != "."]
    directories.sort(
        key=lambda row: (
            len(relative_components(row["path"])),
            path_sort_key(row["path"]),
        )
    )
    for record in directories:
        destination = safe_payload_path(payload_root, record["path"])
        destination.mkdir(mode=0o700)
        destination.chmod(0o700)

    source_root_descriptor, source_root_before = open_root_directory(source_local)
    try:
        assert_stat_matches_record(
            source_root_before,
            root_record,
            source_local,
            "opening source root for copy",
        )
        for record in records:
            if record["path"] == "." or record["type"] == "directory":
                continue
            destination = safe_payload_path(payload_root, record["path"])
            if record["type"] == "file":
                copy_regular_file(
                    source_root_descriptor,
                    source_local,
                    record["path"],
                    destination,
                    record,
                    reporter,
                )
                set_metadata(destination, record)
                fsync_regular_file(destination)
            elif record["type"] == "symlink":
                current_target = read_stable_symlink_at(
                    source_root_descriptor,
                    source_local,
                    record["path"],
                    record,
                )
                os.symlink(current_target, destination)
                set_metadata(destination, record)
            else:
                raise ArchiveError(f"unexpected record type: {record['type']}")
        assert_stable_entry(
            source_root_before,
            os.fstat(source_root_descriptor),
            source_local,
            "source root changed while copying",
        )
        assert_stable_entry(
            source_root_before,
            source_local.lstat(),
            source_local,
            "source root path changed while copying",
        )
    finally:
        os.close(source_root_descriptor)

    # Child creation changes directory mtimes.  Restore directories bottom-up.
    directories.append(root_record)
    directories.sort(
        key=lambda row: (
            len(relative_components(row["path"])),
            path_sort_key(row["path"]),
        ),
        reverse=True,
    )
    for record in directories:
        destination = safe_payload_path(payload_root, record["path"])
        set_metadata(destination, record)
        fsync_directory(destination)


def verify_payload(
    payload_root: Path,
    expected: list[dict[str, Any]],
    reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    actual = scan_tree(payload_root, include_hash=True, reporter=reporter)
    expected_projected = payload_projection(expected)
    actual_projected = payload_projection(actual)
    if actual_projected != expected_projected:
        expected_by_path = record_map(expected_projected)
        actual_by_path = record_map(actual_projected)
        all_paths = sorted(set(expected_by_path) | set(actual_by_path), key=path_sort_key)
        differences = []
        for relative in all_paths:
            if expected_by_path.get(relative) != actual_by_path.get(relative):
                differences.append(
                    {
                        "path": relative,
                        "expected": expected_by_path.get(relative),
                        "actual": actual_by_path.get(relative),
                    }
                )
            if len(differences) == 10:
                break
        raise ArchiveError(
            "payload verification failed: "
            + json.dumps(differences, ensure_ascii=True, sort_keys=True)
        )
    counts = Counter(row["type"] for row in actual)
    return {
        "status": "PASS",
        "entry_count_including_root": len(actual),
        "regular_file_count": counts["file"],
        "directory_count_including_root": counts["directory"],
        "symlink_count": counts["symlink"],
        "regular_file_bytes": sum(row["size"] for row in actual if row["type"] == "file"),
        "source_uid_gid_recorded_in_manifest": True,
        "payload_uid_gid_reapplied": False,
        "destination_uid_gid_is_not_archive_identity": True,
    }


def resolved_link_path(link_path: Path, raw_target: str) -> Path:
    target = Path(raw_target)
    if not target.is_absolute():
        target = link_path.parent / target
    return Path(os.path.abspath(target))


def path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def lstat_if_present(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except OSError as exc:
        if exc.errno in {errno.ENOENT, errno.ENOTDIR}:
            return None
        raise


def optional_stat_identity(info: os.stat_result | None) -> tuple[int, ...] | None:
    if info is None:
        return None
    return tuple(
        getattr(info, field)
        for field in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_uid",
            "st_gid",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
    )


def alias_target_stat_record(info: os.stat_result | None) -> dict[str, Any] | None:
    if info is None:
        return None
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "type": observed_entry_type(info.st_mode),
        "mode": f"{stat.S_IMODE(info.st_mode):04o}",
        "uid": info.st_uid,
        "gid": info.st_gid,
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def observe_alias_target(resolved: Path) -> tuple[Path, os.stat_result | None]:
    """Take a repeatable best-effort observation of a possibly external target."""

    for _ in range(3):
        before = lstat_if_present(resolved)
        real_before = Path(os.path.realpath(resolved))
        real_after = Path(os.path.realpath(resolved))
        after = lstat_if_present(resolved)
        if real_before == real_after and optional_stat_identity(before) == optional_stat_identity(
            after
        ):
            return real_after, after
    raise ArchiveError(f"alias target changed while documenting it: {resolved}")


def audited_worktree_paths(real_path: Path) -> list[dict[str, str]]:
    paths = []
    for root in SOURCE_ROOTS:
        absolute = root.absolute()
        if path_is_within(real_path, absolute):
            value = str(absolute)
            paths.append(
                {
                    "path": value,
                    "path_fsencoded_base64": fsencoded_base64(value),
                }
            )
    return paths


def build_alias_ledger(source_local: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    aliases = []
    for record in records:
        if record["type"] != "symlink":
            continue
        link_path = safe_payload_path(source_local, record["path"])
        resolved = resolved_link_path(link_path, record["target"])
        resolved_real, target_info = observe_alias_target(resolved)
        aliases.append(
            {
                "path": record["path"],
                "path_fsencoded_base64": record["path_fsencoded_base64"],
                "target": record["target"],
                "target_fsencoded_base64": record["target_fsencoded_base64"],
                "resolved_lexical_path": str(resolved),
                "resolved_lexical_path_fsencoded_base64": fsencoded_base64(str(resolved)),
                "resolved_real_path_at_archive_time": str(resolved_real),
                "resolved_real_path_fsencoded_base64": fsencoded_base64(str(resolved_real)),
                "target_lexists_at_archive_time": target_info is not None,
                "target_entry_type_at_archive_time": (
                    observed_entry_type(target_info.st_mode) if target_info is not None else None
                ),
                "target_lstat_at_archive_time": alias_target_stat_record(target_info),
                "outside_source_local_lexically": not path_is_within(resolved, source_local),
                "outside_source_local_after_resolution_at_archive_time": not path_is_within(
                    resolved_real, source_local
                ),
                "resolved_inside_audited_worktrees_at_archive_time": (
                    audited_worktree_paths(resolved_real)
                ),
                "referent_copied": False,
            }
        )
    ledger = {
        "format_version": FORMAT_VERSION,
        "source_local": str(source_local),
        "source_local_fsencoded_base64": fsencoded_base64(str(source_local)),
        "symlink_policy": "archive_link_object_only_do_not_follow",
        "path_encoding": "JSON string plus lossless os.fsencode base64",
        "alias_count": len(aliases),
        "aliases_canonical_sha256": sha256_bytes(canonical_json_bytes(aliases)),
        "aliases": aliases,
    }
    return ledger


def valid_fsencoded_pair(value: Any, encoded: Any) -> bool:
    if not isinstance(value, str) or not isinstance(encoded, str):
        return False
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        return False
    return os.fsdecode(raw) == value and os.fsencode(value) == raw


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArchiveError(f"cannot read valid JSON from {path}: {exc}") from exc


def validate_alias_ledger(
    source_local: Path,
    records: list[dict[str, Any]],
    alias_ledger: dict[str, Any],
    alias_path: Path,
) -> None:
    symlinks = [row for row in records if row["type"] == "symlink"]
    aliases = alias_ledger.get("aliases")
    if (
        alias_ledger.get("format_version") != FORMAT_VERSION
        or alias_ledger.get("source_local") != str(source_local)
        or not valid_fsencoded_pair(
            alias_ledger.get("source_local"),
            alias_ledger.get("source_local_fsencoded_base64"),
        )
        or alias_ledger.get("symlink_policy") != "archive_link_object_only_do_not_follow"
        or alias_ledger.get("path_encoding") != "JSON string plus lossless os.fsencode base64"
        or alias_ledger.get("alias_count") != len(symlinks)
        or not isinstance(aliases, list)
        or len(aliases) != len(symlinks)
        or alias_ledger.get("aliases_canonical_sha256")
        != sha256_bytes(canonical_json_bytes(aliases))
    ):
        raise ArchiveError(f"invalid alias ledger header: {alias_path}")

    alias_by_path = {row.get("path"): row for row in aliases if isinstance(row, dict)}
    if len(alias_by_path) != len(aliases):
        raise ArchiveError(f"duplicate or invalid alias records: {alias_path}")
    if [row.get("path") for row in aliases] != [row["path"] for row in symlinks]:
        raise ArchiveError(f"alias records are not in manifest order: {alias_path}")
    for record in symlinks:
        alias = alias_by_path.get(record["path"])
        if alias is None:
            raise ArchiveError(f"missing alias {record['path']!r}: {alias_path}")
        link_path = safe_payload_path(source_local, record["path"])
        resolved = resolved_link_path(link_path, record["target"])
        static_expected = {
            "path": record["path"],
            "path_fsencoded_base64": record["path_fsencoded_base64"],
            "target": record["target"],
            "target_fsencoded_base64": record["target_fsencoded_base64"],
            "resolved_lexical_path": str(resolved),
            "resolved_lexical_path_fsencoded_base64": fsencoded_base64(str(resolved)),
            "outside_source_local_lexically": not path_is_within(resolved, source_local),
            "referent_copied": False,
        }
        for key, value in static_expected.items():
            if alias.get(key) != value:
                raise ArchiveError(
                    f"invalid alias field {key!r} for {record['path']!r}: {alias_path}"
                )
        real_path = alias.get("resolved_real_path_at_archive_time")
        if not valid_fsencoded_pair(
            real_path, alias.get("resolved_real_path_fsencoded_base64")
        ) or not os.path.isabs(real_path):
            raise ArchiveError(f"invalid archived real path for {record['path']!r}")
        target_exists = alias.get("target_lexists_at_archive_time")
        target_type = alias.get("target_entry_type_at_archive_time")
        target_stat = alias.get("target_lstat_at_archive_time")
        if not isinstance(target_exists, bool):
            raise ArchiveError(f"invalid archived target existence for {record['path']!r}")
        if target_type not in {
            None,
            "file",
            "directory",
            "symlink",
            "fifo",
            "socket",
            "block_device",
            "character_device",
            "unknown",
        }:
            raise ArchiveError(f"invalid archived target type for {record['path']!r}")
        if target_exists != (target_stat is not None):
            raise ArchiveError(f"inconsistent archived target existence for {record['path']!r}")
        if target_stat is not None:
            expected_stat_keys = {
                "device",
                "inode",
                "type",
                "mode",
                "uid",
                "gid",
                "size",
                "mtime_ns",
                "ctime_ns",
            }
            if (
                not isinstance(target_stat, dict)
                or set(target_stat) != expected_stat_keys
                or target_stat.get("type") != target_type
                or not all(
                    isinstance(target_stat.get(key), int)
                    for key in expected_stat_keys - {"type", "mode"}
                )
                or not isinstance(target_stat.get("mode"), str)
            ):
                raise ArchiveError(f"invalid archived target stat for {record['path']!r}")
        elif target_type is not None:
            raise ArchiveError(f"inconsistent archived target type for {record['path']!r}")

        real_path_object = Path(real_path)
        if alias.get("outside_source_local_after_resolution_at_archive_time") != (
            not path_is_within(real_path_object, source_local)
        ):
            raise ArchiveError(f"invalid archived alias containment for {record['path']!r}")
        containing_worktrees = alias.get("resolved_inside_audited_worktrees_at_archive_time")
        if containing_worktrees != audited_worktree_paths(real_path_object):
            raise ArchiveError(f"invalid archived alias resolution for {record['path']!r}")


def publication_bundle_path(final: Path) -> Path:
    return final / PUBLICATION_CONTENT / PUBLICATION_BUNDLE


def _open_real_directory_at(parent_descriptor: int, name: str, path: Path) -> int:
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | DIRECTORY | NOFOLLOW,
            dir_fd=parent_descriptor,
        )
    except OSError as exc:
        raise ArchiveError(f"publication entry is not a real directory: {path}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            raise ArchiveError(f"publication entry is not a real directory: {path}")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _validate_private_directory(info: os.stat_result, path: Path) -> None:
    if not stat.S_ISDIR(info.st_mode):
        raise ArchiveError(f"publication entry is not a real directory: {path}")
    if stat.S_IMODE(info.st_mode) != 0o700:
        raise ArchiveError(f"publication directory is not owner-only mode 0700: {path}")


def _list_directory_names(descriptor: int) -> list[str]:
    """List through a fresh open file description so prior reads cannot hide entries."""

    fresh = os.open(".", os.O_RDONLY | DIRECTORY | NOFOLLOW, dir_fd=descriptor)
    try:
        return os.listdir(fresh)
    finally:
        os.close(fresh)


def _validate_empty_marker(root_descriptor: int, marker: str, final: Path) -> None:
    path = final / marker
    descriptor = _open_real_directory_at(root_descriptor, marker, path)
    try:
        _validate_private_directory(os.fstat(descriptor), path)
        if _list_directory_names(descriptor):
            raise ArchiveError(f"publication state marker is not empty: {path}")
    finally:
        os.close(descriptor)


def _inspect_publication_descriptor(
    final_descriptor: int,
    final: Path,
) -> dict[str, Any]:
    final_info = os.fstat(final_descriptor)
    _validate_private_directory(final_info, final)
    entries = set(_list_directory_names(final_descriptor))
    allowed = {
        PUBLICATION_INCOMPLETE,
        PUBLICATION_COMPLETE,
        PUBLICATION_CONTENT,
    }
    unexpected = sorted(entries - allowed, key=path_sort_key)
    if unexpected:
        raise ArchiveError(f"unexpected entries in publication claim {final}: {unexpected!r}")

    has_incomplete = PUBLICATION_INCOMPLETE in entries
    has_complete = PUBLICATION_COMPLETE in entries
    has_content = PUBLICATION_CONTENT in entries
    if has_incomplete:
        _validate_empty_marker(final_descriptor, PUBLICATION_INCOMPLETE, final)
    if has_complete:
        _validate_empty_marker(final_descriptor, PUBLICATION_COMPLETE, final)

    has_bundle = False
    bundle_identity: tuple[int, int, int, int, int] | None = None
    if has_content:
        content_path = final / PUBLICATION_CONTENT
        content_descriptor = _open_real_directory_at(
            final_descriptor,
            PUBLICATION_CONTENT,
            content_path,
        )
        try:
            _validate_private_directory(os.fstat(content_descriptor), content_path)
            content_entries = set(_list_directory_names(content_descriptor))
            unexpected_content = sorted(
                content_entries - {PUBLICATION_BUNDLE},
                key=path_sort_key,
            )
            if unexpected_content:
                raise ArchiveError(
                    f"unexpected entries in publication content {content_path}: "
                    f"{unexpected_content!r}"
                )
            has_bundle = PUBLICATION_BUNDLE in content_entries
            if has_bundle:
                bundle_path = publication_bundle_path(final)
                bundle_descriptor = _open_real_directory_at(
                    content_descriptor,
                    PUBLICATION_BUNDLE,
                    bundle_path,
                )
                try:
                    bundle_info = os.fstat(bundle_descriptor)
                    _validate_private_directory(bundle_info, bundle_path)
                    if not _list_directory_names(bundle_descriptor):
                        raise ArchiveError(f"publication bundle is empty: {bundle_path}")
                    bundle_identity = (
                        bundle_info.st_dev,
                        bundle_info.st_ino,
                        bundle_info.st_mode,
                        bundle_info.st_mtime_ns,
                        bundle_info.st_ctime_ns,
                    )
                finally:
                    os.close(bundle_descriptor)
        finally:
            os.close(content_descriptor)

    if entries and not has_incomplete:
        raise ArchiveError(f"publication claim lacks INCOMPLETE marker: {final}")
    if has_complete and not has_bundle:
        raise ArchiveError(f"COMPLETE publication has no adopted bundle: {final}")
    return {
        "state": "COMPLETE" if has_complete else "INCOMPLETE",
        "has_incomplete_marker": has_incomplete,
        "has_content_directory": has_content,
        "has_bundle": has_bundle,
        "claim_identity": (
            final_info.st_dev,
            final_info.st_ino,
            final_info.st_mode,
            final_info.st_mtime_ns,
            final_info.st_ctime_ns,
        ),
        "bundle_identity": bundle_identity,
    }


def inspect_directory_publication(final: Path) -> dict[str, Any]:
    """Inspect a claimed final name without following any path component."""

    final = final.absolute()
    parent_descriptor, _ = open_root_directory(final.parent)
    try:
        try:
            info = os.stat(final.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return {
                "state": "ABSENT",
                "has_incomplete_marker": False,
                "has_content_directory": False,
                "has_bundle": False,
                "claim_identity": None,
                "bundle_identity": None,
            }
        _validate_private_directory(info, final)
        final_descriptor = _open_real_directory_at(parent_descriptor, final.name, final)
        try:
            assert_same_file_identity(
                info, os.fstat(final_descriptor), final, "opening publication"
            )
            return _inspect_publication_descriptor(final_descriptor, final)
        finally:
            os.close(final_descriptor)
    finally:
        os.close(parent_descriptor)


def assert_publication_unchanged(final: Path, expected: dict[str, Any]) -> None:
    current = inspect_directory_publication(final)
    keys = ("state", "has_bundle", "claim_identity", "bundle_identity")
    if any(current.get(key) != expected.get(key) for key in keys):
        raise ArchiveError(f"publication changed while being verified: {final}")


def _open_publication_lock(parent_descriptor: int, final: Path) -> int:
    if NOFOLLOW == 0:
        raise ArchiveError("this platform lacks O_NOFOLLOW; refusing unsafe publication")
    lock_name = f".{final.name}.publication.lock"
    descriptor = os.open(
        lock_name,
        os.O_RDWR | os.O_CREAT | NOFOLLOW,
        0o600,
        dir_fd=parent_descriptor,
    )
    try:
        os.fchmod(descriptor, 0o600)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ArchiveError(
                f"publication lock is not a regular file: {final.parent / lock_name}"
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _close_publication_lock(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _assert_claim_name_identity(
    parent_descriptor: int,
    final_descriptor: int,
    final: Path,
) -> None:
    expected = os.fstat(final_descriptor)
    actual = os.stat(final.name, dir_fd=parent_descriptor, follow_symlinks=False)
    assert_same_file_identity(expected, actual, final, "revalidating publication claim name")


def _claim_publication_locked(
    parent_descriptor: int,
    final: Path,
) -> tuple[int, bool]:
    created = False
    try:
        os.mkdir(final.name, 0o700, dir_fd=parent_descriptor)
        os.chmod(final.name, 0o700, dir_fd=parent_descriptor, follow_symlinks=False)
        os.fsync(parent_descriptor)
        created = True
    except FileExistsError:
        pass
    final_descriptor = _open_real_directory_at(parent_descriptor, final.name, final)
    try:
        publication = _inspect_publication_descriptor(final_descriptor, final)
        if not publication["has_incomplete_marker"]:
            if publication["has_content_directory"] or publication["state"] == "COMPLETE":
                raise ArchiveError(f"cannot repair malformed publication claim: {final}")
            os.mkdir(PUBLICATION_INCOMPLETE, 0o700, dir_fd=final_descriptor)
            os.chmod(
                PUBLICATION_INCOMPLETE,
                0o700,
                dir_fd=final_descriptor,
                follow_symlinks=False,
            )
            os.fsync(final_descriptor)
    except BaseException:
        os.close(final_descriptor)
        raise
    return final_descriptor, created


def ensure_directory_publication_claim(final: Path) -> str:
    """Atomically reserve a final directory name and expose INCOMPLETE state."""

    final = final.absolute()
    parent_descriptor, _ = open_root_directory(final.parent)
    try:
        lock_descriptor = _open_publication_lock(parent_descriptor, final)
        try:
            final_descriptor, created = _claim_publication_locked(parent_descriptor, final)
            try:
                _inspect_publication_descriptor(final_descriptor, final)
                _assert_claim_name_identity(parent_descriptor, final_descriptor, final)
            finally:
                os.close(final_descriptor)
            return "MKDIR_CLAIM_CREATED" if created else "MKDIR_CLAIM_VERIFIED"
        finally:
            _close_publication_lock(lock_descriptor)
    finally:
        os.close(parent_descriptor)


def adopt_staging_directory(staging: Path, final: Path) -> str:
    """Move a verified staging tree into a private claimed publication slot."""

    staging = staging.absolute()
    final = final.absolute()
    if staging.parent != final.parent:
        raise ArchiveError("staging and publication claim must be siblings")
    parent_descriptor, _ = open_root_directory(final.parent)
    try:
        staging_info = os.stat(staging.name, dir_fd=parent_descriptor, follow_symlinks=False)
        _validate_private_directory(staging_info, staging)
        staging_descriptor = _open_real_directory_at(parent_descriptor, staging.name, staging)
        try:
            assert_same_file_identity(
                staging_info,
                os.fstat(staging_descriptor),
                staging,
                "opening staging for adoption",
            )
            if not _list_directory_names(staging_descriptor):
                raise ArchiveError(f"refusing to adopt empty staging directory: {staging}")
        finally:
            os.close(staging_descriptor)
        lock_descriptor = _open_publication_lock(parent_descriptor, final)
        try:
            final_descriptor, created = _claim_publication_locked(parent_descriptor, final)
            try:
                publication = _inspect_publication_descriptor(final_descriptor, final)
                if publication["state"] == "COMPLETE":
                    _assert_claim_name_identity(parent_descriptor, final_descriptor, final)
                    return "COMPLETE_ALREADY_PRESENT"
                if publication["has_bundle"]:
                    _assert_claim_name_identity(parent_descriptor, final_descriptor, final)
                    return "BUNDLE_ALREADY_PRESENT"
                if not publication["has_content_directory"]:
                    os.mkdir(PUBLICATION_CONTENT, 0o700, dir_fd=final_descriptor)
                    os.chmod(
                        PUBLICATION_CONTENT,
                        0o700,
                        dir_fd=final_descriptor,
                        follow_symlinks=False,
                    )
                    os.fsync(final_descriptor)
                content_descriptor = _open_real_directory_at(
                    final_descriptor,
                    PUBLICATION_CONTENT,
                    final / PUBLICATION_CONTENT,
                )
                try:
                    if _list_directory_names(content_descriptor):
                        raise ArchiveError(
                            f"publication content slot is not empty: {final / PUBLICATION_CONTENT}"
                        )
                    current_staging = os.stat(
                        staging.name,
                        dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                    assert_stable_entry(
                        staging_info,
                        current_staging,
                        staging,
                        "staging changed before adoption",
                    )
                    os.rename(
                        staging.name,
                        PUBLICATION_BUNDLE,
                        src_dir_fd=parent_descriptor,
                        dst_dir_fd=content_descriptor,
                    )
                    adopted = os.stat(
                        PUBLICATION_BUNDLE,
                        dir_fd=content_descriptor,
                        follow_symlinks=False,
                    )
                    assert_same_file_identity(
                        staging_info,
                        adopted,
                        publication_bundle_path(final),
                        "verifying bundle adoption",
                    )
                    os.fsync(content_descriptor)
                finally:
                    os.close(content_descriptor)
                os.fsync(final_descriptor)
                os.fsync(parent_descriptor)
                _assert_claim_name_identity(parent_descriptor, final_descriptor, final)
                return (
                    "MKDIR_CLAIM_PLUS_PRIVATE_BUNDLE_RENAME"
                    if created
                    else "PRIVATE_BUNDLE_RENAME_INTO_EXISTING_CLAIM"
                )
            finally:
                os.close(final_descriptor)
        finally:
            _close_publication_lock(lock_descriptor)
    finally:
        os.close(parent_descriptor)


def mark_directory_publication_complete(final: Path) -> str:
    """Perform the monotonic INCOMPLETE -> COMPLETE transition with mkdir."""

    final = final.absolute()
    parent_descriptor, _ = open_root_directory(final.parent)
    try:
        lock_descriptor = _open_publication_lock(parent_descriptor, final)
        try:
            final_descriptor, _ = _claim_publication_locked(parent_descriptor, final)
            try:
                publication = _inspect_publication_descriptor(final_descriptor, final)
                if not publication["has_bundle"]:
                    raise ArchiveError(f"cannot complete publication without a bundle: {final}")
                if publication["state"] == "COMPLETE":
                    _assert_claim_name_identity(parent_descriptor, final_descriptor, final)
                    return "COMPLETE_MARKER_ALREADY_PRESENT"
                try:
                    os.mkdir(PUBLICATION_COMPLETE, 0o700, dir_fd=final_descriptor)
                    os.chmod(
                        PUBLICATION_COMPLETE,
                        0o700,
                        dir_fd=final_descriptor,
                        follow_symlinks=False,
                    )
                except FileExistsError:
                    pass
                os.fsync(final_descriptor)
                os.fsync(parent_descriptor)
                completed = _inspect_publication_descriptor(final_descriptor, final)
                if completed["state"] != "COMPLETE":
                    raise ArchiveError(f"publication completion marker was not durable: {final}")
                _assert_claim_name_identity(parent_descriptor, final_descriptor, final)
                return "MKDIR_COMPLETE_MARKER"
            finally:
                os.close(final_descriptor)
        finally:
            _close_publication_lock(lock_descriptor)
    finally:
        os.close(parent_descriptor)


def verify_existing_archive(
    source_root: Path,
    final: Path,
    current_records: list[dict[str, Any]],
    *,
    producer: dict[str, str],
    reporter: ProgressReporter | None = None,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    expected_producer = copy_producer_identity(producer)
    publication = inspect_directory_publication(final)
    if publication["state"] == "ABSENT" or not publication["has_bundle"]:
        raise ArchiveError(f"existing archive has no adopted publication bundle: {final}")
    if publication["state"] != "COMPLETE" and not allow_incomplete:
        raise ArchiveError(f"existing archive publication is INCOMPLETE: {final}")
    bundle = publication_bundle_path(final)
    bundle_entries = {path.name for path in bundle.iterdir()}
    expected_bundle_entries = {
        "manifest.json",
        "alias_ledger.json",
        "archive_audit.json",
        "payload",
    }
    if bundle_entries != expected_bundle_entries:
        raise ArchiveError(
            f"unexpected archive bundle entries in {bundle}: "
            f"{sorted(bundle_entries, key=path_sort_key)!r}"
        )
    payload_parent = bundle / "payload"
    if payload_parent.is_symlink() or not payload_parent.is_dir():
        raise ArchiveError(f"archive payload parent is not a real directory: {payload_parent}")
    if {path.name for path in payload_parent.iterdir()} != {".local"}:
        raise ArchiveError(f"unexpected entries in archive payload parent: {payload_parent}")
    manifest_path = bundle / "manifest.json"
    alias_path = bundle / "alias_ledger.json"
    audit_path = bundle / "archive_audit.json"
    for metadata_path in (manifest_path, alias_path, audit_path):
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise ArchiveError(f"archive metadata is not a real file: {metadata_path}")
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ArchiveError(f"archive manifest is not an object: {manifest_path}")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ArchiveError(f"unsupported archive manifest format: {manifest_path}")
    if manifest.get("producer") != expected_producer:
        raise ArchiveError(f"archive producer identity mismatch: {manifest_path}")
    if manifest.get("source_commit") != expected_producer["source_commit"]:
        raise ArchiveError(f"archive source commit mismatch: {manifest_path}")
    if manifest.get("source_worktree") != str(source_root):
        raise ArchiveError(f"existing archive source mismatch: {final}")
    if manifest.get("source_local") != str(source_root / ".local"):
        raise ArchiveError(f"existing archive .local mismatch: {final}")
    if not valid_fsencoded_pair(
        manifest.get("source_worktree"),
        manifest.get("source_worktree_fsencoded_base64"),
    ) or not valid_fsencoded_pair(
        manifest.get("source_local"),
        manifest.get("source_local_fsencoded_base64"),
    ):
        raise ArchiveError(f"manifest source paths are not lossless: {manifest_path}")
    archived_records = manifest.get("entries")
    if not isinstance(archived_records, list):
        raise ArchiveError(f"existing archive has no valid entries list: {final}")
    if current_records != archived_records:
        raise ArchiveError(
            f"existing archive differs from current source; refusing overwrite: {final}"
        )
    payload_relative = manifest.get("archive_payload_relative_path")
    if payload_relative != "payload/.local":
        raise ArchiveError(f"unexpected payload location in {manifest_path}")
    if manifest.get("entries_canonical_sha256") != sha256_bytes(
        canonical_json_bytes(archived_records)
    ):
        raise ArchiveError(f"manifest entries checksum mismatch: {manifest_path}")
    if manifest.get("path_encoding") != "JSON string plus lossless os.fsencode base64":
        raise ArchiveError(f"unexpected path encoding in {manifest_path}")
    expected_publication_fields = {
        "publication_policy": PUBLICATION_POLICY,
        "publication_state_transition": "INCOMPLETE_TO_COMPLETE",
        "publication_incomplete_marker": PUBLICATION_INCOMPLETE,
        "publication_complete_marker": PUBLICATION_COMPLETE,
        "publication_bundle_relative_path": (f"{PUBLICATION_CONTENT}/{PUBLICATION_BUNDLE}"),
    }
    for key, value in expected_publication_fields.items():
        if manifest.get(key) != value:
            raise ArchiveError(f"unexpected manifest publication field {key!r}: {manifest_path}")
    expected_copy_policy = {
        "regular_files": "byte_copy_with_sha256",
        "directories": "recreated_with_source_metadata",
        "symlinks": "recreated_without_following_referent",
        "source_uid_gid": "recorded_in_manifest_not_reapplied_on_nfs",
        "special_files": "fail_closed",
    }
    if manifest.get("copy_policy") != expected_copy_policy:
        raise ArchiveError(f"unexpected copy policy in {manifest_path}")
    alias_ledger = load_json(alias_path)
    if not isinstance(alias_ledger, dict):
        raise ArchiveError(f"alias ledger is not an object: {alias_path}")
    alias_ledger_canonical_sha256 = sha256_bytes(canonical_json_bytes(alias_ledger))
    if manifest.get("alias_ledger_canonical_sha256") != alias_ledger_canonical_sha256:
        raise ArchiveError(f"manifest does not bind alias ledger: {manifest_path}")
    validate_alias_ledger(source_root / ".local", archived_records, alias_ledger, alias_path)
    if reporter is not None:
        reporter.begin_phase(
            source_root,
            "VERIFY_EXISTING_PAYLOAD",
            sum(row["size"] for row in archived_records if row["type"] == "file"),
        )
    verification = verify_payload(bundle / payload_relative, archived_records, reporter=reporter)
    archive_audit = load_json(audit_path)
    if not isinstance(archive_audit, dict):
        raise ArchiveError(f"archive audit is not an object: {audit_path}")
    expected_audit_fields = {
        "format_version": FORMAT_VERSION,
        "status": "PASS",
        "source_worktree": str(source_root),
        "source_worktree_fsencoded_base64": fsencoded_base64(str(source_root)),
        "source_local": str(source_root / ".local"),
        "source_local_fsencoded_base64": fsencoded_base64(str(source_root / ".local")),
        "source_commit": expected_producer["source_commit"],
        "final_archive": str(final),
        "final_archive_fsencoded_base64": fsencoded_base64(str(final)),
        "payload_verification": verification,
        "alias_ledger_canonical_sha256": alias_ledger_canonical_sha256,
        "source_unchanged_during_copy": True,
        "source_content_rehashed_before_publish": True,
        "source_matches_manifest_at_last_prepublish_validation": True,
        "symlink_referents_followed": False,
        "source_uid_gid_recorded_in_manifest": True,
        "payload_uid_gid_reapplied": False,
        "atomic_publish_policy": PUBLICATION_POLICY,
        "publication_state_transition": "INCOMPLETE_TO_COMPLETE",
        "publication_incomplete_marker": PUBLICATION_INCOMPLETE,
        "publication_complete_marker": PUBLICATION_COMPLETE,
        "publication_bundle_relative_path": (f"{PUBLICATION_CONTENT}/{PUBLICATION_BUNDLE}"),
        "shared_nas_is_independent_disaster_backup": False,
        "shared_nas_limitation": (
            "This verified retirement archive shares the experiment NAS failure domain "
            "and is not an independent disaster-recovery backup."
        ),
        "source_files_deleted": False,
        "worktree_remove_or_prune_executed": False,
        "future_removal_requires_separate_user_authorization": True,
        "producer": expected_producer,
    }
    for key, value in expected_audit_fields.items():
        if archive_audit.get(key) != value:
            raise ArchiveError(f"invalid archive audit field {key!r}: {audit_path}")
    validation_points = archive_audit.get("source_validation_points_utc")
    if (
        not isinstance(validation_points, dict)
        or set(validation_points)
        != {
            "initial_manifest_hash",
            "after_copy_hash",
            "last_prepublish_hash",
        }
        or not all(
            isinstance(value, str) and value.endswith("Z") for value in validation_points.values()
        )
    ):
        raise ArchiveError(f"invalid source validation points: {audit_path}")
    if archive_audit.get("removal_authorized") is not False:
        raise ArchiveError(f"existing archive improperly authorizes removal: {audit_path}")
    if archive_audit.get("removal_executed") is not False:
        raise ArchiveError(f"existing archive claims removal was executed: {audit_path}")
    return {
        "archive": str(final),
        "archive_status": "VERIFIED_EXISTING",
        "manifest_sha256": sha256_file(manifest_path, no_follow=True),
        "alias_ledger_sha256": sha256_file(alias_path, no_follow=True),
        "archive_audit_sha256": sha256_file(audit_path, no_follow=True),
        "payload_verification": verification,
        "publication_state": publication["state"],
    }


def archive_one(
    source_root: Path,
    destination_root: Path,
    *,
    producer: dict[str, str],
    reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    frozen_producer = copy_producer_identity(producer)
    source_root = source_root.absolute()
    destination_root = destination_root.absolute()
    source_local = source_root / ".local"
    if not source_root.is_dir() or source_root.is_symlink():
        raise ArchiveError(f"source worktree missing or not a real directory: {source_root}")
    try:
        destination_descriptor, _ = open_root_directory(destination_root)
    except OSError as exc:
        raise ArchiveError(
            f"destination path contains a missing, aliased, or non-directory component: "
            f"{destination_root}"
        ) from exc
    else:
        os.close(destination_descriptor)

    print(f"[{source_root.name!a}] inventory + SHA-256", flush=True)
    inventory = scan_tree(source_local, include_hash=False)
    total_bytes = sum(row["size"] for row in inventory if row["type"] == "file")
    if reporter is not None:
        reporter.begin_phase(source_root, "HASH_SOURCE_INITIAL", total_bytes)
    records = scan_tree(source_local, include_hash=True, reporter=reporter)
    if metadata_projection(inventory) != metadata_projection(records):
        raise ArchiveError(f"source tree changed during initial inventory: {source_local}")
    initial_hash_completed_at = utc_now()
    final = destination_root / archive_name(source_root)
    publication = inspect_directory_publication(final)
    if publication["state"] == "COMPLETE":
        print(f"[{source_root.name!a}] COMPLETE publication exists; verify only", flush=True)
        result = verify_existing_archive(
            source_root,
            final,
            records,
            producer=frozen_producer,
            reporter=reporter,
        )
        if reporter is not None:
            reporter.begin_phase(source_root, "HASH_SOURCE_AFTER_VERIFY", total_bytes)
        final_source_records = scan_tree(source_local, include_hash=True, reporter=reporter)
        if final_source_records != records:
            raise ArchiveError(f"source changed while verifying existing archive: {source_local}")
        assert_publication_unchanged(final, publication)
        result["source_matches_manifest_after_archive_verification"] = True
        return result
    if publication["has_bundle"]:
        print(f"[{source_root.name!a}] recover INCOMPLETE publication", flush=True)
        result = verify_existing_archive(
            source_root,
            final,
            records,
            producer=frozen_producer,
            reporter=reporter,
            allow_incomplete=True,
        )
        if reporter is not None:
            reporter.begin_phase(source_root, "HASH_SOURCE_BEFORE_RECOVERY_COMPLETE", total_bytes)
        final_source_records = scan_tree(source_local, include_hash=True, reporter=reporter)
        if final_source_records != records:
            raise ArchiveError(
                f"source changed while recovering incomplete archive: {source_local}"
            )
        assert_publication_unchanged(final, publication)
        completion_method = mark_directory_publication_complete(final)
        result["archive_status"] = "RECOVERED_AND_VERIFIED"
        result["publication_state"] = "COMPLETE"
        result["completion_method"] = completion_method
        result["source_matches_manifest_after_archive_verification"] = True
        return result

    claim_method = ensure_directory_publication_claim(final)

    staging = destination_root / (
        f".{archive_name(source_root)}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    )
    staging.mkdir(mode=0o700, parents=False, exist_ok=False)
    staging.chmod(0o700)
    fsync_directory(destination_root)
    payload_parent = staging / "payload"
    payload_parent.mkdir(mode=0o700)
    payload_parent.chmod(0o700)
    payload_root = payload_parent / ".local"

    try:
        print(f"[{source_root.name!a}] copy to {staging.name!a}", flush=True)
        if reporter is not None:
            reporter.begin_phase(source_root, "COPY_TO_STAGING", total_bytes, staging=staging)
        populate_payload(source_local, payload_root, records, reporter=reporter)

        # Bracket the copy with full source hashes.  A metadata-only rescan cannot
        # detect same-size content replacement with restored mtimes.
        if reporter is not None:
            reporter.begin_phase(
                source_root, "HASH_SOURCE_AFTER_COPY", total_bytes, staging=staging
            )
        source_after = scan_tree(source_local, include_hash=True, reporter=reporter)
        if records != source_after:
            raise ArchiveError(f"source tree changed during archive: {source_local}")
        after_copy_hash_completed_at = utc_now()

        print(f"[{source_root.name!a}] verify staged payload", flush=True)
        if reporter is not None:
            reporter.begin_phase(source_root, "VERIFY_STAGED_PAYLOAD", total_bytes, staging=staging)
        verification = verify_payload(payload_root, records, reporter=reporter)
        alias_ledger = build_alias_ledger(source_local, records)
        alias_ledger_canonical_sha256 = sha256_bytes(canonical_json_bytes(alias_ledger))
        created_at = utc_now()
        manifest = {
            "format_version": FORMAT_VERSION,
            "created_at_utc": created_at,
            "source_worktree": str(source_root),
            "source_worktree_fsencoded_base64": fsencoded_base64(str(source_root)),
            "source_local": str(source_local),
            "source_local_fsencoded_base64": fsencoded_base64(str(source_local)),
            "source_commit": frozen_producer["source_commit"],
            "producer": frozen_producer,
            "archive_payload_relative_path": "payload/.local",
            "path_encoding": "JSON string plus lossless os.fsencode base64",
            "publication_policy": PUBLICATION_POLICY,
            "publication_state_transition": "INCOMPLETE_TO_COMPLETE",
            "publication_incomplete_marker": PUBLICATION_INCOMPLETE,
            "publication_complete_marker": PUBLICATION_COMPLETE,
            "publication_bundle_relative_path": (f"{PUBLICATION_CONTENT}/{PUBLICATION_BUNDLE}"),
            "copy_policy": {
                "regular_files": "byte_copy_with_sha256",
                "directories": "recreated_with_source_metadata",
                "symlinks": "recreated_without_following_referent",
                "source_uid_gid": "recorded_in_manifest_not_reapplied_on_nfs",
                "special_files": "fail_closed",
            },
            "alias_ledger_canonical_sha256": alias_ledger_canonical_sha256,
            "entries": records,
        }
        manifest["entries_canonical_sha256"] = sha256_bytes(canonical_json_bytes(records))
        write_json_fsync(staging / "manifest.json", manifest)
        write_json_fsync(staging / "alias_ledger.json", alias_ledger)
        fsync_directory(payload_parent)

        # This is the final source-dependent operation before publishing.  It
        # occurs after payload verification and the other evidence files have
        # been materialised, narrowing the unavoidable live-tree observation
        # window without claiming snapshot semantics.
        if reporter is not None:
            reporter.begin_phase(
                source_root,
                "HASH_SOURCE_LAST_PREPUBLISH",
                total_bytes,
                staging=staging,
            )
        source_before_publish = scan_tree(source_local, include_hash=True, reporter=reporter)
        if records != source_before_publish:
            raise ArchiveError(f"source tree changed before archive publication: {source_local}")
        last_prepublish_hash_completed_at = utc_now()

        archive_audit = {
            "format_version": FORMAT_VERSION,
            "created_at_utc": created_at,
            "status": "PASS",
            "source_worktree": str(source_root),
            "source_worktree_fsencoded_base64": fsencoded_base64(str(source_root)),
            "source_local": str(source_local),
            "source_local_fsencoded_base64": fsencoded_base64(str(source_local)),
            "source_commit": frozen_producer["source_commit"],
            "final_archive": str(final),
            "final_archive_fsencoded_base64": fsencoded_base64(str(final)),
            "payload_verification": verification,
            "alias_ledger_canonical_sha256": alias_ledger_canonical_sha256,
            "source_unchanged_during_copy": True,
            "source_content_rehashed_before_publish": True,
            "source_matches_manifest_at_last_prepublish_validation": True,
            "source_validation_points_utc": {
                "initial_manifest_hash": initial_hash_completed_at,
                "after_copy_hash": after_copy_hash_completed_at,
                "last_prepublish_hash": last_prepublish_hash_completed_at,
            },
            "symlink_referents_followed": False,
            "source_uid_gid_recorded_in_manifest": True,
            "payload_uid_gid_reapplied": False,
            "atomic_publish_policy": PUBLICATION_POLICY,
            "publication_state_transition": "INCOMPLETE_TO_COMPLETE",
            "publication_incomplete_marker": PUBLICATION_INCOMPLETE,
            "publication_complete_marker": PUBLICATION_COMPLETE,
            "publication_bundle_relative_path": (f"{PUBLICATION_CONTENT}/{PUBLICATION_BUNDLE}"),
            "shared_nas_is_independent_disaster_backup": False,
            "shared_nas_limitation": (
                "This verified retirement archive shares the experiment NAS failure domain "
                "and is not an independent disaster-recovery backup."
            ),
            "removal_authorized": False,
            "removal_executed": False,
            "source_files_deleted": False,
            "worktree_remove_or_prune_executed": False,
            "future_removal_requires_separate_user_authorization": True,
            "producer": frozen_producer,
        }
        write_json_fsync(staging / "archive_audit.json", archive_audit)
        fsync_directory(staging)

        print(f"[{source_root.name!a}] adopt verified bundle into INCOMPLETE claim", flush=True)
        publish_method = adopt_staging_directory(staging, final)
        fsync_directory(destination_root)
    except BaseException:
        retained = staging if os.path.lexists(staging) else final
        print(
            f"[{source_root.name!a}] FAILED; retained evidence: {str(retained)!a}",
            file=sys.stderr,
            flush=True,
        )
        raise

    adopted_publication = inspect_directory_publication(final)
    result = verify_existing_archive(
        source_root,
        final,
        records,
        producer=frozen_producer,
        reporter=reporter,
        allow_incomplete=True,
    )
    if reporter is not None:
        reporter.begin_phase(source_root, "HASH_SOURCE_AFTER_PUBLISH_VERIFY", total_bytes)
    final_source_records = scan_tree(source_local, include_hash=True, reporter=reporter)
    if final_source_records != records:
        raise ArchiveError(f"source changed after archive publication: {source_local}")
    assert_publication_unchanged(final, adopted_publication)
    completion_method = mark_directory_publication_complete(final)
    created_methods = {
        "MKDIR_CLAIM_PLUS_PRIVATE_BUNDLE_RENAME",
        "PRIVATE_BUNDLE_RENAME_INTO_EXISTING_CLAIM",
    }
    result["archive_status"] = (
        "CREATED_AND_VERIFIED"
        if publish_method in created_methods
        else "RECOVERED_CONCURRENT_PUBLICATION"
    )
    result["atomic_publish_method"] = publish_method
    result["claim_method"] = claim_method
    result["completion_method"] = completion_method
    result["publication_state"] = "COMPLETE"
    if os.path.lexists(staging):
        result["retained_unused_staging"] = str(staging)
    result["source_matches_manifest_after_archive_verification"] = True
    return result


def build_total_audit(
    results: list[dict[str, Any]],
    destination_root: Path,
    *,
    producer: dict[str, str],
) -> dict[str, Any]:
    frozen_producer = copy_producer_identity(producer)
    expected_archives = [
        str(destination_root / archive_name(source.absolute())) for source in SOURCE_ROOTS
    ]
    if len(results) != len(expected_archives):
        raise ArchiveError(
            f"aggregate audit requires exactly {len(expected_archives)} archives, got {len(results)}"
        )
    observed_archives = [row.get("archive") for row in results]
    if observed_archives != expected_archives or len(set(observed_archives)) != len(
        observed_archives
    ):
        raise ArchiveError(
            f"aggregate archive paths/order mismatch: {observed_archives!r} != {expected_archives!r}"
        )
    for row in results:
        if row.get("publication_state") != "COMPLETE":
            raise ArchiveError(f"aggregate component is not COMPLETE: {row.get('archive')}")
        verification = row.get("payload_verification")
        if not isinstance(verification, dict) or verification.get("status") != "PASS":
            raise ArchiveError(f"aggregate component payload did not pass: {row.get('archive')}")
    return {
        "format_version": FORMAT_VERSION,
        "generated_at_utc": utc_now(),
        "status": "PASS",
        "scope": "S4.3-PI2S worktree-local retirement evidence",
        "source_commit": frozen_producer["source_commit"],
        "producer": frozen_producer,
        "destination_root": str(destination_root),
        "archives": results,
        "publication_policy": PUBLICATION_POLICY,
        "publication_state_transition": "INCOMPLETE_TO_COMPLETE",
        "publication_incomplete_marker": PUBLICATION_INCOMPLETE,
        "publication_complete_marker": PUBLICATION_COMPLETE,
        "publication_bundle_relative_path": (
            f"{PUBLICATION_CONTENT}/{PUBLICATION_BUNDLE}/{AGGREGATE_AUDIT_NAME}"
        ),
        "all_archives_publication_state_complete": all(
            row.get("publication_state") == "COMPLETE" for row in results
        ),
        "all_payloads_sha256_verified": all(
            row["payload_verification"]["status"] == "PASS" for row in results
        ),
        "symlink_referents_followed": False,
        "source_uid_gid_recorded_in_manifests": True,
        "payload_uid_gid_reapplied": False,
        "shared_nas_is_independent_disaster_backup": False,
        "shared_nas_limitation": (
            "The retirement archives are on the same shared experiment NAS and do not "
            "constitute an independent disaster-recovery backup."
        ),
        "removal_authorized": False,
        "removal_executed": False,
        "source_files_deleted": False,
        "worktree_remove_or_prune_executed": False,
        "future_removal_requires_separate_user_authorization": True,
    }


def verify_aggregate_component_evidence(audit: dict[str, Any]) -> None:
    archives = audit.get("archives")
    if not isinstance(archives, list) or len(archives) != len(SOURCE_ROOTS):
        raise ArchiveError("aggregate audit does not contain the exact component count")
    destination_value = audit.get("destination_root")
    if not isinstance(destination_value, str):
        raise ArchiveError("aggregate audit destination_root is invalid")
    destination_root = Path(destination_value)
    expected_paths = [
        str(destination_root / archive_name(source.absolute())) for source in SOURCE_ROOTS
    ]
    observed_paths = [row.get("archive") if isinstance(row, dict) else None for row in archives]
    if observed_paths != expected_paths:
        raise ArchiveError(
            f"aggregate component paths/order mismatch: {observed_paths!r} != {expected_paths!r}"
        )
    seen: set[str] = set()
    for row in archives:
        if not isinstance(row, dict):
            raise ArchiveError("aggregate audit contains a non-object archive row")
        archive_value = row.get("archive")
        if not isinstance(archive_value, str) or archive_value in seen:
            raise ArchiveError(f"invalid or duplicate aggregate archive path: {archive_value!r}")
        seen.add(archive_value)
        if row.get("publication_state") != "COMPLETE":
            raise ArchiveError(f"aggregate archive is not COMPLETE: {archive_value}")
        verification = row.get("payload_verification")
        if not isinstance(verification, dict) or verification.get("status") != "PASS":
            raise ArchiveError(f"aggregate archive payload status is not PASS: {archive_value}")
        archive_path = Path(archive_value)
        publication = inspect_directory_publication(archive_path)
        if publication["state"] != "COMPLETE" or not publication["has_bundle"]:
            raise ArchiveError(f"aggregate component publication is not COMPLETE: {archive_path}")
        bundle = publication_bundle_path(archive_path)
        evidence = {
            "manifest_sha256": bundle / "manifest.json",
            "alias_ledger_sha256": bundle / "alias_ledger.json",
            "archive_audit_sha256": bundle / "archive_audit.json",
        }
        for key, path in evidence.items():
            if path.is_symlink() or not path.is_file():
                raise ArchiveError(f"aggregate component evidence is not a real file: {path}")
            if row.get(key) != sha256_file(path, no_follow=True):
                raise ArchiveError(f"aggregate component evidence hash changed ({key}): {path}")


def verify_total_audit_publication(
    publication_root: Path,
    audit: dict[str, Any],
    *,
    allow_incomplete: bool = False,
) -> Path:
    publication = inspect_directory_publication(publication_root)
    if publication["state"] == "ABSENT" or not publication["has_bundle"]:
        raise ArchiveError(f"aggregate audit publication has no bundle: {publication_root}")
    if publication["state"] != "COMPLETE" and not allow_incomplete:
        raise ArchiveError(f"aggregate audit publication is INCOMPLETE: {publication_root}")
    bundle = publication_bundle_path(publication_root)
    bundle_entries = {path.name for path in bundle.iterdir()}
    if bundle_entries != {AGGREGATE_AUDIT_NAME}:
        raise ArchiveError(f"unexpected aggregate audit bundle entries: {bundle_entries!r}")
    audit_path = bundle / AGGREGATE_AUDIT_NAME
    if audit_path.is_symlink() or not audit_path.is_file():
        raise ArchiveError(f"aggregate audit is not a real file: {audit_path}")
    existing = load_json(audit_path)
    if not isinstance(existing, dict):
        raise ArchiveError(f"existing aggregate audit is not an object: {audit_path}")

    # Timestamps and CREATED/EXISTING labels are run-local.  The evidence
    # evidence identity is the archive path plus its three evidence hashes.
    identity_keys = (
        "archive",
        "manifest_sha256",
        "alias_ledger_sha256",
        "archive_audit_sha256",
    )
    existing_archives = existing.get("archives")
    if not isinstance(existing_archives, list) or not all(
        isinstance(row, dict) for row in existing_archives
    ):
        raise ArchiveError(f"existing aggregate audit archives are invalid: {audit_path}")
    existing_identities = [
        {key: row.get(key) for key in identity_keys} for row in existing_archives
    ]
    current_identities = [{key: row.get(key) for key in identity_keys} for row in audit["archives"]]
    if existing_identities != current_identities:
        raise ArchiveError(f"existing aggregate audit differs; refusing overwrite: {audit_path}")
    for row in existing_archives:
        if row.get("publication_state") != "COMPLETE":
            raise ArchiveError(f"existing aggregate row is not COMPLETE: {audit_path}")
        verification = row.get("payload_verification")
        if not isinstance(verification, dict) or verification.get("status") != "PASS":
            raise ArchiveError(f"existing aggregate row payload status is invalid: {audit_path}")
    safe_existing_fields = {
        "format_version": FORMAT_VERSION,
        "status": "PASS",
        "scope": audit["scope"],
        "source_commit": audit["source_commit"],
        "destination_root": audit["destination_root"],
        "publication_policy": PUBLICATION_POLICY,
        "publication_state_transition": "INCOMPLETE_TO_COMPLETE",
        "publication_incomplete_marker": PUBLICATION_INCOMPLETE,
        "publication_complete_marker": PUBLICATION_COMPLETE,
        "publication_bundle_relative_path": (
            f"{PUBLICATION_CONTENT}/{PUBLICATION_BUNDLE}/{AGGREGATE_AUDIT_NAME}"
        ),
        "all_archives_publication_state_complete": True,
        "all_payloads_sha256_verified": True,
        "symlink_referents_followed": False,
        "source_uid_gid_recorded_in_manifests": True,
        "payload_uid_gid_reapplied": False,
        "shared_nas_is_independent_disaster_backup": False,
        "shared_nas_limitation": audit["shared_nas_limitation"],
        "removal_authorized": False,
        "removal_executed": False,
        "source_files_deleted": False,
        "worktree_remove_or_prune_executed": False,
        "future_removal_requires_separate_user_authorization": True,
        "producer": audit["producer"],
    }
    for key, value in safe_existing_fields.items():
        if existing.get(key) != value:
            raise ArchiveError(f"existing aggregate audit has invalid field {key!r}: {audit_path}")
    return audit_path


def publish_total_audit(destination_root: Path, audit: dict[str, Any]) -> Path:
    verify_aggregate_component_evidence(audit)
    publication_root = destination_root / AGGREGATE_PUBLICATION_NAME
    publication = inspect_directory_publication(publication_root)
    if publication["state"] == "COMPLETE":
        audit_path = verify_total_audit_publication(publication_root, audit)
        assert_publication_unchanged(publication_root, publication)
        return audit_path
    if publication["has_bundle"]:
        audit_path = verify_total_audit_publication(
            publication_root,
            audit,
            allow_incomplete=True,
        )
        verify_aggregate_component_evidence(audit)
        assert_publication_unchanged(publication_root, publication)
        mark_directory_publication_complete(publication_root)
        return audit_path

    ensure_directory_publication_claim(publication_root)
    staging = destination_root / (
        f".{AGGREGATE_PUBLICATION_NAME}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    )
    staging.mkdir(mode=0o700, parents=False, exist_ok=False)
    staging.chmod(0o700)
    try:
        write_json_fsync(staging / AGGREGATE_AUDIT_NAME, audit)
        fsync_directory(staging)
        adopt_staging_directory(staging, publication_root)
        adopted_publication = inspect_directory_publication(publication_root)
        audit_path = verify_total_audit_publication(
            publication_root,
            audit,
            allow_incomplete=True,
        )
        verify_aggregate_component_evidence(audit)
        assert_publication_unchanged(publication_root, adopted_publication)
        mark_directory_publication_complete(publication_root)
        return audit_path
    except BaseException:
        retained = staging if os.path.lexists(staging) else publication_root
        print(
            f"aggregate audit publication FAILED; retained evidence: {str(retained)!a}",
            file=sys.stderr,
            flush=True,
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help="existing NAS destination directory",
    )
    parser.add_argument(
        "--goal-uuid",
        required=True,
        help="active Codex goal/thread UUID recorded in the persistent progress state",
    )
    parser.add_argument(
        "--source-commit",
        required=True,
        help="exact committed main-worktree HEAD that contains this archiver",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    destination_root = args.destination.absolute()
    if not destination_root.is_dir() or destination_root.is_symlink():
        raise ArchiveError(f"destination must be an existing real directory: {destination_root}")
    try:
        destination_descriptor, _ = open_root_directory(destination_root)
    except OSError as exc:
        raise ArchiveError(
            f"destination path contains a missing, aliased, or non-directory component: "
            f"{destination_root}"
        ) from exc
    else:
        os.close(destination_descriptor)
    producer = verify_producer_source_commit(args.source_commit)
    reporter = ProgressReporter(
        destination_root / ".worktree_local_retirement_progress.json",
        goal_uuid=args.goal_uuid,
        producer=producer,
    )
    try:
        results = []
        for source in SOURCE_ROOTS:
            result = archive_one(
                source,
                destination_root,
                producer=producer,
                reporter=reporter,
            )
            results.append(result)
            reporter.note_archive_complete(Path(result["archive"]), result["archive_status"])
        if verify_producer_source_commit(args.source_commit) != producer:
            raise ArchiveError("producer identity changed while archiving")
        audit = build_total_audit(results, destination_root, producer=producer)
        audit_path = publish_total_audit(destination_root, audit)
        if verify_producer_source_commit(args.source_commit) != producer:
            raise ArchiveError("producer identity changed during aggregate publication")
        reporter.finish("PASS")
    except BaseException as exc:
        reporter.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    output = {
        "status": "PASS",
        "aggregate_audit": str(audit_path),
        "aggregate_audit_sha256": sha256_file(audit_path, no_follow=True),
        "progress_state": str(reporter.path),
        "source_commit": producer["source_commit"],
        "producer": producer,
        "archives": results,
        "removal_authorized": False,
        "removal_executed": False,
        "source_files_deleted": False,
        "worktree_remove_or_prune_executed": False,
        "future_removal_requires_separate_user_authorization": True,
    }
    print(json.dumps(output, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
