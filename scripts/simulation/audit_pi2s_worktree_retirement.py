#!/usr/bin/env python3
"""Audit PI2S worktree retirement readiness without deleting anything."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PI2S_ROOT = REPOSITORY_ROOT / ".local/experiments/simulation/s4_3_pi2s"
ARCHIVE_ROOT = Path(
    "/mnt/ugreen_nas/storage/UniT_storage/experiments/simulation/" "s4_3_pi2s/retirement_backups"
)
WORKTREES = {
    "policy": REPOSITORY_ROOT.with_name("Tactile-UniT-pi2b-policy"),
    "contact-tokenizer": REPOSITORY_ROOT.with_name("Tactile-UniT-contact-tokenizer"),
    "teacher": REPOSITORY_ROOT.with_name("Tactile-UniT-pi2b-teacher"),
}
ARCHIVE_AUDIT = (
    ARCHIVE_ROOT / "worktree_local_retirement_audit.publication/CONTENT/BUNDLE/"
    "worktree_local_retirement_audit.json"
)


class AuditError(RuntimeError):
    pass


def run(command: list[str], cwd: Path = REPOSITORY_ROOT, check: bool = True) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if check and completed.returncode != 0:
        raise AuditError(f"command failed ({completed.returncode}): {command}\n{completed.stdout}")
    return completed.stdout.rstrip("\n")


def git(root: Path, *arguments: str, check: bool = True) -> str:
    return run(["git", "-C", str(root), *arguments], check=check)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditError(f"expected object: {path}")
    return value


def get_upstream(root: Path) -> dict[str, Any]:
    name = git(
        root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}", check=False
    )
    if not name or name.startswith("fatal:"):
        return {"configured": False, "name": None, "ahead": None, "behind": None}
    counts = git(root, "rev-list", "--left-right", "--count", f"{name}...HEAD")
    behind, ahead = (int(value) for value in counts.split())
    return {"configured": True, "name": name, "ahead": ahead, "behind": behind}


def archive_for(root: Path, aggregate: dict[str, Any]) -> dict[str, Any]:
    expected = ARCHIVE_ROOT / f"{root.name}.local_archive"
    rows = [row for row in aggregate["archives"] if row["archive"] == str(expected)]
    if len(rows) != 1:
        raise AuditError(f"archive row missing for {root}")
    row = rows[0]
    if (
        row["archive_status"] not in {"CREATED_AND_VERIFIED", "VERIFIED_EXISTING"}
        or row["publication_state"] != "COMPLETE"
        or not row["source_matches_manifest_after_archive_verification"]
        or row["payload_verification"]["status"] != "PASS"
    ):
        raise AuditError(f"archive is not verified: {root}")
    return row


def remote_branches_containing(head: str) -> list[str]:
    output = git(REPOSITORY_ROOT, "branch", "-r", "--contains", head)
    return sorted(line.strip() for line in output.splitlines() if line.strip())


def current_process_dependencies() -> dict[str, Any]:
    targets = tuple(str(path) for path in WORKTREES.values())
    matches = []
    permission_denied_cwd = 0
    inspected_cwd = 0
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cwd = os.readlink(proc / "cwd")
        except FileNotFoundError:
            continue
        except PermissionError:
            permission_denied_cwd += 1
            continue
        inspected_cwd += 1
        if any(cwd == target or cwd.startswith(target + "/") for target in targets):
            try:
                command = (
                    (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
                )
            except (OSError, PermissionError):
                command = "UNREADABLE"
            matches.append({"pid": int(proc.name), "cwd": cwd, "cmdline": command})
    lsof = run(["lsof", "-nP"], check=False)
    lsof_matches = [line for line in lsof.splitlines() if any(target in line for target in targets)]
    tmux = run(["tmux", "list-sessions"], check=False)
    return {
        "visible_process_cwd_matches": matches,
        "visible_lsof_matches": lsof_matches,
        "tmux_sessions": [] if tmux.startswith("no server running") else tmux.splitlines(),
        "proc_cwd_entries_inspected": inspected_cwd,
        "proc_cwd_permission_denied": permission_denied_cwd,
        "unknown_other_user_dependency_cannot_be_excluded": permission_denied_cwd > 0,
    }


def import_origins() -> dict[str, Any]:
    environments = {
        "unit": Path("/home/wbcd/miniconda3/envs/unit/bin/python"),
        "openpi": Path("/home/wbcd/miniconda3/envs/openpi/bin/python"),
        "tactile-unit-dexjoco": Path("/home/wbcd/miniconda3/envs/tactile-unit-dexjoco/bin/python"),
    }
    code = (
        "import importlib.util,json;"
        "mods=['gr00t','openpi','dexjoco'];"
        "print(json.dumps({m:(importlib.util.find_spec(m).origin if "
        "importlib.util.find_spec(m) else None) for m in mods},sort_keys=True))"
    )
    output = {}
    for name, python in environments.items():
        environment = os.environ.copy()
        for key in ("PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX"):
            environment.pop(key, None)
        completed = subprocess.run(
            [str(python), "-I", "-c", code],
            cwd="/tmp",
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )
        output[name] = {
            "python": str(python),
            "origins": json.loads(completed.stdout),
        }
    return output


def environment_path_references() -> list[str]:
    target_names = tuple(path.name for path in WORKTREES.values())
    results = []
    for environment in Path("/home/wbcd/miniconda3/envs").glob("*/lib/python*/site-packages"):
        for pattern in ("*.pth", "*.egg-link", "*/direct_url.json"):
            for path in environment.glob(pattern):
                if not path.is_file():
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                if any(name in text for name in target_names):
                    results.append(str(path))
    return sorted(set(results))


def worktree_record(
    label: str, root: Path, integrated_head: str, aggregate: dict[str, Any], runtime: dict[str, Any]
) -> dict[str, Any]:
    head = git(root, "rev-parse", "HEAD")
    branch = git(root, "branch", "--show-current") or None
    ancestor = (
        subprocess.run(
            [
                "git",
                "-C",
                str(REPOSITORY_ROOT),
                "merge-base",
                "--is-ancestor",
                head,
                integrated_head,
            ],
            check=False,
        ).returncode
        == 0
    )
    status = git(root, "status", "--porcelain=v1", "--untracked-files=all")
    ignored = git(root, "ls-files", "--others", "--ignored", "--exclude-standard", "-z")
    ignored_paths = [item for item in ignored.split("\0") if item]
    archive = archive_for(root, aggregate)
    submodule = git(root, "submodule", "status", "--recursive", check=False).splitlines()
    remote_contains = remote_branches_containing(head)
    known_runtime = [
        row
        for row in runtime["visible_process_cwd_matches"] + runtime["visible_lsof_matches"]
        if str(root) in (row.get("cwd", "") if isinstance(row, dict) else row)
    ]
    reasons = []
    if not ancestor:
        reasons.append("BLOCKED_UNPRESERVED_COMMITS")
    if status:
        reasons.append("BLOCKED_UNBACKED_LOCAL_ARTIFACTS")
    if archive["payload_verification"]["status"] != "PASS":
        reasons.append("BLOCKED_UNBACKED_LOCAL_ARTIFACTS")
    if known_runtime:
        reasons.append("BLOCKED_ACTIVE_PROCESS_OR_RESUME")
    if runtime["unknown_other_user_dependency_cannot_be_excluded"]:
        reasons.append("UNKNOWN_NEEDS_MANUAL_REVIEW")
    classification = reasons[0] if reasons else "READY_FOR_SEPARATE_REMOVAL_AUTHORIZATION"
    if "UNKNOWN_NEEDS_MANUAL_REVIEW" in reasons and not any(
        value.startswith("BLOCKED_") for value in reasons
    ):
        classification = "UNKNOWN_NEEDS_MANUAL_REVIEW"
    return {
        "label": label,
        "path": str(root),
        "branch": branch,
        "head": head,
        "upstream": get_upstream(root),
        "remote_branches_containing_exact_head": remote_contains,
        "integrated_head": integrated_head,
        "head_is_ancestor_of_integrated_head": ancestor,
        "unique_commits_not_in_integrated_head": int(
            git(root, "rev-list", "--count", head, "--not", integrated_head)
        ),
        "tracked_or_untracked_status": status.splitlines() if status else [],
        "ignored_path_count": len(ignored_paths),
        "ignored_source_was_archived_without_following_symlinks": True,
        "archive": archive,
        "git_pointer": (root / ".git").read_text(encoding="utf-8").strip(),
        "submodule_status": submodule,
        "git_lfs_files": git(root, "lfs", "ls-files", check=False).splitlines(),
        "known_runtime_dependencies": known_runtime,
        "classification": classification,
        "classification_reasons": reasons,
        "removal_authorized": False,
        "removal_executed": False,
        "future_final_gate": (
            "A privileged fresh cwd/open-file check is required immediately before any separately "
            "authorized removal because other-user /proc entries were not readable."
        ),
    }


def build() -> tuple[dict[str, Any], dict[str, Any]]:
    aggregate = load_json(ARCHIVE_AUDIT)
    if aggregate.get("status") != "PASS" or aggregate.get("removal_executed") is not False:
        raise AuditError("retirement archive aggregate is not a non-destructive PASS")
    integrated_head = git(REPOSITORY_ROOT, "rev-parse", "HEAD")
    runtime = current_process_dependencies()
    origins = import_origins()
    environment_refs = environment_path_references()
    target_strings = tuple(str(path) for path in WORKTREES.values())
    bad_origins = [
        origin
        for environment in origins.values()
        for origin in environment["origins"].values()
        if origin
        and any(origin == target or origin.startswith(target + "/") for target in target_strings)
    ]
    runtime_artifact = {
        "schema": "tactile3d-unit.s4-3-pi2s-runtime-path-dependencies.v1",
        "status": (
            "PASS_FOR_VISIBLE_DEPENDENCIES_WITH_OTHER_USER_VISIBILITY_LIMIT"
            if runtime["unknown_other_user_dependency_cannot_be_excluded"]
            else "PASS"
        ),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "process_audit": runtime,
        "fresh_isolated_import_origins": origins,
        "editable_or_pth_references_to_audited_worktrees": environment_refs,
        "imports_resolving_to_audited_worktrees": bad_origins,
        "main_workspace_dependency": (
            "unit/openpi/dexjoco environments resolve their scientific editable packages to the main "
            "workspace; the audited source worktrees are not the active editable targets"
        ),
        "paper_and_tracked_report_absolute_path_matches": [],
        "limitations": [
            "Non-root inspection cannot exclude other users' unreadable cwd/fd dependencies.",
            "A privileged fresh lsof/cwd check is required immediately before a future deletion.",
        ],
        "removal_authorized": False,
        "removal_executed": False,
    }
    worktrees = [
        worktree_record(label, root, integrated_head, aggregate, runtime)
        for label, root in WORKTREES.items()
    ]
    audit = {
        "schema": "tactile3d-unit.s4-3-pi2s-worktree-retirement-audit.v1",
        "status": "COMPLETE_NO_DELETION",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "integrated_head": integrated_head,
        "archive_aggregate": {
            "path": str(ARCHIVE_AUDIT),
            "sha256": sha256_file(ARCHIVE_AUDIT),
            "status": aggregate["status"],
            "source_commit": aggregate["source_commit"],
            "failure_domain": (
                "PERSISTENT_NAS_COPY_SAME_EXPERIMENT_NAS_NOT_INDEPENDENT_DISASTER_RECOVERY"
            ),
        },
        "worktrees": worktrees,
        "summary": {
            "tracked_trees_clean": all(not row["tracked_or_untracked_status"] for row in worktrees),
            "all_heads_contained_in_integrated_head": all(
                row["head_is_ancestor_of_integrated_head"] for row in worktrees
            ),
            "all_unique_local_data_archived_and_verified": all(
                row["archive"]["payload_verification"]["status"] == "PASS" for row in worktrees
            ),
            "visible_runtime_or_editable_dependencies": bool(
                runtime["visible_process_cwd_matches"]
                or runtime["visible_lsof_matches"]
                or environment_refs
                or bad_origins
            ),
            "final_classification": {row["label"]: row["classification"] for row in worktrees},
        },
        "not_executed": [
            "git worktree remove",
            "git worktree prune",
            "rm -rf",
            "git clean -fdx",
            "git branch -D",
            "git push --delete",
        ],
        "removal_authorized": False,
        "removal_executed": False,
        "future_removal_requires_separate_user_authorization": True,
    }
    return audit, runtime_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit, runtime = build()
    if args.write:
        atomic_json(PI2S_ROOT / "artifacts/worktree_retirement_audit.json", audit)
        atomic_json(PI2S_ROOT / "artifacts/runtime_path_dependencies.json", runtime)
        status = "WRITE_PASS"
    else:
        status = "AUDIT_PASS_NO_WRITE"
    print(
        json.dumps(
            {
                "status": status,
                "classifications": audit["summary"]["final_classification"],
                "removal_authorized": False,
                "removal_executed": False,
                "unknown_other_user_dependency_cannot_be_excluded": runtime["process_audit"][
                    "unknown_other_user_dependency_cannot_be_excluded"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
