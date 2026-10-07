from __future__ import annotations

import errno
import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from scripts.simulation import archive_pi2s_worktree_locals as archive


def _fixture_producer() -> dict[str, str]:
    return {
        "path": str(Path(archive.__file__).resolve()),
        "repository_relative_path": "scripts/simulation/archive_pi2s_worktree_locals.py",
        "sha256": "a" * 64,
        "source_commit": "0" * 40,
    }


def _fixture_worktree(root: Path, external: Path, name: str = "source-worktree") -> Path:
    worktree = root / name
    local = worktree / ".local"
    (local / "artifacts" / "nested").mkdir(parents=True)
    (local / "artifacts" / "result.json").write_bytes(b'{"status":"PASS"}\n')
    (local / "artifacts" / "nested" / "negative.bin").write_bytes(b"negative-result\x00")
    (local / "external-alias").symlink_to(external)
    return worktree


def test_archive_copies_regular_files_without_following_symlinks(tmp_path: Path) -> None:
    external = tmp_path / "external-evidence"
    external.mkdir()
    (external / "must-not-be-copied.bin").write_bytes(b"external")
    worktree = _fixture_worktree(tmp_path, external)
    destination = tmp_path / "retirement-backups"
    destination.mkdir()

    producer = _fixture_producer()
    result = archive.archive_one(worktree, destination, producer=producer)

    final = Path(result["archive"])
    bundle = archive.publication_bundle_path(final)
    payload = bundle / "payload" / ".local"
    assert result["archive_status"] == "CREATED_AND_VERIFIED"
    assert result["publication_state"] == "COMPLETE"
    assert (payload / "artifacts" / "result.json").read_bytes() == b'{"status":"PASS"}\n'
    assert (payload / "artifacts" / "nested" / "negative.bin").read_bytes() == (
        b"negative-result\x00"
    )
    assert (payload / "external-alias").is_symlink()
    assert os.readlink(payload / "external-alias") == str(external)
    assert not (payload / "external-alias" / "must-not-be-copied.bin").is_symlink()
    assert (external / "must-not-be-copied.bin").read_bytes() == b"external"
    assert (worktree / ".local" / "artifacts" / "result.json").exists()
    assert stat.S_IMODE(final.stat().st_mode) == 0o700
    assert stat.S_IMODE((bundle / "manifest.json").stat().st_mode) == 0o600
    assert archive.inspect_directory_publication(final)["state"] == "COMPLETE"
    assert (final / archive.PUBLICATION_INCOMPLETE).is_dir()
    assert (final / archive.PUBLICATION_COMPLETE).is_dir()
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["source_commit"] == producer["source_commit"]
    assert manifest["producer"] == producer

    verified = archive.archive_one(worktree, destination, producer=producer)
    assert verified["archive_status"] == "VERIFIED_EXISTING"
    assert verified["manifest_sha256"] == result["manifest_sha256"]


def test_existing_archive_is_never_overwritten_when_source_changes(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    worktree = _fixture_worktree(tmp_path, external)
    destination = tmp_path / "retirement-backups"
    destination.mkdir()
    producer = _fixture_producer()
    first = archive.archive_one(worktree, destination, producer=producer)
    manifest_path = archive.publication_bundle_path(Path(first["archive"])) / "manifest.json"
    manifest_before = manifest_path.read_bytes()

    (worktree / ".local" / "artifacts" / "result.json").write_bytes(b"changed\n")

    with pytest.raises(archive.ArchiveError, match="refusing overwrite"):
        archive.archive_one(worktree, destination, producer=producer)
    assert manifest_path.read_bytes() == manifest_before


def test_source_change_before_publish_retains_private_staging_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    worktree = _fixture_worktree(tmp_path, external)
    destination = tmp_path / "retirement-backups"
    destination.mkdir()
    original_build_alias_ledger = archive.build_alias_ledger

    def mutate_source_after_staged_verification(
        source_local: Path, records: list[dict[str, object]]
    ) -> dict[str, object]:
        ledger = original_build_alias_ledger(source_local, records)
        (source_local / "artifacts" / "result.json").write_bytes(b"changed-before-publish\n")
        return ledger

    monkeypatch.setattr(archive, "build_alias_ledger", mutate_source_after_staged_verification)

    with pytest.raises(archive.ArchiveError, match="changed before archive publication"):
        archive.archive_one(worktree, destination, producer=_fixture_producer())

    final = destination / archive.archive_name(worktree)
    publication = archive.inspect_directory_publication(final)
    assert publication["state"] == "INCOMPLETE"
    assert publication["has_bundle"] is False
    staging_directories = list(destination.glob(".*.staging-*"))
    assert len(staging_directories) == 1
    assert stat.S_IMODE(staging_directories[0].stat().st_mode) == 0o700
    assert (worktree / ".local" / "artifacts" / "result.json").read_bytes() == (
        b"changed-before-publish\n"
    )


def test_incomplete_adopted_bundle_is_recovered_without_recopy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    worktree = _fixture_worktree(tmp_path, external)
    destination = tmp_path / "retirement-backups"
    destination.mkdir()
    producer = _fixture_producer()
    real_mark_complete = archive.mark_directory_publication_complete

    def simulated_crash_after_adoption(final: Path) -> str:
        raise archive.ArchiveError(f"simulated crash before COMPLETE: {final}")

    monkeypatch.setattr(
        archive,
        "mark_directory_publication_complete",
        simulated_crash_after_adoption,
    )
    with pytest.raises(archive.ArchiveError, match="simulated crash before COMPLETE"):
        archive.archive_one(worktree, destination, producer=producer)

    final = destination / archive.archive_name(worktree)
    interrupted = archive.inspect_directory_publication(final)
    assert interrupted["state"] == "INCOMPLETE"
    assert interrupted["has_bundle"] is True
    assert (worktree / ".local" / "artifacts" / "result.json").exists()

    monkeypatch.setattr(archive, "mark_directory_publication_complete", real_mark_complete)
    recovered = archive.archive_one(worktree, destination, producer=producer)
    assert recovered["archive_status"] == "RECOVERED_AND_VERIFIED"
    assert recovered["publication_state"] == "COMPLETE"
    assert archive.inspect_directory_publication(final)["state"] == "COMPLETE"


def test_existing_archive_rejects_false_symlink_safety_claim(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    worktree = _fixture_worktree(tmp_path, external)
    destination = tmp_path / "retirement-backups"
    destination.mkdir()
    producer = _fixture_producer()
    result = archive.archive_one(worktree, destination, producer=producer)
    audit_path = archive.publication_bundle_path(Path(result["archive"])) / "archive_audit.json"
    audit = json.loads(audit_path.read_text())
    audit["symlink_referents_followed"] = True
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(archive.ArchiveError, match="symlink_referents_followed"):
        archive.archive_one(worktree, destination, producer=producer)


def test_directory_publication_transitions_without_replacing_bundle(tmp_path: Path) -> None:
    final = tmp_path / "final"
    first_staging = tmp_path / "first-staging"
    first_staging.mkdir(mode=0o700)
    (first_staging / "payload").write_bytes(b"first")

    assert archive.inspect_directory_publication(final)["state"] == "ABSENT"
    assert archive.ensure_directory_publication_claim(final) == "MKDIR_CLAIM_CREATED"
    claimed = archive.inspect_directory_publication(final)
    assert claimed["state"] == "INCOMPLETE"
    assert claimed["has_bundle"] is False
    assert archive.adopt_staging_directory(first_staging, final) == (
        "PRIVATE_BUNDLE_RENAME_INTO_EXISTING_CLAIM"
    )
    adopted = archive.inspect_directory_publication(final)
    assert adopted["state"] == "INCOMPLETE"
    assert adopted["has_bundle"] is True
    assert not first_staging.exists()

    second_staging = tmp_path / "second-staging"
    second_staging.mkdir(mode=0o700)
    (second_staging / "payload").write_bytes(b"second")
    assert archive.adopt_staging_directory(second_staging, final) == "BUNDLE_ALREADY_PRESENT"
    assert second_staging.is_dir()
    assert (second_staging / "payload").read_bytes() == b"second"
    assert (archive.publication_bundle_path(final) / "payload").read_bytes() == b"first"

    assert archive.mark_directory_publication_complete(final) == "MKDIR_COMPLETE_MARKER"
    assert archive.inspect_directory_publication(final)["state"] == "COMPLETE"
    assert archive.mark_directory_publication_complete(final) == ("COMPLETE_MARKER_ALREADY_PRESENT")


def test_publication_rejects_existing_content_and_symlink_claims(tmp_path: Path) -> None:
    occupied = tmp_path / "occupied"
    occupied.mkdir(mode=0o700)
    sentinel = occupied / "do-not-replace"
    sentinel.write_bytes(b"preserved")
    with pytest.raises(archive.ArchiveError, match="unexpected entries"):
        archive.ensure_directory_publication_claim(occupied)
    assert sentinel.read_bytes() == b"preserved"

    referent = tmp_path / "referent"
    referent.mkdir()
    (referent / "sentinel").write_bytes(b"external")
    alias = tmp_path / "alias"
    alias.symlink_to(referent, target_is_directory=True)
    with pytest.raises(archive.ArchiveError, match="not a real directory"):
        archive.ensure_directory_publication_claim(alias)
    assert (referent / "sentinel").read_bytes() == b"external"


def test_publication_rejects_symlink_bundle_without_following_it(tmp_path: Path) -> None:
    final = tmp_path / "final"
    archive.ensure_directory_publication_claim(final)
    content = final / archive.PUBLICATION_CONTENT
    content.mkdir(mode=0o700)
    external = tmp_path / "external"
    external.mkdir()
    (external / "secret").write_bytes(b"must-not-be-read")
    (content / archive.PUBLICATION_BUNDLE).symlink_to(external, target_is_directory=True)

    with pytest.raises(archive.ArchiveError, match="not a real directory"):
        archive.inspect_directory_publication(final)

    assert (external / "secret").read_bytes() == b"must-not-be-read"


def test_empty_bundle_is_never_adopted_over_or_marked_complete(tmp_path: Path) -> None:
    final = tmp_path / "final"
    archive.ensure_directory_publication_claim(final)
    content = final / archive.PUBLICATION_CONTENT
    content.mkdir(mode=0o700)
    (content / archive.PUBLICATION_BUNDLE).mkdir(mode=0o700)
    staging = tmp_path / "staging"
    staging.mkdir(mode=0o700)
    (staging / "evidence").write_bytes(b"preserved")

    with pytest.raises(archive.ArchiveError, match="publication bundle is empty"):
        archive.adopt_staging_directory(staging, final)
    with pytest.raises(archive.ArchiveError, match="publication bundle is empty"):
        archive.mark_directory_publication_complete(final)

    assert (staging / "evidence").read_bytes() == b"preserved"
    assert not (final / archive.PUBLICATION_COMPLETE).exists()


def test_cross_filesystem_rename_failure_keeps_incomplete_claim_and_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final = tmp_path / "final"
    archive.ensure_directory_publication_claim(final)
    staging = tmp_path / "staging"
    staging.mkdir(mode=0o700)
    (staging / "evidence").write_bytes(b"preserved")

    def fail_rename(*args: object, **kwargs: object) -> None:
        raise OSError(errno.EXDEV, "simulated cross-device rename")

    monkeypatch.setattr(archive.os, "rename", fail_rename)
    with pytest.raises(OSError, match="simulated cross-device rename"):
        archive.adopt_staging_directory(staging, final)

    publication = archive.inspect_directory_publication(final)
    assert publication["state"] == "INCOMPLETE"
    assert publication["has_bundle"] is False
    assert (staging / "evidence").read_bytes() == b"preserved"


def test_claim_name_substitution_cannot_report_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final = tmp_path / "final"
    staging = tmp_path / "staging"
    staging.mkdir(mode=0o700)
    (staging / "evidence").write_bytes(b"verified")
    archive.adopt_staging_directory(staging, final)
    moved = tmp_path / "moved-claim"
    external = tmp_path / "external"
    external.mkdir()
    real_mkdir = archive.os.mkdir

    def substitute_after_complete_marker(
        path: object,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        real_mkdir(path, mode, dir_fd=dir_fd)
        if path == archive.PUBLICATION_COMPLETE and dir_fd is not None:
            final.rename(moved)
            final.symlink_to(external, target_is_directory=True)

    monkeypatch.setattr(archive.os, "mkdir", substitute_after_complete_marker)
    with pytest.raises(archive.ArchiveError, match="publication claim name"):
        archive.mark_directory_publication_complete(final)

    assert final.is_symlink()
    assert not list(external.iterdir())
    assert (moved / archive.PUBLICATION_COMPLETE).is_dir()


def test_destination_uid_gid_is_not_payload_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    worktree = _fixture_worktree(tmp_path, external)
    expected = archive.scan_tree(worktree / ".local", include_hash=True)
    remapped = json.loads(json.dumps(expected))
    for row in remapped:
        row["uid"] += 10_000
        row["gid"] += 20_000

    monkeypatch.setattr(archive, "scan_tree", lambda *args, **kwargs: remapped)
    verification = archive.verify_payload(tmp_path / "unused", expected)
    assert verification["status"] == "PASS"
    assert verification["destination_uid_gid_is_not_archive_identity"] is True


def test_publication_modes_do_not_depend_on_process_umask(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    worktree = _fixture_worktree(tmp_path, external)
    destination = tmp_path / "retirement-backups"
    destination.mkdir()

    previous_umask = os.umask(0o777)
    try:
        result = archive.archive_one(worktree, destination, producer=_fixture_producer())
    finally:
        os.umask(previous_umask)

    final = Path(result["archive"])
    bundle = archive.publication_bundle_path(final)
    assert stat.S_IMODE(final.stat().st_mode) == 0o700
    assert stat.S_IMODE((final / archive.PUBLICATION_INCOMPLETE).stat().st_mode) == 0o700
    assert stat.S_IMODE((final / archive.PUBLICATION_COMPLETE).stat().st_mode) == 0o700
    assert stat.S_IMODE(bundle.stat().st_mode) == 0o700
    assert stat.S_IMODE((bundle / "manifest.json").stat().st_mode) == 0o600


def test_aggregate_audit_uses_directory_protocol_without_hardlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "retirement-backups"
    destination.mkdir()
    source_roots = []
    for index in range(3):
        external = tmp_path / f"external-{index}"
        external.mkdir()
        source_roots.append(_fixture_worktree(tmp_path, external, name=f"source-worktree-{index}"))
    monkeypatch.setattr(archive, "SOURCE_ROOTS", tuple(source_roots))

    def forbidden_hardlink(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"hardlink unexpectedly attempted: {args!r} {kwargs!r}")

    monkeypatch.setattr(archive.os, "link", forbidden_hardlink)
    producer = _fixture_producer()
    results = [
        archive.archive_one(source, destination, producer=producer) for source in source_roots
    ]
    audit = archive.build_total_audit(
        results,
        destination,
        producer=producer,
    )

    audit_path = archive.publish_total_audit(destination, audit)
    publication_root = destination / archive.AGGREGATE_PUBLICATION_NAME
    assert audit_path == archive.publication_bundle_path(publication_root) / (
        archive.AGGREGATE_AUDIT_NAME
    )
    assert json.loads(audit_path.read_text())["status"] == "PASS"
    assert archive.inspect_directory_publication(publication_root)["state"] == "COMPLETE"
    assert archive.publish_total_audit(destination, audit) == audit_path
    tampered = json.loads(audit_path.read_text())
    tampered["archives"][0]["publication_state"] = "INCOMPLETE"
    audit_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(archive.ArchiveError, match="aggregate row is not COMPLETE"):
        archive.publish_total_audit(destination, audit)


def test_aggregate_rejects_missing_components(tmp_path: Path) -> None:
    destination = tmp_path / "retirement-backups"
    destination.mkdir()
    with pytest.raises(archive.ArchiveError, match="requires exactly 3 archives"):
        archive.build_total_audit([], destination, producer=_fixture_producer())


def test_archive_rejects_destination_with_symlinked_parent(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    worktree = _fixture_worktree(tmp_path, external)
    real_parent = tmp_path / "real-parent"
    destination = real_parent / "retirement-backups"
    destination.mkdir(parents=True)
    alias = tmp_path / "aliased-parent"
    alias.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(archive.ArchiveError, match="destination path contains"):
        archive.archive_one(
            worktree,
            alias / destination.name,
            producer=_fixture_producer(),
        )

    assert list(destination.iterdir()) == []


def test_producer_must_match_script_blob_in_current_head(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    script = repository / "scripts" / "archive.py"
    script.parent.mkdir(parents=True)
    script.write_bytes(b"print('committed producer')\n")
    subprocess.run(["git", "init", "--quiet"], cwd=repository, check=True)
    subprocess.run(["git", "add", "--", "scripts/archive.py"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Archive Fixture",
            "-c",
            "user.email=archive-fixture@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture producer",
        ],
        cwd=repository,
        check=True,
    )
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()

    producer = archive.verify_producer_source_commit(
        source_commit,
        repository_root=repository,
        script_path=script,
    )
    assert producer["source_commit"] == source_commit
    assert producer["repository_relative_path"] == "scripts/archive.py"
    assert producer["sha256"] == archive.sha256_file(script, no_follow=True)

    script.write_bytes(b"print('uncommitted producer')\n")
    with pytest.raises(archive.ArchiveError, match="differs from its source-commit blob"):
        archive.verify_producer_source_commit(
            source_commit,
            repository_root=repository,
            script_path=script,
        )


def test_progress_state_exposes_observed_rate_and_eta(tmp_path: Path) -> None:
    progress = tmp_path / "progress.json"
    reporter = archive.ProgressReporter(
        progress,
        goal_uuid="fixture-goal",
        producer=_fixture_producer(),
    )
    reporter.begin_phase(tmp_path, "COPY_TO_STAGING", 100)
    reporter.advance(25, tmp_path / "source.bin")
    reporter.persist(force=True)

    state = json.loads(progress.read_text())
    assert state["phase_bytes_completed"] == 25
    assert state["goal_uuid"] == "fixture-goal"
    assert state["source_commit"] == "0" * 40
    assert state["phase_bytes_total"] == 100
    assert state["phase_elapsed_seconds"] >= 0.0
    assert state["phase_rate_bytes_per_second"] > 0.0
    assert state["phase_eta_seconds"] >= 0.0
    assert state["producer"] == _fixture_producer()
    assert stat.S_IMODE(progress.stat().st_mode) == 0o600
