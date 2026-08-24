from pathlib import Path


def test_public_s3_0_files_do_not_contain_private_paths():
    roots = [
        Path("configs/tactile_unit"),
        Path("gr00t/tactile_unit"),
        Path("scripts/tactile_unit"),
        Path("tests/tactile_unit"),
    ]
    forbidden = (
        "/" + "home/",
        "/" + "mnt/",
        "ugreen" + "_nas",
        "Authorization" + ":",
        "Bear" + "er ",
        "github" + "_pat_",
        "gh" + "p_",
    )
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".json", ".md"}:
                text = path.read_text(errors="replace")
                assert not any(term in text for term in forbidden), path
