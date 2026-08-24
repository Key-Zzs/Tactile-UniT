from pathlib import Path


def test_public_s2_files_do_not_contain_private_paths():
    roots = [
        Path("configs/contact_dynamics"),
        Path("gr00t/contact_dynamics"),
        Path("scripts/contact_dynamics"),
        Path("tests/contact_dynamics"),
    ]
    forbidden = (
        "/" + "home/",
        "/" + "mnt/",
        "ugreen" + "_nas",
        "Authorization" + ":",
        "Bear" + "er ",
    )
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".json"}:
                text = path.read_text(errors="replace")
                assert not any(term in text for term in forbidden), path
