"""Store repeated SQL/CPU reference files once, retaining their original paths."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("action", choices=["pack", "check", "restore"])
    args = parser.parse_args()
    root = args.results.resolve()
    archive = root / "reference_fixtures.json.gz"
    if args.action == "pack":
        assert not archive.exists(), "Check or restore the existing archive first"
        contents, files = {}, {}
        for path in sorted(root.rglob("*")):
            if not path.is_file() or not (
                path.suffix == ".sql" or "_reference.json" in path.name
            ):
                continue
            data = path.read_bytes()
            compressed = path.suffix == ".gz"
            text = (gzip.decompress(data) if compressed else data).decode()
            key = digest(text)
            contents[key] = text
            files[path.relative_to(root).as_posix()] = dict(
                sha256=key, compressed=compressed
            )
        payload = dict(version=1, contents=contents, files=files)
        archive.write_bytes(
            gzip.compress(json.dumps(payload, sort_keys=True).encode(), mtime=0)
        )
        assert json.loads(gzip.decompress(archive.read_bytes())) == payload
        for relative in files:
            (root / relative).unlink()
    else:
        payload = json.loads(gzip.decompress(archive.read_bytes()))
    assert payload["version"] == 1
    for key, text in payload["contents"].items():
        assert digest(text) == key
    for relative, record in payload["files"].items():
        path = (root / relative).resolve()
        assert path.is_relative_to(root)
        text = payload["contents"][record["sha256"]]
        if args.action == "restore":
            data = text.encode()
            if record["compressed"]:
                data = gzip.compress(data, mtime=0)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                current = path.read_bytes()
                if record["compressed"]:
                    current = gzip.decompress(current)
                assert current.decode() == text, f"Existing fixture differs: {path}"
            else:
                path.write_bytes(data)
    print(
        f"{archive}: {len(payload['files'])} paths, "
        f"{len(payload['contents'])} unique fixtures; {args.action} verified"
    )


if __name__ == "__main__":
    main()
