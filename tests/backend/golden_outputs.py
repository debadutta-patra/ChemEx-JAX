# ChemEx-JAX fork addition (GPL-3.0-or-later); not part of upstream ChemEx.
"""Generate and compare golden NumPy-backend outputs for every shipped example.

The fork's hard rule is that the NumPy backend stays byte-identical to upstream.
This tool runs each ``examples/*/*/run.sh`` unmodified, through a ``chemex``
shim that redirects ``-o`` to a scratch directory and forces deterministic,
plot-free execution (``--plot nothing --workers 1 --native-threads 1``).  It
then hashes every output file except ``run_info/`` (timestamps and paths).

Statistics ``*.json`` evidence files embed per-run ``uuid4``-derived
occurrence identities (and hashes chained from them), so they differ between
two runs of unmodified upstream code.  For ``*.json`` files only, values of
keys whose name ends in ``identity`` are blanked before hashing; every other
value, and every other file, is compared byte for byte.

Usage::

    uv run python tests/backend/golden_outputs.py generate [--jobs N] [NAMES...]
    uv run python tests/backend/golden_outputs.py compare  [--jobs N] [--reuse] [NAMES...]
    uv run python tests/backend/golden_outputs.py rehash

``rehash`` rebuilds the manifest from the existing ``.golden/`` outputs
without rerunning anything; ``compare --reuse`` hashes ``.golden-check/``
without rerunning.  ``generate`` writes outputs to ``.golden/<group>/<example>/`` and the SHA-256
manifest to ``tests/backend/golden_manifest.json`` (committed).  ``compare``
reruns the examples into ``.golden-check/`` and reports any file whose hash
differs from the manifest, including missing or extra files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
MANIFEST = Path(__file__).with_name("golden_manifest.json")
EXCLUDED_DIRS = {"run_info"}

_SHIM = """#!{python}
import os, sys
args = sys.argv[1:]
out = os.environ["CHEMEX_GOLDEN_OUT"]
rewritten, skip = [], False
for i, arg in enumerate(args):
    if skip:
        skip = False
        continue
    if arg in ("-o", "--output"):
        rewritten += [arg, out]
        skip = True
        continue
    if arg in ("--plot", "--workers", "--native-threads"):
        skip = True
        continue
    rewritten.append(arg)
if "-o" not in rewritten and "--output" not in rewritten:
    rewritten += ["-o", out]
rewritten += ["--plot", "nothing", "--workers", "1", "--native-threads", "1"]
from chemex._entrypoint import main
sys.argv = ["chemex", *rewritten]
sys.exit(main())
"""


def _examples(names: list[str]) -> list[Path]:
    """Run scripts: ``run.sh`` per example, or each ``run*.sh`` if it has none."""
    found = []
    for directory in sorted(p for p in EXAMPLES.glob("*/*") if p.is_dir()):
        if (directory / "run.sh").exists():
            found.append(directory / "run.sh")
        else:
            found += sorted(directory.glob("run*.sh"))
    if names:
        found = [
            p
            for p in found
            if p.parent.name in names or _key(p) in names or p.stem in names
        ]
    return found


def _key(script: Path) -> str:
    example = script.parent
    key = f"{example.parent.name}/{example.name}"
    return key if script.name == "run.sh" else f"{key}/{script.stem}"


def _shim_dir(scratch: Path) -> Path:
    shim_dir = scratch / ".bin"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / "chemex"
    shim.write_text(_SHIM.format(python=sys.executable))
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    return shim_dir


def _run(script: Path, out_root: Path, shim_dir: Path) -> tuple[str, int, str]:
    out = out_root / _key(script)
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PATH"] = f"{shim_dir}{os.pathsep}{env['PATH']}"
    env["CHEMEX_GOLDEN_OUT"] = str(out)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[var] = "1"
    proc = subprocess.run(  # noqa: S603 - repository example script
        ["sh", script.name],  # noqa: S607
        cwd=script.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return _key(script), proc.returncode, proc.stdout[-2000:] + proc.stderr[-4000:]


def _blank_identities(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: "<identity>" if key.endswith("identity") else _blank_identities(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_blank_identities(item) for item in value]
    return value


def _digest(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix == ".json":
        canonical = _blank_identities(json.loads(data))
        data = json.dumps(canonical, indent=1, sort_keys=False).encode()
    return hashlib.sha256(data).hexdigest()


def _hashes(out: Path) -> dict[str, str]:
    result = {}
    for path in sorted(out.rglob("*")):
        rel = path.relative_to(out)
        if not path.is_file() or rel.parts[0] in EXCLUDED_DIRS:
            continue
        result[rel.as_posix()] = _digest(path)
    return result


def _rehash_all(names: list[str], out_root: Path) -> dict[str, dict[str, str]]:
    return {_key(e): _hashes(out_root / _key(e)) for e in _examples(names)}


def _run_all(names: list[str], out_root: Path, jobs: int) -> dict[str, dict[str, str]]:
    examples = _examples(names)
    shim_dir = _shim_dir(out_root)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(lambda e: _run(e, out_root, shim_dir), examples))
    failed = [(key, log) for key, code, log in results if code != 0]
    for key, log in failed:
        print(f"FAILED {key}\n{log}", file=sys.stderr)
    if failed:
        sys.exit(1)
    return {_key(e): _hashes(out_root / _key(e)) for e in examples}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("generate", "compare", "rehash"))
    parser.add_argument("names", nargs="*", help="example names (default: all)")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument(
        "--reuse", action="store_true", help="compare: hash existing outputs only"
    )
    args = parser.parse_args()

    if args.action == "rehash":
        manifest = _rehash_all(args.names, ROOT / ".golden")
        MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        print(f"Rehashed {len(manifest)} examples into {MANIFEST}")
        return

    if args.action == "generate":
        manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
        manifest.update(_run_all(args.names, ROOT / ".golden", args.jobs))
        MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        n_files = sum(len(v) for v in manifest.values())
        print(f"Wrote {len(manifest)} examples / {n_files} files to {MANIFEST}")
        return

    manifest = json.loads(MANIFEST.read_text())
    names = args.names or list(manifest)
    check_root = ROOT / ".golden-check"
    actual = (
        _rehash_all(names, check_root)
        if args.reuse
        else _run_all(names, check_root, args.jobs)
    )
    n_bad = 0
    for key, files in actual.items():
        expected = manifest.get(key)
        if expected is None:
            print(f"NEW      {key} (not in manifest)")
            n_bad += 1
            continue
        for name in sorted(expected.keys() | files.keys()):
            if expected.get(name) != files.get(name):
                state = (
                    "MISSING"
                    if name not in files
                    else "EXTRA"
                    if name not in expected
                    else "DIFFERS"
                )
                print(f"{state:8} {key}/{name}")
                n_bad += 1
    n_files = sum(len(v) for v in actual.values())
    print(f"{len(actual)} examples, {n_files} files, {n_bad} differences")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
