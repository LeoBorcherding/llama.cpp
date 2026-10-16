#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Tests for assemble_metadata.py. Run: python3 scripts/unsloth/test_assemble_metadata.py

The source tarballs in dist/ are the stamped tree the release publishes, not codeload's bytes,
so their sha256 entries must name the publish repo. On a plain (no-pin) build source_repo is
ggml-org/llama.cpp, and an entry under that repo sends the installer's source fallback to an
asset that does not exist there and a codeload tarball that fails the recorded hash.
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "assemble_metadata.py"
FAILS = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  :: {extra}" if extra and not cond else ""))
    if not cond:
        FAILS.append(name)


spec = importlib.util.spec_from_file_location("assemble_metadata", SCRIPT)
am = importlib.util.module_from_spec(spec)
spec.loader.exec_module(am)
# Offline: a plain build of an upstream tag would otherwise list ggml-org's release assets.
am.upstream_assets = lambda tag, token: {}
am.sha256_url = lambda url, token: "c0de" * 16
# main() refuses a dist/ with no CUDA bundle; one placeholder with stubbed embedded info is enough.
am.read_bundle_info = lambda bundle: {
    "bundle_profile": "cuda13-newer", "runtime_line": "cuda13", "coverage_class": "newer",
    "supported_sms": ["90"], "min_sm": 90, "max_sm": 90, "bundle_rank": 1, "toolkit_line": "13.3",
}

TAG, COMMIT, PUBLISH = "b11600", "0123456789abcdef" * 2 + "01234567", "unslothai/llama.cpp"


def run(source_repo, pr_set, with_local):
    d = Path(tempfile.mkdtemp(prefix="am_"))
    dist, out = d / "dist", d / "out"
    dist.mkdir()
    (dist / f"app-{TAG}-linux-x64-cuda13-newer.tar.gz").write_bytes(b"bundle")
    if with_local:
        for name in (f"llama.cpp-source-{TAG}.tar.gz", f"llama.cpp-source-commit-{COMMIT}.tar.gz"):
            (dist / name).write_bytes(b"stamped tree")
    sys.argv = ["assemble_metadata.py", "--tag", TAG, "--source-repo", source_repo,
                "--pr-set", json.dumps(pr_set), "--commit", COMMIT, "--dist", str(dist),
                "--out", str(out), "--publish-repo", PUBLISH]
    assert am.main() == 0
    return json.loads((out / "llama-prebuilt-sha256.json").read_text())["artifacts"]


SOURCES = (f"llama.cpp-source-{TAG}.tar.gz", f"llama.cpp-source-commit-{COMMIT}.tar.gz")

plain = run(am.UPSTREAM_REPO, [], with_local=True)
for name in SOURCES:
    check(f"plain build records {name} under the publish repo",
          plain[name]["repo"] == PUBLISH, plain[name]["repo"])

pin = {"repo": "ggml-org/llama.cpp", "number": 1, "sha": "a" * 40, "url": "u", "title": "t"}
mix = run(PUBLISH, [pin], with_local=True)
for name in SOURCES:
    check(f"mix build still records {name} under the publish repo", mix[name]["repo"] == PUBLISH)

# No dist/ copy means the digest came from codeload, so the bytes live in source_repo.
remote = run(am.UPSTREAM_REPO, [], with_local=False)
for name in SOURCES:
    check(f"a codeload-hashed {name} stays under source_repo",
          remote[name]["repo"] == am.UPSTREAM_REPO, remote[name]["repo"])

print()
print(f"{len(FAILS)} failure(s)" if FAILS else "all assemble_metadata tests passed")
sys.exit(1 if FAILS else 0)
