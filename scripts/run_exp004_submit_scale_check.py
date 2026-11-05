"""Large-scale / offline re-validation for EXP-004 submit bundle."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.submission import validate_submission
from src.utils import ensure_dir, resolve_path

N_SAMPLE = 50_000
SEED = 42
BUNDLE = resolve_path("outputs/submissions/EXP-004_bundle")
ZIP_PATH = resolve_path("outputs/submissions/EXP-004_submit.zip")
REPORT = resolve_path("outputs/submissions/EXP-004_submit_scale_report.md")
TRAIN = resolve_path("data/raw/train.csv")
OFFICIAL_TEST = resolve_path("data/raw/test.csv")
SAMPLE_SUB = resolve_path("data/raw/sample_submission.csv")


def grep_network(script: Path) -> dict:
    text = script.read_text(encoding="utf-8")
    patterns = [
        r"\brequests\b",
        r"\burllib\b",
        r"\bhttp\.client\b",
        r"\bsocket\b",
        r"\bhttpx\b",
        r"\baiohttp\b",
        r"https?://",
        r"\burlopen\b",
        r"\bwget\b",
        r"\bftplib\b",
    ]
    hits = []
    for pat in patterns:
        for i, line in enumerate(text.splitlines(), 1):
            if re.search(pat, line) and not line.strip().startswith("#"):
                # ignore docstring URL-ish in comments already handled; allow Path only
                hits.append({"pattern": pat, "line": i, "text": line.strip()})
    return {"n_hits": len(hits), "hits": hits}


def build_50k_test() -> tuple[Path, Path, list[str]]:
    data_dir = ensure_dir(BUNDLE / "data")
    out_dir = ensure_dir(BUNDLE / "output")
    # clear previous output
    for p in out_dir.glob("*"):
        p.unlink()

    official_test = pd.read_csv(OFFICIAL_TEST, nrows=0, encoding="utf-8-sig")
    test_cols = list(official_test.columns)  # 48 cols, no target
    assert "control_success" not in test_cols

    print(f"Sampling {N_SAMPLE} rows from train...", flush=True)
    t0 = time.perf_counter()
    # chunked sample for memory friendliness
    rng = np.random.default_rng(SEED)
    # Reservoir would be ideal; for simplicity read usecols and sample
    train = pd.read_csv(TRAIN, encoding="utf-8-sig", low_memory=False)
    if len(train) < N_SAMPLE:
        raise ValueError(f"train has only {len(train)} rows")
    idx = rng.choice(len(train), size=N_SAMPLE, replace=False)
    idx.sort()
    part = train.iloc[idx].copy()
    # ensure test-format columns only
    missing = [c for c in test_cols if c not in part.columns]
    if missing:
        raise ValueError(f"train missing test columns: {missing}")
    test_df = part[test_cols].copy()
    # unique row_ids already from train; keep them
    test_path = data_dir / "test.csv"
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    sample = pd.DataFrame(
        {
            "row_id": test_df["row_id"].to_numpy(),
            "control_success": np.full(len(test_df), 0.5, dtype=float),
        }
    )
    sample_path = data_dir / "sample_submission.csv"
    sample.to_csv(sample_path, index=False, encoding="utf-8-sig")
    print(
        f" wrote test={test_path} rows={len(test_df)} cols={test_df.shape[1]} "
        f"prep_s={time.perf_counter() - t0:.1f}",
        flush=True,
    )
    return test_path, sample_path, test_cols


def run_script(env_extra: dict | None = None, label: str = "host") -> dict:
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    if env_extra:
        env.update(env_extra)
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "script.py"],
        cwd=str(BUNDLE),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    wall = time.perf_counter() - t0
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    timing = {}
    for line in stdout.splitlines():
        if "TIMING_SUMMARY" in line:
            for tok in line.split():
                if "=" in tok and tok.split("=")[0].endswith("_s"):
                    k, v = tok.split("=", 1)
                    try:
                        timing[k] = float(v)
                    except ValueError:
                        pass
    return {
        "label": label,
        "returncode": proc.returncode,
        "wall_s": wall,
        "timing": timing,
        "stdout_tail": "\n".join(stdout.splitlines()[-30:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-20:]),
    }


def validate_out(sample_path: Path) -> dict:
    out_path = BUNDLE / "output" / "submission.csv"
    sample = pd.read_csv(sample_path, encoding="utf-8-sig")
    out = pd.read_csv(out_path, encoding="utf-8")
    validate_submission(sample, out, ["row_id"], probability=True)
    return {
        "ok": True,
        "n_rows": len(out),
        "columns": list(out.columns),
        "dtypes": {c: str(out[c].dtype) for c in out.columns},
        "pred_min": float(out["control_success"].min()),
        "pred_max": float(out["control_success"].max()),
        "pred_mean": float(out["control_success"].mean()),
    }


def rebuild_zip() -> None:
    # refresh script.py into zip without wrapping folder
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(BUNDLE / "script.py", arcname="script.py")
        zf.write(BUNDLE / "requirements.txt", arcname="requirements.txt")
        zf.write(
            BUNDLE / "model" / "catboost_primary.joblib",
            arcname="model/catboost_primary.joblib",
        )
        zf.write(
            BUNDLE / "model" / "feature_meta.json",
            arcname="model/feature_meta.json",
        )
    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = zf.namelist()
    bad = [n for n in names if n.startswith("EXP-004") or n.count("/") > 1 and not n.startswith("model/")]
    # only allow script.py, requirements.txt, model/*
    allowed_prefix_ok = all(
        n in {"script.py", "requirements.txt"} or n.startswith("model/") for n in names
    )
    if not allowed_prefix_ok:
        raise RuntimeError(f"Unexpected zip entries: {names}")


def main() -> None:
    net = grep_network(BUNDLE / "script.py")
    print("network_grep", net["n_hits"], flush=True)

    _, sample_path, test_cols = build_50k_test()

    # Normal run
    run_host = run_script(label="host_50k")
    print(run_host["stdout_tail"], flush=True)
    if run_host["returncode"] != 0:
        print(run_host["stderr_tail"], flush=True)
        raise SystemExit(f"script failed rc={run_host['returncode']}")
    val = validate_out(sample_path)

    # Offline-ish: block common proxy/env + NO_NETWORK style
    offline_env = {
        "HTTP_PROXY": "http://127.0.0.1:9",
        "HTTPS_PROXY": "http://127.0.0.1:9",
        "ALL_PROXY": "http://127.0.0.1:9",
        "http_proxy": "http://127.0.0.1:9",
        "https_proxy": "http://127.0.0.1:9",
        "NO_PROXY": "",
        "PIP_NO_INDEX": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
    }
    # clear previous output so we know it rewrote
    out_csv = BUNDLE / "output" / "submission.csv"
    if out_csv.exists():
        out_csv.unlink()
    run_off = run_script(env_extra=offline_env, label="proxy_blocked_50k")
    print(run_off["stdout_tail"], flush=True)
    offline_ok = run_off["returncode"] == 0 and out_csv.exists()
    val_off = validate_out(sample_path) if offline_ok else {"ok": False}

    rebuild_zip()
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zip_names = zf.namelist()

    lines = [
        "# EXP-004 submit — 50k-scale / offline re-check",
        "",
        f"- bundle: `{BUNDLE}`",
        f"- zip: `{ZIP_PATH}`",
        f"- sample size: `{N_SAMPLE}` rows from train → test-format ({len(test_cols)} cols)",
        f"- seed: `{SEED}`",
        "",
        "## 1. Inference timing (50k rows)",
        "",
        "| stage | host_50k (s) | proxy_blocked_50k (s) |",
        "|---|---:|---:|",
    ]
    keys = [
        "model_load_s",
        "data_load_s",
        "feature_prep_s",
        "predict_proba_s",
        "write_s",
        "wall_s",
    ]
    for k in keys:
        a = run_host["timing"].get(k, float("nan"))
        b = run_off["timing"].get(k, float("nan"))
        lines.append(f"| `{k}` | {a:.3f} | {b:.3f} |")
    lines.append(
        f"| subprocess wall (includes python startup) | {run_host['wall_s']:.3f} | "
        f"{run_off['wall_s']:.3f} |"
    )
    lines.extend(
        [
            "",
            "## 2. Output validation vs sample_submission",
            "",
            f"- host validate: `{val}`",
            f"- offline validate: `{val_off}`",
            "",
            "## 3. Network / offline checks",
            "",
            f"- grep hits for requests/urllib/http/socket/etc in `script.py`: "
            f"**{net['n_hits']}**",
        ]
    )
    if net["hits"]:
        for h in net["hits"]:
            lines.append(f"  - L{h['line']}: `{h['text']}`")
    lines.extend(
        [
            f"- re-run with HTTP(S)_PROXY → 127.0.0.1:9: "
            f"**{'OK' if offline_ok else 'FAIL'}** (rc={run_off['returncode']})",
            "",
            "## 4. Zip top-level after script refresh",
            "",
            "```",
            *zip_names,
            "```",
            "",
            "## Notes",
            "",
            "- `model_load_s` = joblib load of CatBoost; `predict_proba_s` = inference only.",
            "- Proxy-block run does not fully sandbox sockets; combined with grep=0 "
            "supports no intentional network use in `script.py`.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORT}", flush=True)
    print(json.dumps({"host_timing": run_host["timing"], "offline_ok": offline_ok, "val": val}, indent=2))


if __name__ == "__main__":
    main()
