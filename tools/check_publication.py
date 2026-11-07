"""Fail closed on unexpected paths, raw data, changed bytes or unsafe staged files."""
import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "publication_manifest.json"
SCRIPTS = {
    "magviz.py", "download.py", "analysis.py", "snapshots.py", "anatomy.py",
    "make_movies.py", "keogram_eq.py", "probes.py", "repro_cache.py",
    "reproduce.py", "figure_diagnostics.py", "recon_redo.py",
    "recon_schematic.py", "recon_slide_figures.py", "location_insets.py",
}
FIGURES = {
    "seq_eq", "seq_mer", "anatomy_mer", "global_mer_t150",
    "fig_keogram_eq_sunearth", "fig_keogram_eq_sunearth_t125_175",
    "repro_pressure_balance", "repro_shue_fit", "repro_shock_rh",
    "repro_qperp_qpar", "repro_brazil", "sheath_eq_t150",
    "day_mer_t90", "day_mer_t125", "day_mer_t150", "repro_fte_probes",
    "khi_eq_t120", "khi_eq_t150", "khi_eq_t180", "repro_khi_criterion",
    "fig_flank_probes", "fig_flank_reynolds", "fig_avg_eq",
    "fig_keogram_mer_sunearth", "fig_tail_xline_eq",
    "tail_mer_t90", "tail_mer_t125", "tail_mer_t150",
    "tail_eq_t85", "tail_eq_t125", "tail_eq_t160", "repro_tail_flows",
    "repro_df_bbf", "recon_rate_v2", "repro_convection", "fig_avg_mer",
    "recon_two_scale_schematic", "hall_quadrupole_v2", "hall_quadrant_ts_v2",
    "loc_s07", "loc_s08", "loc_s11", "loc_s14", "loc_s15", "loc_s16",
    "loc_s17", "loc_s19", "recon_dayside_v2", "recon_anatomy",
}
ARRAY_KEYS = {
    "processed/eq_timeseries.npz": {
        "t", "tsec", "mp", "bs", "flank_mp", "flank_bs", "up_N",
        "up_Vx", "up_Bz", "up_T", "up_Pdyn",
    },
    "processed/mer_timeseries.npz": {"t", "xline", "lobe_B"},
    "processed/figure_diagnostics_profiles.npz": {
        "x", "Pi_xx_ion_magnetic", "scalar",
    },
}
DATA = set(ARRAY_KEYS) | {
    "processed/diagnostics.json", "processed/figure_diagnostics.json",
    "processed/probes.json", "processed/reconnection_candidates.csv",
    "processed/reconnection_summary.json",
}
TEXT = {
    ".gitignore", "README.md", "requirements.txt", "requirements-download.txt",
    "docs/DATA_POLICY.md", "docs/PROVENANCE.md", "docs/REPRODUCING.md",
    "docs/source_snapshot.json", "processed/README.md",
    "output/report.md", "output/slides/script.md",
    "tools/check_publication.py", "tools/plot_summaries.py",
    "tests/test_publication.py",
}
ALLOWED = SCRIPTS | TEXT | DATA | {f"output/figs/{f}.png" for f in FIGURES}
CSV_FIELDS = [
    "t", "x0", "z0", "nX", "vjet_e", "vjet_t", "EN", "ES",
    "TN", "TS", "full", "Ey", "cls",
]


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def validate_file(name, content):
    if name not in ALLOWED:
        raise ValueError(f"Not on the publication allowlist: {name}")
    limit = 5 * 1024**2 if name.endswith(".png") else 256 * 1024
    if len(content) > limit:
        raise ValueError(f"File too large for its publication class: {name}")
    if name in ARRAY_KEYS:
        with np.load(io.BytesIO(content), allow_pickle=False) as arrays:
            if set(arrays.files) != ARRAY_KEYS[name]:
                raise ValueError(f"Unexpected NPZ keys: {name}")
            lengths = set()
            for key in arrays.files:
                a = arrays[key]
                if a.ndim != 1 or a.size > 1000 or a.dtype.kind not in "fiu":
                    raise ValueError(f"Not a reduced numeric 1D series: {name}:{key}")
                lengths.add(a.size)
            expected = 379 if "eq_timeseries" in name else (
                190 if "mer_timeseries" in name else 671
            )
            if lengths != {expected}:
                raise ValueError(f"Unexpected reduced-series length: {name}")
    elif name.endswith(".png"):
        if not content.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"Invalid PNG: {name}")
    else:
        text = content.decode("utf-8")
        for pattern in (
            r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
            r"\b(?:ghp_|github_pat_|gitea_)[A-Za-z0-9_]{20,}",
            r"(?i)\b(?:password|api_key|access_token)\s*[:=]\s*['\"][^'\"\r\n]{8,}['\"]",
            r"(?i)[A-Z]:[\\/]Users[\\/][^\\/\s]+",
        ):
            if re.search(pattern, text):
                raise ValueError(f"Credential or personal path pattern in {name}")
        if name.endswith(".json"):
            json.loads(text)


def validate_reconnection(files):
    table = list(csv.DictReader(io.StringIO(
        files["processed/reconnection_candidates.csv"].decode("utf-8")
    )))
    if len(table) != 90 or list(table[0]) != CSV_FIELDS:
        raise ValueError("Reconnection table must have the documented 90-row schema")
    for row in table:
        numeric = [float(row[k]) for k in CSV_FIELDS if k not in ("full", "cls")]
        if not np.all(np.isfinite(numeric)):
            raise ValueError("Non-finite reconnection table value")
        selected = (
            float(row["vjet_e"]) > 300 and float(row["vjet_t"]) < -300
            and int(row["nX"]) <= 2
        )
        if selected != (row["cls"] == "clean"):
            raise ValueError("Flow selection does not match class")
        full = all(
            float(row[key]) * sign > 0
            for key, sign in zip(("EN", "ES", "TN", "TS"), (1, -1, -1, 1))
        )
        if row["full"] != str(full):
            raise ValueError("Four-sign flag does not match measured quadrants")
    summary = json.loads(files["processed/reconnection_summary.json"])
    for group in ("clean", "inactive", "other", "all"):
        rows = [r for r in table if group == "all" or r["cls"] == group]
        result = summary["classes"][group]
        nfull = sum(r["full"] == "True" for r in rows)
        if result["n"] != len(rows) or result["full_quadrupole"] != nfull:
            raise ValueError(f"Incorrect class counts: {group}")
        for quadrant in ("EN", "ES", "TN", "TS"):
            np.testing.assert_allclose(
                np.mean([float(r[quadrant]) for r in rows]),
                result["By_mean"][quadrant], rtol=1e-10,
            )
    if summary["frames"] != 90:
        raise ValueError("Incorrect total count")
    clean = [r for r in table if r["cls"] == "clean"]
    if len(clean) != 28 or sum(r["full"] == "True" for r in clean) != 21:
        raise ValueError("Expected 21/28 selected frames")
    if sum(r["full"] == "True" for r in table) != 27:
        raise ValueError("Expected 27/90 candidate frames")


def collect(staged):
    top = Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve()
    if top != ROOT:
        raise ValueError("Run inside the separate Open repository, not a parent")
    args = ("ls-files", "--cached", "-z")
    if not staged:
        args = ("ls-files", "--cached", "--others", "--exclude-standard", "-z")
    names = set(git(*args).decode("utf-8").split("\0")) - {"", MANIFEST}
    if names != ALLOWED:
        raise ValueError(
            f"Publication paths differ: missing={sorted(ALLOWED - names)}, "
            f"unexpected={sorted(names - ALLOWED)}"
        )
    files = {}
    for name in sorted(names):
        path = ROOT / name
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f"Symlink or external file: {name}")
        if staged:
            mode = git("ls-files", "--stage", "--", name).decode().split()[0]
            if mode not in ("100644", "100755"):
                raise ValueError(f"Non-regular index entry: {name}")
        files[name] = git("show", ":" + name) if staged else path.read_bytes()
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    if args.staged and args.write_manifest:
        parser.error("Cannot rewrite the manifest during a staged check")
    files = collect(args.staged)
    for name, content in files.items():
        validate_file(name, content)
    if sum(len(v) for k, v in files.items() if k in DATA) > 1024**2:
        raise ValueError("Processed data exceeds 1 MiB")
    validate_reconnection(files)
    entries = [
        {"path": name, "bytes": len(content),
         "sha256": hashlib.sha256(content).hexdigest(),
         "kind": "processed-summary" if name in DATA else (
             "self-generated-figure" if name.endswith(".png") else "source-or-document"
         )}
        for name, content in sorted(files.items())
    ]
    manifest = {"schema": 1, "raw_data_included": False, "files": entries}
    target = ROOT / MANIFEST
    if args.write_manifest:
        target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    else:
        saved = git("show", ":" + MANIFEST) if args.staged else target.read_bytes()
        if json.loads(saved) != manifest:
            raise ValueError("Manifest mismatch; inspect changes before updating it")
    size = sum(len(content) for content in files.values())
    small = sum(len(v) for k, v in files.items() if k in DATA)
    print(
        f"PASS: {len(entries)} allowlisted files; {size / 1024**2:.2f} MiB; "
        f"processed data {small / 1024:.1f} KiB; no raw paths; Hall 21/28 and 27/90."
    )


if __name__ == "__main__":
    main()
