#!/usr/bin/env python3
"""The store helpers the recipe calls, from the lab's analysis-store command line (scripts/analysis/canto/cli.py).

    repaired_rows(of, step)      a derived collection's layer 0: each clip at the file a rewriting step made
    training_set(coll, out, ...) the trainer's --train_jsonl from a collection's ingest, audio and encode layers
    resolve_params(coll, step)   tonefit's templates key made explicit, so it is part of the recipe
    cmd_status(collections)      each step's current layer and its siblings (recipe.py status)

Kept unchanged; the lab's SOURCES (paths on its own disks), its export and its instrument tests are not here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import store as st  # noqa: E402
import steps  # noqa: E402,F401  (registers the steps)


def repaired_rows(of: str = "donor", step: str = "repair_tail", key: str | None = None,
                  exclude: frozenset = frozenset()) -> tuple[list[dict], dict]:
    """A derived collection: the raw collection's clips, each pointing at its repaired file where the repair layer
    made one. Its source names the repair layer's key, so a different repair is a different layer 0. `exclude` names
    clips left out on purpose (a caller's registered exclusions); an errored clip NOT named there is refused."""
    s = st.Store(of)
    key = key or s.refs()[step]
    rep = s.records(step, key, errors="keep" if exclude else "refuse")
    bad = [c for c, r in rep.items() if "error" in r and c not in exclude]
    if bad:
        raise SystemExit(f"{of}/{step}/{key}: {len(bad)} errored row(s) not in the exclusion set, first {bad[0]}")
    rows = []
    for cid, c in s.records("ingest").items():
        if cid in exclude:
            continue
        r = rep[cid]
        rows.append({**{k: v for k, v in c.items() if k not in ("deps", "sha256", "path")},
                     "path": r["path"], "repaired": r["repaired"], "raw_path": c["path"]})
    return rows, {"what": f"{of} through {step}", "repair_key": key, "of_ingest": s.refs()["ingest"],
                  "kind": "derived", "licence": "CC-0",
                  **({"excluded": len(exclude), "excluded_sha256": st.h(sorted(exclude), 64)} if exclude else {})}


def training_set(collection: str, out: Path, speaker_id: str = "cv_voice6", exclude: frozenset = frozenset()) -> int:
    """Write a trainer --train_jsonl from a collection's clips and its current encode layer: audio, text, ref_audio
    (the clip itself, as the speaker set does), speaker_id, duration_sec and audio_codes. `exclude` names clips left
    out on purpose; an errored clip NOT named there is refused (the store's default)."""
    s = st.Store(collection)
    ing = s.refs()["ingest"]
    for step in ("encode", "audio"):
        parent = s.layer_meta(step)["parents"].get("ingest")
        if parent != ing:
            raise SystemExit(f"{collection}: the current {step} layer was made from ingest {parent}, not the current "
                             f"{ing}; run {step} again before joining (review, 2026-10-10)")
    mode = "keep" if exclude else "refuse"
    base, enc, aud = s.records("ingest"), s.records("encode", errors=mode), s.records("audio", errors=mode)
    bad = [c for layer in (enc, aud) for c, r in layer.items() if "error" in r and c not in exclude]
    if bad:
        raise SystemExit(f"{collection}: {len(bad)} errored row(s) not in the exclusion set, first {bad[0]}")
    rows = []
    for cid, c in sorted(base.items()):
        if cid in exclude:
            continue
        codes = s.arrays("encode", cid)["codes"].tolist()
        rows.append({"audio": c["path"], "text": c["text"], "ref_audio": c["path"], "speaker_id": speaker_id,
                     "duration_sec": aud[cid]["duration_s"], "audio_codes": codes})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    meta = {"collection": collection, "ingest": s.refs()["ingest"], "encode": s.refs()["encode"],
            "audio": s.refs()["audio"], "rows": len(rows),
            **({"excluded": len(exclude), "excluded_sha256": st.h(sorted(exclude), 64)} if exclude else {})}
    (out.parent / (out.name + ".provenance.json")).write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(meta))
    return len(rows)


def parse_set(pairs: list[str]) -> dict:
    out = {}
    for kv in pairs or []:
        k, v = kv.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def cmd_status(collections: list[str]) -> None:
    for c in collections:
        s = st.Store(c)
        refs = s.refs()
        print(f"== {c}")
        for step in ["ingest"] + [n for n in st.REGISTRY if n != "ingest"]:
            ls = s.layers(step)
            if not ls:
                continue
            cur = refs.get(step)
            for m in ls:
                mark = "*" if m["key"] == cur else " "
                n = m.get("clips") or (m.get("computed", 0) + m.get("reused", 0))
                print(f"  {mark} {step:10s} {m['key']}  v{m.get('version')}  {'complete' if m['complete'] else 'PARTIAL '}"
                      f"  n={n}  errors={m.get('errors', 0)}  {json.dumps(m.get('params', {}), ensure_ascii=False)[:90]}")


def resolve_params(collection: str, step: str, params: dict, root: Path | None = None) -> dict:
    """Make every cross-collection reference explicit, so it is in the recipe: tonefit's templates key."""
    if step == "tonefit":
        src = params.get("templates_from", st.REGISTRY["tonefit"].params["templates_from"])
        if not params.get("templates_key"):
            params["templates_key"] = st.Store(src, root).refs().get("speaker")
            if not params["templates_key"]:
                raise SystemExit(f"tonefit needs {src}'s speaker layer; run it first")
    return params
