#!/usr/bin/env python3
"""Bake the chosen voice vectors into a merged speaker-phase checkpoint: rows of talker.model.codec_embedding copied
from a source checkpoint, and the config's spk_id naming them.

WHY A BAKE AND NOT THE TRAINED ROW. The speaker phase writes the voice it trained (the mean speaker-encoder vector of
its clips) at row 3000. The recipe ships a CHOSEN single-clip vector instead (owner, 2026-10-17: "choose, never
assume"), today v0.0.1's 343 vector at row 3000, and a second voice row costs nothing (the 777 vector at row 3001).
Both come from a checkpoint that already carries them: in the lab, the v0.0.4 release's merged weights
(exp083.stage_bake). This script is that stage with the source as an argument.

What it does, in the target directory, in place:
  * each named row of talker.model.codec_embedding.weight set to the source's same row, cast to the target's dtype;
  * talker_config.spk_id set to exactly the named voices (and spk_is_dialect to False for each, if the key exists);
  * refuses unless every baked row then equals the source's bit for bit, and unless tts_model_size is 1b7 (Qwen-TTS
    Studio's converter defaults a missing size to 0b6, a silently mislabelled model).

It records, as JSON on stdout and in --record: each row's equality with the source and the cosine between the trained
row 3000 and the baked one (how far the chosen voice is from the voice the phase learned).

    python src/bake_voices.py --target RUN/spk/final_merged --source RELEASE_MERGED \\
        --voice hk_voice_343=3000 --voice hk_voice_777=3001 --record RUN/bake.json
    python src/bake_voices.py --self-test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

KEY = "talker.model.codec_embedding.weight"


def codec_file(d: Path) -> Path:
    """The safetensors shard of `d` holding the codec embedding."""
    from safetensors import safe_open
    for f in sorted(Path(d).glob("model*.safetensors")):
        with safe_open(str(f), "pt") as s:
            if KEY in s.keys():
                return f
    raise SystemExit(f"no {KEY} in {d}")


def bake(target: Path, source: Path, voices: dict[str, int]) -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    cfg_p = target / "config.json"
    cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
    if cfg.get("tts_model_size") != "1b7":
        raise SystemExit(f"{target}: tts_model_size is {cfg.get('tts_model_size')!r}, not '1b7'")
    with safe_open(str(codec_file(source)), "pt") as s:
        src = s.get_tensor(KEY)
    f = codec_file(target)
    with safe_open(str(f), "pt") as s:
        meta = s.metadata()
    t = load_file(str(f))
    if any(r >= t[KEY].shape[0] or r >= src.shape[0] for r in voices.values()):
        raise SystemExit(f"a row in {voices} is past the table ({t[KEY].shape[0]} / {src.shape[0]} rows)")
    trained = t[KEY][3000].clone()
    for row in voices.values():
        t[KEY][row] = src[row].to(t[KEY].dtype)
    save_file(t, str(f), metadata=meta)
    cfg["talker_config"]["spk_id"] = dict(voices)
    if "spk_is_dialect" in cfg["talker_config"]:
        cfg["talker_config"]["spk_is_dialect"] = {v: False for v in voices}
    cfg_p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with safe_open(str(f), "pt") as s:
        back = s.get_tensor(KEY)
    rec = {"target": str(target), "source": str(source), "spk_id": cfg["talker_config"]["spk_id"],
           "rows_equal_source": {v: bool(torch.equal(back[r], src[r].to(back.dtype))) for v, r in voices.items()},
           "row3000_cos_trained_vs_baked": round(float(torch.nn.functional.cosine_similarity(
               trained.float(), back[3000].float(), dim=0)), 6)}
    if not all(rec["rows_equal_source"].values()):
        raise SystemExit(f"the baked rows are not the source's: {rec}")
    return rec


def parse_voices(specs: list[str]) -> dict[str, int]:
    out = {}
    for s in specs:
        name, row = s.split("=", 1)
        if name in out or int(row) in out.values():
            raise SystemExit(f"voice {s} repeats a name or a row")
        out[name] = int(row)
    return out


def self_test() -> int:
    """A two-shard fixture: the bake copies exactly the named rows, sets spk_id, and refuses a non-1b7 config."""
    import tempfile
    import torch
    from safetensors.torch import save_file
    fails = 0

    def check(name, cond):
        nonlocal fails
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        fails += not cond
    with tempfile.TemporaryDirectory() as td:
        src, dst = Path(td) / "src", Path(td) / "dst"
        for d, fill in ((src, 1.0), (dst, 2.0)):
            d.mkdir()
            save_file({"other.weight": torch.zeros(2)}, str(d / "model-00001-of-00002.safetensors"))
            e = torch.full((3072, 8), fill, dtype=torch.bfloat16)
            e[3000] = torch.arange(8) + fill
            save_file({KEY: e}, str(d / "model-00002-of-00002.safetensors"), metadata={"format": "pt"})
            (d / "config.json").write_text(json.dumps({"tts_model_size": "1b7", "talker_config": {
                "spk_id": {"hk_voice_343": 3000}, "spk_is_dialect": {"hk_voice_343": False}}}), encoding="utf-8")
        rec = bake(dst, src, {"hk_voice_343": 3000, "hk_voice_777": 3001})
        from safetensors.torch import load_file
        t = load_file(str(dst / "model-00002-of-00002.safetensors"))[KEY]
        check("the named rows equal the source's", all(rec["rows_equal_source"].values())
              and torch.equal(t[3001], torch.full((8,), 1.0, dtype=torch.bfloat16)))
        check("RED: an unnamed row is untouched", torch.equal(t[2999], torch.full((8,), 2.0, dtype=torch.bfloat16)))
        cfg = json.loads((dst / "config.json").read_text(encoding="utf-8"))["talker_config"]
        check("spk_id names exactly the baked voices", cfg["spk_id"] == {"hk_voice_343": 3000, "hk_voice_777": 3001}
              and cfg["spk_is_dialect"] == {"hk_voice_343": False, "hk_voice_777": False})
        check("the trained-vs-baked cosine is recorded", -1.0 <= rec["row3000_cos_trained_vs_baked"] <= 1.0)
        bad = json.loads((dst / "config.json").read_text(encoding="utf-8"))
        bad["tts_model_size"] = "0b6"
        (dst / "config.json").write_text(json.dumps(bad), encoding="utf-8")
        try:
            bake(dst, src, {"hk_voice_343": 3000})
            check("RED: a 0b6 config is refused", False)
        except SystemExit:
            check("RED: a 0b6 config is refused", True)
    print("PASS" if not fails else f"FAIL: {fails}")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", type=Path, help="the speaker phase's merged export (changed in place)")
    ap.add_argument("--source", type=Path, help="a merged checkpoint carrying the chosen voice rows")
    ap.add_argument("--voice", action="append", default=[], help="NAME=ROW, e.g. hk_voice_343=3000")
    ap.add_argument("--record", type=Path, help="write the record here too")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not (a.target and a.source and a.voice):
        ap.error("--target, --source and at least one --voice are required")
    rec = bake(a.target, a.source, parse_voices(a.voice))
    print(json.dumps(rec, indent=1))
    if a.record:
        a.record.parent.mkdir(parents=True, exist_ok=True)
        a.record.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
