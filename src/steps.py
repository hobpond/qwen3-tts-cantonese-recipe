"""The analysis and cleanup steps the recipe uses. Each is small, versioned and named for what it measures or
makes; a change of method is a new version, which makes a new layer beside the old one rather than rewriting it
(store.py). Copied from the lab's scripts/analysis/canto/steps.py with only the steps below kept; every kept
function is unchanged, so a layer's code hash (store.Step.code_sha) still names exactly what made it.

THE CLEANUP (the D1 recipe, locked 2026-10-22), in the order it runs:

    sidon      Sidon v0.1 speech restoration, every clip rewritten; env_corr records a restoration that wandered  GPU
    dynamics   the raw clip's loudness contour put back on the Sidon output (Sidon compresses dynamics)          CPU
    energy     a 10 ms level track, speech stretches, pauses, edge silence                                       CPU
    edges      stray sounds at both ends, labelled by kind, the speech core                                      CPU
    trim       both ends CUT to edges' speech core, 0.10 s before and 0.30 s after                               CPU
    pad        the cut restored as digital silence, back to the source length (SPEAKER clips only)               CPU
    audio      level, clipping, DC and duration per file                                                         CPU
    encode     the 12 Hz codec's [frames, 16] codes, one clip at a time, for the trainer                         GPU

THE MEASURES (the checkpoint card and the release gate read these):

    tails      the detached-tail detector D and D_loose (tail_detector.py)                    CPU
    f0         pYIN F0 at 10 ms, voicing probability                                          CPU
    cpp        cepstral peak prominence per frame, over voiced frames                         CPU  (phonation, glitches)
    g2p        each character's Jyutping and tone (pycantonese)                               text
    asr        Qwen3-ASR transcript, CER against the text, last character                     GPU
    align      character spans for the text (sensevoice-ctc or the Qwen3 forced aligner)      GPU
    syllables  per character: span, tone target, 10-point F0 contour, level                   CPU  (needs align, f0, energy, g2p)
    speaker    the collection's log-F0 mean and spread, and six tone templates                collection
    tonefit    per syllable: distance to each tone template, nearest tone                     CPU  (templates from a named collection)
    final      the last syllable of the main speech: F0 end, velocity, length                 CPU  (needs energy, tails, f0)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from store import Step, register


CJK = re.compile(r"[㐀-䶿一-鿿\U00020000-\U0002ffff]")
QWEN_ASR = "Qwen/Qwen3-ASR-1.7B"
QWEN_ALIGNER = "Qwen/Qwen3-ForcedAligner-0.6B"


def load(clip: dict, mono: bool = True):
    import soundfile as sf
    y, sr = sf.read(clip["path"], dtype="float32")
    if mono and y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr


def frames_db(y, sr, hop_s):
    hop = int(round(hop_s * sr))
    n = len(y) // hop
    if n == 0:
        return np.zeros(0)
    return 20 * np.log10(np.sqrt(np.mean(np.asarray(y[: n * hop], dtype=np.float64).reshape(n, hop) ** 2, axis=1)) + 1e-12)


def segments(db, hop_s, floor_db, merge_s, min_s):
    act = db > (db.max() + floor_db if len(db) else 0)
    segs, start = [], None
    for i, a in enumerate(act):
        if a and start is None:
            start = i
        if not a and start is not None:
            segs.append([start, i])
            start = None
    if start is not None:
        segs.append([start, len(act)])
    merged = []
    for s, e in segs:
        if merged and (s - merged[-1][1]) * hop_s < merge_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(round(s * hop_s, 3), round(e * hop_s, 3)) for s, e in merged if (e - s) * hop_s >= min_s]


def cjk_only(text: str) -> str:
    return "".join(CJK.findall(text))


# ---------------------------------------------------------------------- audio
def audio(clip, parents, p):
    """Level, clipping and DC offset of the file as stored."""
    y, sr = load(clip, mono=False)
    ch = 1 if y.ndim == 1 else y.shape[1]
    m = y if y.ndim == 1 else y.mean(axis=1)
    return {"sr": sr, "channels": ch, "duration_s": round(len(m) / sr, 3),
            "peak_dbfs": round(float(20 * np.log10(np.abs(y).max() + 1e-12)), 2),
            "rms_dbfs": round(float(20 * np.log10(np.sqrt(np.mean(m.astype(np.float64) ** 2)) + 1e-12)), 2),
            "clipped": int(np.sum(np.abs(y) >= p["clip_level"])), "dc": round(float(m.mean()), 6)}, None


register(Step("audio", "1", audio, params={"clip_level": 0.999}))


# ---------------------------------------------------------------------- energy
def energy(clip, parents, p):
    """A 10 ms level track; speech stretches within floor_db of the loudest frame (gaps under merge_s inside speech);
    the pauses between them; the silence at each edge."""
    y, sr = load(clip)
    db = frames_db(y, sr, p["hop_s"])
    segs = segments(db, p["hop_s"], p["floor_db"], p["merge_s"], p["min_s"])
    dur = len(y) / sr
    pauses = [(a[1], b[0], round(b[0] - a[1], 3)) for a, b in zip(segs, segs[1:])]
    return {"segments": segs, "pauses": pauses, "n_pauses": len(pauses),
            "longest_pause_s": max([x[2] for x in pauses] or [0.0]),
            "lead_s": segs[0][0] if segs else round(dur, 3), "trail_s": round(dur - segs[-1][1], 3) if segs else 0.0,
            "speech_start_s": segs[0][0] if segs else None, "speech_end_s": segs[-1][1] if segs else None,
            "max_db": round(float(db.max()), 2) if len(db) else None}, {"db": db.astype(np.float32)}


register(Step("energy", "1", energy, params={"hop_s": 0.01, "floor_db": -40.0, "merge_s": 0.15, "min_s": 0.04}))


# ---------------------------------------------------------------------- tails
def tails(clip, parents, p):
    """EXP-059's registered detector: D (energy or voiced stretch after a 0.4 s pause) and D_loose."""
    import tail_detector as bp
    y, sr = load(clip)
    t = bp.tails_of(y, sr)
    return {"rule": bp.TAIL_RULE, "D": t["strict"], "D_loose": t["loose"]}, None  # TAIL_RULE, never a hard-coded name


def _tail_code():
    import tail_detector as bp
    return bp.tail_code()


register(Step("tails", "union-v1", tails, code_deps=_tail_code))


# ---------------------------------------------------------------------- f0
def f0(clip, parents, p):
    """pYIN F0 every 10 ms (NaN where unvoiced) and its voicing probability; the clip's voiced share, median and
    spread in semitones."""
    import librosa
    y, sr = load(clip)
    hop = int(round(p["hop_s"] * sr))
    f, v, prob = librosa.pyin(y, sr=sr, fmin=p["fmin"], fmax=p["fmax"], frame_length=p["frame_length"],
                              hop_length=hop)
    fv = f[v & ~np.isnan(f)]
    rec = {"frames": int(len(f)), "voiced_frac": round(float(np.mean(v)), 4) if len(v) else 0.0}
    if len(fv) >= 5:
        st = 12 * np.log2(fv / np.median(fv))
        rec.update(median_hz=round(float(np.median(fv)), 2), p5_hz=round(float(np.percentile(fv, 5)), 2),
                   p95_hz=round(float(np.percentile(fv, 95)), 2),
                   iqr_st=round(float(np.percentile(st, 75) - np.percentile(st, 25)), 3),
                   mean_log_hz=round(float(np.mean(np.log(fv))), 5), n_voiced=int(len(fv)))
    return rec, {"f0": np.where(v, f, np.nan).astype(np.float32), "prob": prob.astype(np.float32)}


register(Step("f0", "pyin-1", f0, params={"hop_s": 0.01, "fmin": 65.0, "fmax": 500.0, "frame_length": 1024}))


# ---------------------------------------------------------------------- cpp
def parent_array(clip: dict, step: str, name: str):
    """A parent layer's per-clip array, from the exact layer this run was planned against."""
    import store as st
    a = st.Store(clip["_collection"], clip["_root"]).arrays(step, clip["clip_id"], clip["_parent_keys"][step])
    return None if a is None else a[name]


def cpp(clip, parents, p):
    """Cepstral peak prominence per 10 ms frame: the cepstrum's peak between the quefrencies of fmax and fmin, in dB
    above its regression line (Hillenbrand's CPP, unsmoothed). Summarized over the frames pYIN calls voiced. Low CPP:
    breathy, rough or glitched phonation."""
    import scipy.signal as ss
    y, sr = load(clip)
    win, hop = int(p["win_s"] * sr), int(p["hop_s"] * sr)
    if len(y) < win:
        return {"n": 0}, None
    fr = np.lib.stride_tricks.sliding_window_view(y, win)[::hop] * ss.get_window("hann", win)
    spec = 20 * np.log10(np.abs(np.fft.rfft(fr, n=2 * win)) + 1e-9)
    cep = 20 * np.log10(np.abs(np.fft.irfft(spec, axis=1))[:, :win] + 1e-9)
    q = np.arange(win) / sr
    lo, hi = int(sr / p["fmax"]), int(sr / p["fmin"])
    peak_i = lo + np.argmax(cep[:, lo:hi], axis=1)
    fit = np.arange(int(0.001 * sr), hi)
    x, Y = q[fit], cep[:, fit]
    xm = x.mean()
    slope = ((Y - Y.mean(axis=1, keepdims=True)) * (x - xm)).sum(axis=1) / ((x - xm) ** 2).sum()
    icpt = Y.mean(axis=1) - slope * xm
    vals = (cep[np.arange(len(cep)), peak_i] - (slope * q[peak_i] + icpt)).astype(np.float32)
    # frame i is centred at i*hop + win/2; pYIN's frame j (centred) at j*hop
    f0 = parent_array(clip, "f0", "f0")
    shift = int(round(win / 2 / hop))
    voiced = np.array([shift + i < len(f0) and not np.isnan(f0[shift + i]) for i in range(len(vals))])
    rec = {"n": int(len(vals)), "n_voiced": int(voiced.sum())}
    if voiced.sum() >= 5:
        v = vals[voiced]
        rec.update(median_db=round(float(np.median(v)), 3), p10_db=round(float(np.percentile(v, 10)), 3))
    return rec, {"cpp": vals}


register(Step("cpp", "1", cpp, parents=("f0",), params={"win_s": 0.04, "hop_s": 0.01, "fmin": 60.0, "fmax": 500.0}))


# ---------------------------------------------------------------------- g2p
TONE_SYL = re.compile(r"[a-z]+[1-6]")


def g2p(clip, parents, p):
    """Each CJK character's Jyutping and tone, from pycantonese's segmenter; None where it has no reading. The
    sentence-final particle is the last CJK character."""
    import pycantonese
    chars, jyut = [], []
    for word, jp in pycantonese.characters_to_jyutping(clip.get("text", "")):
        cs = CJK.findall(word)
        syls = TONE_SYL.findall(jp or "")
        if len(syls) != len(cs):
            syls = [None] * len(cs)
        chars += cs
        jyut += syls
    return {"chars": chars, "jyutping": jyut, "tones": [int(s[-1]) if s else None for s in jyut],
            "coverage": round(sum(s is not None for s in jyut) / max(1, len(jyut)), 3),
            "pycantonese": pycantonese.__version__}, None


register(Step("g2p", "pycantonese-5", g2p, reads_audio=False))


# ---------------------------------------------------------------------- the GPU models (one per process)
_MODELS: dict = {}


def _snapshot(repo: str) -> str:
    import glob
    import os
    hits = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*/"))
    if not hits:
        raise RuntimeError(f"{repo} is not in the local Hugging Face cache")
    return hits[0]


def _asr(device: str):
    if "asr" not in _MODELS:
        import torch
        from qwen_asr import Qwen3ASRModel
        dt = torch.bfloat16 if device.startswith("cuda") else torch.float32
        _MODELS["asr"] = Qwen3ASRModel.from_pretrained(_snapshot(QWEN_ASR), dtype=dt, device_map=device,
                                                       max_inference_batch_size=8)
    return _MODELS["asr"]


def _aligner(device: str):
    if "align" not in _MODELS:
        import torch
        from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner
        dt = torch.bfloat16 if device.startswith("cuda") else torch.float32
        _MODELS["align"] = Qwen3ForcedAligner.from_pretrained(_snapshot(QWEN_ALIGNER), dtype=dt, device_map=device)
    return _MODELS["align"]


def cer(ref: str, hyp: str) -> float:
    r, h_ = list(ref), list(hyp)
    d = list(range(len(h_) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h_) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h_[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h_)] / max(1, len(r))


def asr(clip, parents, p):
    """Qwen3-ASR-1.7B's transcript, language fixed to Cantonese; CER over CJK characters against the text; whether
    the last character (the particle) came back. Blind to tone: a listener, and so an ASR, recovers a sentence with
    wrong tones almost perfectly in quiet."""
    out = _asr(p["device"]).transcribe(clip["path"], language=p["language"])[0]
    ref, hyp = cjk_only(clip.get("text", "")), cjk_only(out.text)
    return {"hyp": out.text, "cer": round(cer(ref, hyp), 4), "last_char_match": bool(ref) and hyp[-1:] == ref[-1:],
            "model": QWEN_ASR}, None


register(Step("asr", "qwen3-asr-1.7b-1", asr, params={"language": "Cantonese", "device": "cuda:0"}, device="cuda"))


SENSEVOICE = "iic/SenseVoiceSmall"
SV_FRAME_S = 0.06  # SenseVoice's encoder frame: 10 ms fbank, low-frame-rate stacked by 6


def _sensevoice(device: str):
    if "sv" not in _MODELS:
        import glob
        import os
        from funasr import AutoModel
        hits = glob.glob(os.path.expanduser(f"~/.cache/modelscope/models/{SENSEVOICE.replace('/', '--')}/snapshots/*/"))
        if not hits:
            raise RuntimeError(f"{SENSEVOICE} is not in the local ModelScope cache")
        _MODELS["sv"] = AutoModel(model=hits[0], device=device, disable_update=True)
        _MODELS["sv"].kwargs["frontend"].dither = 0.0  # its default 1.0 adds fresh noise per call: no-op 17/20
    return _MODELS["sv"]


def sv_logprobs(y16, device: str):
    """SenseVoiceSmall's CTC log-probabilities over 60 ms frames for 16 kHz mono audio, with the four prompt frames
    (language yue, event, emotion, no ITN) dropped, so frame t starts at t * SV_FRAME_S."""
    import torch
    from funasr.utils.load_utils import extract_fbank
    m = _sensevoice(device)
    net, fe = m.model, m.kwargs["frontend"]
    speech, lens = extract_fbank([torch.as_tensor(y16, dtype=torch.float32)], data_type="sound", frontend=fe)
    speech, lens = speech.to(device), lens.to(device)
    q = lambda ids: net.embed(torch.LongTensor([ids]).to(device))
    speech = torch.cat((q([net.lid_dict["yue"]]), q([1, 2]), q([net.textnorm_dict["woitn"]]), speech), dim=1)
    with torch.no_grad():
        enc, el = net.encoder(speech, lens + 4)
        if isinstance(enc, tuple):
            enc = enc[0]
        lp = net.ctc.log_softmax(enc)[0, 4:int(el[0])]
    return lp.float(), net.blank_id, m.kwargs["tokenizer"]


def sv_align(y16, text: str, device: str) -> dict:
    """CTC forced alignment of the REFERENCE text (not a transcript) on SenseVoice's frames. CTC places each token as a
    spike, so a character's span is its onset to the next character's onset; the last runs to its spike's end plus
    tail_s. Each character's spike posterior is kept as its confidence."""
    import torch
    import torchaudio.functional as taf
    lp, blank, tk = sv_logprobs(y16, device)
    ids, owner = [], []
    oov = []
    for k, ch in enumerate(text):
        t = [i for i in tk.encode(ch) if i != blank]  # out of vocabulary encodes to <unk>, which is the blank
        if not t:
            oov.append(k)
        ids += t
        owner += [k] * len(t)
    if not ids or len(ids) > lp.shape[0]:
        return {"error": f"{len(ids)} tokens for {lp.shape[0]} frames"}
    ali, sc = taf.forced_align(lp[None], torch.tensor([ids], device=lp.device), blank=blank)
    spans = taf.merge_tokens(ali[0], sc[0].exp())
    first, last, conf = {}, {}, {}
    for tok, k in zip(spans, owner):
        first.setdefault(k, tok.start)
        last[k] = tok.end
        conf.setdefault(k, round(float(tok.score), 4))
    n_fr = lp.shape[0]
    end_fr = min(n_fr, max(last.values()) + 4)
    onset = {}
    for k in reversed(range(len(text))):  # an unaligned character sits at the next aligned one's onset, zero length
        onset[k] = first.get(k, onset.get(k + 1, end_fr))
    items = []
    for k, ch in enumerate(text):
        b = onset[k + 1] if k + 1 < len(text) else end_fr
        items.append((ch, round(onset[k] * SV_FRAME_S, 3), round(b * SV_FRAME_S, 3)))
    return {"items": items, "conf": [conf.get(k) for k in range(len(text))], "oov": oov, "frames": int(n_fr)}


def speech_bounds(y16, pad_s: float):
    """The speech region by energy (D's -40 dB floor, 40 ms minimum, so a lead click does not count), padded."""
    sg = segments(frames_db(y16, 16000, 0.01), 0.01, -40.0, 0.15, 0.04)
    dur = len(y16) / 16000
    return (0.0, dur) if not sg else (max(0.0, sg[0][0] - pad_s), min(dur, sg[-1][1] + pad_s))


def align_audio(y16, text: str, aligner: str, device: str = "cuda:0", language: str = "Cantonese",
                trim_s: float | None = None) -> dict:
    """One aligner on 16 kHz mono audio: {"items": [(char, start_s, end_s)], "conf": [...] or None}. trim_s: align
    on the speech region padded by trim_s and shift the spans back, so edge silence cannot be absorbed into the
    first or last character (Qwen3-ForcedAligner stretched a first character over 1.7 s of lead silence)."""
    if trim_s is not None:
        a, b = speech_bounds(y16, trim_s)
        out = align_audio(y16[int(a * 16000):int(b * 16000)], text, aligner, device, language)
        if "items" in out:
            out["items"] = [(c, round(x + a, 3), round(e + a, 3)) for c, x, e in out["items"]]
            out["trim"] = [round(a, 3), round(b, 3)]
        return out
    if aligner == "sensevoice-ctc":
        return sv_align(y16, text, device)
    if aligner == "qwen3-fa-0.6b":
        res = _aligner(device).align((np.asarray(y16, np.float32), 16000), text=text, language=language)[0]
        return {"items": [(it.text, round(float(it.start_time), 3), round(float(it.end_time), 3)) for it in res],
                "conf": None}
    raise ValueError(f"unknown aligner {aligner!r}")


def align(clip, parents, p):
    """Each CJK character of the text (not the transcript): its span in seconds. aligner=qwen3-fa-0.6b (the default,
    Qwen3-ForcedAligner-0.6B) or sensevoice-ctc (CTC forced alignment on SenseVoiceSmall's 60 ms frames: coarser,
    onset-to-onset spans, but in the local cache)."""
    text = cjk_only(clip.get("text", ""))
    if p["aligner"] == "qwen3-fa-0.6b" and p.get("trim_s") is None:   # version 2's own path, kept for its layers
        res = _aligner(p["device"]).align(clip["path"], text=text, language=p["language"])[0]
        items = [(it.text, round(float(it.start_time), 3), round(float(it.end_time), 3)) for it in res]
        return {"items": items, "n": len(items), "text_chars": len(text), "model": QWEN_ALIGNER}, None
    import librosa
    y, sr = load(clip)
    y16 = librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y
    out = align_audio(y16, text, p["aligner"], p["device"], p["language"], trim_s=p.get("trim_s"))
    if "error" in out:
        return out, None
    model = SENSEVOICE if p["aligner"] == "sensevoice-ctc" else QWEN_ALIGNER
    return {**{k: v for k, v in out.items() if v is not None}, "n": len(out["items"]), "text_chars": len(text),
            "model": model}, None


register(Step("align", "2", align, params={"aligner": "qwen3-fa-0.6b", "language": "Cantonese", "device": "cuda:0"},
              device="cuda"))


# ---------------------------------------------------------------------- syllables
def syllables(clip, parents, p):
    """Per CJK character: the aligner's span, the G2P's tone target, a 10-point F0 contour (Hz, NaN where
    unvoiced), its voiced share, and its mean level.

    contour=span (the default) samples the contour across the whole span. contour=voiced samples it across the
    longest voiced run inside the span (the rime, which carries the tone), interpolated, when that run is at least
    min_run_s; "voiced" is then the run's own share (1.0) so tonefit admits it, and span_voiced keeps the span's.
    Onset-to-onset spans (sensevoice-ctc) take in the next initial and any pause, which leaves only a third of
    syllables fully voiced across the span.

    span=onset keeps only the aligner's onsets and runs each span to the next onset: Qwen3-FA's end times give 7.3%
    of donor characters zero length (SenseVoice 0.8%), and a zero-length span carries no contour."""
    al, g, en = parents["align"], parents["g2p"], parents["energy"]
    if "items" not in al or "chars" not in g:
        return {"error": "a parent failed"}, None
    f0a, dba = parent_array(clip, "f0", "f0"), parent_array(clip, "energy", "db")
    hop = p["hop_s"]
    chars = g["chars"]
    out = []
    items = al["items"]
    if p.get("span") == "onset":   # the aligner's onsets only: each span runs to the next character's onset
        items = [(ch, a, items[k + 1][1] if k + 1 < len(items) else b) for k, (ch, a, b) in enumerate(items)]
    for k, (ch, a, b) in enumerate(items):
        tone = g["tones"][k] if k < len(chars) and chars[k] == ch else None
        i0, i1 = int(a / hop), max(int(a / hop) + 1, int(b / hop))
        seg = f0a[i0:i1]
        pts = []
        span_voiced = round(float(np.mean(~np.isnan(seg))), 3) if len(seg) else 0.0
        if p["contour"] == "voiced":
            v = ~np.isnan(seg)
            best, run0, cur0 = 0, 0, None
            for j, x in enumerate(np.append(v, False)):
                if x and cur0 is None:
                    cur0 = j
                elif not x and cur0 is not None:
                    if j - cur0 > best:
                        best, run0 = j - cur0, cur0
                    cur0 = None
            if best * hop >= p["min_run_s"] - 1e-9:
                r = seg[run0:run0 + best]
                pts = [round(float(x), 2) for x in np.exp(np.interp(np.linspace(0, best - 1, p["points"]),
                                                                     np.arange(best), np.log(r)))]
            lvl = dba[i0:i1]
            out.append({"char": ch, "tone": tone, "start_s": a, "end_s": b, "dur_s": round(b - a, 3),
                        "voiced": 1.0 if pts else 0.0, "span_voiced": span_voiced,
                        "run_s": round(best * hop, 3), "run_start_s": round((i0 + run0) * hop, 3),
                        "f0": pts, "db": round(float(lvl.mean()), 2) if len(lvl) else None})
            continue
        if len(seg):
            for q in np.linspace(0, len(seg) - 1, p["points"]):
                v = seg[int(round(q))]
                pts.append(None if np.isnan(v) else round(float(v), 2))
        lvl = dba[i0:i1]
        out.append({"char": ch, "tone": tone, "start_s": a, "end_s": b, "dur_s": round(b - a, 3),
                    "voiced": round(float(np.mean(~np.isnan(seg))), 3) if len(seg) else 0.0,
                    "f0": pts, "db": round(float(lvl.mean()), 2) if len(lvl) else None})
    return {"syllables": out, "n": len(out)}, None


register(Step("syllables", "1", syllables, parents=("align", "g2p", "f0", "energy"),
              params={"hop_s": 0.01, "points": 10, "contour": "span", "min_run_s": 0.05}, reads_audio=False))


# ---------------------------------------------------------------------- speaker: normalization and tone templates
def speaker(parents, p):
    """Log-F0 mean and spread over every voiced frame of the collection (the LZ normalization), and the six tones'
    mean normalized 10-point contours over syllables that are neither a clip's last character nor under-voiced."""
    syl = parents["syllables"]
    # mean and spread over the syllables' own F0 points, so both describe one population
    pts = [np.log(v) for r in syl.values() for s in r.get("syllables", []) for v in (s["f0"] or []) if v]
    mu, sd = float(np.mean(pts)), float(np.std(pts))
    by = {t: [] for t in range(1, 7)}
    for r in syl.values():
        ss = r.get("syllables", [])
        for s in ss[:-1]:
            if s["tone"] and s["voiced"] >= p["min_voiced"] and all(v for v in s["f0"]):
                by[s["tone"]].append([(np.log(v) - mu) / sd for v in s["f0"]])
    templates = {t: {"n": len(c), "mean": np.mean(c, axis=0).round(4).tolist() if c else None,
                     "sd": np.std(c, axis=0).round(4).tolist() if c else None} for t, c in by.items()}
    return {"log_f0_mean": round(mu, 5), "log_f0_sd": round(sd, 5), "hz_at_mean": round(float(np.exp(mu)), 2),
            "templates": templates, "clips": len(syl)}, None


register(Step("speaker", "lz-1", speaker, parents=("syllables",), params={"min_voiced": 0.8}, kind="collection"))


# ---------------------------------------------------------------------- tonefit
def tonefit(clip, parents, p):
    """Each syllable's normalized contour against the six templates of a named reference collection (by default the
    donor's own speech): RMS distance to each, the nearest tone, and whether it is the target."""
    import store as st
    ref = st.Store(p["templates_from"], clip["_root"]).summary("speaker", p["templates_key"])
    mu, sd = ref["log_f0_mean"], ref["log_f0_sd"]
    tm = {int(t): np.asarray(v["mean"]) for t, v in ref["templates"].items() if v["mean"]}
    out, hits, n = [], 0, 0
    for s in parents["syllables"].get("syllables", []):
        if not s["f0"] or not all(v for v in s["f0"]) or s["voiced"] < p["min_voiced"]:
            out.append(None)
            continue
        z = (np.log(np.asarray(s["f0"])) - mu) / sd
        d = {t: round(float(np.sqrt(np.mean((z - c) ** 2))), 4) for t, c in tm.items()}
        near = min(d, key=d.get)
        out.append({"target": s["tone"], "nearest": near, "dist": d})
        if s["tone"]:
            n += 1
            hits += near == s["tone"]
    last = next((x for x in reversed(out) if x is not None), None) if out else None
    return {"fits": out, "scored": n, "match_rate": round(hits / n, 4) if n else None,
            "final": last, "final_is_last_char": bool(out) and out[-1] is not None}, None


register(Step("tonefit", "1", tonefit, parents=("syllables",),
              params={"templates_from": "donor", "templates_key": None, "min_voiced": 0.8}, reads_audio=False))

# ---------------------------------------------------------------------- encode: the codec's codes, for training sets
_TOK: dict = {}


def encode(clip, parents, p):
    """The 12 Hz codec's [frames, 16] codes for the clip, one clip at a time (batch 1, so no clip's codes depend on
    its batch neighbours), with the tokenizer and device recorded. Re-encoding is not bit-identical to codes made on
    another device or batch (95-99% of elements agree, measured 2026-10-10), so a training comparison encodes every
    arm in one recipe."""
    import glob
    import os
    if "tok" not in _TOK:
        from qwen_tts import Qwen3TTSTokenizer
        src = p["tokenizer"]
        if not os.path.isdir(src):
            hits = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{src.replace('/', '--')}/snapshots/*/"))
            src = hits[0] if hits else src
        _TOK["tok"] = Qwen3TTSTokenizer.from_pretrained(src, device_map=p["device"])
        _TOK["src"] = src
    enc = _TOK["tok"].encode([clip["path"]])
    codes = np.asarray(enc.audio_codes[0].cpu().tolist(), dtype=np.int16)
    import hashlib
    return {"frames": int(codes.shape[0]), "codebooks": int(codes.shape[1]),
            "codes_sha256": hashlib.sha256(codes.tobytes()).hexdigest(), "tokenizer": _TOK["src"]}, {"codes": codes}


register(Step("encode", "1", encode, params={"tokenizer": "Qwen/Qwen3-TTS-Tokenizer-12Hz", "device": "cuda:0"},
              device="cuda"))


# ---------------------------------------------------------------------- final
def final(clip, parents, p):
    """The last syllable of the MAIN speech (before any detached tail), with no aligner: its final voiced run, F0 at
    its end relative to the clip's median (semitones), the slope over its last 30 ms (semitones per second), and its
    length. The particle profile Wu (2009) and Xu and Mok (2011) point to."""
    en, tl, fr = parents["energy"], parents["tails"], parents["f0"]
    f = parent_array(clip, "f0", "f0")
    hop = p["hop_s"]
    end = (tl["D"] or {}).get("main_end_s") or en.get("speech_end_s")
    if end is None or "median_hz" not in fr:
        return {"error": "no speech or no voicing"}, None
    i_end = min(len(f), int(end / hop) + 1)
    v = ~np.isnan(f[:i_end])
    if not v.any():
        return {"error": "no voiced frame in the main speech"}, None
    last = np.flatnonzero(v)[-1]
    first = last
    while first > 0 and v[first - 1]:
        first -= 1
    run = f[first:last + 1]
    st_run = 12 * np.log2(run / fr["median_hz"])
    k = max(2, int(p["slope_s"] / hop))
    tail = st_run[-k:]
    slope = float(np.polyfit(np.arange(len(tail)) * hop, tail, 1)[0]) if len(tail) >= 2 else None
    return {"main_end_s": round(end, 3), "run_start_s": round(first * hop, 3), "run_s": round(len(run) * hop, 3),
            "end_st": round(float(np.median(st_run[-max(1, int(0.05 / hop)):])), 3),
            "mean_st": round(float(np.mean(st_run)), 3), "slope_st_per_s": round(slope, 2) if slope is not None else None,
            "detached_tail": bool(tl["D"])}, None


register(Step("final", "1", final, parents=("energy", "tails", "f0"), params={"hop_s": 0.01, "slope_s": 0.03},
              reads_audio=False))


# ---------------------------------------------------------------------- edges: stray sounds at BOTH ends
EDGE_KINDS = ("click", "bump", "breath", "voiced", "other")


def _edge_env_db(y, sr, win_s, hop_s):
    """RMS level in dB over win_s windows every hop_s; window i covers samples [i*hop, i*hop + win)."""
    win, hop = max(1, int(round(win_s * sr))), max(1, int(round(hop_s * sr)))
    if len(y) < win:
        return np.zeros(0), win, hop
    c = np.concatenate([[0.0], np.cumsum(np.asarray(y, dtype=np.float64) ** 2)])
    starts = np.arange(0, len(y) - win + 1, hop)
    return 10 * np.log10((c[starts + win] - c[starts]) / win + 1e-20), win, hop


def _edge_periodicity(x, sr, fmin, fmax, frame_s, q=80.0):
    """The q-th percentile, over frames, of the normalized autocorrelation peak at lags 1/fmax to 1/fmin; frames more than
    10 dB under the loudest are skipped. x is already high-passed, so a low-frequency thud cannot read periodic."""
    lo, hi = int(sr / fmax), int(sr / fmin)
    fl = max(int(frame_s * sr), 2 * hi)
    if len(x) < lo + 32:
        return 0.0
    frames = [x[i:i + fl] for i in range(0, len(x) - fl + 1, max(1, fl // 4))] if len(x) >= fl else [x]
    en = np.array([float(np.sum(np.asarray(f, dtype=np.float64) ** 2)) for f in frames])
    vals = []
    for f, e in zip(frames, en):
        if e <= 0 or e < en.max() * 0.1:
            continue
        f = np.asarray(f, dtype=np.float64) - float(np.mean(f))
        L = len(f)
        top = min(hi, L - 16)
        if top < lo:
            continue
        ac = np.fft.irfft(np.abs(np.fft.rfft(f, 2 * L)) ** 2)[:L]          # sum f[n] f[n + lag]
        c = np.concatenate([[0.0], np.cumsum(f ** 2)])
        lags = np.arange(lo, top + 1)
        den = np.sqrt(c[L - lags] * (c[L] - c[lags]))                       # |f[:-lag]| |f[lag:]|
        r = np.where(den > 0, ac[lags] / np.maximum(den, 1e-30), 0.0)
        vals.append(float(r.max()))
    return float(np.percentile(vals, q)) if vals else 0.0


def _edge_kind(ev, p):
    """The v1 kind rules, first match wins (listed in edges())."""
    d, low, per = ev["dur_s"], ev["low_band_share"], ev["periodicity"]
    if d <= p["click_max_s"] and ev["onset_ms"] <= p["click_onset_ms"] and low < p["click_max_low"]:
        return "click"
    if d <= p["bump_max_s"] and low >= p["bump_min_low"]:
        return "bump"
    if d >= p["voiced_min_s"] and per >= p["voiced_min_per"]:
        return "voiced"
    if (p["breath_min_s"] <= d <= p["breath_max_s"] and low < p["breath_max_low"]
            and ev["spectral_centroid_hz"] >= p["breath_min_centroid_hz"]):
        return "breath"
    return "other"


def _edge_levels(x, sr, sp5_db, sp_energy):
    """Graded levels of a region, whether or not anything crosses a threshold: its loudest 5 ms RMS frame (2.5 ms hop)
    re the speech's loudest, and its summed energy re the speech's summed energy, in dB."""
    if len(x) < int(0.005 * sr):
        return None, None
    e5, _, _ = _edge_env_db(x, sr, 0.005, 0.0025)
    tot = float(np.sum(np.asarray(x, dtype=np.float64) ** 2))
    return round(float(e5.max()) - sp5_db, 2), round(10 * np.log10(tot / sp_energy + 1e-30), 2)


def _edge_region(y_dc, y_hp, sr, r0, r1, speech_side, speech_peak, env_peak_db, p, sp5_db=0.0, sp_energy=1.0):
    """The events in one region [r0, r1) samples; speech_side is "after" for the lead, "before" for the trail."""
    import numpy.fft as nf
    out = {"floor_db": None, "attached_s": 0.0, "events": [],
           "level": {"peak_db_rel_speech": None, "energy_db_rel_speech": None, "floor_db": None, "max_event_db": None,
                     "region_s": round(max(0, r1 - r0) / sr, 3)}}
    if r1 - r0 < int(p["min_region_s"] * sr):
        return out
    env, win, hop = _edge_env_db(y_dc[r0:r1], sr, p["win_s"], p["hop_s"])
    if not len(env):
        return out
    env = env - env_peak_db                                          # dB re the core speech's own envelope peak
    floor = max(float(np.percentile(env, p["floor_pct"])), p["floor_clamp_db"])
    thr = max(floor + p["rise_db"], p["min_rel_db"])
    act = (env > thr) | (env > p["loud_rel_db"])
    out["floor_db"] = round(floor, 2)
    runs, s = [], None
    for i, a in enumerate(act):
        if a and s is None:
            s = i
        if not a and s is not None:
            runs.append([s, i])
            s = None
    if s is not None:
        runs.append([s, len(act)])
    merged = []
    for a, b in runs:
        if merged and (a - merged[-1][1]) * hop / sr < p["merge_s"]:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    # a stretch reaching the speech side of the region is the speech's own onset or decay, not a stray sound
    k0, k1 = r0, r1                                                  # the region less the attached stretch
    if merged and speech_side == "after" and merged[-1][1] >= len(act):
        out["attached_s"] = round((merged[-1][1] - merged[-1][0]) * hop / sr, 3)
        k1 = r0 + merged[-1][0] * hop
        merged = merged[:-1]
    if merged and speech_side == "before" and merged[0][0] <= 0:
        out["attached_s"] = round((merged[0][1] - merged[0][0]) * hop / sr, 3)
        k0 = min(r1, r0 + (merged[0][1] - 1) * hop + win)
        merged = merged[1:]
    pk5, en_ = _edge_levels(y_dc[k0:k1], sr, sp5_db, sp_energy)
    out["level"].update(peak_db_rel_speech=pk5, energy_db_rel_speech=en_, floor_db=round(floor, 2))
    for a, b in merged:
        s0, s1 = r0 + a * hop, min(r1, r0 + (b - 1) * hop + win)
        dur = (s1 - s0) / sr
        if dur < p["min_event_s"]:
            continue
        seg = np.asarray(y_dc[s0:s1], dtype=np.float64)
        pk = float(np.abs(seg).max())
        rms = float(np.sqrt(np.mean(seg ** 2))) + 1e-12
        w = np.hanning(len(seg)) if dur > 0.02 else np.ones(len(seg))
        nfft = 1 << int(np.ceil(np.log2(max(len(seg), 4096))))
        P = np.abs(nf.rfft(seg * w, n=nfft)) ** 2
        f = nf.rfftfreq(nfft, 1 / sr)
        band = f >= 20
        tot = float(P[band].sum()) + 1e-30
        e_ev = env[a:b]
        ev = {"start_s": round(s0 / sr, 4), "dur_s": round(dur, 4),
              "peak_db_rel_speech": round(20 * np.log10(pk / speech_peak + 1e-12), 2),
              "env_peak_db_rel_speech": round(float(e_ev.max()), 2),
              "rise_over_floor_db": round(float(e_ev.max()) - floor, 2),
              "crest_db": round(20 * np.log10(pk / rms), 2),
              "spectral_centroid_hz": round(float((f[band] * P[band]).sum()) / tot, 1),
              "low_band_share": round(float(P[band & (f < p["low_band_hz"])].sum()) / tot, 4),
              "periodicity": round(_edge_periodicity(np.asarray(y_hp[s0:s1], dtype=np.float64), sr, p["per_fmin"],
                                                     p["per_fmax"], p["per_frame_s"], p["per_q"]), 3),
              "onset_ms": round(float(np.argmax(e_ev)) * hop / sr * 1000, 1)}
        ev["kind"] = _edge_kind(ev, p)
        out["events"].append(ev)
    if out["events"]:
        out["level"]["max_event_db"] = max(e["env_peak_db_rel_speech"] for e in out["events"])
    return out


def edge_core(segs, db, p):
    """The main speech from the segments. A segment QUALIFIES when it lasts at least core_min_s and its 10 ms peak is
    within core_db of the loudest qualifying segment.

    v1 (core_gap_s None) took the first to the last qualifying segment, so a loud edge sound of 0.1 s or more was
    absorbed into the speech and never reported (review of PR #165, 2026-10-17).

    v2 groups the qualifying segments into runs whose gaps are under core_gap_s (D's 0.4 s pause rule). The core is
    the run holding the loudest segment, extended to any other run spanning at least core_phrase_s (a phrase after a
    pause), first to last. So a loud sound shorter than core_phrase_s, separated from the speech by core_gap_s or
    more, stays at the edge where it is measured. None when there are no segments."""
    if not segs:
        return None
    pk = [float(db[int(a / 0.01):max(int(a / 0.01) + 1, int(b / 0.01))].max()) for a, b in segs]
    long_ = [i for i, (a, b) in enumerate(segs) if b - a >= p["core_min_s"]] or [int(np.argmax(pk))]
    ref = max(pk[i] for i in long_)
    q = [i for i in long_ if pk[i] >= ref + p["core_db"]]
    if p.get("core_gap_s") is None:
        return segs[q[0]][0], segs[q[-1]][1]
    runs = [[q[0]]]
    for i in q[1:]:
        if segs[i][0] - segs[runs[-1][-1]][1] < p["core_gap_s"]:
            runs[-1].append(i)
        else:
            runs.append([i])
    top = max(q, key=lambda i: pk[i])
    keep = [r for r in runs if top in r or segs[r[-1]][1] - segs[r[0]][0] >= p["core_phrase_s"]]
    return segs[keep[0][0]][0], segs[keep[-1][-1]][1]


def edges_of(y, sr, segs, p):
    """edges() on an array: the record, without the store. segs are energy's segments, in seconds."""
    import scipy.signal as ss
    base = {"speech": False, "n_lead": 0, "n_trail": 0, "lead": [], "trail": [],
            "by_kind": {e: {k: 0 for k in EDGE_KINDS} for e in ("lead", "trail")}}
    db = frames_db(y, sr, 0.01)
    if not segs:
        return base
    # energy's segments merge gaps under 0.15 s, which swallows a sound that close to the speech: re-split them by
    # energy's own rule (10 ms frames, -40 dB under the loudest, 40 ms minimum) with gaps under core_merge_s merged
    fine = segments(db, 0.01, p["seg_floor_db"], p["core_merge_s"], 0.04)
    core = edge_core(fine, db, p)
    if core is None:
        return base
    c0, c1 = core
    y_dc = ss.sosfiltfilt(ss.butter(2, 20, "highpass", fs=sr, output="sos"), y)
    y_hp = ss.sosfiltfilt(ss.butter(4, p["per_hp_hz"], "highpass", fs=sr, output="sos"), y)
    i0, i1 = int(c0 * sr), max(int(c0 * sr) + 1, int(c1 * sr))
    speech_peak = float(np.abs(y_dc[i0:i1]).max()) + 1e-12
    env_sp, _, _ = _edge_env_db(y_dc[i0:i1], sr, p["win_s"], p["hop_s"])
    env_peak = float(env_sp.max()) if len(env_sp) else 20 * np.log10(speech_peak)
    sp5, _, _ = _edge_env_db(y_dc[i0:i1], sr, 0.005, 0.0025)
    sp5_db = float(sp5.max()) if len(sp5) else env_peak
    sp_energy = float(np.sum(np.asarray(y_dc[i0:i1], dtype=np.float64) ** 2)) + 1e-30
    g = int(p["guard_s"] * sr)
    lead = _edge_region(y_dc, y_hp, sr, 0, max(0, i0 - g), "after", speech_peak, env_peak, p, sp5_db, sp_energy)
    trail = _edge_region(y_dc, y_hp, sr, min(len(y), i1 + g), len(y), "before", speech_peak, env_peak, p, sp5_db,
                         sp_energy)
    rec = {"speech": True, "core_start_s": round(c0, 3), "core_end_s": round(c1, 3),
           "energy_span_s": [segs[0][0], segs[-1][1]], "duration_s": round(len(y) / sr, 3),
           "lead_floor_db": lead["floor_db"], "trail_floor_db": trail["floor_db"],
           "lead_level": lead["level"], "trail_level": trail["level"],
           "lead_attached_s": lead["attached_s"], "trail_attached_s": trail["attached_s"],
           "lead": [dict(e, end="lead") for e in lead["events"]],
           "trail": [dict(e, end="trail") for e in trail["events"]]}
    rec["n_lead"], rec["n_trail"] = len(rec["lead"]), len(rec["trail"])
    rec["by_kind"] = {e: {k: sum(x["kind"] == k for x in rec[e]) for k in EDGE_KINDS} for e in ("lead", "trail")}
    return rec


def edges(clip, parents, p):
    """Stray sounds at BOTH ends of a clip, short clicks included, each labelled by kind. v2, rule-based.

    WHERE IT LOOKS. The main speech comes from energy's segments, not its raw span, because that span starts at
    whatever first crosses -40 dB, a stray sound included, and its 0.15 s merge swallows a sound that close to the
    speech. So the segments are re-split by energy's rule with gaps under core_merge_s merged, and the CORE is chosen
    by edge_core: the qualifying segments (at least core_min_s long, peak within core_db of the loudest) grouped by
    gaps under core_gap_s; the group holding the loudest segment, plus any group spanning core_phrase_s. A loud
    sound shorter than that, a pause of core_gap_s or more away, is an edge sound, not speech (v1 absorbed it). The
    LEAD is the file start to core start minus guard_s; the TRAIL is core end plus guard_s to the file end.

    HOW. A 4 ms RMS envelope every 2 ms (a 4 ms click spans two to three hops), DC and rumble below 20 Hz removed, in
    dB re the core speech's own envelope peak. Each region's floor is its floor_pct percentile, clamped at
    floor_clamp_db. A hop is active above max(floor + rise_db, min_rel_db), or above loud_rel_db whatever the floor.
    Active runs closer than merge_s merge into one event. A run touching the region's speech side is the speech's own
    onset or decay: recorded as lead_attached_s / trail_attached_s and not counted as an event.

    GRADED LEVELS, per end, whether or not any event crosses the threshold (owner, 2026-10-17: "Boolean is too
    harsh"): lead_level / trail_level hold peak_db_rel_speech (the region's loudest 5 ms RMS frame re the speech's
    loudest), energy_db_rel_speech (the region's summed energy re the speech's; it grows with the region's length),
    floor_db (as above) and max_event_db (the loudest event's envelope peak, None without one). The region here
    excludes the attached stretch, so the speech's own onset and decay do not set it.

    KINDS, first match wins:
      click   dur_s <= click_max_s, envelope peak within click_onset_ms of its start, low_band_share < click_max_low
      bump    dur_s <= bump_max_s and low_band_share (energy under low_band_hz) >= bump_min_low: a low thud
      voiced  dur_s >= voiced_min_s and periodicity >= voiced_min_per: the per_q-th percentile over 40 ms frames of
              the normalized autocorrelation peak at 67-400 Hz lags, on the per_hp_hz high-passed event so a thud's
              low frequency cannot read as pitch (a percentile, not the median: a speech fragment is voiced in part)
      breath  breath_min_s <= dur_s <= breath_max_s, low_band_share < breath_max_low, centroid >=
              breath_min_centroid_hz, and not periodic: noise-like
      other   anything else: a long noise, a slow rumble, a soft-onset knock, a click longer than click_max_s

    Proven on planted sounds by edges_instrument_test() (`cli.py edges-test`) before any layer was read."""
    y, sr = load(clip)
    return edges_of(y, sr, parents["energy"].get("segments"), p), None


EDGES_PARAMS = {
    "seg_floor_db": -40.0, "core_merge_s": 0.05, "core_min_s": 0.1, "core_db": -18.0, "guard_s": 0.03,
    "core_gap_s": 0.4, "core_phrase_s": 0.5,
    "win_s": 0.004, "hop_s": 0.002, "floor_pct": 20.0, "floor_clamp_db": -100.0,
    "rise_db": 10.0, "min_rel_db": -45.0, "loud_rel_db": -20.0, "merge_s": 0.01, "min_event_s": 0.002,
    "min_region_s": 0.05,
    "low_band_hz": 300.0, "per_hp_hz": 300.0, "per_fmin": 67.0, "per_fmax": 400.0, "per_frame_s": 0.04, "per_q": 80.0,
    "click_max_s": 0.03, "click_onset_ms": 6.0, "click_max_low": 0.5,
    "bump_max_s": 0.4, "bump_min_low": 0.9,
    "voiced_min_s": 0.05, "voiced_min_per": 0.45,
    "breath_min_s": 0.12, "breath_max_s": 1.0, "breath_max_low": 0.5, "breath_min_centroid_hz": 500.0,
}
# v2 (2026-10-17): the core no longer absorbs a loud edge sound (edge_core), and a bump may last 0.4 s. v1's layers
# stay on disk beside v2's; v1 is reproduced by core_gap_s=None, bump_max_s=0.3
register(Step("edges", "2", edges, parents=("energy",), params=EDGES_PARAMS))

_EDGES_PARENT_OK: dict = {}


def _edges_parent_check(clip):
    """In a store run, the parent edges layer must be v2 at the parameters repair_edges re-measures with."""
    if "_parent_keys" not in clip:
        return
    k = clip["_parent_keys"]["edges"]
    if k not in _EDGES_PARENT_OK:
        import store as st
        m = st.Store(clip["_collection"], clip["_root"]).layer_meta("edges", k)
        if m.get("version") != "2" or m.get("params") != EDGES_PARAMS:
            raise RuntimeError(f"edges parent {k} is v{m.get('version')} at other params; repair_edges needs v2's")
        _EDGES_PARENT_OK[k] = True


# ---------------------------------------------------------------------- trim: both ends CUT to the speech
TRIM_PARAMS = {"lead_s": 0.10, "tail_s": 0.30}


def trim(clip, parents, p):
    """Both ends cut to the speech (owner, 2026-10-19: "the Sidon output should be trimmed on both end as well"):
    the clip from edges v2's core start, less its attached onset and lead_s, to its core end, plus its attached decay
    and tail_s. Everything outside -- silence, stray sounds and breaths alike -- is removed, unlike repair_edges, which
    replaces sounds and keeps the duration. tail_s is wider than lead_s on purpose: the model learns where to stop
    from what follows the last word, and v0.0.1's raw clips end a median 0.19 s after it (EXP-083 diagnosis).

    A clip the cut would not shorten is NOT rewritten (its row points at the source file). A cut clip is a new FLAC
    in this layer's audio/ at the source's rate and subtype, sample-identical to the source over the kept span."""
    import hashlib
    import soundfile as sf
    rec0 = parents["edges"]
    base = {"repaired": False, "path": clip["path"], "sha256": clip["sha256"]}
    if not rec0.get("speech"):
        return {**base, "why": "no speech"}, None
    _edges_parent_check(clip)
    info_f = sf.info(clip["path"])
    sr, n = info_f.samplerate, info_f.frames
    a = max(0, int(np.floor((rec0["core_start_s"] - rec0.get("lead_attached_s", 0.0) - p["lead_s"]) * sr)))
    b = min(n, int(np.ceil((rec0["core_end_s"] + rec0.get("trail_attached_s", 0.0) + p["tail_s"]) * sr)))
    if b <= a:
        raise RuntimeError(f"empty cut {a}:{b} of {n}")
    out_rec = {"lead_cut_s": round(a / sr, 4), "tail_cut_s": round((n - b) / sr, 4), "raw_s": round(n / sr, 4),
               "kept_s": round((b - a) / sr, 4), "start_sample": a, "end_sample": b}
    if a == 0 and b == n:
        return {**base, **out_rec, "why": "nothing to cut"}, None
    y2d, _ = sf.read(clip["path"], dtype="float32", always_2d=True)
    out = Path(clip["_layer_dir"]) / "audio" / (Path(clip["path"]).stem + ".flac")
    out.parent.mkdir(parents=True, exist_ok=True)
    sub = info_f.subtype if info_f.subtype in ("PCM_16", "PCM_24") else "PCM_16"
    kept = y2d[a:b]
    sf.write(out, kept if kept.shape[1] > 1 else kept[:, 0], sr, format="FLAC", subtype=sub)
    back, sr2 = sf.read(out, dtype="float32", always_2d=True)
    if sr2 != sr or not np.array_equal(back, kept):
        out.unlink()
        raise RuntimeError("the kept span is not sample-identical after the write")
    return {"repaired": True, "path": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "sr": sr,
            "source_subtype": info_f.subtype, **out_rec}, None


register(Step("trim", "1", trim, parents=("edges",), params=TRIM_PARAMS))


def pad(clip, parents, p):
    """The trimmed clip with the cut restored as DIGITAL SILENCE, back to the source's length (EXP-095's arm B, made a
    step 2026-10-20): trimming removed Sidon's noisy edges, which caused EXP-083's runaways, and with them every edge
    silence, which caused EXP-091's skipped text; zeros put the silence back without the noise. A clip trim left
    alone is not rewritten. A padded clip is a new FLAC in this layer's audio/, exactly the source's length, equal to
    the trimmed file over the kept span and zero outside it."""
    import hashlib
    import soundfile as sf
    t = parents["trim"]
    if not t.get("repaired"):
        return {"repaired": False, "path": t.get("path", clip["path"]), "sha256": t.get("sha256", clip["sha256"]),
                "why": t.get("why", "trim left it")}, None
    y2d, sr = sf.read(t["path"], dtype="float32", always_2d=True)
    a, b, n = t["start_sample"], t["end_sample"], sf.info(clip["path"]).frames     # the source this layer's trim cut
    if b - a != len(y2d):
        raise RuntimeError(f"trim record spans {b - a} samples, its file holds {len(y2d)}")
    z = np.zeros((n, y2d.shape[1]), np.float32)
    z[a:b] = y2d
    out = Path(clip["_layer_dir"]) / "audio" / (Path(clip["path"]).stem + ".flac")
    out.parent.mkdir(parents=True, exist_ok=True)
    sub = t.get("source_subtype") if t.get("source_subtype") in ("PCM_16", "PCM_24") else "PCM_16"
    sf.write(out, z if z.shape[1] > 1 else z[:, 0], sr, format="FLAC", subtype=sub)
    back, _ = sf.read(out, dtype="float32", always_2d=True)
    if len(back) != n or not np.array_equal(back[a:b], y2d) or np.any(back[:a]) or np.any(back[b:]):
        out.unlink()
        raise RuntimeError("the padded clip is not the trimmed span between zeros")
    return {"repaired": True, "path": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "sr": sr,
            "lead_pad_s": round(a / sr, 4), "tail_pad_s": round((n - b) / sr, 4), "frames": n}, None


register(Step("pad", "1", pad, parents=("trim",)))


# ---------------------------------------------------------------------- sidon: speech restoration (EXP-083)
class Sidon:
    """Sidon v0.1 as its authors' demo runs it (sarulab-speech/sidon_demo_beta app.py). `models` holds the two
    TorchScript files from sarulab-speech/sidon-v0.1 (feature_extractor_cuda.pt, decoder_cuda.pt) and, in
    w2v-bert-2.0-fe/, facebook/w2v-bert-2.0's preprocessor_config.json. Copied from the lab's build_sidon_page.py."""

    def __init__(self, models: Path):
        import torch
        import transformers
        self.torch = torch
        self.fe = torch.jit.load(str(models / "feature_extractor_cuda.pt"), map_location="cuda").to("cuda")
        self.dec = torch.jit.load(str(models / "decoder_cuda.pt"), map_location="cuda").to("cuda")
        self.pre = transformers.SeamlessM4TFeatureExtractor.from_pretrained(str(models / "w2v-bert-2.0-fe"))

    def __call__(self, y: np.ndarray, sr: int) -> np.ndarray:
        """The authors' demo (sarulab-speech/sidon_demo_beta app.py), returning 48 kHz float32."""
        torch = self.torch
        import torchaudio
        w = torch.tensor(0.9 * y / np.abs(y).max(), dtype=torch.float32).view(1, -1)
        n48 = int(48_000 / sr * w.shape[1])
        w = torchaudio.functional.highpass_biquad(w, sr, 50)
        w16 = torch.nn.functional.pad(torchaudio.functional.resample(w, sr, 16_000), (0, 24000))
        outs, cache = [], None
        with torch.inference_mode():
            for chunk in w16.view(-1).split(16000 * 96):
                inp = self.pre(torch.nn.functional.pad(chunk, (160, 160)), return_tensors="pt", sampling_rate=16_000)
                f = self.fe(inp["input_features"].to("cuda"))["last_hidden_state"]
                if cache is not None:
                    f = torch.cat([cache, f], dim=1)
                outs.append(self.dec(f.transpose(1, 2)).view(-1)[:-960])
                cache = f[:, -1:]
        return torch.cat(outs).float().cpu().numpy()[:n48]


def loud_half_rms(y: np.ndarray, sr: int) -> float:
    """RMS over the loudest half of the clip's 10 ms frames: the level Sidon's output is matched to."""
    k = int(0.01 * sr)
    fr = y[: len(y) // k * k].reshape(-1, k)
    r = np.sqrt((fr ** 2).mean(1))
    return float(np.sqrt((r[r >= np.median(r)] ** 2).mean()))


def sidon(clip, parents, p):
    """The clip restored by Sidon (sarulab-speech/sidon-v0.1, the authors' demo inference: 50 Hz high-pass, 16 kHz in,
    48 kHz out), resampled to the clip's own rate and level-matched to it on the loudest half of its 10 ms frames, as
    BM-SIDON-1 was heard (22 of 24 preferred, blind, no marks). Every clip is rewritten: a NEW FLAC (PCM_16) in this
    layer's audio/; the raw file is never touched. Recorded per clip: the 10 ms energy-envelope correlation with the
    raw clip, and the length check, so a restoration that wandered can be found and excluded later."""
    import hashlib
    import librosa
    import soundfile as sf
    if "sidon" not in _MODELS:
        _MODELS["sidon"] = Sidon(Path(p["models"]).expanduser())
        _MODELS["sidon_loud"] = loud_half_rms
    y, sr = sf.read(clip["path"], dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    z = librosa.resample(_MODELS["sidon"](y, sr), orig_sr=48_000, target_sr=sr)
    z = np.pad(z, (0, max(0, len(y) - len(z))))[:len(y)].astype(np.float32)
    lr, lz = _MODELS["sidon_loud"](y, sr), _MODELS["sidon_loud"](z, sr)
    z = z * (lr / max(lz, 1e-9))
    if np.abs(z).max() > 0.999:
        z = z * (0.999 / np.abs(z).max())
    out = Path(clip["_layer_dir"]) / "audio" / (Path(clip["path"]).stem + ".flac")
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, z, sr, format="FLAC", subtype="PCM_16")
    k = int(0.01 * sr)
    env = lambda v: 20 * np.log10(np.sqrt((v[: len(v) // k * k].reshape(-1, k) ** 2).mean(1)) + 1e-6)
    er, ez = env(y), env(z)
    return {"repaired": True, "path": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "sr": sr,
            "frames": len(z), "same_length": len(z) == len(y), "env_corr": round(float(np.corrcoef(er, ez)[0, 1]), 4),
            "gain_db": round(float(20 * np.log10(lr / max(lz, 1e-9))), 2)}, None


register(Step("sidon", "1", sidon, params={"model": "sarulab-speech/sidon-v0.1", "models": "~/models/sidon-v0.1",
                                           "device": "cuda:0"}, device="cuda"))


# ---------------------------------------------------------------------- dynamics: Sidon's cleaning, the raw clip's emphasis
DYNAMICS_PARAMS = {"frame_s": 0.01, "smooth_s": 0.05, "speech_floor_db": 30.0, "max_gain_db": 8.0}


def _db_env(v, k):
    n = len(v) // k * k
    return 20 * np.log10(np.sqrt((v[:n].reshape(-1, k) ** 2).mean(1)) + 1e-6)


def _smooth(x, w):
    if w <= 1:
        return x
    pad = np.pad(x, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(pad, np.ones(w) / w, mode="valid")


def dynamics_gain(y, z, sr, p):
    """The gain curve (dB per sample) that gives z (a restoration, sample-aligned with y) y's loudness contour:
    both 10 ms envelopes smoothed to smooth_s, their difference taken only where BOTH are within speech_floor_db of
    their own loudest-10% level, capped at +-max_gain_db, centred so z's level over its loudest half is unchanged,
    smoothed again so the gain never steps, and then zeroed wherever z's unsmoothed level is below that floor, so z's
    clean silence stays clean under a noisy y (the control the first two drafts failed: the smoothing leaked up to
    2.7 dB into the frames beside the speech)."""
    k = int(round(p["frame_s"] * sr))
    w = max(1, int(round(p["smooth_s"] / p["frame_s"])))
    ez0 = _db_env(z, k)
    ey, ez = _smooth(_db_env(y, k), w), _smooth(ez0, w)
    speech = (ey > np.percentile(ey, 90) - p["speech_floor_db"]) & (ez > np.percentile(ez, 90) - p["speech_floor_db"])
    g = np.where(speech, np.clip(ey - ez, -p["max_gain_db"], p["max_gain_db"]), 0.0)
    loud = ez >= np.median(ez[speech]) if speech.any() else np.zeros_like(speech)
    if loud.any():
        g = np.where(speech, g - np.median(g[loud & speech]), 0.0)    # keep z's level, change only its contour
    live = ez0 > np.percentile(ez0, 90) - p["speech_floor_db"]
    g = np.where(live, _smooth(g, w), 0.0)                  # the smoothing never reaches z's silence
    t = (np.arange(len(g)) + 0.5) * k
    per_sample = np.interp(np.arange(len(z)), t, g)
    mask = np.zeros(len(z), bool)
    mask[:len(live) * k] = np.repeat(live, k)               # nor does the interpolation between frames
    return np.where(mask, per_sample, 0.0), {"speech_frames": int(speech.sum()), "frames": len(g)}


def dynamics(clip, parents, p):
    """The Sidon output with the raw clip's loudness contour put back (2026-10-20). Measured that day: Sidon's
    restored speaker clips have a loudness range 3.4 dB narrower than the raw ones (loud end -0.5 dB, soft end
    +2.9 dB, relative to each clip's median), and every model trained on them loses its loud peaks (-0.6 to -1.0 dB)
    and its pitch peaks. This keeps Sidon's spectrum and its clean silences and re-applies the raw clip's envelope
    inside the speech (dynamics_gain). Needs a row carrying raw_path, sample-aligned with the clip (sidon keeps
    lengths); refuses otherwise. A NEW FLAC (PCM_16) in this layer's audio/; neither input is touched."""
    import hashlib
    import soundfile as sf
    if not clip.get("raw_path"):
        raise RuntimeError("no raw_path: dynamics needs the raw clip the restoration was made from")
    z, sr = sf.read(clip["path"], dtype="float32", always_2d=True)
    y, sr_y = sf.read(clip["raw_path"], dtype="float32", always_2d=True)
    z, y = z.mean(axis=1), y.mean(axis=1)
    if sr_y != sr or len(y) != len(z):
        raise RuntimeError(f"not sample-aligned: raw {len(y)} at {sr_y}, restored {len(z)} at {sr}")
    g, info = dynamics_gain(y, z, sr, p)
    out_y = z * (10 ** (g / 20)).astype(np.float32)
    if np.abs(out_y).max() > 0.999:
        out_y = out_y * (0.999 / np.abs(out_y).max())
    out = Path(clip["_layer_dir"]) / "audio" / (Path(clip["path"]).stem + ".flac")
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, out_y, sr, format="FLAC", subtype="PCM_16")
    return {"repaired": True, "path": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "sr": sr,
            "frames": len(out_y), "gain_db_p5": round(float(np.percentile(g, 5)), 2),
            "gain_db_p95": round(float(np.percentile(g, 95)), 2), **info}, None


register(Step("dynamics", "1", dynamics, params=DYNAMICS_PARAMS))
