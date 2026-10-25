# SPDX-License-Identifier: Apache-2.0
"""The baked voices of a speaker phase: which voices, which rows, and what goes in them.

ONE CODE PATH FOR ONE VOICE AND FOR SEVERAL (2026-10-17). sft_12hz_lora_mlflow.py used
to keep the voice as two module globals and write it inline at codec_embedding row 3000,
twice, at the two export sites. Several voices per checkpoint lived in a separate fork,
sft_12hz_lora_mlflow_multi_speaker.py, which drifted from the main trainer twice in one
evening (a hardcoded sub-talker weight of 0.3, then each voice taken from ONE clip
instead of the house mean: commits c2a15ad and 49b2908) and, read in full on 2026-10-17,
also lacked every loss fix the main trainer carries. The fix for a fork that drifts is
not a better fork. Everything a voice needs now lives here and the main trainer calls it
for every phase: a single voice is a plan with one entry at row 3000, several voices are
a plan with one entry per `speaker_id`.

THE SINGLE-VOICE PATH IS BYTE-IDENTICAL TO WHAT IT WAS. `VoiceMeans.add` performs the
exact tensor operations the inline code did, in the same order, and `bake_config` /
`bake_rows` write the same keys and the same row. test_voice_bank.py holds the old inline
code verbatim as the reference and requires equality, not closeness.

Needs torch only, so it is testable with no model and no card.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import torch

START_SLOT = 3000

# A NEUTRAL NAME, NEVER A PERSON'S. AGENTS.md forbids attaching a donor's real-world name
# to a voice record. No pattern can tell a pseudonym from a name, so this cannot enforce
# that rule; it enforces the shape the project's pseudonyms already have (hk_voice_343,
# cv_yue_v22, hk_cantonese_speaker) and refuses the shape a real name usually arrives in,
# with a space. The rule itself is the caller's to keep.
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class VoicePlanError(ValueError):
    """A voice plan that would bake something other than what the caller asked for."""


@dataclass
class VoicePlan:
    """Which voices this phase bakes, at which rows, and which data rows feed each.

    `slots` is voice name -> codec_embedding row, in row order. `voice_of` is
    speaker_id -> voice name. `rows_in_data` is speaker_id -> how many training rows
    carry it. A single-voice plan has one slot and no speaker ids: every row is that
    voice, whatever its `speaker_id` says, which is what the trainer has always done.
    """
    slots: dict
    voice_of: dict = field(default_factory=dict)
    rows_in_data: dict = field(default_factory=dict)
    order_rule: str = "single voice"

    @property
    def names(self) -> list:
        return list(self.slots)

    @property
    def multi(self) -> bool:
        return bool(self.voice_of)

    def voice_indices(self, rows) -> list:
        """Index into `names` for each training row, in row order. Refuses a row whose
        speaker_id this plan does not hold, so the data loaded for training cannot
        differ from the data the plan was made from."""
        names = self.names
        pos = {n: i for i, n in enumerate(names)}
        out = []
        for i, r in enumerate(rows):
            sid = r.get("speaker_id") if isinstance(r, dict) else None
            if sid is None or str(sid) not in self.voice_of:
                raise VoicePlanError(f"row {i} has speaker_id {sid!r}, which the voice plan "
                                     f"does not hold ({sorted(self.voice_of)})")
            out.append(pos[self.voice_of[str(sid)]])
        return out

    def record(self, accumulated: dict | None = None) -> dict:
        rec = {"start_slot": min(self.slots.values()),
               "slots": dict(self.slots),
               "order": self.order_rule,
               "speaker_id_to_voice": dict(self.voice_of),
               "rows_in_data": dict(self.rows_in_data),
               "embedding": "running weighted mean of speaker_encoder outputs over every "
                            "training row of that voice, fp32 (VoiceMeans.add)"}
        if accumulated is not None:
            rec["clips_accumulated"] = dict(accumulated)
        return rec


def single_plan(speaker_name: str, slot: int = START_SLOT) -> VoicePlan:
    """The plan every speaker phase in this project has used: one voice, row 3000."""
    return VoicePlan(slots={speaker_name: slot})


def plan_from_rows(speaker_ids, speaker_map: dict | None = None,
                   start_slot: int = START_SLOT) -> VoicePlan:
    """One voice per distinct speaker_id, at rows start_slot, start_slot+1, ...

    ROW ORDER IS SORTED speaker_id, so an export does not change with the manifest's line
    order and the same data always gives the same rows. With a map, each speaker_id's
    voice is the map's name for it; without one, the speaker_id is the name.

    Refused, every problem named rather than only the first:
      * a row with no speaker_id;
      * a map naming a speaker_id that is absent from the data (a voice the caller asked
        for that would not be baked);
      * a speaker_id in the data that the map does not name (rows that would train with
        no voice to bake them into);
      * two speaker_ids mapped to one voice name (a blend wearing a multi-voice config:
        spk_id cannot hold one name twice, so one of them would be silently lost);
      * a voice name that is not a neutral identifier (NAME_RE).
    """
    problems = []
    counts: dict = {}
    for i, sid in enumerate(speaker_ids):
        if sid is None or str(sid) == "":
            problems.append(f"row {i} has no speaker_id")
            continue
        counts[str(sid)] = counts.get(str(sid), 0) + 1
    if not counts and not problems:
        problems.append("no training rows")
    ids = sorted(counts)
    if speaker_map is not None:
        if not isinstance(speaker_map, dict) or not speaker_map:
            raise VoicePlanError("the speaker map must be a non-empty JSON object of "
                                 "speaker_id -> voice name")
        smap = {str(k): v for k, v in speaker_map.items()}
        absent = sorted(set(smap) - set(counts))
        unnamed = sorted(set(counts) - set(smap))
        if absent:
            problems.append(f"the map names speaker_id(s) absent from the data: {absent}")
        if unnamed:
            problems.append(f"the data carries speaker_id(s) the map does not name: {unnamed}")
        voice_of = {sid: smap[sid] for sid in ids if sid in smap}
    else:
        voice_of = {sid: sid for sid in ids}
    for sid, name in voice_of.items():
        if not isinstance(name, str) or not NAME_RE.match(name):
            problems.append(f"voice name {name!r} (speaker_id {sid!r}) is not a neutral "
                            f"identifier matching {NAME_RE.pattern}")
    seen: dict = {}
    for sid, name in voice_of.items():
        seen.setdefault(name, []).append(sid)
    for name, sids in seen.items():
        if len(sids) > 1:
            problems.append(f"speaker_ids {sids} all map to voice {name!r}; one voice name "
                            f"per speaker_id")
    if problems:
        raise VoicePlanError("refusing the voice plan:\n  - " + "\n  - ".join(problems))
    slots = {voice_of[sid]: start_slot + i for i, sid in enumerate(ids)}
    return VoicePlan(slots=slots, voice_of=voice_of, rows_in_data={s: counts[s] for s in ids},
                     order_rule="rows in sorted speaker_id order from start_slot")


def plan_from_jsonl(path: str, speaker_map_path: str | None = None,
                    start_slot: int = START_SLOT) -> VoicePlan:
    """plan_from_rows over a training manifest, read before the model is loaded so a bad
    plan costs a second rather than a model load."""
    speaker_map = None
    if speaker_map_path:
        with open(speaker_map_path, encoding="utf-8") as fh:
            speaker_map = json.load(fh)
    ids = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                ids.append(json.loads(line).get("speaker_id"))
    return plan_from_rows(ids, speaker_map, start_slot)


class VoiceMeans:
    """A running weighted mean of speaker embeddings, one per voice.

    THE ARITHMETIC IS THE SINGLE-VOICE TRAINER'S, unchanged and now shared. It used to be
    inline in the training loop (and still is described there). A first batch sets the
    mean; each later batch of n rows moves it to mean * (count / total) + sum / total.
    Accumulated in fp32, because a bf16 running mean stops moving once the count passes
    ~256 and would silently reduce to "the first few clips".
    """

    def __init__(self, names):
        self.names = list(names)
        self.mean: dict = {}
        self.count: dict = {n: 0 for n in self.names}

    def add(self, name: str, emb: torch.Tensor) -> None:
        """Fold the rows of `emb` (n, d), all of voice `name`, into its mean."""
        _n = emb.size(0)
        if self.mean.get(name) is None:
            self.mean[name] = emb.mean(dim=0, keepdim=True)
            self.count[name] = _n
        else:
            _total = self.count[name] + _n
            self.mean[name] = (
                self.mean[name] * (self.count[name] / _total)
                + emb.sum(dim=0, keepdim=True) / _total
            )
            self.count[name] = _total

    def add_batch(self, emb: torch.Tensor, voice_idx=None) -> None:
        """Fold a batch in. With no voice_idx every row is the plan's only voice; with
        one, rows are grouped by voice and each group is folded into its own mean. A
        batch whose rows are all one voice is folded whole, not indexed, so a one-voice
        multi plan does exactly what the single-voice path does."""
        if voice_idx is None:
            if len(self.names) != 1:
                raise ValueError(f"{len(self.names)} voices and a batch with no voice "
                                 f"index: cannot tell which voice its rows belong to")
            self.add(self.names[0], emb)
            return
        idx = voice_idx.tolist() if torch.is_tensor(voice_idx) else list(voice_idx)
        if len(idx) != emb.size(0):
            raise ValueError(f"{emb.size(0)} embeddings and {len(idx)} voice indices")
        groups: dict = {}
        for row, v in enumerate(idx):
            groups.setdefault(int(v), []).append(row)
        if len(groups) == 1:
            self.add(self.names[next(iter(groups))], emb)
            return
        for v, rows in groups.items():
            sel = torch.tensor(rows, dtype=torch.long, device=emb.device)
            self.add(self.names[v], emb.index_select(0, sel))

    MEANS_FILE = "voice_means.safetensors"

    def state(self) -> tuple[dict, dict]:
        """(tensors, meta) that let a resume continue the SAME running means: each voice's fp32 mean and its clip
        count. Without them a resumed phase bakes each voice from only the batches after the restart (PR #161
        review), which is silent when every voice reappears and fatal when one does not."""
        tensors = {f"mean.{n}": self.mean[n].detach().cpu().float().contiguous() for n in self.names
                   if self.mean.get(n) is not None}
        return tensors, {"names": list(self.names), "count": {n: int(self.count[n]) for n in self.names}}

    def restore(self, tensors: dict, meta: dict) -> None:
        """Continue from a saved state. Refuses a different voice set: the slots would be filled from another
        plan's means."""
        if list(meta.get("names", [])) != self.names:
            raise ValueError(f"the checkpoint's voices {meta.get('names')} are not this run's {self.names}")
        for n in self.names:
            t = tensors.get(f"mean.{n}")
            c = int(meta["count"].get(n, 0))
            if (t is None) != (c == 0):
                raise ValueError(f"voice {n}: count {c} and mean {'absent' if t is None else 'present'} disagree")
            self.mean[n] = t
            self.count[n] = c

    def vectors(self) -> dict:
        """name -> the (d,) vector to bake. Refuses a voice that saw no clip: an export
        with a row never written is the failure verify_multispeaker_export.py exists to
        catch, and it is cheaper to refuse it here than to find it there."""
        empty = [n for n in self.names if self.mean.get(n) is None]
        if empty:
            raise RuntimeError(f"no clip of voice(s) {empty} was seen before this export "
                               f"(--max_steps too small, or a resume that has not yet "
                               f"reached them); refusing to bake an unwritten row")
        return {n: self.mean[n][0] for n in self.names}


def resume_voice_means(resume_dir: str, means: "VoiceMeans", bakes: bool) -> str:
    """Restore `means` from a resume checkpoint, or say why not. A phase that BAKES a voice (an SFT speaker phase
    without --keep_speaker_encoder) must restore them: a checkpoint written before the means were persisted is
    refused rather than resumed into a biased voice. A phase that bakes nothing has nothing to restore."""
    import os
    rs_path = os.path.join(resume_dir, "resume_state.json")
    with open(rs_path, encoding="utf-8") as fh:
        meta = json.load(fh).get("voice_means")
    f = os.path.join(resume_dir, VoiceMeans.MEANS_FILE)
    if not bakes:
        return "this phase bakes no voice; nothing to restore"
    if meta is None or not os.path.isfile(f):
        raise SystemExit(f"--resume_from {resume_dir} carries no voice means, and this phase bakes a voice: the "
                         f"export would bake each voice from only the batches after the restart. Retrain the "
                         f"phase from the start (a speaker phase is minutes), or resume from a checkpoint written "
                         f"by a trainer that saves them.")
    import safetensors.torch as _st
    means.restore(_st.load_file(f), meta)
    return f"restored voice means {meta['count']}"


class VoiceTaggedDataset(torch.utils.data.Dataset):
    """A TTSDataset whose batches also carry each row's voice, as an index into the plan.

    Only the multi-voice mode wraps its dataset. The sampler sees the same length, so
    the shuffle is the same, and the base collate output is untouched: one key is added.
    A LongTensor rather than a list of names, because accelerate moves tensors through
    its dataloader and makes no promise about anything else.
    """

    def __init__(self, base, voice_idx):
        if len(voice_idx) != len(base):
            raise ValueError(f"{len(base)} rows and {len(voice_idx)} voice indices")
        self.base = base
        self.voice_idx = list(voice_idx)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        item["voice_idx"] = self.voice_idx[i]
        return item

    def collate_fn(self, batch):
        out = self.base.collate_fn(batch)
        out["voice_idx"] = torch.tensor([b["voice_idx"] for b in batch], dtype=torch.long)
        return out


def bake_config(config_dict: dict, slots: dict) -> dict:
    """Mark the export custom_voice and give spk_id / spk_is_dialect one entry per voice.
    Same keys, same insertion order and same values as the inline code it replaces."""
    config_dict["tts_model_type"] = "custom_voice"
    talker_config = config_dict.get("talker_config", {})
    talker_config["spk_id"] = dict(slots)
    talker_config["spk_is_dialect"] = {name: False for name in slots}
    config_dict["talker_config"] = talker_config
    return config_dict


def bake_rows(state_dict: dict, slots: dict, vectors: dict) -> list:
    """Drop speaker_encoder and write each voice's vector at its codec_embedding row.

    Refuses rather than skips when there is no codec_embedding.weight or a slot is past
    the table. The inline code it replaces skipped silently with no table, which would
    have exported a custom_voice config naming a row nobody wrote; no checkpoint this
    project has made lacks the table, so no existing export changes.
    """
    keys_to_drop = [k for k in state_dict if k.startswith("speaker_encoder")]
    for k in keys_to_drop:
        del state_dict[k]
    codec_key = None
    for k in state_dict:
        if k.endswith("codec_embedding.weight"):
            codec_key = k
            break
    if codec_key is None:
        raise RuntimeError("no codec_embedding.weight in the state dict: nowhere to bake "
                           f"{list(slots)}")
    weight = state_dict[codec_key]
    missing = [n for n in slots if n not in vectors]
    if missing:
        raise RuntimeError(f"no vector for voice(s) {missing}")
    past = {n: s for n, s in slots.items() if s >= weight.shape[0]}
    if past:
        raise RuntimeError(f"slot(s) {past} past the codec_embedding table's "
                           f"{weight.shape[0]} rows")
    done = []
    for name, slot in slots.items():
        state_dict[codec_key][slot] = vectors[name].detach().to(weight.device).to(weight.dtype)
        done.append((name, slot))
    return done
