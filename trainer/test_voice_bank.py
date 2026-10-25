# SPDX-License-Identifier: Apache-2.0
"""Several baked voices in one speaker phase, through the main trainer's one code path.

    python trainer/test_voice_bank.py

CPU only, no model, no card. Needs torch and safetensors; the CLI refusals at the end
also need the trainer's imports (mlflow, accelerate, peft, qwen_tts) and skip without
them. CUDA is hidden from every subprocess.

EVERY CHECK IS SHOWN BOTH WAYS, as in test_preference_training.py: the real code passes,
and a broken variant -- the defect the check exists to catch -- is required to fail.

Covered here:
* the single-voice path is byte-identical to the inline code it replaced, which is held
  below VERBATIM from sft_12hz_lora_mlflow.py at ffee791 as the reference: the running
  mean at every step, config.json's bytes, and model.safetensors' bytes;
* a one-speaker multi-voice plan produces the same spk_id, row, config bytes and
  safetensors bytes as the single-voice path;
* two speakers give rows 3000 and 3001, in sorted speaker_id order whatever the line
  order, each the MEAN over that speaker's clips against a hand-computed mean, with
  batches that mix the two speakers;
* the plan's refusals: a map naming a speaker absent from the data, data carrying a
  speaker the map does not name, a row with no speaker_id, two ids on one name, a name
  that is not a neutral identifier;
* an export refuses an unseen voice, a slot past the table, and a table that is absent;
* scripts/common/verify_multispeaker_export.py passes a two-voice fixture export made by
  these functions, and FAILS one with a row never written, one whose map names a row
  past the table, and one checked against the wrong count;
* the trainer itself: every bake goes through voice_bank, and the flags refuse what they
  should before any model is loaded.

NOT covered here, and needing a card (or an hour of CPU on the real 1.7B model): a real
multi-voice speaker phase, end to end, and both voices rendered from its export.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save, save_file

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import voice_bank as vb  # noqa: E402

TRAINER = HERE / "sft_12hz_lora_mlflow.py"
VERIFY = REPO / "scripts" / "common" / "verify_multispeaker_export.py"
D = 16          # embedding width of the fixtures; the real model's is 2048
ROWS = 3072     # codec_embedding rows, as the real talker has


# ---------------------------------------------------------------------------------------
# THE REFERENCE: the single-voice code as it stood in sft_12hz_lora_mlflow.py at ffee791,
# copied verbatim apart from the globals becoming a dict and `args`/`pref` becoming
# parameters. Equality with this is what "byte-identical" means.
# ---------------------------------------------------------------------------------------
class LegacyMean:
    def __init__(self):
        self.target_speaker_embedding = None
        self.speaker_embedding_count = 0

    def add(self, speaker_embedding):
        emb = speaker_embedding.float()
        _n = emb.size(0)
        if self.target_speaker_embedding is None:
            self.target_speaker_embedding = emb.mean(dim=0, keepdim=True)
            self.speaker_embedding_count = _n
        else:
            _total = self.speaker_embedding_count + _n
            self.target_speaker_embedding = (
                self.target_speaker_embedding * (self.speaker_embedding_count / _total)
                + emb.sum(dim=0, keepdim=True) / _total
            )
            self.speaker_embedding_count = _total


def legacy_config(config_dict, speaker_name):
    config_dict["tts_model_type"] = "custom_voice"
    talker_config = config_dict.get("talker_config", {})
    talker_config["spk_id"] = {speaker_name: 3000}
    talker_config["spk_is_dialect"] = {speaker_name: False}
    config_dict["talker_config"] = talker_config
    return config_dict


def legacy_rows(state_dict, target_speaker_embedding, v_pref=None, pref=False):
    keys_to_drop = [k for k in state_dict if k.startswith("speaker_encoder")]
    for k in keys_to_drop:
        del state_dict[k]
    codec_key = None
    for k in state_dict:
        if k.endswith("codec_embedding.weight"):
            codec_key = k
            break
    if codec_key:
        weight = state_dict[codec_key]
        state_dict[codec_key][3000] = (v_pref if pref else target_speaker_embedding[0]).detach().to(weight.device).to(weight.dtype)


# ---------------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------------
def base_config():
    return {"architectures": ["Qwen3TTSForConditionalGeneration"], "tts_model_size": "1b7",
            "tts_model_type": "base",
            "talker_config": {"hidden_size": D, "vocab_size": ROWS, "spk_id": {},
                              "spk_is_dialect": {}}}


def base_state(seed=0):
    g = torch.Generator().manual_seed(seed)
    table = torch.zeros(ROWS, D, dtype=torch.bfloat16)
    table[:2048] = (torch.randn(2048, D, generator=g) * 0.02).to(torch.bfloat16)
    return {"talker.model.codec_embedding.weight": table,
            "talker.model.text_embedding.weight": torch.randn(8, D, generator=g).to(torch.bfloat16),
            "speaker_encoder.fc.weight": torch.randn(D, 4, generator=g).to(torch.bfloat16),
            "speaker_encoder.fc.bias": torch.randn(D, generator=g).to(torch.bfloat16)}


def clone_state(sd):
    return {k: v.clone() for k, v in sd.items()}


def encoder_batches(sizes, seed=1, offset=0.0):
    """What the trainer folds in: speaker_encoder outputs in bf16, of varying batch size."""
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(n, D, generator=g) + offset).to(torch.bfloat16) for n in sizes]


def config_bytes(cfg):
    return json.dumps(cfg, indent=2, ensure_ascii=False).encode("utf-8")


def single_export(name, batches, *, multi_ids=None, smap=None):
    """Run the new code path the way the trainer does; return (config bytes, tensors bytes,
    the mean, the plan)."""
    plan = (vb.single_plan(name) if multi_ids is None
            else vb.plan_from_rows(multi_ids, smap))
    vm = vb.VoiceMeans(plan.names)
    for b in batches:
        idx = None if not plan.multi else torch.zeros(b.size(0), dtype=torch.long)
        vm.add_batch(b.float(), idx)
    cfg = vb.bake_config(base_config(), plan.slots)
    sd = clone_state(base_state())
    vb.bake_rows(sd, plan.slots, vm.vectors())
    return config_bytes(cfg), save(sd), vm, plan


def legacy_export(name, batches):
    lm = LegacyMean()
    for b in batches:
        lm.add(b)
    cfg = legacy_config(base_config(), name)
    sd = clone_state(base_state())
    legacy_rows(sd, lm.target_speaker_embedding)
    return config_bytes(cfg), save(sd), lm


SIZES = [1, 3, 2, 4, 1, 1, 5, 2]


# ---------------------------------------------------------------------------------------
class SingleVoiceIsByteIdentical(unittest.TestCase):
    def test_running_mean_equal_at_every_step(self):
        lm, vm = LegacyMean(), vb.VoiceMeans(["hk_cantonese_speaker"])
        for b in encoder_batches(SIZES):
            lm.add(b)
            vm.add_batch(b.float())
            self.assertTrue(torch.equal(lm.target_speaker_embedding,
                                        vm.mean["hk_cantonese_speaker"]))
            self.assertEqual(lm.speaker_embedding_count, vm.count["hk_cantonese_speaker"])

    def test_config_and_safetensors_bytes_equal(self):
        batches = encoder_batches(SIZES)
        new_cfg, new_st, _, _ = single_export("hk_cantonese_speaker", batches)
        old_cfg, old_st, _ = legacy_export("hk_cantonese_speaker", batches)
        self.assertEqual(new_cfg, old_cfg)
        self.assertEqual(new_st, old_st)

    def test_preference_path_writes_v_back_identically(self):
        v = torch.randn(D)
        sd_new, sd_old = clone_state(base_state()), clone_state(base_state())
        vb.bake_rows(sd_new, vb.single_plan("hk_cantonese_speaker").slots,
                     {"hk_cantonese_speaker": v})
        legacy_rows(sd_old, None, v_pref=v, pref=True)
        self.assertEqual(save(sd_new), save(sd_old))

    def test_red_the_comparison_sees_a_one_clip_difference(self):
        batches = encoder_batches(SIZES)
        new_cfg, new_st, _, _ = single_export("hk_cantonese_speaker", batches)
        _, old_st, _ = legacy_export("hk_cantonese_speaker", batches[:-1])
        self.assertNotEqual(new_st, old_st)

    def test_red_the_comparison_sees_another_row_and_another_name(self):
        batches = encoder_batches(SIZES)
        vm = vb.VoiceMeans(["hk_cantonese_speaker"])
        for b in batches:
            vm.add_batch(b.float())
        sd = clone_state(base_state())
        vb.bake_rows(sd, {"hk_cantonese_speaker": 3001}, vm.vectors())
        _, old_st, _ = legacy_export("hk_cantonese_speaker", batches)
        self.assertNotEqual(save(sd), old_st)
        self.assertNotEqual(config_bytes(vb.bake_config(base_config(), {"other": 3000})),
                            config_bytes(legacy_config(base_config(), "hk_cantonese_speaker")))


class OneSpeakerMultiEqualsSingle(unittest.TestCase):
    def test_mapped_one_speaker_equals_single_voice_path(self):
        batches = encoder_batches(SIZES)
        ids = ["d343"] * sum(SIZES)
        m_cfg, m_st, m_vm, m_plan = single_export(
            None, batches, multi_ids=ids, smap={"d343": "hk_cantonese_speaker"})
        s_cfg, s_st, s_vm, s_plan = single_export("hk_cantonese_speaker", batches)
        self.assertEqual(m_plan.slots, s_plan.slots)
        self.assertEqual(m_plan.slots, {"hk_cantonese_speaker": 3000})
        self.assertTrue(torch.equal(m_vm.mean["hk_cantonese_speaker"],
                                    s_vm.mean["hk_cantonese_speaker"]))
        self.assertEqual(m_cfg, s_cfg)
        self.assertEqual(m_st, s_st)
        # and therefore with the legacy inline code too
        o_cfg, o_st, _ = legacy_export("hk_cantonese_speaker", batches)
        self.assertEqual((m_cfg, m_st), (o_cfg, o_st))

    def test_unmapped_one_speaker_uses_its_speaker_id(self):
        plan = vb.plan_from_rows(["cv_yue_v22"] * 4)
        self.assertEqual(plan.slots, {"cv_yue_v22": 3000})

    def test_red_a_different_voice_name_changes_the_bytes(self):
        batches = encoder_batches(SIZES)
        m_cfg, m_st, _, _ = single_export(None, batches, multi_ids=["d343"] * sum(SIZES))
        s_cfg, s_st, _, _ = single_export("hk_cantonese_speaker", batches)
        self.assertNotEqual(m_cfg, s_cfg)
        self.assertEqual(m_st, s_st)          # same row, same mean: only the name moved

    def test_tagged_dataset_keeps_length_and_base_collate(self):
        class Base(torch.utils.data.Dataset):
            def __len__(self):
                return 5

            def __getitem__(self, i):
                return {"x": torch.tensor([i])}

            def collate_fn(self, batch):
                return {"x": torch.cat([b["x"] for b in batch])}

        base = Base()
        tagged = vb.VoiceTaggedDataset(base, [0, 1, 0, 1, 1])
        self.assertEqual(len(tagged), len(base))
        out = tagged.collate_fn([tagged[i] for i in (3, 0, 4)])
        self.assertTrue(torch.equal(out["x"], torch.tensor([3, 0, 4])))
        self.assertTrue(torch.equal(out["voice_idx"], torch.tensor([1, 0, 1])))
        self.assertEqual(set(out), {"x", "voice_idx"})
        with self.assertRaises(ValueError):
            vb.VoiceTaggedDataset(base, [0, 1])


class TwoSpeakers(unittest.TestCase):
    MAP = {"d777": "hk_voice_777", "d343": "hk_voice_343"}

    def setUp(self):
        g = torch.Generator().manual_seed(7)
        # Two clearly separated voices, as bf16 encoder outputs.
        self.a = [(torch.randn(D, generator=g) + 3.0).to(torch.bfloat16) for _ in range(9)]
        self.b = [(torch.randn(D, generator=g) - 3.0).to(torch.bfloat16) for _ in range(6)]
        # Batches that mix them: [(rows, voice per row)], voice 0 = d343, 1 = d777.
        seq = [(self.a[0:2] + self.b[0:1], [0, 0, 1]),
               (self.b[1:3], [1, 1]),
               (self.a[2:3], [0]),
               (self.a[3:6] + self.b[3:4], [0, 0, 0, 1]),
               (self.b[4:6] + self.a[6:9], [1, 1, 0, 0, 0])]
        self.batches = [(torch.stack(r), torch.tensor(v)) for r, v in seq]
        ids = ["d343"] * 9 + ["d777"] * 6
        self.plan = vb.plan_from_rows(ids, self.MAP)

    def run_means(self):
        vm = vb.VoiceMeans(self.plan.names)
        for rows, idx in self.batches:
            vm.add_batch(rows.float(), idx)
        return vm

    def test_rows_and_order(self):
        self.assertEqual(self.plan.slots, {"hk_voice_343": 3000, "hk_voice_777": 3001})
        self.assertEqual(self.plan.voice_of, {"d343": "hk_voice_343", "d777": "hk_voice_777"})
        self.assertEqual(self.plan.rows_in_data, {"d343": 9, "d777": 6})
        shuffled = vb.plan_from_rows(["d777", "d343"] * 3 + ["d343"] * 3, self.MAP)
        self.assertEqual(shuffled.slots, self.plan.slots)

    def test_each_row_is_the_mean_over_that_speakers_clips(self):
        vm = self.run_means()
        hand_a = torch.stack([x.float() for x in self.a]).mean(0)
        hand_b = torch.stack([x.float() for x in self.b]).mean(0)
        vec = vm.vectors()
        self.assertTrue(torch.allclose(vec["hk_voice_343"], hand_a, atol=1e-6))
        self.assertTrue(torch.allclose(vec["hk_voice_777"], hand_b, atol=1e-6))
        self.assertEqual(vm.count, {"hk_voice_343": 9, "hk_voice_777": 6})
        sd = clone_state(base_state())
        done = vb.bake_rows(sd, self.plan.slots, vec)
        self.assertEqual(done, [("hk_voice_343", 3000), ("hk_voice_777", 3001)])
        t = sd["talker.model.codec_embedding.weight"]
        self.assertTrue(torch.equal(t[3000], vec["hk_voice_343"].to(torch.bfloat16)))
        self.assertTrue(torch.equal(t[3001], vec["hk_voice_777"].to(torch.bfloat16)))
        self.assertTrue(torch.allclose(t[3000].float(), hand_a, atol=0.05))
        self.assertTrue(torch.equal(t[3002], torch.zeros(D, dtype=torch.bfloat16)))
        self.assertFalse(any(k.startswith("speaker_encoder") for k in sd))
        cfg = vb.bake_config(base_config(), self.plan.slots)
        self.assertEqual(cfg["tts_model_type"], "custom_voice")
        self.assertEqual(cfg["talker_config"]["spk_id"],
                         {"hk_voice_343": 3000, "hk_voice_777": 3001})
        self.assertEqual(cfg["talker_config"]["spk_is_dialect"],
                         {"hk_voice_343": False, "hk_voice_777": False})

    def test_red_the_first_clip_is_not_the_mean(self):
        # The fork's pre-49b2908 method: one clip per voice.
        hand_a = torch.stack([x.float() for x in self.a]).mean(0)
        self.assertFalse(torch.allclose(self.a[0].float(), hand_a, atol=1e-6))

    def test_red_one_pooled_mean_is_neither_voice(self):
        # A blend wearing a multi-voice config: every row folded into one mean.
        pooled = vb.VoiceMeans(["x"])
        for rows, _ in self.batches:
            pooled.add_batch(rows.float())
        hand_a = torch.stack([x.float() for x in self.a]).mean(0)
        self.assertFalse(torch.allclose(pooled.vectors()["x"], hand_a, atol=1e-2))

    def test_red_a_batch_without_voices_is_refused_in_a_two_voice_plan(self):
        with self.assertRaises(ValueError):
            vb.VoiceMeans(self.plan.names).add_batch(self.batches[0][0].float())

    def test_voice_indices_follow_the_plan(self):
        rows = [{"speaker_id": s} for s in ("d777", "d343", "d343")]
        self.assertEqual(self.plan.voice_indices(rows), [1, 0, 0])
        with self.assertRaises(vb.VoicePlanError):
            self.plan.voice_indices([{"speaker_id": "d999"}])


class PlanRefusals(unittest.TestCase):
    def refused(self, ids, smap=None):
        with self.assertRaises(vb.VoicePlanError) as cm:
            vb.plan_from_rows(ids, smap)
        return str(cm.exception)

    def test_red_map_names_a_speaker_absent_from_the_data(self):
        msg = self.refused(["d343", "d343"], {"d343": "hk_voice_343", "d777": "hk_voice_777"})
        self.assertIn("absent from the data", msg)
        self.assertIn("d777", msg)

    def test_red_data_carries_a_speaker_the_map_does_not_name(self):
        msg = self.refused(["d343", "d777"], {"d343": "hk_voice_343"})
        self.assertIn("does not name", msg)
        self.assertIn("d777", msg)

    def test_red_row_without_speaker_id(self):
        self.assertIn("row 1 has no speaker_id", self.refused(["d343", None, "d343"]))

    def test_red_two_ids_one_name(self):
        self.assertIn("all map to voice", self.refused(["a", "b"], {"a": "v", "b": "v"}))

    def test_red_a_name_with_a_space(self):
        self.assertIn("not a neutral identifier",
                      self.refused(["a"], {"a": "Given Family"}))

    def test_every_problem_named_not_only_the_first(self):
        msg = self.refused(["a", "c", None], {"a": "x y", "b": "v"})
        for want in ("absent from the data", "does not name", "no speaker_id",
                     "not a neutral identifier"):
            self.assertIn(want, msg)

    def test_a_good_map_passes(self):
        vb.plan_from_rows(["d343", "d777"], {"d343": "hk_voice_343", "d777": "hk_voice_777"})


class ExportRefusals(unittest.TestCase):
    def test_red_unseen_voice(self):
        vm = vb.VoiceMeans(["a", "b"])
        vm.add_batch(torch.ones(2, D), torch.tensor([0, 0]))
        with self.assertRaises(RuntimeError):
            vm.vectors()

    def test_red_slot_past_the_table(self):
        with self.assertRaises(RuntimeError):
            vb.bake_rows(clone_state(base_state()), {"a": ROWS}, {"a": torch.ones(D)})

    def test_red_no_table(self):
        sd = clone_state(base_state())
        del sd["talker.model.codec_embedding.weight"]
        with self.assertRaises(RuntimeError):
            vb.bake_rows(sd, {"a": 3000}, {"a": torch.ones(D)})


# The lab's verify_multispeaker_export.py is not part of the recipe (one voice per speaker phase), so these cases
# skip here rather than fail on a missing script.
@unittest.skipUnless(VERIFY.exists(), "verify_multispeaker_export.py is the lab's, not shipped in the recipe")
class VerifierOnFixtureExports(unittest.TestCase):
    """verify_multispeaker_export.py on a checkpoint folder written by these functions."""

    def export(self, root, mutate=None):
        plan = vb.plan_from_rows(["d343"] * 3 + ["d777"] * 2,
                                 {"d343": "hk_voice_343", "d777": "hk_voice_777"})
        vm = vb.VoiceMeans(plan.names)
        vm.add_batch(torch.randn(3, D) + 2.0, torch.tensor([0, 0, 0]))
        vm.add_batch(torch.randn(2, D) - 2.0, torch.tensor([1, 1]))
        cfg = vb.bake_config(base_config(), plan.slots)
        sd = clone_state(base_state())
        vb.bake_rows(sd, plan.slots, vm.vectors())
        if mutate:
            mutate(cfg, sd)
        d = Path(root) / "final_merged"
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        save_file(sd, str(d / "model.safetensors"))
        return Path(root)

    def verify(self, ck, expect):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        r = subprocess.run([sys.executable, str(VERIFY), str(ck), "--expect", str(expect)],
                           capture_output=True, text=True, env=env)
        return r.returncode, r.stdout + r.stderr

    def test_two_voice_export_passes(self):
        with tempfile.TemporaryDirectory() as t:
            rc, out = self.verify(self.export(t), 2)
            self.assertEqual(rc, 0, out)
            self.assertIn("PASS: 2 distinct selectable voices", out)

    def test_red_a_row_not_written(self):
        def unwrite(cfg, sd):
            sd["talker.model.codec_embedding.weight"][3001] = 0
        with tempfile.TemporaryDirectory() as t:
            rc, out = self.verify(self.export(t, unwrite), 2)
            self.assertEqual(rc, 1, out)
            self.assertIn("LOOKS UNWRITTEN", out)

    def test_red_map_names_a_missing_row(self):
        def past(cfg, sd):
            cfg["talker_config"]["spk_id"]["hk_voice_777"] = ROWS + 5
        with tempfile.TemporaryDirectory() as t:
            rc, out = self.verify(self.export(t, past), 2)
            self.assertEqual(rc, 1, out)
            self.assertIn("past the table", out)

    def test_red_map_names_a_third_voice_never_baked(self):
        def third(cfg, sd):
            cfg["talker_config"]["spk_id"]["hk_voice_x"] = 3002
            cfg["talker_config"]["spk_is_dialect"]["hk_voice_x"] = False
        with tempfile.TemporaryDirectory() as t:
            rc, out = self.verify(self.export(t, third), 3)
            self.assertEqual(rc, 1, out)
            self.assertIn("never baked", out)

    def test_red_wrong_count(self):
        with tempfile.TemporaryDirectory() as t:
            rc, out = self.verify(self.export(t), 3)
            self.assertEqual(rc, 1, out)


class TheTrainerUsesIt(unittest.TestCase):
    SRC = TRAINER.read_text(encoding="utf-8")

    def test_every_bake_goes_through_voice_bank(self):
        self.assertEqual(self.SRC.count("vb.bake_config(config_dict, voice_plan.slots)"), 2)
        calls = [ln for ln in self.SRC.splitlines() if ln.strip() == "_bake_rows(state_dict)"]
        self.assertEqual(len(calls), 2)            # the epoch export and final_merged
        self.assertEqual(self.SRC.count("voice_means.add_batch(emb, batch.get(\"voice_idx\"))"), 1)

    def test_red_no_inline_bake_or_mean_survives(self):
        for gone in ("[3000] =", "target_speaker_embedding", "speaker_embedding_count",
                     "talker_config[\"spk_id\"] ="):
            self.assertNotIn(gone, self.SRC, gone)


def _trainer_importable() -> bool:
    try:
        import accelerate, mlflow, peft, qwen_tts  # noqa: F401,E401
        return True
    except Exception:
        return False


@unittest.skipUnless(_trainer_importable(), "the trainer's imports are not in this env")
class TrainerFlagsRefuse(unittest.TestCase):
    """Run the real trainer's argument handling. Every case exits before the model loads,
    before the CUDA check, and with CUDA hidden regardless."""

    def run_trainer(self, *argv):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        env.pop("QWEN_ALLOW_CPU_TRAIN", None)
        return subprocess.run([sys.executable, str(TRAINER), *argv], capture_output=True,
                              text=True, env=env, cwd=str(HERE), timeout=600)

    def manifest(self, t, ids):
        p = Path(t) / "train.jsonl"
        p.write_text("".join(json.dumps({"speaker_id": s, "text": "x"}) + "\n" for s in ids),
                     encoding="utf-8")
        return str(p)

    def test_red_map_without_the_mode(self):
        with tempfile.TemporaryDirectory() as t:
            r = self.run_trainer("--train_jsonl", self.manifest(t, ["a"]),
                                 "--speaker_map", "m.json")
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("--speaker_map names the voices", r.stderr)

    def test_red_with_keep_speaker_encoder(self):
        with tempfile.TemporaryDirectory() as t:
            r = self.run_trainer("--train_jsonl", self.manifest(t, ["a"]),
                                 "--speakers_from_jsonl", "--keep_speaker_encoder")
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("nothing for --speakers_from_jsonl to bake", r.stderr)

    def test_red_map_naming_an_absent_speaker_stops_before_the_model(self):
        with tempfile.TemporaryDirectory() as t:
            m = Path(t) / "map.json"
            m.write_text(json.dumps({"d343": "hk_voice_343", "d777": "hk_voice_777"}),
                         encoding="utf-8")
            r = self.run_trainer("--train_jsonl", self.manifest(t, ["d343"] * 3),
                                 "--speakers_from_jsonl", "--speaker_map", str(m))
            self.assertEqual(r.returncode, 1, r.stderr)
            self.assertIn("absent from the data", r.stderr)
            self.assertNotIn("REFUSING TO TRAIN ON CPU", r.stderr)

    def test_a_good_plan_reaches_the_cpu_refusal_and_prints_its_rows(self):
        # Proves the plan is accepted and printed, then stops at the CPU guard: CUDA is
        # hidden and QWEN_ALLOW_CPU_TRAIN unset, so nothing loads.
        with tempfile.TemporaryDirectory() as t:
            m = Path(t) / "map.json"
            m.write_text(json.dumps({"d343": "hk_voice_343", "d777": "hk_voice_777"}),
                         encoding="utf-8")
            r = self.run_trainer("--train_jsonl", self.manifest(t, ["d777", "d343", "d343"]),
                                 "--speakers_from_jsonl", "--speaker_map", str(m))
            self.assertIn("row 3000 = hk_voice_343 (d343, 2 rows)", r.stdout)
            self.assertIn("row 3001 = hk_voice_777 (d777, 1 rows)", r.stdout)
            self.assertIn("REFUSING TO TRAIN ON CPU", r.stdout + r.stderr)



class ResumeKeepsVoiceMeans(unittest.TestCase):
    """PR #161 review (Codex P1): a resumed phase must bake the mean over every clip it trained on."""

    def _batches(self):
        g = torch.Generator().manual_seed(7)
        return [(torch.randn(4, 8, generator=g), torch.tensor([0, 1, 0, 1])) for _ in range(6)]

    def _save(self, d, means, with_means=True):
        import safetensors.torch as _st
        state = {"optimizer_steps": 3}
        if with_means:
            t, m = means.state()
            _st.save_file(t, os.path.join(d, vb.VoiceMeans.MEANS_FILE))
            state["voice_means"] = m
        with open(os.path.join(d, "resume_state.json"), "w", encoding="utf-8") as fh:
            json.dump(state, fh)

    def test_resume_continues_the_same_means_exactly(self):
        names = ["hk_voice_343", "hk_voice_777"]
        full = vb.VoiceMeans(names)
        for e, i in self._batches():
            full.add_batch(e, i)
        first = vb.VoiceMeans(names)
        for e, i in self._batches()[:3]:
            first.add_batch(e, i)
        with tempfile.TemporaryDirectory() as d:
            self._save(d, first)
            resumed = vb.VoiceMeans(names)
            vb.resume_voice_means(d, resumed, bakes=True)
        for e, i in self._batches()[3:]:
            resumed.add_batch(e, i)
        for n in names:
            self.assertTrue(torch.equal(full.mean[n], resumed.mean[n]), n)
            self.assertEqual(full.count[n], resumed.count[n])

    def test_red_without_restore_the_means_are_biased(self):
        names = ["hk_voice_343", "hk_voice_777"]
        full = vb.VoiceMeans(names)
        for e, i in self._batches():
            full.add_batch(e, i)
        fresh = vb.VoiceMeans(names)                 # what the trainer did before the fix
        for e, i in self._batches()[3:]:
            fresh.add_batch(e, i)
        self.assertFalse(torch.equal(full.mean["hk_voice_343"], fresh.mean["hk_voice_343"]))

    def test_red_a_baking_phase_refuses_a_checkpoint_without_means(self):
        with tempfile.TemporaryDirectory() as d:
            self._save(d, vb.VoiceMeans(["a"]), with_means=False)
            with self.assertRaises(SystemExit):
                vb.resume_voice_means(d, vb.VoiceMeans(["a"]), bakes=True)

    def test_a_phase_that_bakes_nothing_resumes_without_means(self):
        with tempfile.TemporaryDirectory() as d:
            self._save(d, vb.VoiceMeans(["a"]), with_means=False)
            self.assertIn("nothing to restore", vb.resume_voice_means(d, vb.VoiceMeans(["a"]), bakes=False))

    def test_red_a_different_voice_set_is_refused(self):
        a = vb.VoiceMeans(["hk_voice_343", "hk_voice_777"])
        for e, i in self._batches()[:2]:
            a.add_batch(e, i)
        with tempfile.TemporaryDirectory() as d:
            self._save(d, a)
            with self.assertRaises(ValueError):
                vb.resume_voice_means(d, vb.VoiceMeans(["hk_voice_343", "other"]), bakes=True)

    def test_the_trainer_writes_and_restores_them(self):
        src = TRAINER.read_text(encoding="utf-8")
        self.assertIn("voice_means=voice_means)", src)               # the step checkpoint passes them
        self.assertIn("vb.resume_voice_means(", src)                 # the resume restores them
        self.assertIn("bakes=not pref and not args.keep_speaker_encoder", src)

if __name__ == "__main__":
    unittest.main(verbosity=2)
