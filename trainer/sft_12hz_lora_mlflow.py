# coding=utf-8
# Copyright 2026 The Alibaba Qwen team + LoRA modification by Grok
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import math
import os
import shutil
import subprocess
import time
import torch
from accelerate import Accelerator
from dataset import TTSDataset
from epoch_accounting import EpochAccounting
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors import safe_open
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, get_linear_schedule_with_warmup
import mlflow

from peft import LoraConfig, get_peft_model

# THE BAKED VOICES. Every speaker phase, one voice or several, accumulates and bakes
# through voice_bank.py. The single-voice path was two module globals and two inline
# row-3000 writes until 2026-10-17; it now runs the same arithmetic from there.
import voice_bank as vb


def clean_merged_state_dict(peft_model):
    """After merge_adapter(), extract COMPLETELY clean state_dict
    - removes ALL lora_* keys
    - removes base_layer.weight → .weight
    - removes base_model.model. prefix (fixes checkpoint warnings)
    """
    raw_state = peft_model.state_dict()
    clean_dict = {}
    for k, v in raw_state.items():
        if "lora_" in k:
            continue

        clean_k = k
        # Strip PEFT wrapper prefix
        if clean_k.startswith("base_model.model."):
            clean_k = clean_k[len("base_model.model."):]
        if clean_k.startswith("model."):
            clean_k = clean_k[6:]

        # Convert base_layer.weight → weight
        clean_k = clean_k.replace(".base_layer.weight", ".weight")

        clean_dict[clean_k] = v.detach().to("cpu")
    return clean_dict


# ==================== FULL FINE-TUNE SUPPORT (fork addition) ====================
# Everything below this banner is reached only under --full_ft. The LoRA path does
# not call any of it and its behaviour is unchanged.
#
# Why this exists: every checkpoint this project has produced excludes
# `.*code_predictor.*` from the adapter, so the acoustic-realisation layer --
# codebooks 1-15, the thing that turns the talker's plan into a waveform -- has
# never had a gradient applied to it. The remaining unexplained defect is
# sub-phonemic (correct words, correct dialect, wrong realisation), which is
# exactly the layer's job. Testing that hypothesis needs full-weight training, and
# LoRA cannot express it.


def write_resume_checkpoint(out_root, unwrapped, optimizer_steps, epoch, lr,
                            recent_losses, keep_dir=[None], voice_means=None):
    """Write the raw model state plus the JSON a resume needs, atomically.

    DELIBERATELY NOT THE EXPORT PATH. The exported checkpoint is produced by a
    copytree, a config rewrite, a speaker_encoder drop and an embedding injection at
    codec index 3000 -- the code that makes every artifact this project ships. A
    mid-run checkpoint is never shipped; it exists only so a preempted run does not
    re-pay for hours already spent. Routing it through the export path would put that
    code on a new hot path for no benefit.

    WEIGHTS ONLY, AND THE COST OF THAT IS MEASURED RATHER THAN ASSERTED. AdamW moments
    and MasterWeightAdamW's fp32 shadows are not saved: together they are ~19.5 GiB at
    talker scope against 4.3 for the weights, and the shadows live in a plain list on
    the optimizer that accelerate's save_state does not reach -- so a "full state" save
    would silently restore the moments and lose the masters, which is the mechanism that
    makes full-weight training work at all. It would look like it worked. The resume
    instrumentation reports the actual loss discontinuity instead.

    ATOMIC, because a checkpoint half-written when the preemption lands is worse than
    none: it would be resumed from and produce silent garbage. Written to a .tmp
    sibling and renamed, and the previous checkpoint is deleted only after the rename.
    """
    import safetensors.torch as _st
    os.makedirs(out_root, exist_ok=True)
    final = os.path.join(out_root, f"step-{optimizer_steps}")
    tmp = final + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    sd = {k: v.detach().cpu().contiguous() for k, v in unwrapped.state_dict().items()}
    _st.save_file(sd, os.path.join(tmp, "model.safetensors"))
    state = {"optimizer_steps": optimizer_steps, "epoch": epoch, "lr": lr,
             "recent_losses": [round(float(x), 6) for x in recent_losses[-20:]],
             "recent_loss_mean": (round(sum(recent_losses[-20:]) / len(recent_losses[-20:]), 6)
                                  if recent_losses else None),
             "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "restores": "weights, optimiser-step count, LR schedule position"
                         + (", each voice's running mean and clip count" if voice_means is not None else ""),
             "does_not_restore": "AdamW exp_avg/exp_avg_sq, MasterWeightAdamW fp32 masters"}
    if voice_means is not None:
        # The voices' running means, so a resumed phase bakes the mean over EVERY clip it trained on, not only
        # the batches after the restart (PR #161 review). Inside the atomic rename, like the weights.
        _vt, _vm = voice_means.state()
        _st.save_file(_vt, os.path.join(tmp, voice_means.MEANS_FILE))
        state["voice_means"] = _vm
    with open(os.path.join(tmp, "resume_state.json"), "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=1)
    shutil.rmtree(final, ignore_errors=True)
    os.rename(tmp, final)
    print(f"RESUME_CKPT step={optimizer_steps} -> {final} "
          f"(loss_mean={state['recent_loss_mean']})", flush=True)

    # OFF THE INSTANCE THE MOMENT IT EXISTS, for the same reason the export path does
    # it. A resume checkpoint sitting only on the training machine protects against a
    # crashed PROCESS and not against a deleted VM -- and --max-run-duration deletes
    # via the control plane without running any exit handler, so nothing publishes.
    # protect_artifact.sh already exists for exactly this and is never fatal by design:
    # losing the NEXT checkpoint because the last could not be uploaded is this failure
    # inverted. Off-cloud it is a no-op, so local runs pay nothing.
    #
    # The PUSH happens before the previous checkpoint is deleted, so there is never a
    # moment with no complete copy anywhere.
    repo = os.environ.get("REPO_ROOT")
    script = os.path.join(repo, "scripts", "common", "protect_artifact.sh") if repo else None
    if script and os.path.isfile(script):
        try:
            subprocess.run(["bash", script, final], timeout=1800, check=False)
        except Exception as exc:                                  # never fatal
            print(f"RESUME_CKPT push failed, keeping local copy: {exc}", flush=True)

    if keep_dir[0] and keep_dir[0] != final:
        shutil.rmtree(keep_dir[0], ignore_errors=True)
    keep_dir[0] = final
    return final


def full_ft_state_dict(module):
    """State dict for a full fine-tune checkpoint, in the base model's dtype.

    The LoRA path gets its state dict out of merge_and_unload(), which returns the
    base module with adapters folded in, so its keys are already the natural ones
    and its dtype is already the load dtype. A full fine-tune has no merge step, so
    the equivalent is a plain state_dict() -- but the dtype is NOT automatically
    right, because --full_ft_precision fp32 loads the model in float32.

    All 480 tensors in Qwen3-TTS-12Hz-1.7B-Base are BF16. Writing float32 tensors
    instead would double the checkpoint to 7.2 GB and hand the downstream
    consumers -- Studio's convert_tts_to_gguf.py, and this script itself when a
    checkpoint seeds a second stage via --init_model_path -- a file whose dtype
    does not match the config it was copied from. Cast on the way out so an FFT
    checkpoint is byte-for-byte the same *shape* of artefact as a merged LoRA one.
    """
    out = {}
    for k, v in module.state_dict().items():
        v = v.detach().to("cpu")
        if v.is_floating_point():
            v = v.to(torch.bfloat16)
        out[k] = v
    return out


class MasterWeightAdamW(AdamW):
    """AdamW over fp32 master copies of bf16 parameters.

    This is not a refinement, it is a correctness fix, and the measurement is
    worth keeping because it is invisible in every other way.

    bf16 has an 8-bit mantissa, so representable values near a weight of magnitude
    |w| are spaced about |w| * 2^-8 apart. An AdamW step is approximately `lr` in
    magnitude, because Adam divides by sqrt(v) and thereby normalises the update to
    roughly unit scale before scaling by lr. For a pretrained transformer weight of
    |w| ~ 0.02 the bf16 spacing is 1.22e-4, so half an ulp is 6.1e-5 -- and the
    official full-fine-tune rate in Qwen3-TTS's own finetuning/README.md is 2e-6.
    The update is 3% of the smallest change bf16 can represent. Round-to-nearest
    returns the weight unchanged, forever, with no error and no warning.

    Measured, 50k weights ~ N(0, 0.02), 50 AdamW steps at each rate, fraction of
    weights whose value changed at all:

        lr        bf16 params     fp32 params
        2e-6          3.8%           100.0%
        1e-5         15.0%           100.0%
        5e-5         53.9%           100.0%
        1e-4         84.9%           100.0%
        2e-4         97.3%           100.0%

    The sharper measurement is what happens as the run gets LONGER, because the
    intuition that "it will catch up eventually" is precisely wrong. Same weights,
    lr 2e-6, a persistent gradient direction, plain bf16 AdamW against this class:

        steps    plain bf16 AdamW    fp32 masters
           10          3.9%             27.2%
           50          3.9%             82.4%
          100          3.9%             95.7%
          200          3.9%             98.1%

    Plain bf16 is FLAT. Not slow -- flat. A sub-ulp update is discarded at the
    moment it is applied, so it never accumulates and the weight is stuck for the
    entire run however many steps it gets. (The 3.9% that do move are the weights
    near zero, where the ulp is small enough to resolve the step.) With an fp32
    master the identical updates accumulate off to the side and the bf16 weight
    lands in discrete jumps once the accumulation clears half an ulp.

    The LoRA path never met this because it is exempt twice over: lora_B is
    initialised to zero, so the first update is infinitely large relative to the
    parameter, and LoRA runs at 1e-4 to 2e-4 where bf16 resolves the step anyway.
    A full fine-tune at 2e-6 in pure bf16 is a no-op wearing a plausible loss
    curve: the loss does move, because that 3.9% moves, and a mostly-frozen model
    is indistinguishable from a very small learning rate until you try to use it.

    torch.optim.AdamW allocates exp_avg / exp_avg_sq with zeros_like(p), so it
    inherits the parameter dtype: a bf16 model also gets a bf16 second moment,
    where the squared-gradient accumulator loses precision on exactly the small
    values it exists to track. Holding the masters in fp32 fixes both at once.

    Memory, 1.7B, code_predictor scope (175M trainable): bf16 model 3.59 GiB +
    bf16 grads 0.33 + fp32 masters 0.65 + fp32 Adam state 1.30 = 5.87 GiB, against
    9.13 GiB for the same run with the whole model resident in fp32. That
    difference is what decides whether the code-predictor lane fits a 12 GB card.

    The loop calls optimizer.step() and optimizer.zero_grad() through
    accelerate's AcceleratedOptimizer, which forwards both, so overriding them here
    needs no change to the training loop at all.
    """

    def __init__(self, param_groups, **kwargs):
        # Pair each live model parameter with an fp32 shadow. AdamW is given the
        # shadows, so its state -- and therefore its arithmetic -- is fp32
        # throughout; the model keeps its bf16 parameters for forward and backward.
        # EXP-013: offload_master puts the fp32 shadows -- and therefore every AdamW
        # moment, since torch allocates them with zeros_like(p) -- in system RAM.
        # `.clone().float()` alone lands them beside the model, which is where the
        # 6.49 + 12.98 GiB in the memory table comes from.
        self._offload_master = bool(kwargs.pop("offload_master", False))
        # foreach=False, AND IT IS A MEMORY FIX RATHER THAN A TUNING KNOB.
        # torch's multi-tensor AdamW -- `foreach`, which torch selects by default
        # whenever the params are on CUDA -- evaluates
        # `torch._foreach_sqrt(device_exp_avg_sqs)` across every parameter at once.
        # That materialises a SECOND fp32 copy of the entire second moment: 4 bytes
        # per trainable parameter, 6.49 GiB at talker scope, live only inside the
        # step and invisible to any static accounting.
        #
        # EXP-023's full_talker arm OOMed on precisely that allocation, twice, and
        # the arithmetic closes to three figures: 32.79 GiB live at the step, torch
        # reporting 38.12 GiB allocated and requesting 1.16 GiB more --
        # 38.12 - 32.79 = 5.33, and 5.33 + 1.16 = 6.49, one fp32 copy of the
        # trainable parameters, on a 39.49 GiB A100.
        #
        # The single-tensor loop is slower per step by a fraction of a second, over
        # the ~86 optimiser steps a 343-clip speaker phase runs. setdefault, not a
        # hard assignment, so a caller with room to spare can still ask for it.
        kwargs.setdefault("foreach", False)
        self._pairs = []
        shadow_groups = []
        for group in param_groups:
            shadows = []
            for p in group["params"]:
                master = p.detach().clone().float()
                if self._offload_master:
                    master = master.to("cpu")
                master.requires_grad_(True)
                self._pairs.append((p, master))
                shadows.append(master)
            shadow_group = dict(group)
            shadow_group["params"] = shadows
            shadow_groups.append(shadow_group)
        super().__init__(shadow_groups, **kwargs)

    def _sync_devices(self):
        """Follow the model onto whatever device accelerate put it on.

        The shadows are cloned in __init__, which runs BEFORE
        accelerator.prepare() moves the model to CUDA -- so on the first step the
        masters are still on CPU while p.grad is on CUDA, and the assignment below
        raises. The smoke test caught this before four lanes were armed on it.

        Constructing the optimizer after prepare() would also fix it, but that
        reorders the existing LoRA path for the benefit of the FFT path. Following
        the parameter is local to this class and cannot regress the working one.

        Object identity is preserved deliberately: self.state is keyed by the
        master tensor, so the storage is moved in place rather than replaced. A
        fresh tensor would silently orphan the Adam moments -- which on the first
        step are empty, so it would look fine and quietly reset momentum on every
        device change thereafter.
        """
        # EXP-013: with offload on, the masters are SUPPOSED to be on the host and
        # must not be dragged back. Following the parameter is the whole point of
        # this method in the resident case and exactly wrong in the offloaded one.
        if getattr(self, "_offload_master", False):
            return
        for p, master in self._pairs:
            if master.data.device != p.device:
                master.data = master.data.to(p.device)
                st = self.state.get(master)
                if st:
                    for k, v in st.items():
                        if torch.is_tensor(v):
                            st[k] = v.to(p.device)

    @torch.no_grad()
    def step(self, closure=None):
        self._sync_devices()
        # Lift the bf16 gradients onto the masters, step in fp32, write back.
        # The write-back is the only lossy operation, and it is lossy in the way
        # mixed precision is meant to be: the *accumulated* update survives in the
        # master even when a single step rounds away in the bf16 copy.
        # EXP-013 OFFLOAD PATH. The fp32 masters and every AdamW moment live in
        # system RAM; only the gradient crossing and the weight write-back touch the
        # bus. Forward and backward are untouched and stay entirely on the GPU.
        #
        # This is what the memory table's 6.49 + 12.98 GiB of masters and moments
        # buys back. Whether the PCIe round trip is affordable is the whole question
        # EXP-013 exists to answer, and it is deliberately NOT predicted here.
        if getattr(self, "_offload_master", False):
            for p, master in self._pairs:
                master.grad = (None if p.grad is None
                               else p.grad.detach().to("cpu", torch.float32,
                                                       non_blocking=True))
            loss = super().step(closure)
            for p, master in self._pairs:
                p.copy_(master.to(p.device, p.dtype, non_blocking=True))
            return loss

        for p, master in self._pairs:
            master.grad = None if p.grad is None else p.grad.detach().float()
        loss = super().step(closure)
        for p, master in self._pairs:
            p.copy_(master.to(p.dtype))
        return loss

    def zero_grad(self, set_to_none=True):
        # Both sides. Clearing only the shadows would leave the model's own
        # gradients accumulating across every optimizer step for the whole run --
        # a silent, monotonically worsening effective batch size.
        super().zero_grad(set_to_none=set_to_none)
        for p, _ in self._pairs:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()


def apply_full_ft_scope(model, scope):
    """Freeze everything outside `scope` and report what is left trainable.

    Returns (n_trainable, n_total). The scopes, and why each exists:

    `all`             every parameter. Note that speaker_encoder still receives no
                      gradient regardless, because the training loop detaches its
                      output -- so `all` and `talker` differ only in bookkeeping
                      for 12M parameters. Kept because "full fine-tune" should mean
                      what it says.
    `talker`          the talker stack WITHOUT the code predictor: the closest
                      full-weight analogue of the LoRA runs already on disk, and
                      the control that separates "full-weight training helps" from
                      "training the code predictor helps".
    `code_predictor`  175M parameters, 9.1% of the model, and the only scope that
                      fits the 12 GB workstation. This is the lane that answers the
                      standing hypothesis for the price of no cloud spend.

    A trap specific to `code_predictor`, and it will silently halve the experiment:
    the code predictor receives gradient from TWO places. Its input embedding
    tables are read in the main talker path (the codebook 1-15 loop), so they train
    from talker_loss. Its transformer and output head are only reached through
    forward_sub_talker_finetune, whose contribution is scaled by
    --sub_talker_weight. This project's LoRA runs set that weight to 0, which was
    coherent when the code predictor was excluded from the adapter and is
    incoherent here: at weight 0 the code-predictor scope trains 16 embedding
    tables and nothing else. Use the upstream 0.3.
    """
    total = 0
    trainable = 0
    for name, p in model.named_parameters():
        total += p.numel()
        if scope == "all":
            want = True
        elif scope == "code_predictor":
            want = "code_predictor" in name
        elif scope == "talker":
            want = "code_predictor" not in name and (
                name.startswith("talker") or ".talker." in name)
        else:
            raise ValueError(f"unknown full_ft scope: {scope}")
        p.requires_grad_(want)
        if want:
            trainable += p.numel()
    return trainable, total


def train():
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen3-TTS with MLflow + Warmup")
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, default=None,
                        help="The SFT manifest. Required unless --preference_jsonl is given.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate (LoRA typically needs 1e-4–3e-4; full FT uses ~2e-6)")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--speaker_name", type=str, default="hk_cantonese_speaker")
    # SEVERAL VOICES IN ONE SPEAKER PHASE (2026-10-17, the owner: "train two voices into
    # each"). Off by default, and off means the single voice at row 3000 named by
    # --speaker_name, byte-identical to before. On, each row's `speaker_id` is its voice:
    # one running mean per voice, by the same arithmetic as the single voice, baked at
    # rows 3000, 3001, ... in sorted speaker_id order, every voice in spk_id. This
    # replaces sft_12hz_lora_mlflow_multi_speaker.py, a fork that drifted twice and never
    # had this file's loss fixes. See docs/multi-speaker-lora-export.md.
    parser.add_argument("--speakers_from_jsonl", action="store_true",
                        help="Bake one voice per distinct speaker_id in --train_jsonl, at "
                             "codec rows 3000, 3001, ... in sorted speaker_id order. "
                             "--speaker_name is then unused. Off: one voice, "
                             "--speaker_name, row 3000, as always.")
    parser.add_argument("--speaker_map", type=str, default=None,
                        help="With --speakers_from_jsonl: a JSON object of speaker_id -> "
                             "voice name, e.g. {\"d343\": \"hk_voice_343\"}. Every "
                             "speaker_id in the data must be named and every name must "
                             "be in the data. Without it the speaker_id is the name. "
                             "Names are neutral identifiers, never a person's name.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--steptime_warmup", type=int, default=5,
                        help="How many leading optimiser steps the STEPTIME median "
                             "discards. The first steps include allocator growth and "
                             "kernel autotuning and are not representative. A "
                             "pre-registered timing metric must state a number and "
                             "this must match it -- EXP-014 registered 20 while the "
                             "instrument used 5, and the raw times were not logged, "
                             "so the registered median was not recomputable.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Fraction of total steps to use for linear LR warmup (default: 0.1 = 10%%)")
    parser.add_argument("--attn_implementation", type=str, default="sdpa",
                        choices=["sdpa", "flash_attention_2", "eager"],
                        help="Attention backend. Use flash_attention_2 if installed (see flash-attent-install.sh)")
    # Fork addition. Absent upstream, but the model supports it natively:
    # Qwen3TTSPreTrainedModel sets supports_gradient_checkpointing and
    # Qwen3TTSDecoderLayer derives from transformers' GradientCheckpointingLayer.
    # Trades roughly 20-30% step time for a large cut in activation memory, which
    # is the pressure that decides whether 1.7B LoRA fits in 12 GB.
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Recompute activations during backward instead of storing them. "
                             "Slower per step, much lower peak VRAM.")
    # Fork addition. Off by default because adapting the code_predictor destroys
    # intelligibility; retained as a flag so the original behaviour is reproducible
    # rather than merely gone.
    parser.add_argument("--lora_include_code_predictor", action="store_true",
                        help="Also attach LoRA to the code_predictor. This is the "
                             "original behaviour and it produces speech-shaped audio "
                             "with no recoverable words. For reproduction only.")
    # Fork addition. Every checkpoint this project has built is LoRA with
    # exclude_modules=".*code_predictor.*", so the acoustic-realisation layer has
    # never been trained at all. --full_ft skips get_peft_model entirely and
    # trains real weights. Absent, nothing below fires and the LoRA path is
    # bit-for-bit what it was.
    parser.add_argument("--offload_master", action="store_true",
                        help="EXP-013: hold fp32 master weights and AdamW moments "
                             "in system RAM. Forward and backward stay on the GPU; "
                             "only the gradient crossing and weight write-back use "
                             "the bus. Buys back the masters+moments share of VRAM "
                             "at an unmeasured PCIe cost.")
    parser.add_argument("--hybrid_code_predictor", action="store_true",
                        help="LoRA on the talker AND full-weight on the code "
                             "predictor, together. Not compatible with --full_ft.")
    parser.add_argument("--full_ft", action="store_true",
                        help="Full-weight fine-tune: no LoRA adapter, no merge "
                             "step, real parameters updated in place. See "
                             "--full_ft_scope and --full_ft_precision, both of "
                             "which change the answer more than the learning rate "
                             "does.")
    parser.add_argument("--full_ft_scope", type=str, default="all",
                        choices=["all", "talker", "code_predictor"],
                        help="Which parameters --full_ft unfreezes. 'code_predictor' "
                             "is 175M params (9.1%% of the model) and is the only "
                             "scope that fits a 12 GB card; it also REQUIRES "
                             "--sub_talker_weight > 0 or it trains the 16 codebook "
                             "embedding tables and nothing else.")
    parser.add_argument("--full_ft_precision", type=str, default="fp32",
                        choices=["fp32", "bf16_master_fp32", "bf16"],
                        help="Where the master weights live. 'fp32' loads the whole "
                             "model in float32 (7.2 GiB resident, simplest, the "
                             "A100 lane). 'bf16_master_fp32' keeps the model in "
                             "bf16 and the optimizer's copies in fp32 (2.4x less "
                             "memory, the workstation lane). 'bf16' is a measured "
                             "no-op below lr 1e-4 -- 3.9%% of weights move at 2e-6 "
                             "-- and exists only to reproduce that finding.")
    parser.add_argument("--code_predictor_lr_scale", type=float, default=1.0,
                        help="With --lora_include_code_predictor, train the code "
                             "predictor's adapters at lr * this. The codec head is "
                             "far more fragile than the talker: at full lr it "
                             "destroys intelligibility, so a small value (e.g. 0.05) "
                             "is the only defensible way to adapt it at all.")
    # Fork addition. A full epoch is ~40 minutes, which makes falsifying a
    # hypothesis expensive enough that people stop doing it. A few hundred steps
    # answers "does this configuration damage the model at all" for the price of a
    # coffee, and the checkpoint is written exactly as a full run would write it.
    parser.add_argument("--sub_talker_weight", type=float, default=0.3,
                        help="Weight on the code-predictor term. 0 trains the talker "
                             "alone, which isolates whether that term is what damages "
                             "generation. Upstream default is 0.3.")
    parser.add_argument("--checkpoint_every_epochs", type=int, default=1,
                        help="Write a full merged checkpoint every N epochs. Each is "
                             "~4.3GB for the 1.7B model, so 0 (only final_merged) is "
                             "the right setting for any multi-epoch run.")
    # Fork addition. Every export used to unconditionally convert to custom_voice,
    # bake one speaker at codec index 3000, and drop speaker_encoder -- correct for
    # a terminal deliverable, fatal for a checkpoint meant to seed a further
    # training stage: the corpus-pretrain -> speaker-specialize chain crashed with
    # `'NoneType' object is not callable` at the speaker_encoder() call below,
    # because stage one's export had already deleted those weights. This flag
    # keeps the checkpoint as a working base model instead: no forced
    # custom_voice, no dropped speaker_encoder, no baked embedding.
    parser.add_argument("--keep_speaker_encoder", action="store_true",
                        help="Export a reusable base checkpoint instead of a baked "
                             "custom_voice deliverable: keep speaker_encoder weights, "
                             "leave tts_model_type as the source model declared it, "
                             "and skip injecting a speaker embedding. Use this for a "
                             "checkpoint that will be --init_model_path for a further "
                             "training stage.")
    parser.add_argument("--checkpoint_every_steps", type=int, default=0,
                        help="write a RESUME checkpoint every N optimiser steps, 0 to "
                             "disable. This is NOT the exported artifact: it is the raw "
                             "model state plus a small JSON, written outside the "
                             "copytree/config-rewrite/embedding-injection path that "
                             "produces shippable checkpoints, because a mid-run "
                             "checkpoint never needs to be shippable and that path is "
                             "where every artifact this project ships comes from. Only "
                             "the newest is kept.")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="a step-<N> directory written by --checkpoint_every_steps. "
                             "Restores weights, optimiser-step count and the LR schedule "
                             "position. It does NOT restore AdamW moments or the fp32 "
                             "masters -- see the RESUME COST instrumentation, which "
                             "measures what that costs instead of asserting it is small.")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Stop after this many optimizer steps (0 = full epochs). "
                             "The LR schedule is sized to the truncated length, so a "
                             "short probe still warms up and anneals properly.")
    # TASK-PARTICLE-DPO-001, the preference path. Absent --preference_jsonl none of it
    # fires and the SFT path is what it was. What each value means is in
    # docs/okf/planning/task-particle-dpo-001.md, "The frozen objective".
    parser.add_argument("--preference_jsonl", type=str, default=None,
                        help="Train on preference pairs instead of --train_jsonl: one row "
                             "per clip with pair_id, role, text and audio_codes, plus the "
                             "reference scores precompute_ref_logp.py adds. See "
                             "preference_data.py.")
    parser.add_argument("--dpo_beta", type=float, default=None,
                        help="beta in the margin. Fixed at registration, so no default.")
    parser.add_argument("--sft_anchor_weight", type=float, default=None,
                        help="lambda on the chosen-clip anchor, which keeps the chosen "
                             "clips' likelihood, and with it termination, from being "
                             "driven down. Fixed at registration, so no default.")
    parser.add_argument("--residual_weight", type=float, default=0.0,
                        help="w_res. 0 is arm A: codebook 0 and EOS only. Above 0 is arm "
                             "B, which also scores codebooks 1-15 at this weight.")
    parser.add_argument("--preference_weight", type=float, default=1.0,
                        help="Weight on the preference term. 0 leaves only the anchor: "
                             "the matched chosen-only SFT comparator (readiness gate 6).")
    parser.add_argument("--pair_frame_tolerance", type=int, default=2,
                        help="Keep a pair only if its two clips differ by at most this "
                             "many frames. The task's 2 (160 ms): tighten it at "
                             "registration, never loosen it after results.")
    parser.add_argument("--pair_shuffle_seed", type=int, default=0,
                        help="Seed of the pair order. Pairs are shuffled whole, never split.")
    parser.add_argument("--mlflow_tracking_uri", type=str, default="http://localhost:5000")
    # Owner, 2026-10-23: "If we ship, it means we don't have a reproducible recipe." Two trainings of one recipe on
    # near-identical data lost 132 and 158 last characters (EXP-102/104: evaluation noise is about 4.5), and the
    # shuffle was unseeded. --seed fixes the Python, NumPy and torch generators before the model loads (so the LoRA
    # init is fixed) and gives the DataLoader its own seeded generator (so the order is fixed). Absent, nothing
    # changes: the old unseeded behaviour is kept byte for byte.
    parser.add_argument("--seed", type=int, default=None,
                        help="fix the training order and LoRA init; unset keeps the old unseeded behaviour")
    parser.add_argument("--fast_kernels", action="store_true",
                        help="with --seed: keep the fast (non-deterministic) GPU kernels. The order and init stay "
                             "fixed; two runs then differ by about 0.0005 after 30 steps instead of 0.0. On the "
                             "language set exact kernels cost about 60%% per step (1.41 s against 0.87): for "
                             "exploration; a release recipe trains exact")

    args = parser.parse_args()
    if args.seed is not None:
        import random
        import numpy as _np
        random.seed(args.seed)
        _np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    if args.seed is not None and not args.fast_kernels:
        # Seeding alone left two 30-step runs 0.0006 apart (unseeded: 0.046): GPU kernels that sum in a varying
        # order. Deterministic algorithms remove that; an op with no deterministic version warns rather than fails.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Still 0.0005 apart after that: the flash and memory-efficient SDPA backward kernels are not deterministic
        # (torch says so). The math backend is; it is slower and uses more memory.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    # THE RECIPE REPOSITORY TRAINS LoRA SFT ONLY. This file is the lab's trainer unchanged apart from this guard:
    # it is the one module whose pruning could not be checked here without a GPU run, so its full-fine-tune and
    # preference (DPO) paths stay in the source, unreachable. Their helper modules (preference_*.py) are not shipped,
    # so the flags are refused here, before anything loads, rather than failing on an import halfway into a run.
    for _flag, _on in (("--full_ft", args.full_ft), ("--hybrid_code_predictor", args.hybrid_code_predictor),
                       ("--offload_master", args.offload_master), ("--preference_jsonl", bool(args.preference_jsonl))):
        if _on:
            parser.error(f"{_flag} is not part of the recipe (LoRA SFT only); it lives in the lab's trainer")
    if getattr(args, "hybrid_code_predictor", False) and args.full_ft:
        parser.error("--hybrid_code_predictor and --full_ft are mutually exclusive: "
                     "hybrid IS the LoRA path with the code predictor additionally "
                     "unfrozen, so --full_ft would bypass the LoRA setup entirely.")
    pref = bool(args.preference_jsonl)
    if not pref and not args.train_jsonl:
        parser.error("--train_jsonl is required (or --preference_jsonl, for the "
                     "preference path)")
    if pref:
        if args.train_jsonl:
            parser.error("--preference_jsonl replaces --train_jsonl; pass one of them")
        if args.dpo_beta is None or args.sft_anchor_weight is None:
            parser.error("the preference path needs --dpo_beta and --sft_anchor_weight. "
                         "Both are fixed at registration, so neither has a default.")
        if args.full_ft:
            parser.error("--preference_jsonl is built and tested for LoRA, with or "
                         "without --hybrid_code_predictor, not for --full_ft")
        if args.keep_speaker_encoder:
            parser.error("--preference_jsonl exports a baked voice: every export writes "
                         "the starting v back at row 3000, which --keep_speaker_encoder "
                         "would skip")
    if args.speaker_map and not args.speakers_from_jsonl:
        parser.error("--speaker_map names the voices of --speakers_from_jsonl; without it "
                     "this phase bakes the one voice --speaker_name")
    if args.speakers_from_jsonl:
        if pref:
            parser.error("--speakers_from_jsonl is an SFT speaker phase; the preference path "
                         "scores every row under the one voice already at row 3000")
        if args.keep_speaker_encoder:
            parser.error("--keep_speaker_encoder exports a reusable base with NO baked "
                         "voice, so there is nothing for --speakers_from_jsonl to bake")

    # THE VOICE PLAN, before the model loads, so a refused map costs a second rather than
    # a minute. One voice unless --speakers_from_jsonl; the preference path's one voice is
    # v_pref, written back at the same row 3000.
    if args.speakers_from_jsonl:
        try:
            voice_plan = vb.plan_from_jsonl(args.train_jsonl, args.speaker_map)
        except vb.VoicePlanError as exc:
            raise SystemExit(str(exc))
        print("VOICES | " + ", ".join(
            f"row {voice_plan.slots[n]} = {n} ({sid}, {voice_plan.rows_in_data[sid]} rows)"
            for sid, n in voice_plan.voice_of.items()), flush=True)
    else:
        voice_plan = vb.single_plan(args.speaker_name)

    def _driver_says() -> str:
        """Which of the two failures this is, named at the moment it happens.

        "no CUDA device" has two causes with two different fixes: the device is gone,
        or the container cannot reach a device that is still there. On EXP-021 the
        HOST reported a healthy L4 while the container saw nothing -- twice. Without
        this line the refusal sends the next reader to check a driver that was never
        the problem, which is where the last six hours went.
        """
        import subprocess as _sp
        try:
            r = _sp.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                        capture_output=True, text=True, timeout=30)
        except Exception as exc:
            return ("nvidia-smi will not run in here either (%s) -- the device is gone"
                    % type(exc).__name__)
        if r.returncode != 0:
            return ("nvidia-smi exits %d in here: %s -- THE CONTAINER LOST DEVICE"
                    " ACCESS; the host driver is not the problem"
                    % (r.returncode, (r.stderr or r.stdout).strip()[:80]))
        seen = (r.stdout or "").strip().splitlines()
        if seen:
            return ("but nvidia-smi in here still reports %s -- the device is THERE"
                    " and torch cannot initialise against it" % seen[0].strip())
        return "nvidia-smi runs in here but lists no GPU"

    # A CPU FALLBACK IS NOT A DEGRADED RUN, IT IS A WRONG RUN. accelerate selects the
    # device, and with no CUDA it selects CPU and trains happily -- no error, no
    # warning, a plausible loss curve and a valid checkpoint at the end.
    #
    # EXP-021 hit this on a cloud lane. The language phase ran on the GPU at 0.44
    # s/step. The speaker phase that followed ran at 44.2 s/step -- exactly one hundred
    # times slower -- while the host reported 0% utilisation, 0 MiB used and no compute
    # processes: the container had lost the GPU between phases. Nothing failed. Had the
    # phase been short enough to finish, it would have produced a checkpoint, reported
    # success, and cost a hundred times the GPU-hours it should have.
    #
    # check_bootstrap.py cannot catch this: it runs ONCE before the lane, and each
    # phase is a separate process that can lose the device after the gate has passed.
    # The only place that sees every phase is here.
    #
    # QWEN_ALLOW_CPU_TRAIN=1 is the deliberate escape -- a tiny smoke test on a laptop
    # is a real use -- and it must be set on purpose, not arrived at by accident.
    if not torch.cuda.is_available() and os.environ.get("QWEN_ALLOW_CPU_TRAIN") != "1":
        raise SystemExit(
            "REFUSING TO TRAIN ON CPU. torch reports no CUDA device.\n"
            "  %s\n" % _driver_says() +
            "  This is a fine-tune of a 1.7B model: CPU is not a slow path, it is\n"
            "  a hundredfold slower path that still produces a checkpoint, so it\n"
            "  fails silently as an enormous bill rather than loudly as an error.\n"
            "  If the host has a GPU, the container has lost access to it -- check\n"
            "  nvidia-smi on the host and 'docker run --gpus all' in a container.\n"
            "  To train on CPU on purpose, set QWEN_ALLOW_CPU_TRAIN=1.")

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment("Qwen3-TTS-LoRA-Finetune")
    mlflow.config.enable_async_logging()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
    )

    # A run whose name says "lora-r8" but which trained 1.9B real weights would
    # poison the ledger, and the ledger is how this project decides anything. Likewise a
    # multi-voice run is not named after a --speaker_name it does not bake.
    _voice_tag = (f"voices{len(voice_plan.slots)}" if voice_plan.multi
                  else args.speaker_name)
    if args.full_ft:
        run_name = (f"fullft-{args.full_ft_scope}-{args.full_ft_precision}"
                    f"-{_voice_tag}-bs{args.batch_size}-lr{args.lr:g}")
    else:
        run_name = (f"lora-r{args.lora_rank}-{_voice_tag}"
                    f"-bs{args.batch_size}-warmup{args.warmup_ratio}")
    if pref:
        run_name = (f"pref-b{args.dpo_beta:g}-lam{args.sft_anchor_weight:g}"
                    f"-wres{args.residual_weight:g}-{run_name}")

    with mlflow.start_run(run_name=run_name):
        if accelerator.is_main_process:
            mlflow.log_params(vars(args))

        # === Model loading ===
        # bf16 unless --full_ft_precision fp32, which needs float32 parameters
        # resident because that is where its master weights live. The dtype is kept
        # UNIFORM across every module on purpose: this training loop calls
        # model.talker(...) and model.speaker_encoder(...) directly rather than
        # model(...), so accelerate's bf16 autocast -- which it installs by
        # wrapping the prepared model's own .forward -- never fires here. Nothing
        # casts anything. A mixed-dtype model (say an fp32 code_predictor feeding a
        # bf16 talker through the codebook loop) would therefore hit a raw dtype
        # mismatch inside the talker, and the cheaper way to have fp32 masters
        # without that hazard is --full_ft_precision bf16_master_fp32.
        load_dtype = (torch.float32
                      if args.full_ft and args.full_ft_precision == "fp32"
                      else torch.bfloat16)
        qwen3tts = Qwen3TTSModel.from_pretrained(
            args.init_model_path,
            dtype=load_dtype,
            attn_implementation=args.attn_implementation,
        )
        config = AutoConfig.from_pretrained(args.init_model_path)

        # R1: the preference path's voice is the starting weights' row 3000, read once
        # here, before anything can touch it. Every row is scored under it, nothing
        # accumulates a new mean, and both export branches write it back unchanged
        # (readiness gate 7). Imported here so the SFT path's imports are what they were.
        v_pref = None
        if pref:
            import preference_data as pdata
            import preference_forward as pf
            import preference_scoring as pscore
            v_pref = pf.voice_vector(qwen3tts.model)

        # === LoRA setup ===
        # target_modules matches by NAME SUFFIX, so it attaches an adapter to every
        # q_proj/k_proj/... anywhere in the model. Qwen3-TTS is not a plain
        # decoder-only LLM: it has a talker stack AND a code_predictor that share
        # those layer names. The stock list therefore adapts the codec head too.
        #
        # The code_predictor emits the 16 residual codebook tokens the vocoder turns
        # into waveform, so it is the most fragile thing LoRA can reach here, and it
        # took the largest perturbation of any module (0.027 against the talker's
        # 0.016). Excluding it is defensible on those grounds alone.
        #
        # It is NOT established that this is sufficient. An earlier claim that
        # restoring the code predictor produced a character-perfect transcript came
        # from a single sample of a STOCHASTIC decoder and does not replicate: four
        # generations of one line from one checkpoint gave four different mangled
        # transcripts. A run with the code predictor excluded from the start was
        # still unintelligible across 8 of 8 items. Measure with do_sample=False.
        exclude = None if args.lora_include_code_predictor else r".*code_predictor.*"
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            exclude_modules=exclude,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        # lora_config above is built unconditionally and is simply inert under
        # --full_ft; it is a config object, not a model, and leaving it in place
        # keeps the LoRA path's long comment attached to the code it explains.
        if args.full_ft:
            # No wrapper at all. `train_model` is the bare
            # Qwen3TTSForConditionalGeneration, so every downstream use --
            # accelerator.prepare, named_parameters, state_dict -- sees the natural
            # module tree and the natural key names. That is what makes the save
            # path below almost identical to the LoRA one: merge_and_unload()
            # returns this same object, so a merged-LoRA checkpoint and an FFT
            # checkpoint have identical keys by construction.
            train_model = qwen3tts.model
            n_train, n_total = apply_full_ft_scope(train_model, args.full_ft_scope)
            if accelerator.is_main_process:
                accelerator.print(
                    f"FULL FINE-TUNE | scope={args.full_ft_scope} "
                    f"precision={args.full_ft_precision} dtype={load_dtype} | "
                    f"trainable {n_train:,} / {n_total:,} "
                    f"({100 * n_train / n_total:.2f}%)")
                if args.full_ft_scope == "code_predictor" and args.sub_talker_weight == 0:
                    # Loud rather than fatal: the run is still coherent, it just
                    # answers a much smaller question than the flag implies.
                    accelerator.print(
                        "WARNING: --full_ft_scope code_predictor with "
                        "--sub_talker_weight 0. The code predictor's transformer "
                        "and output head are reached ONLY through "
                        "forward_sub_talker_finetune, whose term is scaled by that "
                        "weight. At 0 this run trains the 16 codebook embedding "
                        "tables and leaves 100% of the predictor's compute frozen. "
                        "Upstream uses 0.3.")
                mlflow.log_params({
                    "full_ft_trainable_params": n_train,
                    "full_ft_total_params": n_total,
                    "full_ft_load_dtype": str(load_dtype),
                })
        else:
            train_model = get_peft_model(qwen3tts.model, lora_config)

            # === Hybrid: LoRA talker + full-weight code predictor ===
            # The two were mutually exclusive, and the exclusion cost us a lever.
            # Training the code predictor ALONE at 2e-6 destroys termination: the
            # talker is left bit-identical and still decides when to stop, but the
            # predictor's output feeds back into its context autoregressively, so a
            # frozen, still-correct talker is handed a state distribution it has
            # never seen and never reaches the stop condition. At 2e-7 that is still
            # 30% of clips against a 5% control floor.
            #
            # If desynchronisation is the mechanism, letting the talker ADAPT to the
            # shifted inputs should be the fix -- which needs both trained at once.
            # LoRA keeps the talker's capacity small and cheap (~0.45% of params) so
            # this still fits the 12 GB card, where a full talker scope does not.
            #
            # Use --code_predictor_lr_scale to run the predictor slower than the
            # adapters; the optimizer already splits them into separate groups.
            if args.hybrid_code_predictor:
                n_cp = 0
                for name, prm in qwen3tts.model.named_parameters():
                    if "code_predictor" in name and "lora_" not in name:
                        prm.requires_grad_(True)
                        n_cp += 1
                if accelerator.is_main_process:
                    accelerator.print(
                        f"HYBRID | LoRA talker + full-weight code predictor: "
                        f"{n_cp} code-predictor tensors unfrozen")
            if accelerator.is_main_process:
                train_model.print_trainable_parameters()

        # === Gradient checkpointing (fork addition, opt-in) ===
        # Applied to the talker: that is the module the training loop runs and
        # where the LoRA adapters attach.
        #
        # Enabled AFTER get_peft_model, not before, and the ordering is load
        # bearing. PeftModel.__init__ inspects is_gradient_checkpointing and, if
        # set, calls enable_input_require_grads() -> get_input_embeddings(), which
        # Qwen3TTSForConditionalGeneration does not implement. Enabling first
        # therefore raises NotImplementedError before training starts.
        #
        # use_reentrant=False is likewise required rather than preferred: the
        # reentrant autograd path needs an input carrying requires_grad to trigger
        # the backward recompute, and this loop feeds inputs_embeds built from
        # frozen embedding tables, so reentrant would silently produce no gradients.
        # That is also what makes skipping enable_input_require_grads safe here.
        #
        # Under --full_ft the ordering constraint dissolves, and it is worth being
        # explicit about why rather than leaving the call site looking load-bearing
        # when it is not. There is no PeftModel.__init__ in the full-weight path,
        # so nothing ever calls enable_input_require_grads(), so nothing ever
        # reaches the unimplemented get_input_embeddings(). This block could run
        # before or after the branch above and behave identically. It stays here so
        # both paths read the same.
        #
        # use_reentrant=False remains correct but for a different reason, and the
        # original reason has actually gone away. The reentrant path needs an input
        # tensor carrying requires_grad; the LoRA path had none because
        # inputs_embeds is built from frozen embedding tables. Under --full_ft at
        # least one embedding table in that construction is trainable in every
        # scope -- talker.model.text_embedding under `all`/`talker`, and
        # code_predictor.get_input_embeddings() under `code_predictor` -- so
        # input_embeddings does carry requires_grad and reentrant would in fact
        # work. Non-reentrant is kept anyway: it is the supported path, it does not
        # depend on that argument holding for every future scope, and it is what
        # the measured +50 to +66% step-time figure was taken against.
        #
        # It still pays under --full_ft_scope code_predictor, which is not obvious:
        # that scope freezes every talker parameter, but input_embeddings is built
        # from a TRAINABLE code-predictor embedding table, so backward must still
        # traverse the entire talker to reach it and its activations are still
        # retained. Freezing weights does not free activations when something
        # upstream of them needs a gradient. What checkpointing does NOT cover in
        # any scope is the code predictor itself, which is reached through
        # forward_sub_talker_finetune rather than through the talker's decoder
        # layers, so its activations are stored regardless.
        if args.gradient_checkpointing:
            target = qwen3tts.model.talker
            if not hasattr(target, "gradient_checkpointing_enable"):
                raise RuntimeError(
                    "talker does not expose gradient_checkpointing_enable; "
                    "the submodule's model layout has changed"
                )
            target.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            accelerator.print("Gradient checkpointing enabled on the talker (non-reentrant)")

        # Dataset
        if pref:
            # One pair per batch, never split; pairs outside the frame tolerance are
            # excluded and counted; and every reference score must have been computed
            # under this run's identity (weights, voice, codes, text, w_res), or the run
            # refuses rather than comparing against a stale reference.
            train_data = pdata.load_rows(args.preference_jsonl)
            dataset = pdata.PairDataset(train_data, qwen3tts.processor, config)
            filt = pdata.frame_filter(train_data, pdata.pair_indices(train_data, need_ref=True),
                                      args.pair_frame_tolerance)
            kept = [p["pair"] for p in filt.kept]
            if not kept:
                raise SystemExit(f"no pair is within {args.pair_frame_tolerance} frames; "
                                 f"|F+ - F-| histogram {dict(sorted(filt.histogram.items()))}")
            _weights = pf.checkpoint_digest(args.init_model_path)
            _voice = pf.voice_digest(v_pref)
            stale = pdata.stale_refs(train_data, [i for p in kept for i in p], dataset.text_ids,
                                     weights=_weights, voice=_voice, w_res=args.residual_weight)
            if stale:
                raise SystemExit(
                    f"{len(stale)} of {2 * len(kept)} rows carry a reference score computed "
                    f"under a different identity (weights, voice, codes, text or w_res). "
                    f"Re-run precompute_ref_logp.py on --init_model_path with "
                    f"--residual-weight {args.residual_weight:g}. First: row {stale[0]}")
            pair_sampler = pdata.PairBatchSampler(kept, seed=args.pair_shuffle_seed)
            train_dataloader = DataLoader(dataset, batch_sampler=pair_sampler,
                                          collate_fn=dataset.collate_fn)
            if accelerator.is_main_process:
                accelerator.print(
                    f"PREFERENCE | {len(kept)} pairs kept, {len(filt.excluded)} excluded at "
                    f"{filt.tolerance} frames, |F+ - F-| {dict(sorted(filt.histogram.items()))}"
                    f" | v = codec_embedding[3000] of --init_model_path, sha256 {_voice[:12]}"
                    f" | beta {args.dpo_beta:g} lambda {args.sft_anchor_weight:g} "
                    f"w_res {args.residual_weight:g} pref_weight {args.preference_weight:g}"
                    f" | --batch_size and --sub_talker_weight are not used on this path")
                # THE PAIR-FILTER RECORD GOES BESIDE THE EXPORTS (2026-10-02). With a
                # sqlite tracking URI and no artifact root, log_dict writes to ./mlruns
                # under the CURRENT DIRECTORY, not beside the database. The 2026-09-29 dry
                # run left three such files in a worktree that was later deleted. The
                # 2026-10-01 rehearsal's run points into another worktree. This list of
                # pairs the frame tolerance excluded is the evidence for the primary's
                # sample, so it is written into --output_model_path with the exports. Its
                # scalars go into params and a tag, which live in the database itself.
                # The artifact is still logged, as a convenience copy.
                _pf = {"tolerance": filt.tolerance, "kept": len(kept),
                       "excluded": len(filt.excluded),
                       "frame_diff_histogram": {str(k): v for k, v in
                                                sorted(filt.histogram.items())},
                       "excluded_pairs": [p["pair"] for p in filt.excluded]}
                os.makedirs(args.output_model_path, exist_ok=True)
                _pf_path = os.path.abspath(os.path.join(args.output_model_path,
                                                        "pref_pair_filter.json"))
                with open(_pf_path, "w", encoding="utf-8", newline="\n") as _fh:
                    json.dump(_pf, _fh, indent=1)
                    _fh.write("\n")
                mlflow.log_params({"pref_pairs_kept": len(kept),
                                   "pref_pairs_excluded": len(filt.excluded),
                                   "pref_pair_tolerance": filt.tolerance,
                                   "pref_pair_filter_path": _pf_path,
                                   "pref_voice_sha256": _voice,
                                   "pref_init_digest": _weights})
                mlflow.set_tag("pref_frame_diff_histogram",
                               json.dumps(_pf["frame_diff_histogram"], separators=(",", ":")))
                mlflow.log_dict(_pf, "pref_pair_filter.json")
        else:
            train_data = [json.loads(line) for line in open(args.train_jsonl).readlines()]
            dataset = TTSDataset(train_data, qwen3tts.processor, config)
            if voice_plan.multi:
                # Each batch also carries its rows' voices. Same length, so the same
                # shuffle; the base collate output is unchanged but for that one key.
                dataset = vb.VoiceTaggedDataset(dataset, voice_plan.voice_indices(train_data))
                if accelerator.is_main_process:
                    # THE ROW ORDER IS RECORDED beside the exports, as the preference
                    # path records its pair filter, and in the run's params and tags.
                    os.makedirs(args.output_model_path, exist_ok=True)
                    _vs_path = os.path.abspath(os.path.join(args.output_model_path,
                                                            "voice_slots.json"))
                    with open(_vs_path, "w", encoding="utf-8", newline="\n") as _fh:
                        json.dump(voice_plan.record(), _fh, indent=1, ensure_ascii=False)
                        _fh.write("\n")
                    mlflow.log_params({"voice_count": len(voice_plan.slots),
                                       "voice_slots_path": _vs_path})
                    mlflow.set_tag("voice_slots", json.dumps(voice_plan.slots,
                                                             separators=(",", ":")))
            train_dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=dataset.collate_fn,
                **({"generator": torch.Generator().manual_seed(args.seed)} if args.seed is not None else {}),
            )

        # Separate parameter groups so the code predictor can be adapted at a much
        # smaller rate than the talker. get_linear_schedule_with_warmup wraps each
        # group's own initial_lr, so the ratio is preserved across warmup and decay.
        trainable = [(n, p) for n, p in train_model.named_parameters() if p.requires_grad]
        cp = [p for n, p in trainable if "code_predictor" in n]
        rest = [p for n, p in trainable if "code_predictor" not in n]
        # `if rest` guards --full_ft_scope code_predictor, where the talker group is
        # empty. Inert for LoRA, where `rest` is never empty.
        groups = [{"params": rest, "lr": args.lr}] if rest else []
        if cp:
            groups.append({"params": cp, "lr": args.lr * args.code_predictor_lr_scale})
        if not groups:
            raise RuntimeError(
                f"no trainable parameters (full_ft={args.full_ft}, "
                f"scope={getattr(args, 'full_ft_scope', None)})")
        # See MasterWeightAdamW's docstring for the measurement that makes this a
        # correctness branch rather than a tuning knob: in pure bf16 at the official
        # full-fine-tune rate of 2e-6, 96% of weights never change value at all.
        # Hybrid needs the master weights just as much as --full_ft does, and for
        # the same reason: its code-predictor group trains FULL-WEIGHT bf16 tensors,
        # and at the small rates this layer tolerates, a plain AdamW step is far
        # below bf16's smallest representable change -- 96% of weights never move,
        # while the loss still falls because the other 4% learn. Gating the master
        # weights on --full_ft alone would have made every hybrid run a near-no-op
        # wearing a convincing curve. The LoRA-only path is untouched: `cp` is empty
        # there, so this branch cannot fire.
        needs_master = (args.full_ft and args.full_ft_precision == "bf16_master_fp32")             or (getattr(args, "hybrid_code_predictor", False) and cp)
        if needs_master:
            optimizer = MasterWeightAdamW(
                groups, lr=args.lr, weight_decay=0.01,
                offload_master=getattr(args, "offload_master", False))
            if getattr(args, "offload_master", False) and accelerator.is_main_process:
                # EXP-013 names a void condition for exactly this: "the offload arm
                # silently falls back to keeping shadows on the GPU -- the device of
                # the master tensors must be asserted and logged, not assumed."
                devs = {str(m.device) for _, m in optimizer._pairs}
                accelerator.print(f"offload_master: masters on {sorted(devs)}, "
                                  f"{len(optimizer._pairs)} tensors")
                assert devs == {"cpu"}, f"offload requested but masters on {devs}"
        else:
            optimizer = AdamW(groups, lr=args.lr, weight_decay=0.01)
        if accelerator.is_main_process:
            accelerator.print(
                f"optimizer groups: talker={len(rest)} params @ {args.lr:g}"
                + (f", code_predictor={len(cp)} params @ "
                   f"{args.lr * args.code_predictor_lr_scale:g}" if cp else
                   ", code_predictor excluded"))

        # === Prepare with Accelerator ===
        model, optimizer, train_dataloader = accelerator.prepare(
            train_model, optimizer, train_dataloader
        )

        # === Linear Warmup Scheduler (prevents early divergence) ===
        # total_steps must count OPTIMIZER steps, not dataloader batches.
        # scheduler.step() is called under `if accelerator.sync_gradients`, so it
        # fires once per gradient_accumulation_steps batches. Sizing the schedule
        # by len(dataloader) therefore stretches it by that factor: with accum=6 and
        # 8325 batches, warmup consumed 60% of the run and the decay covered 7% of
        # its intended span, so training ended at 1.85e-4 against a 2e-4 peak --
        # essentially un-annealed. Verified against the run's logged LR curve.
        steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        total_steps = steps_per_epoch * args.num_epochs
        if args.max_steps:
            # Size the schedule to the truncated run, so a short probe still warms
            # up and anneals rather than stopping mid-warmup at peak LR.
            total_steps = min(total_steps, args.max_steps)
        warmup_steps = int(args.warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        scheduler = accelerator.prepare(scheduler)

        model.train()

        global_step = 0
        optimizer_steps = 0
        # === RESUME, AND THE INSTRUMENTATION THAT PRICES IT ===
        # resume_baseline carries the loss the interrupted run was sitting at when it
        # checkpointed. Everything after a resume is compared against it, so the cost of
        # restarting AdamW's moments and the fp32 masters is a MEASURED number in the
        # log rather than a claim in a docstring. Without this the resume path would be
        # exactly the kind of thing that looks fine and is not.
        resume_baseline = None
        resume_at_step = 0
        resume_reported = [False]
        # The trailing loss window a checkpoint records, so the run that resumes from it
        # knows what it was sitting at. A single last-step loss is too noisy at batch 1
        # to be a baseline; the mean of the last 20 optimiser steps is not.
        _recent_losses = []
        if args.resume_from:
            import safetensors.torch as _st
            rs_path = os.path.join(args.resume_from, "resume_state.json")
            if not os.path.isfile(rs_path):
                raise SystemExit(
                    f"--resume_from {args.resume_from} has no resume_state.json. A "
                    f"step-N directory is only resumable if it was written completely; "
                    f"the writer renames into place precisely so a half-written one "
                    f"cannot be found here.")
            with open(rs_path, encoding="utf-8") as fh:
                rs = json.load(fh)
            sd = _st.load_file(os.path.join(args.resume_from, "model.safetensors"))
            missing, unexpected = accelerator.unwrap_model(model).load_state_dict(
                sd, strict=False)
            if missing or unexpected:
                # Never silent. A resume that loads 90% of the tensors and trains on
                # the rest is the failure this whole path exists to avoid.
                raise SystemExit(
                    f"resume state does not match the model: {len(missing)} missing, "
                    f"{len(unexpected)} unexpected. First missing: {list(missing)[:3]}")
            optimizer_steps = int(rs["optimizer_steps"])
            resume_at_step = optimizer_steps
            resume_baseline = rs.get("recent_loss_mean")
            # Fast-forward the LR schedule. The scheduler is stepped once per optimiser
            # step, so replaying it is how its position is restored -- there is no
            # setter, and leaving it at step 0 would restart warmup and undo the anneal.
            for _ in range(optimizer_steps):
                scheduler.step()
            accelerator.print(
                f"RESUME loaded step={optimizer_steps} "
                f"lr={scheduler.get_last_lr()[0]:.3e} "
                f"baseline_loss={resume_baseline} "
                f"| AdamW moments and fp32 masters NOT restored -- "
                f"RESUME_COST lines below measure what that cost")
        # EXP-013 step-time state. A list plus a one-element list for the
        # previous timestamp, so the closure below needs no `nonlocal`.
        _step_times = []
        _last_step = [time.perf_counter()]
        stop_early = False
        # One running mean per voice in the plan; a single-voice phase holds one.
        voice_means = vb.VoiceMeans(voice_plan.names)
        if args.resume_from:
            # A phase that bakes a voice must continue its running means, or refuse (PR #161 review).
            accelerator.print("RESUME " + vb.resume_voice_means(
                args.resume_from, voice_means, bakes=not pref and not args.keep_speaker_encoder))

        def _bake_rows(state_dict):
            """Both export sites bake through here. The preference path writes its
            starting v back at row 3000 (readiness gate 7); every SFT phase writes each
            voice's running mean at its row."""
            vectors = ({n: v_pref for n in voice_plan.slots} if pref
                       else voice_means.vectors())
            for _name, _slot in vb.bake_rows(state_dict, voice_plan.slots, vectors):
                print(f"✓ Speaker embedding injected at index {_slot}"
                      + (f" ({_name}, mean over {voice_means.count[_name]} clips)"
                         if voice_plan.multi else ""))

        for epoch in range(args.num_epochs):
            if accelerator.is_main_process:
                print(f"\n=== Starting Epoch {epoch} ===")

            acct = EpochAccounting()
            if pref:
                pair_sampler.set_epoch(epoch)
                pref_acc = [0.0, 0]

            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(model):
                    if pref:
                        # TASK-PARTICLE-DPO-001: one pair, [chosen, rejected], scored
                        # under the starting v by the same function the reference
                        # precompute used. No speaker encoder, and no running mean.
                        assert batch["roles"] == ["chosen", "rejected"], batch["roles"]
                        sc = pf.score_batch(model.talker, batch, v_pref, args.residual_weight)
                        pol = pscore.sequence_score(sc["s0"], sc["s1"], args.residual_weight)
                        ref = pscore.sequence_score(batch["ref_s0"].to(pol.device),
                                                    batch["ref_s1"].to(pol.device),
                                                    args.residual_weight)
                        loss, pst = pscore.objective(
                            pol[:1], pol[1:], ref[:1], ref[1:], sc["n0"][:1], sc["n1"][:1],
                            beta=args.dpo_beta, lam=args.sft_anchor_weight,
                            w_res=args.residual_weight, pref_weight=args.preference_weight)
                    else:
                        input_ids = batch['input_ids']
                        codec_ids = batch['codec_ids']
                        ref_mels = batch['ref_mels']
                        text_embedding_mask = batch['text_embedding_mask']
                        codec_embedding_mask = batch['codec_embedding_mask']
                        attention_mask = batch['attention_mask']
                        codec_0_labels = batch['codec_0_labels']
                        codec_mask = batch['codec_mask']

                        speaker_embedding = model.speaker_encoder(
                            ref_mels.to(model.device).to(model.dtype)
                        ).detach()

                        # THE BAKED VOICE IS THE MEAN OVER THE PHASE, NOT THE FIRST CLIP.
                        # This used to capture batch 0 and never update again. For a
                        # single-speaker phase that bakes one arbitrary clip's embedding:
                        # roughly the right voice, but carrying that clip's noise and
                        # room. For a BLENDED phase it is actively wrong -- the weights
                        # learn a mixture while the conditioning vector is whichever
                        # contributor the shuffle put first, so the shipped voice is a
                        # pseudonymous speaker chosen at random AND mismatched to what
                        # actually trained.
                        #
                        # The mean over the phase is what makes a blended voice possible:
                        # feed the speaker phase N similar speakers and index 3000 becomes
                        # their centroid, which belongs to none of them.
                        #
                        # Accumulated in fp32 deliberately. A bf16 running mean stops
                        # moving once the count passes ~256, because the increment falls
                        # under the 8-bit mantissa -- which would silently reduce this
                        # back to "the first few clips" on any real run.
                        #
                        # SEVERAL VOICES ARE THE SAME MEAN, ONCE PER VOICE. Under
                        # --speakers_from_jsonl each row folds into its own voice's mean;
                        # a batch whose rows are all one voice folds in whole. The
                        # arithmetic is VoiceMeans.add, which is the code that stood here.
                        emb = speaker_embedding.float()
                        voice_means.add_batch(emb, batch.get("voice_idx"))

                        input_text_ids = input_ids[:, :, 0]
                        input_codec_ids = input_ids[:, :, 1]

                        text_emb = model.talker.model.text_embedding(input_text_ids)
                        input_text_embedding = model.talker.text_projection(text_emb) * text_embedding_mask
                        input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                        input_codec_embedding[:, 6, :] = speaker_embedding

                        input_embeddings = input_text_embedding + input_codec_embedding

                        for i in range(1, 16):
                            codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](
                                codec_ids[:, :, i]
                            )
                            codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                            input_embeddings = input_embeddings + codec_i_embedding

                        # Score the logits directly. Do NOT hand `labels` to the
                        # talker.
                        #
                        # codec_0_labels[:, 1:] is already the correct target for
                        # inputs_embeds[:, :-1]: logits[i] should predict the token at
                        # i+1. The original author computed that shift correctly and
                        # then passed it as `labels`, and Qwen3TTSTalker.forward scores
                        # through HuggingFace's ForCausalLMLoss, which shifts a second
                        # time:
                        #     labels = pad(labels, (0, 1), value=-100)
                        #     shift_labels = labels[..., 1:]
                        # so the model was trained to predict the token AFTER next. Loss
                        # still fell -- t+2 is a perfectly learnable function -- while
                        # generation got worse the better it was learned, which is why no
                        # learning rate ever helped.
                        #
                        # Measured on the untrained base model, mean over 5 batches:
                        #     as-written  labels[:, 1:]    10.2322
                        #     manual CE   targets[:, 1:]    1.3668
                        #
                        # Passing labels[:, :-1] also avoids the double shift and is
                        # WRONG: with batch_size 1 the EOS token sits at the last
                        # position, so that form silently drops the only supervision for
                        # stopping and every clip ran to the token ceiling.
                        outputs = model.talker(
                            inputs_embeds=input_embeddings[:, :-1, :],
                            attention_mask=attention_mask[:, :-1],
                            output_hidden_states=True
                        )
                        talker_logits = outputs.logits
                        talker_loss = torch.nn.functional.cross_entropy(
                            talker_logits.float().reshape(-1, talker_logits.size(-1)),
                            codec_0_labels[:, 1:].reshape(-1),
                            ignore_index=-100,
                        )

                        hidden_states = outputs.hidden_states[0][-1]
                        # Keep only positions where BOTH p and p+1 are real codec frames.
                        #
                        # Without the second term, the last kept position of every clip is
                        # one whose p+1 is NOT a codec frame: measured on 8/8 clips that
                        # frame is all zeros and its codec_0_labels entry is 2150, the EOS
                        # token. Pairing it in supervises the code predictor to emit zeros
                        # for all 15 residual codebooks AT THE TERMINATION BOUNDARY -- the
                        # exact behaviour this whole investigation is about.
                        #
                        # It is ~1% of rows and costs only +0.04 nats overall (5.7130 vs
                        # 5.6738), so it would never have shown up as a bad loss curve. That
                        # row on its own scores 9.2877.
                        sub_mask = codec_mask[:, :-1] & codec_mask[:, 1:]
                        talker_hidden_states = hidden_states[sub_mask]

                        # A SECOND OFF-BY-ONE, in the frame pairing -- independent of the
                        # label double-shift handled below.
                        #
                        # The talker ran on input_embeddings[:, :-1, :] and talker_loss
                        # scores it against codec_0_labels[:, 1:], so hidden_states[p] is
                        # the state that predicts frame p+1. It was being paired with
                        # codec_ids[codec_mask], which is frame p -- conditioning each
                        # frame's residual codebooks on the state for the frame before it.
                        #
                        # The inference path settles which is correct: Qwen3TTSTalker.forward
                        # calls the code predictor with (past_hidden, last_id_hidden) -- the
                        # PREVIOUS step's state together with the codebook-0 token that state
                        # produced -- to generate that frame's codebooks 1-15. Training has to
                        # match, so the state at p pairs with the codec frame at p+1.
                        #
                        # Measured on identical rows, 46,588 frames, mean CE per head:
                        #     as-written (frame p)    6.9008
                        #     corrected  (frame p+1)  5.6439
                        #     marginal entropy floor  7.1787
                        # Head 0 is the tell. It predicts codebook 1, the most predictable of
                        # the fifteen (marginal 6.200), and scored 8.876 as-written -- worse
                        # than ignoring the model, and the worst of all 15 heads. Corrected it
                        # scores 2.857, and the heads become a proper residual-VQ ladder
                        # rising 2.86 -> 6.57 with every head beating its own marginal.
                        #
                        # Same mask on both sides, so the row count is unchanged.
                        talker_codec_ids = codec_ids[:, 1:][sub_mask]

                        sub_talker_logits, sub_talker_loss_buggy = model.talker.forward_sub_talker_finetune(
                            talker_codec_ids, talker_hidden_states
                        )

                        # THE LABELS ARE SHIFTED TWICE UPSTREAM. Do not use the returned loss.
                        #
                        # forward_sub_talker_finetune hands forward_finetune labels that are
                        # ALREADY shifted (codec_ids[:, 1:]), and forward_finetune then applies
                        # ForCausalLMLoss, which shifts again (logits[..., :-1, :] against
                        # labels[..., 1:]). The 15 heads are already aligned to the 15 residual
                        # codebooks -- head j emits codebook j+1 -- so the second shift trains
                        # every head on the NEXT codebook's targets and drops head 15 from the
                        # loss entirely.
                        #
                        # Same bug as ms-swift PR #9752 (merged 2026-07-20). It is still present
                        # in QwenLM/Qwen3-TTS main as of 022e286, so there is nothing to pull;
                        # and ms-swift is not a dependency here, so there is nothing to merge.
                        # It is fixed HERE rather than in the submodule because a working-tree
                        # edit to a gitlinked submodule is captured by no commit and would not
                        # survive a fresh clone or reach the GCP VM, which unpacks its own copy
                        # of the source from GCS.
                        #
                        # The tell: sub-talker loss logged 9.03-11.32 across our runs, against
                        # ln(2048) = 7.62 for uniform random. A pretrained module cannot score
                        # worse than chance on its own task unless the targets are misaligned.
                        #
                        # sub_talker_logits is (N, 15, V) and already lines up with
                        # talker_codec_ids[:, 1:]. Score them directly, with no shift.
                        sub_labels = talker_codec_ids[:, 1:]
                        sub_talker_loss = torch.nn.functional.cross_entropy(
                            sub_talker_logits.float().reshape(-1, sub_talker_logits.size(-1)),
                            sub_labels.reshape(-1),
                        )

                        loss = talker_loss + args.sub_talker_weight * sub_talker_loss

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()          # ← Warmup happens here
                        optimizer.zero_grad()
                        optimizer_steps += 1
                        _recent_losses.append(float(loss.detach()))
                        if len(_recent_losses) > 64:
                            del _recent_losses[:-64]
                        # === RESUME COST, MEASURED ONCE PER RESUME ===
                        # The first steps after a resume are the only place the price of
                        # not restoring AdamW's moments and the fp32 masters is visible.
                        # Reported as the loss against the baseline the interrupted run
                        # recorded, and again once it has recovered, so "the moments
                        # re-warm in tens of steps" becomes a number for THIS run rather
                        # than a general belief.
                        if resume_baseline is not None and not resume_reported[0] \
                           and accelerator.is_main_process:
                            _since = optimizer_steps - resume_at_step
                            _l = float(loss.detach())
                            if _since == 1:
                                accelerator.print(
                                    f"RESUME_COST step={optimizer_steps} since_resume=1 "
                                    f"loss={_l:.4f} baseline={resume_baseline:.4f} "
                                    f"delta={_l - resume_baseline:+.4f} "
                                    f"({100 * (_l - resume_baseline) / resume_baseline:+.1f}%)")
                            elif _l <= resume_baseline or _since >= 200:
                                accelerator.print(
                                    f"RESUME_COST step={optimizer_steps} "
                                    f"recovered_in={_since} steps loss={_l:.4f} "
                                    f"baseline={resume_baseline:.4f} "
                                    + ("(recovered)" if _l <= resume_baseline
                                       else "(NOT recovered within 200 steps -- the "
                                            "weights-only resume is costing more than "
                                            "the design assumed)"))
                                resume_reported[0] = True
                        if (args.checkpoint_every_steps
                                and optimizer_steps % args.checkpoint_every_steps == 0
                                and accelerator.is_main_process):
                            write_resume_checkpoint(
                                os.path.join(args.output_model_path, "resume"),
                                accelerator.unwrap_model(model), optimizer_steps, epoch,
                                scheduler.get_last_lr()[0], _recent_losses,
                                voice_means=voice_means)
                        # EXP-013. Nothing in this trainer measured how long a step
                        # takes, so a throughput question could only be answered by
                        # dividing total wall clock by the step count -- which buries
                        # a ~60 s model load in the number, the exact confound that
                        # made a GGUF benchmark unreadable until load was separated
                        # from generation. CUDA is asynchronous, so the timer has to
                        # synchronise or it measures queueing rather than work.
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        _now = time.perf_counter()
                        _step_times.append(_now - _last_step[0])
                        _last_step[0] = _now
                        if accelerator.is_main_process and optimizer_steps % 10 == 0:
                            _w = (_step_times[args.steptime_warmup:]
                                  or _step_times)
                            _srt = sorted(_w)
                            accelerator.print(
                                f"STEPTIME step={optimizer_steps} "
                                f"last={_step_times[-1]:.3f}s "
                                f"median_after_warmup={_srt[len(_srt)//2]:.3f}s "
                                f"peak_vram={torch.cuda.max_memory_allocated()/2**30:.2f}GiB"
                                if torch.cuda.is_available() else
                                f"STEPTIME step={optimizer_steps} last={_step_times[-1]:.3f}s")

                # ONCE PER MICRO-BATCH, and BEFORE the max_steps break so the batch
                # that completes the final optimiser step is counted. BUG-TRAIN-001:
                # this block sat indented under the epoch-level STEPTIMES print below,
                # so it ran once per epoch against the last micro-batch and every
                # "Avg Loss" and MLflow train_loss point was a single batch. Moved,
                # not rewritten; the optimiser path above is untouched.
                if accelerator.is_main_process and pref:
                    # EpochAccounting's three slots are plain sums; on this path they
                    # carry the preference term, the anchor and the margin.
                    current_loss = acct.add(loss, pst["pref"], pst["sft"], pst["margin_mean"])
                    global_step += 1
                    pref_acc[0] += float(pst["accuracy"])
                    pref_acc[1] += 1
                    # Item 5, every pair: both clips' scores as sums and per token, the
                    # rewards, the margin, and whether chosen beat rejected.
                    _den = (sc["n0"] + args.residual_weight * sc["n1"]).float()
                    _m = {"train_loss": current_loss,
                          "learning_rate": scheduler.get_last_lr()[0],
                          "pref_loss": float(pst["pref"]), "sft_loss": float(pst["sft"]),
                          "pref_margin": float(pst["margin_mean"]),
                          "pref_accuracy": float(pst["accuracy"]),
                          "pref_chosen_logp": float(pol[0]),
                          "pref_rejected_logp": float(pol[1]),
                          "pref_chosen_logp_per_token": float(pol[0] / _den[0]),
                          "pref_rejected_logp_per_token": float(pol[1] / _den[1]),
                          "pref_reward_chosen": args.dpo_beta * float(pol[0] - ref[0]),
                          "pref_reward_rejected": args.dpo_beta * float(pol[1] - ref[1])}
                    mlflow.log_metrics(_m, step=global_step)
                    if step % 10 == 0 or global_step <= 3:
                        accelerator.print(
                            f"PREF_STEP epoch={epoch} step={step} global_step={global_step} "
                            + " ".join(f"{k}={v!r}" for k, v in sorted(_m.items())))
                elif accelerator.is_main_process:
                    current_loss = acct.add(loss, talker_loss, sub_talker_loss,
                                            sub_talker_loss_buggy)
                    global_step += 1
                    if step % 10 == 0:
                        current_lr = scheduler.get_last_lr()[0]
                        accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {current_loss:.4f} | LR: {current_lr:.2e}")
                        mlflow.log_metric("train_loss", current_loss, step=global_step)
                        mlflow.log_metric("learning_rate", current_lr, step=global_step)

                if args.max_steps and optimizer_steps >= args.max_steps:
                    stop_early = True
                    break

            # THE RAW SERIES, so a median over any window can be recomputed later.
            # A summary statistic is only as good as the registration that named it,
            # and registrations have been wrong; the durations themselves are not.
            if accelerator.is_main_process and _step_times:
                accelerator.print(
                    "STEPTIMES " + ",".join(f"{t:.4f}" for t in _step_times))

            if stop_early:
                accelerator.print(
                    f"Stopping early at {optimizer_steps} optimizer steps "
                    f"(--max_steps {args.max_steps})")

            # === Calculate and log average loss ===
            means = acct.means()
            if means is not None:
                epoch_avg_loss = means["loss"]
                if accelerator.is_main_process and pref:
                    print(f"✅ Epoch {epoch} finished | Avg Loss: {epoch_avg_loss:.4f}"
                          f" over {acct.batches} pairs"
                          f" | pref {means['talker']:.4f} | sft {means['sub']:.4f}"
                          f" | margin {means['sub_buggy']:+.4f}"
                          f" | accuracy {pref_acc[0] / max(pref_acc[1], 1):.3f}")
                    mlflow.log_metric("epoch_avg_loss", epoch_avg_loss, step=epoch)
                elif accelerator.is_main_process:
                    print(f"✅ Epoch {epoch} finished | Avg Loss: {epoch_avg_loss:.4f}"
                          f" over {acct.batches} micro-batches"
                          f" | talker {means['talker']:.4f}"
                          f" | sub_talker {means['sub']:.4f}"
                          f" (double-shifted would read"
                          f" {means['sub_buggy']:.4f}; random is 7.62)")
                    mlflow.log_metric("epoch_avg_loss", epoch_avg_loss, step=epoch)

            # ==================== SAVE FULL MERGED CHECKPOINT EVERY EPOCH ====================
            # Each of these is a complete 4.3 GB merged model, not an adapter. That is
            # tolerable for the single-epoch runs this script was written for and a
            # disk bomb for anything longer: a 20-epoch run wrote 89 GB and took the
            # volume to 98% full. --checkpoint_every_epochs 0 keeps only final_merged.
            want_ckpt = (args.checkpoint_every_epochs > 0
                         and (epoch + 1) % args.checkpoint_every_epochs == 0)
            if accelerator.is_main_process and acct.batches > 0 and want_ckpt:
                output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
                print(f"Saving full merged checkpoint to: {output_dir}")
                shutil.copytree(args.init_model_path, output_dir, dirs_exist_ok=True)

                unwrapped = accelerator.unwrap_model(model)
                # There is nothing to merge in a full fine-tune: the weights on
                # disk are the weights that trained. Everything after this point --
                # the config rewrite, the speaker_encoder drop, the embedding
                # injection, save_file -- is shared with the LoRA path verbatim, so
                # the two produce the same checkpoint layout.
                if args.full_ft:
                    state_dict = full_ft_state_dict(unwrapped)
                else:
                    unwrapped.merge_adapter()
                    state_dict = clean_merged_state_dict(unwrapped)

                # Update config
                with open(os.path.join(args.init_model_path, "config.json"), 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                if not args.keep_speaker_encoder:
                    vb.bake_config(config_dict, voice_plan.slots)
                with open(os.path.join(output_dir, "config.json"), 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)

                if not args.keep_speaker_encoder:
                    # Drop speaker_encoder, then write each voice at its row: one
                    # voice at 3000 unless --speakers_from_jsonl.
                    _bake_rows(state_dict)

                save_path = os.path.join(output_dir, "model.safetensors")
                save_file(state_dict, save_path)
                print(f"✅ Full merged model saved at {save_path}")

                # Restore adapter for next epoch (no-op concept under --full_ft:
                # nothing was merged, so nothing needs unmerging).
                if not args.full_ft:
                    unwrapped.unmerge_adapter()

            # BUG-TRAIN-008: --max_steps ends the RUN. The batch loop breaks at the limit;
            # without this every remaining epoch still started, ran a micro-batch (which
            # could take another optimiser step), fed the voice mean another clip and
            # wrote another export. The epoch the limit fell in keeps its summary and its
            # export.
            if stop_early:
                break

        # ==================== FINAL MERGED FOLDER ====================
        if accelerator.is_main_process:
            print("\n=== Creating final_merged folder ===")
            final_dir = os.path.join(args.output_model_path, "final_merged")
            shutil.copytree(args.init_model_path, final_dir, dirs_exist_ok=True)

            unwrapped = accelerator.unwrap_model(model)
            if args.full_ft:
                # merge_and_unload() returns the base module with adapters folded
                # in -- which, in the full-weight path, is `unwrapped` itself. The
                # key names are therefore identical between the two paths by
                # construction, not by coincidence, which is what lets Studio's
                # convert_tts_to_gguf.py consume either without knowing which it
                # got. full_ft_state_dict additionally casts back to bf16, the
                # dtype of all 480 tensors in the base checkpoint.
                state_dict = full_ft_state_dict(unwrapped)
            else:
                # SAVE THE ADAPTER BEFORE THE MERGE, because merge_and_unload() folds
                # the adapters into the base modules IN PLACE and returns that base.
                # After the next line `unwrapped` is a bare model with no adapter left
                # to save -- which is why the call at the end of this block wrote a
                # 40-BYTE EMPTY HEADER into every checkpoint this project has ever
                # exported. 27 of them, undetected for weeks, because nothing reads the
                # file at run time. The adapters were never lost, only never written:
                # each one was recoverable by SVD of merged-minus-base, which is what
                # scripts/release/recover_lora.py had to do to ship the release.
                adapter_dir = os.path.join(final_dir, "lora_adapter")
                unwrapped.save_pretrained(adapter_dir)

                # AN INSTRUMENT, because the defect this replaces was silent for weeks
                # and produced a correctly-named file of the right shape. Size is not
                # the test -- an empty safetensors header IS a valid file -- so count
                # the tensors and refuse a run that would ship another void.
                _ad = os.path.join(adapter_dir, "adapter_model.safetensors")
                with safe_open(_ad, framework="pt") as _f:
                    _n = len(list(_f.keys()))
                if _n == 0:
                    raise RuntimeError(
                        f"{_ad} holds ZERO tensors. The adapter was saved after the "
                        f"merge, or the model carried no adapter. This is the exact "
                        f"defect that produced 27 empty adapters; refusing to write a "
                        f"28th. See docs/okf/operations/the-adapter-is-recoverable.md")
                print(f"✅ LoRA adapter saved at {adapter_dir} ({_n} tensors)")

                clean_model = unwrapped.merge_and_unload()
                state_dict = {k: v.detach().to("cpu") for k, v in clean_model.state_dict().items()}

            with open(os.path.join(args.init_model_path, "config.json"), 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            if not args.keep_speaker_encoder:
                vb.bake_config(config_dict, voice_plan.slots)
            with open(os.path.join(final_dir, "config.json"), 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            if not args.keep_speaker_encoder:
                # Drop speaker_encoder, then write each voice at its row: one
                # voice at 3000 unless --speakers_from_jsonl.
                _bake_rows(state_dict)
            else:
                print("✓ keep_speaker_encoder: exported as a reusable base checkpoint "
                      "(speaker_encoder intact, no baked embedding)")

            save_file(state_dict, os.path.join(final_dir, "model.safetensors"))
            print(f"✅ Final merged model saved at {final_dir}")

            if voice_plan.multi:
                # The plan as written at the start, plus how many clips each voice's
                # final mean is over. Beside the exports, not inside final_merged, so
                # the checkpoint folder holds what a single-voice one holds.
                with open(os.path.join(args.output_model_path, "voice_slots.json"), "w",
                          encoding="utf-8", newline="\n") as _fh:
                    json.dump(voice_plan.record(dict(voice_means.count)), _fh, indent=1,
                              ensure_ascii=False)
                    _fh.write("\n")

            # THE ADAPTER IS SAVED ABOVE, before merge_and_unload(). It used to be
            # saved here, and here is after the merge -- so `unwrapped` was a bare
            # model and save_pretrained wrote a 40-byte void.
            #
            # The comment that stood here was not wrong, it was aimed one hazard to the
            # left: it warned that calling save_pretrained on the bare model would
            # write a SECOND full copy of the weights, 3.9 GB, under a name saying the
            # opposite of what it holds. What actually shipped was a 40-byte file under
            # exactly that misleading name. The author saw that this call site was
            # dangerous and mis-identified which way it would fail.
            #
            # Nothing replaces it. Re-adding a save here would overwrite the real
            # adapter with the void it used to write.

if __name__ == "__main__":
    train()