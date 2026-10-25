# qwen3-tts-cantonese-recipe

The training recipe for a Hong Kong Cantonese fine-tune of
[Qwen3-TTS 12Hz 1.7B Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base): data selection, cleanup,
hyperparameters, the release measures and the listening pages, as one pipeline you run from a clean checkout.
Code and configuration only: no audio, transcripts, clip ids, manifests or weights. You download Common Voice
yourself.

It is the lite repository of a larger research lab, which keeps the evidence for every choice below. This one
states what to do.

## The recipe (H1: D1 + zh-HK, the lab's EXP-100; chosen by the owner for v0.0.5 on 2026-10-22)

The owner locked D1 as the training recipe on 2026-10-22 and, the same day, chose EXP-100's H1 for v0.0.5 ("I think
100 is worth a release"), shipping **both voices**: D1 with Common Voice zh-HK 27.0 added to the language phase. On
the owner's listen (BM-100-p2, scene 6) H1 rated 3.67 against v0.0.1's 3.00, preferred 3 to 0.

| Step | What | Values (all in `recipe.toml`) |
|---|---|---|
| 1. Data | Common Voice `yue` 22.0: v0.0.1's language set (validated, 3-30 s, colloquial transcripts, no down-votes, every speaker; edge silence trimmed at ingest, 4 s floor on the decoded audio); **plus Common Voice zh-HK 27.0** from Mozilla Data Collective (validated, 3-30 s, colloquial, no down-votes, no accent or gender filter, seed 4242); and one donor's clips | 33,104 yue + 29,562 zh-HK + 343 clips in the lab |
| 2. Cleanup | Sidon v0.1 restoration, then `dynamics` (the raw clip's loudness contour put back), edges v2 and `trim` (0.10 s before the speech, 0.30 s after); `pad` back to the source length with digital silence on the **speaker clips only**; then `audio` and `encode` (the 12 Hz codec). Each step is an immutable, re-runnable store layer | exclude a clip whose Sidon envelope correlation is under 0.5 |
| 3. Sets | the language set is the trimmed yue rows followed by the trimmed zh-HK rows, concatenated; the speaker set is the padded donor clips (D1's, unchanged) | 33,097 + 29,511 = 62,608 and 343 rows in the lab; `speaker_id` cv_yue_v22 / cv_zhhk_27 / cv_voice6 |
| 4. Training | language phase from the public base with `--keep_speaker_encoder`, then a speaker phase on the donor's clips, one voice `hk_voice_343` | LoRA r32/a64, lr 1e-4, warmup 0.03, batch 1 x grad-accum 4, 1 epoch, `sub_talker_weight` 0, sdpa, a checkpoint every 500 steps |
| 5. Voices | the chosen vectors copied into `codec_embedding` from a source checkpoint; **both ship** | `hk_voice_343` row 3000, `hk_voice_777` row 3001 |
| 6. Conversion | GGUF q8_0 for [Qwen-TTS Studio](https://github.com/hobpond/qwen-tts-studio), through the pinned qwen3-tts.cpp runtime | `tts_model_size` must be `1b7` |
| 7. Measures | first-draw renders of held-out short lines and long prose at chunk ceiling 45, for each voice and for v0.0.1; the release gate's Part A and the checkpoint card, each voice on its own | thresholds in `[gate]`; A7 waived by the owner (below) |
| 8. Listening | long prose by paragraph, blind, 1-5 ratings, per-character marks and notes, skipped text redrawn for the final page | `[pages]` |

Why each piece is there, in one line each: Sidon cleans the crowd-recorded audio but compresses its dynamics, so
`dynamics` restores the contour; Sidon's noisy edges caused runaways, so `trim` cuts to the speech; the trim also
removed every edge silence, which made the speaker phase skip text, so `pad` puts the silence back as zeros (on the
language set too, it brought runaways back); the voice row is chosen, never assumed; chunks of 70 characters skip
whole clauses, 45 do not; zh-HK adds Hong Kong speech to a `yue` pool drawn from the whole Cantonese-speaking world
(in one early selection only 3.9% of clips were labelled Hong Kong). zh-HK is a language set, so it is trimmed and
not padded.

## Where the v0.0.5 build's data starts

The owner's decision for the from-scratch build (2026-10-22): **"Start from the raw clips on disk."** Two of the three
sets cannot be re-selected by download:

- **yue 22.0**: the Common Voice 22.0 pool the lab selected from is no longer held, so the lab's own selection (its
  55,668 selected clips, `audio/` + `transcripts.tsv`) is given as `LANG_SELECTED_DIR`; ingest with edge-silence
  trimming at a 4.0 s floor should give v0.0.1's 33,104 (a 40-row sample of v0.0.1's set matches the trimmed decode,
  not the untrimmed one; the full count has not been re-run).
- **the donor's 343 clips**: they came from an earlier `yue` release whose number was not recorded. From 22.0 the
  donor has 777 clips, 338 of the 343 among them, so a re-selection would be a different speaker set. The lab's
  selected 348 clips are given as `SPEAKER_SELECTED_DIR`; ingest with edge-silence trimming at a 4.0 s floor gives
  343, with the same transcripts and the same durations to 0.01 s as v0.0.1's speaker set (measured; at 3.0 s, or
  untrimmed, all 348 pass).
- **zh-HK 27.0** is selected from the original MDC archive by `src/select_mdc.py`, as the lab did.

For a pre-selected set `recipe.py select` runs no selector, only `ingest_recordings.py`, and
`records/select.json` says which sets came pre-selected, with the sha256 of each `transcripts.tsv`. Everything from
ingest on runs from scratch in this repository.

## Commands, in order

Everything runs on Linux or WSL with a CUDA card (the lab's: a 12 GB RTX 3080 Ti; the language phase took about two
hours). Set the paths, then run each stage and read its record before the next.

```bash
pip install -r requirements.txt
bash scripts/build_gguf_runtime.sh                 # qwen3-tts.cpp at the pinned commit + the --serve patch

export RECIPE_WORK=~/recipe-work                   # everything the recipe writes; a native disk, never /mnt
export TTS_BASE_MODEL=~/models/Qwen3-TTS-12Hz-1.7B-Base
export SIDON_MODELS=~/models/sidon-v0.1
export CV_ROOT=~/datasets/cv-yue-22                # or CV_MDC_ARCHIVE=<the MDC .tar.gz>
export CV_ZHHK_MDC_ARCHIVE=~/datasets/mdc/common-voice-scripted-speech-27-0-chines-26ba81db.tar.gz
                                                   # MDC dataset cmu5wp2bp00dgnq07ylmgbnrz, downloaded by you
export DONOR_IDS=~/private/donor-ids.txt           # the donor's Common Voice pseudonym; never in this repository
export LANG_SELECTED_DIR=~/datasets/cv-lang-v22    # or: a set already selected (audio/ + transcripts.tsv): no selector
export SPEAKER_SELECTED_DIR=~/datasets/cv-voice6   #     is run for it, only ingest (see above)
export VOICE_SOURCE=~/models/release-merged        # a merged checkpoint carrying the chosen rows 3000 and 3001
export BASELINE_GGUF=~/.qwen-tts-gguf/v0.0.1       # v0.0.1 converted to GGUF
export EVAL_LINES=~/eval/held-out-lines.jsonl      # {"digest", "text"} per line, 200 lines
export EVAL_PROSE=~/eval/prose                     # <id>.txt: title, blank line, paragraphs; plus about.json

python recipe.py check                             # paths, modules, runtime, free card memory
python recipe.py select                            # 1
python recipe.py cleanup                           # 2   (GPU: Sidon, the codec)
python recipe.py sets                              # 3
python recipe.py train-lang                        # 4a  (GPU)
python recipe.py train-spk                         # 4b  (GPU)
python recipe.py bake                              # 5
python recipe.py convert                           # 6
python recipe.py render                            # 7a  (GPU)
python recipe.py gate                              # 7b  (GPU: Qwen3-ASR, the aligner)
python recipe.py card                              # 7c  (GPU: SenseVoice, ECAPA)
python recipe.py pages                             # 8   (GPU: the redraw)
```

Add `--dry-run` to any stage to print what it would run. Every stage writes `$RECIPE_WORK/records/<stage>.json`
(commands, data sha256, MLflow run id, this repository's commit) and logs to `$RECIPE_WORK/logs/`. Those records
ship with any weights this recipe produces.

Fetch the models once (they are not downloaded by the stages):

```bash
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir "$TTS_BASE_MODEL"
hf download Qwen/Qwen3-TTS-Tokenizer-12Hz                           # the encode step's codec, from the HF cache
hf download sarulab-speech/sidon-v0.1 feature_extractor_cuda.pt decoder_cuda.pt --local-dir "$SIDON_MODELS"
hf download facebook/w2v-bert-2.0 preprocessor_config.json --local-dir "$SIDON_MODELS/w2v-bert-2.0-fe"
hf download Qwen/Qwen3-ASR-1.7B                                      # the measures' recogniser, from the HF cache
hf download FunAudioLLM/SenseVoiceSmall \
    --local-dir ~/.cache/modelscope/models/iic--SenseVoiceSmall/snapshots/main   # where src/steps.py looks for it
```

Those locations are where the code looks (the HF cache for the tokenizer and Qwen3-ASR; the snapshot directory
above for SenseVoice, the aligner and second recogniser). The lab's `w2v-bert-2.0-fe/preprocessor_config.json` was
not fetched by a recorded command; it is a `SeamlessM4TFeatureExtractor` config (80 mel bins, 16 kHz, stride 2),
which is what facebook/w2v-bert-2.0 ships. Compare before relying on it.

## The release gate

A candidate is tested against **v0.0.1**, the only release, on the same lines, seeds and backend, and **every voice
it ships** is tested on its own. The tests may veto; they never close. Release needs no veto, the owner's listen,
the GGUF loading in Studio with its voices listed, and weights trained by this repository alone.

Part A, short lines (200 held-out lines x 2 draws per voice), implemented in `src/release_gate.py`:

| # | Test | Rule | Kind |
|---|---|---|---|
| A1 | edges: `peak_db_rel_speech`, lead and trail | Wilcoxon one-sided candidate lower, at 0.05 / 4 | PRIMARY |
| A2 | length bars | worse at p < 0.05 | VETO, **not implemented here** |
| A3 | last character (Qwen3-ASR) | exact McNemar one-sided, worse at p < 0.05 | VETO |
| A4 | CER (Qwen3-ASR) | Wilcoxon one-sided higher at p < 0.05 **and** median more than 0.03 above | VETO |
| A5 | tone (tonefit, the donor's templates) | paired bootstrap 2000, seed 20261018; 95% interval wholly below zero | VETO |
| A6 | runaways | speech over twice v0.0.1's: more than v0.0.1's count the other way + 5 | VETO |
| A7 | liveliness: loudness range p10-p90 | at least 0.5 dB narrower at p < 0.0125 | VETO |
| A8 | skipped text: runs of 4+ characters not heard | more clips than v0.0.1, exact McNemar p < 0.05 | VETO |

DAMAGE_VETO if any veto fires; OBJECTIVELY_BETTER if A1 shows the candidate quieter at an end; NOT_SHOWN otherwise.

**A7 is set aside for this release, by the owner (2026-10-22).** The lab's H1 read a loudness range 0.83 dB narrower
than v0.0.1's, which A7 vetoes. The owner reads that as "likely confounded by the uniform need of no room acoustic
artifact": v0.0.1's renders change room from chunk to chunk, H1's do not. The lab tested that explanation the same
day and found the opposite direction: a planted room **narrows** this measure (-5.0 dB median on 120 of H1's lines,
reverb filling the gaps between syllables), so room acoustics in v0.0.1's renders would lower its range, not raise it.
What makes the raw-trained voice measure livelier is not established. **The waiver is the owner's judgement, not a
measured confound.** `[gate].waived` carries it: A7 is still measured, a firing is reported as `waived_vetoes` and
does not decide the verdict, and the model card must state the measured flatness. Delete the entry to restore the
veto.

Part B, long prose at ceiling 45 (not implemented here; the thresholds, from the lab's gate):

| # | Test | Rule | Kind |
|---|---|---|---|
| B1 | first-draw runaways (over twice the voice's median s/char) and silences (under 0.07 s/char), per chunk | more than v0.0.1's at one-sided Fisher p < 0.05 and by at least 5 points | VETO |
| B2 | chunk CER (Qwen3-ASR) | median more than 0.03 above v0.0.1's | VETO |
| B3 | internal pauses of 1 s or more | | REPORTED |
| B4 | chunks with a lost run of 4+ characters | more than v0.0.1's at one-sided Fisher p < 0.05, or any chunk losing 10+ characters that v0.0.1 does not | VETO |

Part C is the owner's ear: the page `recipe.py pages` builds, registered before it is heard.

**The checkpoint card** (`src/checkpoint_card.py`) reports, never decides: dropped characters both recognisers
miss (long prose), the loud and soft ends and pitch ends, sentence-final particles' final F0, tone match per tone
and the Hong Kong mergers, cepstral peak prominence, and ECAPA similarity to the donor, each against v0.0.1 with a
bootstrap interval.

## What was checked, and what was not

Checked without a GPU (2026-10-22): every module imports; every stage's `--dry-run` prints its commands; the CPU
self-tests pass (`scripts/check_clean_room.py`, `scripts/check_publishable.py`, `src/gguf_backend.py`,
`src/bake_voices.py`, `src/render_short.py`, `src/skipped_text.py`, `src/select_mdc.py`, `src/release_gate.py`,
`src/checkpoint_card.py`, `trainer/test_voice_bank.py`, `trainer/test_epoch_accounting.py`); `cleanup` and `sets`
ran end to end on synthetic yue, zh-HK and speaker clips with Sidon and the codec replaced by CPU stand-ins (the
language set came out as the yue rows then the zh-HK rows, zh-HK trimmed and not padded); the standard page built from a
synthetic clip and round-tripped a code. Read-only on the lab's stored EXP-099 D1 and v0.0.1 layers,
`src/release_gate.py` reproduced every number of the lab's gate read and the card reproduced the lab's card. The
cleanup steps' code hashes equal the lab's (sidon and tails differ only in where their helpers live); the trainer is
the file that trained D1, byte for byte, apart from one argument guard; and `train-lang --dry-run` and
`train-spk --dry-run` print exactly the flags of D1's and H1's recorded commands, paths aside, with the language
phase reading the concatenated `lang-yue-zhhk.jsonl`. Read-only on the lab's stored H1 layers, the gate reads A7 at
-0.828 dB (the lab's -0.828), fires it, and with the owner's waiver reports it as `waived_vetoes` and returns
OBJECTIVELY_BETTER.

**Not run here**: Sidon, the codec, training, conversion, rendering, ASR, alignment, or any page with real audio.
The recipe has not yet produced a model. Acceptance (the lab's TASK-RECIPE-RELEASE-001) is a from-scratch run of these
commands that passes the gate and the owner's listen.

## Open items

- **Gate A2 and Part B** are tables above, not code: the lab's implementations are its experiment chain.
- **A4 on the folded CER**: the recogniser writes Sidon audio in simplified characters, which the CER counts as
  error. Folded to one script, D1's CER equals v0.0.1's. The gate reports the folded CER beside A4; whether A4
  vetoes on it is the owner's decision.
- **The speaker set's selection**: v0.0.1's 343 clips came from an earlier Common Voice release whose number was not
  recorded, and their filters were not recorded either. `[data.speaker]` holds the selector's speaker-phase defaults
  and an `expect_clips` of 343; from 22.0 the same donor has 777 clips.
- **The card's instrument controls** (liveliness and tone plants, the script-switch plant, held-out donor clips) need
  plant collections the lab builds with its own experiment code; `--self-test` here checks the card's arithmetic.
- **hk_voice_777** has not been through the gate, the card or a listen in the lab (its row is the averaged vector the
  lab found 3 dB flat); this recipe renders, gates and cards it beside `hk_voice_343`.
- **The evaluation material** (the held-out lines, the long prose and its `about.json`) is not in this text-free
  repository; candidate and baseline must be rendered from the same files.

## The clean room

Before every commit:

```bash
python scripts/check_clean_room.py --staged
```

It refuses inlined or binary audio, opaque payloads, `audio_codes` arrays, corpus clip paths, speaker ids and donor
names, whatever the file extension (checked: a staged `notes.md` holding a manifest row and a listening page renamed
`.md` are both refused). A bare line of Cantonese is only flagged REVIEW, because no scanner can tell a transcript
from prose about Cantonese; for those the line is `.gitignore`, which blocks text, tables, JSON and JSONL along with
media, weights and pages by extension. There is no hook: running it is the rule.

## Licence

Apache-2.0 (`LICENSE`); third-party components and what is fetched rather than shipped are in `NOTICE`.
