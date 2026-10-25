# SPDX-License-Identifier: Apache-2.0
"""BUG-TRAIN-001: prove the per-epoch accounting counts every micro-batch.

Runs with the standard library only (no torch, accelerate or qwen_tts):

    python trainer/test_epoch_accounting.py

Three layers, and it says plainly what each one does not prove:

1. The helper, on a deterministic fixture: counts and means are exact, and an epoch
   that recorded nothing reads as nothing, not as zero loss.
2. The trainer's STRUCTURE, read with `ast`: the accounting call sits inside the batch
   loop, before the max_steps break, and not under the epoch-level STEPTIMES block
   where the bug put it. The optimiser, scheduler and clipping calls still appear
   exactly once, inside the `sync_gradients` branch. This pins placement; it does not
   run the trainer.
3. A pure-Python replica of the loop's control flow (accumulation, max_steps), showing
   the intended counts, including the micro-batch that completes the final optimiser
   step. It is a replica, not the trainer.

NOT covered here, and reported NOT RUN until done on a machine with the model: a real
run showing identical optimiser updates before and after the fix.
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from epoch_accounting import EpochAccounting  # noqa: E402

TRAINER = HERE / "sft_12hz_lora_mlflow.py"


class HelperFixture(unittest.TestCase):
    def test_counts_and_means_are_exact(self):
        a = EpochAccounting()
        rows = [(2.0, 1.0, 5.0, 9.0), (4.0, 3.0, 7.0, 11.0), (6.0, 5.0, 9.0, 13.0)]
        for r in rows:
            a.add(*r)
        self.assertEqual(a.batches, 3)
        self.assertEqual(a.means(), {"loss": 4.0, "talker": 3.0, "sub": 7.0,
                                     "sub_buggy": 11.0})

    def test_add_returns_the_batch_loss(self):
        self.assertEqual(EpochAccounting().add(1.5, 0, 0, 0), 1.5)

    def test_empty_epoch_is_none_not_zero(self):
        self.assertIsNone(EpochAccounting().means())

    def test_the_bug_shape_is_detectable(self):
        # What the bug reported: the last batch only. A mean over all batches differs.
        a = EpochAccounting()
        for v in (10.0, 20.0, 30.0):
            a.add(v, 0, 0, 0)
        self.assertNotEqual(a.means()["loss"], 30.0)
        self.assertEqual(a.means()["loss"], 20.0)


def _loop_nodes(tree):
    """The epoch loop, the batch loop inside it, and every acct.add call."""
    batch_loop = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.For) and isinstance(node.iter, ast.Call)
                and getattr(node.iter.func, "id", "") == "enumerate"
                and ast.unparse(node.iter.args[0]) == "train_dataloader"):
            batch_loop = node
    adds = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and ast.unparse(n.func) == "acct.add"]
    return batch_loop, adds


class TrainerStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = TRAINER.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.src)
        cls.batch_loop, cls.adds = _loop_nodes(cls.tree)

    def test_one_accounting_call_inside_the_batch_loop(self):
        # TWO CALL SITES SINCE THE LAB ADDED ITS PREFERENCE PATH, ONE PER MICRO-BATCH STILL: the SFT call and the
        # preference call sit in the two arms of one `if`, so exactly one runs. This test asserted a single call and
        # had been failing in the lab since that path landed; the recipe copy asserts the property it was protecting.
        self.assertIsNotNone(self.batch_loop, "batch loop not found")
        inside = [n for n in ast.walk(self.batch_loop) if n in self.adds]
        self.assertEqual(len(self.adds), 2)
        self.assertEqual(len(inside), 2, "every acct.add must be inside the batch loop")
        arms = [n for n in ast.walk(self.batch_loop) if isinstance(n, ast.If)
                and any(x in self.adds for b in n.body for x in ast.walk(b))
                and any(x in self.adds for b in n.orelse for x in ast.walk(b))]
        self.assertTrue(arms, "the two acct.add calls must be the two arms of one if, so one runs per micro-batch")

    def test_accounting_precedes_the_max_steps_break(self):
        breaks = [n for n in ast.walk(self.batch_loop) if isinstance(n, ast.If)
                  and "max_steps" in ast.unparse(n.test)
                  and any(isinstance(b, ast.Break) for b in n.body)]
        self.assertEqual(len(breaks), 1)
        self.assertLess(self.adds[0].lineno, breaks[0].lineno,
                        "the batch completing the final optimiser step must count")

    def test_steptimes_print_is_epoch_level(self):
        steptimes = [n for n in ast.walk(self.tree) if isinstance(n, ast.If)
                     and "_step_times" in ast.unparse(n.test)
                     and "STEPTIMES " in ast.unparse(n)]
        self.assertEqual(len(steptimes), 1)
        in_batch_loop = any(n is steptimes[0] for n in ast.walk(self.batch_loop))
        self.assertFalse(in_batch_loop)
        # The bug's shape, tested by shape rather than by name: anything under this
        # block besides the one print is per-batch work running once per epoch.
        body = steptimes[0].body
        self.assertEqual(len(body), 1, "the STEPTIMES block must hold only its print; "
                         "the bug nested the per-batch accounting under it")
        self.assertEqual(ast.unparse(body[0].value.func), "accelerator.print")

    def test_optimiser_path_unchanged_in_shape(self):
        sync = [n for n in ast.walk(self.batch_loop) if isinstance(n, ast.If)
                and ast.unparse(n.test) == "accelerator.sync_gradients"]
        self.assertEqual(len(sync), 1)
        # Count CALLS in the parsed code, not text: docstrings mention these names.
        def calls(node, name):
            return [n for n in ast.walk(node) if isinstance(n, ast.Call)
                    and ast.unparse(n.func) == name]
        for name in ("accelerator.clip_grad_norm_", "optimizer.step",
                     "scheduler.step", "optimizer.zero_grad"):
            in_sync = calls(sync[0], name)
            self.assertEqual(len(in_sync), 1, name)
        self.assertEqual(len(calls(self.batch_loop, "optimizer.step")), 1)
        self.assertEqual(len(calls(self.batch_loop, "accelerator.backward")), 1)


def replica(n_batches: int, accum: int, max_steps: int) -> tuple[int, int]:
    """The trainer's loop control, reduced to integers. Returns (counted, opt_steps)."""
    acct, opt = EpochAccounting(), 0
    for step in range(n_batches):
        if (step + 1) % accum == 0:          # accelerate's sync_gradients boundary
            opt += 1
        acct.add(float(step), 0, 0, 0)       # the fixed position
        if max_steps and opt >= max_steps:
            break
    return acct.batches, opt


class ControlFlowReplica(unittest.TestCase):
    def test_full_epoch_counts_every_batch(self):
        self.assertEqual(replica(10, 4, 0), (10, 2))

    def test_max_steps_boundary_counts_the_final_micro_batch(self):
        # 2 optimiser steps at accum 4 complete on batch index 7: 8 batches counted.
        self.assertEqual(replica(10, 4, 2), (8, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
