"""Opt-in numerical experiment: verification-window budgets, same checkpoint/engines.

Changing window/batch shapes can change fp16 rounding. Differences are reported, not hidden.
All candidates preserve verification, EOS and existing timestamp constraints.
"""

import argparse
import gc
import json
import os
from collections import Counter, OrderedDict
from pathlib import Path

import torch
from profile_beam import comparison, quality, result_record, save, sha256, stats, wall
from profile_runtime import captures, dispatch

from owsm_trt import OWSM, hub
from owsm_trt import pipeline as pipe
from owsm_trt.audio import load_audio
from owsm_trt.build import engine_tag


class Counted:
    def __init__(self, bound, key, counts):
        self.bound, self.key, self.counts = bound, key, counts

    def __getattr__(self, name):
        return getattr(self.bound, name)

    def __call__(self, *args, **kwargs):
        self.counts[self.key] += 1
        return self.bound(*args, **kwargs)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--lang", default="eng")
    ap.add_argument("--batch", type=int, nargs="+", default=[3, 8])
    ap.add_argument("--budgets", type=int, nargs="+", default=[256, 128, 64, 32])
    ap.add_argument("--repeat", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not os.environ.get("PBS_JOBID") and os.uname().nodename.startswith("login"):
        ap.error("audio/model work requires a compute allocation")
    if 256 not in args.budgets or min(args.budgets) < 1:
        ap.error("positive budgets and reference=256 required")
    model, load_ms = wall(lambda: OWSM.from_pretrained(args.model, lazy_drafts=True))
    files = hub.resolve(args.model)
    engines = hub.cache_dir() / files.name.replace("/", "--") / engine_tag(model.opts)
    save(
        args.out / "environment.json",
        dict(
            commit=os.environ.get("OWSM_PROFILE_COMMIT"),
            checkpoint_sha256=sha256(files.checkpoint),
            engine_sha256={p.name: sha256(p) for p in engines.glob("*.engine")},
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            model_load_ms=load_ms,
        ),
    )
    items = json.loads(Path(args.manifest).read_text())
    waves = [load_audio(it["path"]) for it in items]
    seconds = sum(len(w) for w in waves) / model.cfg.sample_rate
    results = {}
    for b in args.batch:
        for ts in (False, True):
            name = f"b{b}_ts{int(ts)}"
            pools, cold, records, samples = {}, {}, {}, []
            model._sessions.clear()
            gc.collect()
            torch.cuda.empty_cache()

            def call():
                return model.transcribe(waves, lang=args.lang, batch_size=b, timestamps=ts)

            for budget in args.budgets:
                pipe.DRAFT_TOKENS = budget
                model._sessions = pools[budget] = OrderedDict()
                _, ms = wall(call)
                cold[budget] = dict(
                    warmup_ms=ms,
                    captures=captures(model),
                    dispatch=dispatch(model),
                    windows=model._draft_windows(b),
                    stream=len(items) > b
                    and len(model._primer(args.lang, "asr", ts, None)) < max(model._draft_windows(b)),
                )
            for repeat in range(args.repeat):
                for budget in args.budgets if repeat % 2 == 0 else args.budgets[::-1]:
                    pipe.DRAFT_TOKENS = budget
                    model._sessions = pools[budget]
                    before = model.draft_stats[1]
                    out, ms = wall(call)
                    if captures(model) != cold[budget]["captures"]:
                        raise RuntimeError(f"new capture in steady state {name}/{budget}")
                    records_now = {
                        it["id"]: result_record(model, r, it, len(w) / model.cfg.sample_rate)
                        for it, r, w in zip(items, out, waves)
                    }
                    if budget in records and records_now != records[budget]:
                        raise RuntimeError(f"non-deterministic output {name}/{budget}")
                    records[budget] = records_now
                    samples.append(
                        dict(
                            budget=budget,
                            repeat=repeat,
                            wall_ms=ms,
                            decoder_calls=model.draft_stats[1] - before,
                        )
                    )
            actual_dispatch = {}
            for budget in args.budgets:
                pipe.DRAFT_TOKENS = budget
                model._sessions = pools[budget]
                counts = Counter()
                for sess in model._sessions.values():
                    for label in ("_verify", "_verify_solo"):
                        table = getattr(sess, label)
                        for (c, w), bound in list(table.items()):
                            key = f"rows={sess.rows}/cache={c}/window={w}/{label}/{getattr(bound.step, 'dispatch', 'unknown')}"
                            table[c, w] = Counted(bound, key, counts)
                call()  # counters excluded from uninstrumented samples
                actual_dispatch[budget] = dict(counts)
            summary = {}
            for budget in args.budgets:
                ms = [r["wall_ms"] for r in samples if r["budget"] == budget]
                summary[budget] = dict(
                    wall_ms=stats(ms),
                    rtfx=seconds * len(ms) * 1000 / sum(ms),
                    agreement=comparison(records[256], records[budget]),
                    quality=quality(items, records[budget], args.lang),
                )
            path = args.out / name
            save(path / "summary.json", summary)
            save(path / "tokens.json", records)
            save(path / "wall.json", samples)
            save(path / "cold.json", cold)
            save(path / "actual_dispatch.json", actual_dispatch)
            results[name] = summary
            save(args.out / "summary.json", results)
            print(name, {k: round(v["rtfx"], 2) for k, v in summary.items()}, flush=True)
            if progress := os.environ.get("OWSM_PROFILE_PROGRESS"):
                with Path(progress).open("a") as f:
                    f.write(f"draft budgets {args.lang} {name} complete\n")
            pools.clear()
    save(
        args.out / "experiment.json",
        dict(
            classification="N: shape/dispatch changes",
            args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            baseline_budget=256,
            reference_time_annotations=False,
            note="Budget32 with B8 and four-token primer uses fixed batches under existing stream eligibility.",
        ),
    )


if __name__ == "__main__":
    main()
