"""Isolated encoder experiments: fresh fp16 build, no draft head, and packaged FP8 scales.

No production routing changes. The no-head engine lives in an explicitly experimental directory,
keeps zero CTC outputs for the backend binding contract, and only runs with drafts disabled.
"""

import argparse
import copy
import gc
import importlib.metadata
import json
import os
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
from profile_beam import command, comparison, dispatch, quality, result_record, save, sha256, stats, wall
from torch import nn

from owsm_trt import OWSM, build, hub, pipeline
from owsm_trt.audio import load_audio
from owsm_trt.megakernel import MegaBackend
from owsm_trt.modeling import load_model
from owsm_trt.runtime import BeamSession, TRTBackend


class CrossOnlyGraph(nn.Module):
    """Same encoder and cross projection, without evaluating the unused external draft head."""

    def __init__(self, model):
        super().__init__()
        self.encoder, self.decoder = model.encoder, model.decoder
        self.strides = model.cfg.strides

    def forward(self, feats):
        enc = self.encoder(feats)
        kv = self.decoder.cross_kv(enc)  # [L, 2, B, H, T, d_k]
        t = feats.shape[1]
        for stride in self.strides:
            t = (t - 3) // stride + 1
        ids = torch.zeros(feats.shape[0], t + 2, dtype=torch.int32, device=feats.device)
        return kv[:, 0], kv[:, 1], ids


def progress(message):
    print(message, flush=True)
    if path := os.environ.get("OWSM_PROFILE_PROGRESS"):
        with Path(path).open("a") as stream:
            stream.write(message + "\n")


def engine_hashes(directory):
    return {p.name: sha256(p) for p in directory.glob("*.engine")}


@torch.inference_mode()
def encoder_only(models, waves, out):
    """GPU event times around repeated encoder graph replay, excluding features/H2D and capture."""
    results = {}
    ordered = sorted(waves, key=len)
    for batch in (1, 3, 8):
        if len(waves) < batch:
            continue
        for length, selected in (("short", ordered[:batch]), ("long", ordered[-batch:])):
            key = f"b{batch}_{length}"
            sessions, graphs, reference = {}, {}, None
            results[key] = dict(values={}, kv_difference={}, samples=[])
            for name, model in models.items():
                sess = model._session(batch)
                feats = model._features(selected)
                paths = sess.encode(feats)
                torch.cuda.synchronize()
                if name == "no_ctc":
                    assert not paths.any(), "placeholder CTC outputs must be zero"
                if reference is None:
                    reference = (sess.cross_k.clone(), sess.cross_v.clone())
                difference = []
                for actual, expected in zip((sess.cross_k, sess.cross_v), reference):
                    delta = (actual.float() - expected.float()).abs()
                    difference.append(dict(exact=torch.equal(actual, expected), max_abs=delta.max().item(),
                                           mean_abs=delta.mean().item(), finite=bool(torch.isfinite(actual).all())))
                results[key]["kv_difference"][name] = difference
                if not all(x["finite"] for x in difference):
                    raise RuntimeError(f"nonfinite encoder KV: {name}/{key}")
                graphs[name] = sess._enc_graphs[(sess.cross_k.data_ptr(), tuple(feats.shape), 0)][2]
                sessions[name] = sess  # Keep bound outputs/contexts alive until all graph replays finish.
                results[key]["values"][name] = dict(feature_shape=list(feats.shape), ctc_shape=list(paths.shape))
            for repeat in range(7):
                order = list(models)
                order = order[repeat % len(order):] + order[:repeat % len(order)]
                for name in order:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(30):
                        graphs[name].replay()
                    end.record()
                    end.synchronize()
                    results[key]["samples"].append(dict(mode=name, repeat=repeat, cuda_ms=start.elapsed_time(end) / 30))
            for name in models:
                results[key]["values"][name]["cuda_ms"] = stats(
                    [r["cuda_ms"] for r in results[key]["samples"] if r["mode"] == name])
            save(out / "encoder_only.json", results)
            del reference, sessions, graphs
    progress(f"{out.name}: encoder-only graph timings and cross-KV comparison complete")


@torch.inference_mode()
def evaluate(models, items, waves, lang, repeat, out):
    cases = {
        "beam_b1": dict(beam_size=5, batch_size=1, timestamps=True),
        "beam_b3": dict(beam_size=5, batch_size=3, timestamps=True),
        "translation_b1": dict(batch_size=1, task="st_eng"),
        "translation_b8": dict(batch_size=8, task="st_eng"),
    }
    summaries = {}
    audio_seconds = sum(len(w) for w in waves) / next(iter(models.values())).cfg.sample_rate
    for case, kw in cases.items():
        records, cold, samples = {}, {}, []
        torch.cuda.reset_peak_memory_stats()
        for model in models.values():
            model._sessions = OrderedDict()
        gc.collect()
        torch.cuda.empty_cache()

        def call(name):
            return models[name].transcribe(waves, lang=lang, **kw)

        def record(name, outputs):
            model = models[name]
            return {it["id"]: result_record(model, value, it, len(wav) / model.cfg.sample_rate)
                    for it, wav, value in zip(items, waves, outputs)}

        for name in models:
            first, cold[name] = wall(lambda: call(name))
            records[name] = record(name, first)
            warm, _ = wall(lambda: call(name))
            assert record(name, warm) == records[name], f"{name}/{case} warmup changed output"
        save(out / case / "tokens.json", records)
        for rep in range(repeat):
            order = list(models)
            order = order[rep % len(order):] + order[:rep % len(order)]
            if rep % 2:
                order.reverse()
            for name in order:
                result, elapsed = wall(lambda: call(name))
                assert record(name, result) == records[name], f"{name}/{case} timed output changed"
                samples.append(dict(mode=name, repeat=rep, wall_ms=elapsed))
        save(out / case / "wall.json", samples)
        summary = dict(cold_ms=cold, agreement={}, quality={}, timing={}, dispatch={})
        for name, model in models.items():
            summary["agreement"][name] = comparison(records["cached"], records[name])
            summary["quality"][name] = (quality(items, records[name], lang) if kw.get("task", "asr") == "asr"
                                       else {"unmeasured": "No reference English translations"})
            values = [r["wall_ms"] for r in samples if r["mode"] == name]
            summary["timing"][name] = stats(values) | {"rtfx": audio_seconds / (stats(values)["p50"] / 1000)}
            summary["dispatch"][name] = {str(key): dispatch(sess) for key, sess in model._sessions.items()}
            assert not getattr(model.backend, "failed", set()), "unexpected decoder fallback"
        summary["shared_process_peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
        summaries[case] = summary
        save(out / "summary.json", summaries)
        if kw.get("beam_size", 1) > 1:
            for name, model in models.items():
                diagnostic = []

                class ObservedBeam(BeamSession):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **(kwargs | {"diagnostics": True}))

                    def decode(self, *args):
                        tokens = super().decode(*args)
                        data = self.get_diagnostics()
                        for utterance in data["utterances"]:
                            del utterance["history"]
                        diagnostic.append(data)
                        return tokens

                model._sessions = OrderedDict()
                with patch.object(pipeline, "BeamSession", ObservedBeam):
                    result = call(name)
                assert record(name, result) == records[name], "diagnostics changed output"
                save(out / case / f"{name}_diagnostics.json", dict(
                    batches=diagnostic, sorted_input_ids=[items[i]["id"] for i in sorted(range(len(waves)), key=lambda i: len(waves[i]))]))
        progress(f"{out.name}: {case} encoder variant comparison complete")


@torch.inference_mode()
def extra_cases(models, waves, long_audio, out):
    synthetic = [torch.zeros(32000), torch.randn(32000, generator=torch.Generator().manual_seed(77)) * .01]
    cases = [
        ("silence_noise", synthetic + waves[:1], dict(beam_size=5, batch_size=3, timestamps=True), False),
        ("long_prompt", waves[:1], dict(beam_size=5, timestamps=True, prompt="Previous speech context. " * 20), False),
        ("no_timestamps", waves[:3], dict(beam_size=5, batch_size=3), False),
    ]
    if long_audio:
        cases.append(("long_110s", [load_audio(long_audio)], dict(beam_size=5, batch_size=1), True))
    report = {}
    for case, inputs, kwargs, long_form in cases:
        records = {}
        for name, model in models.items():
            fn = model.transcribe_long if long_form else model.transcribe
            outputs = fn(inputs, lang="eng", **kwargs)
            records[name] = {str(i): result_record(model, result, {"id": str(i)}, len(wave) / model.cfg.sample_rate)
                             for i, (wave, result) in enumerate(zip(inputs, outputs))}
        report[case] = dict(records=records, agreement={name: comparison(records["cached"], values)
                                                      for name, values in records.items()})
        save(out / "extra_cases.json", report)
        progress(f"{case}: extra encoder variant outputs recorded")


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--extra-manifest", type=Path)
    ap.add_argument("--long-audio", type=Path)
    ap.add_argument("--repeat", type=int, default=7)
    ap.add_argument("--fp8", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not os.environ.get("PBS_JOBID") or not os.environ.get("PBS_LOCALDIR"):
        ap.error("build/model/audio work requires an allocated PBS compute job")
    if args.repeat < 1:
        ap.error("repeat must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    base, load_ms = wall(lambda: OWSM.from_pretrained(args.model, ctc_draft=False))
    if not isinstance(base.backend, MegaBackend) or base.cfg.ctc_only:
        raise ValueError("this experiment requires the encoder-decoder megakernel backend")
    files = hub.resolve(args.model)
    engine = hub.cache_dir() / files.name.replace("/", "--") / build.engine_tag(base.opts)
    models, directories = {"cached": base}, {"cached": engine}
    metadata = dict(commit=os.environ.get("OWSM_PROFILE_COMMIT"), gpu=torch.cuda.get_device_name(),
                    torch=torch.__version__, cuda=torch.version.cuda, load_ms=load_ms,
                    sm_count=torch.cuda.get_device_properties(base.device).multi_processor_count,
                    versions={n: importlib.metadata.version(n) for n in ("triton", "tensorrt")},
                    nvidia_smi=command(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv"]),
                    checkpoint_sha256=sha256(files.checkpoint), builds={}, engines={},
                    notes=["No CTC drafts in any mode; identical decoder/verify engines and beam rules.",
                           "Fresh fp16 control isolates rebuild variation; removing CTC is E intent/N engine rebuild.",
                           "Packaged FP8 is an independent A experiment; changed outputs are reported, not suppressed.",
                           "Cold observations have unequal shared compiler caches; no cold improvement claimed.",
                           "Process memory includes every variant; encoder-only event time excludes feature/H2D/capture."])
    net = load_model(base.cfg, str(files.checkpoint), base.device, base.dtype)
    for name in ("rebuilt", "no_ctc"):
        directory = Path(os.environ["PBS_LOCALDIR"]) / f"encoder-experiment-{name}"
        context = patch.object(build, "_EncoderGraph", CrossOnlyGraph) if name == "no_ctc" else nullcontext()
        with context:
            _, duration = wall(lambda network=net: build.build_engines(network, directory, base.opts, decoders_from=engine))
        directories[name] = directory
        metadata["builds"][name] = dict(wall_ms=duration, draft_head=name != "no_ctc")
        model = copy.copy(base)
        model._sessions, model._beam_backend = OrderedDict(), None
        source = base.backend
        model.backend = MegaBackend(TRTBackend(base.cfg, directory), source.kernel, source.max_tokens,
                                    source.max_rows, rows_kernel=source.rows_kernel, trial=source.trial)
        models[name] = model
        progress(f"{name}: experimental encoder build complete")
    del net
    gc.collect()
    torch.cuda.empty_cache()
    if args.fp8:
        models["fp8"], duration = wall(lambda: OWSM.from_pretrained(args.model, fp8=True, ctc_draft=False))
        assert models["fp8"].opts.fp8_encoder, "packaged scales unavailable; refusing silent fp16 fallback"
        directories["fp8"] = engine.parent / build.engine_tag(models["fp8"].opts)
        metadata["builds"]["fp8"] = dict(load_build_ms=duration, calibration=models["fp8"].opts.fp8_calibration)
        progress("fp8: packaged-scale encoder build/load complete")
    for name, directory in directories.items():
        hashes = engine_hashes(directory)
        metadata["engines"][name] = dict(sha256=hashes, build=json.loads((directory / "build.json").read_text()))
        for decoder in ("decoder.engine", "verify.engine"):
            assert hashes[decoder] == metadata["engines"]["cached"]["sha256"][decoder]
    save(args.out / "environment.json", metadata)
    for name, path, lang in (("english", args.manifest, "eng"), ("japanese", args.extra_manifest, "jpn")):
        if path is None:
            continue
        items = json.loads(path.read_text())
        assert len({it["id"] for it in items}) == len(items), "manifest IDs must be unique"
        waves, disk_ms = wall(lambda: [load_audio(it["path"]) for it in items])
        save(args.out / name / "input.json", dict(manifest=items, disk_ms=disk_ms))
        if name == "english":
            encoder_only(models, waves, args.out / name)
        evaluate(models, items, waves, lang, args.repeat, args.out / name)
        if name == "english":
            extra_cases(models, waves, args.long_audio, args.out)
    progress("All encoder variant comparisons complete")


if __name__ == "__main__":
    main()
