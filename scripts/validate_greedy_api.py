"""Compute-node regression of the public greedy-kernel default and serial rollback."""

import argparse
import gc
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch
from profile_beam import save, sha256, wall
from profile_runtime import captures

from owsm_trt import OWSM, hub
from owsm_trt.audio import load_audio
from owsm_trt.build import engine_tag
from owsm_trt.runtime import GreedySession


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--decoder-kernel", default="auto")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--lang", default="eng")
    ap.add_argument("--lazy-plain", action="store_true", help="Validate opt-in deferred plain setup")
    ap.add_argument("--finished-stop", action="store_true", help="Assert the public finished-call guard and rollback")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not os.environ.get("PBS_JOBID") and os.uname().nodename.startswith("login"):
        ap.error("model/audio work requires a compute allocation")
    items = json.loads(Path(args.manifest).read_text())[:8]
    waves = [load_audio(x["path"]) for x in items]
    expected, records, engine_hashes = {}, {}, None
    for mode, options in (("serial", {"greedy_kernels": False, "lazy_drafts": False}), ("default", {})):
        model, load_ms = wall(lambda: OWSM.from_pretrained(args.model, decoder_kernel=args.decoder_kernel, **options))
        if args.lazy_plain:
            model.greedy_lazy_plain = mode == "default"
        files = hub.resolve(args.model)
        engines = hub.cache_dir() / files.name.replace("/", "--") / engine_tag(model.opts)
        hashes = {p.name: sha256(p) for p in engines.glob("*.engine")}
        if engine_hashes is not None and hashes != engine_hashes:
            raise RuntimeError("engine changed between public API modes")
        engine_hashes = hashes
        results, dispatch = {}, {}
        for name, kwargs in (
            ("single", {"timestamps": False}),
            ("single_ts", {"timestamps": True}),
            ("stream", {"batch_size": 1}),
            ("stream_ts", {"batch_size": 2, "timestamps": True}),
            ("prompt", {"timestamps": True, "prompt": "Previous speech context. " * 20}),
            ("plain_decoder", {"timestamps": True}),
            ("translation", {"task": "st_eng"}),
            ("plain_group_ts", {"batch_size": 3, "timestamps": True}),
            ("translation_group", {"batch_size": 8, "task": "st_eng"}),
            ("beam_ts", {"beam_size": 5, "timestamps": True}),
            ("single_ts_again", {"timestamps": True}),
        ):
            enabled = name not in ("plain_decoder", "plain_group_ts")
            if model.ctc_draft != enabled:
                # Construction options bind session windows; rebuild only when this test changes them.
                model._sessions.clear()
            model.ctc_draft = enabled
            source = waves if name.startswith("stream") or "group" in name else waves[0]
            out = model.transcribe(source, lang=args.lang, **kwargs)
            results[name] = [asdict(x) for x in out] if isinstance(out, list) else asdict(out)
            dispatch[name] = captures(model)
            if args.finished_stop:
                for sess in model._sessions.values():
                    if not isinstance(sess, GreedySession):
                        continue
                    supported = (bool(getattr(model.backend, 'supports_finished_stop', False)) if sess.batch == 1
                                 else getattr(model.backend, 'supports_finished_groups', lambda rows: False)(sess.batch))
                    expected_guard = mode == 'default' and supported
                    if sess.skip_finished is not expected_guard:
                        raise RuntimeError(f'public finished-stop option mismatch: {mode}/{name}')
                    if any(bool(getattr(fn, 'skips_finished', False)) is not expected_guard
                           for fn in sess._steps.values()):
                        raise RuntimeError(f'actual finished-stop dispatch mismatch: {mode}/{name}')

            if name == "plain_decoder" and model._session(1).windows:
                raise RuntimeError("plain decoder check accidentally used drafted decoding")
            if name == "plain_decoder" and (model._session(1).select_workspace is not None) != (mode == "default"):
                raise RuntimeError("public option did not select expected plain greedy kernel")
            if name == "single_ts":
                if args.lazy_plain and model._session(1)._plain_ready != (mode == "serial"):
                    raise RuntimeError("plain decoder was prepared during drafted ASR")
                windows = [v.bufs for sess in model._sessions.values() for v in sess._verify.values()]
                active = any(w.fuse_accept and w.choose_workspace is not None for w in windows)
                if active != (mode == "default"):
                    raise RuntimeError(f"public option did not select expected kernels: {mode}")
            if mode == "default" and results[name] != expected[name]:
                raise RuntimeError(f"public default changed output after session reuse: {name}")
        results["lid"] = model.detect_language(waves[0])
        if mode == "default" and results["lid"] != expected["lid"]:
            raise RuntimeError("language identification changed")
        if getattr(model.backend, "failed", set()):
            raise RuntimeError("decoder failed and silently fell back during validation")
        expected = results
        records[mode] = dict(load_ms=load_ms, captures=dispatch, outputs=results)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    save(
        args.out / "public_api.json",
        dict(
            all_equal=True,
            decoder_kernel=args.decoder_kernel,
            lazy_plain=args.lazy_plain,
            finished_stop=args.finished_stop,
            checkpoint_sha256=sha256(files.checkpoint),
            engine_sha256=engine_hashes,
            gpu=torch.cuda.get_device_name(),
            commit=os.environ.get("OWSM_PROFILE_COMMIT"),
            modes=records,
            note="Functional public-option/session-reuse regression; no steady-state timing claim.",
        ),
    )
    print("public default/serial rollback: all requests equal", flush=True)


if __name__ == "__main__":
    main()
