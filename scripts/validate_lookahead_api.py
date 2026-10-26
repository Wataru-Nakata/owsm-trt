"""Public lookahead option on mixed speech/silence/noise, prompts and repeated session use."""

import argparse
import gc
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch
from profile_beam import save, sha256

from owsm_trt import OWSM, hub
from owsm_trt.audio import load_audio
from owsm_trt.build import engine_tag


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if not os.environ.get("PBS_JOBID"):
        ap.error("model/audio work requires a compute allocation")
    items = json.loads(args.manifest.read_text())[:5]
    if len(items) < 5:
        ap.error("five speech inputs are required")
    waves = [load_audio(x["path"]) for x in items]
    waves += [torch.zeros(32000), torch.randn(32000, generator=torch.Generator().manual_seed(54)) * .01]
    records, expected, hashes = {}, None, None
    for enabled in (False, True):
        model = OWSM.from_pretrained(args.model, ctc_draft=False, encoder_lookahead=enabled)
        assert model._encoder_lookahead is enabled
        files = hub.resolve(args.model)
        engine = hub.cache_dir() / files.name.replace("/", "--") / engine_tag(model.opts)
        current = (sha256(files.checkpoint), {p.name: sha256(p) for p in engine.glob("*.engine")})
        assert hashes is None or hashes == current
        hashes = current
        outputs, stages = {}, {}
        cases = [
            ("single", waves[:1], dict(beam_size=5, timestamps=True)),
            ("beam_ts", waves, dict(beam_size=5, batch_size=3, timestamps=True)),
            ("beam_notime", waves, dict(beam_size=5, batch_size=3)),
            ("beam_prompt", waves, dict(beam_size=5, batch_size=3, timestamps=True,
                                       prompt="Previous speech context. " * 20)),
            ("plain_ts", waves, dict(batch_size=3, timestamps=True)),
            ("translation", waves, dict(batch_size=3, task="st_eng")),
            ("single_again", waves[:1], dict(beam_size=5, timestamps=True)),
        ]
        for name, inputs, options in cases:
            outputs[name] = [asdict(r) for r in model.transcribe(inputs, lang="eng", **options)]
            stages[name] = sum(len(getattr(s, "_lookahead_stage", {})) for s in model._sessions.values())
            if name == "single":
                assert stages[name] == 0
            elif name != "single_again":
                assert bool(stages[name]) is enabled
            if expected is not None:
                assert outputs[name] == expected[name], f"lookahead changed {name} output"
        assert outputs['single'] == outputs['single_again'], "session reuse changed the single-input output"
        assert not getattr(model.backend, "failed", set()), "unexpected decoder fallback"
        records[str(enabled)] = dict(outputs=outputs, staging_sets_after_request=stages)
        save(args.out / "public_api.json", dict(all_equal=enabled, modes=records,
             checkpoint_sha256=hashes[0], engine_sha256=hashes[1], commit=os.environ.get("OWSM_PROFILE_COMMIT"),
             note="Functional comparison only; source transcripts do not annotate synthetic audio or translations."))
        expected = outputs
        del model
        gc.collect()
        torch.cuda.empty_cache()
    print("public lookahead option: all seven mixed-input cases equal", flush=True)


if __name__ == "__main__":
    main()
