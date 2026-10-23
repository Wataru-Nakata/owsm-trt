"""Check the pure-torch OWSM reimplementation against official ESPnet (CPU, fp32).

Run inside the reference env (tools/espnet_ref), e.g.:

    PYTHONPATH=src tools/espnet_ref/.venv/bin/python scripts/parity_espnet.py \
        --model .cache/models/owsm_v4_base_102M --n 5

Compares the encoder output, the first decoder step and the greedy transcript (encoder-decoder
models), or the encoder output and the CTC best path (OWSM-CTC). ``--timestamps`` decodes with
timestamps and compares the generated token ids.
"""

import argparse
import json
import os

import soundfile as sf
import torch
import torch.nn.functional as F

from owsm_trt.config import load_espnet_config, parse_config
from owsm_trt.hub import resolve
from owsm_trt.modeling import load_model
from owsm_trt.pipeline import ctc_collapse
from owsm_trt.runtime import GreedySession, TimestampRules, TorchBackend
from owsm_trt.tokenizer import Tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=".cache/testdata/librispeech_dummy/manifest.json")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--lang", default="eng")
    ap.add_argument("--task", default="asr")
    ap.add_argument("--timestamps", action="store_true")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    files = resolve(args.model)
    raw = load_espnet_config(files.config)
    cfg = parse_config(raw)
    tok = Tokenizer(
        raw["token_list"],
        files.root / raw["bpemodel"],
        (raw.get("preprocessor_conf") or {}).get("notime_symbol", "<notimestamps>"),
    )
    ours = load_model(cfg, str(files.checkpoint))
    items = json.load(open(args.data))[: args.n]

    from espnet2.bin.s2t_inference import Speech2Text

    cwd = os.getcwd()
    os.chdir(files.root)  # ESPnet config paths (bpemodel, stats) are relative to the model root
    ref = Speech2Text(
        s2t_train_config=str(files.config),
        s2t_model_file=str(files.checkpoint),
        bpemodel=str(files.root / raw["bpemodel"]),
        device="cpu",
        beam_size=1,
        ctc_weight=0.0,
        lang_sym=f"<{args.lang}>",
        task_sym=f"<{args.task}>",
        predict_time=args.timestamps,
        nbest=1,
    )
    os.chdir(cwd)
    ref.s2t_model.eval()
    lang, task = tok.lang_id(args.lang), tok.task_id(args.task)

    if not cfg.ctc_only:
        rules = TimestampRules(tok.notime, tok.first_time, tok.last_time, tok.eos, cfg.vocab_size, "cpu")
        sess = GreedySession(
            TorchBackend(ours), cfg, batch=1, capacity=512, eos=tok.eos, dtype=torch.float32,
            device=torch.device("cpu"), use_cuda_graph=False, rules=rules,
        )  # fmt: skip
        primer = [tok.sos, lang, task] + ([] if args.timestamps else [tok.notime])

    mismatches = 0
    for it in items:
        wav, sr = sf.read(it["path"], dtype="float32")
        assert sr == cfg.sample_rate
        speech = F.pad(torch.from_numpy(wav), (0, cfg.num_samples - len(wav)))[None]  # [1, N]
        feats = ours.frontend(speech)

        if cfg.ctc_only:
            enc_ref, _ = ref._encode(speech, "<na>", f"<{args.lang}>", f"<{args.task}>")
            m = ours
            p = m.embed(torch.tensor([[tok.na]]))
            p = p * m.prompt_xscale + m.prompt_pe[: p.shape[1]]
            memory = m.prompt_proj(m.prompt_encoder(p))
            prefix = m.embed_proj(m.embed(torch.tensor([[lang, task]])))
            enc_ours = m.encoder(feats, prefix, memory, m.ctc)
            ids = ours(feats, torch.tensor([[tok.na]]), torch.tensor([[lang, task]]))[0].tolist()
            text_ref = ref.best_path(speech[0])[0][3]
            text_ours = tok.decode(ctc_collapse(ids, tok.blank))
            extra = ""
        else:
            enc_ref, _ = ref.s2t_model.encode(speech, torch.tensor([speech.shape[1]]))
            enc_ours = ours.encoder(feats)
            ys = torch.tensor([primer])
            logp_ref, _ = ref.s2t_model.decoder.forward_one_step(
                ys, torch.ones(1, len(primer), len(primer)).tril().bool(), enc_ref
            )
            sess.encode(feats)
            logits, _ = ours.decoder.step(
                ys, torch.zeros(1, dtype=torch.long), list(sess.self_k), list(sess.self_v),
                list(sess.cross_k), list(sess.cross_v),
            )  # fmt: skip
            extra = f" logp max|d|={(logp_ref - logits[:, -1].log_softmax(-1)).abs().max().item():.2e}"
            res = ref(speech[0, : len(wav)])[0]
            out = sess.decode(torch.tensor([primer]), 512)[0]
            if args.timestamps:
                text_ref, text_ours = res[2][2:], out
            else:
                text_ref, text_ours = res[3], tok.decode(out)
        assert enc_ref.shape == enc_ours.shape, (enc_ref.shape, enc_ours.shape)
        enc_diff = (enc_ref - enc_ours).abs().max().item()
        same = text_ref == text_ours
        mismatches += not same
        print(f"{it['id']}: enc max|d|={enc_diff:.2e}{extra} same={same}")
        if not same:
            print(f"  ref : {text_ref}\n  ours: {text_ours}")
    print(f"{len(items) - mismatches}/{len(items)} outputs identical")


if __name__ == "__main__":
    main()
