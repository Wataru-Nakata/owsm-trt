"""Token ids <-> text, following ESPnet's token list + SentencePiece conversion."""

import re

import sentencepiece as spm

_TIME = re.compile(r"<\d+\.\d+>")
_TASK = re.compile(r"<(asr|pr|g2p|p2g|st_\w+)>")


class Tokenizer:
    def __init__(self, token_list: list[str], bpe_model: str, notime: str = "<notimestamps>"):
        self.tokens = token_list
        self.ids = {t: i for i, t in enumerate(token_list)}
        self.sp = spm.SentencePieceProcessor(model_file=str(bpe_model))
        self.blank = self.ids["<blank>"]
        self.unk = self.ids["<unk>"]
        self.na = self.ids["<na>"]
        self.sos = self.ids["<sos>"]
        self.eos = self.ids["<eos>"]
        self.sop = self.ids["<sop>"]
        self.notime = self.ids.get(notime)  # CTC-only models have no timestamps
        times = [i for i, t in enumerate(token_list) if _TIME.fullmatch(t)]
        self.first_time, self.last_time = (min(times), max(times)) if times else (None, None)
        # Language tokens sit between <nospeech> (<nolang> in OWSM-CTC) and the first task token.
        self.tasks = {t[1:-1]: i for i, t in enumerate(token_list) if _TASK.fullmatch(t)}
        lo = self.ids["<nospeech>" if "<nospeech>" in self.ids else "<nolang>"] + 1
        self.languages = {token_list[i][1:-1]: i for i in range(lo, min(self.tasks.values()))}
        # tokens a CTC head never writes but the decoder does: drafted decoding skips over them
        times = range(self.first_time, self.last_time + 1) if times else range(0)
        self.not_drafted = frozenset(times) | ({self.notime} if self.notime is not None else set())

    def lang_id(self, lang: str) -> int:
        key = lang.strip("<>")
        if key not in self.languages:
            raise ValueError(f"unknown language {lang!r}; choose from {sorted(self.languages)}")
        return self.languages[key]

    def task_id(self, task: str) -> int:
        key = task.strip("<>")
        if key not in self.tasks:
            raise ValueError(f"unknown task {task!r}; choose from {sorted(self.tasks)}")
        return self.tasks[key]

    def is_time(self, i: int) -> bool:
        return self.first_time is not None and self.first_time <= i <= self.last_time

    def encode(self, text: str) -> list[int]:
        return [self.ids.get(p, self.unk) for p in self.sp.encode(text, out_type=str)]

    def decode(self, ids: list[int]) -> str:
        """Text without special tokens (task, language, timestamps, ...)."""
        pieces = [self.tokens[i] for i in ids if i != self.blank]
        return self.sp.decode_pieces([p for p in pieces if not (p[0] == "<" and p[-1] == ">")])
