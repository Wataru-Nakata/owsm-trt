"""Token ids <-> text, following ESPnet's token list + SentencePiece conversion."""

import re

import sentencepiece as spm

_TIME = re.compile(r"<\d+\.\d+>")


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
        self.notime = self.ids[notime]
        times = [i for i, t in enumerate(token_list) if _TIME.fullmatch(t)]
        self.first_time, self.last_time = min(times), max(times)
        # Language tokens sit between <nospeech> and the first task token <asr>.
        lo, hi = self.ids["<nospeech>"] + 1, self.ids["<asr>"]
        self.languages = {token_list[i][1:-1]: i for i in range(lo, hi)}
        self.tasks = {t[1:-1]: i for i, t in enumerate(token_list) if t == "<asr>" or t.startswith("<st_")}

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
        return self.first_time <= i <= self.last_time

    def encode(self, text: str) -> list[int]:
        return [self.ids.get(p, self.unk) for p in self.sp.encode(text, out_type=str)]

    def decode(self, ids: list[int]) -> str:
        """Text without special tokens (task, language, timestamps, ...)."""
        pieces = [self.tokens[i] for i in ids if i != self.blank]
        return self.sp.decode_pieces([p for p in pieces if not (p[0] == "<" and p[-1] == ">")])
