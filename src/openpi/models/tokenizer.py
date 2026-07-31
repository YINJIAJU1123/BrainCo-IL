import logging

import numpy as np
import sentencepiece

import openpi.shared.download as download


class PaligemmaTokenizer:
    """Tokenizer shared by PI0 and PI0.5."""

    def __init__(self, max_len: int = 48):
        self._max_len = max_len
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as file:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=file.read())

    def tokenize(
        self,
        prompt: str,
        state: np.ndarray | None = None,
        *,
        task_action_prompt: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        elif task_action_prompt:
            # VLASH 官方 PI0.5 state_cond 路径只在 prompt 中保留 task,
            # 连续 state 由 Action Expert 的 adaRMS condition 接收.
            full_prompt = f"Task: {cleaned_text};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")

        token_count = len(tokens)
        if token_count < self._max_len:
            padding = [False] * (self._max_len - token_count)
            mask = [True] * token_count + padding
            tokens += padding
        else:
            if token_count > self._max_len:
                logging.warning(
                    "Token length (%d) exceeds max length (%d), truncating.",
                    token_count,
                    self._max_len,
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)
