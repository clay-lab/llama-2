# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
from logging import getLogger
from typing import List

from sentencepiece import SentencePieceProcessor

import torch

logger = getLogger(__name__)


class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()
    
    @property
    def all_special_tokens(self) -> List[str]:
        special_tokens = [
            getattr(self, f'{pfx}_token')
            for pfx in {'sep', 'pad', 'mask', 'bos'}
                if hasattr(self, f'{pfx}_token')
        ]
        return [t for t in special_tokens if t is not None]
    
    def tokenize(self, *args, **kwargs) -> List[int]:
        '''Wrapper around encode for HF compatibility.'''
        return self.encode(*args, **kwargs)
    
    def encode(self, s: str, bos: bool, eos: bool, one_word: bool = False) -> List[int]:
        assert type(s) is str
        if one_word:
            s = s.strip()
            assert not ' ' in s, f'You are trying to tokenize a string with spaces as a single word! ({s!r})'
        
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        if isinstance(t, torch.Tensor):
            if len(t.size()) > 0:
                t = [x.item() for x in t]
            else:
                t = [t.item()]
        # the llama sp model doesn't decode the pad id, so we need to exclude it here
        t = [t for t in t if not t == self.pad_id]
        return self.sp_model.decode(t)
    
    def batch_decode(self, ts: List[List[int]]) -> List[str]:
        '''
        Allows us to use the same interface for LLaMA 2 tokenizer and HF tokenizers.
        '''
        return [self.decode(t) for t in ts]
