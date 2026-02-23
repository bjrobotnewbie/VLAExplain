# 🧠 VLAExplain — Interpreting Vision-Language-Action (VLA) Models
# Copyright (C) 2026 Yafei Shi
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# For more information, please visit: https://github.com/bjrobotnewbie/VLAExplain

from abc import ABC, abstractmethod
from transformers import AutoTokenizer


class DataProcessor(ABC):
    """Abstract base class for data processors."""
    
    @abstractmethod
    def process(self, data, **kwargs):
        """Abstract method to process data. Must be implemented by subclasses."""
        pass


class TokenDecoder(DataProcessor):
    """Token Decoder for converting tokens to text."""
    
    def __init__(self, tokenizer: AutoTokenizer):
        """Initialize the TokenDecoder with a tokenizer."""
        self.tokenizer = tokenizer

    def process(self, tokens):
        """Decode tokens into a list of text strings.
        
        Args:
            tokens: List of token IDs to decode.
            
        Returns:
            List of decoded text strings with special tokens removed and whitespace stripped.
        """
        decoded_texts = []
        for token in tokens:
            decoded_text = self.tokenizer.decode(token, skip_special_tokens=True).strip()
            decoded_texts.append(decoded_text)
        return decoded_texts


class StateParser(DataProcessor):
    """State Parser for converting discrete states to continuous states."""
    
    def __init__(self):
        """Initialize the StateParser."""
        super().__init__()

    def process(self, discretized_states):
        """Convert discrete states to continuous states.
        
        Args:
            discretized_states: Array of discrete state values.
            
        Returns:
            Array of continuous state values computed using a fixed bin width.
        """
        bin_width = 2.0 / 256  # (1 - (-1)) / 256
        continuous_states = discretized_states * bin_width + (-1.0) + bin_width
        return continuous_states
    
class LanguageInfoLoader(DataProcessor):
    """Singleton loader for language information."""
    _instance = None
    _language_info = None

    def __new__(cls):
        """Ensure only one instance of LanguageInfoLoader exists (Singleton pattern)."""
        if cls._instance is None:
            cls._instance = super(LanguageInfoLoader, cls).__new__(cls)
        return cls._instance

    def process(self):
        """Load and cache language information if not already loaded.
        
        Returns:
            Cached language information.
        """
        if self._language_info is None:
            from lerobot.policies.pi05.collector import ATTENTION_TRACER
            collector = ATTENTION_TRACER
            collector.read_language_info()
            self._language_info = collector.language_info
        return self._language_info