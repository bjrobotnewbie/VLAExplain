# 🧠 VLAExplain — Interpreting Vision-Language-Action (VLA) Models
# Copyright (C) 2026 Yafei Shi
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# For more information, please visit: https://github.com/bjrobotnewbie/VLAExplain

from utils.normalizer import AttentionNormalizer
from core.data_processor import StateParser


class BaseAnalyzer:
    """Base analyzer class for attention analysis functionality"""
    
    def __init__(self, normalization_method="log_normalize"):
        """
        Initialize the base analyzer
        
        Args:
            normalization_method (str): Method for attention weight normalization
        """
        self.attention_normalizer = AttentionNormalizer(method=normalization_method)
        self.is_state_to_origin = True
        self.state_parser = StateParser()
    
    @staticmethod
    def _is_integer(s):
        """
        Check if a string represents an integer
        
        Args:
            s (str): String to check
            
        Returns:
            bool: True if string is an integer, False otherwise
        """
        try:
            int(s)
            return True
        except ValueError:
            return False
        
    def merge_tokens_to_state(self, state_texts) -> None:
        """
        Merge dispersed tokens into original state strings
        
        This method processes a list of state text tokens and groups them according to
        specific delimiters and formatting rules to reconstruct the original state representation.
        
        Args:
            state_texts (list): List of state text tokens to be merged
            
        Returns:
            dict: Dictionary containing merged state information with keys:
                - 'state_texts_list': Grouped text segments
                - 'state_token_indices': Corresponding token indices
                - 'continuous_states': Continuous state values
                - 'discrete_states': Discrete state representations
        """
        item = []
        indices = []
        state_origin_merged = {
            "state_texts_list": [],
            "state_token_indices": [],
            "continuous_states": [],
            "discrete_states": []
        }
        
        for i, text in enumerate(state_texts):
            if i < 2:
                # Handle first two elements specially
                state_origin_merged['state_texts_list'].append([text])
                state_origin_merged['state_token_indices'].append([i])
            elif i == 2:
                # Skip the third element
                continue
            elif text == '':
                # Empty string indicates end of current group
                state_origin_merged['state_texts_list'].append(item)
                state_origin_merged['state_token_indices'].append(indices)
                item = []
                indices = []
            elif text == '-' or text == ';':
                # Delimiter characters indicate end of current group
                state_origin_merged['state_texts_list'].append(item)
                state_origin_merged['state_token_indices'].append(indices)
                item = []
                indices = []
                item.append(text)
                indices.append(i)
            else:
                # Add token to current group
                item.append(text)
                indices.append(i)
                
        # Process the final accumulated group
        for i, parts in enumerate(state_origin_merged['state_texts_list']):
            state_discretized = ''.join(parts)
            if self._is_integer(state_discretized):
                # Convert discrete state to continuous representation
                state_continuous = self.state_parser.process(int(state_discretized))
                state_origin_merged['continuous_states'].append(state_continuous)
            else:
                state_origin_merged['continuous_states'].append(state_discretized)
            state_origin_merged['discrete_states'].append(state_discretized)
            
        return state_origin_merged