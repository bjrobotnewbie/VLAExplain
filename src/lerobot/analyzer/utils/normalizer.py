# 🧠 VLAExplain — Interpreting Vision-Language-Action (VLA) Models
# Copyright (C) 2026 Yafei Shi
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# For more information, please visit: https://github.com/bjrobotnewbie/VLAExplain

import torch
import numpy as np
from core.data_processor import DataProcessor


class AttentionNormalizer(DataProcessor):
    """Attention weight normalizer"""
    
    def __init__(self, method="log_normalize"):
        """
        Initialize the normalizer
        Args:
            method (str): Normalization method
        """
        self.method = method

    def process(self, attn_weights, **kwargs):
        """Normalize attention weights to enhance visual distinction"""
        if torch.is_tensor(attn_weights):
            # Convert tensor to numpy array; if tensor is BFloat16, convert to float32 first
            if attn_weights.dtype == torch.bfloat16:
                attn_weights = attn_weights.to(torch.float32)
            attn_weights = attn_weights.detach().cpu().numpy()  # Use detach() to ensure no gradients
        else:
            attn_weights = np.asarray(attn_weights)  # Ensure it's a numpy array

        # Normalize based on the selected method
        if self.method == "log_normalize":
            return self._log_normalize(attn_weights)
        elif self.method == "min_max":
            return self._min_max_normalize(attn_weights)
        elif self.method == "softmax":
            return self._softmax_normalize(attn_weights)
        elif self.method == "z_score":
            return self._z_score_normalize(attn_weights)
        elif self.method == "robust":
            return self._robust_normalize(attn_weights)
        elif self.method == "power":
            power = kwargs.get("power", 2.0)
            return self._power_normalize(attn_weights, power)
        elif self.method == "sigmoid":
            return self._sigmoid_normalize(attn_weights)
        elif self.method == "unit_vector":
            return self._unit_vector_normalize(attn_weights)
        else:
            # Default to log normalization
            return self._log_normalize(attn_weights)

    def _log_normalize(self, attn_weights):
        """Logarithmic normalization to enhance visibility of small differences"""
        attn_weights = np.clip(attn_weights, 1e-8, np.max(attn_weights))  # Prevent log from going to negative infinity
        log_attn = np.log(attn_weights)
        norm_attn = (log_attn - np.min(log_attn)) / (np.max(log_attn) - np.min(log_attn) + 1e-8)
        return norm_attn

    def _min_max_normalize(self, attn_weights):
        """Min-max normalization to [0, 1]"""
        min_val = np.min(attn_weights)
        max_val = np.max(attn_weights)
        if max_val - min_val == 0:
            return np.zeros_like(attn_weights)
        return (attn_weights - min_val) / (max_val - min_val)

    def _softmax_normalize(self, attn_weights):
        """Softmax normalization to highlight high values"""
        # Prevent numerical overflow
        attn_weights_stable = attn_weights - np.max(attn_weights, axis=-1, keepdims=True)
        exp_vals = np.exp(attn_weights_stable)
        softmax_vals = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
        return softmax_vals

    def _z_score_normalize(self, attn_weights):
        """Z-score normalization (mean 0, standard deviation 1)"""
        mean_val = np.mean(attn_weights)
        std_val = np.std(attn_weights)
        if std_val == 0:
            return np.zeros_like(attn_weights)
        z_scores = (attn_weights - mean_val) / std_val
        # Map z-scores back to [0,1] range
        abs_z_scores = np.abs(z_scores)
        max_abs = np.max(abs_z_scores)
        if max_abs == 0:
            return np.zeros_like(z_scores)
        # Normalize using absolute values, then map to [0,1]
        return (z_scores + max_abs) / (2 * max_abs)

    def _robust_normalize(self, attn_weights):
        """Robust normalization using median and interquartile range to reduce outlier impact"""
        median_val = np.median(attn_weights)
        q75, q25 = np.percentile(attn_weights, [75, 25])
        iqr = q75 - q25  # Interquartile range
        if iqr == 0:
            return np.zeros_like(attn_weights)
        robust_norm = (attn_weights - median_val) / iqr
        # Map robust normalization results back to [0,1] range
        abs_values = np.abs(robust_norm)
        max_abs = np.max(abs_values)
        if max_abs == 0:
            return np.zeros_like(robust_norm)
        return (robust_norm + max_abs) / (2 * max_abs)

    def _power_normalize(self, attn_weights, power=2.0):
        """Power normalization to emphasize large value differences"""
        # Ensure all values are positive
        attn_weights_positive = attn_weights - np.min(attn_weights) + 1e-8
        powered = np.power(attn_weights_positive, power)
        min_powered = np.min(powered)
        max_powered = np.max(powered)
        if max_powered - min_powered == 0:
            return np.zeros_like(powered)
        return (powered - min_powered) / (max_powered - min_powered)

    def _sigmoid_normalize(self, attn_weights):
        """Sigmoid normalization for smooth transitions"""
        sigmoid_vals = 1 / (1 + np.exp(-attn_weights))
        # Map sigmoid results to [0,1]
        min_sigmoid = np.min(sigmoid_vals)
        max_sigmoid = np.max(sigmoid_vals)
        if max_sigmoid - min_sigmoid == 0:
            return np.zeros_like(sigmoid_vals)
        return (sigmoid_vals - min_sigmoid) / (max_sigmoid - min_sigmoid)

    def _unit_vector_normalize(self, attn_weights):
        """Unit vector normalization using L2 norm"""
        # Flatten all dimensions except the last one, normalize along the last dimension
        original_shape = attn_weights.shape
        flattened = attn_weights.reshape(-1, original_shape[-1])
        
        # Compute L2 norm
        norms = np.linalg.norm(flattened, ord=2, axis=1, keepdims=True)
        # Prevent division by zero
        norms = np.where(norms == 0, 1, norms)
        normalized = flattened / norms
        
        return normalized.reshape(original_shape)