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
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Literal, Callable
from config.settings import Settings

def extract_attention_weights(
    attn_weights: torch.Tensor,
    cache_segment_indices: Dict[str, Tuple[int, int]],
) -> Dict[str, torch.Tensor]:
    """
    Extract attention weights for different modules from raw attention tensors.
    
    This function separates the attention weights into query and key components
    for each module based on predefined segment indices.
    
    Args:
        attn_weights: Raw attention weights with shape (1, num_heads, seq_len, seq_len)
        cache_segment_indices: Dictionary mapping module names to their index ranges
        
    Returns:
        Dictionary containing extracted attention weights with keys formatted as 
        "{module}_q" for queries and "{module}_k" for keys
    """
    atten = attn_weights.squeeze(0)  # Remove batch dimension
    extracted_atten = {}

    for module_name, (start_idx, end_idx) in cache_segment_indices.items():
        extracted_atten[f"{module_name}_q"] = atten[:, start_idx:end_idx, :]
        extracted_atten[f"{module_name}_k"] = atten[:, :, start_idx:end_idx]

    return extracted_atten


def _compute_attention_between_modules(
    extracted_atten: Dict[str, torch.Tensor],
    source_modules: list,
    target_modules: list,
    cache_segment_indices: Dict[str, Tuple[int, int]],
    prefix: str = "",
    pool_func: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute attention weights between source and target modules.
    
    This helper function calculates cross-attention patterns between specified
    source and target modules, optionally applying pooling operations.
    
    Args:
        extracted_atten: Dictionary of extracted attention weights
        source_modules: List of source module names
        target_modules: List of target module names
        cache_segment_indices: Module index range dictionary
        prefix: Prefix for output key names (e.g., "global_")
        pool_func: Optional pooling function to apply to attention weights
        
    Returns:
        Dictionary of computed attention weights between module pairs
    """
    result = {}
    for src in source_modules:
        for tgt in target_modules:
            tgt_k_start, tgt_k_end = cache_segment_indices[tgt]
            atten = extracted_atten[f"{src}_q"][:, :, tgt_k_start:tgt_k_end]

            # Apply pooling function if provided
            if pool_func is not None:
                atten = pool_func(atten, dim=1)  # Pool along query dimension

            result[f"{prefix}{src}_to_{tgt}"] = atten
    return result


def compute_fine_grained_attention(
    extracted_atten: Dict[str, torch.Tensor],
    cache_segment_indices: Dict[str, Tuple[int, int]],
) -> Dict[str, torch.Tensor]:
    """
    Compute fine-grained attention weights between different modalities.
    
    Calculates detailed cross-attention patterns including:
    - Image to text attention
    - Text to image attention
    - Bidirectional text attention
    
    Args:
        extracted_atten: Dictionary of extracted attention weights
        cache_segment_indices: Module index range dictionary
        
    Returns:
        Dictionary containing all fine-grained attention patterns
    """
    # Image → Text attention
    image_to_cross = _compute_attention_between_modules(
        extracted_atten, Settings.MODULES_IMAGE, Settings.MODULES_CROSS, cache_segment_indices
    )
    # Text → Image attention
    cross_to_image = _compute_attention_between_modules(
        extracted_atten, Settings.MODULES_CROSS, Settings.MODULES_IMAGE, cache_segment_indices
    )


    return {
        **image_to_cross,
        **cross_to_image,
    }


def compute_global_attention(
    extracted_atten: Dict[str, torch.Tensor],
    cache_segment_indices: Dict[str, Tuple[int, int]],
) -> Dict[str, torch.Tensor]:
    """
    Compute global attention weights using average pooling.
    
    Calculates averaged attention patterns across all heads for:
    - Global image to text attention
    - Global text to image attention
    - Global bidirectional text attention
    
    Args:
        extracted_atten: Dictionary of extracted attention weights
        cache_segment_indices: Module index range dictionary
        
    Returns:
        Dictionary containing all global attention patterns
    """

    def avg_pool_attention(atten: torch.Tensor, dim: int) -> torch.Tensor:
        """Average pooling function for attention tensors"""
        return torch.mean(atten, dim=dim, keepdim=True)

    # Global Image → Text attention
    image_to_cross = _compute_attention_between_modules(
        extracted_atten,
        Settings.MODULES_IMAGE,
        Settings.MODULES_CROSS,
        cache_segment_indices,
        prefix="global_",
        pool_func=avg_pool_attention,
    )
    # Global Text → Image attention
    cross_to_image = _compute_attention_between_modules(
        extracted_atten,
        Settings.MODULES_CROSS,
        Settings.MODULES_IMAGE,
        cache_segment_indices,
        prefix="global_",
        pool_func=avg_pool_attention,
    )

    return {
        **image_to_cross,
        **cross_to_image,
    }


def merge_multi_head_attention(
    atten: torch.Tensor,
    strategy: Literal["mean", "sum", "max", "concat", "weighted"] = "mean",
    head_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Merge multi-head attention weights using specified strategy.
    
    Supports multiple merging approaches including mean averaging, summation,
    maximum selection, concatenation, and weighted combination.
    
    Args:
        atten: Multi-head attention tensor with shape (num_heads, q_len, k_len)
        strategy: Merging strategy to apply
        head_weights: Head importance weights (required for weighted strategy)
        
    Returns:
        Merged attention tensor according to specified strategy
        
    Raises:
        ValueError: If unsupported strategy is provided or missing head_weights
    """
    num_heads, q_len, k_len = atten.shape

    strategies: Dict[str, Callable] = {
        "mean": lambda x: torch.mean(x, dim=0, keepdim=True),
        "sum": lambda x: F.softmax(torch.sum(x, dim=0, keepdim=True).flatten(1), dim=-1).reshape(1, q_len, k_len),
        "max": lambda x: torch.max(x, dim=0, keepdim=True)[0],
        "concat": lambda x: x.permute(1, 0, 2).reshape(q_len, -1),
        "weighted": lambda x: torch.sum(x * F.softmax(head_weights, dim=0).reshape(num_heads, 1, 1), dim=0, keepdim=True),
    }

    if strategy not in strategies:
        raise ValueError(f"Unsupported merging strategy: {strategy}")

    if strategy == "weighted" and head_weights is None:
        raise ValueError("Weighted strategy requires head_weights parameter")

    return strategies[strategy](atten)


def merge_all_attention(
    atten_dict: Dict[str, torch.Tensor],
    strategy: Literal["mean", "sum", "max", "concat", "weighted"] = "mean",
    head_weights: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Batch merge all multi-head attention weights in a dictionary.
    
    Applies the specified merging strategy to all attention tensors in the dictionary.
    
    Args:
        atten_dict: Dictionary containing multi-head attention tensors
        strategy: Merging strategy to apply to all tensors
        head_weights: Head importance weights (required for weighted strategy)
        
    Returns:
        Dictionary with all attention tensors merged according to strategy
    """
    return {
        key: merge_multi_head_attention(atten, strategy, head_weights)
        for key, atten in atten_dict.items()
    }


def compute_and_merge_attention(
    attn_weights: torch.Tensor,
    merge_strategy: Literal["mean", "sum", "max", "concat", "weighted"] = "mean",
    head_weights: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Main pipeline function for complete attention computation and merging.
    
    This function orchestrates the entire attention processing pipeline:
    1. Extracts attention weights from different modules
    2. Computes fine-grained attention patterns
    3. Computes global attention patterns
    4. Merges multi-head attention using specified strategy
    
    Args:
        attn_weights: Raw attention weights with shape (1, num_heads, seq_len, seq_len)
        merge_strategy: Strategy for merging multi-head attention
        head_weights: Head importance weights (required for weighted strategy)
        
    Returns:
        Tuple containing four dictionaries:
        - fine_grained: Original multi-head fine-grained attention
        - global_atten: Original multi-head global attention
        - merged_fine_grained: Merged fine-grained attention
        - merged_global_atten: Merged global attention
        
    Raises:
        ValueError: If input tensor shape is incorrect
    """
    if attn_weights.dim() != 4 or attn_weights.shape[0] != 1:
        raise ValueError("Input attention weights must have shape (1, num_heads, seq_len, seq_len)")

    extracted_atten = extract_attention_weights(attn_weights, Settings.SEGMENTS_INDICES)
    fine_grained = compute_fine_grained_attention(extracted_atten, Settings.SEGMENTS_INDICES)
    global_atten = compute_global_attention(extracted_atten, Settings.SEGMENTS_INDICES)

    if merge_strategy == "weighted" and head_weights is not None:
        head_weights = head_weights.to(attn_weights.device)

    merged_fine_grained = merge_all_attention(fine_grained, merge_strategy, head_weights)
    merged_global_atten = merge_all_attention(global_atten, merge_strategy, head_weights)

    return fine_grained, global_atten, merged_fine_grained, merged_global_atten