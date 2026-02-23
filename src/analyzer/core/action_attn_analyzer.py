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
import cv2
from pathlib import Path
from transformers import AutoTokenizer
from PIL import Image
import pickle

from core.base_analyzer import BaseAnalyzer
from core.data_processor import LanguageInfoLoader
from visualization.bar_chart_visualizer import BarChartVisualizer
from visualization.equal_height_bar_visualizer import EqualHeightBarVisualizer
from visualization.heatmap_overlay_visualizer import HeatmapOverlayVisualizer
from config.settings import Settings
from utils.image_reshaper import ImageReshaper

class ActionAttnAnalyzer(BaseAnalyzer):
    """Action attention analyzer for processing and visualizing attention mechanisms"""
    
    def __init__(self, raw_image_dir, attention_dir, tokenizer_path, normalization_method="log_normalize"):
        """
        Initialize the action attention analyzer
        
        Args:
            raw_image_dir (str): Directory containing raw images
            attention_dir (str): Directory containing attention data files
            tokenizer_path (str): Path to the pre-trained tokenizer
            normalization_method (str): Method for attention weight normalization
        """
        super().__init__(normalization_method)
        self.raw_image_dir = Path(raw_image_dir)
        self.attention_dir = Path(attention_dir)
        self.patch_size = Settings.PATCH_SIZE_GRID
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.language_info = LanguageInfoLoader().process()
        self.bar_chart_visualizer = BarChartVisualizer(figsize=(35, 8), fontsize=24)
        self.equal_height_bar_visualizer = EqualHeightBarVisualizer(
            figsize=(35, 4), fontsize=24
        )
        self.heatmap_visualizer = HeatmapOverlayVisualizer(
            patch_size_grid=Settings.PATCH_SIZE_GRID,
        )
        self.image_reshaper = ImageReshaper(patch_size_grid=self.patch_size)
        
        # Interpolation methods and color schemes
        self.interpolation_methods = {
            'none': None,
            'nearest': cv2.INTER_NEAREST,
            'linear': cv2.INTER_LINEAR,
            'cubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }
        self.interpolation_method = 'cubic'
        self.current_colormap = 'jet'
        self.alpha = 0.5
        self.original_states = {}
        self.text_token_ids = {}
        self._initialize_data()

    def _initialize_data(self):
        """Initialize language information data structures"""
        for step, info_dict in self.language_info.items():
            if not info_dict:
                continue
            self.original_states[step] = info_dict['state'][0]
            self.text_token_ids[step] = info_dict['text_token_ids'][0]

    def set_normalization_method(self, method):
        """Dynamically update the normalization method"""
        self.attention_normalizer.method = method

    def set_interpolation_method(self, method):
        """
        Set the interpolation method for heatmap generation
        
        Args:
            method (str): Interpolation method name
        """
        if method in self.interpolation_methods:
            self.interpolation_method = method
        else:
            print(f"Warning: Unsupported interpolation method '{method}', using default 'cubic'")
            self.interpolation_method = 'cubic'

    def set_colormap(self, colormap):
        """
        Set the color scheme for visualizations
        
        Args:
            colormap (str): Color map name
        """
        self.current_colormap = colormap

    def set_alpha(self, alpha):
        """
        Set the transparency level for overlays
        
        Args:
            alpha (float): Transparency value between 0 and 1
        """
        self.alpha = alpha

    def load_attention_data(self, step):
        """
        Load attention data for a specific step
        
        Args:
            step (int): Step number to load attention data for
            
        Returns:
            dict or None: Processed attention data dictionary or None if file doesn't exist
        """
        attention_file = self.attention_dir / f"{step}_expert_attention.pkl"
        if not attention_file.exists():
            return None
            
        with open(attention_file, 'rb') as f:
            attention_dict = pickle.load(f)
        
        step_attention = attention_dict.get(step, None)
        if step_attention is None:
            return None
        
        # Convert data types
        processed_attention = {}
        for time_step, time_data in step_attention.items():
            processed_time_data = {}
            for layer_idx, attn_tensor in time_data.items():
                if attn_tensor.dtype == torch.bfloat16:
                    attn_tensor = attn_tensor.to(torch.float32)
                processed_time_data[layer_idx] = attn_tensor
            processed_attention[time_step] = processed_time_data
        
        return processed_attention

    def get_available_time_steps(self, attention_data):
        """
        Get list of available time steps
        
        Args:
            attention_data (dict): Attention data dictionary
            
        Returns:
            list: Sorted list of available time steps
        """
        if not attention_data:
            return [0]
        return sorted(attention_data.keys())

    def get_average_attention_head(self, attn_tensor):
        """
        Generate average pooled attention head
        
        Args:
            attn_tensor (torch.Tensor): Input attention tensor
            
        Returns:
            torch.Tensor: Average pooled attention tensor
        """
        return torch.mean(attn_tensor, dim=1, keepdim=True)
    
    def decode_text_tokens(self, token_ids):
        """
        Decode text tokens from token IDs
        
        Args:
            token_ids (list): List of token IDs
            
        Returns:
            str: Decoded text string
        """
        if len(token_ids) < Settings.LAN_INPUT_INDICES["text"][1]:
            return ""
        token_start_idx = Settings.LAN_INPUT_INDICES["text"][0]
        token_end_idx = Settings.LAN_INPUT_INDICES["text"][1]
        text_tokens = token_ids[token_start_idx:token_end_idx]  
        text = self.tokenizer.decode(text_tokens, skip_special_tokens=True)
        
        return text
    
    def decode_state_tokens(self, token_ids):
        """
        Decode state tokens from token IDs
        
        Args:
            token_ids (list): List of token IDs
            
        Returns:
            str: Decoded state string
        """
        if len(token_ids) < Settings.LAN_INPUT_INDICES["state"][1]:
            return ""
        state_start_idx = Settings.LAN_INPUT_INDICES["state"][0]
        state_end_idx = Settings.LAN_INPUT_INDICES["state"][1]
        state_tokens = token_ids[state_start_idx:state_end_idx] 
        state = self.tokenizer.decode(state_tokens, skip_special_tokens=True)
        
        return state

    def normalize_attention(self, attn_weights):
        """
        Normalize attention weights using the configured method
        
        Args:
            attn_weights (torch.Tensor): Raw attention weights
            
        Returns:
            torch.Tensor: Normalized attention weights
        """
        return self.attention_normalizer.process(attn_weights)

    def overlay_attention_on_image(self, img, patch_attn):
        """
        Overlay attention weights on image to generate heatmap while maintaining original image dimensions
        
        Args:
            img (PIL.Image): Original image
            patch_attn (torch.Tensor): Patch attention weights
            
        Returns:
            PIL.Image: Image with attention heatmap overlay
        """
        img = np.array(img)
        img_h, img_w = img.shape[:2]  # Get original image dimensions
        
        # Normalize attention
        patch_attn = self.normalize_attention(patch_attn)
        
        # Calculate patch count
        attn_img = self.image_reshaper.process(
            patch_attn, 
            image_size=(img_h, img_w),
            interpolation_method=self.interpolation_method
        )
        overlay_img = self.heatmap_visualizer.overlay(
            img,
            attn_img, 
            self.alpha,
            self.current_colormap,
            has_grid=False
        )

        return overlay_img

    def visualize_text_attention(self, text, text_attn):
        """
        Generate text attention visualization
        
        Args:
            text (str): Text content
            text_attn (torch.Tensor): Text attention weights
            
        Returns:
            PIL.Image: Text attention visualization image
        """
        # Normalize attention weights
        text_attn = self.normalize_attention(text_attn)

        # Process text tokens
        actual_text_len = len(text_attn)
        text_attn_limited = text_attn[:actual_text_len]

        if text:
            text_tokens = self.tokenizer.tokenize(text)
            text_tokens = [t.replace('▁', '') for t in text_tokens]
            while len(text_tokens) < actual_text_len:
                text_tokens.append('')
            text_tokens = text_tokens[:actual_text_len]
        else:
            text_tokens = [''] * actual_text_len

        text_img = self.bar_chart_visualizer.render(
            values=text_attn_limited,
            labels=text_tokens,
            title="Text Sequence Attention Distribution",
            ylabel="Normalized Attention Weights",
            cmap=self.current_colormap
        )

        return text_img

    def visualize_state_attention(self, state, state_attn, original_state=None):
        """
        Generate state attention visualization
        
        Args:
            state (str): State content
            state_attn (torch.Tensor): State attention weights
            original_state (optional): Original state values for reference
            
        Returns:
            PIL.Image: State attention visualization image
        """
    
        # Process state tokens
        state_tokens = self.tokenizer.tokenize(state)
        state_tokens = [s.replace('▁', '') for s in state_tokens]

        # Parse state tokens according to configuration
        if self.is_state_to_origin:
            state_tokens, state_token_indices = self.get_states_and_indices(state_tokens)
            state_attn_new = []
            for indices in state_token_indices:
                attns = [state_attn[index] for index in indices]
                attns_mean = torch.mean(torch.stack(attns), dim=0)
                state_attn_new.append(attns_mean)
            state_attn_new = torch.tensor(state_attn_new).squeeze()[:10]
            state_attn_normalized = self.normalize_attention(state_attn_new).squeeze()[:10]
            state_tokens = state_tokens[:10]
            if original_state is not None:
                state_tokens = ['State', ':']+original_state.tolist()
        else:
            state_attn_normalized = self.normalize_attention(state_attn)

        state_img = self.bar_chart_visualizer.render(
            values=state_attn_normalized,
            labels=state_tokens,
            title="State Sequence Attention Distribution",
            ylabel="Normalized Attention Weights",
            cmap=self.current_colormap
        )

        return state_img

    def get_states_and_indices(self, state_texts):
        """
        Parse states into original format
        
        Args:
            state_texts (list): List of state text tokens
            
        Returns:
            tuple: (continuous_states, state_token_indices)
        """
        state_origin_merged = self.merge_tokens_to_state(state_texts)
        return state_origin_merged['continuous_states'], state_origin_merged['state_token_indices']

    def get_step_images(self, step, view_key):
        """
        Get images for a specific step and view
        
        Args:
            step (int): Step number
            view_key (str): View identifier ('image1' or 'image2')
            
        Returns:
            PIL.Image: Requested image or gray placeholder if not found
        """
        step_str = f"{step:04d}"
        img_path = self.raw_image_dir / f"step_{step_str}_{view_key}.jpg"
        if img_path.exists():
            return Image.open(img_path).convert('RGB')
        else:
            return Image.new('RGB', (256, 256), color='gray')

    def update_visualization(self, step, time_step, attention_head, layer_idx, alpha, normalization_method, interpolation_method, colormap):
        """
        Core update function: Update all visualizations based on interactive parameters
        
        Args:
            step (int): Current step number
            time_step (int): Time step within the sequence
            attention_head (str): Attention head selection
            layer_idx (int): Layer index
            alpha (float): Transparency level
            normalization_method (str): Attention normalization method
            interpolation_method (str): Heatmap interpolation method
            colormap (str): Color scheme for visualizations
            
        Returns:
            tuple: Tuple of visualization images (raw_img1, raw_img2, view1_img, view2_img, text_img, state_img, action_seq_img)
        """
        # Dynamically update normalization method, interpolation method, and color scheme
        self.set_normalization_method(normalization_method)
        self.set_interpolation_method(interpolation_method)
        self.set_colormap(colormap)
        self.set_alpha(alpha)
        
        # Load attention data
        attention_data = self.load_attention_data(step)
        if attention_data is None:
            empty_img = Image.new('RGB', (256, 256), color='gray')
            return empty_img, empty_img, empty_img, empty_img, empty_img, empty_img, empty_img

        # Check if time step is valid
        available_time_steps = self.get_available_time_steps(attention_data)
        if time_step not in available_time_steps:
            time_step = available_time_steps[0] if available_time_steps else 0

        # Get data for specified time step
        time_data = attention_data.get(time_step, {})
        layer_idx_str = str(layer_idx)
        if layer_idx_str not in time_data:
            layer_idx_str = list(time_data.keys())[0] if time_data else "0"
        attn_tensor = time_data.get(layer_idx_str, None)
        if attn_tensor is None:
            empty_img = Image.new('RGB', (256, 256), color='gray')
            return empty_img, empty_img, empty_img, empty_img, empty_img, empty_img, empty_img

        # Process attention head
        if attention_head == "Average Pooling Head":
            attn_tensor = self.get_average_attention_head(attn_tensor)
            head_idx = 0
        else:
            head_idx = int(attention_head.replace("Head ", "")) - 1

        # Extract single head attention [1, 50, 1018]
        single_head_attn = attn_tensor[:, head_idx:head_idx+1, :, :]
        step_inside = step % Settings.ACTION_NUM
        single_head_attn = single_head_attn[0, 0, step_inside]  # [50, 1018]

        # Load original images
        original_view1_img = self.get_step_images(step, 'image1')
        original_view2_img = self.get_step_images(step, 'image2')

        # Generate attention heatmaps for view1 and view2
        view1_attn = single_head_attn[:Settings.SEGMENTS_INDICES['image1'][1]]
        view1_overlay = self.overlay_attention_on_image(original_view1_img, view1_attn)
        view1_overlay = view1_overlay.resize(original_view1_img.size)

        view2_attn = single_head_attn[Settings.SEGMENTS_INDICES['image2'][0]:Settings.SEGMENTS_INDICES['image2'][1]]
        view2_overlay = self.overlay_attention_on_image(original_view2_img, view2_attn)
        view2_overlay = view2_overlay.resize(original_view2_img.size)

        # Save raw images separately
        raw_img1 = self.heatmap_visualizer.render_image(original_view1_img, f"Step {step} - View1 (Time Step: {time_step})")
        raw_img2 = self.heatmap_visualizer.render_image(original_view2_img, f"Step {step} - View2 (Time Step: {time_step})")
        
        # Save view images separately
        view1_img = self.heatmap_visualizer.render_image(view1_overlay, "View 1 - Attention Heatmap")
        view2_img = self.heatmap_visualizer.render_image(view2_overlay, "View 2 - Attention Heatmap")

        # Decode text and state
        language_tokens_attn = single_head_attn[Settings.SEGMENTS_INDICES['image3'][1]:Settings.SEGMENTS_INDICES['image3'][1]+200]
        text_attn = language_tokens_attn[Settings.LAN_INPUT_INDICES['text'][0]:Settings.LAN_INPUT_INDICES['text'][1]]
        state_attn = language_tokens_attn[Settings.LAN_INPUT_INDICES['state'][0]:Settings.LAN_INPUT_INDICES['state'][1]]

        text_img, state_img = None, None
        if step in self.text_token_ids:
            token_ids = self.language_info[step]['text_token_ids'][0]
            text = self.decode_text_tokens(token_ids)
            text_img = self.visualize_text_attention(text, text_attn)
            state = self.decode_state_tokens(token_ids)
            original_states = self.original_states.get(step)
            state_img = self.visualize_state_attention(state, state_attn, original_states)
        else:
            text_img, state_img = None, None
            
        # Action sequence visualization
        action_attn = single_head_attn[-50:]
        action_attn = self.normalize_attention(action_attn)
        action_seq_img = self.equal_height_bar_visualizer.render(
            action_attn,
            range(len(action_attn)),
            title=f"Action Sequence - Step {step}",
            ylabel="Normalized Attention Weights",
            cmap=self.current_colormap
        )

        # Return all images
        return raw_img1, raw_img2, view1_img, view2_img, text_img, state_img, action_seq_img