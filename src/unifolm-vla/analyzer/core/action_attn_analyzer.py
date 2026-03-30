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
import logging
from pathlib import Path
from transformers import AutoTokenizer
from PIL import Image
import pickle

logger = logging.getLogger(__name__)

from core.base_analyzer import BaseAnalyzer
from core.data_processor import LanguageInfoLoader, TokenDecoder
from visualization.bar_chart_visualizer import BarChartVisualizer
from visualization.equal_height_bar_visualizer import EqualHeightBarVisualizer
from visualization.heatmap_overlay_visualizer import HeatmapOverlayVisualizer
from visualization.module_heatmap_visualizer import ModuleHeatmapVisualizer
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
        self.token_decoder = TokenDecoder(self.tokenizer)  # Add token decoder for clean text labels
        self.language_info = LanguageInfoLoader().process()
        self.bar_chart_visualizer = BarChartVisualizer(figsize=(35, 8), fontsize=24)
        self.equal_height_bar_visualizer = EqualHeightBarVisualizer(
            figsize=(35, 4), fontsize=24
        )
        self.heatmap_visualizer = HeatmapOverlayVisualizer(
            patch_size_grid=Settings.PATCH_SIZE_GRID,
        )
        self.module_heatmap_visualizer = ModuleHeatmapVisualizer(figsize=(20, 6), dpi=150)
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
        
        # DIT expert network configuration
        self.target_action_idx = 34  # 35th action token (0-indexed)
        self.dit_query_dim = 42  # state(2) + future(32) + actions(8)
        self.dit_even_key_dim = 366  # Even layer cross-attention key dimension
        self.dit_odd_key_dim = 42   # Odd layer self-attention key dimension

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
                # Ensure layer index is stored as string for consistent lookup
                processed_time_data[str(layer_idx)] = attn_tensor
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

        # Process text tokens - decode token IDs to get clean text labels
        actual_text_len = len(text_attn)
        text_attn_limited = text_attn[:actual_text_len]

        if text:
            # Get the current step's text token IDs from language_info
            # Find a valid step to get token_ids
            available_steps = list(self.language_info.keys())
            if available_steps:
                sample_step = available_steps[0]
                text_token_ids = self.language_info[sample_step]['text_token_ids'][0]
                
                # Decode token IDs to get clean text (same as Language Analyzer)
                decoded_texts = self.token_decoder.process(text_token_ids)
                
                # Extract only task-related tokens using Settings.LAN_INPUT_INDICES
                if hasattr(Settings, 'LAN_INPUT_INDICES') and 'text' in Settings.LAN_INPUT_INDICES:
                    text_start_idx, text_end_idx = Settings.LAN_INPUT_INDICES['text']
                    # Note: Settings returns (start, format_instruction_start)
                    # We need [start, format_instruction_start - 1] to exclude format instructions
                    text_tokens = decoded_texts[text_start_idx:text_end_idx]  # Use open interval [start:end)
                else:
                    # Fallback: use all decoded texts
                    text_tokens = decoded_texts
            else:
                # Fallback: tokenize on the fly if language_info not available
                text_tokens = self.tokenizer.tokenize(text)
                # Clean special markers
                cleaned_tokens = []
                for token in text_tokens:
                    clean_token = token.replace('Ġ', '').replace('▁', '').replace('</w>', '').replace('<s>', '').replace('</s>', '')
                    cleaned_tokens.append(clean_token)
                text_tokens = cleaned_tokens
                
            # Pad or truncate to match attention length
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
            view_key (str): View identifier ('image11', 'image12', 'image21', 'image22')
                           Format: {window}_image{camera_num}
                           where window=1 or 2, camera_num=1 or 2
            
        Returns:
            PIL.Image: Requested image or gray placeholder if not found
        """
        step_str = f"{step:04d}"
        # Parse view_key to extract window and camera numbers
        # view_key format: 'image11' -> window=1, camera=1
        #                'image12' -> window=1, camera=2
        #                'image21' -> window=2, camera=1
        #                'image22' -> window=2, camera=2
        window_num = view_key[5]  # Extract '1' or '2' from position 5
        camera_num = view_key[6]  # Extract '1' or '2' from position 6
        
        # Construct filename: step_{XXXX}_{window}_image{camera}.jpg
        img_filename = f"step_{step_str}_{window_num}_image{camera_num}.jpg"
        img_path = self.raw_image_dir / img_filename
        
        if img_path.exists():
            return Image.open(img_path).convert('RGB')
        else:
            logger.warning(f"Image not found: {img_path}")
            return Image.new('RGB', (256, 256), color='gray')

    # ========== DIT Expert Network Utility Functions ==========
    
    def extract_action_token_attention(self, attn_tensor, layer_idx):
        """
        Extract attention weights for the target action token (index 34)
        
        Args:
            attn_tensor (torch.Tensor): Attention tensor [batch, heads, query_dim, key_dim]
            layer_idx (int): Layer index to determine even/odd layer type
            
        Returns:
            torch.Tensor: Attention weights for target action token [heads, key_dim]
        """
        # Extract attention for target action token (query dimension index 34)
        # Shape: [batch, heads, query_dim, key_dim] -> [heads, key_dim]
        action_attn = attn_tensor[0, :, self.target_action_idx, :]
        return action_attn
    
    def split_even_layer_dimensions(self, action_attn):
        """
        Split even layer cross-attention dimensions (366D)
        
        Args:
            action_attn (torch.Tensor): Action attention weights [key_dim=366]
            
        Returns:
            dict: Dictionary containing split attention components
        """
        # action_attn is already 1D tensor [366], no need to average heads
        action_attn_avg = action_attn if action_attn.dim() == 1 else torch.mean(action_attn, dim=0)
        
        # Split according to predefined segments
        image11_attn = action_attn_avg[Settings.SEGMENTS_INDICES['image11'][0]:Settings.SEGMENTS_INDICES['image11'][1]]
        image12_attn = action_attn_avg[Settings.SEGMENTS_INDICES['image12'][0]:Settings.SEGMENTS_INDICES['image12'][1]]
        image21_attn = action_attn_avg[Settings.SEGMENTS_INDICES['image21'][0]:Settings.SEGMENTS_INDICES['image21'][1]]
        image22_attn = action_attn_avg[Settings.SEGMENTS_INDICES['image22'][0]:Settings.SEGMENTS_INDICES['image22'][1]]
        
        # Text attention: only task-related tokens (about 10-20 tokens)
        # Extracted from settings.py LAN_INPUT_INDICES['text'] = (start_idx, end_idx)
        # These are the first and last indices of matched task description tokens
        text_start_idx, text_end_idx = Settings.SEGMENTS_INDICES['text']
        text_attn = action_attn_avg[text_start_idx:text_end_idx + 1]  # +1 to include end token
        
        return {
            'image11': image11_attn,
            'image12': image12_attn,
            'image21': image21_attn,
            'image22': image22_attn,
            'text': text_attn
        }
    
    def split_odd_layer_dimensions(self, action_attn):
        """
        Split odd layer self-attention dimensions (42D)
        
        Args:
            action_attn (torch.Tensor): Action attention weights [key_dim=42]
            
        Returns:
            dict: Dictionary containing split attention components
        """
        # action_attn is already 1D tensor [42], no need to average heads
        action_attn_avg = action_attn if action_attn.dim() == 1 else torch.mean(action_attn, dim=0)
        
        # Split according to module structure
        state_attn = action_attn_avg[0:2]      # [2]
        future_attn = action_attn_avg[2:34]    # [32]
        action_attn_module = action_attn_avg[34:42]  # [8]
        
        return {
            'state': state_attn,
            'future': future_attn,
            'actions': action_attn_module,
            'full': action_attn_avg
        }
    
    # ========== DIT Even Layer Visualization Functions ==========
    
    def visualize_even_layer_cross_attention(self, step, time_step, layer_idx, action_attn_split):
        """
        Visualize even layer cross-attention (4 images + text)
        
        Args:
            step (int): Step number
            time_step (int): Time step
            layer_idx (int): Layer index
            action_attn_split (dict): Split attention weights for images and text
            
        Returns:
            tuple: (raw_img1, raw_img2, view1_img, view2_img, text_img, extra_imgs)
                   where extra_imgs contains image21 and image22 visualizations
        """
        # Load original images
        original_image11 = self.get_step_images(step, 'image11')
        original_image12 = self.get_step_images(step, 'image12')
        original_image21 = self.get_step_images(step, 'image21')
        original_image22 = self.get_step_images(step, 'image22')
        
        # Generate heatmaps for each image
        image11_overlay = self.overlay_attention_on_image(original_image11, action_attn_split['image11'])
        image12_overlay = self.overlay_attention_on_image(original_image12, action_attn_split['image12'])
        image21_overlay = self.overlay_attention_on_image(original_image21, action_attn_split['image21'])
        image22_overlay = self.overlay_attention_on_image(original_image22, action_attn_split['image22'])
        
        # Resize overlays to match original image size
        image11_overlay = image11_overlay.resize(original_image11.size)
        image12_overlay = image12_overlay.resize(original_image12.size)
        image21_overlay = image21_overlay.resize(original_image21.size)
        image22_overlay = image22_overlay.resize(original_image22.size)
        
        # Render raw images
        raw_img1 = self.heatmap_visualizer.render_image(
            original_image11, f"Step {step} - Image11 (Time: {time_step})"
        )
        raw_img2 = self.heatmap_visualizer.render_image(
            original_image12, f"Step {step} - Image12 (Time: {time_step})"
        )
        
        # Render overlay images
        view1_img = self.heatmap_visualizer.render_image(
            image11_overlay, "Image11 - Attention Heatmap"
        )
        view2_img = self.heatmap_visualizer.render_image(
            image12_overlay, "Image12 - Attention Heatmap"
        )
        
        # Render additional images (image21 and image22)
        raw_img3 = self.heatmap_visualizer.render_image(
            original_image21, f"Step {step} - Image21 (Time: {time_step})"
        )
        raw_img4 = self.heatmap_visualizer.render_image(
            original_image22, f"Step {step} - Image22 (Time: {time_step})"
        )
        view3_img = self.heatmap_visualizer.render_image(
            image21_overlay, "Image21 - Attention Heatmap"
        )
        view4_img = self.heatmap_visualizer.render_image(
            image22_overlay, "Image22 - Attention Heatmap"
        )
        
        # Visualize text attention
        text_img = None
        if step in self.text_token_ids:
            token_ids = self.language_info[step]['text_token_ids'][0]
            text = self.decode_text_tokens(token_ids)
            text_img = self.visualize_text_attention(text, action_attn_split['text'])
        
        # Return all images as a structured tuple
        return (raw_img1, raw_img2, view1_img, view2_img, text_img, 
                [raw_img3, raw_img4, view3_img, view4_img])
    
    # ========== DIT Odd Layer Visualization Functions ==========
    
    def visualize_odd_layer_self_attention(self, step, time_step, layer_idx, action_attn_split):
        """
        Visualize odd layer self-attention (42D internal structure)
        
        Args:
            step (int): Step number
            time_step (int): Time step
            layer_idx (int): Layer index
            action_attn_split (dict): Split attention weights (state, future, actions)
            
        Returns:
            tuple: (module_heatmap_img, grouped_bar_img, mean_bar_img)
        """
        # Normalize attention weights
        full_attn_normalized = self.normalize_attention(action_attn_split['full'])
        
        # Convert to numpy if it's a tensor
        if torch.is_tensor(full_attn_normalized):
            full_attn_numpy = full_attn_normalized.cpu().numpy()
        else:
            full_attn_numpy = full_attn_normalized  # Already numpy array
        
        # Render modular heatmap
        module_heatmap_img = self.module_heatmap_visualizer.render(
            values=full_attn_numpy,
            title=f"Self-Attention Heatmap - Layer {layer_idx} (Step {step}, Time {time_step})",
            cmap=self.current_colormap,
            highlight_target_action=True,
            target_idx=self.target_action_idx
        )
        
        # Render grouped bar chart (total attention)
        grouped_bar_img = self.module_heatmap_visualizer.render_grouped_bar(
            values=full_attn_numpy,
            title=f"Module Attention Distribution - Layer {layer_idx}"
        )
        
        # Render mean bar chart
        mean_bar_img = self.module_heatmap_visualizer.render_mean_bar(
            values=full_attn_numpy,
            title=f"Mean Module Attention - Layer {layer_idx}"
        )
        
        return module_heatmap_img, grouped_bar_img, mean_bar_img

    def update_visualization(self, step, time_step, attention_head, layer_idx, alpha, normalization_method, interpolation_method, colormap):
        """
        Core update function for DIT expert network: Update all visualizations based on interactive parameters
        Automatically switches between even layer (cross-attention) and odd layer (self-attention) visualization schemes
        
        Args:
            step (int): Current step number
            time_step (int): Time step within the sequence
            attention_head (str): Attention head selection
            layer_idx (int): Layer index (even=cross-attention, odd=self-attention)
            alpha (float): Transparency level
            normalization_method (str): Attention normalization method
            interpolation_method (str): Heatmap interpolation method
            colormap (str): Color scheme for visualizations
            
        Returns:
            tuple: Visualization images depending on layer type:
                   - Even layer: (raw_img1, raw_img2, view1_img, view2_img, text_img, raw_img3, raw_img4, view3_img, view4_img)
                   - Odd layer: (module_heatmap, grouped_bar, None, None, None, None, None, None, None)
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
        # Convert layer_idx to string for dictionary lookup
        # The keys in time_data are strings (e.g., "0", "1", etc.)
        layer_idx_str = str(layer_idx)
        
        if layer_idx_str not in time_data:
            logger.warning(f"Layer {layer_idx_str} not found, available: {list(time_data.keys())}")
            # Fallback to first available layer or default to "0"
            layer_idx_str = list(time_data.keys())[0] if time_data else "0"
            logger.warning(f"Using fallback layer: {layer_idx_str}")
        
        attn_tensor = time_data.get(layer_idx_str, None)
        if attn_tensor is None:
            logger.error(f"Attention tensor is None for layer {layer_idx_str}")
            empty_img = Image.new('RGB', (256, 256), color='gray')
            return empty_img, empty_img, empty_img, empty_img, empty_img, empty_img, empty_img
        
        # logger.info(f"Loaded Layer {layer_idx_str}: shape={attn_tensor.shape}")

        # Process attention head
        if attention_head == "Average Pooling Head":
            attn_tensor = self.get_average_attention_head(attn_tensor)
            head_idx = 0
        else:
            head_idx = int(attention_head.replace("Head ", "")) - 1

        # Determine layer type first to know the key dimension
        layer_idx_int = int(layer_idx)
        is_even_layer = (layer_idx_int % 2 == 0)
        
        # Extract single head attention and specific query position
        # Original shape: [batch, heads, query_dim=42, key_dim]
        # For even layer: key_dim = 366 (cross-attention)
        # For odd layer: key_dim = 42 (self-attention)
        single_head_attn = attn_tensor[:, head_idx:head_idx+1, :, :]  # [1, 1, 42, key_dim]
        
        # Extract the specific query token for target action
        # Result shape: [key_dim] - this is what we need for dimension splitting
        action_query_attn = single_head_attn[0, 0, self.target_action_idx, :]  # [key_dim]
        
        # ========== DIT Expert Network Logic ==========
        # action_query_attn already has correct shape [key_dim]
        # Pass directly to dimension splitting functions
        action_attn = action_query_attn
        
        # Determine layer type and visualize accordingly
        layer_idx_int = int(layer_idx)
        
        if layer_idx_int % 2 == 0:
            # ===== Even Layer: Cross-Attention (Vision + Text) =====
            # Split dimensions for 4 images + text
            action_attn_split = self.split_even_layer_dimensions(action_attn)
            
            # Generate cross-attention visualization
            result = self.visualize_even_layer_cross_attention(
                step, time_step, layer_idx_int, action_attn_split
            )
            raw_img1, raw_img2, view1_img, view2_img, text_img, extra_imgs = result
            raw_img3, raw_img4, view3_img, view4_img = extra_imgs
            
            # Return format compatible with UI (12 outputs)
            # For even layer: return 9 images + 3 None placeholders for module heatmaps
            return (raw_img1, raw_img2, view1_img, view2_img, text_img, 
                    raw_img3, raw_img4, view3_img, view4_img,
                    None, None, None)  # module_heatmap_output, grouped_bar_output, mean_bar_output
            
        else:
            # ===== Odd Layer: Self-Attention (Internal Token Relations) =====
            # Split dimensions for state + future + actions
            action_attn_split = self.split_odd_layer_dimensions(action_attn)
            
            # Generate self-attention visualization
            module_heatmap_img, grouped_bar_img, mean_bar_img = \
                self.visualize_odd_layer_self_attention(
                    step, time_step, layer_idx_int, action_attn_split
                )
            
            # Return format compatible with UI (12 outputs)
            # For odd layer: return module heatmap + grouped bar + mean bar + 9 None placeholders
            return (None, None, None, None, None, 
                    None, None, None, None,
                    module_heatmap_img, grouped_bar_img, mean_bar_img)