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
import cv2
from transformers import AutoTokenizer
from abc import ABC, abstractmethod
# Add project root to Python path
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
src_path = project_root.parent / 'src'
sys.path.insert(0, str(src_path))

from unifolm_vla.model.framework.collector import AnalysisConfig, ATTENTION_TRACER
from core.compute_and_merge_attention import compute_and_merge_attention
from utils.image_reshaper import ImageReshaper
from core.data_processor import TokenDecoder, LanguageInfoLoader
from visualization.heatmap_overlay_visualizer import HeatmapOverlayVisualizer
from visualization.bar_chart_visualizer import BarChartVisualizer
from config.settings import Settings
from core.base_analyzer import BaseAnalyzer

class ImageProcessor:
    """
    Handles image-related operations such as reshaping attention maps to image dimensions,
    processing patch indices from clicks, and drawing selected patches on images.
    """

    def __init__(self, raw_images=None):
        self.image_reshaper = ImageReshaper(Settings.PATCH_SIZE_GRID)
        self.raw_images = raw_images
        self.global_selected_patch_indices = {"image11": [], "image12": [], "image21": [], "image22": []}
        self.interpolation_method = 'cubic'

    def reshape_attention_to_image(self, attn, image_size=(224, 224)):
        """Reshapes patch-level attention into the specified image dimensions."""
        return self.image_reshaper.process(attn, image_size, interpolation_method=self.interpolation_method)

    def get_patch_index_from_click(self, image, click_x, click_y):
        """
        Calculates the corresponding 16x16 patch index (0-255) based on the clicked coordinates.
        Performs boundary checks to ensure valid indices.
        """
        h, w = image.shape[:2]
        patch_w = w // Settings.PATCH_SIZE_GRID
        patch_h = h // Settings.PATCH_SIZE_GRID
        col_idx = click_x // patch_w
        row_idx = click_y // patch_h
        patch_idx = row_idx * Settings.PATCH_SIZE_GRID + col_idx
        patch_idx = max(0, min(patch_idx, Settings.PATCH_SIZE_GRID * Settings.PATCH_SIZE_GRID - 1))
        return int(patch_idx)

    def draw_selected_patches_on_image(self, step, win_num, image_view):
        """Draws selected patches onto the image with grid lines and highlights."""
        if step not in self.raw_images:
            return None
        raw_image = self.raw_images[step][win_num][f"image{image_view}"]
        h, w = raw_image.shape[:2]
        image_with_grid = raw_image.copy()
        if len(image_with_grid.shape) == 2:
            image_with_grid = cv2.cvtColor(image_with_grid, cv2.COLOR_GRAY2BGR)
        patch_h_step = h // Settings.PATCH_SIZE_GRID
        patch_w_step = w // Settings.PATCH_SIZE_GRID
        for i in range(Settings.PATCH_SIZE_GRID):
            cv2.line(image_with_grid, (0, i * patch_h_step), (w, i * patch_h_step), (255, 255, 255), 1)
            cv2.line(image_with_grid, (i * patch_w_step, 0), (i * patch_w_step, h), (255, 255, 255), 1)
        for idx in self.global_selected_patch_indices[f'image{win_num}{image_view}']:
            row_idx = idx // Settings.PATCH_SIZE_GRID
            col_idx = idx % Settings.PATCH_SIZE_GRID
            x1 = int(col_idx * patch_w_step)
            y1 = int(row_idx * patch_h_step)
            x2 = int((col_idx + 1) * patch_w_step)
            y2 = int((row_idx + 1) * patch_h_step)
            cv2.rectangle(image_with_grid, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return image_with_grid

    def image_click_handler(self, step, win_num, image_view, click_event):
        """
        Handles image click events to record selected patch indices.
        Supports multiple input formats for click coordinates.
        """
        if click_event is None or step not in self.raw_images:
            return self.get_image_with_grid(step, int(win_num)), self.global_selected_patch_indices[f'image{win_num}{image_view}']
        raw_image = self.raw_images[step][win_num][f"image{image_view}"]
        if hasattr(click_event, 'data') and isinstance(click_event.data, dict):
            click_x, click_y = click_event.data["x"], click_event.data["y"]
        elif isinstance(click_event, dict):
            click_x, click_y = click_event["x"], click_event["y"]
        elif isinstance(click_event, (list, tuple)) and len(click_event) == 2:
            click_x, click_y = click_event
        else:
            try:
                click_x, click_y = click_event.index[0], click_event.index[1]
            except (TypeError, KeyError, IndexError):
                print(f"Warning: Could not extract coordinates from click_event: {click_event}, type: {type(click_event)}")
                return self.get_image_with_grid(step, int(win_num)), self.global_selected_patch_indices[f'image{win_num}{image_view}']
        patch_idx = self.get_patch_index_from_click(raw_image, click_x, click_y)
        if patch_idx not in self.global_selected_patch_indices[f'image{win_num}{image_view}']:
            self.global_selected_patch_indices[f'image{win_num}{image_view}'].append(patch_idx)
        return self.draw_selected_patches_on_image(step, win_num, image_view), self.global_selected_patch_indices[f'image{win_num}{image_view}']

    def clear_selected_patches(self, step, win_num, image_view):
        """Clears all selected patch indices and redraws the image."""
        self.global_selected_patch_indices[f'image{win_num}{image_view}'] = []
        updated_image = self.draw_selected_patches_on_image(step, win_num, image_view)
        return self.global_selected_patch_indices[f'image{win_num}{image_view}'], updated_image

    def get_image_with_grid(self, step, win_num, image_view):
        """Returns the original image with a 16x16 grid overlay."""
        if step not in self.raw_images:
            return None
        raw_image = self.raw_images[step][win_num][f"image{image_view}"]
        if len(raw_image.shape) == 2:
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
        h, w = raw_image.shape[:2]
        patch_h_step = h // Settings.PATCH_SIZE_GRID
        patch_w_step = w // Settings.PATCH_SIZE_GRID
        for i in range(1, Settings.PATCH_SIZE_GRID):
            cv2.line(raw_image, (0, i * patch_h_step), (w, i * patch_h_step), (255, 255, 255), 1)
            cv2.line(raw_image, (i * patch_w_step, 0), (i * patch_w_step, h), (255, 255, 255), 1)
        return raw_image


class TextProcessor:
    """
    Text Processor
    
    Handles text-related operations:
    - Extract task-related tokens from language_info
    - Provide token index query interface
    - Format token list for UI display
    """

    def __init__(self, decoded_texts_steps):
        """
        Initialize text processor.
        
        Args:
            decoded_texts_steps: Decoded text list for each step
        """
        self.decoded_texts_steps = decoded_texts_steps
        # Get task-related token range from Settings (closed interval)
        self._text_start_index = Settings.LAN_INPUT_INDICES["text"][0]
        self._text_end_index = Settings.LAN_INPUT_INDICES["text"][1]

    def get_token_index_by_text(self, step, target_text):
        """
        Find token indices matching the target text.
        
        Args:
            step: Step index
            target_text: Target text to search
            
        Returns:
            List of matching token indices
        """
        if step not in self.decoded_texts_steps:
            return []
        
        # Extract task-related tokens (closed interval [start:end+1])
        text_tokens = self.decoded_texts_steps[step][
            self._text_start_index:self._text_end_index + 1
        ]
        
        indices = [
            i for i, text in enumerate(text_tokens) 
            if text == target_text
        ]
        
        return indices

    def get_token_list(self, step):
        """
        Get task-related token list for specified step.
        
        Args:
            step: Step index
            
        Returns:
            Decoded text token list (closed interval range)
        """
        if step not in self.decoded_texts_steps:
            return []
        
        # Use closed interval slicing [start:end+1], including end position token
        text_tokens = self.decoded_texts_steps[step][
            self._text_start_index:self._text_end_index + 1
        ]
        
        return text_tokens

    def get_token_list_with_index(self, step):
        """
        Get formatted token list with indices.
        
        Args:
            step: Step index
            
        Returns:
            Formatted string list ["0: token1", "1: token2", ...]
        """
        if step not in self.decoded_texts_steps:
            return []
        
        text_tokens = self.decoded_texts_steps[step][
            self._text_start_index:self._text_end_index + 1
        ]
        
        return [f"{i}: {token}" for i, token in enumerate(text_tokens)]

class AttentionProcessor(BaseAnalyzer):
    """
    Core processor for attention data processing.
    
    Responsible for:
    - Loading and managing attention weights
    - Delegating to specialized processors (ImageProcessor, TextProcessor)
    - Providing unified visualization interface
    """

    def __init__(self, normalization_method="log_normalize"):
        """
        Initialize attention processor.
        
        Args:
            normalization_method: Attention weight normalization method
        """
        super().__init__(normalization_method)
        
        # Initialize utility components
        self.tokenizer = AutoTokenizer.from_pretrained(Settings.TOKENIZER_PATH)
        self.config = AnalysisConfig()
        self.collector = ATTENTION_TRACER
        self.token_decoder = TokenDecoder(self.tokenizer)
        
        # Load data
        self.collector.read_raw_images()
        self.language_info = LanguageInfoLoader().process()
        self._raw_images = self.collector.raw_images
        
        # Initialize data storage
        self.decoded_texts_steps = {}
        self._initialize_data()
        
        # Create specialized processors (only ImageProcessor and TextProcessor)
        self.image_processor = ImageProcessor(self.raw_images)
        self.text_processor = TextProcessor(self.decoded_texts_steps)
        
        # Default interpolation method
        self.interpolation_method = 'cubic'

    def _initialize_data(self):
        """
        Initialize text data.
        
        Extract decoded texts for each step from language_info.
        Note: task_indices calculation is now handled by Settings.LAN_INPUT_INDICES
        """
        for step, info_dict in self.language_info.items():
            if not info_dict:
                continue
                
            # Get text token IDs and decode
            self.text_token_ids = info_dict['text_token_ids'][0]
            decoded_texts = self.token_decoder.process(self.text_token_ids)
            
            # Store decoded texts
            self.decoded_texts_steps[step] = decoded_texts


    def set_interpolation_method(self, method):
        """Sets the interpolation method for image processing."""
        self.image_processor.interpolation_method = method

    def load_attention_weights(self, step, layer_idx):
        """Loads attention weights for the specified step and layer."""
        try:
            self.collector.read_raw_images()
            language_attn_dir = self.collector.language_attn_dir
            attn_weights = None
            for file in language_attn_dir.iterdir():
                if file.suffix == ".pkl":
                    if step == int(file.stem.split('_')[0]):
                        self.collector.read_language_attention(file)
                        temp_attn = self.collector.language_attn
                        temp_attn[step] = {int(k): v for k, v in temp_attn[step].items()}
                        if temp_attn[step] is not None and int(layer_idx) in temp_attn[step]:
                            attn_weights = temp_attn[step][int(layer_idx)]
                            break
            return attn_weights
        except FileNotFoundError as e:
            print(f"Attention weights file not found: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error while loading attention weights: {e}")
            return None

    def get_attention_data(self, step, layer_idx, head_idx, merge_strategy="mean"):
        """
        Retrieves processed attention data including fine-grained and global attention.
        Supports head selection and merging strategies.
        """
        layer_idx_int = int(layer_idx)
        attn_weights = self.load_attention_weights(step, layer_idx_int)
        if attn_weights is None:
            return None, None, None, None
        fine_grained, global_attn, merged_fine, merged_global = compute_and_merge_attention(
            attn_weights, merge_strategy=merge_strategy
        )
        if head_idx == "Average Pooling Head":
            for key in fine_grained:
                fine_grained[key] = torch.mean(fine_grained[key], dim=0, keepdim=True)
            for key in global_attn:
                global_attn[key] = torch.mean(global_attn[key], dim=0, keepdim=True)
        else:
            head_idx_int = int(head_idx)
            for key in fine_grained:
                fine_grained[key] = fine_grained[key][head_idx_int:head_idx_int+1]
            for key in global_attn:
                global_attn[key] = global_attn[key][head_idx_int:head_idx_int+1]
        return fine_grained, global_attn, merged_fine, merged_global

    def select_attention_by_indices(self, selected_indices, fine_grained, attn_key):
        """
        Selects attention for specific indices and computes average pooling.
        """
        attn_data = fine_grained[attn_key]
        attn_data = attn_data.squeeze(0)
        selected_attn = attn_data[selected_indices]
        avg_attn = torch.mean(selected_attn, dim=0, keepdim=True)
        return avg_attn

    def normalize_attention(self, attn_weights):
        """Normalizes attention weights using the configured normalization method."""
        return self.attention_normalizer.process(attn_weights)

    @property
    def raw_images(self):
        return self._raw_images

    @property
    def global_selected_patch_indices(self):
        return self.image_processor.global_selected_patch_indices

    def reshape_attention_to_image(self, attn, image_size=(224, 224)):
        return self.image_processor.reshape_attention_to_image(attn, image_size)

    def get_patch_index_from_click(self, image, click_x, click_y):
        return self.image_processor.get_patch_index_from_click(image, click_x, click_y)

    def draw_selected_patches_on_image(self, step, win_num, image_view):
        return self.image_processor.draw_selected_patches_on_image(step, win_num, image_view)

    def image_click_handler(self, step, win_num, image_view, click_event):
        return self.image_processor.image_click_handler(step, win_num, image_view, click_event)

    def get_image_with_grid(self, step, win_num, image_view):
        return self.image_processor.get_image_with_grid(step, win_num, image_view)

    def clear_selected_patches(self, step, win_num, image_view):
        return self.image_processor.clear_selected_patches(step, win_num, image_view)

    def get_token_list_with_index(self, step):
        return self.text_processor.get_token_list_with_index(step)

    def get_token_index_by_text(self, step, target_text):
        return self.text_processor.get_token_index_by_text(step, target_text)

    def get_token_list(self, step):
        return self.text_processor.get_token_list(step)

class VisualizationStrategy(ABC):
    """
    Abstract base class for visualization strategies.
    Defines a common interface for different types of attention visualizations.
    """

    def __init__(self, processor):
        self.processor = processor
        self.bar_chart_visualizer = BarChartVisualizer(figsize=(12, 8), fontsize=8)
        self.heatmap_visualizer = HeatmapOverlayVisualizer(
            patch_size_grid=Settings.PATCH_SIZE_GRID,
        )

    @abstractmethod
    def visualize(self, step, layer_idx, head_idx, alpha, cmap, **kwargs):
        pass

class TextToVisionVisualizationStrategy(VisualizationStrategy):
    """
    Visualizes attention from text to vision modalities.
    Supports both fine-grained and global attention types.
    """

    def visualize(self, step, layer_idx, head_idx, selected_text_indices, attention_type, alpha, cmap):
        fine_grained, global_attn, merged_fine, merged_global = self.processor.get_attention_data(
            step, layer_idx, head_idx
        )
        if fine_grained is None:
            return None, None, None, None, None
        raw_image11 = self.processor.raw_images[step][1]["image1"]
        raw_image12 = self.processor.raw_images[step][1]["image2"]
        raw_image21 = self.processor.raw_images[step][2]["image1"]
        raw_image22 = self.processor.raw_images[step][2]["image2"]
        image_size11 = (raw_image11.shape[1], raw_image11.shape[0])
        image_size12 = (raw_image12.shape[1], raw_image12.shape[0])
        image_size21 = (raw_image21.shape[1], raw_image21.shape[0])
        image_size22 = (raw_image22.shape[1], raw_image22.shape[0])
        overlay_img11, overlay_img12, overlay_img21, overlay_img22 = None, None, None, None
        if attention_type == "Fine-grained":
            attn_key11 = "text_to_image11"
            text_to_image11_attn = self.processor.select_attention_by_indices(selected_text_indices, fine_grained, attn_key11)
            if text_to_image11_attn is None:
                return None, None, None, None, None, None, None, None
            attn_image11 = self.processor.reshape_attention_to_image(text_to_image11_attn, image_size11)
            attn_image11 = self.processor.normalize_attention(attn_image11)
            overlay_img11 = self.heatmap_visualizer.overlay(raw_image11, attn_image11, alpha, cmap)
            attn_key12 = "text_to_image12"
            text_to_image12_attn = self.processor.select_attention_by_indices(selected_text_indices, fine_grained, attn_key12)
            attn_image12 = self.processor.reshape_attention_to_image(text_to_image12_attn, image_size12)
            attn_image12 = self.processor.normalize_attention(attn_image12)
            overlay_img12 = self.heatmap_visualizer.overlay(raw_image12, attn_image12, alpha, cmap)
            attn_key21 = "text_to_image21"
            text_to_image21_attn = self.processor.select_attention_by_indices(selected_text_indices, fine_grained, attn_key21)
            attn_image21 = self.processor.reshape_attention_to_image(text_to_image21_attn, image_size21)
            attn_image21 = self.processor.normalize_attention(attn_image21)
            overlay_img21 = self.heatmap_visualizer.overlay(raw_image21, attn_image21, alpha, cmap)
            attn_key22 = "text_to_image22"
            text_to_image22_attn = self.processor.select_attention_by_indices(selected_text_indices, fine_grained, attn_key22)
            attn_image22 = self.processor.reshape_attention_to_image(text_to_image22_attn, image_size22)
            attn_image22 = self.processor.normalize_attention(attn_image22)
            overlay_img22 = self.heatmap_visualizer.overlay(raw_image22, attn_image22, alpha, cmap)

        else:
            attn_key11 = "global_text_to_image11"
            global_text_to_image11 = global_attn[attn_key11]
            attn_image11 = self.processor.reshape_attention_to_image(global_text_to_image11, image_size11)
            attn_image11 = self.processor.normalize_attention(attn_image11)
            overlay_img11 = self.heatmap_visualizer.overlay(raw_image11, attn_image11, alpha, cmap)
            attn_key12 = "global_text_to_image12"
            global_text_to_image12 = global_attn[attn_key12]
            attn_image12 = self.processor.reshape_attention_to_image(global_text_to_image12, image_size12)
            attn_image12 = self.processor.normalize_attention(attn_image12)
            overlay_img12 = self.heatmap_visualizer.overlay(raw_image12, attn_image12, alpha, cmap)
            attn_key21 = "global_text_to_image21"
            global_text_to_image21 = global_attn[attn_key21]
            attn_image21 = self.processor.reshape_attention_to_image(global_text_to_image21, image_size21)
            attn_image21 = self.processor.normalize_attention(attn_image21)
            overlay_img21 = self.heatmap_visualizer.overlay(raw_image21, attn_image21, alpha, cmap)
            attn_key22 = "global_text_to_image22"
            global_text_to_image22 = global_attn[attn_key22]
            attn_image22 = self.processor.reshape_attention_to_image(global_text_to_image22, image_size22)
            attn_image22 = self.processor.normalize_attention(attn_image22)
            overlay_img22 = self.heatmap_visualizer.overlay(raw_image22, attn_image22, alpha, cmap)
  
        return raw_image11, overlay_img11, raw_image12, overlay_img12, raw_image21, overlay_img21, raw_image22, overlay_img22

class VisionToTextVisualizationStrategy(VisualizationStrategy):
    """
    Visualizes attention from vision to text modalities.
    Handles single-image and multi-image scenarios.
    """

    def _visualize_single_image(self, step, layer_idx, head_idx, image_key, selected_patches, fine_grained, global_attn, text_tokens, attention_type, alpha, cmap):
        text_attn_img = None
        if attention_type == "Fine-grained":
            attn_key_text = f"{image_key}_to_text"
            attn_data_text = fine_grained[attn_key_text].squeeze(0) 
            valid_patches = [idx for idx in selected_patches if 0 <= idx < attn_data_text.shape[0]]
            if valid_patches:
                selected_attn_text = attn_data_text[valid_patches]
                avg_attn_text = torch.mean(selected_attn_text, dim=0, keepdim=True)
                attn_normalized_text = self.processor.normalize_attention(avg_attn_text).squeeze()
                text_attn_img = self.bar_chart_visualizer.render(
                    attn_normalized_text, 
                    text_tokens,
                    f"{image_key.capitalize()} Patches {valid_patches} → Text Attention (Step {step}, Layer {layer_idx})",
                    cmap=cmap,
                )

        else:
            global_attn_key_text = f"global_{image_key}_to_text"
            global_attn_text = global_attn[global_attn_key_text].squeeze()
            attn_normalized_text = self.processor.normalize_attention(global_attn_text)
            text_attn_img = self.bar_chart_visualizer.render(
                attn_normalized_text, 
                text_tokens,
                f"Global {image_key.capitalize()} → Text Attention (Step {step}, Layer {layer_idx})",
                cmap=cmap,
            )

        return text_attn_img

    def visualize(self, step, layer_idx, head_idx, attention_type, alpha, cmap):
        fine_grained, global_attn, merged_fine, merged_global = self.processor.get_attention_data(
            step, layer_idx, head_idx
        )
        text_tokens = self.processor.decoded_texts_steps[step][Settings.LAN_INPUT_INDICES['text'][0]:Settings.LAN_INPUT_INDICES['text'][1]]
        selected_patches11 = self.processor.global_selected_patch_indices["image11"].copy()
        selected_patches12 = self.processor.global_selected_patch_indices["image12"].copy()
        selected_patches21 = self.processor.global_selected_patch_indices["image21"].copy()
        selected_patches22 = self.processor.global_selected_patch_indices["image22"].copy()
        if not selected_patches11 and attention_type == 'Fine-grained':
            text_attn_img11 = None
        else:
            text_attn_img11 = self._visualize_single_image(
                step, layer_idx, head_idx, "image11", selected_patches11,
                fine_grained, global_attn, text_tokens, attention_type, alpha, cmap
            )
        if not selected_patches12 and attention_type == 'Fine-grained':
            text_attn_img12 = None
        else:
            text_attn_img12 = self._visualize_single_image(
                step, layer_idx, head_idx, "image12", selected_patches12,
                fine_grained, global_attn, text_tokens, attention_type, alpha, cmap
            )
        if not selected_patches21 and attention_type == 'Fine-grained':
            text_attn_img21 = None
        else:
            text_attn_img21 = self._visualize_single_image(
                step, layer_idx, head_idx, "image21", selected_patches21,
                fine_grained, global_attn, text_tokens, attention_type, alpha, cmap
            )
        if not selected_patches22 and attention_type == 'Fine-grained':
            text_attn_img22 = None
        else:
            text_attn_img22 = self._visualize_single_image(
                step, layer_idx, head_idx, "image22", selected_patches22,
                fine_grained, global_attn, text_tokens, attention_type, alpha, cmap
            )
        return text_attn_img11, text_attn_img12, text_attn_img21, text_attn_img22


class LanguageAttentionAnalyzer:
    """
    Main application class for analyzing language attention.
    Integrates all processors and visualization strategies.
    """

    def __init__(self, normalization_method='log_normalize'):
        self.processor = AttentionProcessor(normalization_method)
        self.text_to_vision_visualizer = TextToVisionVisualizationStrategy(self.processor)
        self.vision_to_text_visualizer = VisionToTextVisualizationStrategy(self.processor)

    def update_normalization_method(self, method):
        """Updates the normalization method used by the processor."""
        self.processor.attention_normalizer.method = method
        return f"Normalization method updated to: {method}"

    def update_setting_indices_by_step(self, step):
        """Updates the indices used by the processor based on the given step."""
        Settings.initialize_lan_input_indices(step)
        
    def text_vis_wrapper(self, step, layer, head, dropdown_text, attn_type, alpha, cmap, normalization_method, interpolation_method):
        """Wrapper function for text-to-vision visualization."""
        self.update_setting_indices_by_step(step)
        self.update_normalization_method(normalization_method)
        self.processor.set_interpolation_method(interpolation_method)
        if dropdown_text:
            selected_text_indices = []
            for item in dropdown_text:
                if ": " in item:
                    try:
                        index_part = item.split(": ", 1)[0]
                        idx = int(index_part)
                        selected_text_indices.append(idx)
                    except ValueError:
                        continue
            head_idx = int(head.replace("Head ", "")) - 1 if head != 'Average Pooling Head' else head
            return self.text_to_vision_visualizer.visualize(step, layer, head_idx, selected_text_indices, attn_type, alpha, cmap)
        else:
            return None, None, None, None, None, None, None, None

    def vision_vis_wrapper(self, step, layer_idx, head, attention_type, alpha, cmap, normalization_method, interpolation_method):
        """Wrapper function for vision-to-text visualization."""
        self.update_setting_indices_by_step(step)
        self.update_normalization_method(normalization_method)
        self.processor.set_interpolation_method(interpolation_method)
        head_idx = int(head.replace("Head ", "")) - 1 if head != 'Average Pooling Head' else head
        return self.vision_to_text_visualizer.visualize(step, layer_idx, head_idx, attention_type, alpha, cmap)
