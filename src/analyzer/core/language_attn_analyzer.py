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
from lerobot.policies.pi05.collector import AnalysisConfig, ATTENTION_TRACER
from core.compute_and_merge_attention import compute_and_merge_attention
from utils.normalizer import AttentionNormalizer
from utils.image_reshaper import ImageReshaper
from core.data_processor import TokenDecoder, StateParser, LanguageInfoLoader
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
        self.global_selected_patch_indices = {"image1": [], "image2": []}
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

    def draw_selected_patches_on_image(self, step, image_view):
        """Draws selected patches onto the image with grid lines and highlights."""
        if step not in self.raw_images:
            return None
        raw_image = self.raw_images[step][f"image{image_view}"]
        h, w = raw_image.shape[:2]
        image_with_grid = raw_image.copy()
        if len(image_with_grid.shape) == 2:
            image_with_grid = cv2.cvtColor(image_with_grid, cv2.COLOR_GRAY2BGR)
        patch_h_step = h // Settings.PATCH_SIZE_GRID
        patch_w_step = w // Settings.PATCH_SIZE_GRID
        for i in range(Settings.PATCH_SIZE_GRID):
            cv2.line(image_with_grid, (0, i * patch_h_step), (w, i * patch_h_step), (255, 255, 255), 1)
            cv2.line(image_with_grid, (i * patch_w_step, 0), (i * patch_w_step, h), (255, 255, 255), 1)
        for idx in self.global_selected_patch_indices[f'image{image_view}']:
            row_idx = idx // Settings.PATCH_SIZE_GRID
            col_idx = idx % Settings.PATCH_SIZE_GRID
            x1 = int(col_idx * patch_w_step)
            y1 = int(row_idx * patch_h_step)
            x2 = int((col_idx + 1) * patch_w_step)
            y2 = int((row_idx + 1) * patch_h_step)
            cv2.rectangle(image_with_grid, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return image_with_grid

    def image_click_handler(self, step, image_view, click_event):
        """
        Handles image click events to record selected patch indices.
        Supports multiple input formats for click coordinates.
        """
        if click_event is None or step not in self.raw_images:
            return self.get_image_with_grid(step, int(image_view)), self.global_selected_patch_indices[f'image{image_view}']
        raw_image = self.raw_images[step][f"image{image_view}"]
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
                return self.get_image_with_grid(step, int(image_view)), self.global_selected_patch_indices[image_view]
        patch_idx = self.get_patch_index_from_click(raw_image, click_x, click_y)
        if patch_idx not in self.global_selected_patch_indices[f'image{image_view}']:
            self.global_selected_patch_indices[f'image{image_view}'].append(patch_idx)
        return self.draw_selected_patches_on_image(step, image_view), self.global_selected_patch_indices[f'image{image_view}']

    def clear_selected_patches(self, image_view, step):
        """Clears all selected patch indices and redraws the image."""
        self.global_selected_patch_indices[f'image{image_view}'] = []
        updated_image = self.draw_selected_patches_on_image(step, image_view)
        return self.global_selected_patch_indices[f'image{image_view}'], updated_image

    def get_image_with_grid(self, step, image_view):
        """Returns the original image with a 16x16 grid overlay."""
        if step not in self.raw_images:
            return None
        raw_image = self.raw_images[step][f"image{image_view}"]
        if len(raw_image.shape) == 2:
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
        h, w = raw_image.shape[:2]
        patch_h_step = h // Settings.PATCH_SIZE_GRID
        patch_w_step = w // Settings.PATCH_SIZE_GRID
        for i in range(1, Settings.PATCH_SIZE_GRID):
            cv2.line(raw_image, (0, i * patch_h_step), (w, i * patch_h_step), (255, 255, 255), 1)
            cv2.line(raw_image, (i * patch_w_step, 0), (i * patch_w_step, h), (255, 255, 255), 1)
        return raw_image

class StateProcessor:
    """
    Processes state-related data including normalization of attention weights,
    mapping text to state indices, and handling fine-grained vs. merged state attention.
    """

    def __init__(self,
                 language_task_state_indices,
                 decoded_texts_steps,
                 is_state_to_origin,
                 normalization_method
                 ):
        self.language_task_state_indices = language_task_state_indices
        self.decoded_texts_steps = decoded_texts_steps
        self.is_state_to_origin = is_state_to_origin
        self.attention_normalizer = AttentionNormalizer(method=normalization_method)
        self.state_parser = StateParser()
        self._state_start_index = Settings.LAN_INPUT_INDICES["state"][0]
        self._state_end_index = Settings.LAN_INPUT_INDICES["state"][1]

    def normalize_attention(self, attn_weights):
        """Normalizes attention weights using the configured normalization method."""
        return self.attention_normalizer.process(attn_weights)

    def get_state_index_by_text(self, step, target_text):
        """Finds the indices of state tokens matching the given target text."""
        if step not in self.decoded_texts_steps:
            return []
        state_tokens = self.decoded_texts_steps[step][self._state_start_index:self._state_end_index]
        indices = [i for i, text in enumerate(state_tokens) if text == target_text]
        return indices

    def normalize_state_attention(self, step, attn_tensor):
        """
        Normalizes state attention, supporting both original and merged state representations.
        Returns normalized attention values and associated state texts.
        """
        if self.is_state_to_origin:
            state_token_indices = self.language_task_state_indices[step]['state_origin_merged']['state_token_indices']
            attns = []
            for indices in state_token_indices:
                selected = [attn_tensor[0][i] for i in indices if i < attn_tensor.shape[1]]
                if selected:
                    attns.append(torch.mean(torch.stack(selected), dim=0))
            if not attns:
                return torch.zeros(10)
            state_attn_new = torch.stack(attns).unsqueeze(0)
            normalized = self.normalize_attention(state_attn_new).squeeze()[:10]
            state_texts = self.language_task_state_indices[step]['state_origin_merged']['continuous_states'][:10]
        else:
            normalized = self.normalize_attention(attn_tensor).squeeze()
            state_texts = self.decoded_texts_steps[step][self._state_start_index:self._state_end_index]
        return normalized, state_texts

    def select_state_by_content(self, step, selected_state_indices, fine_grained, attn_key):
        """
        Selects attention for specific state content based on provided indices.
        Computes average pooling over selected indices.
        """
        attn_data = fine_grained[attn_key]
        attn_data = attn_data.squeeze(0)
        if self.is_state_to_origin:
            state_token_indices = self.language_task_state_indices[step]['state_origin_merged']['state_token_indices']
            state_selected_indices_list = [state_token_indices[index] for index in selected_state_indices]
            selected_attn = []
            attns = []
            for indices in state_selected_indices_list:
                attns += [attn_data[index] for index in indices]
            selected_attn = torch.mean(torch.stack(attns), dim=0).unsqueeze(0)
        else:
            selected_attn = attn_data[selected_state_indices]
            selected_attn = torch.tensor(selected_attn).unsqueeze(0)
        return selected_attn

    def select_merged_state_attn(self, step, merged_fine, attn_key):
        """
        Selects merged state attention by averaging across relevant token indices.
        """
        attn_data = merged_fine[attn_key]
        attn_data = attn_data.squeeze(0)
        state_token_indices = self.language_task_state_indices[step]['state_origin_merged']['state_token_indices']
        attns = []
        for indices in state_token_indices:
            attns += [attn_data[index] for index in indices]
        return torch.mean(torch.stack(attns), dim=0).unsqueeze(0)

    def get_state_list(self, step):
        """Returns a list of decoded state texts for the given step."""
        if step not in self.decoded_texts_steps:
            return []
        if self.is_state_to_origin:
            continuous_states = self.language_task_state_indices[step]['state_origin_merged']['continuous_states']
        else:
            continuous_states = self.decoded_texts_steps[step][self._state_start_index:self._state_end_index]
        return continuous_states

    def get_state_list_with_index(self, step):
        """Returns a formatted list of state texts with their indices."""
        if step not in self.decoded_texts_steps:
            return []
        if self.is_state_to_origin:
            continuous_states = self.language_task_state_indices[step]['state_origin_merged']['continuous_states']
        else:
            continuous_states = self.decoded_texts_steps[step][self._state_start_index:self._state_end_index]
        return [f"{i}: {token}" for i, token in enumerate(continuous_states)]

class TextProcessor:
    """
    Processes text-related data such as token decoding, index mapping, and text-to-token conversion.
    """

    def __init__(self, decoded_texts_steps):
        self.decoded_texts_steps = decoded_texts_steps
        self._text_start_index = Settings.LAN_INPUT_INDICES["text"][0]
        self._text_end_index = Settings.LAN_INPUT_INDICES["text"][1]

    def get_token_index_by_text(self, step, target_text):
        """Finds the indices of text tokens matching the given target text."""
        if step not in self.decoded_texts_steps:
            return []
        text_tokens = self.decoded_texts_steps[step][self._text_start_index:self._text_end_index]
        indices = [i for i, text in enumerate(text_tokens) if text == target_text]
        return indices

    def get_token_list(self, step):
        """Returns a list of decoded text tokens for the given step."""
        if step not in self.decoded_texts_steps:
            return []
        text_tokens = self.decoded_texts_steps[step][self._text_start_index:self._text_end_index]
        return text_tokens

    def get_token_list_with_index(self, step):
        """Returns a formatted list of text tokens with their indices."""
        if step not in self.decoded_texts_steps:
            return []
        text_tokens = self.decoded_texts_steps[step][self._text_start_index:self._text_end_index]
        return [f"{i}: {token}" for i, token in enumerate(text_tokens)]

class AttentionProcessor(BaseAnalyzer):
    """
    Core class for processing attention data, including loading weights, selecting attention,
    and delegating tasks to specialized processors (ImageProcessor, StateProcessor, TextProcessor).
    """

    def __init__(self, normalization_method="log_normalize"):
        super().__init__(normalization_method)
        self.tokenizer = AutoTokenizer.from_pretrained(Settings.TOKENIZER_PATH)
        self.config = AnalysisConfig()
        self.collector = ATTENTION_TRACER
        self.collector.read_raw_images()
        self.language_info = LanguageInfoLoader().process()
        self._raw_images = self.collector.raw_images
        self.decoded_texts_steps = {}
        self.language_task_state_indices = {}
        self.original_states = {}
        self.token_decoder = TokenDecoder(self.tokenizer)
        self._initialize_data()
        self.image_processor = ImageProcessor(self.raw_images)
        self.state_processor = StateProcessor(
            language_task_state_indices=self.language_task_state_indices,
            decoded_texts_steps=self.decoded_texts_steps,
            is_state_to_origin=self.is_state_to_origin,
            normalization_method=normalization_method
        )
        self.text_processor = TextProcessor(self.decoded_texts_steps)
        self.interpolation_method = 'cubic'

    def _initialize_data(self):
        """Initializes decoded texts and state indices from loaded language information."""
        for step, info_dict in self.language_info.items():
            if not info_dict:
                continue
            self.original_states[step] = info_dict['state'][0].tolist()
            self.text_token_ids = info_dict['text_token_ids'][0]
            decoded_texts = self.token_decoder.process(self.text_token_ids)
            if step not in self.language_task_state_indices:
                self.language_task_state_indices[step] = {}
                self.language_task_state_indices[step]["decoded_texts"] = decoded_texts
                task_indices = []
                state_indices = []
                for i, text in enumerate(decoded_texts):
                    if text == 'Task' and decoded_texts[i+1] == ':':
                        task_indices.append(i)
                    if text == 'State' and decoded_texts[i+1] == ':':
                        state_indices.append(i)
                        task_indices.append(i)
                    if text == 'Action' and decoded_texts[i+1] == ':':
                        state_indices.append(i)
                if len(task_indices) == 2:
                    self.language_task_state_indices[step]["task_indices"] = task_indices
                else:
                    self.language_task_state_indices[step]["task_indices"] = []
                if len(state_indices) == 2:
                    self.language_task_state_indices[step]["state_indices"] = state_indices
                else:
                    self.language_task_state_indices[step]["state_indices"] = []
            self._parse_sate_tokens(step)
            self.decoded_texts_steps[step] = decoded_texts

    def _parse_sate_tokens(self, step):
        """Parses state tokens into original format and merges them."""
        if step in self.language_task_state_indices:
            decoded_texts = self.language_task_state_indices[step]['decoded_texts']
            state_indices = self.language_task_state_indices[step]['state_indices']
            state_texts = decoded_texts[state_indices[0]:state_indices[1]]
            state_origin_merged = self.merge_tokens_to_state(state_texts)
            state_origin_merged['continuous_states'] = ['State', ':']+self.original_states[step] if step in self.original_states else state_origin_merged['continuous_states']
            if 'state_origin_merged' not in self.language_task_state_indices[step]:
                self.language_task_state_indices[step]['state_origin_merged'] = state_origin_merged

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

    def draw_selected_patches_on_image(self, step, image_view):
        return self.image_processor.draw_selected_patches_on_image(step, image_view)

    def image_click_handler(self, step, image_view, click_event):
        return self.image_processor.image_click_handler(step, image_view, click_event)

    def get_image_with_grid(self, step, image_view):
        return self.image_processor.get_image_with_grid(step, image_view)

    def clear_selected_patches(self, image_view, step):
        self.image_processor.clear_selected_patches(image_view, step)

    def get_state_index_by_text(self, step, target_text):
        return self.state_processor.get_state_index_by_text(step, target_text)

    def normalize_state_attention(self, step, attn_tensor):
        return self.state_processor.normalize_state_attention(step, attn_tensor)

    def select_state_by_content(self, step, selected_state_indices, fine_grained, attn_key):
        return self.state_processor.select_state_by_content(step, selected_state_indices, fine_grained, attn_key)

    def select_merged_state_attn(self, step, merged_fine, attn_key):
        return self.state_processor.select_merged_state_attn(step, merged_fine, attn_key)

    def get_state_list(self, step):
        return self.state_processor.get_state_list(step)

    def get_state_list_with_index(self, step):
        return self.state_processor.get_state_list_with_index(step)

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

class TextToVisionStateVisualizationStrategy(VisualizationStrategy):
    """
    Visualizes attention from text to vision/state modalities.
    Supports both fine-grained and global attention types.
    """

    def visualize(self, step, layer_idx, head_idx, selected_text_indices, attention_type, alpha, cmap):
        fine_grained, global_attn, merged_fine, merged_global = self.processor.get_attention_data(
            step, layer_idx, head_idx
        )
        if fine_grained is None:
            return None, None, None, None, None
        raw_image1 = self.processor.raw_images[step]["image1"]
        raw_image2 = self.processor.raw_images[step]["image2"]
        image_size1 = (raw_image1.shape[1], raw_image1.shape[0])
        image_size2 = (raw_image2.shape[1], raw_image2.shape[0])
        overlay_img1, overlay_img2, state_attn_img = None, None, None
        if attention_type == "Fine-grained":
            attn_key1 = "text_to_image1"
            text_to_image1_attn = self.processor.select_attention_by_indices(selected_text_indices, fine_grained, attn_key1)
            if text_to_image1_attn is None:
                return None, None, None, None, None
            attn_image1 = self.processor.reshape_attention_to_image(text_to_image1_attn, image_size1)
            attn_image1 = self.processor.normalize_attention(attn_image1)
            overlay_img1 = self.heatmap_visualizer.overlay(raw_image1, attn_image1, alpha, cmap)
            attn_key2 = "text_to_image2"
            text_to_image2_attn = self.processor.select_attention_by_indices(selected_text_indices, fine_grained, attn_key2)
            attn_image2 = self.processor.reshape_attention_to_image(text_to_image2_attn, image_size2)
            attn_image2 = self.processor.normalize_attention(attn_image2)
            overlay_img2 = self.heatmap_visualizer.overlay(raw_image2, attn_image2, alpha, cmap)
            state_attn_key = "text_to_state"
            text_to_state_attn = self.processor.select_attention_by_indices(selected_text_indices, fine_grained, state_attn_key)
            state_attn_normalized, state_texts = self.processor.normalize_state_attention(
                step,
                text_to_state_attn
            )
            state_attn_img = self.bar_chart_visualizer.render(
                state_attn_normalized, 
                state_texts,
                f"Selected Text → State Attention (Step {step}, Layer {layer_idx})",
                cmap=cmap,
            )
        else:
            attn_key1 = "global_text_to_image1"
            global_text_to_image1 = global_attn[attn_key1]
            attn_image1 = self.processor.reshape_attention_to_image(global_text_to_image1, image_size1)
            attn_image1 = self.processor.normalize_attention(attn_image1)
            overlay_img1 = self.heatmap_visualizer.overlay(raw_image1, attn_image1, alpha, cmap)
            attn_key2 = "global_text_to_image2"
            global_text_to_image2 = global_attn[attn_key2]
            attn_image2 = self.processor.reshape_attention_to_image(global_text_to_image2, image_size2)
            attn_image2 = self.processor.normalize_attention(attn_image2)
            overlay_img2 = self.heatmap_visualizer.overlay(raw_image2, attn_image2, alpha, cmap)
            state_attn_key = "global_text_to_state"
            global_text_to_state = global_attn[state_attn_key].squeeze(0)
            state_attn_normalized, state_texts = self.processor.normalize_state_attention(
                step,
                global_text_to_state
            )
            state_attn_img = self.bar_chart_visualizer.render(
                state_attn_normalized, 
                state_texts,
                f"Global Text → State Attention (Step {step}, Layer {layer_idx})",
                cmap=cmap,
            )
        return raw_image1, overlay_img1, raw_image2, overlay_img2, state_attn_img

class VisionToTextStateVisualizationStrategy(VisualizationStrategy):
    """
    Visualizes attention from vision to text/state modalities.
    Handles single-image and multi-image scenarios.
    """

    def _visualize_single_image(self, step, layer_idx, head_idx, image_key, selected_patches, fine_grained, global_attn, text_tokens, attention_type, alpha, cmap):
        text_attn_img, state_attn_img = None, None
        if attention_type == "Fine-grained":
            attn_key_text = f"{image_key}_to_text"
            attn_key_state = f"{image_key}_to_state"
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
            attn_data_state = fine_grained[attn_key_state].squeeze(0)
            if valid_patches:
                selected_attn_state = attn_data_state[valid_patches]
                avg_attn_state = torch.mean(selected_attn_state, dim=0, keepdim=True)
                state_attn_normalized, state_texts = self.processor.normalize_state_attention(
                    step,
                    avg_attn_state
                )
                state_attn_img = self.bar_chart_visualizer.render(
                    state_attn_normalized, 
                    state_texts,
                    f"{image_key.capitalize()} Patches {valid_patches} → State Attention (Step {step}, Layer {layer_idx})",
                    cmap=cmap,
                )
        else:
            global_attn_key_text = f"global_{image_key}_to_text"
            global_attn_key_state = f"global_{image_key}_to_state"
            global_attn_text = global_attn[global_attn_key_text].squeeze()
            attn_normalized_text = self.processor.normalize_attention(global_attn_text)
            text_attn_img = self.bar_chart_visualizer.render(
                attn_normalized_text, 
                text_tokens,
                f"Global {image_key.capitalize()} → Text Attention (Step {step}, Layer {layer_idx})",
                cmap=cmap,
            )
            global_attn_state = global_attn[global_attn_key_state].squeeze(0)
            state_attn_normalized, state_texts = self.processor.normalize_state_attention(
                step,
                global_attn_state
            )
            state_attn_img = self.bar_chart_visualizer.render(
                state_attn_normalized, 
                state_texts,
                f"Global {image_key.capitalize()} → State Attention (Step {step}, Layer {layer_idx})",
                cmap=cmap,
            )
        return text_attn_img, state_attn_img

    def visualize(self, step, layer_idx, head_idx, attention_type, alpha, cmap):
        fine_grained, global_attn, merged_fine, merged_global = self.processor.get_attention_data(
            step, layer_idx, head_idx
        )
        text_tokens = self.processor.decoded_texts_steps[step][Settings.LAN_INPUT_INDICES['text'][0]:Settings.LAN_INPUT_INDICES['text'][1]]
        selected_patches1 = self.processor.global_selected_patch_indices["image1"].copy()
        selected_patches2 = self.processor.global_selected_patch_indices["image2"].copy()
        if not selected_patches1 and attention_type == 'Fine-grained':
            text_attn_img1, state_attn_img1 = None, None
        else:
            text_attn_img1, state_attn_img1 = self._visualize_single_image(
                step, layer_idx, head_idx, "image1", selected_patches1,
                fine_grained, global_attn, text_tokens, attention_type, alpha, cmap
            )
        if not selected_patches2 and attention_type == 'Fine-grained':
            text_attn_img2, state_attn_img2 = None, None
        else:
            text_attn_img2, state_attn_img2 = self._visualize_single_image(
                step, layer_idx, head_idx, "image2", selected_patches2,
                fine_grained, global_attn, text_tokens, attention_type, alpha, cmap
            )
        return text_attn_img1, text_attn_img2, state_attn_img1, state_attn_img2

class StateToTextVisionVisualizationStrategy(VisualizationStrategy):
    """
    Visualizes attention from state to text/vision modalities.
    Supports both fine-grained and global attention types.
    """

    def visualize(self, step, layer_idx, head_idx, selected_state_indices, attention_type, alpha, cmap):
        fine_grained, global_attn, merged_fine, merged_global = self.processor.get_attention_data(
            step, layer_idx, head_idx
        )
        if fine_grained is None:
            return None, None, None, None, None
        raw_image1 = self.processor.raw_images[step]["image1"]
        raw_image2 = self.processor.raw_images[step]["image2"]
        image_size1 = (raw_image1.shape[1], raw_image1.shape[0])
        image_size2 = (raw_image2.shape[1], raw_image2.shape[0])
        text_tokens = self.processor.decoded_texts_steps[step][Settings.LAN_INPUT_INDICES['text'][0]:Settings.LAN_INPUT_INDICES['text'][1]]
        text_attn_img, overlay_img1, overlay_img2 = None, None, None
        if attention_type == "Fine-grained":
            attn_key = "state_to_text"
            state_to_text_attn = self.processor.select_state_by_content(step, selected_state_indices, fine_grained, attn_key)
            if state_to_text_attn is None:
                return None, None, None, None, None
            attn_normalized = self.processor.normalize_attention(state_to_text_attn).squeeze()
            text_attn_img = self.bar_chart_visualizer.render(
                attn_normalized, 
                text_tokens,
                f"Selected State → Text Attention (Step {step}, Layer {layer_idx})",
                cmap=cmap,
            )
            vision_attn_key1 = "state_to_image1"
            state_to_image1_attn = self.processor.select_state_by_content(step, selected_state_indices, fine_grained, vision_attn_key1)
            attn_image1 = self.processor.reshape_attention_to_image(state_to_image1_attn, image_size1)
            attn_image1 = self.processor.normalize_attention(attn_image1)
            overlay_img1 = self.heatmap_visualizer.overlay(raw_image1, attn_image1, alpha, cmap)
            vision_attn_key2 = "state_to_image2"
            state_to_image2_attn = self.processor.select_state_by_content(step, selected_state_indices, fine_grained, vision_attn_key2)
            attn_image2 = self.processor.reshape_attention_to_image(state_to_image2_attn, image_size2)
            attn_image2 = self.processor.normalize_attention(attn_image2)
            overlay_img2 = self.heatmap_visualizer.overlay(raw_image2, attn_image2, alpha, cmap)
        else:
            if self.processor.is_state_to_origin:
                attn_key = "state_to_text"
                global_state_to_text = self.processor.select_merged_state_attn(step, merged_fine, attn_key).squeeze()
            else:
                attn_key = "global_state_to_text"
                global_state_to_text = global_attn[attn_key].squeeze()
            attn_normalized = self.processor.normalize_attention(global_state_to_text)
            text_attn_img = self.bar_chart_visualizer.render(
                attn_normalized, 
                text_tokens,
                f"Global State → Text Attention (Step {step}, Layer {layer_idx})",
                cmap=cmap,
            )
            if self.processor.is_state_to_origin:
                vision_attn_key1 = "state_to_image1"
                global_state_to_image1 = self.processor.select_merged_state_attn(step, merged_fine, vision_attn_key1)
            else:
                vision_attn_key1 = "global_state_to_image1"
                global_state_to_image1 = global_attn[vision_attn_key1]
            attn_image1 = self.processor.reshape_attention_to_image(global_state_to_image1, image_size1)
            attn_image1 = self.processor.normalize_attention(attn_image1)
            overlay_img1 = self.heatmap_visualizer.overlay(raw_image1, attn_image1, alpha, cmap)
            if self.processor.is_state_to_origin:
                vision_attn_key2 = "state_to_image2"
                global_state_to_image2 = self.processor.select_merged_state_attn(step, merged_fine, vision_attn_key2)
            else:
                vision_attn_key2 = "global_state_to_image2"
                global_state_to_image2 = global_attn[vision_attn_key2]
            attn_image2 = self.processor.reshape_attention_to_image(global_state_to_image2, image_size2)
            attn_image2 = self.processor.normalize_attention(attn_image2)
            overlay_img2 = self.heatmap_visualizer.overlay(raw_image2, attn_image2, alpha, cmap)
        return text_attn_img, raw_image1, overlay_img1, raw_image2, overlay_img2

class LanguageAttentionAnalyzer:
    """
    Main application class for analyzing language attention.
    Integrates all processors and visualization strategies.
    """

    def __init__(self, normalization_method='log_normalize'):
        self.processor = AttentionProcessor(normalization_method)
        self.text_to_vision_state_visualizer = TextToVisionStateVisualizationStrategy(self.processor)
        self.vision_to_text_state_visualizer = VisionToTextStateVisualizationStrategy(self.processor)
        self.state_to_text_vision_visualizer = StateToTextVisionVisualizationStrategy(self.processor)

    def update_normalization_method(self, method):
        """Updates the normalization method used by the processor."""
        self.processor.attention_normalizer.method = method
        return f"Normalization method updated to: {method}"

    def update_setting_indices_by_step(self, step):
        """Updates the indices used by the processor based on the given step."""
        Settings.initialize_lan_input_indices(step)
    def text_vis_wrapper(self, step, layer, head, dropdown_text, attn_type, alpha, cmap, normalization_method, interpolation_method):
        """Wrapper function for text-to-vision/state visualization."""
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
            return self.text_to_vision_state_visualizer.visualize(step, layer, head_idx, selected_text_indices, attn_type, alpha, cmap)
        else:
            return None, None, None, None, None

    def vision_vis_wrapper(self, step, layer_idx, head, attention_type, alpha, cmap, normalization_method, interpolation_method):
        """Wrapper function for vision-to-text/state visualization."""
        self.update_setting_indices_by_step(step)
        self.update_normalization_method(normalization_method)
        self.processor.set_interpolation_method(interpolation_method)
        head_idx = int(head.replace("Head ", "")) - 1 if head != 'Average Pooling Head' else head
        return self.vision_to_text_state_visualizer.visualize(step, layer_idx, head_idx, attention_type, alpha, cmap)

    def state_vis_wrapper(self, step, layer, head, dropdown_state, attn_type, alpha, cmap, normalization_method, interpolation_method):
        """Wrapper function for state-to-text/vision visualization."""
        self.update_setting_indices_by_step(step)
        self.update_normalization_method(normalization_method)
        self.processor.set_interpolation_method(interpolation_method)
        if dropdown_state:
            selected_state_indices = []
            for item in dropdown_state:
                if ": " in item:
                    try:
                        index_part = item.split(": ", 1)[0]
                        idx = int(index_part)
                        selected_state_indices.append(idx)
                    except ValueError:
                        continue
            head_idx = int(head.replace("Head ", "")) - 1 if head != 'Average Pooling Head' else head
            return self.state_to_text_vision_visualizer.visualize(step, layer, head_idx, selected_state_indices, attn_type, alpha, cmap)
        else:
            return None, None, None, None, None