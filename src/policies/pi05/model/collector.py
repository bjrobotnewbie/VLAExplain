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
from collections import defaultdict
from pathlib import Path
import numpy as np
import pickle
from dataclasses import dataclass
from PIL import Image
from typing import Dict, Optional, Any
import logging
import os
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Configuration class for analysis directories"""
    # main_dir: Path = Path('/home/robot/myapp/llm/code/lerobot/analysis_data')
    
    def __init__(self, main_dir='.'):
        self.main_dir = Path(main_dir)
        self.language_attn_dir = self.main_dir / 'language_attention'
        self.raw_image_dir = self.main_dir / 'raw_images'
        self.key_language_info_dir = self.main_dir / 'language_info'
        self.expert_attn_dir = self.main_dir / 'expert_attention'


class AttentionTracer:
    """Attention data tracer - collects, stores, and manages attention weights"""
    
    def __init__(self):
        """Initialize the attention tracer"""
        self.reset()
        self.is_collect_expert_attn = True
        self.is_collect_language_attn = True
        self.is_collect_raw_images = True
        self.set_main_dir(os.getenv("LEROBOT_DATA_DIR"))

    def set_main_dir(self, main_dir):
        self.config = AnalysisConfig(main_dir)
        self._setup_directories()
    
    def _setup_directories(self):
        """Set up and create all necessary directories"""
        self.main_dir = self.config.main_dir
        self.main_dir.mkdir(exist_ok=True)
        
        self.language_attn_dir = self.config.language_attn_dir
        self.raw_image_dir = self.config.raw_image_dir
        self.key_language_info_dir = self.config.key_language_info_dir
        self.expert_attn_dir = self.config.expert_attn_dir

        for path in [
            self.language_attn_dir,
            self.raw_image_dir,
            self.key_language_info_dir,
            self.expert_attn_dir,
        ]:
            path.mkdir(exist_ok=True)
    def reset(self):
        """Reset the collector"""
        self.attention_data = defaultdict(dict)
        self.current_step = 0
        self.current_time_step = 0
        
        self.layer_outputs: Dict[int, Dict[int, torch.Tensor]] = {}
        self.language_attn: Dict[int, Dict[int, torch.Tensor]] = {}
        self.raw_images: Dict[int, Dict[str, np.ndarray]] = {}
        self.language_info: Dict[int, Dict[int, torch.Tensor]] = {}
        self.expert_attn: Dict[int, Dict[int, Dict[int, torch.Tensor]]] = {}
    
    def update_step(self, step: int):
        """Update current step"""
        self.current_step = step
        # Ensure all dictionaries have corresponding step keys
        for container in [
                            self.layer_outputs, 
                            self.language_attn, 
                            self.language_info, 
                            self.expert_attn
                         ]:
            if self.current_step not in container:
                container[self.current_step] = {}
    
    def update_time_step(self, time_step: int):
        """Update time step"""
        self.current_time_step = time_step
        if self.current_step not in self.expert_attn:
            self.expert_attn[self.current_step] = {}
        if self.current_time_step not in self.expert_attn[self.current_step]:
            self.expert_attn[self.current_step][self.current_time_step] = {}
    
    def update_images(self, direction: str, images: torch.Tensor):
        """Update images"""
        try:
            # Convert tensor to numpy array and save
            img_array = images.detach().cpu().numpy()
            img_array = (img_array.transpose(1, 2, 0) * 255).astype(np.uint8)
            img = Image.fromarray(img_array)
            img.save(self.raw_image_dir / f"step_{self.current_step:04d}_{direction}.jpg")
        except Exception as e:
            logger.error(f"Error saving image for step {self.current_step}, direction {direction}: {e}")
    
    
    def update_language_attention(self, layer_idx: int, attn_weights: torch.Tensor, 
                                 device: torch.device = torch.device('cpu')):
        """
        Update language attention
        
        Args:
            layer_idx: Layer index
            attn_weights: Attention weights
            device: Target device, default is CPU
        """
        # logger.debug(f"Updating language attention - Step: {self.current_step}, Layer: {layer_idx}")
        # logger.debug(f"Attention weights shape: {attn_weights.shape}")
        # logger.debug(f"Attention weights device: {attn_weights.device}")
    

        if self.current_step not in self.language_attn:
            self.language_attn[self.current_step] = {}
        
        # Use directly if already on target device
        if attn_weights.device == device:
            self.language_attn[self.current_step][layer_idx] = attn_weights
            return
        
        # Move to target device
        if hasattr(attn_weights, 'to') and callable(getattr(attn_weights, 'to')):
            moved_weights = attn_weights.to(device).clone()
            self.language_attn[self.current_step][layer_idx] = moved_weights
            
            # Clean GPU memory if moving from GPU to CPU
            if attn_weights.device.type == 'cuda' and device.type == 'cpu':
                del attn_weights
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def update_expert_attention(self, layer_idx: int, attn_weights: torch.Tensor, 
                               device: torch.device = torch.device('cpu')):
        """Update expert attention"""
        # Use directly if already on target device
        if attn_weights.device == device:
            self.expert_attn[self.current_step][self.current_time_step][layer_idx] = attn_weights
            return
        
        # Move to target device
        if hasattr(attn_weights, 'to') and callable(getattr(attn_weights, 'to')):
            moved_weights = attn_weights.to(device).clone()
            self.expert_attn[self.current_step][self.current_time_step][layer_idx] = moved_weights
            
            # Clean GPU memory if moving from GPU to CPU
            if attn_weights.device.type == 'cuda' and device.type == 'cpu':
                del attn_weights
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def update_language_info(self, language_info: Dict[int, torch.Tensor]):
        """Update language information"""
        self.language_info[self.current_step] = language_info
    
    
    def save_expert_attention(self):
        """Save expert attention"""
        if self.current_step not in self.expert_attn or not self.expert_attn[self.current_step]:
            logger.info(f"No expert attention data to save for step {self.current_step}")
            return
                    
        try:
            attn_path = self.expert_attn_dir / f"{self.current_step}_expert_attention.pkl"
            with open(attn_path, 'wb') as f:
                pickle.dump(self.expert_attn, f)
            self.expert_attn = {}
            
        except Exception as e:
            logger.error(f"Error saving expert attention for step {self.current_step}: {e}")

    def save_language_attention(self):
        """Save language attention"""
        if self.current_step not in self.language_attn or not self.language_attn[self.current_step]:
            logger.info(f"No language attention data to save for step {self.current_step}")
            return
                    
        try:
            attn_path = self.language_attn_dir / f"{self.current_step}_language_attention.pkl"
            with open(attn_path, 'wb') as f:
                pickle.dump(self.language_attn, f)
            self.language_attn = {}
            
        except Exception as e:
            logger.error(f"Error saving language attention for step {self.current_step}: {e}")
    
    def save_language_info(self):
        """Save language information"""
        try:
            with open(self.key_language_info_dir / "language_info.pkl", "wb") as f:
                pickle.dump(self.language_info, f)
        except Exception as e:
            logger.error(f"Error saving language info: {e}")
    
    def read_expert_attention(self, attn_path: Optional[Path] = None):
        """Read expert attention from disk"""
        if attn_path is None:
            # If no path specified, try to load all expert attention data
            if not self.expert_attn_dir.exists():
                logger.warning(f"Expert attention directory does not exist: {self.expert_attn_dir}")
                return
            
            for path in self.expert_attn_dir.glob("*.pkl"):
                try:
                    with open(path, 'rb') as f:
                        # Parse step number from filename
                        step_str = path.stem.replace("_expert_attention", "")
                        step = int(step_str)
                        self.expert_attn[step] = pickle.load(f)
                except ValueError:
                    logger.warning(f"Could not parse step number from filename: {path}")
                except Exception as e:
                    logger.error(f"Error reading expert attention from {path}: {e}")
        elif attn_path.exists():
            try:
                with open(attn_path, 'rb') as f:
                    self.expert_attn = pickle.load(f)
            except Exception as e:
                logger.error(f"Error reading expert attention from {attn_path}: {e}")
    
    def read_language_attention(self, attn_path: Optional[Path] = None):
        """Read language attention from disk"""
        if attn_path is None:
            attn_path = self.language_attn_dir / "language_attention.pkl"
        
        if not attn_path.exists():
            logger.warning(f"Language attention file does not exist: {attn_path}")
            return
        
        try:
            with open(attn_path, 'rb') as f:
                self.language_attn = pickle.load(f)
        except Exception as e:
            logger.error(f"Error reading language attention from {attn_path}: {e}")
    
    def read_language_info(self):
        """Read language information from disk"""
        info_path = self.key_language_info_dir / "language_info.pkl"
        if not info_path.exists():
            logger.warning(f"Language info file does not exist: {info_path}")
            return
        
        try:
            with open(info_path, "rb") as f:
                self.language_info = pickle.load(f)
        except Exception as e:
            logger.error(f"Error reading language info from {info_path}: {e}")
    
    def read_raw_images(self):
        """Read images from disk"""
        if not self.raw_image_dir.exists():
            logger.warning(f"Raw image directory does not exist: {self.raw_image_dir}")
            return
        
        for image_path in self.raw_image_dir.glob("*.jpg"):
            try:
                image = np.array(Image.open(image_path))
                # Parse filename to extract direction and step
                filename = image_path.stem
                parts = filename.split("_")
                if len(parts) >= 3:
                    direction = parts[-1]
                    step = int(parts[1])
                    
                    if step not in self.raw_images:
                        self.raw_images[step] = {}
                    self.raw_images[step][direction] = image
            except Exception as e:
                logger.error(f"Error reading raw image from {image_path}: {e}")
    
    def get_raw_images(self) -> Dict[int, Dict[str, np.ndarray]]:
        """Get all raw images"""
        return dict(self.raw_images)
    
    def get_language_attention(self) -> Dict[int, Dict[int, torch.Tensor]]:
        """Get all language attention"""
        return dict(self.language_attn)
    
    def get_attention_data(self) -> Dict[int, Dict[str, Any]]:
        """Get all attention data"""
        return dict(self.attention_data)
    
    def collect_from_model(self, model_attention_data: Dict[int, Dict[str, Any]]):
        """Collect attention data from model"""
        # Merge model attention data by step into analysis collector
        for step, layer_data in model_attention_data.items():
            for layer_name, data in layer_data.items():
                if step not in self.attention_data:
                    self.attention_data[step] = {}
                self.attention_data[step][layer_name] = data
    
    @staticmethod
    def convert_bfloat16(data: Any) -> Any:
        """Convert bfloat16 to float32"""
        if data is not None and isinstance(data, torch.Tensor) and data.dtype == torch.bfloat16:
            return data.float()
        return data


    def get_main_dir(self) -> Path:
        """Get main directory"""
        return self.main_dir


ATTENTION_TRACER = AttentionTracer()
