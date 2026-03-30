# 🧠 VLAExplain — Interpreting Vision-Language-Action (VLA) Models
# Copyright (C) 2026 Yafei Shi
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# For more information, please visit: https://github.com/bjrobotnewbie/VLAExplain

import os
from pathlib import Path
from core.data_processor import LanguageInfoLoader
from core.data_processor import TokenDecoder
from transformers import AutoTokenizer

class Settings:
    """System configuration class"""
    
    # Model path configuration
    TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "")
    
    # Data path configuration
    DATA_MAIN_PATH = Path(os.getenv("VLA_DATA_DIR", "/mnt/data/myapp/code/lerobot_g1/unifolm-vla/analyzer_data"))
    RAW_IMAGE_DIR = DATA_MAIN_PATH / "raw_images"
    LANGUAGE_INFO_PATH = DATA_MAIN_PATH / "language_info/language_info.pkl"
    EXPERT_ATTN_DIR = DATA_MAIN_PATH / "expert_attention"
    LANGUAGE_ATTN_DIR = DATA_MAIN_PATH / "language_attention"
    

    MODULES_IMAGE = ["image11", "image12", "image21", "image22"]
    # MODULES_CROSS = ["text", "state"]
    MODULES_CROSS = ["text"]
    ACTION_NUM = 50
    # System parameters
    PATCH_SIZE_GRID = int(os.getenv("PATCH_SIZE_GRID", "8"))
    IGNORE_IMAGE3 = os.getenv("IGNORE_IMAGE3", "True").lower() == "true"  # Whether to ignore view 3
    LAN_MODEL_LAYER_NUM = int(os.getenv("LAN_MODEL_LAYER_NUM", "28"))
    LAN_NUM_ATTENTION_HEADS = int(os.getenv("LAN_NUM_ATTENTION_HEADS", "28"))
    
    # DIT expert network configuration
    DIT_NUM_ATTENTION_HEADS = int(os.getenv("DIT_NUM_ATTENTION_HEADS", "32"))  # Number of attention heads in DIT model
    
    # UI parameters
    DEFAULT_ALPHA = float(os.getenv("DEFAULT_ALPHA", "0.5"))
    DEFAULT_COLORMAP = os.getenv("DEFAULT_COLORMAP", "jet")
    DEFAULT_NORMALIZATION_METHOD = os.getenv("DEFAULT_NORMALIZATION_METHOD", "log_normalize")
    DEFAULT_INTERPOLATION = os.getenv("DEFAULT_INTERPOLATION", "cubic")
    
    # Web server configuration
    SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT = int(os.getenv("SERVER_PORT", "7862"))

    # Cached initialization results as class attributes
    _lan_input_indices = None
    _last_initialized_step = 0 

    @classmethod
    def initialize_lan_input_indices(cls, step):
        """Initialize data (execute only once per unique step)"""
        if cls._lan_input_indices is not None and cls._last_initialized_step == step:
            return cls._lan_input_indices

        language_info = LanguageInfoLoader().process()
        text_token_ids = language_info[step]['text_token_ids'][0]
        tokenizer = AutoTokenizer.from_pretrained(cls.TOKENIZER_PATH)
        token_decoder = TokenDecoder(tokenizer)
        decoded_texts = token_decoder.process(text_token_ids)

        task_indices = []
        # state_indices = []


        TASK_START_PATTERN = ['task', 'is', '"']
        FORMAT_INSTRUCTION_PATTERN = ['Please', 'predict', 'up', 'to', '', '1', '0', 'key', 'trajectory', 'points']

        for i, text in enumerate(decoded_texts):
            if text == 'The' and decoded_texts[i+1:i+4] == TASK_START_PATTERN:
                task_indices.append(i)
            if decoded_texts[i:i+10] == FORMAT_INSTRUCTION_PATTERN:
                task_indices.append(i)


        cls._lan_input_indices = {
            "text": tuple(task_indices),
            # "state": tuple(state_indices)
        }
        cls._last_initialized_step = step
        return cls._lan_input_indices

    @classmethod
    @property
    def LAN_INPUT_INDICES(cls):
        """Provide read-only access interface"""
        return cls.initialize_lan_input_indices(step=cls._last_initialized_step)
    
    @classmethod
    @property
    def SEGMENTS_INDICES(cls):
        """Get segment indices for images/text/state"""
        return {
            "image11": (15, 79),
            "image12": (81, 145),
            "image21": (147, 211),
            "image22": (213, 277),
            "text": (cls.LAN_INPUT_INDICES['text'][0], cls.LAN_INPUT_INDICES['text'][1]),
        }

    @classmethod
    def validate_paths(cls):
        """Validate whether all paths exist"""
        paths_to_check = [
            cls.TOKENIZER_PATH,
            cls.RAW_IMAGE_DIR,
            cls.EXPERT_ATTN_DIR,
            cls.LANGUAGE_INFO_PATH,
            cls.LANGUAGE_ATTN_DIR
        ]
        
        missing_paths = []
        for path in paths_to_check:
            if not Path(path).exists():
                missing_paths.append(str(path))
        
        if missing_paths:
            raise FileNotFoundError(f"The following paths do not exist: {', '.join(missing_paths)}")
        
        return True