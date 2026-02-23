# 🧠 VLAExplain — Interpreting Vision-Language-Action (VLA) Models
# Copyright (C) 2026 Yafei Shi
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# For more information, please visit: https://github.com/bjrobotnewbie/VLAExplain

from pathlib import Path
import pickle
from core.action_attn_analyzer import ActionAttnAnalyzer
from config.settings import Settings

def load_attention_data(step, attention_dir):
    """Load attention data for a specified step"""
    attention_file = Path(attention_dir) / f"{step}_expert_attention.pkl"
    if not attention_file.exists():
        return None
        
    with open(attention_file, 'rb') as f:
        attention_dict = pickle.load(f)
    
    return attention_dict.get(step, None)


def get_available_steps():
    """Retrieve a list of available steps"""
    attention_dir = Path(Settings.EXPERT_ATTN_DIR)
    attention_files = [f for f in attention_dir.iterdir() if f.suffix == '.pkl']
    steps = []
    for f in attention_files:
        try:
            step = int(f.stem.split('_')[0])
            steps.append(step)
        except:
            continue
    steps.sort()
    return [step for step in steps if step % 50 == 0 or step == steps[-1]] if steps else []


def get_available_time_steps_for_step(step):
    """Get available time steps for a given step"""
    analyzer = ActionAttnAnalyzer(
        raw_image_dir=Settings.RAW_IMAGE_DIR,
        attention_dir=Settings.EXPERT_ATTN_DIR,
        tokenizer_path=Settings.TOKENIZER_PATH
    )
    attention_data = analyzer.load_attention_data(step)
    available_time_steps = analyzer.get_available_time_steps(attention_data)
    return available_time_steps if available_time_steps else [0]


def get_available_layers(step, time_step):
    """Get available network layers for a specified step and time step"""
    analyzer = ActionAttnAnalyzer(
        raw_image_dir=Settings.RAW_IMAGE_DIR,
        attention_dir=Settings.EXPERT_ATTN_DIR,
        tokenizer_path=Settings.TOKENIZER_PATH
    )
    attention_data = analyzer.load_attention_data(step)
    if attention_data is None:
        return ["0"]
    
    available_time_steps = analyzer.get_available_time_steps(attention_data)
    if time_step not in available_time_steps:
        time_step = available_time_steps[0] if available_time_steps else 0
    
    time_data = attention_data.get(time_step, {})
    if not time_data:
        return ["0"]
    return [int(layer) for layer in time_data.keys()]