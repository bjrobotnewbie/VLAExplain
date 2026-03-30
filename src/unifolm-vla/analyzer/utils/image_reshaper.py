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
from core.data_processor import DataProcessor


class ImageReshaper(DataProcessor):
    """Image reshaping processor"""
    
    def __init__(self, patch_size_grid=None):
        super().__init__()
        self.patch_size_grid = patch_size_grid
        self.interpolation_methods = {
            'none': None,
            'nearest': cv2.INTER_NEAREST,
            'linear': cv2.INTER_LINEAR,
            'cubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }

    def process(self, 
                attn, 
                image_size=(224, 224), 
                interpolation_method='cubic'                
                ):
        """
        Reshape patch-level attention to image dimensions 
        (256 patches correspond to a 16x16 grid, upsampled to original image size)
        """
        attn_flat = attn.reshape(-1)
        # Convert to float32 if attn_flat is bfloat16
        attn_flat = attn_flat if attn_flat.dtype != torch.bfloat16 else attn_flat.to(torch.float32)
        # Convert to numpy array
        if torch.is_tensor(attn_flat):
            attn_flat = attn_flat.cpu().numpy()
        # Ensure data type is float32, required by cv2.resize
        attn_flat = attn_flat.astype(np.float32)

        patch_num = int(np.sqrt(len(attn_flat)))
        img_h, img_w = image_size[0], image_size[1]
        
        # Generate heatmap based on interpolation method
        if interpolation_method == 'none':
            attn_image = np.zeros((img_h, img_w), dtype=np.float32)
            for i in range(patch_num):
                for j in range(patch_num):
                    patch_idx = i * patch_num + j
                    if patch_idx >= len(attn_flat):
                        continue
                    x1, y1 = j * self.patch_size_grid, i * self.patch_size_grid
                    x2, y2 = min((j+1) * self.patch_size_grid, img_w), min((i+1) * self.patch_size_grid, img_h)
                    attn_image[y1:y2, x1:x2] = attn_flat[patch_idx]
        else:
            # Use interpolation: first generate small heatmap, then upscale to original image size
            patch_attn_reshaped = attn_flat[:patch_num*patch_num].reshape(patch_num, patch_num)
            attn_image = cv2.resize(
                                    patch_attn_reshaped, 
                                    (img_w, img_h), 
                                    interpolation=self.interpolation_methods[interpolation_method]
                                )
        return attn_image