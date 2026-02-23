# 🧠 VLAExplain — Interpreting Vision-Language-Action (VLA) Models
# Copyright (C) 2026 Yafei Shi
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# For more information, please visit: https://github.com/bjrobotnewbie/VLAExplain

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

class HeatmapOverlayVisualizer:
    """
    A generic heatmap overlay visualizer for overlaying attention maps onto original images.
    """

    def __init__(self, patch_size_grid=16):
        """
        Initialize the visualizer.

        Args:
            patch_size_grid (int): The image is divided into a patch_size_grid x patch_size_grid grid.
        """
        self.patch_size_grid = patch_size_grid

    def overlay(self, raw_image, attn_map, alpha=0.5, cmap="jet", has_grid=True):
        """
        Overlay the attention heatmap onto the original image.

        Args:
            raw_image (numpy.ndarray): The original BGR or grayscale image.
            attn_map (numpy.ndarray or torch.Tensor): The attention map, with shape [H, W] or [1, H*W].
            alpha (float): The transparency of the heatmap.
            cmap (str): The name of the Matplotlib colormap.

        Returns:
            numpy.ndarray: The BGR image with the heatmap and grid overlaid.
        """

        # Convert the normalized attention map to a colored heatmap
        colormap = plt.cm.get_cmap(cmap)
        heatmap_colored = colormap(attn_map)
        heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR)

        # Overlay the heatmap onto the original image
        if len(raw_image.shape) == 2:
            raw_image_bgr = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
        else:
            raw_image_bgr = raw_image.copy()

        overlay_image = cv2.addWeighted(raw_image_bgr, alpha, heatmap_colored, 1 - alpha, 0)
        if has_grid:
            # Draw patch grid
            h, w = overlay_image.shape[:2]
            patch_h_step = h // self.patch_size_grid
            patch_w_step = w // self.patch_size_grid

            for i in range(1, self.patch_size_grid):
                # Horizontal lines
                cv2.line(overlay_image, (0, i * patch_h_step), (w, i * patch_h_step), (255, 255, 255), 1)
                # Vertical lines
                cv2.line(overlay_image, (i * patch_w_step, 0), (i * patch_w_step, h), (255, 255, 255), 1)

        return Image.fromarray(overlay_image)

    @staticmethod
    def render_image(image, title):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image)
        ax.set_title(title)
        ax.axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        return img