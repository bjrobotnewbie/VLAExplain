# 🧠 VLAExplain — Interpreting Vision-Language-Action (VLA) Models
# Copyright (C) 2026 Yafei Shi
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# For more information, please visit: https://github.com/bjrobotnewbie/VLAExplain

import io
import cv2
import numpy as np
import matplotlib.pyplot as plt


class BarChartVisualizer:
    """
    A generic bar chart visualizer for rendering data such as attention weights into images.
    """

    def __init__(self, figsize=(12, 8), fontsize=8, dpi=150):
        """
        Initialize the visualizer parameters.

        Args:
            figsize (tuple): The size of the Matplotlib figure.
            fontsize (int): The font size for Y-axis labels.
            dpi (int): The resolution of the output image.
        """
        self.figsize = figsize
        self.fontsize = fontsize
        self.dpi = dpi
    
    def render(
        self,
        values,
        labels,
        title="Attention Weights",
        ylabel="Attention Weight",
        cmap="jet"
    ):
        """
        Render the given values and labels into an OpenCV (BGR) formatted image.

        Args:
            values (array-like): The numerical values for the bar chart, e.g., normalized attention weights.
            labels (list of str): The labels corresponding to each bar.
            title (str): The chart title.
            ylabel (str): The Y-axis label.
            cmap (str): The name of the Matplotlib colormap.

        Returns:
            numpy.ndarray: An image array in BGR format, suitable for direct use with OpenCV or Gradio display.
        """        
        if len(values) != len(labels):
            raise ValueError("The length of 'values' must match the length of 'labels'.")

        fig, ax = plt.subplots(figsize=self.figsize)
        x_pos = np.arange(len(labels))  # Changed to x_pos

        # Use colormap to set colors
        colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(labels)))

        ax.bar(x_pos, values, align='center', color=colors)  # Changed to vertical bar chart
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=self.fontsize, rotation=45, ha='right')  # Changed to x-axis labels
        ax.set_ylabel(ylabel)  # Changed to Y-axis label
        ax.set_title(title)
        plt.tight_layout()

        # Convert Matplotlib figure to OpenCV image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        buf.seek(0)
        img_arr = np.frombuffer(buf.read(), dtype=np.uint8)
        plot_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        plt.close(fig)

        return plot_img