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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image


class EqualHeightBarVisualizer:
    """
    A visualization tool class for generating equal-height bar charts.
    All bar heights are fixed at 1, with color mapping representing weight magnitudes.
    """

    def __init__(self, figsize=(30, 3), fontsize=12, dpi=100):
        """
        Initialize the visualizer parameters.

        Args:
            figsize (tuple): The size of the Matplotlib figure.
            fontsize (int): The font size for X-axis labels.
            dpi (int): The resolution of the output image.
        """
        self.figsize = figsize
        self.fontsize = fontsize
        self.dpi = dpi

    def render(
        self,
        values,
        labels,
        title="Equal Height Bar Chart",
        ylabel="Normalized Attention Weights",
        cmap="jet"
    ):
        """
        Render an equal-height bar chart.

        Args:
            values (array-like): The color weight values for the bars (should be normalized to [0, 1]).
            labels (list of str): The labels corresponding to each bar.
            title (str): The chart title.
            ylabel (str): The Y-axis label.
            cmap (str): The name of the Matplotlib colormap.

        Returns:
            PIL.Image.Image: The generated equal-height bar chart image.
        """
        if len(values) != len(labels):
            raise ValueError("The length of 'values' must match the length of 'labels'.")

        # Create the figure
        fig, ax = plt.subplots(figsize=self.figsize)
        fig.subplots_adjust(bottom=0.3)

        x_pos = np.arange(len(labels))

        # Map colors using colormap
        colormap = plt.cm.get_cmap(cmap)
        colors = colormap(values)

        # Draw equal-height bars (height fixed at 1)
        bars = ax.bar(x_pos, np.ones(len(values)), color=colors, edgecolor='black', linewidth=0.5)

        # Set axes and labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=self.fontsize)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, 1.2)

        # Add color bar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Normalized Attention Weights')

        # Save as image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=self.dpi)
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)

        return img