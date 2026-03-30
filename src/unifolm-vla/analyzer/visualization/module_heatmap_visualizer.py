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
from matplotlib.patches import Rectangle


class ModuleHeatmapVisualizer:
    """
    Modular heatmap visualizer for self-attention visualization.
    Displays attention weights as a segmented bar with distinct color blocks.
    """

    def __init__(self, figsize=(20, 6), dpi=150):
        """
        Initialize the module heatmap visualizer.

        Args:
            figsize (tuple): Figure size (width, height)
            dpi (int): Image resolution
        """
        self.figsize = figsize
        self.dpi = dpi
        
        # Module configuration for 42-dimensional structure
        self.modules = {
            'state': (0, 2, 'State'),           # [0:2] - 2 dimensions
            'future': (2, 34, 'Future Tokens'),  # [2:34] - 32 dimensions
            'action': (34, 42, 'Actions')        # [34:42] - 8 dimensions
        }
        
        # Color schemes for different modules
        self.module_colors = {
            'state': '#FF6B6B',      # Red
            'future': '#4ECDC4',     # Teal
            'action': '#FFE66D'       # Yellow
        }

    def render(self, values, title="Self-Attention Heatmap", cmap="jet", 
               highlight_target_action=True, target_idx=34):
        """
        Render modular heatmap with segmented color blocks.

        Args:
            values (array-like): Attention weights for 42 tokens
            title (str): Chart title
            cmap (str): Colormap for attention intensity
            highlight_target_action (bool): Whether to highlight target action token
            target_idx (int): Index of target action token (default: 34)

        Returns:
            numpy.ndarray: BGR image array
        """
        if len(values) != 42:
            raise ValueError(f"Expected 42 values, got {len(values)}")

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Create heatmap background
        x_pos = np.arange(42)
        values_array = np.array(values).reshape(1, -1)
        
        # Display heatmap
        im = ax.imshow(values_array, aspect='auto', cmap=cmap, origin='lower')
        
        # Set y-axis limits to provide space for labels above and below
        ax.set_ylim(-1.0, 1.0)
        
        # Add module segmentation rectangles
        for module_name, (start, end, label) in self.modules.items():
            rect = Rectangle(
                (start - 0.5, -0.5),
                end - start,
                1,
                fill=False,
                edgecolor=self.module_colors[module_name],
                linewidth=3,
                linestyle='--',
                label=label
            )
            ax.add_patch(rect)
            
            # Add module label at center (positioned below heatmap to avoid overlap)
            center_x = (start + end) / 2 - 0.5
            ax.text(center_x, -0.8, label, ha='center', va='center', 
                   fontsize=9, fontweight='bold',
                   color=self.module_colors[module_name])
        
        # Highlight target action token
        if highlight_target_action and 34 <= target_idx < 42:
            action_idx_in_module = target_idx - 34  # 0-7
            rect_highlight = Rectangle(
                (target_idx - 0.5, -0.5),
                1,
                1,
                fill=True,
                facecolor='red',
                alpha=0.5,
                edgecolor='darkred',
                linewidth=4,
                label=f'Target Action (idx={target_idx})'
            )
            ax.add_patch(rect_highlight)
            
            # Add annotation (positioned to the right and slightly up to avoid overlap)
            ax.annotate(f'Target\nAction\n(idx={target_idx})',
                       xy=(target_idx, 0.5),
                       xytext=(target_idx + 1.8, 0.8),  # Move further right and higher up
                       ha='center',
                       va='center',
                       fontsize=8,
                       fontweight='bold',
                       color='red',
                       arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
        
        # Configure axes
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(i) for i in range(42)], fontsize=6, rotation=45, ha='right')
        ax.set_yticks([])
        ax.set_xlabel('Token Index', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Attention Weight', fontsize=10)
        
        # Add legend (moved to bottom to avoid overlap)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), 
                 fontsize=8, framealpha=0.9, ncol=3)
        
        plt.tight_layout()

        # Convert to OpenCV image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        buf.seek(0)
        img_arr = np.frombuffer(buf.read(), dtype=np.uint8)
        plot_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        plt.close(fig)

        return plot_img

    def render_mean_bar(self, values, title="Mean Attention per Module"):
        """
        Render mean bar chart showing average attention by module.

        Args:
            values (array-like): Attention weights for 42 tokens
            title (str): Chart title

        Returns:
            numpy.ndarray: BGR image array
        """
        if len(values) != 42:
            raise ValueError(f"Expected 42 values, got {len(values)}")

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Calculate module-level statistics
        module_stats = {}
        for module_name, (start, end, label) in self.modules.items():
            module_values = values[start:end]
            module_stats[module_name] = {
                'mean': np.mean(module_values),
                'std': np.std(module_values),
                'min': np.min(module_values),
                'max': np.max(module_values),
                'values': module_values,
                'label': label
            }
        
        # Create mean bar chart
        x_pos = np.arange(len(self.modules))
        width = 0.6
        
        # Plot mean bars
        means = [stats['mean'] for stats in module_stats.values()]
        stds = [stats['std'] for stats in module_stats.values()]
        bars = ax.bar(x_pos, means, width, yerr=stds, capsize=5, 
                     label='Mean Attention ± Std',
                     color=[self.module_colors[m] for m in self.modules.keys()],
                     alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for i, (bar, name) in enumerate(zip(bars, self.modules.keys())):
            height = bar.get_height()
            std_val = stds[i]
            ax.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.002,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Configure axes
        ax.set_xticks(x_pos)
        ax.set_xticklabels([stats['label'] for stats in module_stats.values()], 
                          fontsize=10, rotation=15, ha='right')
        ax.set_ylabel('Mean Attention Weight', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        plt.tight_layout()

        # Convert to OpenCV image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        buf.seek(0)
        img_arr = np.frombuffer(buf.read(), dtype=np.uint8)
        plot_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        plt.close(fig)

        return plot_img

    def render_grouped_bar(self, values, title="Grouped Attention Distribution"):
        """
        Render grouped bar chart showing attention by module.

        Args:
            values (array-like): Attention weights for 42 tokens
            title (str): Chart title

        Returns:
            numpy.ndarray: BGR image array
        """
        if len(values) != 42:
            raise ValueError(f"Expected 42 values, got {len(values)}")

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Calculate module-level statistics
        module_stats = {}
        for module_name, (start, end, label) in self.modules.items():
            module_values = values[start:end]
            module_stats[module_name] = {
                'sum': np.sum(module_values),
                'mean': np.mean(module_values),
                'max': np.max(module_values),
                'values': module_values,
                'label': label
            }
        
        # Create grouped bar chart
        x_pos = np.arange(len(self.modules))
        width = 0.6
        
        # Plot sum bars
        sums = [stats['sum'] for stats in module_stats.values()]
        bars = ax.bar(x_pos, sums, width, label='Total Attention',
                     color=[self.module_colors[m] for m in self.modules.keys()],
                     alpha=0.7)
        
        # Add value labels on bars
        for i, (bar, name) in enumerate(zip(bars, self.modules.keys())):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Configure axes
        ax.set_xticks(x_pos)
        ax.set_xticklabels([stats['label'] for stats in module_stats.values()], 
                          fontsize=10, rotation=15, ha='right')
        ax.set_ylabel('Attention Weight Sum', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()

        # Convert to OpenCV image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        buf.seek(0)
        img_arr = np.frombuffer(buf.read(), dtype=np.uint8)
        plot_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        plt.close(fig)

        return plot_img
