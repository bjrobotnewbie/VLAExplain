# 🧠 VLAExplain — Interpreting Vision-Language-Action (VLA) Models
# Copyright (C) 2026 Yafei Shi
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# For more information, please visit: https://github.com/bjrobotnewbie/VLAExplain

import gradio as gr
from core.action_attn_analyzer import ActionAttnAnalyzer
from utils.file_loader import get_available_steps, get_available_time_steps_for_step, get_available_layers
from config.settings import Settings
from core.language_attn_analyzer import LanguageAttentionAnalyzer


def initialize_action_components(action_analyzer):
    """Initialize components for the action attribution analysis tab."""
    available_steps = get_available_steps()
    default_step = available_steps[0] if available_steps else 0
    available_time_steps = get_available_time_steps_for_step(default_step)
    default_time_step = available_time_steps[0] if available_time_steps else 0
    available_layers = get_available_layers(default_step, default_time_step)

    return {
        "step": default_step,
        "time_step": default_time_step,
        "layers": available_layers,
        "steps": available_steps,
        "time_steps": available_time_steps,
    }


def create_action_tab(vis, action_analyzer, n_action_steps=1):
    """Create the action attribution analysis tab."""
    
    def on_step_change(step):
            Settings.initialize_lan_input_indices(step)
            time_steps = get_available_time_steps_for_step(step)
            layers = get_available_layers(step, time_steps[0] if time_steps else 0)
            return (
                gr.update(choices=time_steps),
                gr.update(choices=layers),
                step
            )

    def on_time_step_change(step, time_step):
        layers = get_available_layers(step, time_step)
        return gr.update(choices=layers)
    
    def on_layer_change(layer_idx):
        """Update visualization titles based on layer type (even/odd)"""
        layer_idx_int = int(layer_idx)
        if layer_idx_int % 2 == 0:
            # Even layer: cross-attention (4 images + text)
            return (
                gr.update(label="Win1 Raw1 Original", visible=True),
                gr.update(label="Win1 Raw2 Original", visible=True),
                gr.update(label="Win1 View1 Attention Heatmap", visible=True),
                gr.update(label="Win1 View2 Attention Heatmap", visible=True),
                gr.update(label="Text Attention Distribution", visible=True),
                gr.update(label="Win2 Raw1 Original", visible=True),
                gr.update(label="Win2 Raw2 Original", visible=True),
                gr.update(label="Win2 View1 Attention Heatmap", visible=True),
                gr.update(label="Win2 View2 Attention Heatmap", visible=True),
                gr.update(visible=False),  # module_heatmap_output
                gr.update(visible=False),  # grouped_bar_output
                gr.update(visible=False)   # mean_bar_output
            )
        else:
            # Odd layer: self-attention (Module Heatmap + Grouped Bar + Mean Bar)
            return (
                gr.update(visible=False),  # raw_img1_output
                gr.update(visible=False),  # raw_img2_output
                gr.update(visible=False),  # view1_attn_output
                gr.update(visible=False),  # view2_attn_output
                gr.update(visible=False),  # text_attn_output
                gr.update(visible=False),  # raw_img3_output
                gr.update(visible=False),  # raw_img4_output
                gr.update(visible=False),  # view3_attn_output
                gr.update(visible=False),  # view4_attn_output
                gr.update(label="Module Heatmap (42D Structure)", visible=True),
                gr.update(label="Module Distribution Bar Chart (Total)", visible=True),
                gr.update(label="Module Distribution Bar Chart (Mean)", visible=True)
            )

    components = initialize_action_components(action_analyzer)

    with gr.TabItem("Action Attribution Analysis"):
        gr.Markdown("## Parameter Selection")

        # State variables
        step_state = gr.State(components["step"])

        # Interactive component row - Two rows for better organization
        with gr.Row():
            # First row: Attention Head, Layer, Time Step, Step Selection
            with gr.Column(scale=1):
                # Generate head choices dynamically based on Settings (use Dropdown for better UX with many heads)
                head_choices = [f"Head {i+1}" for i in range(Settings.DIT_NUM_ATTENTION_HEADS)] + ["Average Pooling Head"]
                head_dropdown_action = gr.Dropdown(
                    choices=head_choices,
                    value="Average Pooling Head",
                    label="Attention Head Selection",
                    info=f"Select from {Settings.DIT_NUM_ATTENTION_HEADS} attention heads"
                )
            with gr.Column(scale=1):
                layer_selector = gr.Dropdown(
                    choices=components["layers"],
                    value=components["layers"][0] if components["layers"] else 0,
                    label="Layer Selection"
                )
            with gr.Column(scale=1):
                time_step_selector = gr.Dropdown(
                    choices=components["time_steps"],
                    value=components["time_step"],
                    label="Time Step Selection"
                )
            with gr.Column(scale=2):
                step_slider = gr.Slider(
                    minimum=min(components["steps"]) if components["steps"] else 0,
                    maximum=max(components["steps"]) if components["steps"] else 100,
                    step=n_action_steps if n_action_steps else 1,
                    value=components["step"],
                    label="Step Selection",
                    interactive=True
                )
        
        # Second row: Normalization, Interpolation, Color Map, Heatmap Transparency
        with gr.Row():
            normalization_method_dropdown = gr.Dropdown(
                choices=[
                    "log_normalize", "min_max", "softmax", "z_score",
                    "robust", "power", "sigmoid", "unit_vector"
                ],
                value=Settings.DEFAULT_NORMALIZATION_METHOD,
                label="Normalization Method",
                info="Select attention weight normalization method to highlight feature differences"
            )
            interpolation_method_dropdown = gr.Dropdown(
                choices=["none", "nearest", "linear", "cubic", "lanczos"],
                value=Settings.DEFAULT_INTERPOLATION,
                label="Interpolation Method",
                info="Select image upscaling interpolation algorithm to affect visual effect"
            )
            colormap_dropdown = gr.Dropdown(
                choices=[
                    "jet", "viridis", "plasma", "inferno", "magma",
                    "cividis", "rainbow", "coolwarm", "seismic", "hot", "cool"
                ],
                value=Settings.DEFAULT_COLORMAP,
                label="Color Map",
                info="Select heatmap color scheme to highlight subtle attention differences"
            )
            alpha_slider = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                step=0.1,
                value=Settings.DEFAULT_ALPHA,
                label="Heatmap Transparency",
                info="Adjust transparency of attention heatmap over raw image"
            )

        # Visualization layout - Dynamic display based on layer type
        with gr.Row():
            gr.Markdown("## DIT Expert Network Attention Visualization")
        
        with gr.Row():
            raw_img1_output = gr.Image(type="pil", 
                                       label="Win1 Raw1 Original", 
                                       height=300,
                                       visible=True,
                                       interactive=False
                                       )
            view1_attn_output = gr.Image(type="pil", 
                                         label="Win1 View1 Attention Heatmap", 
                                         height=300,
                                         visible=True,
                                         interactive=False
                                         )
            raw_img2_output = gr.Image(type="pil", 
                                       label="Win1 Raw2 Original", 
                                       height=300,
                                       visible=True,
                                       interactive=False
                                       )
            view2_attn_output = gr.Image(type="pil", 
                                         label="Win1 View2 Attention Heatmap", 
                                         height=300,
                                         visible=True,
                                         interactive=False
                                         )
        
        with gr.Row():
            raw_img3_output = gr.Image(type="pil", 
                                       label="Win2 Raw1 Original", 
                                       height=300,
                                       visible=True,
                                       interactive=False
                                       )
            view3_attn_output = gr.Image(type="pil", 
                                         label="Win2 View1 Attention Heatmap", 
                                         height=300,
                                         visible=True,
                                         interactive=False
                                         )
            raw_img4_output = gr.Image(type="pil", 
                                       label="Win2 Raw2 Original", 
                                       height=300,
                                       visible=True,
                                       interactive=False
                                       )
            view4_attn_output = gr.Image(type="pil", 
                                         label="Win2 View2 Attention Heatmap", 
                                         height=300,
                                         visible=True,
                                         interactive=False
                                         )
        
        with gr.Row():
            text_attn_output = gr.Image(type="pil", label="Text Attention Distribution", height=300, visible=True, interactive=False)

        with gr.Row():
            module_heatmap_output = gr.Image(type="pil", 
                                             label="Module Heatmap (42D Structure)", 
                                             height=400,
                                             visible=False,
                                             interactive=False
                                             )
        
        with gr.Row():
            grouped_bar_output = gr.Image(type="pil", 
                                          label="Module Distribution Bar Chart (Total)", 
                                          height=400,
                                          visible=False,
                                          interactive=False
                                          )
        
            mean_bar_output = gr.Image(type="pil", 
                                       label="Module Distribution Bar Chart (Mean)", 
                                       height=400,
                                       visible=False,
                                       interactive=False
                                       )
        
        

        # Event bindings
        step_slider.change(
            fn=on_step_change,
            inputs=[step_slider],
            outputs=[time_step_selector, layer_selector, step_state]
        )
        time_step_selector.change(
            fn=on_time_step_change,
            inputs=[step_state, time_step_selector],
            outputs=[layer_selector]
        )
        # Add layer change event to update UI labels and visibility
        layer_selector.change(
            fn=on_layer_change,
            inputs=[layer_selector],
            outputs=[
                raw_img1_output, raw_img2_output, 
                view1_attn_output, view2_attn_output,
                text_attn_output, 
                raw_img3_output, raw_img4_output,
                view3_attn_output, view4_attn_output,
                module_heatmap_output, grouped_bar_output, mean_bar_output
            ]
        )
        
        all_inputs = [
            step_slider, time_step_selector, head_dropdown_action, layer_selector,
            alpha_slider, normalization_method_dropdown, interpolation_method_dropdown, colormap_dropdown
        ]

        for component in [step_slider, time_step_selector, head_dropdown_action, layer_selector,
                          alpha_slider, normalization_method_dropdown, interpolation_method_dropdown, colormap_dropdown]:
            component.change(
                fn=action_analyzer.update_visualization,
                inputs=all_inputs,
                outputs=[
                            raw_img1_output,
                            raw_img2_output, 
                            view1_attn_output, 
                            view2_attn_output,
                            text_attn_output, 
                            raw_img3_output,
                            raw_img4_output,
                            view3_attn_output, 
                            view4_attn_output,
                            module_heatmap_output,
                            grouped_bar_output,
                            mean_bar_output
                         ]
            )

        vis.load(
            fn=action_analyzer.update_visualization,
            inputs=all_inputs,
            outputs=[
                        raw_img1_output,
                        raw_img2_output, 
                        view1_attn_output, 
                        view2_attn_output,
                        text_attn_output, 
                        raw_img3_output,
                        raw_img4_output,
                        view3_attn_output, 
                        view4_attn_output,
                        module_heatmap_output,
                        grouped_bar_output,
                        mean_bar_output
                    ]
        )


def create_language_tab(language_analyzer, n_action_steps=1):
    """Create the language model analysis tab."""
    # === Text → Vision Functions ===
    def update_text_dropdown_lang(step):
        choices = language_analyzer.processor.get_token_list_with_index(step)
        return gr.update(choices=choices, value=[choices[0]])
    # === Vision → Text Functions ===
    def init_image_lang(step, win_num, image_view):
        return language_analyzer.processor.get_image_with_grid(step, win_num, image_view)
    def update_image_lang(step, win_num, image_view):
        updated_image = language_analyzer.processor.draw_selected_patches_on_image(step, win_num, image_view)
        return updated_image
    def handle_image11_click_lang(img, step, event: gr.SelectData):
        updated_image, selected_patches = language_analyzer.processor.image_click_handler(step, 1, "1", event)
        return updated_image, str(selected_patches)
    def clear_patches11_lang(image_view, win_num, step):
        _, image = language_analyzer.processor.clear_selected_patches(step, win_num, image_view)
        return str(language_analyzer.processor.global_selected_patch_indices["image11"]), image
    def handle_image12_click_lang(img, step, event: gr.SelectData):
        updated_image, selected_patches = language_analyzer.processor.image_click_handler(step, 1, "2", event)
        return updated_image, str(selected_patches)
    def clear_patches12_lang(image_view, win_num, step):
        _, image = language_analyzer.processor.clear_selected_patches(step, win_num, image_view)
        return str(language_analyzer.processor.global_selected_patch_indices["image12"]), image
    def handle_image21_click_lang(img, step, event: gr.SelectData):
        updated_image, selected_patches = language_analyzer.processor.image_click_handler(step, 2, "1", event)
        return updated_image, str(selected_patches)
    def clear_patches21_lang(image_view, win_num, step):
        _, image = language_analyzer.processor.clear_selected_patches(step, win_num, image_view)
        return str(language_analyzer.processor.global_selected_patch_indices["image21"]), image
    def handle_image22_click_lang(img, step, event: gr.SelectData):
        updated_image, selected_patches = language_analyzer.processor.image_click_handler(step, 2, "2", event)
        return updated_image, str(selected_patches)
    def clear_patches22_lang(image_view, win_num, step):
        _, image = language_analyzer.processor.clear_selected_patches(step, win_num, image_view)
        return str(language_analyzer.processor.global_selected_patch_indices["image22"]), image


    with gr.TabItem("Language Model Analysis"):
        gr.Markdown("## Parameter Selection")

        available_steps_lang = list(language_analyzer.processor.decoded_texts_steps.keys())

        # Global parameter settings
        with gr.Row():
            with gr.Column(scale=1):
                head_dropdown_lang = gr.Dropdown(
                    choices=[f"Head {i+1}" for i in range(Settings.LAN_NUM_ATTENTION_HEADS)] + ["Average Pooling Head"],
                    value="Average Pooling Head",
                    label="Attention Head Selection",
                    info=f"Select from {Settings.LAN_NUM_ATTENTION_HEADS} attention heads"
                )

            with gr.Column(scale=4):
                with gr.Row():
                    with gr.Column(scale=1):
                        layer_dropdown_lang = gr.Dropdown(
                            choices=[str(i) for i in range(Settings.LAN_MODEL_LAYER_NUM)],
                            value="0",
                            label="Layer Selection",
                            info="Select hidden layer of the language model"
                        )
                    with gr.Column(scale=3):
                        step_slider_lang = gr.Slider(
                            minimum=0,
                            maximum=(
                                max(language_analyzer.processor.decoded_texts_steps.keys()) 
                                if language_analyzer.processor.decoded_texts_steps 
                                else 10
                            ) // n_action_steps * n_action_steps,
                            value=0,
                            step=n_action_steps if n_action_steps else 1,
                            label="Step Selection",
                            info="Slide to select action step"
                        )

        with gr.Row():
            normalization_method_dropdown_lang = gr.Dropdown(
                choices=[
                    "log_normalize", "min_max", "softmax", "z_score",
                    "robust", "power", "sigmoid", "unit_vector"
                ],
                value="log_normalize",
                label="Normalization Method",
                info="Select attention weight normalization method to highlight feature differences"
            )
            interpolation_method_dropdown_lang = gr.Dropdown(
                choices=["none", "nearest", "linear", "cubic", "lanczos"],
                value=Settings.DEFAULT_INTERPOLATION,
                label="Interpolation Method",
                info="Select image upscaling interpolation algorithm to affect visual effect"
            )
            cmap_dropdown_lang = gr.Dropdown(
                choices=[
                    "jet", "viridis", "plasma", "inferno", "magma",
                    "cividis", "rainbow", "coolwarm", "seismic", "hot", "cool"
                ],
                value="jet",
                label="Color Map",
                info="Select heatmap color scheme to highlight subtle attention differences"
            )
            alpha_slider_lang = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                step=0.1,
                value=Settings.DEFAULT_ALPHA,
                label="Heatmap Transparency",
                info="Adjust transparency of attention heatmap over raw image"
            )

        # Text preview
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Raw Text Preview at Current Step")
                token_list_box_lang = gr.Textbox(
                    label="Text",
                    value=" ".join(language_analyzer.processor.get_token_list(0)),
                    interactive=False,
                    lines=2,
                    max_lines=10
                )


        # Multimodal cross-attention visualization tabs
        with gr.Tabs():
            with gr.TabItem("Text → Vision"):
                gr.Markdown("### Text Attention to Win1/2 View1/View2")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        text_dropdown_lang = gr.Dropdown(
                            choices=language_analyzer.processor.get_token_list_with_index(available_steps_lang[0]) if available_steps_lang else [],
                            value=[language_analyzer.processor.get_token_list_with_index(available_steps_lang[0])[0]] if available_steps_lang and language_analyzer.processor.get_token_list_with_index(available_steps_lang[0]) else [],
                            label="Select Text (Supports Multi-select)",
                            multiselect=True,
                            info="Choose from dropdown or input manually"
                        )
                        attention_type_1_lang = gr.Radio(
                            choices=["Fine-grained", "Global"],
                            value="Fine-grained",
                            label="Attention Type",
                            info="Fine-grained: Selected text; Global: Average of all text"
                        )
                        submit_1_lang = gr.Button("Generate Visualization", variant="primary")

                    with gr.Column(scale=4):
                        with gr.Row():
                            image_overlay_11_o_lang = gr.Image(label="Win1 View1 Original", type="numpy")
                            image_overlay_11_lang = gr.Image(label="Text→Win1 View1 Attention Heatmap", type="numpy")
                            image_overlay_12_o_lang = gr.Image(label="Win1 View2 Original", type="numpy")
                            image_overlay_12_lang = gr.Image(label="Text→Win1 View2 Attention Heatmap", type="numpy")
                        with gr.Row():
                            image_overlay_21_o_lang = gr.Image(label="Win2 View1 Original", type="numpy")
                            image_overlay_21_lang = gr.Image(label="Text→Win2 View1 Attention Heatmap", type="numpy")
                            image_overlay_22_o_lang = gr.Image(label="Win2 View2 Original", type="numpy")
                            image_overlay_22_lang = gr.Image(label="Text→Win2 View2 Attention Heatmap", type="numpy")

            with gr.TabItem("Vision → Text"):
                gr.Markdown("### Click Win1/2 View1/View2 Patches to Visualize Attention to Text")    
                with gr.Row():            
                    with gr.Column(scale=1):
                        with gr.Row():
                            with gr.Column(scale=1):
                                image11_display_lang = gr.Image(
                                    label="Win1 View1 (with Patch Grid)",
                                    type="numpy",
                                    interactive=True,
                                    height=300,
                                    value=init_image_lang(0, 1, "1")
                                )
                                selected_patches_text11_lang = gr.Textbox(
                                    label="Selected Patches in Win1 View1",
                                    interactive=False,
                                    value=str(language_analyzer.processor.global_selected_patch_indices["image11"])
                                )
                                clear_patch11_btn_lang = gr.Button("Clear Selected Patches in Win1 View1")

                    with gr.Column(scale=4):
                        with gr.Row():
                            text_attn_1_1_lang = gr.Image(label="Win1 View1 → Text Attention Distribution", type="numpy", interactive=False)

                    with gr.Column(scale=1):
                        with gr.Row():
                            with gr.Column(scale=1):
                                image12_display_lang = gr.Image(
                                    label="Win1 View2 (with Patch Grid)",
                                    type="numpy",
                                    interactive=True,
                                    height=300,
                                    value=init_image_lang(0, 1, "2")
                                )
                                selected_patches_text12_lang = gr.Textbox(
                                    label="Selected Patches in Win1 View2",
                                    interactive=False,
                                    value=str(language_analyzer.processor.global_selected_patch_indices["image12"])
                                )
                                clear_patch12_btn_lang = gr.Button("Clear Selected Patches in Win1 View2")
                            
                    with gr.Column(scale=4):
                        with gr.Row():
                            text_attn_1_2_lang = gr.Image(label="Win1 View2 → Text Attention Distribution", type="numpy", interactive=False)
                            
                with gr.Row():            
                    with gr.Column(scale=1):
                        with gr.Row():
                            with gr.Column(scale=1):
                                image21_display_lang = gr.Image(
                                    label="Win2 View1 (with Patch Grid)",
                                    type="numpy",
                                    interactive=True,
                                    height=300,
                                    value=init_image_lang(0, 2, "1")
                                )
                                selected_patches_text21_lang = gr.Textbox(
                                    label="Selected Patches in Win1 View1",
                                    interactive=False,
                                    value=str(language_analyzer.processor.global_selected_patch_indices["image21"])
                                )
                                clear_patch21_btn_lang = gr.Button("Clear Selected Patches in Win2 View1")

                    with gr.Column(scale=4):
                        with gr.Row():
                            text_attn_2_1_lang = gr.Image(label="Win2 View1 → Text Attention Distribution", type="numpy", interactive=False)

                    with gr.Column(scale=1):
                        with gr.Row():
                            with gr.Column(scale=1):
                                image22_display_lang = gr.Image(
                                    label="Win2 View2 (with Patch Grid)",
                                    type="numpy",
                                    interactive=True,
                                    height=300,
                                    value=init_image_lang(0, 2, "2")
                                )
                                selected_patches_text22_lang = gr.Textbox(
                                    label="Selected Patches in Win2 View2",
                                    interactive=False,
                                    value=str(language_analyzer.processor.global_selected_patch_indices["image22"])
                                )
                                clear_patch22_btn_lang = gr.Button("Clear Selected Patches in Win2 View2")
                            
                    with gr.Column(scale=4):
                        with gr.Row():
                            text_attn_2_2_lang = gr.Image(label="Win2 View2 → Text Attention Distribution", type="numpy", interactive=False)

                with gr.Row():
                    with gr.Column(scale=1):
                        attention_type_2_lang = gr.Radio(
                            choices=["Fine-grained", "Global"],
                            value="Fine-grained",
                            label="Attention Type",
                            info="Fine-grained: Selected patches; Global: Average of all patches"
                        )
                        
                        submit_2_lang = gr.Button("Generate Visualization", variant="primary")
                    with gr.Column(scale=9):
                        pass

            
                            
            # === Text Preview Event Binding ===
            step_slider_lang.change(
                    fn=lambda step: " ".join(language_analyzer.processor.get_token_list(step)),
                    inputs=step_slider_lang,
                    outputs=token_list_box_lang
                )

            # === Text → Vision Event Binding ===
            step_slider_lang.change(
                            fn=update_text_dropdown_lang,
                            inputs=step_slider_lang,
                            outputs=text_dropdown_lang
                        )
            submit_1_lang.click(
                fn=language_analyzer.text_vis_wrapper,
                inputs=[
                    step_slider_lang, 
                    layer_dropdown_lang, 
                    head_dropdown_lang,
                    text_dropdown_lang, 
                    attention_type_1_lang,
                    alpha_slider_lang, 
                    cmap_dropdown_lang, 
                    normalization_method_dropdown_lang,
                    interpolation_method_dropdown_lang
                ],
                outputs=[image_overlay_11_o_lang, image_overlay_11_lang, image_overlay_12_o_lang, image_overlay_12_lang,
                    image_overlay_21_o_lang, image_overlay_21_lang, image_overlay_22_o_lang, image_overlay_22_lang
                ]
            )

            # === Vision → Text Event Binding ===
            step_slider_lang.change(
                fn=update_image_lang,
                inputs=[step_slider_lang, gr.State(value=1), gr.State(value="1")],
                outputs=image11_display_lang
            )
            
            image11_display_lang.select(
                fn=handle_image11_click_lang,
                inputs=[image11_display_lang, step_slider_lang],
                outputs=[image11_display_lang, selected_patches_text11_lang]
            )

            clear_patch11_btn_lang.click(
                fn=clear_patches11_lang,
                inputs=[gr.State("1"), gr.State(1),step_slider_lang],
                outputs=[selected_patches_text11_lang, image11_display_lang]
            )

            step_slider_lang.change(
                fn=update_image_lang,
                inputs=[step_slider_lang, gr.State(value=1), gr.State(value="2")],
                outputs=image12_display_lang
            )

            image12_display_lang.select(
                fn=handle_image12_click_lang,
                inputs=[image12_display_lang, step_slider_lang],
                outputs=[image12_display_lang, selected_patches_text12_lang]
            )

            clear_patch12_btn_lang.click(
                fn=clear_patches12_lang,
                inputs=[gr.State("2"), gr.State(1), step_slider_lang],
                outputs=[selected_patches_text12_lang, image12_display_lang]
            )
            # win2 view1/view2
            step_slider_lang.change(
                fn=update_image_lang,
                inputs=[step_slider_lang, gr.State(value=2), gr.State(value="1")],
                outputs=image11_display_lang
            )
            
            image21_display_lang.select(
                fn=handle_image21_click_lang,
                inputs=[image21_display_lang, step_slider_lang],
                outputs=[image21_display_lang, selected_patches_text21_lang]
            )

            clear_patch21_btn_lang.click(
                fn=clear_patches21_lang,
                inputs=[gr.State("1"), gr.State(2),step_slider_lang],
                outputs=[selected_patches_text21_lang, image21_display_lang]
            )

            step_slider_lang.change(
                fn=update_image_lang,
                inputs=[step_slider_lang, gr.State(value=2), gr.State(value="2")],
                outputs=image22_display_lang
            )

            image22_display_lang.select(
                fn=handle_image22_click_lang,
                inputs=[image22_display_lang, step_slider_lang],
                outputs=[image22_display_lang, selected_patches_text22_lang]
            )

            clear_patch22_btn_lang.click(
                fn=clear_patches22_lang,
                inputs=[gr.State("2"), gr.State(2), step_slider_lang],
                outputs=[selected_patches_text22_lang, image22_display_lang]
            )

            submit_2_lang.click(
                fn=language_analyzer.vision_vis_wrapper,
                inputs=[
                    step_slider_lang, 
                    layer_dropdown_lang, 
                    head_dropdown_lang,
                    attention_type_2_lang, 
                    alpha_slider_lang, 
                    cmap_dropdown_lang, 
                    normalization_method_dropdown_lang,
                    interpolation_method_dropdown_lang
                ],
                outputs=[text_attn_1_1_lang, text_attn_1_2_lang,
                    text_attn_2_1_lang, text_attn_2_2_lang,
                ]
            )

def create_unified_interface(n_action_steps=1):
    """Create the unified attention analysis interface."""
    action_analyzer = ActionAttnAnalyzer(
        raw_image_dir=Settings.RAW_IMAGE_DIR,
        attention_dir=Settings.EXPERT_ATTN_DIR,
        tokenizer_path=Settings.TOKENIZER_PATH,
        normalization_method=Settings.DEFAULT_NORMALIZATION_METHOD,
    )

    language_analyzer = LanguageAttentionAnalyzer(
        normalization_method=Settings.DEFAULT_NORMALIZATION_METHOD,
    )

    with gr.Blocks(title="VLAExplain - Unifolm VLA Attention Analysis Platform") as vis:
        gr.Markdown("# VLAExplain - Unifolm VLA Attention Analysis Platform")

        with gr.Tabs():
            # Tab 1: Action Attribution Analysis (DIT Expert Network)
            create_action_tab(vis, action_analyzer, n_action_steps=n_action_steps)
            # Tab 2: Language Model Analysis
            create_language_tab(language_analyzer, n_action_steps=n_action_steps)

    return vis