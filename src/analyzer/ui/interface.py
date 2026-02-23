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

    components = initialize_action_components(action_analyzer)

    with gr.TabItem("Action Attribution Analysis"):
        gr.Markdown("## Parameter Selection")

        # State variables
        step_state = gr.State(components["step"])

        # Interactive component row
        with gr.Row():
            with gr.Column(scale=1):
                attention_head = gr.Radio(
                    choices=["Head 1", "Head 2", "Head 3", "Head 4",
                             "Head 5", "Head 6", "Head 7", "Head 8", "Average Pooling Head"],
                    value="Average Pooling Head",
                    label="Attention Head Selection"
                )
            with gr.Column(scale=4):
                with gr.Row():
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

        # Visualization layout
        with gr.Row():
            gr.Markdown("## Action Attribution Attention Visualization")
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    raw_img1_output = gr.Image(type="pil", 
                                               label="Raw Image1", 
                                               height=300, 
                                            #    elem_classes=["rotate-180"]
                                               )
                    raw_img2_output = gr.Image(type="pil", 
                                               label="Raw Image2", 
                                               height=300, 
                                            #    elem_classes=["rotate-180"]
                                               )
                with gr.Row():
                    view1_attn_output = gr.Image(type="pil", 
                                                 label="View1 Attention Heatmap", 
                                                 height=300,
                                                #  elem_classes=["rotate-180"]
                                                 )
                    view2_attn_output = gr.Image(type="pil", 
                                                 label="View2 Attention Heatmap", 
                                                 height=300,
                                                #  elem_classes=["rotate-180"]
                                                 )
            with gr.Column(scale=2):
                text_attn_output = gr.Image(type="pil", label="Text Attention", height=300)
                state_attn_output = gr.Image(type="pil", label="State Attention", height=300)
        with gr.Row():
            action_seq_output = gr.Image(type="pil", label="Action Sequence Attention Distribution", height=200)

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
        all_inputs = [
            step_slider, time_step_selector, attention_head, layer_selector,
            alpha_slider, normalization_method_dropdown, interpolation_method_dropdown, colormap_dropdown
        ]

        for component in [step_slider, time_step_selector, attention_head, layer_selector,
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
                            state_attn_output, 
                            action_seq_output
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
                        state_attn_output, 
                        action_seq_output
                    ]
        )


def create_language_tab(language_analyzer, n_action_steps=1):
    """Create the language model analysis tab."""
    # === Text → Vision/State Functions ===
    def update_text_dropdown_lang(step):
        choices = language_analyzer.processor.get_token_list_with_index(step)
        return gr.update(choices=choices, value=[choices[0]])
    # === Vision → Text/State Functions ===
    def init_image_lang(step, image_view):
        return language_analyzer.processor.get_image_with_grid(step, image_view)
    def update_image_lang(step, image_view):
        updated_image = language_analyzer.processor.draw_selected_patches_on_image(step, image_view)
        return updated_image
    def handle_image1_click_lang(img, step, event: gr.SelectData):
        updated_image, selected_patches = language_analyzer.processor.image_click_handler(step, "1", event)
        return updated_image, str(selected_patches)
    def clear_patches1_lang(image_view, step):
        _, image = language_analyzer.processor.clear_selected_patches(image_view, step)
        return str(language_analyzer.processor.global_selected_patch_indices["image1"]), image
    def handle_image2_click_lang(step, event: gr.SelectData):
        updated_image, selected_patches = language_analyzer.processor.image_click_handler(step, "2", event)
        return updated_image, str(selected_patches)
    def clear_patches2_lang(image_view, step):
        _, image = language_analyzer.processor.clear_selected_patches(image_view, step)
        return str(language_analyzer.processor.global_selected_patch_indices["image2"]), image
    
    # === State → Vision/Text Functions ===
    def update_state_dropdown_lang(step):
        choices = language_analyzer.processor.get_state_list_with_index(step)
        return gr.update(choices=choices, value=[choices[0]])


    with gr.TabItem("Language Model Analysis"):
        gr.Markdown("## Parameter Selection")

        available_steps_lang = list(language_analyzer.processor.decoded_texts_steps.keys())

        # Global parameter settings
        with gr.Row():
            with gr.Column(scale=1):
                head_dropdown_lang = gr.Radio(
                    choices=["Head 1", "Head 2", "Head 3", "Head 4",
                             "Head 5", "Head 6", "Head 7", "Head 8", "Average Pooling Head"],
                    value="Average Pooling Head",
                    label="Attention Head Selection"
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

        # Text/state preview
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

            with gr.Column(scale=1):
                gr.Markdown("## Original State Preview")
                state_list_box_lang = gr.Textbox(
                    label="Origial State",
                    value=" ".join([str(s) for s in language_analyzer.processor.get_state_list(0)]),
                    interactive=False,
                    lines=2,
                    max_lines=10
                )

        # Multimodal cross-attention visualization tabs
        with gr.Tabs():
            with gr.TabItem("Text → Vision/State"):
                gr.Markdown("### Text Attention to View1/View2 and State Visualization")
                
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
                            image_overlay_1_o_lang = gr.Image(label="View1 Original", type="numpy")
                            image_overlay_1_lang = gr.Image(label="Text→View1 Attention Heatmap", type="numpy")
                            image_overlay_2_o_lang = gr.Image(label="View2 Original", type="numpy")
                            image_overlay_2_lang = gr.Image(label="Text→View2 Attention Heatmap", type="numpy")
                        with gr.Row():
                            state_attn_1_lang = gr.Image(label="Text→State Attention Distribution", type="numpy")

            with gr.TabItem("Vision → Text/State"):
                gr.Markdown("### Click View1/View2 Patches to Visualize Attention to Text and State")                

                with gr.Row():
                    with gr.Column(scale=1):
                        image1_display_lang = gr.Image(
                            label="View1 (with Patch Grid)",
                            type="numpy",
                            interactive=True,
                            height=300,
                            value=init_image_lang(0, "1")
                        )
                        selected_patches_text1_lang = gr.Textbox(
                            label="Selected Patches in View1",
                            interactive=False,
                            value=str(language_analyzer.processor.global_selected_patch_indices["image1"])
                        )
                        clear_patch1_btn_lang = gr.Button("Clear Selected Patches in View1")

                    with gr.Column(scale=4):
                        with gr.Row():
                            text_attn_2_1_lang = gr.Image(label="View1 → Text Attention Distribution", type="numpy")
                            state_attn_2_1_lang = gr.Image(label="View1 → State Attention Distribution", type="numpy")
                            

                with gr.Row():
                    with gr.Column(scale=1):
                        image2_display_lang = gr.Image(
                            label="View2 (with Patch Grid)",
                            type="numpy",
                            interactive=True,
                            height=300,
                            value=init_image_lang(0, "2")
                        )
                        selected_patches_text2_lang = gr.Textbox(
                            label="Selected Patches in View2",
                            interactive=False,
                            value=str(language_analyzer.processor.global_selected_patch_indices["image2"])
                        )
                        clear_patch2_btn_lang = gr.Button("Clear Selected Patches in View2")
                    
                    with gr.Column(scale=4):
                        with gr.Row():
                            text_attn_2_2_lang = gr.Image(label="View2 → Text Attention Distribution", type="numpy")
                            state_attn_2_2_lang = gr.Image(label="View2 → State Attention Distribution", type="numpy")

                with gr.Row():
                    with gr.Column(scale=1):
                        attention_type_2_lang = gr.Radio(
                            choices=["Fine-grained", "Global"],
                            value="Fine-grained",
                            label="Attention Type",
                            info="Fine-grained: Selected patches; Global: Average of all patches"
                        )
                        
                        submit_2_lang = gr.Button("Generate Visualization", variant="primary")
                    with gr.Column(scale=4):
                        pass

            with gr.TabItem("State → Vision/Text"):
                gr.Markdown("### Decoded State Attention to Text and View1/View2 Visualization")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            state_dropdown_lang = gr.Dropdown(
                                choices=language_analyzer.processor.get_state_list_with_index(available_steps_lang[0]) if available_steps_lang else [],
                                value=[language_analyzer.processor.get_state_list_with_index(available_steps_lang[0])[0]] if available_steps_lang and language_analyzer.processor.get_state_list_with_index(available_steps_lang[0]) else [],
                                label="Select State (Supports Multi-select)",
                                multiselect=True,
                                info="Choose from dropdown or input manually"
                            )
                        with gr.Row():
                            attention_type_3_lang = gr.Radio(
                                choices=["Fine-grained", "Global"],
                                value="Fine-grained",
                                label="Attention Type",
                                info="Fine-grained: Selected states; Global: Average of all states"
                            )
                        with gr.Row():
                            submit_3_lang = gr.Button("Generate Visualization", variant="primary")

                    with gr.Column(scale=4):
                        with gr.Row():
                            image_origin_3_1_lang = gr.Image(label="View1 Original", type="numpy")
                            image_overlay_3_1_lang = gr.Image(label="State→View1 Attention Heatmap", type="numpy")
                            image_origin_3_2_lang = gr.Image(label="View2 Original", type="numpy")
                            image_overlay_3_2_lang = gr.Image(label="State→View2 Attention Heatmap", type="numpy")
                        with gr.Row():
                            text_attn_3_lang = gr.Image(label="State→Text Attention Distribution", type="numpy")
                            
            # === Text State Preview Event Binding ===
            step_slider_lang.change(
                    fn=lambda step: " ".join(language_analyzer.processor.get_token_list(step)),
                    inputs=step_slider_lang,
                    outputs=token_list_box_lang
                )
            step_slider_lang.change(
                    fn=lambda step: " ".join([str(s) for s in language_analyzer.processor.get_state_list(step)]),
                    inputs=step_slider_lang,
                    outputs=state_list_box_lang
                )
            # === Text → Vision/State Event Binding ===
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
                outputs=[image_overlay_1_o_lang, image_overlay_1_lang, image_overlay_2_o_lang, image_overlay_2_lang, state_attn_1_lang]
            )

            # === Vision → Text/State Event Binding ===
            step_slider_lang.change(
                fn=update_image_lang,
                inputs=[step_slider_lang, gr.State(value="1")],
                outputs=image1_display_lang
            )
            
            image1_display_lang.select(
                fn=handle_image1_click_lang,
                inputs=[image1_display_lang, step_slider_lang],
                outputs=[image1_display_lang, selected_patches_text1_lang]
            )

            clear_patch1_btn_lang.click(
                fn=clear_patches1_lang,
                inputs=[gr.State("1"), step_slider_lang],
                outputs=[selected_patches_text1_lang, image1_display_lang]
            )

            step_slider_lang.change(
                fn=update_image_lang,
                inputs=[step_slider_lang, gr.State(value="2")],
                outputs=image2_display_lang
            )

            image2_display_lang.select(
                fn=handle_image2_click_lang,
                inputs=[step_slider_lang],
                outputs=[image2_display_lang, selected_patches_text2_lang]
            )

            clear_patch2_btn_lang.click(
                fn=clear_patches2_lang,
                inputs=[gr.State("2"), step_slider_lang],
                outputs=[selected_patches_text2_lang, image2_display_lang]
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
                outputs=[text_attn_2_1_lang, text_attn_2_2_lang, state_attn_2_1_lang, state_attn_2_2_lang]
            )

            # === State → Vision/Text Event Binding ===
            step_slider_lang.change(
                fn=update_state_dropdown_lang,
                inputs=step_slider_lang,
                outputs=state_dropdown_lang
            )

            submit_3_lang.click(
                fn=language_analyzer.state_vis_wrapper,
                inputs=[
                    step_slider_lang, 
                    layer_dropdown_lang, 
                    head_dropdown_lang,
                    state_dropdown_lang, 
                    attention_type_3_lang,
                    alpha_slider_lang, 
                    cmap_dropdown_lang, 
                    normalization_method_dropdown_lang,
                    interpolation_method_dropdown_lang
                ],
                outputs=[text_attn_3_lang, image_origin_3_1_lang, image_overlay_3_1_lang, image_origin_3_2_lang, image_overlay_3_2_lang]
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

    with gr.Blocks(title="VLAExplain-Pi05 Attention Analysis Platform") as vis:
        gr.Markdown("# VLAExplain-Pi05 Attention Analysis Platform")

        with gr.Tabs():
            create_action_tab(vis, action_analyzer, n_action_steps=n_action_steps)
            create_language_tab(language_analyzer, n_action_steps=n_action_steps)

    return vis