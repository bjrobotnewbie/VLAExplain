# 🧠 VLAExplain — UnifoLM-VLA Model Attention Analysis

UnifoLM-VLA model attention analysis supports:

- **Action Attribution**: For the current action, analyze the cross-attention and self-attention patterns of the action expert network DiT
    - DiT even layers use cross-attention mode, analyzing the attention correlation between the current action and visual and task text descriptions
    - DiT odd layers use self-attention mode, including token-level self-attention correlations for three modules: state, future tokens, and actions, as well as module-level attention intensity comparison
      - Module-level attention intensity includes total intensity (reflecting overall distribution) and mean intensity (eliminating dimensional differences)


- **Cross-Modal Attention in Language Model**: Analyze attention relationships between modalities—e.g., text → image, image → text.

This project has been tested in the LIBERO environment and has not been experimented on real robots due to hardware limitations.


## 🎥 Demo

Check out our Unifolm VLA demo to see VLAExplain in action:

<img src="../../assets/unifolm_vla_demo.gif" alt="Demo GIF" width="100%" />


---

## 🚀 Installation

Follow these steps to set up the environment:

### Step 1: Clone Unifolm-VLA Project and Install Dependencies

```bash
cd projects
git clone https://github.com/unitreerobotics/unifolm-vla
```

Install the project development environment according to the description in the README file of the project, including the LIBERO environment. After installation, return to the main directory.


### Step 2: Replace Key Files

Copy and overwrite the following files:

```bash
bash src/unifolm-vla/scripts/run_eval_libero.sh
```

---

## ▶️ Usage

### Run Inference & Collect Attention Data


```bash
bash src/unifolm-vla/scripts/run_eval_libero.sh
```

Key Parameter Descriptions:
- `LIBERO_HOME`: Path to the LIBERO project directory
- `VLA_DATA_DIR`: Directory path for saving attention weights and other information
- `POLICY_PATH`: Path to UnifoLM-VLA-Libero model weights, e.g., `UnifoLM-VLA-Libero/checkpoints/pytorch_model.pt`
- `VLM_PRETRAINED_PATH`: Path to UnifoLM-VLM-Base directory
- `TASK_SUITE_NAME`: LIBERO task suite
- `UNORM_KEY`: {TASK_SUITE_NAME}_no_noops
- `TASK_ID`: Task ID (required). Currently, the project only supports analysis of one task ID at a time

### 📁 Output Directory

Data is saved under the path specified by the environment variable `VLA_DATA_DIR`, containing:

- `expert_attention/` — Action attribution attention weights
- `language_attention/` — Cross-modal attention within the language model
- `language_info/` — input_ids and state values at each action step
- `raw_images/` — Original RGB frames from robot execution


### ⚠️ Storage Note

One task inference consumes ~4.5 GB of disk space due to high-resolution attention maps.

---

## 🖼️ Visualization App Features

Launch the attention visualization tool with:
```bash
bash src/unifolm-vla/scripts/run_analyzer.sh
```

Key Parameter Descriptions:
- `VLA_DATA_DIR`: Directory path for saving attention weights and other information
- `TOKENIZER_PATH`: Directory path of tokenizer, "Qwen2.5-VL-3B-Instruct"
- `N_ACTION_STEPS`: Number of action steps per prediction, default is 8

### Supported Features

| Feature                 | Action Attribution | Cross-Modal Attention |
|-------------------------|--------------------|-----------------------|
| Step Selector           | ✅                 | ✅                    |
| Time Step Filter        | ✅                 | ❌                    |
| Head Number             | ✅                 | ✅                    |
| Layer Index             | ✅                 | ✅                    |
| Normalization Method    | ✅                 | ✅                    |
| Interpolation Method    | ✅                 | ✅                    |
| Color Map               | ✅                 | ✅                    |
| Opacity Control         | ✅                 | ✅                    |

### Supported Modalities

| Modality  | Action Attribution Self-attention |Action Attribution Cross-attention | Cross-Modal Attention |
|-----------|--------------------|-------------------|-----------------------|
| Image     | ❌                 | ✅                | ✅                    |
| Text      | ❌                 | ✅                | ✅                    |
| State     | ✅                 | ❌                | ❌                    |
| Future tokens | ✅                 | ❌                | ❌                    |
| Action    | ✅                 | ✅                | ❌                    |

---

## 🙌 Support the Project

If you find VLAExplain useful, please give us a ⭐ on GitHub!

We warmly welcome everyone to contribute to the development of VLAExplain! 

---

## 📚 Citation

If you use this code in your research or project, please cite:

```bibtex
@misc{shi2026vlaexplain,
  title        = {{VLAExplain}: Interpreting Vision-Language-Action ({VLA}) Models},
  author       = {Shi, Yafei},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/bjrobotnewbie/VLAExplain}}
}
```

---

## 📜 License

This project is licensed under the [**AGPLv3 License**](LICENSE).

---

## 🙏 Acknowledgements

We sincerely thank the developers of [UnifoLM-VLA](https://github.com/unitreerobotics/unifolm-vla) and [Gradio](https://github.com/gradio-app/gradio).
This project builds directly on their excellent open-source framework and contributions to the robotics community.
