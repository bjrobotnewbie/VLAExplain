# 🧠 VLAExplain — Interpreting Vision-Language-Action (VLA) Models

VLAExplain is an interpretability toolkit designed to help users visually understand the inner workings of Vision-Language-Action (VLA) models.

Currently in its early stage, it supports two types of attention analysis for the Pi05 model:

- **ActionCode Attribution**: Understand how a predicted action attends to inputs from vision, language, robot state, and future actions.
- **Cross-Modal Attention in Language Model**: Analyze attention relationships between modalities—e.g., text → image/state, image → text/state, state → text/image.

✨ Many features and code quality improvements are still in progress and will be gradually implemented.

## 🎥 Demo

Check out our demo video to see VLAExplain in action:

<video src="assets/demos/demo_video.mp4" controls width="100%">
  Your browser does not support the video tag.
</video>

---

## 🚀 Installation

Follow these steps to set up the environment:

### Step 1: Clone LeRobot

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout v0.4.4  # Recommended version
```

### Step 2: Install Dependencies

Install in editable mode as per LeRobot’s [installation guide](https://huggingface.co/docs/lerobot/installation):

```bash
pip install -e .
```

Then install Pi05 and Libero dependencies:

- Follow the [Libero setup](https://huggingface.co/docs/lerobot/main/en/libero), install Libero dependencies:

```bash
pip install -e ".[libero]"
```

- Follow the [Pi05 policy setup](https://huggingface.co/docs/lerobot/main/en/pi05), install Pi05-specific dependencies:

```bash
pip install -e ".[pi]"
```

Finally, install visualization app requirements:
```bash
cd ..
pip install -r requirements.txt
```

### Step 3: Replace Key Files

Copy and overwrite the following files:

```bash
# Model files
cp src/policies/pi05/model/* lerobot/src/lerobot/policies/pi05/

# Evaluation script
cp src/policies/pi05/infer/lerobot_eval.py lerobot/src/lerobot/scripts/lerobot_eval.py
```

---

## ▶️ Usage

### Run Inference & Collect Attention Data

```bash
bash libero_eval.sh
```

### 📁 Output Directory

Data is saved under the path specified by the environment variable `LEROBOT_DATA_DIR`, containing:

- `expert_attention/` — Action-to-modality attention weights (image, text, state)
- `language_attention/` — Cross-modal attention within the language model
- `language_info/` — input_ids and state values at each action step
- `raw_images/` — Original RGB frames from robot execution

### 💡 Recommendations

- Set `eval.batch_size = 1` and `eval.n_episodes = 1`
- Choose one `env.task_ids` for cleaner analysis
  (Multi-episode support will be added later)

### ⚠️ Storage Note

Each inference step consumes ~420 MB of disk space due to high-resolution attention maps.

---

## 🖼️ Visualization App Features

Launch the attention visualization tool with:
```bash
bash run_app.sh
```

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

| Modality  | Action Attribution | Cross-Modal Attention |
|-----------|--------------------|-----------------------|
| Image     | ✅                 | ✅                    |
| Text      | ✅                 | ✅                    |
| State     | ✅                 | ✅                    |
| Action    | ✅                 | ❌                    |

---

## 🙌 Support the Project

If you find VLAExplain useful, please give us a ⭐ on GitHub!

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

We sincerely thank the developers of [LeRobot](https://github.com/huggingface/lerobot) and [Gradio](https://github.com/gradio-app/gradio).
This project builds directly on their excellent open-source framework and contributions to the robotics community.
