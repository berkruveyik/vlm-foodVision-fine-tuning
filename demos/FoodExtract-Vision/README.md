---
title: FoodExtract-Vision
emoji: 🍕
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: "5.50.0"
python_version: "3.12"
app_file: app.py
pinned: false
---

# 🍕🔍 FoodExtract-Vision v1: Fine-tuned SmolVLM2-500M for Structured Food Tag Extraction

[![Model on HuggingFace](https://img.shields.io/badge/🤗%20Model-FoodExtract--Vision--SmolVLM2--500M-blue)](https://huggingface.co/berkeruveyik/FoodExtract-Vision-SmolVLM2-500M-fine-tune-v3)
[![Dataset on HuggingFace](https://img.shields.io/badge/🤗%20Dataset-vlm--food--4k--not--food-green)](https://huggingface.co/datasets/berkeruveyik/vlm-food-4k-not-food-dataset)
[![Base Model](https://img.shields.io/badge/🧠%20Base-SmolVLM2--500M--Video--Instruct-orange)](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)
[![License](https://img.shields.io/badge/📄%20License-Apache%202.0-lightgrey)](https://www.apache.org/licenses/LICENSE-2.0)

---

## 📋 Overview

**FoodExtract-Vision** is a fine-tuned Vision-Language Model (VLM) that takes any image as input and produces **structured JSON output** classifying whether food/drink items are visible and extracting them into organized lists.

Built on top of [SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct), this project demonstrates that even **small (~500M parameter) VLMs** can be fine-tuned to reliably produce structured outputs for domain-specific tasks — without needing PEFT/LoRA adapters.

> 💡 **Key Insight:** The base model often fails to follow the required JSON output structure, producing inconsistent or unstructured responses. After two-stage fine-tuning, the model **reliably generates valid JSON** matching the specified schema.

---

## 🎯 What Does It Do?

| | Input | Output |
|---|---|---|
| 📸 | Any image (food or non-food) | Structured JSON |

### Output Schema

```json
{
  "is_food": 1,
  "image_title": "Tandoori chicken with naan bread",
  "food_items": ["tandoori chicken", "naan bread", "rice", "salad"],
  "drink_items": ["lassi"]
}
```

| Field | Type | Description |
|---|---|---|
| `is_food` | `int` | `0` = no food/drink visible, `1` = food/drink visible |
| `image_title` | `str` | Short food-related caption (blank if no food) |
| `food_items` | `list[str]` | List of visible edible food item nouns |
| `drink_items` | `list[str]` | List of visible edible drink item nouns |

---

## 🛠️ What Was Done — End-to-End Pipeline

This project covers the **full ML lifecycle** from dataset creation to deployment:

### Step 1: 📊 Dataset Creation (`00_create_vlm_dataset.ipynb`)

1. 🏷️ Loaded food labels from `data/food_dataset-2.jsonl` (generated via Qwen3-VL-8B inference on Food270 images)
2. 📝 Added metadata fields (`image_id`, `image_name`, `food270_class_name`, `image_source`)
3. 🖼️ Sampled **not-food images** from `data/not_food/` and created empty labels with `is_food = 0`
4. 🔀 Merged food + not-food labels into a unified dataset
5. 📁 Copied all images into `data/food_all/` and wrote `metadata.jsonl` for HuggingFace `imagefolder` format
6. 🚀 Pushed to HuggingFace Hub as [`berkeruveyik/vlm-food-4k-not-food-dataset`](https://huggingface.co/datasets/berkeruveyik/vlm-food-4k-not-food-dataset)

**Final dataset:** ~3,698 image-JSON pairs across **270 food categories** + not-food images

### Step 2: 🧪 Base Model Evaluation (`01_fine_tune_vlm_v3_smolVLM_500m.ipynb`)

- Tested `SmolVLM2-500M-Video-Instruct` on the food extraction task
- **Result:** The base model produced unstructured text like *"The given image is a food or drink item."* instead of valid JSON
- ❌ Base model **cannot** follow the structured output format

### Step 3: 📐 Data Formatting for SFT

Converted each sample to a **conversational message format** with three roles:

```
[SYSTEM] → Expert food extractor persona
[USER]   → Image + JSON extraction prompt
[ASSISTANT] → Ground truth JSON output
```

- Used `PIL.Image` objects directly (not bytes) to preserve image quality
- 80/20 train/validation split with `random.seed(42)` for reproducibility

### Step 4: 🧊 Stage 1 Training — Frozen Vision Encoder

- **Froze** the vision encoder (`model.model.vision_model`)
- **Trained** only the LLM + connector layers
- **Goal:** Teach the language model to output valid JSON structure
- Used `SFTTrainer` from TRL with custom `collate_fn` for image-text batching

### Step 5: 🔥 Stage 2 Training — Full Model Fine-tuning

- **Unfroze** the vision encoder
- **Trained** all parameters with a **100x lower learning rate** (`2e-6` vs `2e-4`)
- **Goal:** Allow the vision encoder to adapt for better food recognition without catastrophic forgetting

### Step 6: 📈 Evaluation & Comparison

- Compared outputs from 3 models side-by-side:
  - 🔴 **Pre-trained** (base model) — fails at structured output
  - 🟡 **Stage 1** (frozen vision) — learns JSON format
  - 🟢 **Stage 2** (full fine-tune) — best food recognition + JSON format

### Step 7: 🚀 Deployment

- Uploaded fine-tuned model to HuggingFace Hub
- Built Gradio demo with side-by-side comparison
- Deployed as a HuggingFace Space

---

## 🏗️ Architecture & Training Details

### 🧠 Base Model

| Property | Value |
|---|---|
| Model | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` |
| Parameters | ~500M |
| Precision | `bfloat16` |
| Attention | `eager` |

### 📊 Dataset

| Property | Value |
|---|---|
| Source | [`berkeruveyik/vlm-food-4k-not-food-dataset`](https://huggingface.co/datasets/berkeruveyik/vlm-food-4k-not-food-dataset) |
| Total Samples | ~3,698 image-JSON pairs |
| Train / Val Split | 80% / 20% |
| Food Categories | 270 (from Food270 dataset) |
| Non-food Images | Random internet images |
| Label Source | Qwen3-VL-8B inference outputs |

### 🔧 Two-Stage Training Strategy

Inspired by the [SmolVLM Docling paper](https://arxiv.org/pdf/2503.11576):

#### 🧊 Stage 1: LLM Alignment (Frozen Vision Encoder)

| Parameter | Value |
|---|---|
| Vision Encoder | ❄️ Frozen |
| Trainable | LLM + connector layers |
| Learning Rate | `2e-4` |
| Epochs | 2 |
| Batch Size | 8 × 4 gradient accumulation = effective 32 |
| Optimizer | `adamw_torch_fused` |
| LR Scheduler | `constant` |
| Warmup Ratio | `0.03` |
| Precision | `bf16` |

#### 🔥 Stage 2: Full Model Fine-tuning (Unfrozen Vision Encoder)

| Parameter | Value |
|---|---|
| Vision Encoder | 🔥 Unfrozen |
| Trainable | All parameters |
| Learning Rate | `2e-6` (100x lower than Stage 1) |
| Epochs | 2 |
| Batch Size | 8 × 4 gradient accumulation = effective 32 |
| Optimizer | `adamw_torch_fused` |
| LR Scheduler | `constant` |
| Warmup Ratio | `0.03` |
| Precision | `bf16` |

---

## 🚀 Quick Start

### 📦 Installation

```bash
pip install transformers torch gradio spaces accelerate
```

### 🔮 Inference with Pipeline

```python
import torch
from transformers import pipeline
from PIL import Image

FINE_TUNED_MODEL_ID = "berkeruveyik/FoodExtraqt-Vision-SmoLVLM2-500M-fine-tune-v3"

pipe = pipeline(
    "image-text-to-text",
    model=FINE_TUNED_MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
)

prompt = """Classify the given input image into food or not and if edible food or drink items are present, extract those to a list. If no food/drink items are visible, return empty lists.

Only return valid JSON in the following form:

```json
{
  "is_food": 0,
  "image_title": "",
  "food_items": [],
  "drink_items": []
}
```
"""

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "path/to/your/image.jpg"},
            {"type": "text", "text": prompt},
        ],
    }
]

output = pipe(text=messages, max_new_tokens=256)
print(output[0][0]["generated_text"][-1]["content"])
```

### 🧪 Inference without Pipeline

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

FINE_TUNED_MODEL_ID = "berkeruveyik/FoodExtraqt-Vision-SmoLVLM2-500M-fine-tune-v3"

model = AutoModelForImageTextToText.from_pretrained(
    FINE_TUNED_MODEL_ID,
    attn_implementation="eager",
    dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(FINE_TUNED_MODEL_ID)

image = Image.open("path/to/your/image.jpg")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "YOUR_PROMPT_HERE"},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=256, do_sample=False)

decoded = processor.decode(output[0][input_len:], skip_special_tokens=True)
print(decoded)
```

---

## 🎮 Gradio Demo

This Space runs a **side-by-side comparison** between the base model and the fine-tuned model.

### ▶️ Running Locally

```bash
cd demos/FoodExtract-Vision
pip install -r requirements.txt
python app.py
```

### 🖥️ What the Demo Shows

1. 📤 **Upload** any image
2. 🔄 **Compare** outputs from the base model vs. the fine-tuned model side-by-side
3. 📊 See how fine-tuning enables **reliable structured JSON extraction**

### 📸 Example Images Included

The demo comes with pre-loaded examples to try instantly.

---

## 📁 Project Structure

```
vlm_finetune/
├── 📓 00_create_vlm_dataset.ipynb          # Dataset creation pipeline
├── 📓 01-fine_tune_vlm.ipynb               # First fine-tuning experiment (Gemma-3n)
├── 📓 01-fine_tune_vlm-v2-smolVLM.ipynb    # SmolVLM 256M experiment
├── 📓 01_fine_tune_vlm_v3_smolVLM_500m.ipynb # ✅ Final: SmolVLM 500M two-stage training
├── 📓 qwen3-food270-inference-viewer.ipynb  # Dataset visualization tool
├── 📄 README.md                            # Root project README
├── 📁 data/
│   ├── food_dataset-2.jsonl                # Qwen3-VL-8B inference outputs
│   ├── food_labels_updated.json            # Processed food labels
│   ├── 📁 10_images_270_class/             # 10 sample images per category
│   ├── 📁 food_all/                        # Merged dataset (food + not-food)
│   │   └── metadata.jsonl                  # HuggingFace imagefolder metadata
│   └── 📁 not_food/                        # Non-food images
└── 📁 demos/
    └── 📁 FoodExtract-Vision/
        ├── app.py                          # 🚀 Gradio demo application
        ├── README.md                       # 📖 This file
        ├── requirements.txt                # 📦 Python dependencies
        └── 📁 examples/                    # 🖼️ Example images
            ├── 36741.jpg
            ├── IMG_3808.JPG
            └── istockphoto-175500494-612x612.jpg
```

---

## 📝 Key Learnings & Notes

### ✅ What Worked

- 🏗️ **Two-stage training** significantly improved output quality compared to single-stage
- 🧊 **Freezing the vision encoder first** let the LLM learn JSON format without vision interference
- 🐢 **100x lower learning rate in Stage 2** (`2e-6` vs `2e-4`) prevented catastrophic forgetting
- 🤏 Even a **500M parameter model** can learn reliable structured output generation
- 📝 **Custom `collate_fn`** with proper label masking (pad tokens + image tokens → `-100`) was essential
- 🔀 **`remove_unused_columns = False`** is critical when using a custom data collator with `SFTTrainer`

### ⚠️ Important Notes

- **Dtype consistency:** Model inputs must match the model's dtype (e.g., `bfloat16` inputs for a `bfloat16` model)
- **System prompt handling:** When not using `transformers.pipeline`, the system prompt may need to be folded into the user prompt
- **PIL images over bytes:** Using `format_data()` as a list comprehension instead of `dataset.map()` preserves PIL image types
- **Gradient checkpointing:** Set `use_reentrant=False` to avoid warnings and ensure compatibility

### 🧪 Experiments Tried

| Notebook | Model | Approach | Result |
|---|---|---|---|
| `01-fine_tune_vlm.ipynb` | Gemma-3n-E2B | QLoRA + PEFT | ✅ Works but larger model |
| `01-fine_tune_vlm-v2-smolVLM.ipynb` | SmolVLM2-256M | Full fine-tune | 🟡 Limited capacity |
| `01_fine_tune_vlm_v3_smolVLM_500m.ipynb` | SmolVLM2-500M | **Two-stage full fine-tune** | ✅ **Best results** |

---

## 🔗 Links

| Resource | URL |
|---|---|
| 🤗 Fine-tuned Model | [berkeruveyik/FoodExtraqt-Vision-SmoLVLM2-500M-fine-tune-v3](https://huggingface.co/berkeruveyik/FoodExtraqt-Vision-SmoLVLM2-500M-fine-tune-v3) |
| 🤗 Dataset | [berkeruveyik/vlm-food-4k-not-food-dataset](https://huggingface.co/datasets/berkeruveyik/vlm-food-4k-not-food-dataset) |
| 🤗 Base Model | [HuggingFaceTB/SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) |
| 📄 SmolVLM Docling Paper | [arxiv.org/pdf/2503.11576](https://arxiv.org/pdf/2503.11576) |
| 📚 TRL Documentation | [huggingface.co/docs/trl](https://huggingface.co/docs/trl/main/en/index) |
| 📚 PEFT GitHub | [github.com/huggingface/peft](https://github.com/huggingface/peft) |
| 📚 HF Vision Fine-tune Guide | [ai.google.dev/gemma/docs](https://ai.google.dev/gemma/docs/core/huggingface_vision_finetune_qlora?hl=tr) |

---

## 📄 License

This project uses Apache 2.0 license. Please refer to the respective model and dataset cards for additional licensing information.

---

*Built with ❤️ using 🤗 Transformers, TRL, and Gradio — by [Berker Üveyik](https://huggingface.co/berkeruveyik)*
