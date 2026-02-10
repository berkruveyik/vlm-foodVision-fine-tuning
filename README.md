# FoodExtract-Vision: Model Evolution and Final Selection

This document explains the full experiment path for this repository, from the first Gemma-based trial to the final SmolVLM2-500M model and deployed demo.

The notebooks were developed in this order:

1. `01-fine_tune_vlm-gemma-3n.ipynb`
2. `01-fine_tune_vlm-v2-smolVLM.ipynb`
3. `01_fine_tune_vlm_v2_smolVLM_Without_Peft.ipynb`
4. `01_fine_tune_vlm_v3_smolVLM_500m.ipynb` (final)

Space demo (final): https://huggingface.co/spaces/berkeruveyik/FoodExtract-Vision

## 1. Project Goal

The target task is structured food information extraction from images.

Input:
- Any image (food or non-food)

Output:
- Strict JSON with:
  - `is_food` (0 or 1)
  - `image_title` (short title)
  - `food_items` (list)
  - `drink_items` (list)

Core challenge:
- Base VLMs can describe images, but they often fail to follow strict JSON formatting reliably.

## 2. Shared Pipeline Across Notebooks

Across all four notebooks, the same core pipeline is used:

1. Load dataset from Hugging Face:
   - `berkeruveyik/vlm-food-4k-not-food-dataset`
2. Build a conversation-style training format:
   - `system` role: extractor behavior
   - `user` role: image + schema prompt
   - `assistant` role: target JSON
3. Keep images as PIL objects (not bytes) using list-comprehension formatting.
4. Custom collate function with `processor.apply_chat_template(...)`.
5. 80/20 split with `random.seed(42)`.
6. Train with `trl.SFTTrainer`.

This continuity makes the model comparisons meaningful, because changes are mainly in model family, fine-tuning strategy, and hyperparameters.

## 3. Notebook-by-Notebook Evolution

## 3.1 `01-fine_tune_vlm-gemma-3n.ipynb` (First Attempt)

Base model:
- `google/gemma-3n-E2B-it`

Approach:
- PEFT/LoRA fine-tuning (adapter-based)
- LoRA setup:
  - `r=16`
  - `lora_alpha=8`
  - `lora_dropout=0.05`
  - `target_modules="all-linear"`
- Training via `SFTTrainer` with `peft_config`

Training config highlights:
- `num_train_epochs=1`
- `per_device_train_batch_size=8`
- `gradient_accumulation_steps=8`
- `learning_rate=2e-5`
- validation/checkpoint every 5 steps
- `load_best_model_at_end=True`

What this notebook established:
- End-to-end VLM fine-tuning flow for strict structured output.
- Adapter loading/testing workflow after training.

Limitations observed in practice:
- Heavier base model and PEFT complexity for deployment/maintenance.
- Good proof-of-concept, but not the best practical balance for your target demo.

## 3.2 `01-fine_tune_vlm-v2-smolVLM.ipynb` (Move to Smaller Model)

Base model:
- `HuggingFaceTB/SmolVLM2-256M-Video-Instruct`

Main changes from previous notebook:
- Switched from Gemma-3n to SmolVLM2-256M for a lighter model.
- Continued using LoRA (PEFT), but adapted config:
  - `r=16`
  - `lora_alpha=16`
  - `lora_dropout=0.05`
  - `target_modules="all-linear"`
  - `modules_to_save=["lm_head", "embed_tokens"]`
- Added explicit vision encoder freezing step:
  - freeze `model.model.vision_model` parameters
- Fixed data collator masking behavior for image tokens (`<image>` ID).

Training config highlights:
- `num_train_epochs=1`
- `per_device_train_batch_size=4`
- `gradient_accumulation_steps=4`
- `learning_rate=2e-4`
- epoch-based eval/save

What improved:
- Lower compute footprint.
- Cleaner collation/token masking behavior.
- Better control over what gets trained.

Remaining issues:
- 256M capacity is limited for robust extraction quality on harder images.
- PEFT path still adds adapter-management overhead.

## 3.3 `01_fine_tune_vlm_v2_smolVLM_Without_Peft.ipynb` (Remove PEFT, Two-Stage Full Fine-Tuning)

Base model:
- `HuggingFaceTB/SmolVLM2-256M-Video-Instruct`

Main changes from previous notebook:
- PEFT/LoRA removed entirely.
- Switched to full fine-tuning with two-stage strategy:
  - Stage 1: freeze vision encoder, train language-side alignment
  - Stage 2: unfreeze vision encoder, train whole model at low LR

Stage 1 config:
- `num_train_epochs=1`
- `per_device_train_batch_size=4`
- `gradient_accumulation_steps=4`
- `learning_rate=2e-4`

Stage 2 config:
- `num_train_epochs=1`
- `per_device_train_batch_size=4`
- `gradient_accumulation_steps=4`
- `learning_rate=2e-6` (100x lower)

What improved:
- Simpler final artifact (no adapter merge/load complexity).
- Two-stage training aligns output format first, then improves visual adaptation.
- Includes side-by-side comparison and demo scaffolding for this version.

Remaining issue:
- Better than prior 256M setup, but still constrained by model size.

## 3.4 `01_fine_tune_vlm_v3_smolVLM_500m.ipynb` (Final Model)

Base model:
- `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`

Main changes from previous notebook:
- Model scaled from 256M to 500M.
- Kept successful two-stage full fine-tuning strategy.
- Increased training depth and throughput.

Stage 1 config:
- `num_train_epochs=2`
- `per_device_train_batch_size=8`
- `gradient_accumulation_steps=4`
- `learning_rate=2e-4`

Stage 2 config:
- `num_train_epochs=2`
- `per_device_train_batch_size=8`
- `gradient_accumulation_steps=4`
- `learning_rate=2e-6`

Additional outcomes in this notebook:
- Model upload flow to Hugging Face Hub.
- Demo app generation in `demos/FoodExtract-Vision/`.
- Hugging Face Space upload flow.

Why this became the final model:
- Best balance of:
  - structured JSON reliability
  - stronger visual recognition than 256M versions
  - practical deployability
- Keeps the simpler no-PEFT deployment path while improving capacity.

## 4. Cross-Notebook Comparison

| Notebook | Base Model | Fine-Tuning Strategy | Key Difference vs Previous | Practical Outcome |
|---|---|---|---|---|
| `01-fine_tune_vlm-gemma-3n.ipynb` | Gemma-3n-E2B | LoRA/PEFT | Baseline end-to-end setup | Strong first proof, heavier stack |
| `01-fine_tune_vlm-v2-smolVLM.ipynb` | SmolVLM2-256M | LoRA/PEFT + frozen vision | Smaller model, improved collator handling | Cheaper/faster, but limited capacity |
| `01_fine_tune_vlm_v2_smolVLM_Without_Peft.ipynb` | SmolVLM2-256M | Two-stage full FT (no PEFT) | Removed adapters, stage-1/stage-2 training | Simpler deployment, better behavior |
| `01_fine_tune_vlm_v3_smolVLM_500m.ipynb` | SmolVLM2-500M | Two-stage full FT (no PEFT) | Capacity increase + longer training | Final selected model |

## 5. Why the Final Model Was Selected

Your final selection (`01_fine_tune_vlm_v3_smolVLM_500m.ipynb`) is justified by the combination of:

1. Better capacity (500M vs 256M) for more accurate item extraction.
2. Two-stage training strategy that first enforces output structure, then improves visual adaptation.
3. No-PEFT final architecture, which simplifies serving and reproducibility.
4. End-to-end deployability already validated through the demo space.

## 6. How to Use the Final Model

## 6.1 Live Demo

Use the deployed Space:

- https://huggingface.co/spaces/berkeruveyik/FoodExtract-Vision

## 6.2 Run the demo locally

From repository root:

```bash
cd demos/FoodExtract-Vision
pip install -r requirements.txt
python app.py
```

## 6.3 Programmatic inference (pipeline)

```python
import torch
from transformers import pipeline

MODEL_ID = "berkeruveyik/FoodExtraqt-Vision-SmoLVLM2-500M-fine-tune-v3"

pipe = pipeline(
    "image-text-to-text",
    model=MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
)

prompt = """Classify the given input image into food or not and if edible food or drink items are present, extract those to a list. If no food/drink items are visible, return empty lists.

Only return valid JSON in the following form:
{
  'is_food': 0,
  'image_title': '',
  'food_items': [],
  'drink_items': []
}"""

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "path/to/image.jpg"},
        {"type": "text", "text": prompt},
    ],
}]

out = pipe(text=[messages], max_new_tokens=256)
print(out[0][0]["generated_text"][-1]["content"])
```

Note:
- In notebooks/code, the model repo is referenced as `FoodExtraqt...`.
- In some descriptions, it appears as `FoodExtract...`.
- Keep IDs consistent with your actual Hub repository naming.

## 7. Related Files

- Dataset preparation: `00_create_vlm_dataset.ipynb`
- Final training notebook: `01_fine_tune_vlm_v3_smolVLM_500m.ipynb`
- Final demo app: `demos/FoodExtract-Vision/app.py`
- Final demo Space metadata/readme: `demos/FoodExtract-Vision/README.md`

