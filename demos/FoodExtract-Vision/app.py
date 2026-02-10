import torch
import gradio as gr
import spaces
from transformers import pipeline

BASE_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
FINE_TUNED_MODEL_ID = 'berkeruveyik/FoodExtraqt-Vision-SmoLVLM2-500M-fine-tune-v3'
OUTPUT_TOKENS = 256

print(f"[INFO] Loading Original Model")
original_pipeline = pipeline("image-text-to-text", model=BASE_MODEL_ID, dtype=torch.bfloat16, device_map="auto")

print(f"[INFO] Loading Fine-tuned Model")
ft_pipe = pipeline("image-text-to-text", model=FINE_TUNED_MODEL_ID, dtype=torch.bfloat16, device_map="auto")

def create_message(input_image):
    return [{'role': 'user', 'content': [{'type': 'image', 'image': input_image}, {'type': 'text', 'text': "Classify the given input image into food or not and if edible food or drink items are present, extract those to a list. If no food/drink items are visible, return empty lists.\n\nOnly return valid JSON in the following form:\n\n```json\n{\n  'is_food': 0, # int - 0 or 1 based on whether food/drinks are present (0 = no foods visible, 1 = foods visible)\n  'image_title': '', # str - short food-related title for what foods/drinks are visible in the image, leave blank if no foods present\n  'food_items': [], # list[str] - list of visible edible food item nouns\n  'drink_items': [] # list[str] - list of visible edible drink item nouns\n}\n```\n"}]}]

@spaces.GPU
def extract_foods_from_image(input_image):
    input_image = input_image.resize(size=(512, 512))
    input_message = create_message(input_image=input_image)
    original_pipeline_output = original_pipeline(text=[input_message], max_new_tokens=OUTPUT_TOKENS)
    outputs_pretrained = original_pipeline_output[0][0]["generated_text"][-1]["content"]
    ft_pipe_output = ft_pipe(text=[input_message], max_new_tokens=OUTPUT_TOKENS)
    outputs_fine_tuned = ft_pipe_output[0][0]["generated_text"][-1]["content"]
    return outputs_pretrained, outputs_fine_tuned

demo_title = "🍕🔍 FoodExtract-Vision: Fine-tuned SmolVLM2-500M"
demo_description = """* **Base model:** https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct\n* **Fine-tuning dataset:** https://huggingface.co/datasets/berkeruveyik/vlm-food-4k-not-food-dataset\n* **Fine-tuned model:** https://huggingface.co/berkeruveyik/FoodExtract-Vision-SmolVLM2-500M-fine-tune-v3\n\n## 📋 Overview\n\nThis demo showcases the power of fine-tuning for structured output generation. Compare a base vision-language model against its fine-tuned version specialized in extracting food and drink items from images in JSON format.\n\nThe **base model** often fails to follow the required output structure, producing inconsistent or unstructured responses. The **fine-tuned model** reliably generates valid JSON outputs matching the specified schema.\n\n## 🎯 Task Description\n\nBoth models receive identical input prompts requesting food/drink classification and extraction:\n\n````\nClassify the given input image into food or not and if edible food or drink items are present, extract those to a list. If no food/drink items are visible, return empty lists.\n\nOnly return valid JSON in the following form:\n\n```json\n{\n  'is_food': 0, # int - 0 or 1 based on whether food/drinks are present (0 = no foods visible, 1 = foods visible)\n  'image_title': '', # str - short food-related title for what foods/drinks are visible in the image, leave blank if no foods present\n  'food_items': [], # list[str] - list of visible edible food item nouns\n  'drink_items': [] # list[str] - list of visible edible drink item nouns\n}\n```\n````\n\n## 🔧 Training Details\n\nThe fine-tuned model was trained on **3,698 images** from the vlm-food-4k-not-food-dataset:\n- **Food images:** Multiple categories from the Food270 dataset including various cuisines, ingredients, and prepared dishes\n- **Non-food images:** Random internet images to teach the model to correctly identify non-food content\n- Each image is labeled with structured JSON outputs including classification, titles, and extracted food/drink items"""

demo = gr.Interface(fn=extract_foods_from_image, inputs=gr.Image(type="pil"), title=demo_title, description=demo_description, outputs=[gr.Textbox(lines=4, label="Original Model (not fine-tuned)"), gr.Textbox(lines=4, label="Fine-tuned Model")], examples=[["./examples/36741.jpg"], ["./examples/IMG_3808.JPG"], ["./examples/istockphoto-175500494-612x612.jpg"]])

if __name__ == "__main__":
    demo.launch(share=True)
