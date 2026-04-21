import os
import copy
import random
import math
import numpy as np
import torch
import argparse
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from FSC import FSC147

# Configuration
MODEL_NAME = "llava_qwen"
CONV_TEMPLATE = "qwen_1_5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COUNTING_BOUND = 5000

PROMPT_TEMPLATE = (
    "\nHint: Please answer the question and provide the final answer at the end. \nQuestion: "
    "Based on the image, generate a title that predicts the counts of {}, "
    "formatted as 'a photo of [XX] [objects]' (for example: a photo of 15 apples)."
)

def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA Counting Inference")
    parser.add_argument("--pretrained_model", type=str, required=True, help="Path to pretrained model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_prediction(output_text):
    try:
        numbers = [float(x) for x in output_text.split() if x.replace('.', '', 1).isdigit()]
        pred = numbers[0] if numbers else 0
    except ValueError:
        pred = 0
    return pred if pred < COUNTING_BOUND else 0

def divide_image(image, grid_size=2):
    width, height = image.size
    pw, ph = width // grid_size, height // grid_size
    return [
        image.crop((i * pw, j * ph, (i + 1) * pw, (j + 1) * ph))
        for i in range(grid_size) for j in range(grid_size)
    ]

def process_image(image, problem, processor, model, tokenizer):
    image_tensor = process_images([image], processor, model.config)
    image_tensor = [img.to(dtype=torch.float16, device=DEVICE) for img in image_tensor]
    
    conv = copy.deepcopy(conv_templates[CONV_TEMPLATE])
    conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{problem}")
    conv.append_message(conv.roles[1], None)
    
    input_ids = tokenizer_image_token(
        conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

def evaluate(dataset, model, processor, tokenizer):
    rmse, mae = 0, 0
    
    for i in tqdm(range(len(dataset)), desc="Evaluating"):
        _, gt_map, _, _, text, image_pil = dataset[i]
        gt_count = gt_map.sum().item()
        problem = PROMPT_TEMPLATE.format(text)
        
        output = process_image(image_pil, problem, processor, model, tokenizer)
        pred = get_prediction(output)

        if pred > 100:
            patches = divide_image(image_pil, 2)
            local_preds = [
                get_prediction(process_image(p, problem, processor, model, tokenizer))
                for p in patches
            ]
            pred = (pred + sum(local_preds)) / 2

        rmse += (gt_count - pred) ** 2
        mae += abs(gt_count - pred)

    print(f"RMSE: {math.sqrt(rmse / len(dataset)):.2f}, MAE: {mae / len(dataset):.2f}")

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    print(f"Loading model from {args.pretrained_model}...")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.pretrained_model, None, MODEL_NAME, device_map="auto"
    )
    model.eval()

    test_dataset = FSC147(split="test")
    evaluate(test_dataset, model, image_processor, tokenizer)
