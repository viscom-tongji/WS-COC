import os
import json
import torch
from tqdm import tqdm
import numpy as np
from llava.constants import DEFAULT_IMAGE_TOKEN
import random
import math



from FSC import FSC147

device = "cuda" if torch.cuda.is_available() else "cpu"

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

train_datas = FSC147(split='train', resize_val=False)
im_dir = train_datas.im_dir
json_list = []
item_id = 0

# Save name
json_name = "fsc_train_convs_merged.json"

class_dict = {}
max_gt = 0

def convservation(conv_id, im_id, problem, answer):
    if isinstance(problem, list):
        conversations = []
        for p, a in zip(problem, answer):
            conversations.append({"from": "human", "value": DEFAULT_IMAGE_TOKEN + p})
            conversations.append({"from": "gpt", "value": a})
        conversation = {
            "id": str(conv_id),
            "image": im_id,
            "conversations": conversations
        }
    else:
        conversation = {
            "id": str(conv_id),
            "image": im_id,
            "conversations": [
                {
                    "from": "human",
                    "value": problem
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
        }
    return conversation


# Part 1: Process Single Image Samples 
for i in tqdm(range(int(len(train_datas))), desc="Single Image Sample Processing"):
    ele = train_datas[i]
    img, _, rects, dots, image_pil, text, im_id = ele
    gt_count = len(dots)
    
    max_gt = max(max_gt, gt_count)
    
    if text not in class_dict:
        class_dict[text] = []
    class_dict[text].append({"im_id": im_id, "GT": gt_count, "rects": rects})
    
    # ------------------ type7: basic count and string match ------------------
    problem_type7 = (
        f"\nHint: Please answer the question and provide the final answer at the end. \nQuestion: "
        f"Based on the image, generate a title that estimates the counts of {text}, "
        f"formatted as 'a photo of [XX] [objects]' (for example: a photo of 15 apples)."
    )
    answer_type7 = f"a photo of {gt_count} {text}"
    conv_type7 = convservation(item_id, im_id, [problem_type7], [answer_type7])
    json_list.append(conv_type7)
    item_id += 1
    
    # ------------------ type16: iterative bounding ------------------
    # Only process samples with counts > 100
    if gt_count > 100:
        target_precision = 0.2  # 20% precision
        lower_bound = 1
        upper_bound = 2000  # set a reasonable upper bound
        
        conversations = []
        round_num = 1
        
        while (upper_bound - lower_bound) / gt_count > target_precision:
            mid_point = (lower_bound + upper_bound) // 2
            
            if round_num == 1:
                problem = (
                    f"{DEFAULT_IMAGE_TOKEN}\nHint: Please answer the question step by step. "
                    f"\nQuestion: Looking at this image, do you think there are more than {mid_point} {text} in total? "
                    f"Please answer 'Yes' if you think there are more than {mid_point}, or 'No' if you think there are {mid_point} or fewer."
                )
            else:
                problem = (
                    f"\nQuestion: Based on your previous observations, do you think there are more than {mid_point} {text} in the image? "
                    f"Please answer 'Yes' if you think there are more than {mid_point}, or 'No' if you think there are {mid_point} or fewer."
                )
            
            if gt_count > mid_point:
                answer = "Yes"
                lower_bound = mid_point + 1
            else:
                answer = "No"
                upper_bound = mid_point
            
            conversations.append({"from": "human", "value": problem})
            conversations.append({"from": "gpt", "value": answer})
            round_num += 1
            
            # Prevent infinite loop
            if round_num > 10:
                break
                
        # Final round: provide specific estimate within the determined range
        final_range_mid = (lower_bound + upper_bound) // 2
        final_problem = (
            f"\nQuestion: Based on our discussion, the count should be between {lower_bound} and {upper_bound}. "
            f"Please provide your final estimate of the number of {text} in the image, "
            f"formatted as 'This photo has [XX] {text}' (for example:This photo has 150 {text})."
        )
        final_answer = f"This photo has {gt_count} {text}."
        
        conversations.append({"from": "human", "value": final_problem})
        conversations.append({"from": "gpt", "value": final_answer})
        
        conversation_type16 = {
            "id": str(item_id),
            "image": im_id,
            "conversations": conversations
        }
        json_list.append(conversation_type16)
        item_id += 1


# Part 2: Process Multiple Image Samples (rank_type2)
# Rank images by count of objects
k_type2 = 4  # Number of images to rank
extra_counting_bound = 100
extra_number_limited = 40
total_countsss = 0

for key, value in class_dict.items():
    temp_list = sorted(value, key=lambda x: x["GT"])
    
    count100 = sum(1 for ele in temp_list if ele["GT"] >= extra_counting_bound)
    if count100 >= 3:
        cur_comb = min(math.comb(count100, 3), 20)
        total_countsss += cur_comb
        print(f"Processing class: {key} with more than 100 samples: {count100} and {cur_comb} combinations")
        
print("Total valid combinations for rank_type2 info logging:", total_countsss)

for key, value in class_dict.items():
    temp_list = sorted(value, key=lambda x: x["GT"])
    sample_num = len(temp_list)
    
    if sample_num < k_type2 or temp_list[-1]["GT"] < extra_counting_bound:
        continue
        
    weights = [2 ** (k_type2 - i) for i in range(1, k_type2 + 1)]
    total_weight = sum(weights)
    total_elements = len(temp_list)
    target_sizes = [total_elements * w / total_weight for w in weights]
    
    groups = []
    start_idx = 0
    for i, target in enumerate(target_sizes):
        if i == k_type2 - 1:
            group = temp_list[start_idx:]
        else:
            end_idx = start_idx + int(round(target))
            # Ensure we don't break apart dictionary elements with the same GT value
            while end_idx < total_elements and temp_list[end_idx]["GT"] == temp_list[end_idx - 1]["GT"]:
                end_idx += 1
            group = temp_list[start_idx:end_idx]
        if len(group) == 0:
            continue
        groups.append(group)
        start_idx = end_idx
        
    group_counts = [len(g) for g in groups]
    potential_number_combins = min(math.prod(group_counts), extra_number_limited)
    
    selected_set = set()
    while len(selected_set) < potential_number_combins:
        sample = tuple(random.randrange(count) for count in group_counts)
        selected_set.add(sample)
        
    selected_int_arrays = [list(arr) for arr in selected_set]
    
    for id_arr in selected_int_arrays:
        id_list_extra = [group[idx]["im_id"] for idx, group in zip(id_arr, groups)]
        
        image_placeholders = " ".join([f"{DEFAULT_IMAGE_TOKEN} This is the {i+1}{'st' if i==0 else 'nd' if i==1 else 'rd' if i==2 else 'th'} image." for i in range(len(id_list_extra))])
        
        problem_extra_type2 = (
            f"\nHint: You will be provided with multiple images. Carefully analyze the images and determine the order of the {key} counts in each image. "
            f"{image_placeholders}"
            f"\nQuestion: Based on the provided images, rank the images in ascending order of the counts of {key}. "
            "Format your answer as a sequence of image indices (e.g., Image 1 < Image 3 < Image 2 < Image 4). "
        )
        
        shuffled_ids_extra = id_list_extra.copy()
        random.shuffle(shuffled_ids_extra)
        
        ranked_shuffled_ids_extra = []
        for image_id_val in id_list_extra:
            index = shuffled_ids_extra.index(image_id_val) if image_id_val in shuffled_ids_extra else None
            ranked_shuffled_ids_extra.append(f"Image {index + 1}")
            
        answer_extra = " < ".join(ranked_shuffled_ids_extra)
        
        conversation_muti = {
            "id": str(item_id),
            "image": shuffled_ids_extra,
            "conversations": [
                {
                    "from": "human",
                    "value": problem_extra_type2
                },
                {
                    "from": "gpt",
                    "value": answer_extra
                }
            ]
        }
        item_id += 1
        json_list.append(conversation_muti)

# Save the unified JSON list to a file
with open(json_name, 'w', encoding='utf-8') as f:
    json.dump(json_list, f, indent=2, ensure_ascii=False)
print(f"JSON list saved to {json_name} with {len(json_list)} items.")
