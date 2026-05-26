import torch
import json
import os
import re
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import decord
import gc

# ==================== 配置区 ====================
MODEL_PATH = "/share/hjx/models/qwen3VL-30B"
VIDEO_ROOT = "/share/hjx"

DATASET_LIST = ["015", "016", "017", "018", "019", "020", "021", "023", "024"]

BASE_JSON_DIR = "/share/hjx/global_json_list"

OUTPUT_DIR = "/share/hjx/project/qwen3vl-30b/results"
LOG_DIR = "/share/hjx/project/qwen3vl-30b/logs"

VIDEO_CONFIG = {"max_frames": 256}
# ===============================================


def extract_option(response):
    match = re.search(r'\b([A-D])\b', response)
    return match.group(1).upper() if match else "FAIL"


def setup_logger(dataset_id):
    os.makedirs(LOG_DIR, exist_ok=True)

    log_file = os.path.join(LOG_DIR, f"qwen3vl_{dataset_id}.log")

    logger = logging.getLogger(dataset_id)
    logger.setLevel(logging.INFO)

    # 避免重复 handler
    if logger.handlers:
        logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    return logger


def get_finished_keys(csv_path):
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        return set((row['video_id'].strip() + row['question'].strip()) for _, row in df.iterrows())
    except Exception:
        return set()


def get_video_duration(path):
    try:
        vr = decord.VideoReader(path, ctx=decord.cpu(0))
        return len(vr) / vr.get_avg_fps()
    except Exception:
        return 1.0


def run_dataset(dataset_id, model, processor, logger):

    json_path = os.path.join(BASE_JSON_DIR, dataset_id, "QA.json")

    output_csv = os.path.join(OUTPUT_DIR, f"qwen3vl_{dataset_id}.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fieldnames = [
        'video_id', 'question', 'task_type', 'difficulty',
        'ground_truth', 'prediction', 'raw_output', 'is_correct'
    ]

    logger.info(f"🚀 Start dataset {dataset_id}")
    logger.info(f"JSON: {json_path}")
    logger.info(f"CSV: {output_csv}")

    finished_keys = get_finished_keys(output_csv)

    if not os.path.exists(output_csv):
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in tqdm(data, desc=f"Dataset {dataset_id}"):

        v_id = str(item.get('video_id', '')).strip()
        q_text = str(item.get('question', '')).strip()

        if (v_id + q_text) in finished_keys:
            continue

        v_paths = item.get('video_path', [])
        if isinstance(v_paths, str):
            v_paths = [v_paths]

        full_paths = [
            os.path.join(VIDEO_ROOT, v if v.endswith('.mp4') else v + ".mp4")
            for v in v_paths
        ]

        valid_paths = [p for p in full_paths if os.path.exists(p)]

        try:
            if not valid_paths:
                response, pred_option = "Missing Videos", "MISSING"

            else:
                v_durations = [get_video_duration(p) for p in valid_paths]
                total_dur = sum(v_durations)

                content = []

                for path, dur in zip(valid_paths, v_durations):
                    v_frames = max(4, int((dur / total_dur) * VIDEO_CONFIG["max_frames"]))

                    content.append({
                        "type": "video",
                        "video": path,
                        "max_frames": v_frames,
                        "fps": 1.0
                    })

                clean_q = re.sub(r'<.*?>', '', q_text).strip()

                prompt_text = (
                    f"{clean_q}\nOptions:\n"
                    + "\n".join(item['options'])
                    + "\nAnswer with A, B, C, or D only."
                )

                content.append({"type": "text", "text": prompt_text})

                messages = [{"role": "user", "content": content}]

                text_prompt = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                image_inputs, video_inputs = process_vision_info(messages)

                inputs = processor(
                    text=[text_prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )

                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=32,
                        do_sample=False
                    )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                ]

                response = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0].strip()

                pred_option = extract_option(response)

                del inputs, generated_ids, generated_ids_trimmed
                torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            response, pred_option = f"Error: {str(e)}", "ERROR"
            logger.error(f"{dataset_id} | {v_id} | {e}")
            torch.cuda.empty_cache()

        res_entry = {
            "video_id": v_id,
            "question": q_text,
            "task_type": item.get('task_type', 'unknown'),
            "difficulty": item.get('difficulty', 'unknown'),
            "ground_truth": item.get('correct_options'),
            "prediction": pred_option,
            "raw_output": response.replace('\n', ' '),
            "is_correct": (pred_option == item.get('correct_options'))
        }

        with open(output_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(res_entry)
            f.flush()
            os.fsync(f.fileno())

    logger.info(f"✅ Finished dataset {dataset_id}")


def main():

    print("📦 Loading model once...")

    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    ).eval()

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    print("✅ Model loaded")

    for dataset_id in DATASET_LIST:

        logger = setup_logger(dataset_id)

        try:
            run_dataset(dataset_id, model, processor, logger)
        except Exception as e:
            logger.error(f"🔥 Fatal error in dataset {dataset_id}: {e}")


if __name__ == "__main__":
    main()