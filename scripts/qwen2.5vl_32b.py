import os
import json
import csv
import logging
from datetime import datetime
from tqdm import tqdm
import re

import torch
import numpy as np
from PIL import Image
from decord import VideoReader, cpu

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# =========================
# 🔧 DATASET LIST（核心新增）
# =========================
DATASET_LIST = ["013", "023"]   # ⭐ 在这里改


# =========================
# 🔧 CONFIG（固定部分）
# =========================
CONFIG = {
    "model_path": "/share/hjx/models/qwen2.5VL_32B",
    "video_root": "/share/hjx",

    "num_frames": 256,
    "max_new_tokens": 10,

    "dtype": torch.bfloat16,
}


# =========================
# 📝 Logger（每个数据集独立）
# =========================
def setup_logger(log_path):
    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers = []

    ch = logging.StreamHandler()
    fh = logging.FileHandler(log_path, mode='w')

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


# =========================
# 🎥 抽帧（不变）
# =========================
def load_multi_videos(video_paths, total_frames, size=224):
    frames_all = []
    lengths, vrs = [], []

    for vp in video_paths:
        vr = VideoReader(vp, ctx=cpu(0))
        vrs.append(vr)
        lengths.append(len(vr))

    total_len = sum(lengths)

    frames_per_video = [
        max(1, int(total_frames * l / total_len))
        for l in lengths
    ]

    while sum(frames_per_video) < total_frames:
        frames_per_video[np.argmax(lengths)] += 1
    while sum(frames_per_video) > total_frames:
        frames_per_video[np.argmax(frames_per_video)] -= 1

    for vr, nf in zip(vrs, frames_per_video):
        total = len(vr)

        if total <= nf:
            indices = list(range(total))
        else:
            step = total / nf
            indices = [int(i * step + step / 2) for i in range(nf)]

        batch = vr.get_batch(indices).asnumpy()

        for frame in batch:
            img = Image.fromarray(frame).convert("RGB")
            img = img.resize((size, size))
            frames_all.append(img)

    return frames_all


# =========================
# 🧠 Prompt
# =========================
def build_prompt(question, options):
    q = re.sub(r'^<.*?>', '', question).strip()
    q = re.sub(r'<.*?>', '', q)

    opts = "\n".join(options)

    return f"""Answer a multiple-choice question based on the video.

Question: {q}

Options:
{opts}

IMPORTANT:
- Only output ONE letter: A, B, C, or D
- Do NOT output any explanation

Answer:"""


def extract_answer(text):
    match = re.search(r'\b([A-D])\b', text.upper())
    return match.group(1) if match else "UNKNOWN"


# =========================
# 🚀 单个数据集处理函数（核心拆分）
# =========================
def run_single_dataset(dataset_name, model, processor):
    data_path = f"/share/hjx/global_json_list/{dataset_name}/QA.json"

    output_path = f"{dataset_name}.csv"
    log_path = f"qwen2.5_32b_eval_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = setup_logger(log_path)

    logger.info(f"🚀 Start dataset: {dataset_name}")
    logger.info(f"📂 Data path: {data_path}")

    with open(data_path, "r") as f:
        data = json.load(f)

    results = []
    correct = 0

    for idx, item in enumerate(tqdm(data)):

        try:
            video_paths = [
                os.path.join(CONFIG["video_root"], v)
                for v in item["video_path"]
            ]

            logger.info("=" * 60)
            logger.info(f"[{idx+1}/{len(data)}] Processing")

            # ---------- 抽帧 ----------
            frames = load_multi_videos(
                video_paths,
                CONFIG["num_frames"]
            )

            logger.info(f"🎬 Frames: {len(frames)}")

            # ---------- Prompt ----------
            prompt = build_prompt(item["question"], item["options"])

            content = [{"type": "image", "image": img} for img in frames]
            content.append({"type": "text", "text": prompt})

            messages = [{"role": "user", "content": content}]

            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )

            # ---------- 推理 ----------
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=CONFIG["max_new_tokens"]
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True
            )[0]

            pred = extract_answer(output_text)
            gt = item["correct_options"]

            if pred == gt:
                correct += 1

        except Exception as e:
            pred = "ERROR"
            gt = item.get("correct_options", "UNKNOWN")
            output_text = str(e)

        logger.info(f"Pred: {pred} | GT: {gt}")
        logger.info(f"Raw: {output_text}\n")

        results.append({
            "question": item["question"],
            "pred": pred,
            "gt": gt,
            "correct": int(pred == gt),
            "raw": output_text
        })

    # ---------- 保存 ----------
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    acc = correct / len(results)

    logger.info(f"\n🎯 Accuracy: {acc:.4f}")
    logger.info(f"📁 Log: {log_path}")
    logger.info(f"📊 CSV: {output_path}")


# =========================
# 🚀 主函数
# =========================
def main():

    print("🚀 Loading 32B model (only once)...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CONFIG["model_path"],
        torch_dtype=CONFIG["dtype"],
        device_map="auto",
        low_cpu_mem_usage=True
    )

    processor = AutoProcessor.from_pretrained(CONFIG["model_path"])

    print("✅ Model loaded once!")

    # =========================
    # 🔁 循环多个数据集
    # =========================
    for dataset_name in DATASET_LIST:
        run_single_dataset(dataset_name, model, processor)


if __name__ == "__main__":
    main()