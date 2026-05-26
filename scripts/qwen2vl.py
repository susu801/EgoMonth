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
# 🔧 CONFIG
# =========================
DATA_PATH = "/share/hjx/global_json_list/021/QA.json"

# 自动提取样本集名，例如 024
DATASET_NAME = os.path.basename(os.path.dirname(DATA_PATH))

CONFIG = {
    "model_path": "/share/hjx/models/qwen2VL",
    "data_path": DATA_PATH,
    "video_root": "/share/hjx",

    "output_path": f"{DATASET_NAME}.csv",
    "log_path": f"qwen2.5_eval_{DATASET_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",

    "num_frames": 256,
    "max_new_tokens": 10,

    "device": "cuda:0"
}


# =========================
# 📝 Logger
# =========================
def setup_logger(log_path):
    logger = logging.getLogger("eval")
    logger.setLevel(logging.INFO)

    # StreamHandler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    # FileHandler
    fh = logging.FileHandler(log_path, mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


# =========================
# 🎥 手动抽帧
# =========================
def load_multi_videos(video_paths, total_frames, size=224):
    frames_all = []

    lengths = []
    vrs = []

    for vp in video_paths:
        vr = VideoReader(vp, ctx=cpu(0))
        vrs.append(vr)
        lengths.append(len(vr))

    total_len = sum(lengths)

    frames_per_video = [
        max(1, int(total_frames * l / total_len))
        for l in lengths
    ]

    # 修正
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
    # 去掉开头 <xxx.mp4> 标签
    q = re.sub(r'^<.*?>', '', question).strip()
    
    # 去掉其他 HTML 标签
    q = re.sub(r'<.*?>', '', q)
    
    opts = "\n".join(options)

    return f"""Answer a multiple-choice question based on the video.

Question: {q}

Options:
{opts}

IMPORTANT:
- Only output ONE letter: A, B, C, or D
- Do NOT output any explanation or extra words

Answer:"""


# =========================
# 🔍 提取答案
# =========================
def extract_answer(text):
    match = re.search(r'\b([A-D])\b', text.upper())
    return match.group(1) if match else "UNKNOWN"


# =========================
# 🚀 主函数
# =========================
def main():
    cfg = CONFIG
    logger = setup_logger(cfg["log_path"])

    device = torch.device(cfg["device"])
    torch.cuda.set_device(device)

    # =========================
    # 🔧 加载模型
    # =========================
    logger.info("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg["model_path"],
        torch_dtype=torch.float16,
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(cfg["model_path"])
    logger.info("Model loaded!")

    # =========================
    # 📂 数据
    # =========================
    with open(cfg["data_path"], "r") as f:
        data = json.load(f)

    results = []
    correct = 0

    # =========================
    # 🔁 推理
    # =========================
    for idx, item in enumerate(tqdm(data)):
        try:
            video_paths = [
                os.path.join(cfg["video_root"], v)
                for v in item["video_path"]
            ]

            # ---------- 抽帧 ----------
            frames = load_multi_videos(
                video_paths,
                cfg["num_frames"]
            )

            # ---------- Prompt ----------
            prompt = build_prompt(item["question"], item["options"])

            # ---------- 构造 Qwen 输入 ----------
            content = []

            for img in frames:
                content.append({
                    "type": "image",
                    "image": img
                })

            content.append({
                "type": "text",
                "text": prompt
            })

            messages = [{
                "role": "user",
                "content": content
            }]

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
            ).to(device)

            # ---------- 推理 ----------
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=cfg["max_new_tokens"]
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

        # ---------- 写日志 ----------
        logger.info(f"[{idx+1}/{len(data)}] Question: {item['question']}")
        logger.info(f"Video paths: {video_paths}")
        logger.info(f"Prediction: {pred}, Ground Truth: {gt}")
        logger.info(f"Raw output: {output_text}\n")

        results.append({
            "question": item["question"],
            "pred": pred,
            "gt": gt,
            "correct": int(pred == gt),
            "raw": output_text
        })

    # =========================
    # 💾 保存
    # =========================
    with open(cfg["output_path"], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    acc = correct / len(results)
    logger.info(f"\n✅ Accuracy: {acc:.4f}")
    logger.info(f"Log saved at: {cfg['log_path']}")
    logger.info(f"CSV results saved at: {cfg['output_path']}")


if __name__ == "__main__":
    main()