import os
import json
import torch
import pandas as pd
import re
import logging
import time
import numpy as np
from tqdm import tqdm
from PIL import Image
from decord import VideoReader, cpu

from transformers import (
    LlavaNextVideoForConditionalGeneration,
    LlavaNextVideoProcessor
)

# ==============================
# 只需要改这里
# ==============================

TARGET_JSON = "024"
GPU_ID = 0

NUM_FRAMES = 32

MODEL_PATH = "/share/cwt/models/LLaVA-NeXT-Video-7B-hf"
VIDEO_ROOT = "/share/hjx"
JSON_FOLDER = "/share/hjx/global_json_list"

# ==============================
# 日志
# ==============================

def setup_logger(json_number):

    log_file = f"llava_next_video_eval_{json_number}.log"

    logger = logging.getLogger("LLaVA")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ==============================
# Prompt
# ==============================

def build_prompt(question, options):

    question = re.sub(r"<.*?>", "", question).strip()

    clean_options = []

    for opt in options:

        opt = re.sub(r"([ABCD])\.", r"\1. ", opt)

        clean_options.append(opt)

    option_text = "\n".join(clean_options)

    prompt = f"""<video>

Answer the question based on the video.

Question: {question}

Options:
{option_text}

Reply with only the letter (A, B, C or D).
Answer:"""

    return prompt


# ==============================
# 答案解析
# ==============================

def extract_answer(text):

    text = text.upper()

    match = re.search(r"ANSWER:\s*([ABCD])", text)

    if match:
        return match.group(1)

    match = re.search(r"\b([ABCD])\b\s*$", text)

    if match:
        return match.group(1)

    return "ERROR"


# ==============================
# 视频读取
# ==============================

def fast_read_video(video_path, num_frames):

    vr = VideoReader(video_path, ctx=cpu(0))

    total_frames = len(vr)

    indices = np.linspace(
        0,
        total_frames - 1,
        num_frames,
        dtype=int
    )

    frames = vr.get_batch(indices).asnumpy()

    images = []

    for frame in frames:

        img = Image.fromarray(frame).convert("RGB")

        img = img.resize((336,336))

        images.append(img)

    del vr

    return images


# ==============================
# 推理
# ==============================

def inference(model, processor, video_paths, question, options, logger):

    n_videos = len(video_paths)

    frames_per_video = [NUM_FRAMES // n_videos] * n_videos

    for i in range(NUM_FRAMES % n_videos):

        frames_per_video[i] += 1

    all_frames = []

    for vp, nf in zip(video_paths, frames_per_video):

        frames = fast_read_video(vp, nf)

        all_frames.extend(frames)

    prompt = build_prompt(question, options)

    logger.info("===== PROMPT =====")
    logger.info(prompt)
    logger.info("==================")

    inputs = processor(
        text=[prompt],
        videos=all_frames,
        return_tensors="pt",
        padding=True
    )

    for k, v in inputs.items():

        if not isinstance(v, torch.Tensor):
            continue

        if v.dtype == torch.float32:
            inputs[k] = v.to(model.device, dtype=torch.float16)
        else:
            inputs[k] = v.to(model.device)

    with torch.no_grad():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            temperature=0
        )

    output_text = processor.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    logger.info(f"MODEL RAW OUTPUT: {output_text}")

    pred = extract_answer(output_text)

    logger.info(f"EXTRACTED ANSWER: {pred}")

    return pred, output_text


# ==============================
# 主函数
# ==============================

# ==============================
# 主函数
# ==============================

def main():

    logger = setup_logger(TARGET_JSON)

    device = f"cuda:{GPU_ID}"

    logger.info(f"Loading model on GPU {GPU_ID}")

    # ===== 改动：去掉 repo_type/revision，确保本地加载 =====
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map={"": device},
        low_cpu_mem_usage=True,
        local_files_only=True  # 本地加载
    )

    processor = LlavaNextVideoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True  # 本地加载
    )
    # =======================================================

    model.eval()

    json_path = os.path.join(
        JSON_FOLDER,
        TARGET_JSON,
        "QA.json"
    )

    with open(json_path) as f:
        data = json.load(f)

    logger.info(f"Dataset size: {len(data)}")

    results = []
    correct = 0

    for idx, sample in enumerate(tqdm(data)):

        start_time = time.time()

        video_paths = [
            os.path.join(VIDEO_ROOT, v)
            for v in sample["video_path"]
        ]

        question = sample["question"]
        options = sample["options"]
        gt = sample["correct_options"]

        logger.info(
            f"\n========== Sample {idx+1}/{len(data)} - {sample['video_id']} =========="
        )

        try:
            pred, raw = inference(
                model,
                processor,
                video_paths,
                question,
                options,
                logger
            )
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            pred = "ERROR"
            raw = str(e)

        correct_flag = pred == gt
        if correct_flag:
            correct += 1

        results.append({
            "video_id": sample["video_id"],
            "prediction": pred,
            "ground_truth": gt,
            "correct": correct_flag,
            "raw_output": raw
        })

        acc = correct / (idx + 1) * 100

        logger.info(
            f"Prediction:{pred} | GT:{gt} | Correct:{correct_flag}"
        )

        logger.info(
            f"Acc:{acc:.2f}% | Time:{time.time()-start_time:.2f}s"
        )

    df = pd.DataFrame(results)
    out_csv = f"llava_next_video_results_{TARGET_JSON}.csv"
    df.to_csv(out_csv, index=False)

    logger.info("Evaluation Finished")
    logger.info(
        f"Final Accuracy: {correct/len(results)*100:.2f}%"
    )

if __name__ == "__main__":
    main()