import argparse
import json
import csv
import re
import os
from datetime import datetime
from tqdm import tqdm

import torch

from stllm.common.config import Config
from stllm.common.registry import registry
from stllm.conversation.conversation import Chat, CONV_instructblip_Vicuna0

# imports modules for registration
from stllm.datasets.builders import *
from stllm.models import *
from stllm.processors import *
from stllm.runners import *
from stllm.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="STLLM Video QA Evaluation")

    parser.add_argument(
        "--cfg-path",
        default="config/instructblipbase_stllm_conversation.yaml",
        help="path to configuration file."
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="specify the gpu to load the model."
    )
    parser.add_argument(
        "--ckpt-path",
        required=True,
        help="path to STLLM checkpoint."
    )
    parser.add_argument(
        "--qa-json",
        required=True,
        help="path to QA json file."
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="path to output csv file."
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=64,
        help="number of video frames used by STLLM."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=300
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="override config options."
    )

    return parser.parse_args()


def load_qa_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "data" in data:
            data = data["data"]
        elif "questions" in data:
            data = data["questions"]
        elif "qa" in data:
            data = data["qa"]
        else:
            data = list(data.values())

    if not isinstance(data, list):
        raise ValueError("QA JSON should be a list or a dict containing data/questions/qa.")

    return data


def get_field(item, candidates, default=""):
    for key in candidates:
        if key in item and item[key] is not None:
            return item[key]
    return default


def normalize_options(options):
    """
    支持以下几种格式：
    1. ["xxx", "yyy", "zzz", "www"]
    2. {"A": "xxx", "B": "yyy", ...}
    3. 已经是字符串
    """
    if isinstance(options, list):
        labels = ["A", "B", "C", "D", "E", "F"]
        return "\n".join([f"{labels[i]}. {opt}" for i, opt in enumerate(options)])

    if isinstance(options, dict):
        lines = []
        for k in sorted(options.keys()):
            lines.append(f"{k}. {options[k]}")
        return "\n".join(lines)

    if isinstance(options, str):
        return options

    return ""


def build_prompt(question, options_text):
    return f"""Please answer the following multiple-choice question based on the video.

Question:
{question}

Options:
{options_text}

Please output the answer option only, such as A, B, C, or D.
"""


def extract_answer(text):
    """
    从模型输出中解析 A/B/C/D。
    """
    if text is None:
        return ""

    text = text.strip()

    patterns = [
        r"answer\s*[:：]\s*([A-F])",
        r"option\s*[:：]?\s*([A-F])",
        r"\(([A-F])\)",
        r"\b([A-F])\b",
    ]

    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    return ""


def normalize_gt_answer(ans):
    if ans is None:
        return ""

    if isinstance(ans, int):
        labels = ["A", "B", "C", "D", "E", "F"]
        if 0 <= ans < len(labels):
            return labels[ans]

    ans = str(ans).strip()

    m = re.search(r"\b([A-F])\b", ans, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return ans


def init_model(args):
    print("Initializing STLLM...")

    cfg = Config(args)

    ckpt_path = args.ckpt_path
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_config.ckpt = ckpt_path
    model_config.llama_model = ckpt_path

    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(f"cuda:{args.gpu_id}")
    model.to(torch.float16)

    chat = Chat(model, device=f"cuda:{args.gpu_id}")

    print("Initialization Finished.")
    return chat


def run_one_sample(chat, video_path, prompt, num_frames, max_new_tokens):
    chat_state = CONV_instructblip_Vicuna0.copy()
    img_list = []

    chat.upload_video(
        video_path,
        chat_state,
        img_list,
        num_frames,
        text=prompt
    )

    chat.ask("###Human: " + prompt + " ###Assistant: ", chat_state)

    llm_message = chat.answer(
        conv=chat_state,
        img_list=img_list,
        num_beams=5,
        do_sample=False,
        temperature=1,
        max_new_tokens=max_new_tokens,
        max_length=2000
    )[0]

    return llm_message


def main():
    args = parse_args()

    qa_data = load_qa_data(args.qa_json)

    if args.output_csv is None:
        dataset_name = os.path.splitext(os.path.basename(args.qa_json))[0]
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_csv = f"stllm_eval_{dataset_name}_{time_str}.csv"

    chat = init_model(args)

    results = []
    correct_count = 0
    total_with_gt = 0

    for idx, item in enumerate(tqdm(qa_data, desc="Evaluating")):
        video_path = get_field(
            item,
            ["video_path", "video", "video_file", "video_name", "path"]
        )

        question = get_field(
            item,
            ["question", "query", "Question", "qa_question"]
        )

        options = get_field(
            item,
            ["options", "choices", "candidate_answers", "answers"]
        )

        gt_answer = get_field(
            item,
            ["answer", "gt_answer", "correct_answer", "label", "ground_truth"],
            default=""
        )

        options_text = normalize_options(options)
        prompt = build_prompt(question, options_text)

        if not video_path:
            print(f"[Warning] Sample {idx} has no video path. Skipped.")
            continue

        if not os.path.exists(video_path):
            print(f"[Warning] Video not found: {video_path}")

        try:
            raw_response = run_one_sample(
                chat=chat,
                video_path=video_path,
                prompt=prompt,
                num_frames=args.num_frames,
                max_new_tokens=args.max_new_tokens
            )

            pred_answer = extract_answer(raw_response)
            gt_answer_norm = normalize_gt_answer(gt_answer)

            is_correct = ""
            if gt_answer_norm:
                total_with_gt += 1
                is_correct = pred_answer == gt_answer_norm
                if is_correct:
                    correct_count += 1

            results.append({
                "index": idx,
                "video_path": video_path,
                "question": question,
                "options": options_text,
                "gt_answer": gt_answer_norm,
                "model_raw_response": raw_response,
                "pred_answer": pred_answer,
                "is_correct": is_correct
            })

        except Exception as e:
            print(f"[Error] Sample {idx} failed: {e}")

            results.append({
                "index": idx,
                "video_path": video_path,
                "question": question,
                "options": options_text,
                "gt_answer": normalize_gt_answer(gt_answer),
                "model_raw_response": "",
                "pred_answer": "",
                "is_correct": "",
                "error": str(e)
            })

    fieldnames = [
        "index",
        "video_path",
        "question",
        "options",
        "gt_answer",
        "model_raw_response",
        "pred_answer",
        "is_correct"
    ]

    with open(args.output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print("=" * 60)
    print(f"Saved results to: {args.output_csv}")
    print(f"Total samples: {len(results)}")

    if total_with_gt > 0:
        acc = correct_count / total_with_gt
        print(f"Samples with GT: {total_with_gt}")
        print(f"Correct: {correct_count}")
        print(f"Accuracy: {acc:.4f}")
    else:
        print("No ground-truth answer found, only saved model predictions.")

    print("=" * 60)


if __name__ == "__main__":
    main()