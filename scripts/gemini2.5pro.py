import os
import json
import csv
import re
import time
from tqdm import tqdm
from datetime import datetime

import vertexai
from vertexai.generative_models import GenerativeModel, Part

# =========================
# 🔧 CONFIG
# =========================
DATA_PATH = "/share/hjx/global_json_list/113/sub.json"

DATASET_NAME = os.path.basename(os.path.dirname(DATA_PATH))

CONFIG = {
    "data_path": DATA_PATH,

    "output_path": f"gemini_{DATASET_NAME}.csv",
    "log_path": f"gemini_eval_{DATASET_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",

    "model_name": "gemini-2.5-pro",
}

# =========================
# 🧠 init
# =========================
vertexai.init(
    project="ego-video-qa",
    location="us-central1"
)

model = GenerativeModel(CONFIG["model_name"])


# =========================
# 🧠 prompt
# =========================
def build_prompt(question, options):
    q = re.sub(r'^<.*?>', '', question).strip()
    q = re.sub(r'<.*?>', '', q)

    return f"""
Answer a multiple-choice question based on the video.

Question: {q}

Options:
{chr(10).join(options)}

Only output your choice: A, B, C, or D.
"""


# =========================
# 🔍 extract
# =========================
def extract_answer(text):
    match = re.search(r'\b([A-D])\b', text.upper())
    return match.group(1) if match else "UNKNOWN"


# =========================
# 🚀 main
# =========================
def main():

    with open(CONFIG["data_path"], "r") as f:
        data = json.load(f)

    results = []
    correct = 0

    for idx, item in tqdm(enumerate(data)):

        try:
            # =========================
            # ✅ 改动1：直接读取 gs://
            # =========================
            gcs_uri = item["video_path"]

            # 兼容 list 格式
            if isinstance(gcs_uri, list):
                gcs_uri = gcs_uri[0]

            video_part = Part.from_uri(
                gcs_uri,
                mime_type="video/mp4"
            )

            # =========================
            # 🧠 prompt
            # =========================
            prompt = build_prompt(
                item["question"],
                item["options"]
            )

            # =========================
            # 🚀 inference
            # =========================
            response = model.generate_content([video_part, prompt])

            output_text = response.text
            pred = extract_answer(output_text)
            gt = item["correct_options"]

            if pred == gt:
                correct += 1

        except Exception as e:
            pred = "ERROR"
            gt = item.get("correct_options", "UNKNOWN")
            output_text = str(e)

        results.append({
            "question": item["question"],
            "pred": pred,
            "gt": gt,
            "correct": int(pred == gt),
            "raw": output_text,
            "task_type": item["task_type"]
        })

        print(f"[{idx}] pred={pred}, gt={gt}")

        time.sleep(30)

    # =========================
    # 💾 save
    # =========================
    with open(CONFIG["output_path"], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    acc = correct / len(results)
    print(f"\n✅ Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()