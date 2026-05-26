import torch
import json
import os
import re
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import decord
import gc

# ==================== 配置区 ====================
MODEL_PATH = "/share/cwt/models/Qwen3-VL-8B-Instruct"
VIDEO_ROOT = "/share/hjx"
JSON_FILE = "/share/hjx/global_json_list/021/QA.json" 
OUTPUT_CSV = "/data1/student/cwt/ego_month/result_fix/Qwen3-VL-8B/021/qwen3vl_transformers_results.csv"

# 采样配置：针对 48G 显存推荐 256 帧
VIDEO_CONFIG = {
    "max_frames": 256, 
}
# ===============================================

def extract_option(response):
    """从回复中提取 A/B/C/D 选项"""
    match = re.search(r'\b([A-D])\b', response)
    return match.group(1).upper() if match else "FAIL"

def get_finished_keys(csv_path):
    """读取已完成任务，实现断点续测"""
    if not os.path.exists(csv_path): return set()
    try:
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        return set((row['video_id'].strip() + row['question'].strip()) for _, row in df.iterrows())
    except Exception: return set()

def get_video_duration(path):
    """获取视频时长用于比例采样"""
    try:
        vr = decord.VideoReader(path, ctx=decord.cpu(0))
        return len(vr) / vr.get_avg_fps()
    except Exception: return 1.0

def main():
    # 1. 初始化路径与 CSV 表头
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    fieldnames = ['video_id', 'question', 'task_type', 'difficulty', 'ground_truth', 'prediction', 'raw_output', 'is_correct']
    
    finished_keys = get_finished_keys(OUTPUT_CSV)
    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            f.flush()
            os.fsync(f.fileno())

    # 2. 加载模型与处理器 (开启 Flash Attention 2)
    print(f"📦 正在加载 Qwen3-VL-8B (Transformers 版): {MODEL_PATH}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", # 48G 显存强烈建议开启
        device_map="auto",
        trust_remote_code=True
    ).eval()
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 3. 读取测评任务
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 4. 测评主循环
    for item in tqdm(data):
        v_id, q_text = str(item.get('video_id', '')).strip(), str(item.get('question', '')).strip()
        if (v_id + q_text) in finished_keys: continue

        v_paths = item.get('video_path', [])
        if isinstance(v_paths, str): v_paths = [v_paths]
        full_paths = [os.path.join(VIDEO_ROOT, v if v.endswith('.mp4') else v + ".mp4") for v in v_paths]
        valid_paths = [p for p in full_paths if os.path.exists(p)]

        try:
            if not valid_paths:
                response, pred_option = "Missing Videos", "MISSING"
            else:
                # 🚀 时长比例采样逻辑
                v_durations = [get_video_duration(p) for p in valid_paths]
                total_dur = sum(v_durations)
                
                content = []
                for path, dur in zip(valid_paths, v_durations):
                    v_frames = max(4, int((dur / total_dur) * VIDEO_CONFIG["max_frames"]))
                    content.append({
                        "type": "video",
                        "video": path,
                        "max_frames": v_frames,
                        "fps": 1.0, # 配合抽帧
                    })
                
                clean_q = re.sub(r'<.*?>', '', q_text).strip()
                prompt_text = f"{clean_q}\nOptions:\n" + "\n".join(item['options']) + \
                              "\nAnswer with the option letter (A, B, C, or D) only."
                content.append({"type": "text", "text": prompt_text})

                messages = [{"role": "user", "content": content}]

                # 5. 推理预处理
                text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                
                inputs = processor(
                    text=[text_prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)

                # 6. 生成回复
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    response = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0].strip()
                
                pred_option = extract_option(response)

                # 显存释放防止累积
                del inputs, generated_ids, generated_ids_trimmed
                torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            response, pred_option = f"Error: {str(e)}", "ERROR"
            torch.cuda.empty_cache()

        # 7. 结果持久化 (字段对齐)
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
        
        with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(res_entry)
            f.flush()
            os.fsync(f.fileno())

    print(f"✅ Qwen3-VL 测评完成，结果保存至: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
