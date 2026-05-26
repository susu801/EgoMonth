import torch
import json
import os
import re
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import decord
from scipy.spatial import cKDTree
import gc

# ==================== 配置区 ====================
MODEL_PATH = '/share/cwt/models/MiniCPM-V-4_5'
VIDEO_ROOT = "/share/hjx"
JSON_FILE = "/share/hjx/global_json_list/021/QA.json" 
OUTPUT_CSV = "/data1/student/cwt/ego_month/result_fix/MiniCPM-V-4_5/021/minicpmv45_videomme_results.csv"

# 采样配置：设定总帧数上限
VIDEO_CONFIG = {
    "max_frames": 256, 
    "packing_nums": 3,   # MiniCPM-V 4.5 特有的 3D-Packing 参数
    "time_scale": 0.1
} 
# ===============================================

def load_model_local():
    print(f"📦 正在加载 MiniCPM-V 4.5 (SDPA 模式): {MODEL_PATH}...")
    model = AutoModel.from_pretrained(
        MODEL_PATH, 
        trust_remote_code=True, 
        attn_implementation='sdpa', # 兼容性更好的模式
        torch_dtype=torch.bfloat16
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return model, tokenizer

def map_to_nearest_scale(values, scale):
    tree = cKDTree(np.asarray(scale)[:, None])
    _, indices = tree.query(np.asarray(values)[:, None])
    return np.asarray(scale)[indices]

def group_array(arr, size):
    return [arr[i:i+size] for i in range(0, len(arr), size)]

def get_video_info(path):
    """获取视频时长和总帧数"""
    try:
        vr = decord.VideoReader(path, ctx=decord.cpu(0))
        duration = len(vr) / vr.get_avg_fps()
        return vr, duration
    except Exception:
        return None, 0.0

def extract_option(response):
    match = re.search(r'\b([A-D])\b', response)
    return match.group(1).upper() if match else "FAIL"

def get_finished_keys(csv_path):
    if not os.path.exists(csv_path): return set()
    try:
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        return set((row['video_id'].strip() + row['question'].strip()) for _, row in df.iterrows())
    except Exception: return set()

def main():
    # 1. 初始化路径与文件
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    fieldnames = ['video_id', 'question', 'task_type', 'difficulty', 'ground_truth', 'prediction', 'raw_output', 'is_correct']
    
    finished_keys = get_finished_keys(OUTPUT_CSV)
    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            f.flush()
            os.fsync(f.fileno())

    # 2. 加载模型
    model, tokenizer = load_model_local()
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 3. 测评循环
    for item in tqdm(data):
        v_id, q_text = str(item.get('video_id', '')).strip(), str(item.get('question', '')).strip()
        if (v_id + q_text) in finished_keys: continue

        v_paths = item.get('video_path', [])
        if isinstance(v_paths, str): v_paths = [v_paths]
        full_paths = [os.path.join(VIDEO_ROOT, v if v.endswith('.mp4') else v + ".mp4") for v in v_paths]
        
        try:
            # 🚀 比例采样逻辑
            valid_infos = []
            for p in full_paths:
                if os.path.exists(p):
                    vr, dur = get_video_info(p)
                    if vr: valid_infos.append((p, vr, dur))
            
            if not valid_infos:
                response, pred_option = "Missing Files", "MISSING"
            else:
                total_dur = sum(info[2] for info in valid_infos)
                all_frames = []
                all_temporal_ids = []

                for path, vr, dur in valid_infos:
                    # 计算当前视频应分配的帧数
                    v_target_frames = max(4, int((dur / total_dur) * VIDEO_CONFIG["max_frames"]))
                    
                    # 均匀采样索引
                    total_f = len(vr)
                    fps = vr.get_avg_fps()
                    indices = np.linspace(0, total_f - 1, v_target_frames, dtype=int)
                    
                    # 提取并转换图像
                    frames_raw = vr.get_batch(indices).asnumpy()
                    all_frames.extend([Image.fromarray(f).convert('RGB') for f in frames_raw])
                    
                    # 计算 MiniCPM 特有的时间戳 ID
                    frame_ts = indices / fps
                    scale = np.arange(0, dur + VIDEO_CONFIG["time_scale"], VIDEO_CONFIG["time_scale"])
                    ts_ids = (map_to_nearest_scale(frame_ts, scale) / VIDEO_CONFIG["time_scale"]).astype(np.int32)
                    all_temporal_ids.extend(ts_ids.tolist())

                # 构建消息并调用 chat 接口
                clean_q = re.sub(r'<.*?>', '', q_text).strip()
                prompt = f"{clean_q}\nOptions:\n" + "\n".join(item['options']) + \
                         "\nAnswer with the option letter (A, B, C, or D) only."
                
                msgs = [{'role': 'user', 'content': all_frames + [prompt]}]
                
                # 3D-Packing 分组处理
                ts_id_group = group_array(all_temporal_ids, VIDEO_CONFIG["packing_nums"])

                with torch.inference_mode():
                    response = model.chat(
                        msgs=msgs,
                        tokenizer=tokenizer,
                        use_image_id=False,
                        max_slice_nums=1, # 视频场景固定为 1
                        temporal_ids=ts_id_group
                    )
                pred_option = extract_option(response)

                del all_frames, msgs
                torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            print(f"\n❌ Error {v_id}: {e}")
            response, pred_option = str(e), "ERROR"

        # 4. 字段完全对齐保存
        res_entry = {
            "video_id": v_id, "question": q_text,
            "task_type": item.get('task_type', 'unknown'),
            "difficulty": item.get('difficulty', 'unknown'),
            "ground_truth": item.get('correct_options'),
            "prediction": pred_option,
            "raw_output": response.replace('\n', ' '),
            "is_correct": str(pred_option == item.get('correct_options'))
        }
        
        with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(res_entry)
            f.flush()
            os.fsync(f.fileno())

    print(f"\n✅ MiniCPM-V 4.5 测评完成，采样策略已对齐。")

if __name__ == "__main__":
    main()
