import torch
import json
import os
import re
import csv
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor
import gc
import decord

# ==================== 配置区 ====================
MODEL_PATH = "/share/cwt/models/VideoLLaMA3-7B/VideoLLaMA3-7B"
VIDEO_ROOT = "/share/hjx"
JSON_FILE = "/share/hjx/global_json_list/011/QA.json" 
OUTPUT_CSV = "/data1/student/cwt/ego_month/result_fix/VideoLLaMA3-7B/011/videollama3_videomme_results.csv"

VIDEO_CONFIG = {"max_frames": 256} 
# ===============================================

def load_model_local():
    print(f"📦 正在加载模型: {MODEL_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", 
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return model, processor

def extract_option(response):
    match = re.search(r'\b([A-D])\b', response)
    return match.group(1).upper() if match else "FAIL"

def get_finished_keys(csv_path):
    """通过标准化 key 避免重复"""
    if not os.path.exists(csv_path):
        return set()
    try:
        # 强制读取所有列为字符串，避免科学计数法或浮点数导致匹配失败
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        # 组合 ID 和 Question 并去除首尾空格
        return set((row['video_id'].strip() + row['question'].strip()) for _, row in df.iterrows())
    except Exception as e:
        print(f"⚠️ 读取旧结果失败: {e}")
        return set()

def get_video_duration(path):
    try:
        vr = decord.VideoReader(path, ctx=decord.cpu(0))
        return len(vr) / vr.get_avg_fps()
    except Exception:
        return 1.0

def main():
    # 1. 立即创建目录
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # 2. 字段对齐
    fieldnames = ['video_id', 'question', 'task_type', 'difficulty', 'ground_truth', 'prediction', 'raw_output', 'is_correct']
    
    finished_keys = get_finished_keys(OUTPUT_CSV)
    
    # 初始化文件头
    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            f.flush()
            os.fsync(f.fileno()) # 🚀 强制物理写入磁盘

    # 3. 加载模型
    model, processor = load_model_local()

    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"📊 总任务: {len(data)} | 已跳过: {len(finished_keys)}")

    # 4. 推理循环
    for item in tqdm(data):
        v_id = str(item.get('video_id', '')).strip()
        q_text = str(item.get('question', '')).strip()
        
        # 🚀 严格校验：标准化后对比
        if (v_id + q_text) in finished_keys:
            continue

        v_paths = item.get('video_path', [])
        if isinstance(v_paths, str): v_paths = [v_paths]
        full_paths = [os.path.join(VIDEO_ROOT, v if v.endswith('.mp4') else v + ".mp4") for v in v_paths]
        valid_paths = [p for p in full_paths if os.path.exists(p)]

        try:
            if not valid_paths:
                response, pred_option = "Missing Files", "MISSING"
            else:
                # 动态分配帧数
                v_durations = [get_video_duration(p) for p in valid_paths]
                total_dur = sum(v_durations)
                content = []
                for path, dur in zip(valid_paths, v_durations):
                    v_frames = max(4, int((dur / total_dur) * VIDEO_CONFIG["max_frames"]))
                    content.append({"type": "video", "video": {"video_path": path, "max_frames": v_frames}})
                
                prompt_text = f"{re.sub(r'<.*?>', '', q_text)}\nOptions:\n" + "\n".join(item['options']) + \
                              "\nAnswer with the option letter (A, B, C, or D) only."
                content.append({"type": "text", "text": prompt_text})

                inputs = processor(conversation=[{"role":"user", "content":content}], return_tensors="pt")
                inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                if "pixel_values" in inputs: inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

                with torch.inference_mode():
                    output_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
                
                response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                pred_option = extract_option(response)

                del inputs, output_ids
                torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            print(f"\n❌ 错误 {v_id}: {e}")
            response, pred_option = str(e), "ERROR"

        # 5. 保存结果：对齐 002.csv 字段
        res_entry = {
            "video_id": v_id,
            "question": q_text,
            "task_type": item.get('task_type', 'unknown'),
            "difficulty": item.get('difficulty', 'unknown'),
            "ground_truth": item.get('correct_options'),
            "prediction": pred_option,
            "raw_output": response.replace('\n', ' '),
            "is_correct": str(pred_option == item.get('correct_options'))
        }
        
        # 🚀 强化写入逻辑：每条数据都确保落盘
        with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(res_entry)
            f.flush()
            os.fsync(f.fileno()) # 💡 关键：解决日志显示完成但文件没数据的问题

    print(f"\n✅ 任务处理完毕，已同步至磁盘。")

if __name__ == "__main__":
    main()

# import torch
# import json
# import os
# import re
# import csv
# import pandas as pd
# from tqdm import tqdm
# from transformers import AutoModelForCausalLM, AutoProcessor

# # ==================== 配置区 ====================
# MODEL_PATH = "/share/cwt/models/VideoLLaMA3-7B/VideoLLaMA3-7B"
# VIDEO_ROOT = "/share/hjx"
# JSON_FILE = "/share/hjx/global_json_list/004/QA.json" 
# OUTPUT_CSV = "/data1/student/cwt/ego_month/result/004/videollama3_videomme_results.csv"

# # 采样配置：移除 fps 限制，强制在视频总时长内均匀抽取 256 帧
# VIDEO_CONFIG = {
#     "max_frames": 128,
# } 
# # ===============================================

# def load_model_local():
#     print(f"📦 正在加载模型: {MODEL_PATH}...")
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_PATH,
#         trust_remote_code=True,
#         device_map="auto",
#         torch_dtype=torch.bfloat16,
#         attn_implementation="flash_attention_2", # 请确保已安装 flash-attn
#     )
#     processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
#     return model, processor

# def extract_option(response):
#     """提取 A/B/C/D 选项"""
#     match = re.search(r'\b([A-D])\b', response)
#     if match:
#         return match.group(1).upper()
#     return "FAIL"

# def main():
#     model, processor = load_model_local()
    
#     if not os.path.exists(os.path.dirname(OUTPUT_CSV)):
#         os.makedirs(os.path.dirname(OUTPUT_CSV))

#     # 初始化 CSV 文件头（如果文件不存在）
#     if not os.path.exists(OUTPUT_CSV):
#         with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
#             writer = csv.DictWriter(f, fieldnames=["sample_id", "video_id", "ground_truth", "prediction", "is_correct", "raw_output"])
#             writer.writeheader()

#     with open(JSON_FILE, 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     print(f"📊 开始均匀采样测评（Max Frames: {VIDEO_CONFIG['max_frames']}），总计 {len(data)} 条任务...")

#     for item in tqdm(data):
#         v_paths = item.get('video_path', [])
#         if isinstance(v_paths, str): v_paths = [v_paths]
        
#         full_paths = [os.path.join(VIDEO_ROOT, v if v.endswith('.mp4') else v + ".mp4") for v in v_paths]
#         valid_paths = [p for p in full_paths if os.path.exists(p)]

#         try:
#             if not valid_paths:
#                 raise FileNotFoundError(f"未找到视频: {full_paths}")

#             # 构建对话内容
#             content = []
#             for path in valid_paths:
#                 content.append({
#                     "type": "video", 
#                     "video": {
#                         "video_path": path, 
#                         **VIDEO_CONFIG # 这里不再包含 fps，触发模型全长采样
#                     }
#                 })
            
#             clean_question = re.sub(r'<.*?>', '', item['question']).strip()
#             prompt_text = (
#                 f"{clean_question}\n"
#                 f"Options:\n" + "\n".join(item['options']) + "\n"
#                 "Answer with the option letter (A, B, C, or D) only."
#             )
#             content.append({"type": "text", "text": prompt_text})

#             conversation = [
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": content},
#             ]

#             # 预处理与推理
#             inputs = processor(conversation=conversation, return_tensors="pt")
#             inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
#             if "pixel_values" in inputs:
#                 inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

#             with torch.inference_mode():
#                 output_ids = model.generate(
#                     **inputs, 
#                     max_new_tokens=32, 
#                     do_sample=False,
#                     # temperature=0.0 # 在 do_sample=False 时无需设置
#                 )
            
#             response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
#             pred_option = extract_option(response)

#         except Exception as e:
#             print(f"\n❌ ID {item.get('sample_id')} 出错: {e}")
#             response, pred_option = str(e), "ERROR"

#         # 逐行追加保存结果，防止内存堆积和意外丢失
#         res_entry = {
#             "sample_id": item.get('sample_id'),
#             "video_id": item.get('video_id'),
#             "ground_truth": item.get('correct_options'),
#             "prediction": pred_option,
#             "is_correct": (pred_option == item.get('correct_options')),
#             "raw_output": response.replace('\n', ' ')
#         }
        
#         with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
#             writer = csv.DictWriter(f, fieldnames=res_entry.keys())
#             writer.writerow(res_entry)

#     print(f"\n✅ 测评结束。结果保存在: {OUTPUT_CSV}")

# if __name__ == "__main__":
#     main()


# import torch
# import json
# import os
# import re
# import pandas as pd
# from tqdm import tqdm
# from transformers import AutoModelForCausalLM, AutoProcessor

# # ==================== 配置区 ====================
# # 指向你本地已下载好的模型目录
# MODEL_PATH = "/share/cwt/models/VideoLLaMA3-7B/VideoLLaMA3-7B"

# VIDEO_ROOT = "/share/hjx"
# JSON_FILE = "/share/hjx/global_json_list/003/QA.json" 
# OUTPUT_CSV = "/data1/student/cwt/ego_month/result/003/videollama3_videomme_results.csv"

# # 采样配置
# # fps=1: 每秒抽1帧; max_frames=128: 1小时视频约28秒抽1帧
# VIDEO_CONFIG = {"fps": 1, "max_frames": 256} 
# # ===============================================

# def load_model_local():
#     print(f"📦 正在从本地路径加载模型: {MODEL_PATH}...")
    
#     if not os.path.exists(MODEL_PATH):
#         raise FileNotFoundError(f"❌ 找不到本地模型路径，请确认是否下载完成: {MODEL_PATH}")

#     # 直接从本地路径加载模型权重和配置
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_PATH,
#         trust_remote_code=True,
#         device_map="auto",
#         torch_dtype=torch.bfloat16,
#         attn_implementation="flash_attention_2", # 确保你已成功安装 flash-attn
#         # 如果显存依然不足(低于24GB)，请取消下面一行的注释开启4bit量化
#         # load_in_4bit=True, 
#     )
    
#     # 从本地路径加载处理器
#     processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
#     return model, processor

# def extract_option(response):
#     """提取 A/B/C/D 选项"""
#     match = re.search(r'\b([A-D])\b', response)
#     if match:
#         return match.group(1).upper()
#     return "FAIL"

# def main():
#     model, processor = load_model_local()
    
#     if not os.path.exists(os.path.dirname(OUTPUT_CSV)):
#         os.makedirs(os.path.dirname(OUTPUT_CSV))

#     with open(JSON_FILE, 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     results = []
#     print(f"📊 开始测评，总计 {len(data)} 条任务...")

#     for item in tqdm(data):
#         # 1. 路径补全与校验
#         v_paths = item.get('video_path', [])
#         if isinstance(v_paths, str): 
#             v_paths = [v_paths]
        
#         # 这里的逻辑会处理你在 JSON 更新后生成的列表路径
#         full_paths = [os.path.join(VIDEO_ROOT, v if v.endswith('.mp4') else v + ".mp4") for v in v_paths]
#         valid_paths = [p for p in full_paths if os.path.exists(p)]

#         try:
#             if not valid_paths:
#                 raise FileNotFoundError(f"未找到视频文件: {full_paths}")

#             # 2. 构建对话内容
#             content = []
#             for path in valid_paths:
#                 content.append({
#                     "type": "video", 
#                     "video": {
#                         "video_path": path, 
#                         **VIDEO_CONFIG
#                     }
#                 })
            
#             clean_question = re.sub(r'<.*?>', '', item['question']).strip()
#             prompt_text = (
#                 f"{clean_question}\n"
#                 f"Options:\n" + "\n".join(item['options']) + "\n"
#                 "Answer with the option letter (A, B, C, or D) only."
#             )
#             content.append({"type": "text", "text": prompt_text})

#             conversation = [
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": content},
#             ]

#             # 3. 预处理
#             inputs = processor(conversation=conversation, return_tensors="pt")
#             # 将 Tensor 转移到 GPU
#             inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
#             if "pixel_values" in inputs:
#                 inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

#             # 4. 生成
#             with torch.inference_mode():
#                 output_ids = model.generate(
#                     **inputs, 
#                     max_new_tokens=32, 
#                     do_sample=False,
#                     temperature=0.0
#                 )
            
#             response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
#             pred_option = extract_option(response)

#         except Exception as e:
#             print(f"\n❌ 处理 ID {item.get('sample_id')} 时出错: {e}")
#             response, pred_option = str(e), "ERROR"

#         # 5. 保存结果
#         results.append({
#             "sample_id": item.get('sample_id'),
#             "video_id": item.get('video_id'),
#             "ground_truth": item.get('correct_options'),
#             "prediction": pred_option,
#             "raw_output": response,
#             "is_correct": (pred_option == item.get('correct_options'))
#         })
        
#         # 实时写入 CSV
#         pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

#     print(f"\n✅ 测评结束。结果已保存至: {OUTPUT_CSV}")

# if __name__ == "__main__":
#     main()
