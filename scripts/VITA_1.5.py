import os
import json
import csv
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.conv
import torch.nn.modules.linear
import time
import numpy as np
from PIL import Image
from decord import VideoReader, cpu

# ==========================================
# 0. 核心算子补丁 (解决 Whale 内部精度回跳)
# ==========================================
_orig_conv2d = F.conv2d
def _safe_conv2d(input, weight, bias=None, *args, **kwargs):
    if input.dtype != weight.dtype:
        input = input.to(weight.dtype)
    if bias is not None and bias.dtype != weight.dtype:
        bias = bias.to(weight.dtype)
    return _orig_conv2d(input, weight, bias, *args, **kwargs)

F.conv2d = _safe_conv2d
torch.nn.modules.conv.F.conv2d = _safe_conv2d

_orig_linear = F.linear
def _safe_linear(input, weight, bias=None, *args, **kwargs):
    if input.dtype != weight.dtype:
        input = input.to(weight.dtype)
    if bias is not None and bias.dtype != weight.dtype:
        bias = bias.to(weight.dtype)
    return _orig_linear(input, weight, bias, *args, **kwargs)

F.linear = _safe_linear
torch.nn.modules.linear.F.linear = _safe_linear

def skip_init(*args, **kwargs): pass
nn.init.kaiming_uniform_ = skip_init
nn.init.kaiming_normal_ = skip_init
nn.init.uniform_ = skip_init
nn.init.normal_ = skip_init

from vita.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, MAX_IMAGE_LENGTH
from vita.conversation import conv_templates
from vita.model.builder import load_pretrained_model
from vita.util.mm_utils import get_model_name_from_path, tokenizer_image_token
from vita.util.utils import disable_torch_init

# ==========================================
# 1. 配置区
# ==========================================
CONFIG = {
    "model_path": "/share/cwt/models/VITA-1.5", 
    "model_type": "qwen2p5_instruct", 
    "conv_mode": "qwen2p5_instruct", 
    "load_8bit": False,
    "max_frames_total": 16, 
    "json_path": "/share/hjx/global_json_list/021/QA.json",             
    "output_csv": "/data1/student/cwt/ego_month/result_fix/VITA-1.5/021/vita1.5_output_fp32.csv", 
    "video_base_dir": "/share/hjx/",
}

def load_multiple_video_frames(video_paths, image_processor, max_frames_total=16):
    valid_paths = [p for p in video_paths if os.path.exists(p)]
    num_videos = len(valid_paths)
    
    if num_videos == 0:
        raise ValueError(f"提供的所有视频均不存在或无法访问: {video_paths}")
        
    frames_allocation = []
    for i in range(num_videos):
        base = max_frames_total // num_videos
        extra = 1 if i < (max_frames_total % num_videos) else 0
        frames_allocation.append(base + extra)

    all_processed_images = []
    total_frames_all = 0
    
    for v_path, allocated_frames in zip(valid_paths, frames_allocation):
        if allocated_frames == 0:
            continue
            
        try:
            vreader = VideoReader(v_path, ctx=cpu(0))
            total_frames = len(vreader)
            if total_frames == 0:
                continue
                
            total_frames_all += total_frames
            
            sample_indices = [int(i) for i in np.linspace(0, total_frames - 1, min(total_frames, allocated_frames))]
            frames = vreader.get_batch(sample_indices).asnumpy()
            
            for f in frames:
                all_processed_images.append(
                    image_processor.preprocess(Image.fromarray(f), return_tensors="pt")["pixel_values"][0]
                )
        except Exception as e:
            print(f" -> [错误] 读取视频 {v_path} 时失败: {e}")
            continue
            
    if not all_processed_images:
        raise ValueError(f"无法从列表中提取到任何有效帧: {valid_paths}")
        
    return torch.stack(all_processed_images), len(all_processed_images), total_frames_all

def extract_prediction(raw_output):
    if not raw_output:
        return "None"
    match = re.search(r'(?:选项|Option|Answer|答案)?[^\w]*([A-D])', raw_output, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r'\b([A-D])\b', raw_output.upper())
    if match:
        return match.group(1).upper()
    return "None"

def main():
    disable_torch_init()
    torch.cuda.empty_cache()

    print(f"\n--- [1/4] 加载模型: {CONFIG['model_path']} ---")
    model_name = get_model_name_from_path(CONFIG['model_path'])
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        CONFIG['model_path'], None, model_name, CONFIG['model_type'],
        load_8bit=CONFIG['load_8bit'], device_map="auto"
    )
    model.eval()

    vision_tower = model.get_model().vision_tower
    audio_tower = model.get_model().audio_encoder
    v_device = next(vision_tower.parameters()).device
    a_device = next(audio_tower.parameters()).device
    target_dtype = next(model.parameters()).dtype

    print(f"--- [2/4] 环境检查 ---")
    print(f"[LOG] 视觉端设备: {v_device}, 模型精度: {target_dtype}, 全局帧数上限: {CONFIG['max_frames_total']}")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"--- [3/4] 加载 JSON 数据 ---")
    with open(CONFIG['json_path'], 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    total_tasks = len(qa_data)

    csv_headers = ["video_id", "question", "task_type", "difficulty", "ground_truth", "prediction", "raw_output", "is_correct"]
    
    # 【新增功能 1】检查并自动创建输出目录
    output_dir = os.path.dirname(CONFIG['output_csv'])
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"[提示] 自动创建输出目录: {output_dir}")

    print(f"--- [4/4] 开始批量 QA 推理测试 (共 {total_tasks} 题) ---")
    
    # 记录总循环开始的时间，用于计算 ETA
    global_start_time = time.time()
    correct_count = 0

    with open(CONFIG['output_csv'], 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader()
        
        for idx, item in enumerate(qa_data):
            video_id = item.get("video_id")
            raw_vpath = item.get("video_path", video_id)
            
            if isinstance(raw_vpath, list):
                v_paths = raw_vpath
            elif isinstance(raw_vpath, str):
                v_paths = [p.strip() for p in raw_vpath.split(",")]
            else:
                v_paths = [str(raw_vpath)]
                
            video_paths = []
            for p in v_paths:
                if not p.endswith(".mp4"):
                    p += ".mp4"
                video_paths.append(os.path.join(CONFIG["video_base_dir"], p))
            
            question_text = item.get("question", "")
            options = item.get("options", [])
            ground_truth = item.get("correct_options", "")
            task_type = item.get("task_type", "")
            difficulty = item.get("difficulty", "")
            
            # 【新增功能 2】打印百分比进度
            progress_pct = (idx + 1) / total_tasks * 100
            print(f"\n[{idx + 1}/{total_tasks}] ({progress_pct:.1f}%) Video ID: {video_id} | 难度: {difficulty} | 题型: {task_type}")
            print(f" -> 待处理视频列表: {v_paths}")
            
            try:
                image_tensor, slice_len, total_frames = load_multiple_video_frames(
                    video_paths, image_processor, max_frames_total=CONFIG['max_frames_total']
                )
                image_tensor = image_tensor.to(v_device, dtype=target_dtype)
                
                print(f" -> [DEBUG] 实际提取总帧数: {slice_len}")
                
                audios = {
                    "audios": torch.zeros(1, 400, 80).to(a_device, dtype=target_dtype),
                    "lengths": torch.tensor([400]).to(a_device),
                    "lengths_for_llm": torch.tensor([0]).to(a_device) 
                }

                opts_str = "\n".join(options)
                prompt_text = f"你是一个精准的视频问答助手。请仔细观看视频，并用简体中文回答下面的选择题。\n\n问题：{question_text}\n{opts_str}\n\n指令：请仅仅输出正确选项的字母（A、B、C 或 D），不要做任何多余的解释，绝对不要描述视频内容！"

                qs = DEFAULT_IMAGE_TOKEN * slice_len + "\n" + prompt_text
                conv = conv_templates[CONFIG['conv_mode']].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt(modality="video")

                input_device = next(model.parameters()).device
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(input_device)

                start_time = time.time()
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor, 
                        audios=audios,
                        do_sample=True,          
                        temperature=0.3,         
                        max_new_tokens=2,       
                        repetition_penalty=1.0,  
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                
                if output_ids.shape[1] > input_ids.shape[1] and torch.all(output_ids[0, :5] == input_ids[0, :5]):
                    generated_ids = output_ids[:, input_ids.shape[1]:]
                else:
                    generated_ids = output_ids
                
                raw_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                
                prediction = extract_prediction(raw_output)
                is_correct = (prediction == ground_truth)
                if is_correct:
                    correct_count += 1
                
                print(f" -> 耗时: {time.time() - start_time:.2f}s | 模型原输出: {raw_output}")
                print(f" -> 预测: {prediction} | 标答: {ground_truth} | 判定结果: {is_correct}")
                
                writer.writerow({
                    "video_id": video_id,
                    "question": question_text,
                    "task_type": task_type,
                    "difficulty": difficulty,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "raw_output": raw_output,
                    "is_correct": str(is_correct)
                })
                csvfile.flush()
                
            except Exception as e:
                print(f" -> [错误] 处理 {video_id} 时发生异常: {e}")
                import traceback
                traceback.print_exc()

            # 【新增功能 2】计算并打印 ETA（预计剩余时间）与当前准确率
            elapsed_time = time.time() - global_start_time
            avg_time_per_task = elapsed_time / (idx + 1)
            eta_seconds = avg_time_per_task * (total_tasks - idx - 1)
            current_acc = (correct_count / (idx + 1)) * 100
            print(f" ===> [统计] 进度: {idx+1}/{total_tasks} | 当前准确率: {current_acc:.1f}% | 预计剩余时间: {eta_seconds/60:.1f} 分钟 <===")

if __name__ == "__main__":
    main()

# import os
# import json
# import csv
# import re
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.nn.modules.conv
# import torch.nn.modules.linear
# import time
# import numpy as np
# from PIL import Image
# from decord import VideoReader, cpu

# # ==========================================
# # 0. 核心算子补丁 (解决 Whale 内部精度回跳)
# # ==========================================
# _orig_conv2d = F.conv2d
# def _safe_conv2d(input, weight, bias=None, *args, **kwargs):
#     if input.dtype != weight.dtype:
#         input = input.to(weight.dtype)
#     if bias is not None and bias.dtype != weight.dtype:
#         bias = bias.to(weight.dtype)
#     return _orig_conv2d(input, weight, bias, *args, **kwargs)

# F.conv2d = _safe_conv2d
# torch.nn.modules.conv.F.conv2d = _safe_conv2d

# _orig_linear = F.linear
# def _safe_linear(input, weight, bias=None, *args, **kwargs):
#     if input.dtype != weight.dtype:
#         input = input.to(weight.dtype)
#     if bias is not None and bias.dtype != weight.dtype:
#         bias = bias.to(weight.dtype)
#     return _orig_linear(input, weight, bias, *args, **kwargs)

# F.linear = _safe_linear
# torch.nn.modules.linear.F.linear = _safe_linear

# def skip_init(*args, **kwargs): pass
# nn.init.kaiming_uniform_ = skip_init
# nn.init.kaiming_normal_ = skip_init
# nn.init.uniform_ = skip_init
# nn.init.normal_ = skip_init

# from vita.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, MAX_IMAGE_LENGTH
# from vita.conversation import conv_templates
# from vita.model.builder import load_pretrained_model
# from vita.util.mm_utils import get_model_name_from_path, tokenizer_image_token
# from vita.util.utils import disable_torch_init

# # ==========================================
# # 1. 配置区
# # ==========================================
# CONFIG = {
#     "model_path": "/share/cwt/models/VITA-1.5", 
#     "model_type": "qwen2p5_instruct", 
#     "conv_mode": "qwen2p5_instruct", 
#     "load_8bit": True,
#     # 【关键修改】设定所有视频加起来的全局最高抽帧总数
#     "max_frames_total": 16, 
#     "json_path": "/share/hjx/global_json_list/002/QA.json",             
#     "output_csv": "/data1/student/cwt/ego_month/result/VITA-1.5/002/vita1.5_output.csv", 
#     "video_base_dir": "/share/hjx/",
# }

# def load_multiple_video_frames(video_paths, image_processor, max_frames_total=16):
#     """
#     将全局抽帧数平摊到各个视频上，确保拼接后的总张量帧数严格不超过 max_frames_total。
#     """
#     valid_paths = [p for p in video_paths if os.path.exists(p)]
#     num_videos = len(valid_paths)
    
#     if num_videos == 0:
#         raise ValueError(f"提供的所有视频均不存在或无法访问: {video_paths}")
        
#     # 动态分配每个视频应抽取的帧数 (解决非整除问题)
#     frames_allocation = []
#     for i in range(num_videos):
#         base = max_frames_total // num_videos
#         extra = 1 if i < (max_frames_total % num_videos) else 0
#         frames_allocation.append(base + extra)

#     all_processed_images = []
#     total_frames_all = 0
    
#     for v_path, allocated_frames in zip(valid_paths, frames_allocation):
#         if allocated_frames == 0:
#             continue # 如果视频数量多于最大帧数，排在后面的直接跳过
            
#         try:
#             vreader = VideoReader(v_path, ctx=cpu(0))
#             total_frames = len(vreader)
#             if total_frames == 0:
#                 continue
                
#             total_frames_all += total_frames
            
#             # 按照分配到的固定名额均匀抽帧
#             sample_indices = [int(i) for i in np.linspace(0, total_frames - 1, min(total_frames, allocated_frames))]
#             frames = vreader.get_batch(sample_indices).asnumpy()
            
#             # 预处理并追加
#             for f in frames:
#                 all_processed_images.append(
#                     image_processor.preprocess(Image.fromarray(f), return_tensors="pt")["pixel_values"][0]
#                 )
#         except Exception as e:
#             print(f" -> [错误] 读取视频 {v_path} 时失败: {e}")
#             continue
            
#     if not all_processed_images:
#         raise ValueError(f"无法从列表中提取到任何有效帧: {valid_paths}")
        
#     # 堆叠成完整 Tensor，其尺寸第一维将严格限制在 max_frames_total 以内
#     return torch.stack(all_processed_images), len(all_processed_images), total_frames_all

# def extract_prediction(raw_output):
#     """提取模型回答中的选项 (A/B/C/D)"""
#     if not raw_output:
#         return "None"
#     match = re.search(r'(?:选项|Option|Answer|答案)?[^\w]*([A-D])', raw_output, re.IGNORECASE)
#     if match:
#         return match.group(1).upper()
#     match = re.search(r'\b([A-D])\b', raw_output.upper())
#     if match:
#         return match.group(1).upper()
#     return "None"

# def main():
#     disable_torch_init()
#     torch.cuda.empty_cache()

#     print(f"\n--- [1/4] 加载模型: {CONFIG['model_path']} ---")
#     model_name = get_model_name_from_path(CONFIG['model_path'])
#     tokenizer, model, image_processor, context_len = load_pretrained_model(
#         CONFIG['model_path'], None, model_name, CONFIG['model_type'],
#         load_8bit=CONFIG['load_8bit'], device_map="auto"
#     )
#     model.eval()

#     vision_tower = model.get_model().vision_tower
#     audio_tower = model.get_model().audio_encoder
#     v_device = next(vision_tower.parameters()).device
#     a_device = next(audio_tower.parameters()).device
#     target_dtype = next(model.parameters()).dtype

#     print(f"--- [2/4] 环境检查 ---")
#     print(f"[LOG] 视觉端设备: {v_device}, 模型精度: {target_dtype}, 全局帧数上限: {CONFIG['max_frames_total']}")

#     if tokenizer.pad_token_id is None:
#         tokenizer.pad_token_id = tokenizer.eos_token_id

#     print(f"--- [3/4] 加载 JSON 数据 ---")
#     with open(CONFIG['json_path'], 'r', encoding='utf-8') as f:
#         qa_data = json.load(f)

#     csv_headers = ["video_id", "question", "task_type", "difficulty", "ground_truth", "prediction", "raw_output", "is_correct"]
    
#     print(f"--- [4/4] 开始批量 QA 推理测试 (共 {len(qa_data)} 题) ---")
#     with open(CONFIG['output_csv'], 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
#         writer.writeheader()
        
#         for idx, item in enumerate(qa_data):
#             video_id = item.get("video_id")
#             raw_vpath = item.get("video_path", video_id)
            
#             # 解析单视频、多视频列表或逗号分隔的视频路径
#             if isinstance(raw_vpath, list):
#                 v_paths = raw_vpath
#             elif isinstance(raw_vpath, str):
#                 v_paths = [p.strip() for p in raw_vpath.split(",")]
#             else:
#                 v_paths = [str(raw_vpath)]
                
#             video_paths = []
#             for p in v_paths:
#                 if not p.endswith(".mp4"):
#                     p += ".mp4"
#                 video_paths.append(os.path.join(CONFIG["video_base_dir"], p))
            
#             question_text = item.get("question", "")
#             options = item.get("options", [])
#             ground_truth = item.get("correct_options", "")
#             task_type = item.get("task_type", "")
#             difficulty = item.get("difficulty", "")
            
#             print(f"\n[{idx + 1}/{len(qa_data)}] Video ID: {video_id} | 难度: {difficulty} | 题型: {task_type}")
#             print(f" -> 待处理视频列表: {v_paths}")
            
#             try:
#                 # 传入更新后的多视频分配函数
#                 image_tensor, slice_len, total_frames = load_multiple_video_frames(
#                     video_paths, image_processor, max_frames_total=CONFIG['max_frames_total']
#                 )
#                 image_tensor = image_tensor.to(v_device, dtype=target_dtype)
                
#                 print(f" -> [DEBUG] 实际提取总帧数: {slice_len}")
                
#                 audios = {
#                     "audios": torch.zeros(1, 400, 80).to(a_device, dtype=target_dtype),
#                     "lengths": torch.tensor([400]).to(a_device),
#                     "lengths_for_llm": torch.tensor([0]).to(a_device) 
#                 }

#                 opts_str = "\n".join(options)
#                 prompt_text = f"你是一个精准的视频问答助手。请仔细观看视频，并用简体中文回答下面的选择题。\n\n问题：{question_text}\n{opts_str}\n\n指令：请仅仅输出正确选项的字母（A、B、C 或 D），不要做任何多余的解释，绝对不要描述视频内容！"

#                 qs = DEFAULT_IMAGE_TOKEN * slice_len + "\n" + prompt_text
#                 conv = conv_templates[CONFIG['conv_mode']].copy()
#                 conv.append_message(conv.roles[0], qs)
#                 conv.append_message(conv.roles[1], None)
#                 prompt = conv.get_prompt(modality="video")

#                 input_device = next(model.parameters()).device
#                 input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(input_device)

#                 start_time = time.time()
#                 with torch.inference_mode():
#                     output_ids = model.generate(
#                         input_ids,
#                         images=image_tensor, 
#                         audios=audios,
#                         do_sample=False,          
#                         temperature=0.0,         
#                         max_new_tokens=10,       
#                         repetition_penalty=1.0,  
#                         use_cache=True,
#                         pad_token_id=tokenizer.pad_token_id,
#                         eos_token_id=tokenizer.eos_token_id,
#                     )
                
#                 if output_ids.shape[1] > input_ids.shape[1] and torch.all(output_ids[0, :5] == input_ids[0, :5]):
#                     generated_ids = output_ids[:, input_ids.shape[1]:]
#                 else:
#                     generated_ids = output_ids
                
#                 raw_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                
#                 prediction = extract_prediction(raw_output)
#                 is_correct = (prediction == ground_truth)
                
#                 print(f" -> 耗时: {time.time() - start_time:.2f}s | 模型原输出: {raw_output}")
#                 print(f" -> 预测: {prediction} | 标答: {ground_truth} | 判定结果: {is_correct}")
                
#                 writer.writerow({
#                     "video_id": video_id,
#                     "question": question_text,
#                     "task_type": task_type,
#                     "difficulty": difficulty,
#                     "ground_truth": ground_truth,
#                     "prediction": prediction,
#                     "raw_output": raw_output,
#                     "is_correct": str(is_correct)
#                 })
#                 csvfile.flush()
                
#             except Exception as e:
#                 print(f" -> [错误] 处理 {video_id} 时发生异常: {e}")
#                 import traceback
#                 traceback.print_exc()

# if __name__ == "__main__":
#     main()



