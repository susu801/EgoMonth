import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import re
import time
import multiprocessing
from tqdm import tqdm
from PIL import Image
from decord import VideoReader, cpu, bridge
import logging
from logging.handlers import RotatingFileHandler

# ==============================
# 全局配置（核心：只需修改这里！）
# ==============================
# 1. 要处理的JSON编号列表（你想跑的003、002、005等都填这里）
TARGET_JSON_NUMBERS = ["001", "002", "003", "004","005","009","010"]  # 示例：处理这4个JSON
# 2. 可用GPU列表（按优先级排序，先用完3、7，再用2、5）
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6]
# 3. 固定路径前缀（无需修改）
JSON_ROOT = "/share/hjx/global_json_list"
VIDEO_ROOT = "/share/hjx"
MODEL_PATH = "/data1/hjx/models/Chat-UniVi-7B-v1.5"

# 视频参数（和你原脚本一致）
MAX_PIXELS_THRESHOLD = 360 * 420
MAX_FRAME_THRESHOLD = 256
FINAL_FRAME_LIMIT = 512

sys.path.append("/data1/hjx/project/Chat-UniVi")

# ==============================
# 日志系统（按JSON编号生成独立日志）
# ==============================
def setup_logger(json_number):
    """为每个JSON生成独立日志文件：{编号}_chatunivi_eval.log"""
    log_file = f"{json_number}_chatunivi_eval.log"
    logger = logging.getLogger(f"ChatUniVi_Eval_{json_number}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 终端输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# 主进程日志（记录整体调度）
main_logger = logging.getLogger("ChatUniVi_Main")
main_logger.setLevel(logging.INFO)
main_logger.handlers.clear()
main_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
main_console = logging.StreamHandler()
main_console.setFormatter(main_formatter)
main_logger.addHandler(main_console)

# ==============================
# 视频读取（和你原脚本完全一致）
# ==============================
def read_video_frames(video_path, max_frames=MAX_FRAME_THRESHOLD, logger=None):
    logger.info(f"📌 开始读取视频文件：{video_path}")

    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        logger.info(f"视频总帧数：{total_frames}")

        if total_frames == 0:
            return []

        indices = np.linspace(
            0,
            total_frames - 1,
            min(max_frames, total_frames)
        ).astype(int)

        frames = []
        for idx in indices:
            frame = vr[idx].asnumpy()
            frames.append(Image.fromarray(frame).convert("RGB"))

        logger.info(f"✅ 视频读取完成，抽帧数：{len(frames)}")
        return frames

    except Exception as e:
        logger.error(f"❌ 视频读取失败 {video_path}: {e}", exc_info=True)
        return []

# ==============================
# 拼接视频（和你原脚本完全一致）
# ==============================
def concat_and_sample_frames(video_list, logger=None):
    logger.info(f"开始处理视频列表：{video_list}")

    all_frames = []

    # 第一阶段：每个视频最多256帧
    for rel_path in video_list:
        full_path = os.path.join(VIDEO_ROOT, rel_path)

        if not os.path.exists(full_path):
            logger.error(f"文件不存在: {full_path}")
            continue

        frames = read_video_frames(full_path, logger=logger)
        all_frames.extend(frames)

        logger.info(f"✅ 视频帧已合并，当前累计帧数：{len(all_frames)}")

    # 第二阶段：全局抽帧到512
    total_frames = len(all_frames)
    if total_frames > FINAL_FRAME_LIMIT:
        logger.info(
            f"⚠ 总帧数 {total_frames} 超出安全上限 {FINAL_FRAME_LIMIT}，开始全局均匀抽帧"
        )
        indices = np.linspace(0, total_frames - 1, FINAL_FRAME_LIMIT).astype(int)
        all_frames = [all_frames[i] for i in indices]
        logger.info(f"✅ 全局抽帧完成，当前帧数：{len(all_frames)}")

    # 第三阶段：缩放像素
    logger.info(f"开始缩放 {len(all_frames)} 帧到安全像素数（{MAX_PIXELS_THRESHOLD}）")
    processed_frames = []
    for frame in all_frames:
        w, h = frame.size
        raw_pixels = w * h
        if raw_pixels > MAX_PIXELS_THRESHOLD:
            scale = (MAX_PIXELS_THRESHOLD / raw_pixels) ** 0.5
            frame = frame.resize(
                (int(w * scale), int(h * scale)),
                Image.Resampling.LANCZOS
            )
        processed_frames.append(frame)
    logger.info(f"✅ 帧缩放完成，最终有效帧数：{len(processed_frames)}")

    return processed_frames

# ==============================
# Prompt（和你原脚本完全一致）
# ==============================
def build_prompt(question, options):
    if ">" in question:
        question = question.split(">")[-1].strip()
    option_text = "\n".join(options)
    return (
        f"{question}\n"
        f"{option_text}\n"
        f"Answer with the option's letter (A/B/C/D) directly.\n"
        f"Only output a single letter."
    )

def extract_answer(text):
    text = text.strip().upper()
    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1)
    return "ERROR"

# ==============================
# 推理（适配指定GPU）
# ==============================
def chatunivi_inference(model, tokenizer, image_processor, frames, question, options, device, logger=None):
    from ChatUniVi.mm_utils import process_images
    from ChatUniVi.conversation import conv_templates

    conv = conv_templates["v1"].copy()
    prompt_text = build_prompt(question, options)
    conv.append_message(conv.roles[0], prompt_text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    logger.info(f"🚀 开始模型推理（Logits模式），Prompt长度：{len(prompt)}")

    # 处理图像（指定GPU）
    images_tensor = process_images(
        frames,
        image_processor,
        model.config
    ).to(device, dtype=torch.float16)

    # 编码文本（指定GPU）
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, images=images_tensor)

    # 计算选项分数（和你原脚本一致）
    logits = outputs.logits[:, -1, :]
    choice_tokens = tokenizer(["A", "B", "C", "D"], add_special_tokens=False).input_ids
    choice_token_ids = [token[0] for token in choice_tokens]
    choice_logits = logits[:, choice_token_ids]
    scores = choice_logits[0].tolist()
    pred_id = torch.argmax(choice_logits, dim=-1).item()
    pred_letter = ["A", "B", "C", "D"][pred_id]

    logger.info(
        f"📊 选项打分 -> A:{scores[0]:.4f} | B:{scores[1]:.4f} | C:{scores[2]:.4f} | D:{scores[3]:.4f}"
    )
    logger.info(f"✅ 最终预测答案：{pred_letter}")

    return pred_letter

# ==============================
# 单个JSON的处理函数（绑定指定GPU）
# ==============================
def process_single_json(json_number, gpu_id):
    """处理单个JSON文件，绑定到指定GPU"""
    # 1. 配置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = f"cuda:0"  # 因为限制了GPU，所以固定为0
    logger = setup_logger(json_number)
    
    # 2. 拼接路径（自动生成，无需手动改）
    json_path = os.path.join(JSON_ROOT, json_number, "QA.json")
    output_csv = f"{json_number}_results.csv"
    
    logger.info(f"\n=====================================")
    logger.info(f"🚀 开始处理 JSON {json_number} (绑定GPU{gpu_id})")
    logger.info(f"📁 JSON路径：{json_path}")
    logger.info(f"📁 输出CSV：{output_csv}")
    logger.info(f"=====================================\n")

    # 3. 加载模型（每个GPU独立加载）
    logger.info("🚀 开始加载Chat-UniVi模型...")
    try:
        from ChatUniVi.model.builder import load_pretrained_model
        from ChatUniVi.utils import disable_torch_init
        disable_torch_init()

        tokenizer, model, image_processor, context_len = load_pretrained_model(
            MODEL_PATH, None, "ChatUniVi"
        )
        model = model.to(device, dtype=torch.float16)
        model.eval()
        logger.info("✅ 模型加载完成！")
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}", exc_info=True)
        return

    # 4. 加载JSON数据
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        logger.info(f"📊 加载测试数据完成，共 {len(data)} 条样本")
    except Exception as e:
        logger.error(f"❌ 加载JSON失败: {e}", exc_info=True)
        return

    # 5. 处理样本（和你原脚本逻辑完全一致）
    bridge.set_bridge("native")
    results = []
    correct_count = 0

    for idx, sample in enumerate(data):
        start_time = time.time()
        video_id = sample["video_id"]
        question = sample["question"]
        options = sample["options"]
        ground_truth = sample["correct_options"]

        logger.info("\n========== 开始处理样本："
                    f"{idx+1}/{len(data)} - {video_id} ==========")

        # 抽帧
        frames = concat_and_sample_frames(sample["video_path"], logger=logger)
        logger.info(f"📌 样本 {video_id} 提取帧数量：{len(frames)}")

        if not frames:
            logger.warning("⚠ 未提取到帧，跳过")
            continue

        # 推理
        try:
            pred_letter = chatunivi_inference(
                model, tokenizer, image_processor, frames, question, options, device, logger=logger
            )
            pred = extract_answer(pred_letter)
            is_correct = (pred == ground_truth)
            if is_correct:
                correct_count += 1

            logger.info(
                f"✅ 样本 {video_id} 推理完成，预测：{pred}，"
                f"真实标签：{ground_truth}，是否正确：{is_correct}"
            )
        except Exception as e:
            logger.error(f"❌ 推理失败: {e}", exc_info=True)
            pred = "ERROR"
            pred_letter = str(e)
            is_correct = False

        # 统计耗时和准确率
        elapsed = time.time() - start_time
        logger.info(f"⏱ 本样本耗时：{elapsed:.2f} 秒")
        current_acc = correct_count / (idx+1) * 100 if (idx+1) > 0 else 0
        logger.info(f"📈 当前累计准确率：{current_acc:.2f}%\n")

        # 保存结果
        results.append({
            "video_id": video_id,
            "prediction": pred,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "raw_output": pred_letter
        })

    # 6. 保存结果（自动生成{编号}_results.csv）
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    
    # 7. 统计最终结果
    final_acc = correct_count / len(results) * 100 if len(results) > 0 else 0
    logger.info(f"\n🎉 JSON {json_number} 评测完成！")
    logger.info(f"📊 最终统计 -> 总样本：{len(results)}，正确：{correct_count}，准确率：{final_acc:.2f}%")
    logger.info(f"📁 结果已保存到：{output_csv}")

# ==============================
# 主函数（调度多GPU处理多个JSON）
# ==============================
def main():
    main_logger.info(f"🚀 主进程启动！待处理JSON列表：{TARGET_JSON_NUMBERS}")
    main_logger.info(f"🖥️ 可用GPU列表：{AVAILABLE_GPUS}")

    # 1. 检查JSON数量和GPU数量
    num_jsons = len(TARGET_JSON_NUMBERS)
    num_gpus = len(AVAILABLE_GPUS)
    if num_jsons > num_gpus:
        main_logger.warning(f"⚠ JSON数量({num_jsons})超过GPU数量({num_gpus})，将串行处理剩余JSON")

    # 2. 创建进程池（最多同时运行num_gpus个进程）
    multiprocessing.set_start_method('spawn', force=True)
    pool = multiprocessing.Pool(processes=num_gpus)

    # 3. 分配GPU并启动进程
    for idx, json_number in enumerate(TARGET_JSON_NUMBERS):
        gpu_id = AVAILABLE_GPUS[idx % num_gpus]  # 循环分配GPU
        main_logger.info(f"📌 分配 GPU{gpu_id} 处理 JSON {json_number}")
        pool.apply_async(process_single_json, args=(json_number, gpu_id))

    # 4. 等待所有进程完成
    pool.close()
    pool.join()
    main_logger.info("✅ 所有JSON处理完成！")

if __name__ == "__main__":
    main()