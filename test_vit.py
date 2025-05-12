import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5" # Optional: Select GPU

# --- 从你的代码中导入 QAEVAVisionTransformer 及其依赖 ---
# 假设 qvit_eva.py 文件与此测试脚本在同一目录或已在PYTHONPATH中
# 你可能需要调整导入路径
from qvit_eva import create_qa_eva_vit_g, QAEVABlock # 确保导入 QAEVABlock 用于 isinstance 检查

# --- 配置 ---
EVA_IMG_SIZE = 224  # 与你的模型配置一致
INSTRUCTION_DIM = 4096 # 与你的模型配置一致
QA_INTEGRATION_POINT = 'late'
DROP_PATH_RATE = 0.0
QVIT_CHECKPOINT_PATH = "../models/eva_vit_g.pth" # 你的EVA ViT-G预训练权重路径

# --- 测试参数 (你可以修改这些来进行实验) ---
TEST_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DTYPE = torch.float16  # *** 关键测试点: torch.float32, torch.float16, torch.bfloat16 ***
USE_GRAD_SCALER = True     # *** 关键测试点: True 或 False (仅当 TEST_DTYPE 为 float16 时相关) ***
USE_CHECKPOINT = True     # 是否在 QAEVAVisionTransformer 中使用梯度检查点
LEARNING_RATE = 1e-4
BATCH_SIZE = 2             # 模拟的批次大小
NUM_FRAMES = 1             # 每个样本的帧数 (对于独立测试，1帧即可)
INSTR_SEQ_LEN = 16         # 指令序列长度

# --- 辅助函数：打印参数状态 ---
def print_param_status(model, prefix=""):
    print(f"\n--- {prefix} Parameter Status ---")
    found_trainable = False
    for name, param in model.named_parameters():
        if "instruct_dim_reduce" in name or "parallel_mlp" in name or "parallel_gate" in name:
            print(f"Param: {name:<60} | Requires Grad: {param.requires_grad} | Dtype: {param.dtype} | Device: {param.device} | Value (sum): {param.data.sum().item():.4f}")
            if param.requires_grad:
                found_trainable = True
    if not found_trainable:
        print("No QA-specific trainable parameters found with the specified names.")
    elif not any(p.requires_grad for p in model.parameters()):
        print("Warning: No parameters in the model require gradients at all!")


if __name__ == "__main__":
    print(f"--- Starting QAEVAVisionTransformer Parameter Update Test ---")
    print(f"Device: {TEST_DEVICE}, Dtype: {TEST_DTYPE}, Use GradScaler: {USE_GRAD_SCALER}, LR: {LEARNING_RATE}")

    # 1. 实例化 QAEVAVisionTransformer 模型
    #    注意: create_qa_eva_vit_g 内部的 precision 处理需要根据你的修改来激活
    try:
        print(f"\nLoading QAEVAVisionTransformer with precision={TEST_DTYPE}...")
        qvit_model = create_qa_eva_vit_g(
            img_size=EVA_IMG_SIZE,
            drop_path_rate=DROP_PATH_RATE,
            use_checkpoint=USE_CHECKPOINT,
            precision=TEST_DTYPE, # 传递测试的dtype
            instruction_dim=INSTRUCTION_DIM,
            integration_point=QA_INTEGRATION_POINT,
            cached_file=QVIT_CHECKPOINT_PATH
        )
        # 手动转换模型到测试设备和数据类型 (如果 create_qa_eva_vit_g 内部没有完全处理)
        # 这一步非常重要，确保模型参数是你测试的 DTYPE
        # if TEST_DTYPE == torch.float16:
        #     qvit_model.half()
        # elif TEST_DTYPE == torch.bfloat16:
        #     qvit_model.bfloat16()
        # else: float32 is default

        qvit_model = qvit_model.to(TEST_DEVICE)
        print(f"QAEVAVisionTransformer loaded and moved to {TEST_DEVICE}.")

    except Exception as e:
        print(f"Error creating QAEVAVisionTransformer: {e}")
        import traceback
        traceback.print_exc()
        exit()

    # 2. 冻结基础模型参数 (与你的主模型逻辑一致)
    if hasattr(qvit_model, 'freeze_base_model'):
        qvit_model.freeze_base_model()
        print("Called freeze_base_model().")
    else:
        print("Warning: QAEVAVisionTransformer does not have freeze_base_model method.")

    # 打印冻结后的参数状态 (初始状态)
    print_param_status(qvit_model, "Initial (After Freeze)")

    # 3. 准备优化器 (只优化可训练参数)
    trainable_params = [p for p in qvit_model.parameters() if p.requires_grad]
    if not trainable_params:
        print("\nError: No trainable parameters found in qvit_model. Cannot proceed with test.")
        # 进一步检查 freeze_base_model 的逻辑或参数命名
        print("Inspect the output of 'Initial (After Freeze)' above.")
        exit()
    else:
        print(f"\nFound {len(trainable_params)} trainable parameters for the optimizer.")

    optimizer = AdamW(qvit_model.parameters(), lr=LEARNING_RATE)

    # 4. 准备 GradScaler (如果使用)
    scaler = GradScaler(enabled=(TEST_DTYPE == torch.float16 and USE_GRAD_SCALER))
    print(f"GradScaler enabled: {scaler.is_enabled()}")

    # 5. 模拟输入数据
    # (BATCH_SIZE * NUM_FRAMES, Channels, Height, Width)
    dummy_pixel_values = torch.randn(
        BATCH_SIZE * NUM_FRAMES, 3, EVA_IMG_SIZE, EVA_IMG_SIZE,
        device=TEST_DEVICE, dtype=TEST_DTYPE # 输入数据也用目标 DTYPE
    )
    # (BATCH_SIZE * NUM_FRAMES, SeqLen, InstructionDim)
    dummy_instruct_states = torch.randn(
        BATCH_SIZE * NUM_FRAMES, INSTR_SEQ_LEN, INSTRUCTION_DIM,
        device=TEST_DEVICE, dtype=TEST_DTYPE # 指令也用目标 DTYPE (LLM输出通常是fp16/bf16或fp32)
    )
    dummy_instruct_masks = torch.ones(
        BATCH_SIZE * NUM_FRAMES, INSTR_SEQ_LEN,
        device=TEST_DEVICE, dtype=torch.long
    )

    # 6. 获取可训练参数的初始值 (深拷贝)
    initial_param_values = {}
    for name, param in qvit_model.named_parameters():
        if param.requires_grad:
            initial_param_values[name] = param.data.clone()

    # 7. 执行一个训练步骤
    qvit_model.train() # 设置为训练模式
    optimizer.zero_grad()

    print("\n--- Performing a single training step ---")
    try:
        with autocast(enabled=(TEST_DTYPE != torch.float32)):
            # 模型输出 (BATCH_SIZE * NUM_FRAMES, NumPatches+1, EmbedDim)
            output_features = qvit_model(
                x=dummy_pixel_values,
                instruct_states=dummy_instruct_states,
                instruct_masks=dummy_instruct_masks
            )
            # 创建一个简单的模拟损失
            # 例如，让模型输出的CLS token的范数接近某个值
            # loss = (output_features[:, 0].norm() - 10.0).pow(2)
            # 或者更简单，直接对输出求和作为loss
            loss = output_features.sum() * 0.001 # 乘以一个小数防止梯度过大
            print(f"Calculated Loss: {loss.item()}")

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Error: Loss is {loss}. Stopping.")
        else:
            scaler.scale(loss).backward()
            # scaler.unscale_(optimizer) # 如果需要梯度裁剪
            # torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0) # 如果需要梯度裁剪
            scaler.step(optimizer)
            scaler.update()
            print("Optimizer step and scaler update completed.")

    except Exception as e:
        print(f"Error during training step: {e}")
        import traceback
        traceback.print_exc()
        print_param_status(qvit_model, "After Error")
        exit()

    # 8. 检查参数是否更新
    print("\n--- Checking Parameter Updates ---")
    updated_count = 0
    not_updated_count = 0
    params_changed_info = []

    for name, param in qvit_model.named_parameters():
        if name in initial_param_values: # 即 param.requires_grad is True
            initial_val = initial_param_values[name]
            current_val = param.data
            # 比较张量是否不相等
            # 使用 torch.equal 对于浮点数可能过于严格，建议检查差值
            # if not torch.equal(initial_val, current_val):
            diff = torch.abs(initial_val - current_val).sum().item()
            if diff > 1e-7: # 允许一个小的容差，以防极小的数值变化
                updated_count += 1
                change_magnitude = (current_val - initial_val).norm().item()
                params_changed_info.append(f"  Updated: {name:<55} | Change Mag: {change_magnitude:.2e} | Initial Sum: {initial_val.sum().item():.4f} -> Current Sum: {current_val.sum().item():.4f}")
            else:
                not_updated_count += 1
                params_changed_info.append(f"  NOT Updated: {name:<55} | Diff Sum: {diff:.2e} | Initial Sum: {initial_val.sum().item():.4f} == Current Sum: {current_val.sum().item():.4f}")


    if updated_count > 0:
        print(f"Success: {updated_count} trainable parameter(s) were updated.")
    else:
        print(f"Failure: No trainable parameters were updated significantly (checked {len(initial_param_values)} params).")
    print(f"({not_updated_count} parameters showed no significant change.)")

    print("\nDetailed Parameter Change Info:")
    for info in params_changed_info:
        print(info)

    # 打印更新后的参数状态
    print_param_status(qvit_model, "Final (After Optimizer Step)")

    print("\n--- Test Finished ---")