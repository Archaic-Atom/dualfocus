import os
import yaml
import argparse
import logging
import shutil
import json
from datetime import datetime
import torch
import math
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration,DistributedType
from accelerate.utils import DummyOptim, DummyScheduler # 导入 DummyOptim
from transformers import get_scheduler #虽然ds接管 但accelerator prepare可能需要
import deepspeed # 确保已安装

# --- Import your modules ---
from model_builder2 import DualFocusVideoQA # 假设你的模型代码在 dualfocus_model.py
from basedataset import collate_fn
from msrvttqa import MSRVTTQADataset
# from msvd import MSVDQADataset
# Add imports for the custom save/load methods if needed (e.g., import types)

# torch.autograd.set_detect_anomaly(True)

# --- Setup Logging ---
# 使用 accelerate 的 logger
logger = get_logger(__name__)

# --- 辅助函数 ---
def load_config(config_path):
    """Loads YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(config, accelerator):
    """Sets up logging for the training process."""
    log_level = logging.INFO
    if accelerator.is_local_main_process:
        log_level = logging.DEBUG # 更详细的主进程日志

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=log_level,
    )
    logger.info(f"Accelerator state: {accelerator.state}", main_process_only=True)

    # 在主进程中记录到文件
    if accelerator.is_main_process:
        log_dir = os.path.join(config['output_dir'], config['experiment_name'], "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler) # Add handler to root logger
        logger.info(f"Logging to file: {log_file}", main_process_only=True)

def save_trainable_weights(accelerator,model, save_dir, step):
    """Saves trainable weights of specified model components."""
    unwrapped_model = accelerator.unwrap_model(model) # 获取底层模型

    save_paths = {
        "vit": os.path.join(save_dir, f"vit_step_{step}.pth"),
        "qvit": os.path.join(save_dir, f"qvit_step_{step}.pth"),
        "qformer": os.path.join(save_dir, f"qformer_step_{step}.pth"),
        "mapper": os.path.join(save_dir, f"mapper_step_{step}.pth"),
    }

    # --- 用save_trainable_weight在每个相关组件上调 ---
    # Check if component exists and has the method before calling
    if hasattr(unwrapped_model, 'vit_feature_extractor') and hasattr(unwrapped_model.vit_feature_extractor, 'save_trainable_weight'):
        unwrapped_model.vit_feature_extractor.save_trainable_weight(save_paths["vit"])
    else:
        logger.warning("Could not find vit_feature_extractor or its save_trainable_weight method.")

    if hasattr(unwrapped_model, 'qvit_model') and hasattr(unwrapped_model.qvit_model, 'save_trainable_weight'):
        # 假设 freeze_qvit_base 控制这个;必要时进行调整
        # 如果只有部分被冻结，你可能需要更精细的检查
        # 为简单起见，如果任何部分可能是可训练的，请保存
        unwrapped_model.qvit_model.save_trainable_weight(save_paths["qvit"])
    else:
        logger.warning("Could not find qvit_model or its save_trainable_weight method.")

    if hasattr(unwrapped_model, 'instruct_qformer') and hasattr(unwrapped_model.instruct_qformer, 'save_trainable_weight'):
         # Assuming freeze_qformer_base controls this
        unwrapped_model.instruct_qformer.save_trainable_weight(save_paths["qformer"])
    else:
        logger.warning("Could not find instruct_qformer or its save_trainable_weight method.")

    if hasattr(unwrapped_model, 'feature_mapper') and hasattr(unwrapped_model.feature_mapper, 'save_trainable_weight'):
        unwrapped_model.feature_mapper.save_trainable_weight(save_paths["mapper"])
    else:
        logger.warning("Could not find feature_mapper or its save_trainable_weight method.")

    logger.info(f"Saved trainable weights for step {step} to {save_dir}", main_process_only=True)

def load_trainable_weights(accelerator,model, load_dir, step):
    """Loads the latest trainable weights for specified model components."""
    unwrapped_model = accelerator.unwrap_model(model) # Get the underlying model

    load_paths = {
        "vit": os.path.join(load_dir, f"vit_step_{step}.pth"),
        "qvit": os.path.join(load_dir, f"qvit_step_{step}.pth"),
        "qformer": os.path.join(load_dir, f"qformer_step_{step}.pth"),
        "mapper": os.path.join(load_dir, f"mapper_step_{step}.pth"),
    }
    loaded_successfully = True

    logger.info(f"Attempting to load trainable weights from step {step} in {load_dir}")

    if hasattr(unwrapped_model, 'vit_feature_extractor') and hasattr(unwrapped_model.vit_feature_extractor, 'load_trainable_weight'):
        if os.path.exists(load_paths["vit"]):
            unwrapped_model.vit_feature_extractor.load_trainable_weight(load_paths["vit"])
        elif not unwrapped_model.vit_feature_extractor.is_frozen: # Only warn if it should have been loaded
            logger.warning(f"Trainable weight file not found for VIT: {load_paths['vit']}")
    else:
         logger.warning("Could not find vit_feature_extractor or its load_trainable_weight method.")


    if hasattr(unwrapped_model, 'qvit_model') and hasattr(unwrapped_model.qvit_model, 'load_trainable_weight'):
        if os.path.exists(load_paths["qvit"]):
            unwrapped_model.qvit_model.load_trainable_weight(load_paths["qvit"])
        else:
            logger.warning(f"Trainable weight file not found for QVIT: {load_paths['qvit']}")
    else:
         logger.warning("Could not find qvit_model or its load_trainable_weight method.")


    if hasattr(unwrapped_model, 'instruct_qformer') and hasattr(unwrapped_model.instruct_qformer, 'load_trainable_weight'):
        if os.path.exists(load_paths["qformer"]):
            unwrapped_model.instruct_qformer.load_trainable_weight(load_paths["qformer"])
        else:
            logger.warning(f"Trainable weight file not found for QFormer: {load_paths['qformer']}")
    else:
         logger.warning("Could not find instruct_qformer or its load_trainable_weight method.")


    if hasattr(unwrapped_model, 'feature_mapper') and hasattr(unwrapped_model.feature_mapper, 'load_trainable_weight'):
        if os.path.exists(load_paths["mapper"]):
            unwrapped_model.feature_mapper.load_trainable_weight(load_paths["mapper"])
        else:
            logger.warning(f"Trainable weight file not found for Mapper: {load_paths['mapper']}")
    else:
         logger.warning("Could not find feature_mapper or its load_trainable_weight method.")


    return loaded_successfully

def get_last_checkpoint_info(experiment_dir):
    """Finds the latest saved step and associated info."""
    state_file = os.path.join(experiment_dir, "training_state.json")
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            logger.info(f"Found training state: {state}")
            # 验证 last_step 是否存在相应的 weight 文件
            last_step = state.get("last_step")
            weights_dir = os.path.join(experiment_dir, "checkpoints")
            expected_files = [
                 os.path.join(weights_dir, f"vit_step_{last_step}.pth"),
                 os.path.join(weights_dir, f"qvit_step_{last_step}.pth"),
                 os.path.join(weights_dir, f"qformer_step_{last_step}.pth"),
                 os.path.join(weights_dir, f"mapper_step_{last_step}.pth"),
            ]
            # 检查是否存在，考虑到有些可能被冻结且未保存
            # 如果文件丢失，我们将依赖 load 函数的警告
            # 但我们确实需要加速器状态目录
            accelerator_state_dir = os.path.join(weights_dir, f"accelerator_step_{last_step}")
            if os.path.isdir(accelerator_state_dir):
                 return state, accelerator_state_dir
            else:
                 logger.warning(f"Accelerator state directory not found for step {last_step}: {accelerator_state_dir}")

        except Exception as e:
            logger.error(f"Error reading training state file {state_file}: {e}")
    logger.warning("Could not find valid previous checkpoint state.")
    return None, None


def evaluate(model, dataloader, accelerator):
    """Runs evaluation on the validation set."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    nan_loss_count = 0
    all_predictions = []
    all_references = []
    step_idx=0
    max_step=50

    logger.info("Starting evaluation...")
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            # Move batch to device (Accelerator handles dataloader device placement if prepared)
            # Manual placement might still be needed depending on collate_fn
            # batch = {k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            # ^^ Above is tricky with lists of PIL images. Handle inside model forward.

            # Forward pass for evaluation (loss calculation)
            # If generation is needed, call model without answers and decode outputs
            try:
                 preprocessed_batch = model.data_preprocess(batch)
                 outputs = model(preprocessed_batch)
                 loss = outputs.get("loss")

                 if loss is not None:
                     # 1. Gather the single loss value from each process
                     #    假设 loss 是一个标量张量。gathered_loss 形状通常是 [world_size]
                     gathered_losses = accelerator.gather(loss)

                     # 2. Identify NaN and valid (non-NaN) losses
                     is_nan_mask = torch.isnan(gathered_losses)
                     valid_losses = gathered_losses[~is_nan_mask]  # 选取所有非 NaN 的损失

                     # 3. Accumulate sum of valid losses
                     #    仅当存在有效损失时才进行求和，避免对空张量求和（虽然结果通常是0，但显式检查更安全）
                     if valid_losses.numel() > 0:
                         total_loss += valid_losses.sum().item()  # .item() 获取 Python float

                     # 4. Accumulate count of valid samples
                     total_samples += valid_losses.numel()  # .numel() 直接给出非 NaN 损失的数量

                     # 5. Accumulate count of NaN samples
                     current_nan_count = is_nan_mask.sum().item()  # 对布尔掩码求和 (True=1, False=0) 得到 NaN 的数量
                     nan_loss_count += current_nan_count
                 else:
                      logger.warning(f"Eval step {step}: Loss not found in model output.")

                 step_idx=step_idx+1
                 if step_idx>=max_step:
                     break
                 # --- Optional: Generation Metric Calculation ---
                 # outputs_gen = model(r_video_frames=batch['video_r'], m_video_frames=batch['video_m'], questions=batch['question'], answers=None)
                 # generated_texts = outputs_gen.get("generated_text")
                 # if generated_texts:
                 #     gathered_preds = accelerator.gather_for_metrics(generated_texts) # Check API
                 #     gathered_refs = accelerator.gather_for_metrics(batch['answer']) # Check API
                 #     all_predictions.extend(gathered_preds)
                 #     all_references.extend(gathered_refs)
                 # ---------------------------------------------

            except Exception as e:
                 logger.error(f"Error during evaluation step {step}: {e}")
                 # Potentially skip batch or re-raise

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    logger.info(f"Evaluation finished. Average Loss: {avg_loss:.4f}", main_process_only=True)
    logger.info(f"NaN losses num: {nan_loss_count}", main_process_only=True)

    # --- 可选：计算生成指标 ---
    # 如果 all_predictions 和 all_references 和 accelerator.is_main_process：
    # # 使用 rouge_score、nltk （for bleu）、pycocoevalcap （cider） 等库
    # logger.info（“正在计算生成指标...”）
    # bleu = compute_bleu（all_references， all_predictions）
    # # 胭脂 = compute_rouge（all_references， all_predictions）
    # # logger.info（f“BLEU： {bleu}， ROUGE： {rouge}”） # 示例
    # -----------------------------------------

    model.train() # Set back to train mode
    return {"eval_loss": avg_loss}


# --- 主要训练功能 ---
def main(config):
    # --- Accelerator 初始化 ---
    # ProjectConfiguration 帮助管理日志/检查点的输出目录
    project_config = ProjectConfiguration(
        project_dir=config['output_dir'],
        logging_dir=os.path.join(config['output_dir'], config['experiment_name'], "logs")
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=config['dataloader'].get('gradient_accumulation_steps', 1), # Get from config or default to 1
        log_with="tensorboard", # Or "wandb", "all", None
        project_config=project_config,
        deepspeed_plugin=None # 让 Accelerator 通过配置文件处理
    )

    # --- Setup Logging (after accelerator init) ---
    setup_logging(config, accelerator)
    logger.info("Starting training script...")
    # logger.info(f"Using DeepSpeed: {accelerator.use_deepspeed}", main_process_only=True)
    logger.info(f"Using DeepSpeed: {accelerator.distributed_type == DistributedType.DEEPSPEED }", main_process_only=True)
    logger.info(f"Number of processes: {accelerator.num_processes}", main_process_only=True)
    logger.info(f"Device: {accelerator.device}", main_process_only=True)
    logger.info(f"Mixed precision: {accelerator.mixed_precision}", main_process_only=True)

    # --- Set Seed ---
    if config.get('seed') is not None:
        set_seed(config['seed'])
        logger.info(f"Set random seed to {config['seed']}", main_process_only=True)

    # --- Load Datasets ---
    logger.info("Loading datasets...")
    dataset_name = config['dataset']['name']
    if dataset_name == 'MSRVTT':
        train_dataset = MSRVTTQADataset(
            json_path=config['dataset']['train_json_path'],
            video_dir=config['dataset']['video_dir'],
            num_frames_r=config['dataset']['num_frames_r'],
            num_frames_m=config['dataset']['num_frames_m']
        )
        val_dataset = MSRVTTQADataset(
            json_path=config['dataset']['val_json_path'],
            video_dir=config['dataset']['video_dir'],
            num_frames_r=config['dataset']['num_frames_r'],
            num_frames_m=config['dataset']['num_frames_m']
        )
    # elif dataset_name == 'MSVD':
    #      train_dataset = MSVDQADataset(
    #         json_path=config['dataset']['train_json_path'],
    #         video_dir=config['dataset']['video_dir'],
    #         num_frames_r=config['dataset']['num_frames_r'],
    #         num_frames_m=config['dataset']['num_frames_m']
    #      )
    #      val_dataset = MSVDQADataset(
    #         json_path=config['dataset']['val_json_path'],
    #         video_dir=config['dataset']['video_dir'],
    #         num_frames_r=config['dataset']['num_frames_r'],
    #         num_frames_m=config['dataset']['num_frames_m']
    #      )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    # --- Create Dataloaders ---
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['dataloader']['batch_size_per_device'],
        collate_fn=collate_fn,
        num_workers=config['dataloader']['num_workers'],
        shuffle=True,
        pin_memory=True # 通常对性能有好处
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['dataloader']['batch_size_per_device'],# 使用相同或更小的批量大小进行评估
        collate_fn=collate_fn,
        num_workers=config['dataloader']['num_workers'],
        shuffle=False,
        pin_memory=True
    )

    # --- Instantiate Model ---
    # 显式传递设备？Accelerator 应该可以处理它，但请检查 DualFocusVideoQA 需求
    # 根据 accelerator.mixed_precision 显式传递 dtype？
    model_dtype = torch.float32 # Default
    if accelerator.mixed_precision == 'fp16':
        model_dtype = torch.float16
    elif accelerator.mixed_precision == 'bf16':
        model_dtype = torch.bfloat16

    logger.info(f"Initializing model with dtype: {model_dtype}")
    # 确保 DualFocusVideoQA 的内部 dtype 逻辑对齐或接受此参数
    model = DualFocusVideoQA(
        qvit_model_config=config['model']['qvit_model_config'],
        vit_model_id=config['model']['vit_model_id'],
        llm_model_id=config['model']['llm_model_id'],
        qformer_model_id=config['model']['qformer_model_id'],
        qformer_num_query=config['model']['qformer_num_query'],
        qformer_max_txt_len=config['model']['qformer_max_txt_len'],
        dinov2_output_layer=config['model']['dinov2_output_layer'],
        max_frames=config['model']['max_frames'],
        feature_map_intermediate_dim=config['model']['feature_map_intermediate_dim'],
        freeze_llm=config['model']['freeze_llm'],
        freeze_vit=config['model']['freeze_vit'],
        freeze_qvit_base=config['model']['freeze_qvit_base'],
        freeze_qformer_base=config['model']['freeze_qformer_base'],
        device=accelerator.device, # 让 accelerator/deepspeed 处理放置
        dtype=model_dtype # Pass dtype if model init supports it
    )
    # 如果不是 defs 类的一部分，则动态添加 save/load 方法
    # 导入类型
    # model.vit_feature_extractor.save_trainable_weight = types.MethodType（方法类型save_trainable_weight， model.vit_feature_extractor）
    # model.vit_feature_extractor.load_trainable_weight = types.MethodType（方法类型load_trainable_weight， model.vit_feature_extractor）
    # ...对 qvit_model、instruct_qformer feature_mapper 执行此作 ...
    # 首先检查基类是否已经有这些方法了！

    # --- Optimizer 和 Scheduler（由 DeepSpeed/Accelerator 处理）---
    # 优化器和调度器在 ds_config.json 中定义
    # Accelerator 的 prepare 将处理它们的创建和包装。
    # 如果 DS config 没有明确指定它们，我们可能需要占位符优化器/调度器来准备
    # 但通常 DS config 会处理这个问题。让我们创建占位符以防万一。
    # 为 accelerator.prepare 创建一个虚拟优化器，DS 可能会覆盖
    logger.info("Using DummyOptim as optimizer is defined in DeepSpeed config.")
    # 使用 DummyOptim 代替实际的优化器
    optimizer = DummyOptim(model.parameters())  # 注意这里仍然需要传入 model.parameters()

    # 如果你的 ds_config.json 中也定义了 scheduler，同样使用 DummyScheduler
    scheduler = DummyScheduler(optimizer, warmup_num_steps=100) # 参数可能需要调整

    # --- Prepare Components ---
    logger.info("Preparing components with Accelerator...")
    if accelerator.state.deepspeed_plugin is not None:
        logger.info(f"DeepSpeed config being used: {accelerator.state.deepspeed_plugin.deepspeed_config}")
    else:
        logger.warning("DeepSpeed plugin not found in accelerator state!")
    # Note: DeepSpeed might create its own optimizer/scheduler based on ds_config.json
    # The ones passed here might be ignored or used as base types.
    model, optimizer,scheduler, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer,scheduler, train_dataloader, val_dataloader
        # Pass 调度器（如果你在 ds_config 外部定义了一个）
    )
    # 准备后，优化器可能是 DeepSpeedCPUAdam 或类似对象。
    # Scheduler 也可能被包装。通常通过 optimizer.param_groups[0]['lr'] 访问 LR。
    logger.info("Components prepared.")

    # --- 计算总训练步数 ---
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps)
    max_train_steps = config['training']['num_epochs'] * num_update_steps_per_epoch
    logger.info(f"Total training steps: {max_train_steps}")

    # --- Resume Logic ---
    start_epoch = 0
    global_step = 0
    resume_step = 0
    previous_accelerator_save_dir = None
    experiment_dir = os.path.join(config['output_dir'], config['experiment_name'])
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    if config.get('resume_from_checkpoint', False):
        logger.info(f"Attempting to resume training from {experiment_dir}")
        training_state, accelerator_state_dir = get_last_checkpoint_info(experiment_dir)

        if training_state and accelerator_state_dir:
            try:
                logger.info(f"Resuming from epoch {training_state['last_epoch']}, step {training_state['last_step']}")
                # 1.加载加速器状态（Optimizer、Scheduler、RNG）
                accelerator.load_state(accelerator_state_dir)
                logger.info(f"Loaded accelerator state from {accelerator_state_dir}")

                # 2.加载可训练的模型权重 AFTER prepare 和 AFTER load_state
                if load_trainable_weights(accelerator,model, checkpoints_dir, training_state['last_step']):
                    # 3.更新训练进度变量
                    start_epoch = training_state['last_epoch']
                    # 根据 DeepSpeed/Accelerator 处理步数调整步数
                    # 如果 load_state reset，我们需要根据 epoch 内的 epoch/step 重新计算
                    # 假设 global_step 是优化器步数
                    resume_step = training_state['last_step'] # 加载步骤之后的步数
                    global_step = resume_step # 从这里开始计数
                    logger.info(f"Successfully resumed training state. Starting from epoch {start_epoch}, step {global_step}.")
                else:
                     logger.error("Failed to load trainable model weights. Starting training from scratch.")
                     start_epoch = 0
                     global_step = 0

            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}. Starting training from scratch.")
                start_epoch = 0
                global_step = 0
        else:
            logger.info("No valid checkpoint found or resume disabled. Starting training from scratch.")

    # --- 初始化进度条 ---
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process, desc="Training Steps")
    if resume_step > 0:
        progress_bar.update(resume_step)


    # --- TensorBoard Logging ---
    if accelerator.is_main_process:
        try:
             # Use accelerator's tracker
             tracker = accelerator.get_tracker("tensorboard", unwrap=True)
             # Or initialize manually if needed:
             # from torch.utils.tensorboard import SummaryWriter
             # tb_log_dir = os.path.join(experiment_dir, "tensorboard_logs")
             # writer = SummaryWriter(log_dir=tb_log_dir)
        except Exception as e:
             logger.warning(f"Could not initialize TensorBoard: {e}")
             # writer = None
             tracker = None
    else:
        tracker = None


    # --- 训练循环 ---
    logger.info("***** Starting Training *****")
    logger.info(f"  Num epochs = {config['training']['num_epochs']}")
    logger.info(f"  Instantaneous batch size per device = {config['dataloader']['batch_size_per_device']}")
    # logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}") # 计算
    logger.info(f"  Gradient Accumulation steps = {accelerator.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f"  Starting epoch = {start_epoch}")
    logger.info(f"  Starting step = {global_step}")


    model.train()
    for epoch in range(start_epoch, config['training']['num_epochs']):
        logger.info(f"--- Starting Epoch {epoch}/{config['training']['num_epochs']} ---")
        epoch_loss = 0.0
        steps_in_epoch = 0

        outputs = None
        loss = None
        is_problematic_loss_local_tensor = None  # 使用张量进行 all_reduce

        for step, batch in enumerate(train_dataloader):
            # updata_back = False
            # 如果在 epoch 内恢复，则跳过已处理的步骤
            # 这个逻辑可能很复杂，梯度累积很复杂，更容易从下一个 epoch 的第 0 步继续
            # 如果 global_step 正确反映了优化器步骤，则此循环条件会隐式处理恢复。
            preprocessed_batch = model.data_preprocess(batch)
            with accelerator.accumulate(model):
                # 将批处理移动到设备 - 在模型内部处理 PIL 图像
                # Forward pass
                outputs = model(preprocessed_batch)
                loss = outputs.get("loss")
                # loss_for_backward=loss
                print('loss:',loss)

                # 默认执行反向传播
                perform_backward_local = True
                is_problematic_loss_local = torch.tensor(0, device=accelerator.device)  # 0 for good, 1 for bad

                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    logger.warning(
                        f"Step {global_step}: Loss is problematic (local rank value: {loss}), "
                        f"skipping backward pass and gradient update for this step."
                    )
                    is_problematic_loss_local = torch.tensor(1, device=accelerator.device)
                    perform_backward_local = False  # 本地决定不进行
                    # 如果损失为nan，则跳过梯度更新
                    # loss_for_backward = sum(p.sum() for p in model.parameters() if p.requires_grad ) * 0.0

                # --- NaN 处理逻辑开始 ---
                # 1. 收集所有 rank 的 loss 值 (确保 loss 是标量或单元素张量)
                #    gather 会将每个 rank 的 loss 张量收集到一个列表中（如果 async_gather=True）
                #    或一个拼接后的大张量中（默认 async_gather=False）
                #    我们假设 loss 是一个标量 tensor
                gathered_losses = accelerator.gather(loss)  # gathered_losses 形状通常是 [world_size]
                # 2. 识别并过滤 NaN 值
                is_nan_mask = torch.isnan(gathered_losses)
                valid_losses = gathered_losses[~is_nan_mask]  # 使用布尔掩码选取非 NaN 值
                # 3. 计算平均损失
                if valid_losses.numel() > 0:
                    # 如果存在有效的 (非 NaN) loss 值，则计算它们的平均值
                    avg_loss = valid_losses.mean()
                else:
                    # 如果所有收集到的 loss 都是 NaN
                    avg_loss = torch.tensor(float('nan'), device=gathered_losses.device,
                                         dtype=gathered_losses.dtype)
                    logger.warning(f"Step {global_step}: All gathered losses were NaN. avg_loss set to NaN.")
                # --- NaN 处理逻辑结束 ---

                # 只有在 avg_loss 不是 NaN 的情况下才累加 epoch_loss
                if not torch.isnan(avg_loss):
                    # 注意：config['dataloader']['batch_size_per_device'] 在这里可能不是必需的
                    # avg_loss 已经是所有设备上的平均损失了
                    # epoch_loss 的累加逻辑需要确认是否正确，通常是累加当前step的平均损失
                    epoch_loss += avg_loss.item() / accelerator.gradient_accumulation_steps
                    # 注意：原代码中 loss.repeat(...) 看起来不正确，这里直接用 avg_loss
                else:
                    # 如果 avg_loss 是 NaN，可以选择记录日志但不累加
                    logger.warning(f"Step {global_step}: avg_loss is NaN. Skipping accumulation into epoch_loss.")

                # steps_in_epoch 的增加逻辑保持不变，表示完成了一个计算步骤（即使loss可能是NaN）
                steps_in_epoch += 1  # 计算此 epoch 运行中处理的实际步骤数
                avg_loss2 = accelerator.gather(loss.repeat(config['dataloader']['batch_size_per_device'])).mean()

                # 反向传递
                torch.distributed.all_reduce(is_problematic_loss_local, op=torch.distributed.ReduceOp.MAX)
                # 如果任何 rank 报告了问题，则所有 rank 都跳过
                if is_problematic_loss_local.item() == 1:
                    # accelerator.print(
                    #     f"Rank {accelerator.process_index}: Global decision to skip backward due to problematic loss on at least one rank.")
                    perform_backward_global = False
                else:
                    perform_backward_global = True
                if perform_backward_global:
                    if perform_backward_local:
                        accelerator.backward(loss)
                    else:
                        pass
                else:
                    pass

                # --- 清理逻辑 ---
                # 如果这个rank (或者所有ranks根据全局决定) 将跳过 backward
                if not perform_backward_global:
                    # accelerator.print(
                    #     f"Rank {accelerator.process_index}: Skipping backward pass globally. Explicitly clearing potentially large tensors.")
                    # 在这里，我们知道 backward 不会发生，所以可以安全地删除相关张量
                    # 以释放显存。

                    # 1. 删除模型输出 (outputs)
                    # outputs 通常是一个字典，包含多个张量
                    if outputs is not None:
                        # 如果 outputs 是一个字典
                        if isinstance(outputs, dict):
                            for k in list(
                                    outputs.keys()):  # list(keys()) to avoid RuntimeError: dictionary changed size during iteration
                                tensor_val = outputs.pop(k, None)
                                if torch.is_tensor(tensor_val):
                                    del tensor_val
                        elif torch.is_tensor(outputs):  # 如果 outputs 本身就是张量
                            del outputs
                        outputs = None  # 清除引用

                    # 2. 删除原始损失 (loss)
                    if loss is not None:
                        del loss
                        loss = None  # 清除引用

                    del is_problematic_loss_local_tensor
                    is_problematic_loss_local_tensor = None

                    # 4. 调用 PyTorch 的 CUDA缓存清理 (谨慎使用，可能影响性能，但有助于调试OOM)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        # accelerator.print(f"Rank {accelerator.process_index}: Called torch.cuda.empty_cache().")
                    # 注意: torch.cuda.empty_cache() 只是释放 PyTorch 缓存的但未被占用的显存给操作系统，
                    # 它不会释放那些仍然被张量引用的显存。所以，del 语句更重要


                # print('global_step:',global_step)
                # accelerator.wait_for_everyone()
                # 优化器步骤（仅每 gradient_accumulation_steps发生一次）
                if accelerator.sync_gradients:
                    # 梯度裁剪（由 DeepSpeed 配置处理，如果需要，可以手动进行）
                    # if accelerator.sync_gradients and config.get("max_grad_norm"):
                    #     accelerator.clip_grad_norm_(model.parameters(), config["max_grad_norm"])

                    optimizer.step()
                    scheduler.step() # 调度器步骤 - 检查 DS 是否处理此问题或需要手动调用
                    optimizer.zero_grad()

                    global_step += 1
                    progress_bar.update(1)

                    # --- Logging ---
                    if global_step % 5 == 1: # 减少记录频率
                         lr = optimizer.param_groups[0]['lr']
                         logger.info(f"Epoch {epoch} | Step {global_step}/{max_train_steps} | Loss: {avg_loss.item():.4f}/{avg_loss2.item():.4f} | LR: {lr:.2e}")
                         if tracker: # 记录到 TensorBoard
                             accelerator.log({
                                 "train/loss": avg_loss.item(),
                                 "train/learning_rate": lr,
                             }, step=global_step)


                    # --- 检查点和评估---
                    if global_step > 0 and global_step % config['training']['save_steps'] == 0:
                        accelerator.wait_for_everyone()
                        logger.info(f"--- Saving Checkpoint at Step {global_step} ---")
                        # 1.使用自定义函数保存可训练的权重
                        save_trainable_weights(accelerator, model, checkpoints_dir, global_step)
                        # 2.保存加速器状态 （optimizer、scheduler、rng）
                        current_accelerator_save_dir = os.path.join(checkpoints_dir, f"accelerator_step_{global_step}")
                        accelerator.save_state(current_accelerator_save_dir)
                        logger.info(f"Saved accelerator state to {current_accelerator_save_dir}")
                        # 等待所有进程完成保存操作 ，确保在主进程尝试删除旧目录前，所有进程都已完成写入
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:

                            # 3.保存训练进度状态
                            training_state = {
                                "last_epoch": epoch,
                                "last_step": global_step,
                                "max_train_steps": max_train_steps,
                            }
                            state_file = os.path.join(experiment_dir, "training_state.json")
                            with open(state_file, 'w') as f:
                                json.dump(training_state, f, indent=4)
                            logger.info(f"Saved training state to {state_file}")

                            # 4. 删除上一个 accelerator_state 保存（如果存在）
                            if previous_accelerator_save_dir:  # 确保不是第一次保存
                                if os.path.exists(previous_accelerator_save_dir):
                                    try:
                                        logger.info(
                                            f"Removing previous accelerator state directory: {previous_accelerator_save_dir}")
                                        shutil.rmtree(previous_accelerator_save_dir)
                                        logger.info(f"Successfully removed {previous_accelerator_save_dir}")
                                    except OSError as e:
                                        logger.error(f"Error removing directory {previous_accelerator_save_dir}: {e}")
                                else:
                                    # 如果目录因为某些原因不存在了，记录一个警告
                                    logger.warning(
                                        f"Attempted to remove {previous_accelerator_save_dir}, but it was not found.")

                            # 4. 更新 'previous' 目录为当前刚保存的目录，为下一次清理做准备
                            previous_accelerator_save_dir = current_accelerator_save_dir

                        accelerator.wait_for_everyone()
                        # --- 运行评估 ---
                        if global_step % config['training']['eval_steps'] == 0:
                            logger.info(f"--- Evaluating at Step {global_step} ---")
                            eval_metrics = evaluate(model, val_dataloader, accelerator)
                            if tracker:
                                accelerator.log({"eval/loss": eval_metrics["eval_loss"]}, step=global_step)
                            logger.info(f"Step {global_step} Eval Metrics: {eval_metrics}", main_process_only=True)
                            model.train() # Ensure model is back in training mode


            # 检查是否达到最大步数
            if global_step >= max_train_steps:
                logger.info("Maximum training steps reached.")
                break # 退出内循环

        logger.info(f"--- Epoch {epoch} Summary ---")
        avg_epoch_loss = epoch_loss / steps_in_epoch if steps_in_epoch > 0 else 0.0
        logger.info(f"Average Training Loss: {avg_epoch_loss:.4f}")
        if tracker:
            accelerator.log({"epoch/train_loss": avg_epoch_loss}, step=epoch)


        # 再次检查 epoch 完成后是否达到最大步数
        if global_step >= max_train_steps:
            break # Exit outer loop


    # --- End of Training ---
    logger.info("***** Training Finished *****")
    accelerator.wait_for_everyone()
    # 保存最终的可训练权重
    save_trainable_weights(accelerator, model, checkpoints_dir, global_step)
    # 保存最终的加速器状态
    accelerator_save_dir = os.path.join(checkpoints_dir, f"accelerator_step_{global_step}_final")
    accelerator.save_state(accelerator_save_dir)
    logger.info(f"Saved final accelerator state to {accelerator_save_dir}")
    if accelerator.is_main_process:
        logger.info("--- Saving Final Checkpoint ---")
        # 更新最终训练状态
        training_state = {
            "last_epoch": epoch,
            "last_step": global_step,
             "max_train_steps": max_train_steps,
             "status": "completed"
         }
        state_file = os.path.join(experiment_dir, "training_state.json")
        with open(state_file, 'w') as f:
            json.dump(training_state, f, indent=4)
        logger.info(f"Saved final training state to {state_file}")


    # 如果手动初始化，请清理 TensorBoard writer
    # if writer and accelerator.is_main_process:
    #     writer.close()

    logger.info("Script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DualFocusVideoQA Model")
    parser.add_argument("--config", type=str, default="train_config2.yaml", help="Path to the training configuration file.")
    args = parser.parse_args()

    # Load Base 配置
    config = load_config(args.config)

    # 如果需要，您可以在此处通过命令行覆盖配置值
    # e.g., parser.add_argument("--experiment_name", type=str, default=None)
    # cli_args = parser.parse_args()
    # if cli_args.experiment_name:
    #     config['experiment_name'] = cli_args.experiment_name

    main(config)