import os
import yaml
import argparse
import logging
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
from model_builder import DualFocusVideoQA # 假设你的模型代码在 dualfocus_model.py
from basedataset import collate_fn
from msrvttqa import MSRVTTQADataset
# from msvd import MSVDQADataset
# Add imports for the custom save/load methods if needed (e.g., import types)

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
        if not unwrapped_model.vit_feature_extractor.is_frozen: # Only save if not fully frozen
             unwrapped_model.vit_feature_extractor.save_trainable_weight(save_paths["vit"])
        else:
             logger.debug(f"VIT Feature Extractor is frozen, skipping save_trainable_weight.")
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
            if not unwrapped_model.vit_feature_extractor.load_trainable_weight(load_paths["vit"]):
                loaded_successfully = False
        elif not unwrapped_model.vit_feature_extractor.is_frozen: # Only warn if it should have been loaded
             logger.warning(f"Trainable weight file not found for VIT: {load_paths['vit']}")
    else:
         logger.warning("Could not find vit_feature_extractor or its load_trainable_weight method.")


    if hasattr(unwrapped_model, 'qvit_model') and hasattr(unwrapped_model.qvit_model, 'load_trainable_weight'):
        if os.path.exists(load_paths["qvit"]):
            if not unwrapped_model.qvit_model.load_trainable_weight(load_paths["qvit"]):
                loaded_successfully = False
        else:
             logger.warning(f"Trainable weight file not found for QVIT: {load_paths['qvit']}")
    else:
         logger.warning("Could not find qvit_model or its load_trainable_weight method.")


    if hasattr(unwrapped_model, 'instruct_qformer') and hasattr(unwrapped_model.instruct_qformer, 'load_trainable_weight'):
        if os.path.exists(load_paths["qformer"]):
            if not unwrapped_model.instruct_qformer.load_trainable_weight(load_paths["qformer"]):
                loaded_successfully = False
        else:
             logger.warning(f"Trainable weight file not found for QFormer: {load_paths['qformer']}")
    else:
         logger.warning("Could not find instruct_qformer or its load_trainable_weight method.")


    if hasattr(unwrapped_model, 'feature_mapper') and hasattr(unwrapped_model.feature_mapper, 'load_trainable_weight'):
        if os.path.exists(load_paths["mapper"]):
            if not unwrapped_model.feature_mapper.load_trainable_weight(load_paths["mapper"]):
                loaded_successfully = False
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
    all_predictions = []
    all_references = []

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
                 outputs = model(
                    r_video_frames=batch['video_r'],
                    m_video_frames=batch['video_m'],
                    questions=batch['question'],
                    answers=batch['answer'], # Calculate loss during eval for simplicity
                 )
                 loss = outputs.get("loss")

                 if loss is not None:
                     # Gather loss across processes
                     gathered_loss = accelerator.gather(loss.repeat(config['dataloader']['batch_size_per_device']))
                     total_loss += gathered_loss.sum().item()
                     total_samples += gathered_loss.numel()
                 else:
                      logger.warning(f"Eval step {step}: Loss not found in model output.")

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

        for step, batch in enumerate(train_dataloader):
            # 如果在 epoch 内恢复，则跳过已处理的步骤
            # 这个逻辑可能很复杂，梯度累积很复杂，更容易从下一个 epoch 的第 0 步继续
            # 如果 global_step 正确反映了优化器步骤，则此循环条件会隐式处理恢复。

            with accelerator.accumulate(model):
                # 将批处理移动到设备 - 在模型内部处理 PIL 图像
                # Forward pass
                outputs = model(
                    r_video_frames=batch['video_r'],
                    m_video_frames=batch['video_m'],
                    questions=batch['question'],
                    answers=batch['answer'], # 提供损失计算的答案
                )
                loss = outputs.get("loss")
                print('loss:', loss)

                if loss is None:
                     logger.warning(f"Step {global_step}: Loss is None, skipping backward pass.")
                     continue # 如果丢失，则跳过梯度更新

                avg_loss = accelerator.gather(loss.repeat(config['dataloader']['batch_size_per_device'])).mean()
                epoch_loss += avg_loss.item() / accelerator.gradient_accumulation_steps
                steps_in_epoch += 1 # 计算此 epoch 运行中处理的实际步骤数

                # 反向传递
                accelerator.backward(loss)

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
                    if global_step % 50 == 1: # 减少记录频率
                         lr = optimizer.param_groups[0]['lr']
                         logger.info(f"Epoch {epoch} | Step {global_step}/{max_train_steps} | Loss: {avg_loss.item():.4f} | LR: {lr:.2e}")
                         if tracker: # 记录到 TensorBoard
                             accelerator.log({
                                 "train/loss": avg_loss.item(),
                                 "train/learning_rate": lr,
                             }, step=global_step)


                    # --- 检查点和评估---
                    if global_step > 0 and global_step % config['training']['save_steps'] == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            logger.info(f"--- Saving Checkpoint at Step {global_step} ---")
                            # 1.使用自定义函数保存可训练的权重
                            save_trainable_weights(accelerator,model, checkpoints_dir, global_step)

                            # 2.保存加速器状态 （optimizer、scheduler、rng）
                            accelerator_save_dir = os.path.join(checkpoints_dir, f"accelerator_step_{global_step}")
                            accelerator.save_state(accelerator_save_dir)
                            logger.info(f"Saved accelerator state to {accelerator_save_dir}")


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
    if accelerator.is_main_process:
        logger.info("--- Saving Final Checkpoint ---")
        # 保存最终的可训练权重
        save_trainable_weights(accelerator,model, checkpoints_dir, global_step)
        # 保存最终的加速器状态
        accelerator_save_dir = os.path.join(checkpoints_dir, f"accelerator_step_{global_step}_final")
        accelerator.save_state(accelerator_save_dir)
        logger.info(f"Saved final accelerator state to {accelerator_save_dir}")

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