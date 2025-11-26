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
from accelerate.utils import set_seed, ProjectConfiguration, DistributedType
from accelerate.utils import DummyOptim, DummyScheduler
from transformers import get_scheduler
import deepspeed
from deepspeed.accelerator import get_accelerator
from model_builder import DualFocusVideoQA
from basedataset import collate_fn,StatefulSampler
from pretrain_videodata import VideoDataset

# torch.autograd.set_detect_anomaly(True)

# --- Setup Logging ---
logger = get_logger(__name__)


def load_config(config_path):
    """Loads YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config, accelerator):
    """Sets up logging for the training process."""
    log_level = logging.INFO
    if accelerator.is_local_main_process:
        log_level = logging.DEBUG

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
        logging.getLogger().addHandler(file_handler)  # Add handler to root logger
        logger.info(f"Logging to file: {log_file}", main_process_only=True)


def save_trainable_weights_2(accelerator, model, train_sampler, save_dir, step):
    save_paths = {
        "vit": os.path.join(save_dir, f"vit_step_{step}.pth"),
        "qvit": os.path.join(save_dir, f"qvit_step_{step}.pth"),
        "qformer": os.path.join(save_dir, f"qformer_step_{step}.pth"),
        "mapper": os.path.join(save_dir, f"mapper_step_{step}.pth"),
    }

    unwrapped_model = accelerator.unwrap_model(model)  # 获取底层模型
    full_state_dict = accelerator.get_state_dict(model)

    component_configs = {
        "vit": ("vit_feature_extractor",
                unwrapped_model.vit_feature_extractor if hasattr(unwrapped_model, 'vit_feature_extractor') else None),
        "qvit": ("qvit_model", unwrapped_model.qvit_model if hasattr(unwrapped_model, 'qvit_model') else None),
        "qformer": (
            "instruct_qformer",
            unwrapped_model.instruct_qformer if hasattr(unwrapped_model, 'instruct_qformer') else None),
        "mapper": (
            "feature_mapper", unwrapped_model.feature_mapper if hasattr(unwrapped_model, 'feature_mapper') else None),
    }
    if accelerator.is_main_process:
        for key, (attr_name, component_module) in component_configs.items():
            if component_module is not None:
                component_trainable_state_dict = {}
                # Iterate through the component's parameters to identify trainable ones
                # and map their names to the full_state_dict keys.
                for name, param in component_module.named_parameters():
                    if param.requires_grad:
                        # Construct the full key as it would appear in the full_state_dict
                        # This assumes standard naming conventions where component names are prefixes.
                        # For DeepSpeed, the unwrapped model might have a 'module.' prefix if it was a DeepSpeedEngine.
                        # accelerator.get_state_dict should handle this.
                        # The keys in full_state_dict will be like "vit_feature_extractor.encoder.layer.0..."
                        full_param_name = f"{attr_name}.{name}"

                        # If the model was wrapped multiple times (e.g. by DDP then DeepSpeed),
                        # the prefix might be more complex or absent if get_state_dict resolved it.
                        # We check for existence and try common variations if needed.
                        # A robust way is to ensure `attr_name` matches the prefix in `full_state_dict`.
                        # For simplicity, we assume `full_state_dict` has keys like `vit_feature_extractor.actual_param_name`.

                        if full_param_name in full_state_dict:
                            component_trainable_state_dict[name] = full_state_dict[
                                full_param_name].cpu().clone()  # Save to CPU
                        else:
                            # Fallback: Sometimes, the attr_name might not be part of the key
                            # if the component is the top-level module or get_state_dict flattens names differently.
                            # This part might need adjustment based on your exact model structure and full_state_dict keys.
                            # For now, we'll log a warning if the prefixed name isn't found.
                            logger.warning(
                                f"Parameter {full_param_name} (derived from {attr_name}.{name}) not found in full_state_dict for component '{key}'. Keys available: {list(full_state_dict.keys())[:5]}...")
                            # As a simpler fallback if prefixes are tricky: if component_module is the main model itself for some reason
                            if name in full_state_dict and attr_name == "":  # if component IS the model
                                component_trainable_state_dict[name] = full_state_dict[name].cpu().clone()

                if component_trainable_state_dict:
                    torch.save(component_trainable_state_dict, save_paths[key])
                    logger.info(
                        f"Saved trainable weights for {key} ({len(component_trainable_state_dict)} params) to {save_paths[key]}")
                else:
                    logger.warning(
                        f"No trainable weights found or extracted for component {key} to save to {save_paths[key]}. Check prefixing and requires_grad flags.")
            else:
                logger.warning(f"Component for '{key}' (attribute '{attr_name}') not found in unwrapped_model.")

        train_checkpoint = {
            'sampler_state': train_sampler.state_dict(),
        }
        torch.save(train_checkpoint, os.path.join(save_dir, f"dataload_step_{step}.pth"))
        logger.info(f"Custom trainable weights saved for step {step} to {save_dir}",
                    main_process_only=False)  # Log on main, but info applies to all


def load_trainable_weights_2(model, train_sampler, load_dir, step, device='cpu'):
    """
    Loads trainable weights (saved component-wise) into the specified model.

    Args:
        model (torch.nn.Module): The model instance to load weights into.
                                 This should be the raw model, before accelerator.prepare().
        load_dir (str): The directory from which to load the weights.
        step (int): The training step number of the checkpoint to load.
        device (str or torch.device): The device to load the weights onto initially.
                                      Defaults to 'cpu'. It's often safer to load to CPU
                                      first and then let Accelerator/DeepSpeed handle
                                      moving the model to the correct training devices.
    """
    logger.info(f"Attempting to load trainable weights for step {step} from directory: {load_dir}")

    # Define paths and corresponding model attributes
    component_configs = {
        "vit": {
            "path": os.path.join(load_dir, f"vit_step_{step}.pth"),
            "attribute_name": "vit_feature_extractor",
        },
        "qvit": {
            "path": os.path.join(load_dir, f"qvit_step_{step}.pth"),
            "attribute_name": "qvit_model",
        },
        "qformer": {
            "path": os.path.join(load_dir, f"qformer_step_{step}.pth"),
            "attribute_name": "instruct_qformer",
        },
        "mapper": {
            "path": os.path.join(load_dir, f"mapper_step_{step}.pth"),
            "attribute_name": "feature_mapper",
        },
    }

    for component_key, config in component_configs.items():
        checkpoint_path = config["path"]
        attr_name = config["attribute_name"]

        if not hasattr(model, attr_name):
            logger.warning(
                f"Model does not have attribute '{attr_name}' for component '{component_key}'. Skipping loading for this component.")
            continue

        component_module = getattr(model, attr_name)

        if component_module is None:  # Should not happen if hasattr is true, but good practice
            logger.warning(f"Model attribute '{attr_name}' for component '{component_key}' is None. Skipping loading.")
            continue

        if os.path.exists(checkpoint_path):
            logger.info(
                f"Loading weights for component '{component_key}' (attribute: {attr_name}) from {checkpoint_path}")
            try:
                # Load the state dict for the specific component
                # map_location ensures weights are loaded to the specified device (e.g., CPU)
                # before the model might be moved to GPUs by Accelerator.
                state_dict_to_load = torch.load(checkpoint_path, map_location=device)

                # Since we only saved trainable weights, there might be missing keys
                # (e.g., for frozen parts of the component or buffers).
                # Thus, strict=False is important.
                missing_keys, unexpected_keys = component_module.load_state_dict(state_dict_to_load, strict=False)

                if missing_keys:
                    logger.warning(f"For component '{component_key}', missing keys while loading: {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"For component '{component_key}', unexpected keys while loading: {unexpected_keys}")

                if not missing_keys and not unexpected_keys:
                    logger.info(
                        f"Successfully loaded weights for component '{component_key}' with no missing/unexpected keys.")
                else:
                    logger.info(
                        f"Loaded weights for component '{component_key}' (check warnings for missing/unexpected keys).")

            except FileNotFoundError:
                logger.error(
                    f"File not found error for {checkpoint_path} (should have been caught by os.path.exists, but being defensive).")
            except Exception as e:
                logger.error(f"Failed to load weights for component '{component_key}' from {checkpoint_path}: {e}",
                             exc_info=True)
        else:
            logger.warning(f"Checkpoint file not found for component '{component_key}': {checkpoint_path}. Skipping.")

    logger.info(f"Finished attempting to load trainable weights for step {step}.")
    CHECKPOINT_PATH=os.path.join(load_dir, f"dataload_step_{step}.pth")
    if os.path.exists(CHECKPOINT_PATH):
        data_checkpoint = torch.load(CHECKPOINT_PATH)
        train_sampler.load_state_dict(data_checkpoint.get('sampler_state'))

    return True, model, train_sampler


def save_trainable_weights(accelerator, model, save_dir, step):
    """Saves trainable weights of specified model components."""
    unwrapped_model = accelerator.unwrap_model(model)

    save_paths = {
        "vit": os.path.join(save_dir, f"vit_step_{step}.pth"),
        "qvit": os.path.join(save_dir, f"qvit_step_{step}.pth"),
        "qformer": os.path.join(save_dir, f"qformer_step_{step}.pth"),
        "mapper": os.path.join(save_dir, f"mapper_step_{step}.pth"),
    }

    # Check if component exists and has the method before calling
    if hasattr(unwrapped_model, 'vit_feature_extractor') and hasattr(unwrapped_model.vit_feature_extractor,
                                                                     'save_trainable_weight'):
        unwrapped_model.vit_feature_extractor.save_trainable_weight(save_paths["vit"])
    else:
        logger.warning("Could not find vit_feature_extractor or its save_trainable_weight method.")

    if hasattr(unwrapped_model, 'qvit_model') and hasattr(unwrapped_model.qvit_model, 'save_trainable_weight'):

        unwrapped_model.qvit_model.save_trainable_weight(save_paths["qvit"])
    else:
        logger.warning("Could not find qvit_model or its save_trainable_weight method.")

    if hasattr(unwrapped_model, 'instruct_qformer') and hasattr(unwrapped_model.instruct_qformer,
                                                                'save_trainable_weight'):
        # Assuming freeze_qformer_base controls this
        unwrapped_model.instruct_qformer.save_trainable_weight(save_paths["qformer"])
    else:
        logger.warning("Could not find instruct_qformer or its save_trainable_weight method.")

    if hasattr(unwrapped_model, 'feature_mapper') and hasattr(unwrapped_model.feature_mapper, 'save_trainable_weight'):
        unwrapped_model.feature_mapper.save_trainable_weight(save_paths["mapper"])
    else:
        logger.warning("Could not find feature_mapper or its save_trainable_weight method.")

    logger.info(f"Saved trainable weights for step {step} to {save_dir}", main_process_only=True)


def load_trainable_weights(unwrapped_model, load_dir, step):
    """Loads the latest trainable weights for specified model components."""
    # unwrapped_model = accelerator.unwrap_model(model) # Get the underlying model

    load_paths = {
        "vit": os.path.join(load_dir, f"vit_step_{step}.pth"),
        "qvit": os.path.join(load_dir, f"qvit_step_{step}.pth"),
        "qformer": os.path.join(load_dir, f"qformer_step_{step}.pth"),
        "mapper": os.path.join(load_dir, f"mapper_step_{step}.pth"),
    }
    loaded_successfully = True

    logger.info(f"Attempting to load trainable weights from step {step} in {load_dir}")

    if hasattr(unwrapped_model, 'vit_feature_extractor') and hasattr(unwrapped_model.vit_feature_extractor,
                                                                     'load_trainable_weight'):
        if os.path.exists(load_paths["vit"]):
            unwrapped_model.vit_feature_extractor.load_trainable_weight(load_paths["vit"])
        elif not unwrapped_model.vit_feature_extractor.is_frozen:  # Only warn if it should have been loaded
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

    if hasattr(unwrapped_model, 'instruct_qformer') and hasattr(unwrapped_model.instruct_qformer,
                                                                'load_trainable_weight'):
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

    return loaded_successfully, unwrapped_model


def get_last_checkpoint_info(experiment_dir):
    """Finds the latest saved step and associated info."""
    state_file = os.path.join(experiment_dir, "training_state.json")
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            logger.info(f"Found training state: {state}")
            last_step = state.get("last_step")
            weights_dir = os.path.join(experiment_dir, "checkpoints")
            expected_files = [
                os.path.join(weights_dir, f"vit_step_{last_step}.pth"),
                os.path.join(weights_dir, f"qvit_step_{last_step}.pth"),
                os.path.join(weights_dir, f"qformer_step_{last_step}.pth"),
                os.path.join(weights_dir, f"mapper_step_{last_step}.pth"),
            ]
            accelerator_state_dir = os.path.join(weights_dir, f"accelerator_step_{last_step}")
            return state, accelerator_state_dir
            # if os.path.isdir(accelerator_state_dir):
            #      return state, accelerator_state_dir
            # else:
            #      logger.warning(f"Accelerator state directory not found for step {last_step}: {accelerator_state_dir}")

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
    step_idx = 0
    max_step = 50

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
                    gathered_losses = accelerator.gather(loss)

                    # 2. Identify NaN and valid (non-NaN) losses
                    is_nan_mask = torch.isnan(gathered_losses)
                    valid_losses = gathered_losses[~is_nan_mask]

                    # 3. Accumulate sum of valid losses
                    if valid_losses.numel() > 0:
                        total_loss += valid_losses.sum().item()

                    # 4. Accumulate count of valid samples
                    total_samples += valid_losses.numel()

                    # 5. Accumulate count of NaN samples
                    current_nan_count = is_nan_mask.sum().item()
                    nan_loss_count += current_nan_count
                else:
                    logger.warning(f"Eval step {step}: Loss not found in model output.")

                step_idx = step_idx + 1
                if step_idx >= max_step:
                    break

            except Exception as e:
                logger.error(f"Error during evaluation step {step}: {e}")
                # Potentially skip batch or re-raise

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    logger.info(f"Evaluation finished. Average Loss: {avg_loss:.4f}", main_process_only=True)
    logger.info(f"NaN losses num: {nan_loss_count}", main_process_only=True)

    model.train()  # Set back to train mode
    return {"eval_loss": avg_loss}


def main(config):

    project_config = ProjectConfiguration(
        project_dir=config['output_dir'],
        logging_dir=os.path.join(config['output_dir'], config['experiment_name'], "logs")
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=config['dataloader'].get('gradient_accumulation_steps', 1),
        # Get from config or default to 1
        log_with="tensorboard",  # Or "wandb", "all", None
        project_config=project_config,
        deepspeed_plugin=None
    )

    # --- Setup Logging (after accelerator init) ---
    setup_logging(config, accelerator)
    logger.info("Starting training script...")
    # logger.info(f"Using DeepSpeed: {accelerator.use_deepspeed}", main_process_only=True)
    logger.info(f"Using DeepSpeed: {accelerator.distributed_type == DistributedType.DEEPSPEED}", main_process_only=True)
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
    if dataset_name == 'VideoInstruction':
        train_dataset = VideoDataset(
            qa_path=config['dataset']['train_json_path'],
            video_dir=config['dataset']['video_dir'],
            num_frames_r=config['dataset']['num_frames_r'],
            num_frames_m=config['dataset']['num_frames_m']
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    logger.info(f"Train dataset size: {len(train_dataset)}")
    train_sampler = StatefulSampler(train_dataset, shuffle=True, seed=config['seed'])

    model_dtype = torch.float32
    if accelerator.mixed_precision == 'fp16':
        model_dtype = torch.float16
    elif accelerator.mixed_precision == 'bf16':
        model_dtype = torch.bfloat16

    logger.info(f"Initializing model with dtype: {model_dtype}")
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
        device=accelerator.device,
        dtype=model_dtype  # Pass dtype if model init supports it
    )

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
            logger.info(
                f"Resuming from epoch {training_state['last_epoch']}, step {training_state['last_step']}")
            _, model, train_sampler = load_trainable_weights_2(model, train_sampler, checkpoints_dir,
                                                                  training_state['last_step'])
            start_epoch = training_state['last_epoch']
            resume_step = training_state['last_step']
            global_step = resume_step
            logger.info(
                f"Successfully resumed training state. Starting from epoch {start_epoch}, step {global_step}.")
        else:
            logger.info("No valid checkpoint found or resume disabled. Starting training from scratch.")

    logger.info("Using DummyOptim as optimizer is defined in DeepSpeed config.")
    optimizer = DummyOptim(model.parameters())

    scheduler = DummyScheduler(optimizer, warmup_num_steps=100)

    # --- Create Dataloaders ---
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['dataloader']['batch_size_per_device'],
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=config['dataloader']['num_workers'],
        shuffle=False,
        batch_sampler=None,
        pin_memory=True  # 通常对性能有好处
    )

    # --- Prepare Components ---
    logger.info("Preparing components with Accelerator...")
    if accelerator.state.deepspeed_plugin is not None:
        logger.info(f"DeepSpeed config being used: {accelerator.state.deepspeed_plugin.deepspeed_config}")
    else:
        logger.warning("DeepSpeed plugin not found in accelerator state!")

    model, optimizer, scheduler, train_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader
    )
    logger.info("Components prepared.")


    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps)
    max_train_steps = config['training']['num_epochs'] * num_update_steps_per_epoch
    logger.info(f"Total training steps: {max_train_steps}")

    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process, desc="Training Steps")
    if resume_step > 0:
        progress_bar.update(resume_step)

    # --- TensorBoard Logging ---
    if accelerator.is_main_process:
        try:
            tracker = accelerator.get_tracker("tensorboard", unwrap=True)
        except Exception as e:
            logger.warning(f"Could not initialize TensorBoard: {e}")
            tracker = None
    else:
        tracker = None

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
            preprocessed_batch = model.data_preprocess(batch)
            with accelerator.accumulate(model):
                outputs = model(preprocessed_batch)
                loss = outputs.get("loss")
                print('loss:', loss.item())

                gathered_losses = accelerator.gather(loss)
                is_nan_mask = torch.isnan(gathered_losses)
                valid_losses = gathered_losses[~is_nan_mask]
                if valid_losses.numel() > 0:
                    avg_loss = valid_losses.mean()
                else:
                    avg_loss = torch.tensor(float('nan'), device=gathered_losses.device,
                                            dtype=gathered_losses.dtype)
                    logger.warning(f"Step {global_step}: All gathered losses were NaN. avg_loss set to NaN.")


                if not torch.isnan(avg_loss):
                    epoch_loss += avg_loss.item() / accelerator.gradient_accumulation_steps
                else:

                    logger.warning(f"Step {global_step}: avg_loss is NaN. Skipping accumulation into epoch_loss.")

                steps_in_epoch += 1
                avg_loss2 = accelerator.gather(loss.repeat(config['dataloader']['batch_size_per_device'])).mean()

                accelerator.backward(loss)

                if accelerator.sync_gradients:

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    global_step += 1
                    progress_bar.update(1)

                    # --- Logging ---
                    if global_step % 5 == 1:
                        lr = optimizer.param_groups[0]['lr']
                        logger.info(
                            f"Epoch {epoch} | Step {global_step}/{max_train_steps} | Loss: {avg_loss.item():.4f}/{avg_loss2.item():.4f} | LR: {lr:.2e}")
                        if tracker:
                            accelerator.log({
                                "train/loss": avg_loss.item(),
                                "train/learning_rate": lr,
                            }, step=global_step)

                    if global_step > 0 and global_step % config['training']['save_steps'] == 0:
                        accelerator.wait_for_everyone()
                        logger.info(f"--- Saving Checkpoint at Step {global_step} ---")
                        save_trainable_weights_2(accelerator, model, train_sampler, checkpoints_dir, global_step)
                        if accelerator.is_main_process:
                            training_state = {
                                "last_epoch": epoch,
                                "last_step": global_step,
                                "max_train_steps": max_train_steps,
                            }
                            state_file = os.path.join(experiment_dir, "training_state.json")
                            with open(state_file, 'w') as f:
                                json.dump(training_state, f, indent=4)
                            logger.info(f"Saved training state to {state_file}")

                        accelerator.wait_for_everyone()
                        get_accelerator().empty_cache()


            if global_step >= max_train_steps:
                logger.info("Maximum training steps reached.")
                break

        logger.info(f"--- Epoch {epoch} Summary ---")
        avg_epoch_loss = epoch_loss / steps_in_epoch if steps_in_epoch > 0 else 0.0
        logger.info(f"Average Training Loss: {avg_epoch_loss:.4f}")
        if tracker:
            accelerator.log({"epoch/train_loss": avg_epoch_loss}, step=epoch)

        if global_step >= max_train_steps:
            break

    # --- End of Training ---
    logger.info("***** Training Finished *****")
    accelerator.wait_for_everyone()
    save_trainable_weights_2(accelerator, model, train_sampler, checkpoints_dir, global_step)
    accelerator.wait_for_everyone()
    accelerator_save_dir = os.path.join(checkpoints_dir, f"accelerator_step_{global_step}_final")
    accelerator.save_state(accelerator_save_dir)
    logger.info(f"Saved final accelerator state to {accelerator_save_dir}")
    if accelerator.is_main_process:
        logger.info("--- Saving Final Checkpoint ---")
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


    logger.info("Script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DualFocusVideoQA Model")
    parser.add_argument("--config", type=str, default="train_config.yaml",
                        help="Path to the training configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)

    main(config)