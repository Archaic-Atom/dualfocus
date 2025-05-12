import os
import torch
from accelerate.utils import save_model_state_dict # For older Accelerate versions, or use accelerator.save_state / torch.save for newer ones.
# Make sure logger is defined, e.g.:
import logging
logger = logging.getLogger(__name__)
# Configure logger if not already configured, e.g., for basic output:
# logging.basicConfig(level=logging.INFO)


def save_trainable_weights(accelerator, model, save_dir, step):
    """Saves trainable weights of specified model components, handling DeepSpeed ZeRO-3."""
    os.makedirs(save_dir, exist_ok=True) # Ensure save_dir exists

    # Wait for all processes to sync before saving. Crucial for distributed training.
    accelerator.wait_for_everyone()

    save_paths = {
        "vit": os.path.join(save_dir, f"vit_step_{step}.pth"),
        "qvit": os.path.join(save_dir, f"qvit_step_{step}.pth"),
        "qformer": os.path.join(save_dir, f"qformer_step_{step}.pth"),
        "mapper": os.path.join(save_dir, f"mapper_step_{step}.pth"),
    }

    # Only the main process should perform the actual saving to disk.
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)

        # Get the full state dict. accelerator.get_state_dict gathers parameters from all ranks.
        # For ZeRO-3, it's recommended to get the state_dict of the original model passed to prepare.
        full_state_dict = accelerator.get_state_dict(model)

        component_configs = {
            "vit": ("vit_feature_extractor", unwrapped_model.vit_feature_extractor if hasattr(unwrapped_model, 'vit_feature_extractor') else None),
            "qvit": ("qvit_model", unwrapped_model.qvit_model if hasattr(unwrapped_model, 'qvit_model') else None),
            "qformer": ("instruct_qformer", unwrapped_model.instruct_qformer if hasattr(unwrapped_model, 'instruct_qformer') else None),
            "mapper": ("feature_mapper", unwrapped_model.feature_mapper if hasattr(unwrapped_model, 'feature_mapper') else None),
        }

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
                            component_trainable_state_dict[name] = full_state_dict[full_param_name].cpu().clone() # Save to CPU
                        else:
                            # Fallback: Sometimes, the attr_name might not be part of the key
                            # if the component is the top-level module or get_state_dict flattens names differently.
                            # This part might need adjustment based on your exact model structure and full_state_dict keys.
                            # For now, we'll log a warning if the prefixed name isn't found.
                            logger.warning(f"Parameter {full_param_name} (derived from {attr_name}.{name}) not found in full_state_dict for component '{key}'. Keys available: {list(full_state_dict.keys())[:5]}...")
                            # As a simpler fallback if prefixes are tricky: if component_module is the main model itself for some reason
                            if name in full_state_dict and attr_name == "": # if component IS the model
                                component_trainable_state_dict[name] = full_state_dict[name].cpu().clone()


                if component_trainable_state_dict:
                    torch.save(component_trainable_state_dict, save_paths[key])
                    logger.info(f"Saved trainable weights for {key} ({len(component_trainable_state_dict)} params) to {save_paths[key]}")
                else:
                    logger.warning(f"No trainable weights found or extracted for component {key} to save to {save_paths[key]}. Check prefixing and requires_grad flags.")
            else:
                logger.warning(f"Component for '{key}' (attribute '{attr_name}') not found in unwrapped_model.")

        logger.info(f"Custom trainable weights saved for step {step} to {save_dir}", main_process_only=False) # Log on main, but info applies to all

    # Ensure all processes wait for the main process to finish saving before proceeding.
    accelerator.wait_for_everyone()


def load_trainable_weights(model, load_dir, step, device='cpu'):
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