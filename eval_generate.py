import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
import torch
import numpy as np
import random
import yaml
import argparse
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

from basedataset import collate_fn
from msrvttqa import MSRVTTQADataset
from msvdqa import MSVDQADataset
from activitynetqa import ActivityNetQADataset
from benchmarkqa import BenchmarkQADataset

from model_builder import DualFocusVideoQA, set_seed, seed_worker

from accelerate import Accelerator, dispatch_model
from accelerate.utils import set_seed as accelerate_set_seed


def load_trainable_weights(model, weight_path, step):
    print((f"Attempting to load trainable weights for step {step} from directory: {weight_path}"))
    component_configs = {
        "vit": {
            "path": os.path.join(weight_path, f"vit_step_{step}.pth"),
            "attribute_name": "vit_feature_extractor",
        },
        "qvit": {
            "path": os.path.join(weight_path, f"qvit_step_{step}.pth"),
            "attribute_name": "qvit_model",
        },
        "qformer": {
            "path": os.path.join(weight_path, f"qformer_step_{step}.pth"),
            "attribute_name": "instruct_qformer",
        },
        "mapper": {
            "path": os.path.join(weight_path, f"mapper_step_{step}.pth"),
            "attribute_name": "feature_mapper",
        },
    }
    for component_key, config in component_configs.items():
        checkpoint_path = config["path"]
        attr_name = config["attribute_name"]

        if hasattr(model, 'module'):
            component_module = getattr(model.module, attr_name)
        else:
            component_module = getattr(model, attr_name)

        if os.path.exists(checkpoint_path):
            state_dict_to_load = torch.load(checkpoint_path, map_location='cpu')

            # strict=False
            missing_keys, unexpected_keys = component_module.load_state_dict(state_dict_to_load, strict=False)

            if missing_keys:
                print(f"For component '{component_key}', missing keys while loading: {missing_keys}")
            if unexpected_keys:
                print(f"For component '{component_key}', unexpected keys while loading: {unexpected_keys}")
        else:
            print(f"Warning: Checkpoint path for component {component_key} not found: {checkpoint_path}")

    print(f"Finished attempting to load trainable weights for step {step}.")
    return model


def load_model_for_evaluation(config_model_init, config_load_trained, accelerator: Accelerator):
    print("Initializing DualFocusVideoQA model for evaluation...")

    dtype_str = config_model_init.get('dtype', 'float16')
    if dtype_str == "float16":
        dtype = torch.float16
    elif dtype_str == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    config_model_init['dtype'] = dtype

    model = DualFocusVideoQA(
        qvit_model_config=config_model_init['qvit_model_config'],
        vit_model_id=config_model_init['vit_model_id'],
        llm_model_id=config_model_init['llm_model_id'],
        qformer_model_id=config_model_init['qformer_model_id'],
        qformer_num_query=config_model_init['qformer_num_query'],
        max_frames=config_model_init['max_frames'],
        feature_map_intermediate_dim=config_model_init.get('feature_map_intermediate_dim'),
        device='cpu',
        dtype=config_model_init['dtype'],
        freeze_llm=config_model_init.get('freeze_llm', True),
        freeze_vit=config_model_init.get('freeze_vit', True),
        freeze_qvit_base=config_model_init.get('freeze_qvit_base', True),
        freeze_qformer_base=config_model_init.get('freeze_qformer_base', True)
    )
    print("Model structure initialized on CPU.")


    if config_load_trained.get('load_trained_components', False):
        load_dir = config_load_trained.get('trained_components_dir')
        load_dir = os.path.join(load_dir, "checkpoints")
        step = config_load_trained.get('trained_components_step')
        if load_dir and os.path.exists(load_dir) and step is not None:
            model = load_trainable_weights(model, load_dir, step)
        else:
            print(
                f"Warning: Trained components directory '{load_dir}' for step '{step}' not valid or step not specified. Skipping weight loading.")


    print("Dispatching model across available devices using Accelerate...")

    device_map_strategy = gen_config.get('device_map_strategy', "custom")  # 从配置读取策略
    print(f"Using device_map_strategy: {device_map_strategy}")
    if device_map_strategy=='custom':
        custom_device_map = {
            "llm_model": 1,
            "vit_feature_extractor": 0,
            "qvit_model": 0,
            "instruct_qformer": 0,
            "feature_mapper": 0
        }
        model = dispatch_model(model, device_map=custom_device_map)
    else:
        raise 'device_map_strategy error...'

    model.eval()
    print("Model dispatched and set to evaluation mode.")
    return model


def main(config_path):
    accelerator = Accelerator()
    print(
        f"Accelerator initialized. Device: {accelerator.device}, Num processes: {accelerator.num_processes}, Distributed type: {accelerator.distributed_type}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    global gen_config
    gen_config = config['generator_settings']
    common_config = config


    accelerate_set_seed(common_config.get('seed', 42))


    model = load_model_for_evaluation(
        gen_config['model_init_params'],
        gen_config.get('load_trained_components_config', {}),
        accelerator
    )

    if hasattr(model, 'set_generation_params'):
        model.set_generation_params(gen_config.get('llm_generation_params', {}))
    else:
        llm_gen_params = gen_config.get('llm_generation_params', {})

    data_loader_cfg = gen_config['data_loader']
    if data_loader_cfg['dataset_name'] == 'MSRVTT':
        dataset = MSRVTTQADataset(
            json_path=data_loader_cfg['val_json_path'],
            video_dir=data_loader_cfg['video_dir'],
            num_frames_r=data_loader_cfg['num_frames_r'],
            num_frames_m=data_loader_cfg['num_frames_m'],
            transform=None
        )
    elif data_loader_cfg['dataset_name'] == 'MSVD':
        dataset = MSVDQADataset(
            annotation_path=data_loader_cfg['val_json_path'],
            video_dir=data_loader_cfg['video_dir'],
            mapping_path=data_loader_cfg['mapping_path'],
            num_frames_r=data_loader_cfg['num_frames_r'],
            num_frames_m=data_loader_cfg['num_frames_m']
        )
    elif data_loader_cfg['dataset_name'] == 'Activity':
        dataset = ActivityNetQADataset(
            q_path=data_loader_cfg['q_path'],
            a_path=data_loader_cfg['a_path'],
            video_dir=data_loader_cfg['video_dir'],
            num_frames_r=data_loader_cfg['num_frames_r'],
            num_frames_m=data_loader_cfg['num_frames_m'],
            transform=None
        )
    elif data_loader_cfg['dataset_name'] == 'Benchmark':
        dataset = BenchmarkQADataset(
            annotation_path=data_loader_cfg['annotation_path'],
            video_dir=data_loader_cfg['video_dir'],
            num_frames_r=data_loader_cfg['num_frames_r'],
            num_frames_m=data_loader_cfg['num_frames_m'],
            transform=None
        )
    else:
        raise ValueError(f"Unsupported dataset: {data_loader_cfg['dataset_name']}")

    dataloader = DataLoader(
        dataset,
        batch_size=data_loader_cfg['batch_size'],
        shuffle=False,
        num_workers=data_loader_cfg['num_workers'],
        worker_init_fn=seed_worker,
        collate_fn=collate_fn
    )


    output_dir = os.path.join(common_config['output_base_dir'], "generated_preds")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, gen_config['generation_output']['filename'])

    results = []
    print(f"Starting generation. Results will be saved to: {output_path}")
    max_data_len = gen_config.get('max_data_len', -1)


    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Generating Predictions"):

            ground_truth_answers = batch['answer']
            batch['answer'] = None

            preprocessed_batch = model.data_preprocess(batch)


            outputs = model(preprocessed_batch)

            del preprocessed_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            generated_texts = outputs.get("generated_text")

            questions = batch['question']
            video_ids = batch.get('video_id', [f"video_{batch_idx * data_loader_cfg['batch_size'] + i}" for i in
                                               range(len(questions))])
            question_ids = batch.get('question_id', [f"q_{batch_idx * data_loader_cfg['batch_size'] + i}" for i in
                                                     range(len(questions))])

            if generated_texts:
                for i in range(len(generated_texts)):
                    results.append({
                        "video_id": video_ids[i],
                        "question_id": question_ids[i],
                        "question": questions[i],
                        "answer": ground_truth_answers[i] if ground_truth_answers and i < len(
                            ground_truth_answers) else "N/A",
                        "pred": generated_texts[i].strip()
                    })
            else:
                print(f"Warning: No generated text for batch {batch_idx}")

            if (batch_idx + 1) % 50 == 0:

                with open(output_path, 'w', encoding='utf-8') as f:
                    for res_item in results:
                        f.write(json.dumps(res_item) + '\n')
                print(f"Saved intermediate results at batch {batch_idx + 1}")

            if max_data_len != -1 and batch_idx >= (max_data_len / data_loader_cfg['batch_size'] - 1):
                print(f"Reached max_data_len ({max_data_len} items). Stopping.")
                break


    with open(output_path, 'w', encoding='utf-8') as f:
        for res_item in results:
            f.write(json.dumps(res_item) + '\n')
    print(f"Generation complete. All results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate model predictions for video QA.")
    parser.add_argument(
        "--config",
        type=str,
        default="eval_config.yaml",
        help="Path to the evaluation configuration YAML file."
    )
    args = parser.parse_args()
    main(args.config)