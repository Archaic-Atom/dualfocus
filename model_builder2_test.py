import os
import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModel, AutoImageProcessor, AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    SiglipVisionConfig
)
from PIL import Image
import math
from typing import List, Optional, Union, Dict, Any
import warnings

# --- Import previously defined modules ---
from basedataset import collate_fn
from msrvttqa import MSRVTTQADataset
# from msvd import MSVDQADataset
# Assume these files exist and contain the respective classes
from dino import Dinov2FeatureExtractor  # Contains Dinov2FeatureExtractor and PositionalEncoding
from siglip import SiglipFeatureExtractor
from qvit_siglip import QASiglipVisionModel  # Contains QASiglipVisionModel and its sub-components
from qvit_eva import create_qa_eva_vit_g
from instruct_qformer import InstructBlipQFormer  # Contains InstructBlipQFormer
from projector import FeatureMappingFramework, FeatureMappingFramework2  # Contains FeatureMappingFramework and MLP

def set_seed(seed: int):
    """
    设置所有相关的随机种子以确保结果可复现。

    Args:
        seed (int): 要设置的种子值。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 设置所有 GPUs 的种子 (如果你有多个 GPU)
    torch.cuda.manual_seed_all(seed)
    # 如果只使用一个特定的 GPU，也可以用:
    # torch.cuda.manual_seed(seed)

    # CuDNN 相关设置
    # 这些设置确保 CuDNN 使用确定性的卷积算法
    # 注意：这可能会降低训练速度
    torch.backends.cudnn.deterministic = True
    # 禁用 CuDNN 的基准测试功能，因为它可能选择非确定性算法
    torch.backends.cudnn.benchmark = False

    # 有时还需要设置环境变量 PYTHONHASHSEED
    # 这通常在脚本启动时设置，而不是在函数内部
    # os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Set random seed to {seed}")


# --- 如何使用 ---

# 1. 在你的脚本最开始调用 set_seed
seed_value = 42 # 选择一个种子值
set_seed(seed_value)

# 2. 如果你使用了 DataLoader 并且 num_workers > 0，需要为 worker 设置种子
#    否则，每个 worker 会有自己独立的、可能基于时间的种子

def seed_worker(worker_id):
    """
    DataLoader worker 的初始化函数，用于设置 worker 内部的种子。
    """
    # 使用主种子和 worker ID 结合生成 worker 特定的种子
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# --- Main DualFocus Model ---
class DualFocusVideoQA(nn.Module):
    def __init__(self,
                 qvit_model_config: dict = {},
                 vit_model_id: str = "facebook/dinov2-base",
                 llm_model_id: str = "mistralai/Mistral-7B-v0.1",
                 qformer_model_id: str = "bert-base-uncased", # Q-Former base
                 qformer_num_query: int = 32,
                 qformer_max_txt_len: int = 128,
                 dinov2_output_layer: int = -1,
                 max_frames: int = 32,
                 feature_map_intermediate_dim: Optional[int] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 dtype=torch.float16,
                 freeze_llm: bool = True,
                 freeze_vit: bool = True,
                 freeze_qvit_base: bool = True,
                 freeze_qformer_base: bool = True, # Freeze Q-Former BERT part
                 ):
        super().__init__()

        self.device = device
        self.dtype = dtype # 或者根据你的设置
        print(f"Initializing DualFocusVideoQA on device: {self.device} with dtype: {self.dtype}")

        # --- 1. Load LLM and Tokenizer ---
        print(f"\nLoading LLM and Tokenizer: {llm_model_id}")
        try:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id,use_fast=False)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_model_id,
                torch_dtype=self.dtype,
                attn_implementation="flash_attention_2",
                # device_map="auto", # 移到模型末尾的 .to(device)
            )
            self.llm_hidden_dim = self.llm_model.config.hidden_size
            print(f"LLM loaded. Hidden dim: {self.llm_hidden_dim}")

            if self.llm_tokenizer.pad_token is None:
                print("LLM tokenizer does not have a pad token. Setting to eos_token.")
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            self.llm_tokenizer.padding_side = "right"

            # --- 注册特殊 Token ---
            # 使用 <video_feat> 作为视觉占位符，这是一个常见的选择
            self.visual_placeholder = "<video_feat>"
            special_tokens_dict = {'additional_special_tokens': [self.visual_placeholder]}
            num_added_toks = self.llm_tokenizer.add_special_tokens(special_tokens_dict)
            # 获取注册后的 ID
            self.video_token_id = self.llm_tokenizer.convert_tokens_to_ids(self.visual_placeholder)
            if num_added_toks > 0:
                self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
                print(f"Added {num_added_toks} special token(s): {self.visual_placeholder}. Resized LLM embeddings.")
            else:
                 print(f"Special token '{self.visual_placeholder}' already exists in tokenizer.")
            self.llm_model.gradient_checkpointing_enable()

        except Exception as e:
            print(f"Error loading LLM: {e}")
            raise

        # --- 2. Load VIT Feature Extractor ---
        print(f"\nLoading VIT feature extractor: {vit_model_id}")
        try:
            # 确保 feature extractor 初始化时传递 device 和 dtype
            if 'dino' in vit_model_id.lower():
                self.vit_feature_extractor = Dinov2FeatureExtractor(
                    model_id=vit_model_id,
                    output_layer_index=dinov2_output_layer,
                    max_frames=max_frames + 10,
                    device=self.device,
                    dtype=self.dtype
                )
                self.vit_hidden_dim = self.vit_feature_extractor.hidden_dim
            elif 'siglip' in vit_model_id.lower():
                 self.vit_feature_extractor = SiglipFeatureExtractor(
                     model_id=vit_model_id,
                     output_layer_index=dinov2_output_layer, # 这个参数对Siglip也合适
                     max_frames=max_frames + 10,
                     device=self.device,
                     dtype=self.dtype
                 )
                 self.vit_hidden_dim = self.vit_feature_extractor.hidden_dim
            else:
                raise ValueError(f"Unsupported vit_model_id type: {vit_model_id}")

            if freeze_vit:
                 print("VIT model is frozen (handled within extractor).")
            else:
                 print("VIT model is NOT frozen.")

        except Exception as e:
            print(f"Error loading VIT feature extractor: {e}")
            raise

        # --- 3. Load QA-ViT ---
        print(f"\nLoading QA-VIT vision model")
        try:
            self.qvit_hidden_dim = qvit_model_config['hidden_dim']
            self.qvit_image_processor = AutoImageProcessor.from_pretrained(qvit_model_config['image_processor_id'])
            self.qvit_model = create_qa_eva_vit_g(
                img_size=qvit_model_config['eva_img_size'],
                drop_path_rate=qvit_model_config['drop_path_rate'],
                use_checkpoint=True,
                precision=self.dtype, # 传递 self.dtype
                instruction_dim=qvit_model_config['instruction_dim'],
                integration_point=qvit_model_config['qa_instruction_point'],
                cached_file=qvit_model_config['cached_file']
            )

            if freeze_qvit_base:
                if hasattr(self.qvit_model, 'freeze_base_model'):
                    self.qvit_model.freeze_base_model()
                    print("QA-ViT base model frozen.")
                else:
                    print("Warning: QA-ViT model does not have 'freeze_base_model' method.")
            else:
                 print("QA-ViT base model parameters are NOT frozen.")

        except Exception as e:
            print(f"Error loading QA-VIT model: {e}")
            raise

        # --- 4. Load InstructBLIP Q-Former ---
        print(f"\nLoading InstructBLIP Q-Former (base: {qformer_model_id})")
        try:
            qformer_vision_width = self.qvit_hidden_dim
            self.instruct_qformer = InstructBlipQFormer(
                num_query_token=qformer_num_query,
                vision_width=qformer_vision_width,
                max_txt_len=qformer_max_txt_len,
                qformer_pretrained_model='bert-base-uncased', # 或者根据你的配置
                qformer_text_input=True,
                freeze_qformer_base=freeze_qformer_base
            )
            self.qformer_hidden_dim = self.instruct_qformer.Qformer.config.hidden_size
            print(f"Q-Former initialized. Hidden dim: {self.qformer_hidden_dim}")

            if os.path.exists(qformer_model_id): # 检查是否是文件路径
                self.instruct_qformer.load_checkpoint(qformer_model_id)
                print(f"Loaded Q-Former checkpoint from: {qformer_model_id}")
            else:
                print(f"Warning: Q-Former checkpoint path {qformer_model_id} not found. Using initialized Q-Former.")


        except Exception as e:
            print(f"Error loading InstructBLIP Q-Former: {e}")
            raise

        # --- 5. Feature Mapping Framework ---
        print("\nInitializing Feature Mapping Framework...")
        try:
            t_input_dim = self.vit_hidden_dim
            s_input_dim = self.qformer_hidden_dim
            self.feature_mapper = FeatureMappingFramework(
                t_feat_dim=t_input_dim,
                s_feat_dim=s_input_dim,
                output_dim=self.llm_hidden_dim,
                intermediate_proj_dim=feature_map_intermediate_dim
            )
            print("Feature Mapping Framework initialized.")
        except Exception as e:
            print(f"Error initializing Feature Mapping Framework: {e}")
            raise

        # # --- 6. Move models to device and set dtype ---
        # self.qvit_model = self.qvit_model.to(self.device, dtype=self.dtype)
        # self.instruct_qformer = self.instruct_qformer.to(self.device, dtype=self.dtype)
        # self.feature_mapper = self.feature_mapper.to(self.device, dtype=self.dtype)
        # # LLM 和 VIT feature extractor 通常在内部或加载时处理 device/dtype
        # # 如果 LLM 使用 device_map='auto'，它会自动分配
        # # 确保 VIT feature extractor 内部也正确设置了 device 和 dtype
        # self.llm_model = self.llm_model.to(self.device) # 如果没有用device_map，需要手动移动

        # --- 7. Freeze LLM ---
        if freeze_llm:
            print("\nFreezing LLM parameters...")
            # self.llm_model.eval() # 不需要在init里设置eval
            for param in self.llm_model.parameters():
                param.requires_grad = False
            print("LLM frozen.")
        else:
            print("\nLLM parameters are NOT frozen.")
            # 可能需要解冻特定层或使用 LoRA


    def data_preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocesses the raw input batch for the DualFocusVideoQA model.

        Handles image processing, tokenization, text embedding lookups,
        and prepares inputs for various components before the main forward pass.

        Args:
            batch: A dictionary containing raw data:
                'video_r': List[List[PIL.Image]] for QA-ViT path.
                'video_m': List[List[PIL.Image]] for ViT path.
                'question': List[str] of questions.
                'answer': Optional[List[str]] of answers (for training).

        Returns:
            A dictionary containing preprocessed tensors and metadata:
            - pixel_values_r_batch: Tensor [TotalFrames, C, H, W] for QA-ViT.
            - r_frame_lens: List[int] frames per video in QA-ViT path.
            - instruct_states_qvit_expanded: Tensor [TotalFrames, L_q_instr, D_llm] question embeds for QA-ViT.
            - instruct_masks_qvit_expanded: Tensor [TotalFrames, L_q_instr] question masks for QA-ViT.
            - questions_repeated_for_qformer: List[str] repeated questions.
            - m_video_frames: Passed through List[List[PIL.Image]].
            - embeds_before_list: List[Tensor [1, len_before, D_llm]] text embeds before placeholder.
            - embeds_after_list: List[Tensor [1, len_after, D_llm]] text embeds after placeholder.
            - labels_full_text_list: List[Tensor [1, SeqLen]] initial labels or None.
            - is_training: bool indicating if answers were provided.
            - batch_size: int.
        """
        r_video_frames = batch['video_r']
        m_video_frames = batch['video_m']
        questions = batch['question']
        answers = batch.get('answer') # Use .get for optional answers
        batch_size = len(questions)
        is_training = answers is not None
        preprocessed = {}

        # --- 1. Prepare QA-ViT Inputs ---
        # --- 1a. Question Instructions ---
        question_tokens = self.llm_tokenizer(
            questions,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=32 # Instruction length limit
        )
        # Move question tokens to the correct device *once*
        question_token_ids = question_tokens['input_ids'].to(self.device)
        instruct_masks_qvit = question_tokens['attention_mask'].to(self.device)

        word_embeddings = self.llm_model.get_input_embeddings()

        # Get instruction embeddings (usually no grad needed for lookup)
        with torch.no_grad():
            # Ensure embeddings are on the correct device and dtype
            instruct_states_qvit = word_embeddings(question_token_ids).to(device=self.device, dtype=self.dtype) # [B, L_q_instr, D_llm]

        # --- 1b. Video Frames ---
        all_r_frames_tensors = []
        r_frame_lens = []
        for frames in r_video_frames:
            if not frames: # Handle empty frame lists for a sample
                r_frame_lens.append(0)
                continue
            # Process frames for this video
            processed_frames = [
                self.qvit_image_processor(img, return_tensors="pt")['pixel_values']
                for img in frames
            ]
            # Stack frames for this video, move to device/dtype together
            video_tensor = torch.cat(processed_frames, dim=0).to(device=self.device, dtype=self.dtype)
            all_r_frames_tensors.append(video_tensor)
            r_frame_lens.append(len(frames))

        if not all_r_frames_tensors:
            # Handle case where the entire batch has 0 frames for QA-ViT
            # This might require special handling in forward pass or raising an error
            warnings.warn("Entire batch has 0 frames for the QA-ViT path.")
            # Set dummy values or handle appropriately later
            preprocessed['pixel_values_r_batch'] = torch.empty((0, 3, QVIT_CONFIG['eva_img_size'], QVIT_CONFIG['eva_img_size']), device=self.device, dtype=self.dtype) # Example shape
            preprocessed['instruct_states_qvit_expanded'] = torch.empty((0, 32, self.llm_hidden_dim), device=self.device, dtype=self.dtype)
            preprocessed['instruct_masks_qvit_expanded'] = torch.empty((0, 32), device=self.device, dtype=torch.long)
        else:
            # Concatenate all frame tensors across the batch
            pixel_values_r_batch = torch.cat(all_r_frames_tensors, dim=0) # [TotalFrames, C, H, W]

            # Expand instructions to match total frames
            instruct_states_qvit_expanded = torch.cat(
                [instruct_states_qvit[i].repeat(r_frame_lens[i], 1, 1) for i in range(batch_size) if r_frame_lens[i] > 0],
                dim=0
            )
            instruct_masks_qvit_expanded = torch.cat(
                [instruct_masks_qvit[i].repeat(r_frame_lens[i], 1) for i in range(batch_size) if r_frame_lens[i] > 0],
                dim=0
            )
            preprocessed['pixel_values_r_batch'] = pixel_values_r_batch
            preprocessed['instruct_states_qvit_expanded'] = instruct_states_qvit_expanded
            preprocessed['instruct_masks_qvit_expanded'] = instruct_masks_qvit_expanded

        preprocessed['r_frame_lens'] = r_frame_lens

        # --- 2. Prepare Q-Former Text Inputs ---
        questions_repeated_for_qformer = []
        for i, q in enumerate(questions):
            questions_repeated_for_qformer.extend([q] * r_frame_lens[i])
        preprocessed['questions_repeated_for_qformer'] = questions_repeated_for_qformer

        # --- 3. Prepare ViT Path Inputs (Pass through PIL list) ---
        preprocessed['m_video_frames'] = m_video_frames # Pass raw list

        # --- 4. Prepare LLM Text Inputs & Initial Labels ---
        embeds_before_list = []
        embeds_after_list = []
        labels_full_text_list = [] if is_training else None

        for i in range(batch_size):
            question = questions[i]

            # --- 4a. Format Prompt ---
            if is_training:
                answer = answers[i]
                prompt_template = (
                    "The assistant gives helpful, detailed, and brief answers to the user's questions based on the video.\n\n"
                    "USER: {video_token}\n Based on the video, answer the question: {question}\n"
                    "ASSISTANT: {answer}{eos_token}"
                )
                full_text = prompt_template.format(
                    video_token=self.visual_placeholder, # Use placeholder
                    question=question,
                    answer=answer,
                    eos_token=self.llm_tokenizer.eos_token
                )
            else:
                prompt_template = (
                    "The assistant gives helpful, detailed, and brief answers to the user's questions based on the video.\n\n"
                    "USER: {video_token}\n Based on the video, answer the question: {question}\n"
                    "ASSISTANT:"
                )
                full_text = prompt_template.format(
                    video_token=self.visual_placeholder, # Use placeholder
                    question=question
                )

            # --- 4b. Tokenize and Find Placeholder ---
            # Tokenize without adding special tokens automatically if template handles BOS/EOS
            encoding = self.llm_tokenizer(full_text, return_tensors="pt", add_special_tokens=True) # Let tokenizer handle BOS/EOS
            input_ids = encoding.input_ids # Shape: [1, SeqLen]
            video_token_indices = torch.where(input_ids == self.video_token_id)[1]

            if len(video_token_indices) == 0:
                 warnings.warn(f"'{self.visual_placeholder}' not found in prompt for sample {i}. Skipping.")
                 # Add empty placeholders or handle error appropriately
                 embeds_before_list.append(torch.empty((1, 0, self.llm_hidden_dim), device=self.device, dtype=self.dtype))
                 embeds_after_list.append(torch.empty((1, 0, self.llm_hidden_dim), device=self.device, dtype=self.dtype))
                 if is_training:
                     labels_full_text_list.append(torch.empty((1, 0), device=self.device, dtype=torch.long))
                 continue # Skip this sample in preprocessing

            video_token_index = video_token_indices[0].item()

            # --- 4c. Split IDs and Get Text Embeddings ---
            ids_before_video = input_ids[:, :video_token_index].to(self.device)
            ids_after_video = input_ids[:, video_token_index + 1:].to(self.device)

            with torch.no_grad(): # Text embedding lookup
                embeds_before = word_embeddings(ids_before_video).to(device=self.device, dtype=self.dtype)
                embeds_after = word_embeddings(ids_after_video).to(device=self.device, dtype=self.dtype)

            embeds_before_list.append(embeds_before) # Shape: [1, len_before, D_llm]
            embeds_after_list.append(embeds_after)   # Shape: [1, len_after, D_llm]

            # --- 4d. Create Initial Labels (Masking everything initially) ---
            if is_training:
                # Labels match the original tokenized sequence length
                labels_single = torch.full_like(input_ids, -100, dtype=torch.long)
                # 计算答案部分在拼接后序列中的起始索引
                # 答案部分的 token id 在 ids_after_video 中
                # 答案文本（不含开头的 ASSISTANT:）
                answer_prompt = answer + self.llm_tokenizer.eos_token
                answer_encoding = self.llm_tokenizer(answer_prompt, add_special_tokens=False, return_tensors="pt")
                answer_ids = answer_encoding.input_ids  # Shape: [1, len_ans]
                answer_len = answer_ids.shape[1]

                # 答案在最终序列中的起始索引 = 非答案部分长度
                # 非答案部分 = embeds_before + p_feat + embeds_after[:-answer_len]
                answer_start_index_in_final = embeds_before.shape[1]+embeds_after.shape[1] - answer_len
                answer_end_index_in_final = answer_start_index_in_final + answer_len
                labels_single[:, answer_start_index_in_final+1:answer_end_index_in_final+1] = answer_ids
                labels_full_text_list.append({'id':labels_single,'answer_len':answer_len})

        preprocessed['embeds_before_list'] = embeds_before_list
        preprocessed['embeds_after_list'] = embeds_after_list
        preprocessed['labels_full_text_list'] = labels_full_text_list
        preprocessed['is_training'] = is_training
        preprocessed['batch_size'] = batch_size

        return preprocessed

    def forward(self, preprocessed_batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a preprocessed batch for video question answering.

        Args:
            preprocessed_batch: Dictionary containing tensors and metadata
                                from the `data_preprocess` method.

        Returns:
            A dictionary containing loss (if training) or generated text (if inferring),
            and potentially logits.
        """
        # Unpack preprocessed data
        pixel_values_r_batch = preprocessed_batch['pixel_values_r_batch']
        r_frame_lens = preprocessed_batch['r_frame_lens']
        instruct_states_qvit_expanded = preprocessed_batch['instruct_states_qvit_expanded']
        instruct_masks_qvit_expanded = preprocessed_batch['instruct_masks_qvit_expanded']
        questions_repeated_for_qformer = preprocessed_batch['questions_repeated_for_qformer']
        m_video_frames = preprocessed_batch['m_video_frames']  # Raw PIL list
        embeds_before_list = preprocessed_batch['embeds_before_list']
        embeds_after_list = preprocessed_batch['embeds_after_list']
        labels_full_text_list = preprocessed_batch['labels_full_text_list']  # Initial labels
        is_training = preprocessed_batch['is_training']
        batch_size = preprocessed_batch['batch_size']

        outputs = {}

        # --- Step 1: QA-ViT Path ---
        with torch.cuda.amp.autocast(enabled=(self.dtype == torch.float16 or self.dtype == torch.bfloat16)):
            qvit_output_frames = self.qvit_model(
                x=pixel_values_r_batch,  # Shape: [TotalFrames, C, H, W]
                instruct_states=instruct_states_qvit_expanded,  # Shape: [TotalFrames, L_q_instr, D_llm]
                instruct_masks=instruct_masks_qvit_expanded,  # Shape: [TotalFrames, L_q_instr]
            )  # Shape: [TotalFrames, N_patch+1, D_qvit]

        # --- Step 2: Q-Former Interaction ---
        with torch.cuda.amp.autocast(enabled=(self.dtype == torch.float16 or self.dtype == torch.bfloat16)):
            # QFormer input [TotalFrames, N_patch+1, D_qvit] and List[str] (len=TotalFrames)
            s_feat = self.instruct_qformer(qvit_output_frames, questions_repeated_for_qformer)
            # s_feat_frames shape: [TotalFrames, N_query, D_qformer]

        # Split s_feat according to original video lengths in the batch
        # Handle cases where r_frame_lens contains zeros
        s_feat_list = list(torch.split(s_feat, r_frame_lens, dim=0))
        # valid_s_feat_parts = torch.split(s_feat, [l for l in r_frame_lens if l > 0], dim=0)
        # s_feat_list = []
        # valid_idx = 0
        # for length in r_frame_lens:
        #     s_feat_list.append(valid_s_feat_parts[valid_idx])
        #     valid_idx += 1

        # --- Step 3: ViT Path ---
        t_feat_list = []
        with torch.cuda.amp.autocast(enabled=(self.dtype == torch.float16 or self.dtype == torch.bfloat16)):
            for video_frames in m_video_frames:  # Process batch sample by sample
                # Feature extractor expects List[PIL], runs model, returns tensor on self.device
                feat = self.vit_feature_extractor(video_frames,device=self.device)  # Already on self.device  ##sd
                t_feat_list.append(feat)  # feat shape: [1 or N_frames', D_vit] - Mapper needs to handle this
                # Shape depends on extractor's reduction strategy

        # --- Step 4: Feature Mapping ---
        p_feat_list = []  # List to store projected features per sample
        with torch.cuda.amp.autocast(enabled=(self.dtype == torch.float16 or self.dtype == torch.bfloat16)):
            for i in range(batch_size):
                # Ensure t_feat and s_feat are valid before passing to mapper
                t_feat_sample = t_feat_list[i]  # Shape [N_frames_t, D_vit]
                s_feat_sample = s_feat_list[i]  # Shape [N_frames_s, N_query, D_qformer]

                # Mapper expects t_feat=[B(=1), Nt, Dt], s_feat=[B(=1), Ns, Nq, Ds] or similar
                # Adjust inputs if necessary based on FeatureMappingFramework requirements
                # Example: if mapper needs batch dim 1
                # t_feat_sample = t_feat_sample.unsqueeze(0) # If needed
                # s_feat_sample = s_feat_sample.unsqueeze(0) # If needed
                p_feat_sample = self.feature_mapper(t_feat=t_feat_sample, s_feat=s_feat_sample)
                # Assume p_feat_sample shape: [1, N_vis_tokens, D_llm]

                p_feat_list.append(p_feat_sample)

        # --- Step 5: Assemble LLM Inputs and Pad ---
        assembled_inputs_embeds = []
        assembled_labels = [] if is_training else None
        max_seq_len = 0

        for i in range(batch_size):

            embeds_before = embeds_before_list[i]  # [1, len_before, D_llm]
            p_feat_single = p_feat_list[i]  # [1, N_vis_tokens, D_llm]
            embeds_after = embeds_after_list[i]  # [1, len_after, D_llm]

            # Concatenate embeddings
            inputs_embeds_single = torch.cat([embeds_before, p_feat_single, embeds_after], dim=1)
            # Shape: [1, len_before + N_vis_tokens + len_after, D_llm]
            assembled_inputs_embeds.append(inputs_embeds_single)
            current_seq_len = inputs_embeds_single.shape[1]
            max_seq_len = max(max_seq_len, current_seq_len)

            if is_training:
                labels_initial = labels_full_text_list[i]['id']  # [1, OriginalSeqLen]
                answer_len=labels_full_text_list[i]['answer_len']
                # Create new labels tensor matching the assembled length
                labels_single = torch.full((1, current_seq_len), -100, dtype=torch.long, device=self.device)

                labels_single[:,(current_seq_len-answer_len):]=labels_initial[:,(labels_initial.shape[1]-answer_len):]

                assembled_labels.append(labels_single)

        # Pad the batch
        # Ensure there's something to pad
        if not assembled_inputs_embeds:
            if is_training:
                outputs["loss"] = torch.tensor(0.0, device=self.device, requires_grad=True)  # Dummy loss
            else:
                outputs["generated_text"] = ["Error: No valid inputs processed."] * batch_size
            return outputs  # Early exit if batch is empty after filtering

        final_inputs_embeds = torch.zeros(len(assembled_inputs_embeds), max_seq_len, self.llm_hidden_dim,
                                          dtype=self.dtype, device=self.device)
        final_attention_mask = torch.zeros(len(assembled_inputs_embeds), max_seq_len, dtype=torch.long,
                                           device=self.device)
        if is_training:
            final_labels = torch.full((len(assembled_inputs_embeds), max_seq_len), -100, dtype=torch.long,
                                      device=self.device)

        for i in range(len(assembled_inputs_embeds)):
            seq_len = assembled_inputs_embeds[i].shape[1]
            final_inputs_embeds[i, :seq_len] = assembled_inputs_embeds[i][0]
            final_attention_mask[i, :seq_len] = 1
            if is_training:
                final_labels[i, :seq_len] = assembled_labels[i][0]

        # --- Step 6: LLM Forward / Generation ---
        if is_training:
            self.train()  # Ensure train mode
            with torch.cuda.amp.autocast(enabled=(self.dtype == torch.float16 or self.dtype == torch.bfloat16)):
                llm_outputs = self.llm_model(
                    inputs_embeds=final_inputs_embeds,
                    attention_mask=final_attention_mask,
                    labels=final_labels,
                    return_dict=True,
                )
            outputs["loss"] = llm_outputs.loss
            outputs["logits"] = llm_outputs.logits  # Optional

        else:  # Inference
            self.eval()  # Ensure eval mode
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=(self.dtype == torch.float16 or self.dtype == torch.bfloat16)):
                    # Find appropriate terminators
                    terminators = [
                        self.llm_tokenizer.eos_token_id,
                        self.llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")  # Example for Llama 3.1
                    ]
                    terminators = [t for t in terminators if
                                   isinstance(t, int) and t != self.llm_tokenizer.unk_token_id]
                    if not terminators:
                        terminators = self.llm_tokenizer.eos_token_id

                    generated_ids = self.llm_model.generate(
                        inputs_embeds=final_inputs_embeds,
                        attention_mask=final_attention_mask,
                        max_new_tokens=100,
                        eos_token_id=terminators,
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.9,
                        pad_token_id=self.llm_tokenizer.pad_token_id
                    )

            # Decode only the generated part
            # Input length is final_inputs_embeds.shape[1] (padded length)
            # Need actual prompt length used for generation, which is variable per sample.
            # It's safer to decode full and then parse, or pass prompt_input_ids to generate.
            # For simplicity here, decode all:
            generated_texts = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            outputs["generated_text"] = generated_texts  # List[str]

        # --- Cleanup (Optional but recommended for debugging leaks) ---
        try:
            del pixel_values_r_batch, instruct_states_qvit_expanded, instruct_masks_qvit_expanded
            del qvit_output_frames, s_feat, s_feat_list
            del t_feat_list, p_feat_list
            del assembled_inputs_embeds, final_inputs_embeds, final_attention_mask
            if is_training:
                del assembled_labels, final_labels
            torch.cuda.empty_cache() # Use very cautiously if leaks persist
        except NameError:
            pass  # Ignore if a variable wasn't created due to empty input etc.

        return outputs

# --- (主函数部分保持不变，但输入需要调整为批次) ---
if __name__=='__main__':
    print("--- Running DualFocusVideoQA Integration Example (Batch Processing) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.cuda.set_device(device)

    # device = torch.device("cpu")

    # --- 配置 (与之前相同) ---
    # 确保路径正确，模型文件存在
    QVIT_CONFIG = {
        "qa_instruction_point": 'late',
        'instruction_dim': 4096, # 应由 LLM hidden dim 决定，在 init 中设置
        'hidden_dim': 1408,      # QA-ViT output dim
        'eva_img_size': 224,
        'drop_path_rate': 0.0,
        'image_processor_id': "../models/vit_image_processor", # 确保这个 processor 和你的 QA-ViT 匹配
        'cached_file': "../models/eva_vit_g.pth" # QA-ViT 预训练权重路径
    }
    # 使用 SigLIP 作为第二视觉路径的特征提取器
    VIT_ID = "../models/siglip" # SigLIP 模型路径
    LLM_ID = "../models/vicuna" # Vicuna 模型路径
    # InstructBLIP Q-Former 检查点路径 (确保与使用的 bert-base 兼容)
    QFORMER_ID = "../models/instruct_blip_vicuna7b_trimmed.pth" # 或 'bert-base-uncased' 如果不加载检查点


    # --- 实例化模型 ---
    try:
        model = DualFocusVideoQA(
            qvit_model_config=QVIT_CONFIG,
            vit_model_id=VIT_ID,
            llm_model_id=LLM_ID,
            qformer_model_id=QFORMER_ID, # 用于加载检查点或指定bert基础
            qformer_num_query=32,       # 明确指定 QFormer query 数量
            max_frames=32, # 这个参数可能影响内部 buffer，但实际帧数由输入决定
            device=device,
            feature_map_intermediate_dim=None, # 可选，映射器中间维度
            freeze_llm=True,
            freeze_vit=True,           # 假设 FeatureExtractor 内部处理冻结
            freeze_qvit_base=True,     # 冻结QA-ViT的基础部分
            freeze_qformer_base=True   # 冻结Q-Former的BERT部分
        ).to(device)
        # .to(device) 在 __init__ 中已经处理

    except Exception as e:
        print(f"\nError initializing DualFocusVideoQA model: {e}")
        import traceback
        traceback.print_exc()
        print("Ensure models exist, paths are correct, sufficient RAM/VRAM.")
        exit()

    # # 统计可训练和冻结的参数数量
    # trainable_params = 0
    # frozen_params = 0
    # total_params = 0
    #
    # print("\n--- 参数可训练性检查 ---")
    # print(f"{'参数名称':<80} {'是否可训练':<10} {'参数数量':<15}")
    # print("-" * 110)
    #
    # # 存储可训练和冻结参数的名称（可选，如果参数量太大可能不适合打印）
    # trainable_param_names = []
    # frozen_param_names = []
    #
    # for name, param in model.named_parameters():
    #     num_elements = param.numel()  # 获取参数包含的元素数量
    #     total_params += num_elements
    #     is_trainable = param.requires_grad
    #
    #     if is_trainable:
    #         trainable_params += num_elements
    #         trainable_param_names.append(name)
    #         status = "是 (True)"
    #     else:
    #         frozen_params += num_elements
    #         frozen_param_names.append(name)
    #         status = "否 (False)"
    #
    #     # 打印每个参数的信息（如果参数过多，可以注释掉这行）
    #     # print(f"{name:<80} {status:<10} {num_elements:<15}")
    #
    # print("-" * 110)
    # print("\n--- 统计结果 ---")
    # print(f"总参数数量: {total_params}")
    # print(f"可训练参数数量: {trainable_params}")
    # print(f"冻结参数数量: {frozen_params}")
    #
    # if total_params > 0:
    #     trainable_percent = 100 * trainable_params / total_params
    #     frozen_percent = 100 * frozen_params / total_params
    #     print(f"可训练参数占比: {trainable_percent:.2f}%")
    #     print(f"冻结参数占比: {frozen_percent:.2f}%")
    # else:
    #     print("模型没有参数。")
    #
    # # (可选) 打印部分可训练/冻结参数的名称以供抽查
    # print("\n--- 部分可训练参数名称示例 (最多显示10个) ---")
    # for name in trainable_param_names[:10]:
    #     print(name)
    # if len(trainable_param_names) > 10:
    #     print("...")
    #
    # print("\n--- 部分冻结参数名称示例 (最多显示10个) ---")
    # for name in frozen_param_names[:10]:
    #     print(name)
    # if len(frozen_param_names) > 10:
    #     print("...")
    #
    # print("\n--- 检查完毕 ---")

    DATASET='MSRVTT'
    TRAIN_JSON_PATH = '../video_data/MSRVTT/MSRVTT-QA/val_qa.json'  # Or val_qa.json, test_qa.json
    VIDEO_DIR = '../video_data/MSRVTT/MSRVTT/MSRVTT_Videos'  # Path to videoXXXX.mp4 files
    NUM_FRAMES_R = 4
    NUM_FRAMES_M = 4
    BATCH_SIZE = 1
    if DATASET=='MSRVTT':
        test_dataset = MSRVTTQADataset(
            json_path=TRAIN_JSON_PATH,
            video_dir=VIDEO_DIR,
            num_frames_r=NUM_FRAMES_R,
            num_frames_m=NUM_FRAMES_M,
            transform=None  # Assuming HF processors will handle transforms later
        )
    # elif DATASET=='MSVD':
    #     test_dataset = MSVDQADataset(
    #         json_path=TRAIN_JSON_PATH,
    #         video_dir=VIDEO_DIR,
    #         num_frames_r=NUM_FRAMES_R,
    #         num_frames_m=NUM_FRAMES_M,
    #         transform=None  # Assuming HF processors will handle transforms later
    #     )

    dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # Set True for training
        num_workers=2,  # Adjust based on system
        worker_init_fn=seed_worker,
        collate_fn=collate_fn  # Use the custom collate function
    )
    num_batches_to_test = 2
    for i, batch in enumerate(dataloader):
        print("--- Testing Refactored MSRVTTQADataset ---")

        # --- 测试训练前向传播 ---
        print(f"\n--- Testing Training Forward Pass (B={BATCH_SIZE}) ---")
        if i==2:
            print(1)
        preprocessed_batch = model.data_preprocess(batch)
        model.train() # 设置为训练模式（即使部分冻结）
        outputs_train = model(preprocessed_batch)
        loss = outputs_train.get("loss")
        if loss is not None:
            # 首先检查 loss 是否为 NaN
            if torch.isnan(loss):
                print('is in:',i)
                print("数据异常：计算出的 Loss 为 NaN！")
                # 这里可以添加更多的调试信息，比如打印相关的输入或模型参数
                # print(f"NaN Loss Tensor: {loss}") # 查看具体的 NaN 张量

            # 其次检查 loss 是否为无穷大 (inf)，这也是一个常见的问题
            elif torch.isinf(loss):
                print('is in:', i)
                print(f"数据异常：计算出的 Loss 为 Infinite ({loss.item()})！")
                # print(f"Infinite Loss Tensor: {loss}")

            # 如果 loss 既不是 NaN 也不是 Inf，那么它是一个有效的数值
            else:
                print(f"Calculated Loss: {loss.item()}")  # 使用 .item() 获取 Python 标量值
                # --- 原有的反向传播逻辑 ---
                # try:
                #     loss.backward() # 取消注释以测试反向传播
                #     print("Backward pass successful (if uncommented).")
                #     # model.zero_grad() # 通常在 optimizer.step() 之前或之后调用
                # except RuntimeError as e:
                #     print(f"Error during backward pass: {e}")
                #     # 在这里可以打印更多关于梯度的信息来帮助调试
        else:
            print("Loss was not calculated.")
            # print(f"Logits shape: {outputs_train.get('logits').shape}") # 打印 logits 形状

        # # --- 测试推理前向传播 ---
        # print(f"\n--- Testing Inference Forward Pass (B={BATCH_SIZE}) ---")
        # batch_infer = batch.copy()
        # if 'answer' in batch_infer:
        #     del batch_infer['answer']  # Remove answers for inference preprocessing
        # preprocessed_batch_infer = model.data_preprocess(batch_infer)
        # model.eval() # 设置为评估模式
        # outputs_infer = model(preprocessed_batch_infer) # 无答案
        # generated_texts = outputs_infer.get("generated_text")
        # if generated_texts:
        #     print("Generated Texts:")
        #     for i, txt in enumerate(generated_texts):
        #         print(f"  Sample {i}: {txt}")
        # else:
        #     print("Generated text not found in output.")


    print("\n--- DualFocusVideoQA Integration Example Finished ---")