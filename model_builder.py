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
# from dino import Dinov2FeatureExtractor
# from siglip import SiglipFeatureExtractor
from eva import EvaFeatureExtractor
# from qvit_siglip import QASiglipVisionModel
from qvit_eva import create_qa_eva_vit_g
from instruct_qformer import InstructBlipQFormer
from projector import FeatureMappingFramework, AttentionalFeatureFusionFramework

def set_seed(seed: int):
    """
    Set all relevant random seeds to ensure that the results are reproducible.

    Args:
        seed (int): The seed value to be set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set random seed to {seed}")



def seed_worker(worker_id):
    """
    DataLoader worker initialization function to set the seed inside the worker.
    """
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
        self.dtype = dtype
        print(f"Initializing DualFocusVideoQA on device: {self.device} with dtype: {self.dtype}")

        # --- 1. Load LLM and Tokenizer ---
        print(f"\nLoading LLM and Tokenizer: {llm_model_id}")
        try:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id,use_fast=False)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_model_id,
                torch_dtype=self.dtype,
                attn_implementation="flash_attention_2",
                # device_map="auto",
            )
            self.llm_hidden_dim = self.llm_model.config.hidden_size
            print(f"LLM loaded. Hidden dim: {self.llm_hidden_dim}")

            if self.llm_tokenizer.pad_token is None:
                print("LLM tokenizer does not have a pad token. Setting to eos_token.")
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            self.llm_tokenizer.padding_side = "right"

            self.visual_placeholder = "<video_feat>"
            special_tokens_dict = {'additional_special_tokens': [self.visual_placeholder]}
            num_added_toks = self.llm_tokenizer.add_special_tokens(special_tokens_dict)
            self.video_token_id = self.llm_tokenizer.convert_tokens_to_ids(self.visual_placeholder)
            if num_added_toks > 0:
                self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
                print(f"Added {num_added_toks} special token(s): {self.visual_placeholder}. Resized LLM embeddings.")
            else:
                 print(f"Special token '{self.visual_placeholder}' already exists in tokenizer.")
            self.llm_model.gradient_checkpointing_enable()
            self.max_tokens = 100
            self.temperature = 0.6
            self.top_p = 0.9

        except Exception as e:
            print(f"Error loading LLM: {e}")
            raise

        # --- 2. Load VIT Feature Extractor ---
        print(f"\nLoading VIT feature extractor: {vit_model_id}")
        try:
            if 'eva' in vit_model_id.lower():
                self.vit_feature_extractor = EvaFeatureExtractor(
                    model_path="../models/eva_vit_g.pth",
                    processor_id="../models/vit_image_processor",
                    output_layer_index=dinov2_output_layer,
                    max_frames=max_frames + 10, # Buffer for PE
                    feature_type='patch',
                    embedding_type='num',
                    device=self.device,
                    dtype=self.dtype
                )
                self.vit_hidden_dim = self.vit_feature_extractor.hidden_dim
            # elif 'dino' in vit_model_id.lower():
            #     self.vit_feature_extractor = Dinov2FeatureExtractor(
            #         model_id=vit_model_id,
            #         output_layer_index=dinov2_output_layer,
            #         max_frames=max_frames + 10,
            #         device=self.device,
            #         dtype=self.dtype
            #     )
            #     self.vit_hidden_dim = self.vit_feature_extractor.hidden_dim
            # elif 'siglip' in vit_model_id.lower():
            #      self.vit_feature_extractor = SiglipFeatureExtractor(
            #          model_id=vit_model_id,
            #          output_layer_index=dinov2_output_layer, # 这个参数对Siglip也合适
            #          max_frames=max_frames + 10,
            #          device=self.device,
            #          dtype=self.dtype
            #      )
            #      self.vit_hidden_dim = self.vit_feature_extractor.hidden_dim
            # else:
            #     raise ValueError(f"Unsupported vit_model_id type: {vit_model_id}")

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
            # self.feature_mapper = FeatureMappingFramework(
            #     t_feat_dim=t_input_dim,
            #     s_feat_dim=s_input_dim,
            #     output_dim=self.llm_hidden_dim,
            #     intermediate_proj_dim=feature_map_intermediate_dim
            # )
            self.feature_mapper = AttentionalFeatureFusionFramework(
                t_feat_dim=t_input_dim,
                s_feat_dim=s_input_dim,
                proj_dim=self.llm_hidden_dim,
                num_cycles=2
            )
            print("Feature Mapping Framework initialized.")
        except Exception as e:
            print(f"Error initializing Feature Mapping Framework: {e}")
            raise

        if freeze_llm:
            print("\nFreezing LLM parameters...")
            for param in self.llm_model.parameters():
                param.requires_grad = False
            print("LLM frozen.")
        else:
            print("\nLLM parameters are NOT frozen.")


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
            preprocessed['pixel_values_r_batch'] = torch.empty((0, 3, 224, 224), device=self.device, dtype=self.dtype)
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

        questions_repeated_for_qformer = []
        for i, q in enumerate(questions):
            questions_repeated_for_qformer.extend([q] * r_frame_lens[i])
        preprocessed['questions_repeated_for_qformer'] = questions_repeated_for_qformer

        preprocessed['m_video_frames'] = m_video_frames

        embeds_before_list = []
        embeds_after_list = []
        labels_full_text_list = [] if is_training else None

        for i in range(batch_size):
            question = questions[i]

            if is_training:
                answer = answers[i]
                prompt_template = (
                    "The assistant gives helpful, detailed, and brief answers to the user's questions based on the video.\n\n"
                    "USER: {video_token}\n Based on the video, answer the question: {question}\n"
                    "ASSISTANT: {answer}{eos_token}"
                )
                full_text = prompt_template.format(
                    video_token=self.visual_placeholder,
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

            ids_before_video = input_ids[:, :video_token_index].to(self.device)
            ids_after_video = input_ids[:, video_token_index + 1:].to(self.device)

            with torch.no_grad(): # Text embedding lookup
                embeds_before = word_embeddings(ids_before_video).to(device=self.device, dtype=self.dtype)
                embeds_after = word_embeddings(ids_after_video).to(device=self.device, dtype=self.dtype)

            embeds_before_list.append(embeds_before) # Shape: [1, len_before, D_llm]
            embeds_after_list.append(embeds_after)   # Shape: [1, len_after, D_llm]

            if is_training:
                labels_single = torch.full_like(input_ids, -100, dtype=torch.long)
                answer_prompt = answer + self.llm_tokenizer.eos_token
                answer_encoding = self.llm_tokenizer(answer_prompt, add_special_tokens=False, return_tensors="pt")
                answer_ids = answer_encoding.input_ids
                answer_len = answer_ids.shape[1]

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

        with torch.cuda.amp.autocast(enabled=(self.dtype == torch.float16 or self.dtype == torch.bfloat16)):
            qvit_output_frames = self.qvit_model(
                x=pixel_values_r_batch,  # Shape: [TotalFrames, C, H, W]
                instruct_states=instruct_states_qvit_expanded,  # Shape: [TotalFrames, L_q_instr, D_llm]
                instruct_masks=instruct_masks_qvit_expanded,  # Shape: [TotalFrames, L_q_instr]
            )  # Shape: [TotalFrames, N_patch+1, D_qvit]

        with torch.cuda.amp.autocast(enabled=(self.dtype == torch.float16 or self.dtype == torch.bfloat16)):
            s_feat = self.instruct_qformer(qvit_output_frames, questions_repeated_for_qformer)

        s_feat_list = list(torch.split(s_feat, r_frame_lens, dim=0))
        t_feat_list = []
        with torch.cuda.amp.autocast(enabled=(self.dtype == torch.float16 or self.dtype == torch.bfloat16)):
            for video_frames in m_video_frames:  # Process batch sample by sample
                feat = self.vit_feature_extractor(video_frames,device=self.device)  # Already on self.device  ##sd
                t_feat_list.append(feat)

        p_feat_list = []
        with torch.cuda.amp.autocast(enabled=(self.dtype == torch.float16 or self.dtype == torch.bfloat16)):
            for i in range(batch_size):
                t_feat_sample = t_feat_list[i]  # Shape [N_frames_t, D_vit]
                s_feat_sample = s_feat_list[i]  # Shape [N_frames_s, N_query, D_qformer]
                p_feat_sample = self.feature_mapper(t_feat=t_feat_sample, s_feat=s_feat_sample)

                p_feat_list.append(p_feat_sample)

        assembled_inputs_embeds = []
        assembled_labels = [] if is_training else None
        max_seq_len = 0

        for i in range(batch_size):

            embeds_before = embeds_before_list[i]
            p_feat_single = p_feat_list[i]
            embeds_after = embeds_after_list[i]

            inputs_embeds_single = torch.cat([embeds_before, p_feat_single, embeds_after], dim=1)
            assembled_inputs_embeds.append(inputs_embeds_single)
            current_seq_len = inputs_embeds_single.shape[1]
            max_seq_len = max(max_seq_len, current_seq_len)

            if is_training:
                labels_initial = labels_full_text_list[i]['id']
                answer_len=labels_full_text_list[i]['answer_len']
                labels_single = torch.full((1, current_seq_len), -100, dtype=torch.long, device=self.device)

                labels_single[:,(current_seq_len-answer_len):]=labels_initial[:,(labels_initial.shape[1]-answer_len):]

                assembled_labels.append(labels_single)

        if not assembled_inputs_embeds:
            if is_training:
                outputs["loss"] = torch.tensor(0.0, device=self.device, requires_grad=True)  # Dummy loss
            else:
                outputs["generated_text"] = ["Error: No valid inputs processed."] * batch_size
            return outputs

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

        if is_training:
            with torch.cuda.amp.autocast(enabled=(self.dtype == torch.float16 or self.dtype == torch.bfloat16)):
                llm_outputs = self.llm_model(
                    inputs_embeds=final_inputs_embeds,
                    attention_mask=final_attention_mask,
                    labels=final_labels,
                    return_dict=True,
                )
            outputs["loss"] = llm_outputs.loss
            outputs["logits"] = llm_outputs.logits

        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=(self.dtype == torch.float16 or self.dtype == torch.bfloat16)):
                    terminators = [
                        self.llm_tokenizer.eos_token_id,
                        self.llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]
                    terminators = [t for t in terminators if
                                   isinstance(t, int) and t != self.llm_tokenizer.unk_token_id]
                    if not terminators:
                        terminators = self.llm_tokenizer.eos_token_id

                    generated_ids = self.llm_model.generate(
                        inputs_embeds=final_inputs_embeds,
                        attention_mask=final_attention_mask,
                        max_new_tokens=self.max_tokens,
                        eos_token_id=terminators,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        pad_token_id=self.llm_tokenizer.pad_token_id
                    )

            generated_texts = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            outputs["generated_text"] = generated_texts

        try:
            del pixel_values_r_batch, instruct_states_qvit_expanded, instruct_masks_qvit_expanded
            del qvit_output_frames, s_feat, s_feat_list,t_feat_sample,s_feat_sample,p_feat_sample,feat
            del t_feat_list, p_feat_list,embeds_before,p_feat_single,embeds_after,inputs_embeds_single
            del assembled_inputs_embeds, final_inputs_embeds, final_attention_mask
            if is_training:
                del assembled_labels, final_labels
            else:
                del terminators,generated_ids
        except NameError:
            pass
        torch.cuda.empty_cache()  # Use very cautiously if leaks persist
        return outputs

    def set_generation_params(self,args):
        self.max_tokens=args['max_new_tokens']
        self.temperature=args['temperature']
        self.top_p=args['top_p']

