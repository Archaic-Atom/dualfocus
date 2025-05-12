import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, AutoImageProcessor, AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    SiglipVisionConfig
)
from PIL import Image
import math
from typing import List, Optional, Union, Dict, Any
import warnings

# --- Import previously defined modules ---
# Assume these files exist and contain the respective classes
from dino import Dinov2FeatureExtractor  # Contains Dinov2FeatureExtractor and PositionalEncoding
from siglip import SiglipFeatureExtractor
from qvit_siglip import QASiglipVisionModel  # Contains QASiglipVisionModel and its sub-components
from qvit_eva import create_qa_eva_vit_g
from instruct_qformer import InstructBlipQFormer  # Contains InstructBlipQFormer
from projector import FeatureMappingFramework, FeatureMappingFramework2  # Contains FeatureMappingFramework and MLP

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
                 freeze_llm: bool = True,
                 freeze_vit: bool = True,
                 freeze_qvit_base: bool = True,
                 freeze_qformer_base: bool = True, # Freeze Q-Former BERT part
                 ):
        super().__init__()

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 # 或者根据你的设置
        print(f"Initializing DualFocusVideoQA on device: {self.device} with dtype: {self.dtype}")

        # --- 1. Load LLM and Tokenizer ---
        print(f"\nLoading LLM and Tokenizer: {llm_model_id}")
        try:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id,use_fast=False)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_model_id,
                torch_dtype=self.dtype,
                # attn_implementation="flash_attention_2"
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
                     output_layer_index=dinov2_output_layer, # 确认这个参数对Siglip是否合适
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
                use_checkpoint=False,
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
                qformer_text_input=True
            )
            self.qformer_hidden_dim = self.instruct_qformer.Qformer.config.hidden_size
            print(f"Q-Former initialized. Hidden dim: {self.qformer_hidden_dim}")

            if os.path.exists(qformer_model_id): # 检查是否是文件路径
                 try:
                    self.instruct_qformer.load_checkpoint(qformer_model_id)
                    print(f"Loaded Q-Former checkpoint from: {qformer_model_id}")
                 except Exception as load_err:
                    print(f"Warning: Failed to load Q-Former checkpoint from {qformer_model_id}. Error: {load_err}")
            else:
                print(f"Warning: Q-Former checkpoint path {qformer_model_id} not found. Using initialized Q-Former.")

            if freeze_qformer_base:
                 print("Freezing Q-Former base BERT parameters...")
                 for name, param in self.instruct_qformer.Qformer.named_parameters():
                     # param.requires_grad = False
                     if 'bert' in name.lower(): # 更精确地冻结 BERT 部分 ##sd
                        param.requires_grad = False
                 # 确保 query_tokens 可训练
                 self.instruct_qformer.query_tokens.requires_grad = True
                 print("Q-Former base BERT frozen. Query tokens remain trainable.")
            else:
                 print("Q-Former base BERT parameters are NOT frozen.")

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

        # --- 6. Move models to device and set dtype ---
        self.qvit_model = self.qvit_model.to(self.device, dtype=self.dtype)
        self.instruct_qformer = self.instruct_qformer.to(self.device, dtype=self.dtype)
        self.feature_mapper = self.feature_mapper.to(self.device, dtype=self.dtype)
        # LLM 和 VIT feature extractor 通常在内部或加载时处理 device/dtype
        # 如果 LLM 使用 device_map='auto'，它会自动分配
        # 确保 VIT feature extractor 内部也正确设置了 device 和 dtype
        self.llm_model = self.llm_model.to(self.device) # 如果没有用device_map，需要手动移动

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



    def forward(self,
                r_video_frames: List[List[Image.Image]], # Batch of rough PIL Image lists
                m_video_frames: List[List[Image.Image]], # Batch of meticulous PIL Image lists
                questions: List[str],                   # Batch of question strings
                answers: Optional[List[str]] = None     # Batch of answer strings (for training)
                ) -> Dict[str, Any]:
        """
        Processes a batch of videos and questions for video question answering.

        Args:
            r_video_frames: 粗略视觉路径的视频帧列表批次 (List[List[PIL.Image]])
            m_video_frames: 精细视觉路径的视频帧列表批次 (List[List[PIL.Image]])
            questions: 问题字符串批次 (List[str])
            answers: 答案字符串批次 (Optional[List[str]], 训练时需要)

        Returns:
            包含损失（如果提供了答案）和 logits/outputs 的字典 (Dict[str, Any])
        """
        batch_size = len(questions)
        outputs = {}

        # --- Step 1: 编码问题 (用于 QA-ViT 指令) ---
        # 使用 padding=True 让 tokenizer 处理批次，自动填充到最长问题的长度
        question_tokens = self.llm_tokenizer(
            questions,
            return_tensors="pt",
            padding="max_length", # 填充到批次中最长问题的长度
            truncation=True,
            max_length=32 # 指令长度限制 (可调整)
        ).to(self.device)
        question_token_ids = question_tokens['input_ids'] # Shape: [B, L_q_instr]
        instruct_masks_qvit = question_tokens['attention_mask'] # Shape: [B, L_q_instr]

        with torch.no_grad():
            instruct_states_qvit = self.llm_model.get_input_embeddings()(question_token_ids) # Shape: [B, L_q_instr, D_llm]

        # --- Step 2: 第一视觉路径 (QA-VIT) ---
        all_r_frames = []
        r_frame_lens = [] # 记录每个视频的帧数
        for frames in r_video_frames:
            # 处理每个视频的帧
            processed = [self.qvit_image_processor(img, return_tensors="pt")['pixel_values'].to(device=self.device, dtype=self.dtype)
                         for img in frames]
            all_r_frames.extend(processed)
            r_frame_lens.append(len(frames))

        if not all_r_frames:
            raise ValueError("QA-ViT path received no frames across the entire batch.")

        # 将所有帧合并成一个大批次
        pixel_values_qvit_batch = torch.cat(all_r_frames, dim=0) # Shape: [TotalFrames, C, H, W]

        # 扩展指令嵌入和掩码以匹配总帧数
        # [B, L_q_instr, D_llm] -> [TotalFrames, L_q_instr, D_llm]
        instruct_states_qvit_expanded = torch.cat(
            [instruct_states_qvit[i].repeat(r_frame_lens[i], 1, 1) for i in range(batch_size) if r_frame_lens[i] > 0],
            dim=0
        )
        # [B, L_q_instr] -> [TotalFrames, L_q_instr]
        instruct_masks_qvit_expanded = torch.cat(
            [instruct_masks_qvit[i].repeat(r_frame_lens[i], 1) for i in range(batch_size) if r_frame_lens[i] > 0],
            dim=0
        )

        # 使用 autocast 进行混合精度计算
        with torch.cuda.amp.autocast(enabled=(self.dtype == torch.float16 or self.dtype == torch.bfloat16)):
            # 传递给 QA-ViT
            qvit_output_frames = self.qvit_model(
                x=pixel_values_qvit_batch, # Shape: [TotalFrames, C, H, W]
                instruct_states=instruct_states_qvit_expanded, # Shape: [TotalFrames, L_q_instr, D_llm]
                instruct_masks=instruct_masks_qvit_expanded,   # Shape: [TotalFrames, L_q_instr]
            ) # 输出形状假设是 [TotalFrames, N_patch+1, D_qvit] 或类似

        # --- Step 3: Q-Former 交互 ---
        # 准备 Q-Former 的文本输入：将每个问题重复其对应的帧数次
        questions_repeated_for_qformer = []
        for i, q in enumerate(questions):
            questions_repeated_for_qformer.extend([q] * r_frame_lens[i])

        with torch.cuda.amp.autocast(enabled=(self.dtype == torch.float16 or self.dtype == torch.bfloat16)):
            # QFormer 输入 [TotalFrames, N_patch+1, D_qvit] 和 List[str] (len=TotalFrames)
            s_feat = self.instruct_qformer(qvit_output_frames, questions_repeated_for_qformer)
            # s_feat_frames shape: [TotalFrames, N_query, D_qformer]
        # 按照 r_frame_lens 分割 s_feat
        s_feat_list = list(torch.split(s_feat, r_frame_lens, dim=0))

        # --- Step 4: 第二视觉路径 (VIT) ---
        # *** 假设 self.vit_feature_extractor 已更新以处理批量的 List[List[Image]] ***
        # *** 并输出 [B, D_vit] ***
        m_frame_lens = []
        with torch.cuda.amp.autocast(enabled=(self.dtype == torch.float16 or self.dtype == torch.bfloat16)):
            # t_feat = self.vit_feature_extractor(m_video_frames) # Shape: [B, D_vit]
            # --- 如果 vit_feature_extractor 不能处理批处理，需要循环 ---
            t_feat_list = []
            for video_frames in m_video_frames:
                # 假设 extractor 输入 List[PIL] 输出 [1, D_vit]
                feat = self.vit_feature_extractor(video_frames)
                t_feat_list.append(feat) # feat shape: [1, D_vit]
                m_frame_lens.append(feat.shape[0])
            # t_feat = torch.cat(t_feat_list, dim=0) # Shape: [B, D_vit]
            # --- 结束循环处理 ---

        p_feat_list=[]
        # --- Step 5: 特征映射 ---
        with torch.cuda.amp.autocast(enabled=(self.dtype == torch.float16 or self.dtype == torch.bfloat16)):
            for i in range(batch_size):
                 p_feat = self.feature_mapper(t_feat=t_feat_list[i], s_feat=s_feat_list[i])
                 p_feat_list.append(p_feat)
                 # p_feat shape: [B, N_vis_tokens, D_llm]
                 # N_vis_tokens 由 FeatureMappingFramework 决定

        # --- Step 6 & 7: 准备 LLM 输入并执行 ---

        # 获取 LLM 词嵌入层 (只需一次)
        word_embeddings = self.llm_model.get_input_embeddings()

        all_inputs_embeds = []
        all_labels = []
        all_attention_masks = []
        max_seq_len = 0 # 记录批次中的最大序列长度

        is_training = answers is not None

        for i in range(batch_size):
            question = questions[i]
            p_feat_single = p_feat_list[i]

            if is_training:
                answer = answers[i]
                # 准备训练 prompt
                prompt_template = (
                    # "A chat between a curious user and an artificial intelligence assistant. "
                    "The assistant gives helpful, detailed, and brief answers to the user's questions based on the video.\n\n"
                    "USER: <video_feat>\n Based on the video, answer the question: {question}\n"  # 加入<video_feat>
                    "ASSISTANT: {answer}{eos_token}"
                )
                full_text = prompt_template.format(
                    question=question,
                    answer=answer,
                    eos_token=self.llm_tokenizer.eos_token
                )
            else:
                # 准备推理 prompt
                prompt_template = (
                    # "A chat between a curious user and an artificial intelligence assistant. "
                    "The assistant gives helpful, detailed, and brief answers to the user's questions based on the video.\n\n"
                    "user: <video_feat>\n Based on the video, answer the question: {question}\n"  # 加入<video_feat>
                    "assistant:"
                )
                full_text = prompt_template.format(
                    video_token=self.visual_placeholder,
                    question=question
                ) # 推理时不需要答案和eos

            # Tokenize 完整文本
            # 注意：这里暂时不加 padding/truncation，因为我们要手动拼接
            # encoding = self.llm_tokenizer(full_text, return_tensors="pt", add_special_tokens=False) # 不自动加 BOS/EOS  ##sd
            encoding = self.llm_tokenizer(full_text, return_tensors="pt")
            input_ids = encoding.input_ids.to(self.device) # Shape: [1, SeqLen]

            # 找到 video_token 在 input_ids 中的索引
            video_token_indices = torch.where(input_ids == self.video_token_id)[1]
            if len(video_token_indices) == 0:
                raise ValueError(f"'{self.visual_placeholder}' not found in the tokenized prompt!")
            video_token_index = video_token_indices[0].item()

            # 分割 input_ids
            ids_before_video = input_ids[:, :video_token_index]
            ids_after_video = input_ids[:, video_token_index + 1:]

            # 获取文本部分的嵌入
            with torch.no_grad(): # 文本嵌入通常不需要梯度，除非微调embedding层
                embeds_before = word_embeddings(ids_before_video.to(self.device))  # Shape: [1, len_before, D_llm]
                embeds_after = word_embeddings(ids_after_video.to(self.device))    # Shape: [1, len_after, D_llm]

            # 拼接最终的 inputs_embeds for this sample
            inputs_embeds_single = torch.cat([embeds_before, p_feat_single, embeds_after], dim=1)
            # Shape: [1, len_before + N_vis_tokens + len_after, D_llm]

            current_seq_len = inputs_embeds_single.shape[1]
            max_seq_len = max(max_seq_len, current_seq_len)
            all_inputs_embeds.append(inputs_embeds_single)

            # --- 创建 Labels (仅训练时) ---
            if is_training:
                labels_single = torch.full((1, current_seq_len), -100, dtype=torch.long, device=self.device)
                # 计算答案部分在拼接后序列中的起始索引
                # 答案部分的 token id 在 ids_after_video 中
                # 答案文本（不含开头的 ASSISTANT:）
                answer_prompt = answer + self.llm_tokenizer.eos_token
                answer_encoding = self.llm_tokenizer(answer_prompt, add_special_tokens=False, return_tensors="pt")
                answer_ids = answer_encoding.input_ids.to(self.device) # Shape: [1, len_ans]
                answer_len = answer_ids.shape[1]

                # 答案在最终序列中的起始索引 = 非答案部分长度
                # 非答案部分 = embeds_before + p_feat + embeds_after[:-answer_len]
                answer_start_index_in_final = embeds_before.shape[1] + p_feat_single.shape[1] + (embeds_after.shape[1] - answer_len)
                answer_end_index_in_final = answer_start_index_in_final + answer_len
                labels_single[:, answer_start_index_in_final:answer_end_index_in_final] = answer_ids
                all_labels.append(labels_single)

        # --- 填充批次 ---
        final_inputs_embeds = torch.zeros(batch_size, max_seq_len, self.llm_hidden_dim, dtype=self.dtype, device=self.device)
        final_attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=self.device)
        if is_training:
            final_labels = torch.full((batch_size, max_seq_len), -100, dtype=torch.long, device=self.device)

        for i in range(batch_size):
            seq_len = all_inputs_embeds[i].shape[1]
            # 右填充嵌入和注意力掩码
            final_inputs_embeds[i, :seq_len] = all_inputs_embeds[i][0] # all_inputs_embeds[i] is [1, seq_len, D]
            final_attention_mask[i, :seq_len] = 1
            if is_training:
                final_labels[i, :seq_len] = all_labels[i][0] # all_labels[i] is [1, seq_len]

        # --- LLM 前向传播 / 生成 ---
        if is_training:
            self.train() # 确保模型处于训练模式
            with torch.cuda.amp.autocast(enabled=(self.dtype == torch.float16 or self.dtype == torch.bfloat16)):
                llm_outputs = self.llm_model(
                    inputs_embeds=final_inputs_embeds,
                    attention_mask=final_attention_mask,
                    labels=final_labels,
                    return_dict=True,
                )
            outputs["loss"] = llm_outputs.loss
            outputs["logits"] = llm_outputs.logits # Shape: [B, MaxSeqLen, VocabSize]

        else: # 推理模式 ##sd
            self.eval() # 确保模型处于评估模式
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=(self.dtype == torch.float16 or self.dtype == torch.bfloat16)):
                    # 使用 LLM 的 generate 函数
                    # 需要为 Llama 3.1 等模型设置正确的终止符
                    terminators = [
                         self.llm_tokenizer.eos_token_id,
                         # 尝试获取 Llama 3.1 的 eot_id
                         self.llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]
                    # 过滤掉可能无法转换的 token (例如返回了 unk_token_id) #sd
                    terminators = [t for t in terminators if isinstance(t, int) and t != self.llm_tokenizer.unk_token_id]
                    if not terminators: # 如果都无效，至少用 eos
                        terminators = self.llm_tokenizer.eos_token_id
                        warnings.warn("Could not find specific terminators, using only EOS.")


                    # 使用准备好的 prompt embeddings 进行生成
                    # generate 函数的输入是 prompt 部分
                    generated_ids = self.llm_model.generate(
                        inputs_embeds=final_inputs_embeds, # 包含 prompt 的嵌入
                        attention_mask=final_attention_mask, # prompt 的掩码
                        max_new_tokens=100, # 生成的最大新 token 数
                        eos_token_id=terminators,
                        do_sample=True, # 可以设为 False 进行确定性束搜索
                        temperature=0.6,
                        top_p=0.9,
                        pad_token_id=self.llm_tokenizer.pad_token_id # 必须提供 pad_token_id
                    ) # Output shape: [B, PromptLen + GenLen]

            # 解码生成的 token

            generated_texts = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            outputs["generated_text"] = generated_texts # List[str]

        return outputs

# --- (主函数部分保持不变，但输入需要调整为批次) ---
if __name__=='__main__':
    print("--- Running DualFocusVideoQA Integration Example (Batch Processing) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    MAX_FRAMES_TEST = 3 # 测试用的帧数
    try:
        model = DualFocusVideoQA(
            qvit_model_config=QVIT_CONFIG,
            vit_model_id=VIT_ID,
            llm_model_id=LLM_ID,
            qformer_model_id=QFORMER_ID, # 用于加载检查点或指定bert基础
            qformer_num_query=32,       # 明确指定 QFormer query 数量
            max_frames=MAX_FRAMES_TEST, # 这个参数可能影响内部 buffer，但实际帧数由输入决定
            device=device,
            feature_map_intermediate_dim=None, # 可选，映射器中间维度
            freeze_llm=True,
            freeze_vit=True,           # 假设 FeatureExtractor 内部处理冻结
            freeze_qvit_base=True,     # 冻结QA-ViT的基础部分
            freeze_qformer_base=True   # 冻结Q-Former的BERT部分
        )
        # .to(device) 在 __init__ 中已经处理

    except Exception as e:
        print(f"\nError initializing DualFocusVideoQA model: {e}")
        import traceback
        traceback.print_exc()
        print("Ensure models exist, paths are correct, sufficient RAM/VRAM.")
        exit()

    # --- 创建伪数据 (B=2) ---
    BATCH_SIZE = 2
    # r_dummy_videos: List[List[Image]]
    r_dummy_videos = [
        [Image.new('RGB', (384, 384), color=(100, 50, 50)) for _ in range(MAX_FRAMES_TEST)],
        [Image.new('RGB', (384, 384), color=(50, 100, 50)) for _ in range(MAX_FRAMES_TEST - 1)] # 第二个视频帧数不同
    ]
    # m_dummy_videos: List[List[Image]] (假设 VIT 输入尺寸也是 384x384)
    # **重要**: 这里的尺寸应匹配 VIT 特征提取器的期望输入
    m_dummy_videos = [
        [Image.new('RGB', (384, 384), color=(120, 60, 60)) for _ in range(MAX_FRAMES_TEST)], # 假设 VIT 特征提取器内部会处理帧
        [Image.new('RGB', (384, 384), color=(60, 120, 60)) for _ in range(MAX_FRAMES_TEST -1)]
    ]
    dummy_questions = ["What color dominates the first video?", "Describe the second video content."]
    dummy_answers = ["Reddish gray.", "Greenish gray squares."] # 用于训练损失计算

    # --- 测试训练前向传播 ---
    print(f"\n--- Testing Training Forward Pass (B={BATCH_SIZE}) ---")
    try:
        model.train() # 设置为训练模式（即使部分冻结）
        outputs_train = model(r_video_frames=r_dummy_videos, m_video_frames=m_dummy_videos, questions=dummy_questions, answers=dummy_answers)
        loss = outputs_train.get("loss")
        if loss is not None:
            print(f"Calculated Loss: {loss.item()}")
            # loss.backward() # 取消注释以测试反向传播
            # print("Backward pass successful (if uncommented).")
            # model.zero_grad()
        else:
            print("Loss was not calculated.")
        # print(f"Logits shape: {outputs_train.get('logits').shape}") # 打印 logits 形状

    except Exception as e:
        print(f"Error during training forward pass test: {e}")
        import traceback
        traceback.print_exc()

    # # --- 测试推理前向传播 ---
    # print(f"\n--- Testing Inference Forward Pass (B={BATCH_SIZE}) ---")
    # try:
    #     model.eval() # 设置为评估模式
    #     outputs_infer = model(r_video_frames=r_dummy_videos, m_video_frames=m_dummy_videos, questions=dummy_questions, answers=None) # 无答案
    #     generated_texts = outputs_infer.get("generated_text")
    #     if generated_texts:
    #         print("Generated Texts:")
    #         for i, txt in enumerate(generated_texts):
    #             print(f"  Sample {i}: {txt}")
    #     else:
    #         print("Generated text not found in output.")
    #
    # except Exception as e:
    #     print(f"Error during inference forward pass test: {e}")
    #     import traceback
    #     traceback.print_exc()

    print("\n--- DualFocusVideoQA Integration Example Finished ---")