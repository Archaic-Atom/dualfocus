import torch
import torch.nn as nn
import os
from transformers import BertTokenizer, BertConfig
# Assuming Qformer.py contains the BertLMHeadModel implementation provided
from Qformer import BertLMHeadModel
import logging

# "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth"
# Basic LayerNorm implementation (as used in blip2.py)
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class InstructBlipQFormer(nn.Module):
    """
    Decoupled Q-Former module for InstructBLIP, focusing on instruction-aware
    visual feature extraction.

    Takes image embeddings and instruction text as input, and outputs
    instruction-conditioned visual features (query outputs).
    """
    @classmethod
    def init_tokenizer(cls, truncation_side="right"):
        """Initializes the BERT tokenizer used for processing instructions."""
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        # Note: InstructBLIP might not strictly need [DEC] for Q-Former input processing,
        # but retaining it for consistency with Blip2Base if loading checkpoints.
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def __init__(
        self,
        num_query_token=32,
        vision_width=1408, # Example width for ViT-G/14
        cross_attention_freq=2,
        embed_dim=256, # Optional: for potential projection layer definition later
        max_txt_len=128, # Max length for instruction text tokenization
        qformer_pretrained_model="bert-base-uncased", # Base model for Q-Former
        qformer_text_input=True, # Crucial for InstructBLIP
        freeze_qformer_base=True
    ):
        super().__init__()

        self.qformer_text_input = qformer_text_input
        self.tokenizer = self.init_tokenizer()
        self.max_txt_len = max_txt_len
        self.num_query_token = num_query_token

        # --- Initialize Q-Former Core ---
        encoder_config = BertConfig.from_pretrained(qformer_pretrained_model)
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token

        # Load Q-Former model structure
        # Using BertLMHeadModel as per blip2.py and blip2_qformer.py
        # We might only use the 'bert' part for feature extraction.
        # self.Qformer = BertLMHeadModel.from_pretrained(
        #     qformer_pretrained_model, config=encoder_config
        # )
        self.Qformer = BertLMHeadModel(config=encoder_config)

        # --- Initialize Query Tokens ---
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        # --- Adapt Q-Former for Text Input (Instruction) ---
        if self.qformer_text_input:
             # Resize token embeddings if instruction text is used as input
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
            logging.info(f"Resized Q-Former token embeddings to {len(self.tokenizer)}")
        else:
            # If not using text input (like original BLIP-2 stage 1),
            # certain components might be disabled (though InstructBLIP needs it).
            logging.warning("qformer_text_input is False. InstructBLIP typically requires text input to Q-Former.")
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None

        # Remove the language model head (cls) if only using for feature extraction
        self.Qformer.cls = None
        logging.info("Removed Q-Former's LM head (if any).")
        if freeze_qformer_base:
            print("Freezing Q-Former base BERT parameters...")
            for name, param in self.Qformer.named_parameters():
                # param.requires_grad = False
                if 'bert' in name.lower():  # 更精确地冻结 BERT 部分 ##sd
                    param.requires_grad = False
            # 确保 query_tokens 可训练
            self.query_tokens.requires_grad = True
            print("Q-Former base BERT frozen. Query tokens remain trainable.")
        else:
            print("Q-Former base BERT parameters are NOT frozen.")


    def forward(self, image_embeds: torch.Tensor, instruction: list[str]):
        """
        Forward pass for instruction-aware visual feature extraction.

        Args:
            image_embeds (torch.Tensor): Visual features from the image encoder
                                          (e.g., ViT output), after LayerNorm.
                                          Shape: (batch_size, num_patches, vision_width)
            instruction (list[str]): A list of instruction strings, one per image.
                                      Length: batch_size

        Returns:
            torch.Tensor: Instruction-aware query outputs (visual features).
                          Shape: (batch_size, num_query_token, hidden_size)
        """
        batch_size = image_embeds.shape[0]
        device = image_embeds.device

        # --- Prepare Inputs for Q-Former ---
        # Expand query tokens to match batch size
        query_tokens = self.query_tokens.expand(batch_size, -1, -1).to(device)

        # Attention mask for image features
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)

        if self.qformer_text_input:
            # Tokenize the instruction text
            text_Qformer = self.tokenizer(
                instruction,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(device)

            # Attention mask for query tokens
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=device)

            # Concatenate query attention mask and text attention mask
            # This mask defines which elements (query tokens + text tokens)
            # attend to each other in self-attention and which attend
            # to the image in cross-attention.
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

            # --- Run Q-Former ---
            # The core of InstructBLIP's Q-Former operation:
            # - text_Qformer.input_ids: Instruction tokens act as part of the input sequence.
            # - query_embeds: Learnable query tokens also act as part of the input sequence.
            # - attention_mask (Qformer_atts): Combined mask for self-attention.
            # - encoder_hidden_states (image_embeds): Visual features for cross-attention.
            # - encoder_attention_mask (image_atts): Mask for visual features.
            query_output = self.Qformer.bert(
                input_ids=text_Qformer.input_ids, # Instruction is input here
                attention_mask=Qformer_atts,      # Combined mask
                query_embeds=query_tokens,        # Queries are input here
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            # Fallback for Q-Former without text input (not typical for InstructBLIP)
             query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        # --- Extract Instruction-Aware Visual Features ---
        # The output hidden states corresponding to the initial query tokens
        # are considered the instruction-aware visual features.
        instruct_visual_features = query_output.last_hidden_state[:, :self.num_query_token, :]

        return instruct_visual_features

    def load_checkpoint(self, filename):
        """Loads a checkpoint, filtering for Q-Former and query_tokens weights."""
        if filename is None or not filename:
             logging.warning("No checkpoint filename provided for QFormer.")
             return
        checkpoint = torch.load(filename, map_location="cpu")
        state_dict = checkpoint["model"]

        # Filter state dict for QFormer weights and query_tokens
        qformer_weights = {}
        for k, v in state_dict.items():
            if k.startswith("Qformer.") or k.startswith("query_tokens"):
                qformer_weights[k] = v

        msg = self.load_state_dict(qformer_weights, strict=False)
        logging.info(f"Loaded Q-Former checkpoint from {filename}")
        logging.info(f"Q-Former Load Msg: {msg}")

    # --- 新增函数 ---
    def save_trainable_weight(self, save_path):
        """
        保存模型中可训练的参数到指定路径。
        仅保存 requires_grad = True 的参数。

        Args:
            save_path (str): 保存权重的文件路径。
        """
        trainable_state_dict = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_state_dict[name] = param.data  # 保存参数的数据部分

        if not trainable_state_dict:
            logging.warning("在模型中没有找到可训练的参数！无法保存。")
            return

        try:
            # 确保目录存在
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)

            torch.save(trainable_state_dict, save_path)
            logging.info(f"模型的可训练参数已成功保存到: {save_path}")
            print(f"模型的可训练参数已成功保存到: {save_path}")  # 添加打印信息
            print(f"共保存 {len(trainable_state_dict)} 个可训练参数张量。")
        except Exception as e:
            logging.error(f"保存可训练参数到 {save_path} 时出错: {e}")
            print(f"保存可训练参数到 {save_path} 时出错: {e}")  # 添加打印信息

    def load_trainable_weight(self, load_path):
        """
        从指定路径加载可训练的参数。
        只会加载文件中存在的、且在当前模型中也存在的参数。
        使用 strict=False 来允许只加载部分权重。

        Args:
            load_path (str): 加载权重的文件路径。
        """
        if not os.path.exists(load_path):
            logging.error(f"权重文件未找到: {load_path}")
            print(f"错误：权重文件未找到: {load_path}")  # 添加打印信息
            return

        try:
            # 加载保存的、仅包含可训练参数的状态字典
            trainable_state_dict = torch.load(load_path, map_location='cpu')

            # 获取当前模型状态字典，检查哪些参数是可训练的
            current_trainable_params = {name for name, param in self.named_parameters() if param.requires_grad}

            # 过滤加载的 state_dict，确保只加载到当前模型中实际可训练的参数上
            # （虽然 load_state_dict(strict=False) 也能处理，但这提供了一层额外的明确性）
            filtered_state_dict = {}
            loaded_keys = set(trainable_state_dict.keys())
            keys_to_load_actually = []

            for name, param_data in trainable_state_dict.items():
                if name in self.state_dict():  # 检查参数是否存在于当前模型
                    if name in current_trainable_params:  # 检查参数在当前模型中是否可训练
                        filtered_state_dict[name] = param_data
                        keys_to_load_actually.append(name)
                    else:
                        logging.warning(f"跳过加载参数 '{name}' 因为它在当前模型中被冻结 (requires_grad=False)。")
                else:
                    logging.warning(f"跳过加载参数 '{name}' 因为它不在当前模型结构中。")

            if not filtered_state_dict:
                logging.warning(f"文件 {load_path} 中没有找到与当前模型可训练参数匹配的权重。")
                print(f"警告：文件 {load_path} 中没有找到与当前模型可训练参数匹配的权重。")
                return

            # 使用 load_state_dict 加载过滤后的权重，strict=False 允许多余/缺失键
            # 因为我们已经过滤了，理论上应该都能匹配上 filtered_state_dict 中的键
            msg = self.load_state_dict(filtered_state_dict, strict=False)

            logging.info(f"从 {load_path} 加载可训练参数完成。")
            print(f"从 {load_path} 加载可训练参数完成。")  # 添加打印信息
            print(f"成功加载了 {len(keys_to_load_actually)} / {len(loaded_keys)} 个来自文件的参数张量。")

            # 打印加载状态信息，对于调试很有用
            # logging.info(f"Load Status: {msg}")
            # if msg.missing_keys:
            #     # 这些应该是被冻结的参数，或者是不在 filtered_state_dict 中的参数
            #     logging.warning(
            #         f"Load state dict report - Missing Keys (expected for frozen params): {msg.missing_keys}")
            if msg.unexpected_keys:
                # 这不应该发生，因为我们是基于加载的文件来构建 filtered_state_dict 的
                logging.error(f"Load state dict report - Unexpected Keys (issue): {msg.unexpected_keys}")
                print(f"错误：加载时遇到意外的键: {msg.unexpected_keys}")

        except FileNotFoundError:
            logging.error(f"权重文件未找到: {load_path}")
            print(f"错误：权重文件未找到: {load_path}")  # 添加打印信息
        except Exception as e:
            logging.error(f"加载可训练参数从 {load_path} 时出错: {e}")
            print(f"加载可训练参数从 {load_path} 时出错: {e}")  # 添加打印信息


# --- Example Usage (Conceptual) ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cpu")

    # Dummy inputs
    B = 2
    FRAME=8
    IMG_SIZE = 224
    PATCH_SIZE = 14
    NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
    VISION_WIDTH = 1408 # Example for ViT-L
    NUM_QUERY_TOKENS = 32
    QFORMER_HIDDEN = 768 # BERT-base hidden size

    # Simulate output from Vision Encoder + LayerNorm
    dummy_image_embeds = torch.randn(B, NUM_PATCHES+1, VISION_WIDTH)

    # Example instructions
    instructions = [
        "Describe the main object in the image.",
        "What color is the car?"
    ]

    # Initialize the decoupled Q-Former
    instruct_qformer = InstructBlipQFormer(
        num_query_token=NUM_QUERY_TOKENS,
        vision_width=VISION_WIDTH,
        # embed_dim=256, # Not needed within the QFormer itself
        max_txt_len=128,
        qformer_pretrained_model="bert-base-uncased" # Or path to specific checkpoint
    ).to(device)

    # --- Load pretrained weights (Important for performance) ---
    # Example: Load parts from a BLIP-2 or InstructBLIP checkpoint if available
    instruct_qformer.load_checkpoint("../models/instruct_blip_vicuna7b_trimmed.pth") # Replace with actual path


    # --- Freeze parts if needed (as done in InstructBLIP finetuning) ---
    # During InstructBLIP finetuning, only QFormer is trained.
    # Here, we assume the module itself might be fine-tuned.
    # If used within a larger model, you'd freeze Vision Encoder and LLM outside this module.
    instruct_qformer.train() # Set to train mode


    # Get instruction-aware visual features
    with torch.amp.autocast('cuda',dtype=torch.float16): # Use autocast like in the original code
         instruct_visual_features = instruct_qformer(dummy_image_embeds.to(device), instructions)

    print("Decoupled InstructBLIP Q-Former executed.")
    print("Output shape:", instruct_visual_features.shape) # Expected: [B, NUM_QUERY_TOKENS, QFORMER_HIDDEN]

    # --- Next Steps (Outside this module in full InstructBLIP) ---
    # 1. Project `instruct_visual_features` using `llm_proj` layer.
    # 2. Feed the projected features as soft prompts to the frozen LLM.
    # 3. Tokenize the instruction again (using LLM's tokenizer) and feed to LLM.
    # 4. Generate text output using the LLM.

    # # --- 演示保存和加载可训练权重 ---
    # print("\n--- 测试保存和加载可训练权重 ---")
    # trainable_params_list = [name for name, param in instruct_qformer.named_parameters() if param.requires_grad]
    # print(f"当前模型可训练参数 ({len(trainable_params_list)} 个): {trainable_params_list}")
    # # 当 freeze_qformer_base=True 时，应该只包含 'query_tokens' 和可能的 Qformer 交叉注意力层参数等
    #
    # save_file = "./test/instruct_qformer_trainable_weights.pth"  # 临时保存路径
    #
    # # 1. 保存可训练权重
    # print(f"\n保存可训练权重到 {save_file}...")
    # instruct_qformer.save_trainable_weight(save_file)
    #
    # # 可选: 验证保存 - 尝试修改可训练参数，然后加载回来
    # original_query_tokens_sum = 0
    # if instruct_qformer.query_tokens.requires_grad:
    #     original_query_tokens_sum = instruct_qformer.query_tokens.data.sum().item()
    #     print(f"修改前 query_tokens sum: {original_query_tokens_sum}")
    #     # 轻微修改 (仅当 query_tokens 可训练时)
    #     with torch.no_grad():
    #         instruct_qformer.query_tokens.data += 0.1
    #     print(f"修改后 query_tokens sum: {instruct_qformer.query_tokens.data.sum().item()}")
    # else:
    #     print("query_tokens 不可训练，跳过修改步骤。")
    #
    # # 2. 加载可训练权重
    # print(f"\n从 {save_file} 加载可训练权重...")
    # instruct_qformer.load_trainable_weight(save_file)
    #
    # # 验证加载
    # if instruct_qformer.query_tokens.requires_grad:
    #     restored_query_tokens_sum = instruct_qformer.query_tokens.data.sum().item()
    #     print(f"加载后 query_tokens sum: {restored_query_tokens_sum}")
    #     # 使用 torch.allclose 进行更鲁棒的比较
    #     # assert abs(restored_query_tokens_sum - original_query_tokens_sum) < 1e-5, "加载后的权重与原始权重不匹配！"
    #     # 注意：浮点数比较可能需要容忍度
    #     if torch.isclose(torch.tensor(restored_query_tokens_sum), torch.tensor(original_query_tokens_sum)):
    #         print("加载验证成功：query_tokens 已恢复。")
    #     else:
    #         print(
    #             f"加载验证警告：query_tokens sum ({restored_query_tokens_sum}) 与原始值 ({original_query_tokens_sum}) 略有不同，可能是浮点精度问题。")
    # else:
    #     print("query_tokens 不可训练，跳过加载验证。")