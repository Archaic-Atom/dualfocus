import torch
import torch.nn as nn
import os
from transformers import BertTokenizer, BertConfig
from Qformer import BertLMHeadModel
import logging

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
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def __init__(
        self,
        num_query_token=32,
        vision_width=1408,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=128,
        qformer_pretrained_model="bert-base-uncased",
        qformer_text_input=True,
        freeze_qformer_base=True
    ):
        super().__init__()

        self.qformer_text_input = qformer_text_input
        self.tokenizer = self.init_tokenizer()
        self.max_txt_len = max_txt_len
        self.num_query_token = num_query_token

        encoder_config = BertConfig.from_pretrained(qformer_pretrained_model)
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token


        self.Qformer = BertLMHeadModel(config=encoder_config)

        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        if self.qformer_text_input:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
            logging.info(f"Resized Q-Former token embeddings to {len(self.tokenizer)}")
        else:
            logging.warning("qformer_text_input is False. InstructBLIP typically requires text input to Q-Former.")
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None

        self.Qformer.cls = None
        logging.info("Removed Q-Former's LM head (if any).")
        if freeze_qformer_base:
            print("Freezing Q-Former base BERT parameters...")
            for name, param in self.Qformer.named_parameters():
                if 'bert' in name.lower():
                    param.requires_grad = False
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

        query_tokens = self.query_tokens.expand(batch_size, -1, -1).to(device)

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

            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

            query_output = self.Qformer.bert(
                input_ids=text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
             query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        instruct_visual_features = query_output.last_hidden_state[:, :self.num_query_token, :]

        return instruct_visual_features

    def load_checkpoint(self, filename):
        """Loads a checkpoint, filtering for Q-Former and query_tokens weights."""
        if filename is None or not filename:
             logging.warning("No checkpoint filename provided for QFormer.")
             return
        checkpoint = torch.load(filename, map_location="cpu")
        state_dict = checkpoint["model"]

        qformer_weights = {}
        for k, v in state_dict.items():
            if k.startswith("Qformer.") or k.startswith("query_tokens"):
                qformer_weights[k] = v

        msg = self.load_state_dict(qformer_weights, strict=False)
        logging.info(f"Loaded Q-Former checkpoint from {filename}")
        logging.info(f"Q-Former Load Msg: {msg}")

    def save_trainable_weight(self, save_path):
        """
                Save the trainable parameters in the model to the specified path.
                Only the parameters of requires_grad = True are saved.

        Args:
            save_path (str): The file path where the weights are stored.
        """
        trainable_state_dict = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_state_dict[name] = param.data

        if not trainable_state_dict:
            logging.warning("No trainable parameters found in the model! Cannot be saved.")
            return

        try:
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)

            torch.save(trainable_state_dict, save_path)
            logging.info(f"The trainable parameters of the model have been successfully saved to: {save_path}")
        except Exception as e:
            logging.error(f"Error saving trainable parameters to {save_path}: {e}")

    def load_trainable_weight(self, load_path):
        if not os.path.exists(load_path):
            logging.error(f"Weight file not found: {load_path}")
            print(f"Error: Weight file not found: {load_path}")
            return

        try:
            trainable_state_dict = torch.load(load_path, map_location='cpu')

            current_trainable_params = {name for name, param in self.named_parameters() if param.requires_grad}

            filtered_state_dict = {}
            loaded_keys = set(trainable_state_dict.keys())
            keys_to_load_actually = []

            for name, param_data in trainable_state_dict.items():
                if name in self.state_dict():
                    if name in current_trainable_params:
                        filtered_state_dict[name] = param_data
                        keys_to_load_actually.append(name)
                    else:
                        logging.warning(f"Skip loading parameter '{name}' because it is frozen in the current model (requires_grad=False).")
                else:
                    logging.warning(f"Skip loading parameter '{name}' because it is not in the current model structure.")

            if not filtered_state_dict:
                logging.warning(f"No weights were found in the file {load_path} that matched the current model's trainable parameters.")
                print(f"Warning: No weights were found in the file {load_path} that match the current model trainable parameters.")
                return

            msg = self.load_state_dict(filtered_state_dict, strict=False)

            logging.info(f"Loading trainable parameters from {load_path} is completed.")
            print(f"Loading trainable parameters from {load_path} is completed.")
            print(f"{len(keys_to_load_actually)} / {len(loaded_keys)} parameter tensors from the file were successfully loaded.")

            if msg.unexpected_keys:
                logging.error(f"Load state dict report - Unexpected Keys (issue): {msg.unexpected_keys}")
                print(f"Error: Unexpected key encountered on load: {msg.unexpected_keys}")

        except FileNotFoundError:
            logging.error(f"weightFileNotFoundLoad_path")
            print(f"errorWeightFileNotFoundLoad_path")
        except Exception as e:
            logging.error(f"Error loading trainable parameters from {load_path}: {e}")
            print(f"Error loading trainable parameters from {load_path}: {e}")

