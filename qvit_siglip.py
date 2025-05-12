# --- Start: Adapted QA-ViT components for SigLIP ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipVisionConfig, SiglipVisionModel, SiglipPreTrainedModel
from transformers.models.siglip.modeling_siglip import SiglipAttention, SiglipMLP, SiglipVisionEmbeddings, SiglipEncoderLayer
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutput
from typing import Optional, Tuple, Union

# Helper MLP (similar to paper's MLP and clip_qvit.py's FeedForward)
# Use SiglipMLP structure as a base for the parallel path
class QASiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.activation_fn = F.gelu # SigLIP uses GELU
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

# Modified Attention to handle instruction fusion via concatenation  ##sd
class QASiglipAttention(SiglipAttention):
    """
    Adapts SiglipAttention to concatenate instruction tokens (kv_states)
    before the self-attention mechanism, similar to MMCLIPAttention.
    The parallel gated projection part is handled in the EncoderLayer now.
    """
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None, # SigLIP Attention doesn't typically use mask
        output_attentions: Optional[bool] = False,
        # QA-ViT specific inputs
        kv_states: Optional[torch.Tensor] = None, # Projected instruction features
        kv_masks: Optional[torch.Tensor] = None, # Mask for instruction features (unused here)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if kv_states is None:
            # Fallback to standard SiglipAttention if no instruction provided
            return super().forward(hidden_states, attention_mask, output_attentions)

        # --- QA-ViT Fusion Logic ---
        # Concatenate instruction tokens (kv_states) with visual tokens (hidden_states)
        # kv_states are assumed to be already projected to the correct dimension
        batch_size, vis_seq_len, embed_dim = hidden_states.size()
        _, instr_seq_len, _ = kv_states.size()
        concat_states = torch.cat([kv_states, hidden_states], dim=1)
        concat_seq_len = concat_states.size(1) # vis_seq_len + instr_seq_len

        # --- Standard SigLIP Attention applied to concatenated sequence ---
        head_dim = embed_dim // self.num_heads
        if head_dim * self.num_heads != embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        scale = head_dim**-0.5

        # Project concatenated states to Q, K, V
        query_states = self.q_proj(concat_states) * scale
        key_states = self.k_proj(concat_states)
        value_states = self.v_proj(concat_states)

        # Reshape for multi-head attention
        # (bs, seq_len, embed_dim) -> (bs, num_heads, seq_len, head_dim)
        query_states = query_states.view(batch_size, concat_seq_len, self.num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, concat_seq_len, self.num_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, concat_seq_len, self.num_heads, head_dim).transpose(1, 2)

        # Compute attention scores
        # (bs, num_heads, seq_len, head_dim) x (bs, num_heads, head_dim, seq_len)
        # -> (bs, num_heads, seq_len, seq_len)
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))

        # --- QA-ViT Output Selection ---
        # We only care about the output for the original visual tokens
        # The queries corresponding to visual tokens attend to all keys (visual + instruction)
        # Attention weights shape: (bs, num_heads, concat_seq_len, concat_seq_len)
        # We need the weights where query index corresponds to visual tokens (last vis_seq_len rows)
        attn_weights_vis_queries = attn_weights[:, :, instr_seq_len:, :]

        if attn_weights_vis_queries.size() != (batch_size, self.num_heads, vis_seq_len, concat_seq_len):
             raise ValueError(
                f"Attention weights shape error. Expected: "
                f"{(batch_size, self.num_heads, vis_seq_len, concat_seq_len)}, "
                f"Got: {attn_weights_vis_queries.size()}"
            )

        # Apply softmax
        attn_weights_vis_queries = nn.functional.softmax(attn_weights_vis_queries, dim=-1)

        # Optional: Dropout (apply to the selected weights)
        attn_probs = nn.functional.dropout(attn_weights_vis_queries, p=self.dropout, training=self.training)

        # Compute attention output using the probabilities for visual queries and *all* value states
        # (bs, num_heads, vis_seq_len, concat_seq_len) x (bs, num_heads, concat_seq_len, head_dim)
        # -> (bs, num_heads, vis_seq_len, head_dim)
        attn_output = torch.matmul(attn_probs, value_states)

        # Reshape back to (bs, vis_seq_len, embed_dim)
        if attn_output.size() != (batch_size, self.num_heads, vis_seq_len, head_dim):
             raise ValueError(
                f"`attn_output` shape error. Expected: "
                f"{(batch_size, self.num_heads, vis_seq_len, head_dim)}, Got: {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous() # Ensure contiguity
        attn_output = attn_output.reshape(batch_size, vis_seq_len, embed_dim)

        # Apply output projection
        attn_output = self.out_proj(attn_output)
        # --- End QA-ViT Specific Attention Logic ---

        # Return attention output and optionally the weights (only for vis queries)
        if output_attentions:
            # Note: Returning weights corresponding only to visual queries
            return attn_output, attn_weights_vis_queries
        else:
            return attn_output, None

# Modified Encoder Layer with optional QA-ViT components
class QASiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig, instruction_dim: int):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # Use the modified Attention layer
        self.self_attn = QASiglipAttention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # Standard SigLIP MLP
        self.mlp = SiglipMLP(config)

        # --- QA-ViT Specific Components ---
        # MLP to project instruction features to ViT dimension (if needed)
        # Paper suggests independent MLP per layer, but for simplicity here, one type.
        # Input dim is instruction_dim, output dim is ViT hidden_size
        self.instruct_dim_reduce = nn.Sequential(
            nn.LayerNorm(instruction_dim), # Normalize instruction features first
            nn.Linear(instruction_dim, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )

        # Parallel Gated Projection Path (parallel to self.mlp)
        self.parallel_mlp = QASiglipMLP(config) # Using SigLIP's MLP structure
        self.parallel_gate = nn.Parameter(torch.zeros(1)) # Initialize beta to 0

        # Initialize parallel path like the original one? Maybe. Paper initializes beta=0.
        # Let's initialize weights of parallel_mlp randomly for now.
        # Alternatively, could copy weights from self.mlp if desired.


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        # QA-ViT specific inputs
        instruct_states: Optional[torch.Tensor] = None,
        instruct_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:

        residual = hidden_states
        hidden_states_norm = self.layer_norm1(hidden_states)

        # Project instructions if provided
        kv_states = None
        if instruct_states is not None:
             kv_states = self.instruct_dim_reduce(instruct_states)

        # Self Attention (using QASiglipAttention)
        attn_output, attn_weights = self.self_attn(
            hidden_states=hidden_states_norm,
            output_attentions=output_attentions,
            kv_states=kv_states,
            kv_masks=instruct_masks, # Pass mask if needed by attention impl.
        )

        # First residual connection
        hidden_states = attn_output + residual

        # MLP part with Parallel Gated Projection
        residual = hidden_states
        hidden_states_norm = self.layer_norm2(hidden_states)

        # Original MLP path
        mlp_output = self.mlp(hidden_states_norm)

        # Parallel Gated Path
        parallel_output = self.parallel_mlp(hidden_states_norm)
        gate_value = torch.tanh(self.parallel_gate) # tanh(beta)
        gated_parallel_output = parallel_output * gate_value

        # Combine original MLP output, gated parallel output, and residual
        hidden_states = residual + mlp_output + gated_parallel_output # Eq 4 adaptation

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

# The Encoder that uses the modified layers
class QASiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig, instruction_dim: int, integration_point: str):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        num_hidden_layers = config.num_hidden_layers
        self.gradient_checkpointing = False # Can be set to True for training

        for layer_idx in range(num_hidden_layers):
            # Determine if this layer should be QA-ViT enabled
            is_qa_layer = False
            if integration_point == 'all':
                is_qa_layer = True
            elif integration_point == 'late':
                # Example: last half of the layers
                is_qa_layer = (layer_idx >= num_hidden_layers // 2)
            elif integration_point == 'early':
                 is_qa_layer = (layer_idx < num_hidden_layers // 2)
            # Add other integration points like 'late2', 'sparse' if needed
            elif integration_point == 'none': # For ablation/baseline
                is_qa_layer = False
            else:
                raise ValueError(f"Unsupported integration_point: {integration_point}")

            if is_qa_layer:
                layer = QASiglipEncoderLayer(config, instruction_dim)
            else:
                # Use the standard SigLIP layer
                layer = SiglipEncoderLayer(config)
            self.layers.append(layer)

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None, # Usually None for ViT
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # QA-ViT specific inputs
        instruct_states: Optional[torch.Tensor] = None,
        instruct_masks: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds

        for idx, layer_module in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            layer_outputs = None
            # Handle gradient checkpointing if enabled
            if self.gradient_checkpointing and self.training:
                 # Wrapper for checkpointing
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # Pass relevant args based on layer type
                        if isinstance(module, QASiglipEncoderLayer):
                            return module(inputs[0], output_attentions=output_attentions,
                                          instruct_states=inputs[1], instruct_masks=inputs[2])
                        else: # Standard SiglipEncoderLayer
                             return module(inputs[0], output_attentions=output_attentions)
                    return custom_forward

                if isinstance(layer_module, QASiglipEncoderLayer):
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer_module),
                        hidden_states, instruct_states, instruct_masks,
                        use_reentrant=False # Recommended for newer PyTorch versions
                    )
                else:
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer_module),
                        hidden_states, None, None, # Pass placeholders for non-QA layers
                        use_reentrant=False
                    )
            else:
                # Standard forward pass
                if isinstance(layer_module, QASiglipEncoderLayer):
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask=attention_mask, # Pass if SigLIP layer uses it
                        output_attentions=output_attentions,
                        instruct_states=instruct_states,
                        instruct_masks=instruct_masks,
                    )
                else: # Standard SiglipEncoderLayer
                     layer_outputs = layer_module(
                        hidden_states,
                        attention_mask=attention_mask,
                        output_attentions=output_attentions,
                    )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


# The main Vision Model using the QA-Encoder
class QASiglipVisionTransformer(nn.Module):
    """ Vision Transformer based on SigLIP using QASiglipEncoder """
    def __init__(self, config: SiglipVisionConfig, instruction_dim: int, integration_point: str):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        # SigLIP applies layer norm *before* the encoder layers
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = QASiglipEncoder(config, instruction_dim, integration_point)
        # SigLIP applies layer norm *after* the encoder layers for the pooled output
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)


    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # QA-ViT specific inputs
        instruct_states: Optional[torch.Tensor] = None,
        instruct_masks: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Standard SigLIP embedding process
        hidden_states = self.embeddings(pixel_values) # Patch + Position embeddings
        hidden_states = self.pre_layrnorm(hidden_states)

        # Pass through the potentially QA-enabled Encoder
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # Pass instruction features down
            instruct_states=instruct_states,
            instruct_masks=instruct_masks,
        )

        last_hidden_state = encoder_outputs[0]

        # SigLIP uses the CLS token (index 0) for pooling
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output) # Apply final layer norm

        if not return_dict:
            # Return according to standard HF vision model output format
            outputs = (last_hidden_state, pooled_output) + encoder_outputs[1:]
            return tuple(output for output in outputs if output is not None)


        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output, # Use the pooled output
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# Wrapper class similar to InstructCLIPVisionModel
class QASiglipVisionModel(SiglipPreTrainedModel):
    config_class = SiglipVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: SiglipVisionConfig, instruction_dim: int, integration_point: str):
        super().__init__(config)
        # Store QA-ViT specific config within the main config for convenience
        self.config.instruction_dim = instruction_dim
        self.config.integration_point = integration_point
        # Instantiate the custom Vision Transformer
        self.vision_model = QASiglipVisionTransformer(config, instruction_dim, integration_point)
        # Initialize weights and apply final processing (standard HF practice)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        # Return the patch embedding layer
        return self.vision_model.embeddings.patch_embedding

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model. Not implemented for QA layers yet. """
        raise NotImplementedError("Pruning heads is not supported for QASiglip layers yet.")


    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # QA-ViT specific inputs
        instruct_states: Optional[torch.Tensor] = None,
        instruct_masks: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Delegate the forward pass to the internal QASiglipVisionTransformer
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # Pass QA-ViT inputs
            instruct_states=instruct_states,
            instruct_masks=instruct_masks,
        )

    # Add a method to handle freezing easily
    def freeze_base_model(self):
        """Freezes all parameters except those specific to QA-ViT."""
        print("Freezing base SigLIP Vision Model parameters...")
        print("All SigLIP Vision Model parameters...")
        # for name, param in self.named_parameters():
        #     print(f"  - : {name}")
        print("Keeping trainable SigLIP Vision Model parameters...")
        for name, param in self.named_parameters():
            is_qa_param = False
            # Identify QA-ViT specific parameters (adapt names if changed)
            if "instruct_dim_reduce" in name: is_qa_param = True
            if "parallel_mlp" in name: is_qa_param = True
            if "parallel_gate" in name: is_qa_param = True
            # Add more specific checks if needed (e.g., if QASiglipAttention had trainable params)

            if not is_qa_param:
                param.requires_grad = False
            else:
                 print(f"  - Keeping trainable: {name}")


# --- End: Adapted QA-ViT components for SigLIP ---

if __name__=="__main__":
    # --- Now, modify your loading code ---
    import torch,os
    from transformers import AutoImageProcessor, AutoModel, SiglipVisionConfig, SiglipVisionModel

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 这会让程序只看到4号GPU

    # Configuration
    SIGLIP_MODEL_ID = "../models/siglip"
    # --- QA-ViT Configuration ---
    QA_INTEGRATION_POINT = 'late' # Options: 'late', 'all', 'early', 'none', etc.
    # Define the dimension of your instruction features AFTER text encoding/embedding
    # Example: If using T5-large encoder, dim is 1024. If using SigLIP text encoder, dim might be different.
    # For this example, let's assume it matches the vision model's hidden dim for simplicity,
    # but this *must* match the output of your chosen text encoder.
    INSTRUCTION_DIM = 1152 # SigLIP-so400m has hidden_size 1152

    # Other settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = torch.float16 if device.type == "cuda" else torch.float32

    # 1. Load Image Processor (remains the same)
    print(f"Loading SigLIP Image Processor: {SIGLIP_MODEL_ID}...")
    try:
        siglip_processor = AutoImageProcessor.from_pretrained(SIGLIP_MODEL_ID)
        print("Image Processor loaded.")
    except Exception as e:
        print(f"Error loading SigLIP processor: {e}")
        exit()

    # 2. Load SigLIP Vision Configuration
    print(f"Loading SigLIP Vision Config: {SIGLIP_MODEL_ID}...")
    try:
        siglip_vision_config = SiglipVisionConfig.from_pretrained(SIGLIP_MODEL_ID)
        print("SigLIP Vision Config loaded.")
    except Exception as e:
        print(f"Error loading SigLIP vision config: {e}")
        exit()

    # 3. Instantiate the Custom QASiglipVisionModel
    print(f"Instantiating QASiglipVisionModel with integration='{QA_INTEGRATION_POINT}'...")
    qa_siglip_model = QASiglipVisionModel(
        config=siglip_vision_config,
        instruction_dim=INSTRUCTION_DIM,
        integration_point=QA_INTEGRATION_POINT
    )
    print("QASiglipVisionModel instantiated.")

    # 4. Load Pre-trained Weights from Original SigLIP Vision Model
    print(f"Loading pre-trained weights from {SIGLIP_MODEL_ID} into QASiglipVisionModel...")
    try:
        # Load the standard model temporarily to get its state dict
        original_siglip_vision_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_ID)
        original_state_dict = original_siglip_vision_model.state_dict()

        # Load into our custom model, ignoring missing/unexpected keys  ##sd
        missing_keys, unexpected_keys = qa_siglip_model.load_state_dict(original_state_dict, strict=False)

        print("\nWeight loading report:")
        print(f"  - Missing keys (expected QA-ViT params): {missing_keys}")
        print(f"  - Unexpected keys (should be empty): {unexpected_keys}")
        if len(unexpected_keys) > 0:
            print("WARNING: Unexpected keys found during weight loading!")

        # Explicitly initialize the parallel gate to zero (as per paper)
        # Other QA params (MLPs) are randomly initialized by default, which is fine for training
        for name, param in qa_siglip_model.named_parameters():
            if "parallel_gate" in name:
                with torch.no_grad():
                    param.zero_()
            # Optional: Initialize parallel MLP path weights similar to original MLP
            # if 'parallel_mlp' in name and 'weight' in name:
            #     original_name = name.replace('parallel_mlp', 'mlp')
            #     if original_name in original_state_dict:
            #          with torch.no_grad():
            #              param.copy_(original_state_dict[original_name])
            # if 'parallel_mlp' in name and 'bias' in name:
            #      original_name = name.replace('parallel_mlp', 'mlp')
            #      if original_name in original_state_dict:
            #          with torch.no_grad():
            #              param.copy_(original_state_dict[original_name])

        print("Pre-trained weights loaded successfully.")
        del original_siglip_vision_model # Free memory
    except Exception as e:
        print(f"Error loading SigLIP weights: {e}")
        exit()

    # 5. Move model to device and set dtype
    qa_siglip_model = qa_siglip_model.to(device=device, dtype=compute_dtype)

    # 6. Freeze base model parameters for training (Optional but Recommended)
    qa_siglip_model.freeze_base_model()

    # 7. Set to train or eval mode
    # qa_siglip_model.train() # If you are going to train the QA params
    qa_siglip_model.eval() # If you are just doing inference (QA params untrained)

    # --- Placeholder for Text Encoding ---
    # You NEED to add your text processing pipeline here
    # Example:
    # from transformers import AutoTokenizer, T5EncoderModel
    # text_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    # text_encoder = T5EncoderModel.from_pretrained("google/flan-t5-large").to(device).eval() # Or train it?
    # instruction_dim = text_encoder.config.d_model # Should match INSTRUCTION_DIM above

    def get_instruction_features(question_text, tokenizer, encoder, device):
        """Placeholder function to encode question"""
        if not isinstance(question_text, list):
            question_text = [question_text] # Handle single question case
        inputs = tokenizer(question_text, return_tensors="pt", padding=True, truncation=True, max_length=32) # Adjust max_length
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad(): # Assuming text encoder is frozen during VQA finetuning
            outputs = encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            instruct_states = outputs.last_hidden_state
        instruct_masks = inputs["attention_mask"] # Use tokenizer mask
        return instruct_states, instruct_masks
    # --- End Placeholder ---


    # --- Example Usage ---
    print("\n--- Example Usage ---")
    # Load a sample image (replace with your image loading)
    from PIL import Image
    url = "./000000039769.jpg"
    try:
        image = Image.open(url).convert("RGB")
        print("Sample image loaded.")

        # Prepare image input
        image_inputs = siglip_processor(images=image, return_tensors="pt").to(device=device, dtype=compute_dtype)

        # Prepare question input (using placeholder function)
        question = "What color is the bus?"
        # instruct_states, instruct_masks = get_instruction_features(
        #     question, text_tokenizer, text_encoder, device
        # )
        # Since we don't have the text encoder here, let's create dummy tensors
        # IMPORTANT: Replace this with your actual text encoding!
        batch_size = image_inputs['pixel_values'].shape[0]
        instr_seq_len = 10 # Dummy sequence length for instruction
        instruct_states = torch.randn(batch_size, instr_seq_len, INSTRUCTION_DIM, device=device, dtype=compute_dtype)
        instruct_masks = torch.ones(batch_size, instr_seq_len, device=device, dtype=torch.long)
        print(f"Using DUMMY instruction features for '{question}'")


        # Forward pass through the QA-SigLIP model
        print("Performing forward pass...")
        with torch.no_grad(): # Use no_grad for inference example
            outputs = qa_siglip_model(
                pixel_values=image_inputs['pixel_values'],
                # Pass the encoded question features
                instruct_states=instruct_states,
                instruct_masks=instruct_masks,
                output_hidden_states=True, # Optional: get all hidden states
                return_dict=True
            )

        # Access outputs
        last_hidden_state = outputs.last_hidden_state # Shape: (batch, num_patches + 1, hidden_size)
        pooled_output = outputs.pooler_output       # Shape: (batch, hidden_size) - CLS token output

        print(f"Forward pass successful.")
        print(f"  - Last Hidden State shape: {last_hidden_state.shape}")
        print(f"  - Pooled Output shape: {pooled_output.shape}")
        # You would typically feed these features (e.g., last_hidden_state) into your decoder/LLM

    except Exception as e:
        print(f"Error during example usage: {e}")