# --- Start: Adapted QA-ViT components for SigLIP ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipVisionConfig, SiglipVisionModel, SiglipPreTrainedModel
# Make sure to import the correct base classes from SigLIP modeling file
from transformers.models.siglip.modeling_siglip import SiglipAttention, SiglipMLP, SiglipVisionEmbeddings, SiglipEncoderLayer, SiglipVisionTransformer as HFSiglipVisionTransformer
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutput
from typing import Optional, Tuple, Union
import logging # Use logging for better messages

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper MLP (similar to paper's MLP and clip_qvit.py's FeedForward)
# Use SigLIP MLP structure as a base for the parallel path
class QASiglipMLP(nn.Module):
    """Parallel MLP for QA-ViT specific path, mimicking SiglipMLP structure."""
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        # SigLIP uses approximate GELU (check config, but F.gelu is common)
        # Use config._attn_implementation for specific activation if needed, otherwise GELU
        self.activation_fn = F.gelu
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

# Modified Attention to handle instruction fusion via concatenation
class QASiglipAttention(SiglipAttention):
    """
    Adapts SiglipAttention to concatenate instruction tokens (kv_states)
    before the self-attention mechanism, similar to MMCLIPAttention.
    """
    # Inherits __init__ from SiglipAttention (q_proj, k_proj, v_proj, projection, dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None, # SigLIP Attention doesn't typically use mask
        output_attentions: Optional[bool] = False,
        # QA-ViT specific inputs
        kv_states: Optional[torch.Tensor] = None, # Projected instruction features
        kv_masks: Optional[torch.Tensor] = None, # Mask for instruction features (currently unused)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if kv_states is None:
            # Fallback to standard SiglipAttention if no instruction provided
            # Note: Standard SiglipAttention forward signature might differ slightly,
            # ensure compatibility or call super() appropriately if needed.
            # Calling the original logic explicitly here for clarity:
            bsz, tgt_len, embed_dim = hidden_states.size()
            head_dim = embed_dim // self.num_heads
            scale = head_dim**-0.5
            query_states = self.q_proj(hidden_states) * scale
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, tgt_len, self.num_heads, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, tgt_len, self.num_heads, head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, tgt_len, self.num_heads, head_dim).transpose(1, 2)

            attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))
            # No attention mask applied in standard SiglipAttention

            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
            attn_output = torch.matmul(attn_probs, value_states)

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

            # *** Fix: Use self.projection instead of self.o_proj ***
            attn_output = self.projection(attn_output)

            if output_attentions:
                return attn_output, attn_weights
            else:
                return attn_output, None
            # --- End Fallback ---


        # --- QA-ViT Fusion Logic ---
        batch_size, vis_seq_len, embed_dim = hidden_states.size()
        _, instr_seq_len, _ = kv_states.size()

        # Concatenate instruction tokens (kv_states) with visual tokens (hidden_states)
        # kv_states are assumed to be already projected to the correct dimension (embed_dim)
        concat_states = torch.cat([kv_states, hidden_states], dim=1)
        concat_seq_len = concat_states.size(1) # vis_seq_len + instr_seq_len

        # --- Standard SigLIP Attention applied to concatenated sequence ---
        head_dim = embed_dim // self.num_heads
        # No need to check divisibility, SiglipAttention init does this
        scale = head_dim**-0.5

        # Project concatenated states to Q, K, V
        query_states = self.q_proj(concat_states) * scale
        key_states = self.k_proj(concat_states)
        value_states = self.v_proj(concat_states)

        # Reshape for multi-head attention
        # (bs, concat_seq_len, embed_dim) -> (bs, num_heads, concat_seq_len, head_dim)
        query_states = query_states.view(batch_size, concat_seq_len, self.num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, concat_seq_len, self.num_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, concat_seq_len, self.num_heads, head_dim).transpose(1, 2)

        # Compute attention scores
        # (bs, num_heads, concat_seq_len, head_dim) x (bs, num_heads, head_dim, concat_seq_len)
        # -> (bs, num_heads, concat_seq_len, concat_seq_len)
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))

        # --- QA-ViT Output Selection ---
        # We only care about the output for the original visual tokens.
        # Select weights where query index corresponds to visual tokens (last vis_seq_len rows)
        # Visual tokens attend to ALL keys (instruction + visual)
        attn_weights_vis_queries = attn_weights[:, :, instr_seq_len:, :] # Shape: (bs, num_heads, vis_seq_len, concat_seq_len)

        # Check shape (optional, good for debugging)
        # if attn_weights_vis_queries.size() != (batch_size, self.num_heads, vis_seq_len, concat_seq_len):
        #      logger.warning( # Use logger
        #         f"Attention weights shape error. Expected: "
        #         f"{(batch_size, self.num_heads, vis_seq_len, concat_seq_len)}, "
        #         f"Got: {attn_weights_vis_queries.size()}"
        #     )

        # Apply softmax over the keys dimension (last dim)
        attn_weights_vis_queries = nn.functional.softmax(attn_weights_vis_queries, dim=-1)

        # Optional: Dropout (apply to the selected probabilities)
        attn_probs = nn.functional.dropout(attn_weights_vis_queries, p=self.dropout, training=self.training)

        # Compute attention output using the probabilities for visual queries and *all* value states
        # (bs, num_heads, vis_seq_len, concat_seq_len) x (bs, num_heads, concat_seq_len, head_dim)
        # -> (bs, num_heads, vis_seq_len, head_dim)
        attn_output = torch.matmul(attn_probs, value_states)

        # Reshape back to (bs, vis_seq_len, embed_dim)
        # Check shape (optional)
        # if attn_output.size() != (batch_size, self.num_heads, vis_seq_len, head_dim):
        #      logger.warning(
        #         f"`attn_output` shape error. Expected: "
        #         f"{(batch_size, self.num_heads, vis_seq_len, head_dim)}, Got: {attn_output.size()}"
        #     )

        attn_output = attn_output.transpose(1, 2).contiguous() # Ensure contiguity
        attn_output = attn_output.reshape(batch_size, vis_seq_len, embed_dim)

        # Apply output projection
        # *** Fix: Use self.projection instead of self.o_proj ***
        attn_output = self.projection(attn_output)
        # --- End QA-ViT Specific Attention Logic ---

        # Return attention output and optionally the weights (only for vis queries)
        if output_attentions:
            # Note: Returning weights corresponding only to visual queries attending to all keys
            return attn_output, attn_probs # Return probs after softmax/dropout
        else:
            return attn_output, None

# Modified Encoder Layer with optional QA-ViT components
class QASiglipEncoderLayer(nn.Module):
    """ Siglip Encoder Layer adapted for QA-ViT. """
    def __init__(self, config: SiglipVisionConfig, instruction_dim: int):
        super().__init__()
        self.embed_dim = config.hidden_size
        # Layer Norm before attention
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # Use the modified Attention layer
        self.self_attn = QASiglipAttention(config)
        # Layer Norm before MLP
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # Standard SigLIP MLP
        self.mlp = SiglipMLP(config)

        # --- QA-ViT Specific Components ---
        # MLP to project instruction features to ViT dimension
        # Paper suggests independent MLP per layer, but for simplicity here, one type.
        # Input dim is instruction_dim, output dim is ViT hidden_size
        # Added LayerNorm before projection as in clip_qvit.py FeedForward
        self.instruct_dim_reduce = nn.Sequential(
            nn.LayerNorm(instruction_dim, eps=config.layer_norm_eps), # Normalize instruction features first
            nn.Linear(instruction_dim, config.hidden_size),
            nn.GELU(), # Or use config activation
            nn.Linear(config.hidden_size, config.hidden_size)
        )

        # Parallel Gated Projection Path (parallel to self.mlp)
        self.parallel_mlp = QASiglipMLP(config) # Using specific QA MLP class
        self.parallel_gate = nn.Parameter(torch.zeros(1)) # Initialize beta to 0

        # Initialization: parallel_mlp weights are random, gate is zero.
        # This matches paper's intent (start identical to base model).

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None, # Usually None for ViT
        output_attentions: Optional[bool] = False,
        # QA-ViT specific inputs
        instruct_states: Optional[torch.Tensor] = None,
        instruct_masks: Optional[torch.Tensor] = None, # Pass mask if needed by attention impl.
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:

        residual = hidden_states

        # Layer Norm 1
        hidden_states_norm = self.layer_norm1(hidden_states)

        # Project instructions if provided
        kv_states = None
        if instruct_states is not None:
             # Ensure instruct_states are on the correct device/dtype
             instruct_states = instruct_states.to(hidden_states_norm.device, dtype=hidden_states_norm.dtype)
             kv_states = self.instruct_dim_reduce(instruct_states)

        # Self Attention (using QASiglipAttention)
        attn_output, attn_weights = self.self_attn(
            hidden_states=hidden_states_norm,
            # attention_mask=attention_mask, # Pass if QASiglipAttention uses it
            output_attentions=output_attentions,
            kv_states=kv_states,
            kv_masks=instruct_masks,
        )

        # First residual connection (Attention output + Input)
        # SigLIP applies dropout here if configured (config.attention_dropout) - check if needed
        # attn_output = nn.functional.dropout(attn_output, p=self.config.attention_dropout, training=self.training)
        hidden_states = attn_output + residual

        # MLP part with Parallel Gated Projection
        residual = hidden_states
        # Layer Norm 2
        hidden_states_norm = self.layer_norm2(hidden_states)

        # Original MLP path
        mlp_output = self.mlp(hidden_states_norm)
        # SigLIP applies dropout here too if configured (config.dropout) - check if needed
        # mlp_output = nn.functional.dropout(mlp_output, p=self.config.dropout, training=self.training)


        # Parallel Gated Path
        parallel_output = self.parallel_mlp(hidden_states_norm)
        gate_value = torch.tanh(self.parallel_gate) # tanh(beta)
        # Ensure gate_value is broadcastable if needed, though usually scalar * tensor works
        gated_parallel_output = parallel_output * gate_value.to(parallel_output.dtype) # Match dtype

        # Combine original MLP output, gated parallel output, and residual
        # Second residual connection (MLP output(s) + Input to MLP)
        hidden_states = residual + mlp_output + gated_parallel_output # Eq 4 adaptation

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

# The Encoder that uses the modified layers
class QASiglipEncoder(nn.Module):
    """ Siglip Encoder adapted to use QASiglipEncoderLayer at specified points. """
    def __init__(self, config: SiglipVisionConfig, instruction_dim: int, integration_point: str):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        num_hidden_layers = config.num_hidden_layers
        # Default gradient checkpointing to False, can be enabled externally
        self.gradient_checkpointing = False

        integration_point = integration_point.lower() # Normalize case
        logger.info(f"Initializing QASiglipEncoder with integration_point='{integration_point}'")

        # Calculate start layer for 'late' based on total layers
        # Example: If 27 layers, // 2 gives 13. Layers 13-26 are 'late'. (14 layers)
        # If 26 layers, // 2 gives 13. Layers 13-25 are 'late'. (13 layers)
        late_fusion_start_layer = num_hidden_layers // 2

        for layer_idx in range(num_hidden_layers):
            is_qa_layer = False
            if integration_point == 'all':
                is_qa_layer = True
            elif integration_point == 'late':
                is_qa_layer = (layer_idx >= late_fusion_start_layer)
            elif integration_point == 'early':
                 is_qa_layer = (layer_idx < late_fusion_start_layer)
            # Add other integration points like 'late2', 'sparse' if needed
            # Example: late2 = last quarter
            # late2_start_layer = 3 * num_hidden_layers // 4
            # elif integration_point == 'late2':
            #     is_qa_layer = (layer_idx >= late2_start_layer)
            elif integration_point == 'none': # For ablation/baseline
                is_qa_layer = False
            # else: # Keep it simple or add explicit checks
                # raise ValueError(f"Unsupported integration_point: {integration_point}")

            if is_qa_layer:
                logger.debug(f"  Layer {layer_idx}: Using QASiglipEncoderLayer")
                layer = QASiglipEncoderLayer(config, instruction_dim)
            else:
                logger.debug(f"  Layer {layer_idx}: Using standard SiglipEncoderLayer")
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
            layer_outputs = None
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            # Handle gradient checkpointing if enabled
            if self.gradient_checkpointing and self.training:
                 # Wrapper for checkpointing needs to handle different layer signatures
                def create_custom_forward(module, is_qa):
                    def custom_forward(*inputs):
                        _hidden_states = inputs[0]
                        _instruct_states = inputs[1] if len(inputs) > 1 else None
                        _instruct_masks = inputs[2] if len(inputs) > 2 else None
                        if is_qa:
                            return module(_hidden_states, output_attentions=output_attentions,
                                          instruct_states=_instruct_states, instruct_masks=_instruct_masks)
                        else:
                             # Standard Siglip layer expects hidden_states, attention_mask, output_attentions
                             # attention_mask is often None for ViTs, pass it along if provided
                             return module(_hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
                    return custom_forward

                is_qa_layer = isinstance(layer_module, QASiglipEncoderLayer)
                if is_qa_layer:
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer_module, is_qa=True),
                        hidden_states, instruct_states, instruct_masks,
                        use_reentrant=False # Recommended for newer PyTorch versions
                    )
                else:
                    # Non-QA layer only needs hidden_states for its core computation path in checkpoint
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer_module, is_qa=False),
                        hidden_states, None, None, # Pass placeholders for non-QA inputs
                        use_reentrant=False
                    )
                # Checkpoint returns only the first output (hidden_states)
                # To get attentions, need to run forward pass again or disable checkpointing for those layers
                # Simplification: Assume layer_outputs is tuple only if not checkpointing
                hidden_states = layer_outputs
                layer_outputs = (hidden_states, None) # Create a tuple structure, # attention is lost

            else:
                # Standard forward pass
                if isinstance(layer_module, QASiglipEncoderLayer):
                    layer_outputs = layer_module(
                        hidden_states,
                        # attention_mask=attention_mask, # Pass if needed by QASiglipAttention
                        output_attentions=output_attentions,
                        instruct_states=instruct_states,
                        instruct_masks=instruct_masks,
                    )
                else: # Standard SiglipEncoderLayer
                     layer_outputs = layer_module(
                        hidden_states,
                        attention_mask=attention_mask, # Standard layer might use it
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

                if output_attentions:
                    # Ensure attentions are collected correctly even with mixed layer types
                    current_attn = layer_outputs[1] if len(layer_outputs) > 1 else None
                    all_attentions = all_attentions + (current_attn,)


        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            outputs = [hidden_states]
            if output_hidden_states:
                outputs.append(encoder_states)
            if output_attentions:
                outputs.append(all_attentions)
            return tuple(v for v in outputs if v is not None)


        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


# The main Vision Model using the QA-Encoder
class QASiglipVisionTransformer(nn.Module):
    """ Vision Transformer based on SigLIP using QASiglipEncoder """
    # This class now mirrors HFSiglipVisionTransformer structure but uses QASiglipEncoder
    def __init__(self, config: SiglipVisionConfig, instruction_dim: int, integration_point: str):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        # SigLIP applies layer norm *before* the encoder layers
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps) # Name matches HF
        self.encoder = QASiglipEncoder(config, instruction_dim, integration_point)
        # SigLIP applies layer norm *after* the encoder layers for the pooled output
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps) # Name matches HF


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
        # pixel_values: (batch_size, num_channels, height, width)
        hidden_states = self.embeddings(pixel_values)
        # hidden_states: (batch_size, seq_len, hidden_size), where seq_len = num_patches + 1 (for CLS token)

        # Apply pre-norm *before* encoder
        hidden_states = self.pre_layrnorm(hidden_states)

        # Pass through the potentially QA-enabled Encoder
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            # attention_mask=None, # Usually None for ViT
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # Pass instruction features down
            instruct_states=instruct_states,
            instruct_masks=instruct_masks,
        )

        # encoder_outputs can be tuple or BaseModelOutput
        if isinstance(encoder_outputs, tuple):
            last_hidden_state = encoder_outputs[0]
        else: # BaseModelOutput
            last_hidden_state = encoder_outputs.last_hidden_state

        # SigLIP uses the CLS token output (index 0) after the encoder blocks
        # And applies a final layer norm to this pooled output
        pooled_output = self.post_layernorm(last_hidden_state[:, 0, :]) # Apply final layer norm

        if not return_dict:
            # Mimic HF tuple output format: (last_hidden_state, pooled_output, hidden_states, attentions)
            outputs = (last_hidden_state, pooled_output)
            if output_hidden_states:
                outputs = outputs + (encoder_outputs[1] if isinstance(encoder_outputs, tuple) else encoder_outputs.hidden_states,)
            if output_attentions:
                outputs = outputs + (encoder_outputs[-1] if isinstance(encoder_outputs, tuple) else encoder_outputs.attentions,)
            return tuple(output for output in outputs if output is not None)


        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output, # Use the post-normed pooled output
            hidden_states=encoder_outputs.hidden_states if not isinstance(encoder_outputs, tuple) else encoder_outputs[2 if output_hidden_states else 1], # Adjust index based on tuple contents
            attentions=encoder_outputs.attentions if not isinstance(encoder_outputs, tuple) else encoder_outputs[-1],
        )


# Wrapper class similar to InstructCLIPVisionModel
class QASiglipVisionModel(SiglipPreTrainedModel):
    """
    QA-ViT enabled SiglipVisionModel. It replaces the standard vision_model
    with a QASiglipVisionTransformer that incorporates question awareness.
    """
    config_class = SiglipVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: SiglipVisionConfig, instruction_dim: int, integration_point: str):
        super().__init__(config)
        # Store QA-ViT specific config within the main config for convenience
        # This doesn't modify the saved config file, just the in-memory object
        self.config.instruction_dim = instruction_dim
        self.config.integration_point = integration_point

        # Instantiate the custom Vision Transformer backbone
        # The internal model structure is now self.vision_model like the original
        self.vision_model = QASiglipVisionTransformer(config, instruction_dim, integration_point)

        # Initialize weights and apply final processing (standard HF practice)
        # This handles tie_weights, etc. if applicable
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        # Return the patch embedding layer from the underlying vision model
        return self.vision_model.embeddings.patch_embedding

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model. Needs adaptation for QA layers. """
        # This would need careful implementation if pruning is required for QA layers
        logger.warning("Pruning heads is not fully implemented for QASiglip layers.")
        # Delegate to standard model's method if possible, but QA layers won't be pruned
        # Or iterate through layers and call prune_heads if the method exists on the layer
        for layer in self.vision_model.encoder.layers:
             if hasattr(layer.self_attn, "prune_heads"):
                 layer.self_attn.prune_heads(heads_to_prune)
        # raise NotImplementedError("Pruning heads is not supported for QASiglip layers yet.")


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
        # The QASiglipVisionModel acts as the main interface matching SiglipVisionModel
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
        logger.info("Freezing base SigLIP Vision Model parameters...")
        qa_param_prefixes = ("instruct_dim_reduce", "parallel_mlp", "parallel_gate")
        trainable_params = []
        total_params = 0
        trainable_count = 0

        for name, param in self.named_parameters():
            total_params += param.numel()
            is_qa_param = any(qp in name for qp in qa_param_prefixes)

            if not is_qa_param:
                param.requires_grad = False
            else:
                 param.requires_grad = True
                 trainable_params.append(name)
                 trainable_count += param.numel()

        logger.info(f"Total parameters: {total_params}")
        logger.info(f"Trainable QA-ViT parameters ({len(trainable_params)} tensors): {trainable_count}")
        # logger.debug(f"Trainable parameter names: {trainable_params}")


# --- End: Adapted QA-ViT components for SigLIP ---

if __name__=="__main__":
    # --- Configuration ---
    import torch, os
    from transformers import AutoImageProcessor, SiglipVisionConfig, SiglipVisionModel # Keep original for loading
    from PIL import Image

    os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Limit to GPU 0 if desired

    # Model ID for the pre-trained SigLIP model
    SIGLIP_MODEL_ID = "google/siglip-so400m-patch14-384"
    # Local path if downloaded: "../models/siglip"

    # --- QA-ViT Configuration ---
    QA_INTEGRATION_POINT = 'late' # Options: 'late', 'all', 'early', 'none', etc.
    # Dimension of instruction features (output of your text encoder)
    # SigLIP-so400m-patch14-384 has hidden_size 1152
    INSTRUCTION_DIM = 1152

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use float32 for stability, especially if pre_layrnorm loading fails.
    # Can switch back to float16 if GPU memory is tight and things work.
    compute_dtype = torch.float32 # Changed from float16 for potentially more stability
    logger.info(f"Using device: {device}, compute dtype: {compute_dtype}")


    # 1. Load Image Processor (remains the same)
    logger.info(f"Loading SigLIP Image Processor: {SIGLIP_MODEL_ID}...")
    try:
        siglip_processor = AutoImageProcessor.from_pretrained(SIGLIP_MODEL_ID)
        logger.info("Image Processor loaded.")
    except Exception as e:
        logger.error(f"Error loading SigLIP processor: {e}", exc_info=True)
        exit()

    # 2. Load SigLIP Vision Configuration
    logger.info(f"Loading SigLIP Vision Config: {SIGLIP_MODEL_ID}...")
    try:
        siglip_vision_config = SiglipVisionConfig.from_pretrained(SIGLIP_MODEL_ID)
        logger.info("SigLIP Vision Config loaded.")
    except Exception as e:
        logger.error(f"Error loading SigLIP vision config: {e}", exc_info=True)
        exit()

    # 3. Instantiate the Custom QASiglipVisionModel
    logger.info(f"Instantiating QASiglipVisionModel with integration='{QA_INTEGRATION_POINT}'...")
    # Pass the loaded config, instruction dim, and integration point
    qa_siglip_model = QASiglipVisionModel(
        config=siglip_vision_config,
        instruction_dim=INSTRUCTION_DIM,
        integration_point=QA_INTEGRATION_POINT
    )
    logger.info("QASiglipVisionModel instantiated.")
    # logger.debug(f"Model structure:\n{qa_siglip_model}") # Print structure if needed

    # 4. Load Pre-trained Weights from Original SigLIP Vision Model
    logger.info(f"Loading pre-trained weights from {SIGLIP_MODEL_ID} into QASiglipVisionModel...")
    try:
        # Load the standard model temporarily to get its state dict
        logger.info("  Loading original SiglipVisionModel to extract weights...")
        original_siglip_vision_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_ID)
        original_state_dict = original_siglip_vision_model.state_dict()
        logger.info("  Original state_dict loaded.")

        # Load into our custom model, ignoring missing/unexpected keys
        missing_keys, unexpected_keys = qa_siglip_model.load_state_dict(original_state_dict, strict=False)

        # Refined report: Separate expected missing from unexpected
        expected_missing_qa_prefixes = ("instruct_dim_reduce", "parallel_mlp", "parallel_gate")
        missing_qa_keys = [k for k in missing_keys if any(p in k for p in expected_missing_qa_prefixes)]
        missing_non_qa_keys = [k for k in missing_keys if k not in missing_qa_keys]

        logger.info("\nWeight loading report:")
        logger.info(f"  - Expected Missing Keys (QA-ViT params): {len(missing_qa_keys)} keys")
        # logger.debug(f"    {missing_qa_keys}") # Uncomment for full list
        if missing_non_qa_keys:
             logger.warning(f"  - UNEXPECTED Missing Keys (Check these!): {missing_non_qa_keys}")
             # Specifically check for pre_layrnorm
             if any("pre_layrnorm" in k for k in missing_non_qa_keys):
                 logger.error("    >---> CRITICAL: 'pre_layrnorm' weights seem missing from loaded state_dict!")
        else:
            logger.info("  - No unexpected missing keys found.")

        # Report unexpected keys (likely the contrastive head, which is OK to ignore)
        logger.info(f"  - Unexpected Keys (Ignored): {len(unexpected_keys)} keys")
        # logger.debug(f"    {unexpected_keys}") # Uncomment for full list
        if unexpected_keys and not all("head." in k for k in unexpected_keys):
             logger.warning(f"    >---> Some unexpected keys might not be from the head: {[k for k in unexpected_keys if 'head.' not in k]}")


        # Explicitly initialize the parallel gate to zero (as per paper)
        logger.info("  Initializing QA parallel gates to zero...")
        for name, param in qa_siglip_model.named_parameters():
            if "parallel_gate" in name:
                with torch.no_grad():
                    param.zero_()

        logger.info("Pre-trained weights loaded successfully (with expected mismatches).")
        del original_siglip_vision_model, original_state_dict # Free memory
        torch.cuda.empty_cache() # Try to clear cache

    except Exception as e:
        logger.error(f"Error loading SigLIP weights: {e}", exc_info=True)
        exit()

    # 5. Move model to device and set dtype
    logger.info(f"Moving model to {device} with dtype {compute_dtype}...")
    qa_siglip_model = qa_siglip_model.to(device=device, dtype=compute_dtype)
    logger.info("Model moved to device.")

    # 6. Freeze base model parameters for training (Optional but Recommended for PEFT)
    qa_siglip_model.freeze_base_model()

    # 7. Set to train or eval mode
    # qa_siglip_model.train() # If you are going to train the QA params
    qa_siglip_model.eval() # For inference
    logger.info(f"Model set to {'train' if qa_siglip_model.training else 'eval'} mode.")


    # --- Placeholder for Text Encoding ---
    # You MUST replace this with your actual text encoding pipeline.
    # The output dimension must match INSTRUCTION_DIM.
    def get_dummy_instruction_features(question_text, batch_size, seq_len, dim, device, dtype):
        """Generates dummy instruction features."""
        logger.warning(f"Using DUMMY instruction features for '{question_text}'")
        instruct_states = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)
        # Mask usually indicates valid tokens (1) vs padding (0)
        instruct_masks = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)
        return instruct_states, instruct_masks
    # --- End Placeholder ---


    # --- Example Usage ---
    logger.info("\n--- Example Usage ---")
    # Load a sample image
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg" # COCO bus image
    logger.info(f"Loading sample image from: {image_url}")
    try:
        # Handle potential download errors
        from PIL import Image
        import requests
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        logger.info("Sample image loaded.")

        # Prepare image input
        image_inputs = siglip_processor(images=image, return_tensors="pt")
        # Move inputs to device and cast dtype AFTER processing
        image_inputs = {k: v.to(device=device) for k, v in image_inputs.items()}
        pixel_values = image_inputs['pixel_values'].to(dtype=compute_dtype)


        # Prepare question input (using dummy function)
        question = "What color is the bus?"
        batch_size = pixel_values.shape[0] # Should be 1 for this example
        instr_seq_len = 16 # Dummy sequence length for instruction
        instruct_states, instruct_masks = get_dummy_instruction_features(
            question, batch_size, instr_seq_len, INSTRUCTION_DIM, device, compute_dtype
        )

        # Forward pass through the QA-SigLIP model
        logger.info("Performing forward pass...")
        with torch.no_grad(): # Use no_grad for inference
            outputs = qa_siglip_model(
                pixel_values=pixel_values,
                # Pass the encoded question features
                instruct_states=instruct_states,
                instruct_masks=instruct_masks, # Pass mask if your attention impl uses it
                output_hidden_states=False, # Optional: get all hidden states
                output_attentions=False, # Optional: get attention weights
                return_dict=True
            )

        # Access outputs
        last_hidden_state = outputs.last_hidden_state # Shape: (batch, num_patches + 1, hidden_size)
        pooled_output = outputs.pooler_output       # Shape: (batch, hidden_size) - CLS token output after post_layernorm

        logger.info(f"Forward pass successful.")
        logger.info(f"  - Last Hidden State shape: {last_hidden_state.shape}")
        logger.info(f"  - Pooled Output shape: {pooled_output.shape}")
        # You would typically feed these features (e.g., last_hidden_state[:, 1:, :])
        # into your decoder/LLM, possibly after a projection layer.

    except Exception as e:
        logger.error(f"Error during example usage: {e}", exc_info=True)