# --- Start: Adapted QA-ViT components for EVA-ViT-G ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from functools import partial
import torch.utils.checkpoint as checkpoint
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from typing import Optional, Tuple, Union
from transformers import AutoImageProcessor
from torch.cuda.amp import autocast

# --- Original EVA-ViT-G Components (Modified slightly or used as base) ---
# Based on EVA, BEIT, timm and DeiT code bases
# ... (Keep DropPath, Mlp, PatchEmbed, RelativePositionBias as provided in the prompt) ...
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Attention(nn.Module):
    # ... (no changes needed here, autocast will handle F.linear) ...
     def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            # Initialize bias parameters directly with the desired dtype later if needed,
            # but autocast usually handles the F.linear call correctly even if bias is fp32.
            # Let's keep the original init for now.
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            # Use indexing='ij' for newer torch versions if needed, keep original if it works
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

     def forward(self, x, rel_pos_bias=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            # Ensure bias components match the expected type for F.linear within autocast
            # Usually autocast handles this, but if issues persist, try casting bias:
            # bias_dtype = x.dtype if torch.is_autocast_enabled() else self.q_bias.dtype
            # Or more simply, match the weight dtype if available: bias_dtype = self.qkv.weight.dtype
            # bias_dtype = self.q_bias.dtype # Keep original for now
            # qkv_bias = torch.cat((
            #     self.q_bias.to(bias_dtype),
            #     torch.zeros_like(self.v_bias, requires_grad=False).to(bias_dtype),
            #     self.v_bias.to(bias_dtype)
            # ))
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))

        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # Autocast handles the F.linear dtype matching
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            # Ensure relative pos bias matches attention score dtype
            # Autocast might handle this promotion, but explicit cast is safer if needed
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0).to(attn.dtype) # Explicit cast

        if rel_pos_bias is not None:
             # Ensure external rel_pos_bias also matches attention score dtype
             attn = attn + rel_pos_bias.to(attn.dtype) # Explicit cast

        # Softmax is usually kept in fp32 by autocast for stability
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class Mlp(nn.Module): # Original MLP from EVA-ViT-G
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class RelativePositionBias(nn.Module):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij')) # Use indexing='ij' for newer torch versions
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 use_mean_pooling=True, init_scale=0.001, use_checkpoint=False):
        super().__init__()
        self.image_size = img_size
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None
        self.use_checkpoint = use_checkpoint

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None)
            for i in range(depth)])
        #         self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        #         self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        #         self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)
        #         if isinstance(self.head, nn.Linear):
        #             trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

    #         if isinstance(self.head, nn.Linear):
    #             self.head.weight.data.mul_(init_scale)
    #             self.head.bias.data.mul_(init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, rel_pos_bias)
            else:
                x = blk(x, rel_pos_bias)
        return x

    #         x = self.norm(x)

    #         if self.fc_norm is not None:
    #             t = x[:, 1:, :]
    #             return self.fc_norm(t.mean(1))
    #         else:
    #             return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        #         x = self.head(x)
        return x

    def get_intermediate_layers(self, x):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        features = []
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias)
            features.append(x)

        return features

    def get_num_layer(self, var_name=""):
        if var_name in ("cls_token", "mask_token", "pos_embed"):
            return 0
        elif var_name.startswith("patch_embed"):
            return 0
        elif var_name.startswith("rel_pos_bias"):
            return len(self.blocks) - 1
        elif var_name.startswith("blocks"):
            layer_id = int(var_name.split('.')[1])
            return layer_id + 1
        else:
            return len(self.blocks)

# --- QA-ViT specific Components Adapted for EVA-ViT-G ---

# Helper MLP (similar to original QA-SigLIP MLP and EVA's Mlp)
class QAEVAMlp(Mlp): # Inherits from EVA's Mlp for structure consistency
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__(in_features, hidden_features, out_features, act_layer, drop)
        # Structure is identical to EVA's Mlp, just a separate class name for clarity

# Modified Attention to handle instruction fusion via concatenation
class QAEVAAttention(Attention): # Inherit from EVA's Attention
    """
    Adapts EVA's Attention to concatenate instruction tokens (kv_states)
    before the self-attention mechanism.
    Skips relative position bias when instructions are provided for simplicity.
    """
    def forward(
        self,
        x: torch.Tensor,
        rel_pos_bias: Optional[torch.Tensor] = None,
        # QA-ViT specific inputs
        kv_states: Optional[torch.Tensor] = None, # Projected instruction features
        kv_masks: Optional[torch.Tensor] = None, # Mask for instruction features (unused here)
    ) -> torch.Tensor:

        B, N, C = x.shape # N = num_visual_tokens (including CLS)
        num_visual_tokens = N

        if kv_states is None:
            # Fallback to standard EVA Attention if no instruction provided
            return super().forward(x, rel_pos_bias=rel_pos_bias)

        # --- QA-ViT Fusion Logic ---
        # Concatenate instruction tokens (kv_states) with visual tokens (x)
        # kv_states are assumed to be already projected to the correct dimension (C)
        _, num_instr_tokens, _ = kv_states.size()
        concat_states = torch.cat([kv_states, x], dim=1)
        concat_seq_len = concat_states.size(1) # num_instr_tokens + num_visual_tokens

        # --- Standard EVA Attention applied to concatenated sequence (modified QKV step) ---
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))

        # Project concatenated states to Q, K, V
        qkv = F.linear(input=concat_states, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, concat_seq_len, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # Shape: (B, num_heads, concat_seq_len, head_dim)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # Shape: (B, num_heads, concat_seq_len, concat_seq_len)

        # --- Relative Position Bias Handling (Skipped for QA-ViT path) ---
        # NOTE: Applying rel_pos_bias correctly to the concatenated sequence is complex.
        # For simplicity, we skip adding relative position bias when kv_states are present.
        # The interaction between visual tokens and instruction tokens doesn't have a
        # natural relative positional encoding in the original EVA sense.
        # if rel_pos_bias is not None:
        #     print("Warning: Skipping relative position bias addition in QAEVAAttention due to instruction fusion.")
            # attn = attn + rel_pos_bias # This would be incorrect dimensionally

        # --- QA-ViT Output Selection ---
        # We only care about the output for the original visual tokens (queries)
        # Attention weights for visual queries attending to all keys (instruction + visual)
        attn_weights_vis_queries = attn[:, :, num_instr_tokens:, :] # Shape: (B, num_heads, N, concat_seq_len)

        # Apply softmax to the attention scores corresponding to visual queries
        attn_probs = attn_weights_vis_queries.softmax(dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        # Compute attention output using the probabilities for visual queries and *all* value states
        # (B, num_heads, N, concat_seq_len) x (B, num_heads, concat_seq_len, head_dim) -> (B, num_heads, N, head_dim)
        attn_output = attn_probs @ v # Note: v has shape (B, num_heads, concat_seq_len, head_dim)

        # Reshape back to (B, N, C) where N is the original number of visual tokens
        attn_output = attn_output.transpose(1, 2).reshape(B, num_visual_tokens, -1) # -1 calculates C = num_heads * head_dim
        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output)
        # --- End QA-ViT Specific Attention Logic ---

        return attn_output


# Modified Transformer Block with optional QA-ViT components
class QAEVABlock(Block): # Inherit from EVA's Block
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None,
                 # QA-ViT specific args
                 instruction_dim: int = 0 # Pass the dimension of the raw instruction features
                ):
        # Initialize the original Block first
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                         drop_path, init_values, act_layer, norm_layer, window_size, attn_head_dim)

        # --- Replace standard Attention with QA-Attention ---
        self.attn = QAEVAAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)

        # --- QA-ViT Specific Components ---
        # MLP to project instruction features to ViT dimension (dim)
        # Input dim is instruction_dim, output dim is ViT hidden_size (dim)
        if instruction_dim > 0:
             self.instruct_dim_reduce = nn.Sequential(
                nn.LayerNorm(instruction_dim), # Normalize instruction features first
                nn.Linear(instruction_dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim)
             )
        else:
            self.instruct_dim_reduce = None # No projection needed if dim matches or no instructions

        # Parallel Gated Projection Path (parallel to self.mlp)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.parallel_mlp = QAEVAMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.parallel_gate = nn.Parameter(torch.zeros(1)) # Initialize beta to 0

    def forward(
        self,
        x: torch.Tensor,
        rel_pos_bias: Optional[torch.Tensor] = None,
        # QA-ViT specific inputs
        instruct_states: Optional[torch.Tensor] = None,
        instruct_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # Project instructions if provided
        kv_states = None
        if instruct_states is not None and self.instruct_dim_reduce is not None:
             kv_states = self.instruct_dim_reduce(instruct_states)
        elif instruct_states is not None: # Assume instruction_dim already matches dim
            kv_states = instruct_states

        # --- Attention Path ---
        # Note: LayerScale (gamma_1) is applied *after* the attention + residual
        attn_input = self.norm1(x)
        attn_output = self.attn(
            attn_input,
            rel_pos_bias=rel_pos_bias,
            kv_states=kv_states,
            kv_masks=instruct_masks # Pass mask if needed by attention impl. (unused here)
        )

        if self.gamma_1 is None:
            x = x + self.drop_path(attn_output)
        else:
            x = x + self.drop_path(self.gamma_1 * attn_output)


        # --- MLP Path with Parallel Gated Projection ---
        mlp_input = self.norm2(x)

        # Original MLP path
        mlp_output = self.mlp(mlp_input)

        # Parallel Gated Path
        parallel_output = self.parallel_mlp(mlp_input)
        gate_value = torch.tanh(self.parallel_gate) # tanh(beta)
        gated_parallel_output = parallel_output * gate_value

        # Combine original MLP, gated parallel output, residual, and LayerScale (gamma_2)
        if self.gamma_2 is None:
            x = x + self.drop_path(mlp_output + gated_parallel_output) # Add parallel output here
        else:
            # Apply gamma_2 to the combined MLP outputs before adding residual
            x = x + self.drop_path(self.gamma_2 * (mlp_output + gated_parallel_output))

        return x

# The main Vision Transformer using the QA-Blocks
class QAEVAVisionTransformer(VisionTransformer): # Inherit from EVA's VisionTransformer
    """
    Vision Transformer based on EVA-ViT-G, adapted with QA-ViT components.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12, # num_classes=0 for feature extraction
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 use_mean_pooling=False, init_scale=0.001, use_checkpoint=False,
                 # QA-ViT specific args
                 instruction_dim: int = 0,
                 integration_point: str = 'none' # 'all', 'late', 'early', 'none'
                ):
        # Initialize the base VisionTransformer but DON'T create blocks yet
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, embed_dim=embed_dim, depth=depth,
                         num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=0., norm_layer=norm_layer, init_values=init_values, # drop_path applied later
                         use_abs_pos_emb=use_abs_pos_emb, use_rel_pos_bias=use_rel_pos_bias, use_shared_rel_pos_bias=use_shared_rel_pos_bias,
                         use_mean_pooling=use_mean_pooling, init_scale=init_scale, use_checkpoint=use_checkpoint)

        # Store QA config
        self.instruction_dim = instruction_dim
        self.integration_point = integration_point

        # --- Override the blocks creation ---
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList()
        for i in range(depth):
            # Determine if this layer should be QA-ViT enabled
            is_qa_layer = False
            if integration_point == 'all':
                is_qa_layer = True
            elif integration_point == 'late':
                is_qa_layer = (i >= depth // 2)
            elif integration_point == 'early':
                 is_qa_layer = (i < depth // 2)
            elif integration_point == 'none':
                is_qa_layer = False
            else:
                raise ValueError(f"Unsupported integration_point: {integration_point}")

            if is_qa_layer:
                print(f"Layer {i}: Using QAEVABlock")
                layer = QAEVABlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                    instruction_dim=instruction_dim # Pass instruction_dim here
                )
            else:
                # Use the standard EVA Block
                print(f"Layer {i}: Using standard Block")
                layer = Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None)
            self.blocks.append(layer)

        # --- Re-apply post-initialization steps from original __init__ ---
        # (These were likely done in the original super().__init__ but might need re-triggering or were done after block creation)
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights) # Re-apply initialization
        self.fix_init_weight()      # Re-apply weight fixing

        # --- Add final norm if needed (original EVA model structure implies it might be outside the loop) ---
        # Based on the original provided code, the final norm was commented out or applied differently.
        # Let's add a final norm for consistency with typical ViT outputs before pooling.
        self.norm = norm_layer(embed_dim)


    # Override forward_features to pass instruction features
    def forward_features(self, x, instruct_states=None, instruct_masks=None):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        for blk in self.blocks:
            # Pass instruction features only to QAEVABlock instances
            if isinstance(blk, QAEVABlock):
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x, rel_pos_bias, instruct_states, instruct_masks,use_reentrant=False)
                else:
                    x = blk(x, rel_pos_bias=rel_pos_bias, instruct_states=instruct_states, instruct_masks=instruct_masks)
            else: # Standard Block
                if self.use_checkpoint:
                    # Checkpoint standard block signature (might need adjustment if rel_pos_bias is optional)
                     x = checkpoint.checkpoint(blk, x, rel_pos_bias,use_reentrant=False)
                else:
                    x = blk(x, rel_pos_bias=rel_pos_bias)

        # Apply final layer norm
        x = self.norm(x)
        return x # Return all token features (including CLS) after final norm

    # Override forward to accept instruction features
    def forward(self, x, instruct_states=None, instruct_masks=None):
        x = self.forward_features(x, instruct_states=instruct_states, instruct_masks=instruct_masks)
        # The original EVA forward had a head, but we removed it for feature extraction
        # Output: (batch_size, num_tokens, embed_dim)
        # Typically for downstream tasks, you might take the CLS token: x[:, 0]
        # Or perform mean pooling: x[:, 1:].mean(dim=1)
        return x

    # --- Add QA-ViT specific methods ---
    def freeze_base_model(self):
        """Freezes all parameters except those specific to QA-ViT."""
        print("Freezing base EVA-ViT-G Model parameters...")
        print("Keeping trainable EVA-ViT-G Model parameters:")
        for name, param in self.named_parameters():
            is_qa_param = False
            # Identify QA-ViT specific parameters
            if "instruct_dim_reduce" in name: is_qa_param = True
            if "parallel_mlp" in name: is_qa_param = True
            if "parallel_gate" in name: is_qa_param = True
            # Check if it's within a QAEVABlock's QAEVAAttention (less direct way)
            # A simpler check is based on module names defined only in QA blocks.

            if not is_qa_param:
                param.requires_grad = False
            else:
                 print(f"  - Keeping trainable: {name}")

    # --- NEW: Method to save only trainable weights ---
    def save_trainable_weight(self, save_path):
        """
        仅保存模型中 requires_grad=True 的参数。

        Args:
            save_path (str): 保存权重的文件路径。
        """
        trainable_state_dict = {}
        saved_keys = []
        print(f"准备保存可训练权重到: {save_path}")
        for name, param in self.named_parameters():
            if param.requires_grad:
                # 推荐：克隆张量并移至 CPU 保存，以提高兼容性
                trainable_state_dict[name] = param.data.clone().cpu()
                saved_keys.append(name)
                # print(f"  - 将保存: {name}") # 可以取消注释以获得更详细的输出

        if not saved_keys:
            print("警告：模型中没有找到可训练的权重 (requires_grad=True)，未保存任何内容。")
            return

        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(trainable_state_dict, save_path)
            print(f"成功将 {len(saved_keys)} 个可训练参数保存到 {save_path}")
            # print(f"  - 保存的键: {saved_keys}") # 可以取消注释以查看具体键名
        except Exception as e:
            print(f"错误：保存可训练权重到 {save_path} 失败: {e}")

    # --- NEW: Method to load only trainable weights ---
    def load_trainable_weight(self, load_path, strict=True):
        """
        加载权重到模型中 requires_grad=True 的参数。

        Args:
            load_path (str): 要加载的权重文件路径。
            strict (bool): 如果为 True，确保所有 requires_grad=True 的参数
                           都在加载的文件中找到对应的键。
                           如果为 False，则只加载能匹配上的键。
        """
        if not os.path.exists(load_path):
            print(f"错误：找不到权重文件 {load_path}，无法加载。")
            return

        print(f"准备从 {load_path} 加载可训练权重...")
        # 加载到 CPU 以避免设备问题，稍后会移动到参数的实际设备
        loaded_state_dict = torch.load(load_path, map_location='cpu')
        loaded_keys = set(loaded_state_dict.keys())
        current_trainable_keys = set()
        not_found_in_file = []
        loaded_count = 0

        # self.eval() # 通常在加载权重时设置为评估模式
        with torch.no_grad(): # 禁用梯度计算以进行加载
            for name, param in self.named_parameters():
                if param.requires_grad:
                    current_trainable_keys.add(name)
                    if name in loaded_state_dict:
                        # 将加载的张量移动到参数所在的设备
                        loaded_tensor = loaded_state_dict[name].to(dtype=param.dtype)
                        # 检查形状是否匹配
                        if loaded_tensor.shape == param.data.shape:
                             param.data.copy_(loaded_tensor)
                             # print(f"  - 已加载: {name}") # 可以取消注释以获得更详细的输出
                             loaded_count += 1
                        else:
                             print(f"  - 警告：形状不匹配，跳过加载 {name}。"
                                   f" 模型需要: {param.data.shape}, 文件提供: {loaded_tensor.shape}")
                    else:
                        not_found_in_file.append(name)
                        # print(f"  - 警告: 可训练参数 {name} 在加载文件中未找到。") # 可以取消注释

        print(f"权重加载完成。共加载了 {loaded_count} 个参数。")

        # 严格模式检查
        if strict and not_found_in_file:
            raise KeyError(f"严格模式错误：以下可训练参数在文件 {load_path} 中未找到: {not_found_in_file}")
        elif not_found_in_file:
            print(f"警告：以下 {len(not_found_in_file)} 个当前可训练参数在加载文件中未找到: {not_found_in_file}")

        # 检查是否有加载文件中存在但当前模型中不可训练的键 (可选，但有助于调试)
        unused_keys = loaded_keys - current_trainable_keys
        if unused_keys:
            print(f"提示：加载文件中的以下 {len(unused_keys)} 个键在当前模型中不是可训练参数或不存在: {unused_keys}")

# --- Factory function to create and load the QA-EVA-ViT-G model ---
# (Adapting the original create_eva_vit_g function)

def create_qa_eva_vit_g(
    img_size=224,
    drop_path_rate=0.4,
    use_checkpoint=False,
    precision=torch.float16, # Default to fp32 for broader compatibility, change if needed
    # QA-ViT Config
    instruction_dim: int = 1408, # Default to EVA-G embed_dim, **MUST** match text encoder output
    integration_point: str = 'late',
    cached_file: str = r'../models/eva_vit_g.pth'
    ):

    # Instantiate the custom QA-ViT model
    print(f"Instantiating QAEVAVisionTransformer with integration='{integration_point}'...")
    model = QAEVAVisionTransformer(
        img_size=img_size,
        patch_size=14,
        use_mean_pooling=False, # Get all tokens
        embed_dim=1408,
        depth=39,
        num_heads=1408//88,
        mlp_ratio=4.3637,
        qkv_bias=True,
        drop_path_rate=drop_path_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_checkpoint=use_checkpoint,
        use_abs_pos_emb=True,     # EVA-G uses absolute pos embedding
        use_rel_pos_bias=False,   # EVA-G does not use relative pos bias blocks
        init_values=None,         # EVA-G does not use LayerScale init values (gammas are learned)
        # QA-ViT Args
        instruction_dim=instruction_dim,
        integration_point=integration_point
    )
    print("QAEVAVisionTransformer instantiated.")

    cached_file = cached_file
    try:
        state_dict = torch.load(cached_file, map_location="cpu")

        # 如果需要，插入位置嵌入（调整原始函数） ##sd
        interpolate_pos_embed(model, state_dict)

        # Load into our custom model, ignoring missing/unexpected keys
        print("Loading state dict (strict=False)...")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        print("\nWeight loading report:")
        print(f"  - Missing keys (expected QA-ViT params): {missing_keys}")
        print(f"  - Unexpected keys (should be empty or related to head/norm): {unexpected_keys}")
        # It's normal for 'norm.weight', 'norm.bias', 'head.weight', 'head.bias' to be unexpected
        # if the original checkpoint had them and our QAEVAVisionTransformer doesn't define 'head'.
        # Our QAEVAVisionTransformer *does* define `self.norm`, so those should load if present.

        # 显式地将并行门初始化为零（根据论文）
        for name, param in model.named_parameters():
            # if "parallel_gate" in name:
            #     with torch.no_grad():
            #         param.zero_()
            # Optional: Initialize parallel MLP path weights similar to original MLP (might help convergence)
            # This requires careful name matching between parallel_mlp and mlp within the same QAEVABlock
            # Example (needs verification based on actual parameter names):
            if 'parallel_mlp' in name:
                 original_mlp_name = name.replace('parallel_mlp', 'mlp')
                 if original_mlp_name in state_dict:
                     with torch.no_grad():
                          param.copy_(state_dict[original_mlp_name])

        print("Pre-trained weights loaded successfully.")

    except Exception as e:
        print(f"Error loading EVA-ViT-G weights: {e}")
        # Decide how to handle error - maybe return uninitialized model or raise exception
        raise e # Re-raise the exception

    # # Handle precision
    # if precision == "fp16":
    #     print("Converting model weights to FP16...")
    #     convert_weights_to_fp16(model) # Use the provided conversion function
    # elif precision == "bf16":
    #     print("Converting model weights to BF16...")
    #     model = model.to(dtype=torch.bfloat16)

    return model

def interpolate_pos_embed(model, checkpoint_model):
    # Adapt the function provided in the prompt slightly
    if 'pos_embed' in checkpoint_model and model.pos_embed is not None:
        pos_embed_checkpoint = checkpoint_model['pos_embed'].float()
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches # Should be 1 for CLS token

        if pos_embed_checkpoint.shape[-2] == num_extra_tokens + num_patches:
             print("Position embedding size matches. No interpolation needed.")
             return # Sizes match, no need to interpolate

        print(f"Position embedding size mismatch: Checkpoint {pos_embed_checkpoint.shape} vs Model {model.pos_embed.shape}. Interpolating...")

        # height (== width) for the checkpoint position embedding
        orig_num_patches = pos_embed_checkpoint.shape[-2] - num_extra_tokens
        # Handle cases where orig_num_patches might not be a perfect square
        if orig_num_patches <= 0:
             print(f"Warning: Cannot interpolate position embedding with non-positive number of patches in checkpoint ({orig_num_patches}). Skipping.")
             # Copy the original embedding without the spatial part if possible, or skip entirely
             checkpoint_model['pos_embed'] = model.pos_embed.data # Fallback or error
             return

        orig_size = int(math.sqrt(orig_num_patches))
        if orig_size * orig_size != orig_num_patches:
             print(f"Warning: Checkpoint position embedding patch count ({orig_num_patches}) is not a perfect square. Interpolation may be inaccurate.")
             # Attempt interpolation anyway, but results might be suboptimal

        # height (== width) for the new position embedding
        new_size = int(math.sqrt(num_patches))
        if new_size * new_size != num_patches:
             print(f"Error: Model position embedding patch count ({num_patches}) is not a perfect square. Cannot interpolate.")
             raise ValueError("Model patch count not a perfect square.")


        print(f"Interpolating position embedding from {orig_size}x{orig_size} to {new_size}x{new_size}")
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed
    elif 'pos_embed' not in checkpoint_model:
        print("Warning: 'pos_embed' not found in checkpoint. Cannot interpolate.")
    elif model.pos_embed is None:
         print("Info: Model does not use absolute position embeddings. Skipping interpolation.")


def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()
        # Add conversion for LayerNorm if needed, though often kept in fp32 for stability
        # if isinstance(l, nn.LayerNorm):
        #     l.weight.data = l.weight.data.half()
        #     l.bias.data = l.bias.data.half()

    model.apply(_convert_weights_to_fp16)
# --- End: Adapted QA-ViT components for EVA-ViT-G ---


if __name__=="__main__":
    import os
    from PIL import Image
    # Assuming a compatible image processor (e.g., from CLIP or BLIP2)
    # For demonstration, we'll use a simple transform. Replace with your actual processor.
    from torchvision import transforms

    os.environ["CUDA_VISIBLE_DEVICES"] = "5" # Optional: Select GPU

    # --- QA-ViT Configuration for EVA-ViT-G ---
    QA_INTEGRATION_POINT = 'late' # Options: 'late', 'all', 'early', 'none'
    # EVA-ViT-G embed_dim is 1408. Instruction dim must match text encoder output.
    INSTRUCTION_DIM = 1408 # Example: if using EVA-G itself or compatible encoder
    EVA_IMG_SIZE = 224
    DROP_PATH_RATE = 0.0 # Set drop path rate (original EVA-G used 0.4 during pre-training)
    PRECISION = torch.float16 # "torch.float16", "torch.bfloat16", "torch.float32"
    IMAGE_PROCESSOR_ID="../models/vit_image_processor"

    # Other settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # 1. Instantiate and Load the Custom QA-EVA-ViT-G Model
    print(f"Creating QA-EVA-ViT-G model...")
    try:
        qa_eva_model = create_qa_eva_vit_g(
            img_size=EVA_IMG_SIZE,
            drop_path_rate=DROP_PATH_RATE,
            use_checkpoint=False, # Set to True if needed for memory saving during training
            precision=PRECISION,
            instruction_dim=INSTRUCTION_DIM,
            integration_point=QA_INTEGRATION_POINT
        )
        print("QA-EVA-ViT-G model created and weights loaded.")
    except Exception as e:
        print(f"Error creating or loading QA-EVA-ViT-G model: {e}")
        exit()

    # 2. Move model to device
    # Note: create_qa_eva_vit_g handles precision conversion, just move the structure
    qa_eva_model = qa_eva_model.to(device)
    print(f"Model moved to {device}.")

    # 3. Freeze base model parameters for training (Optional but Recommended)
    qa_eva_model.freeze_base_model()

    # 4. Set to train or eval mode
    # qa_eva_model.train() # If you are going to train the QA params
    qa_eva_model.eval() # If you are just doing inference (QA params untrained)

    # --- Placeholder for Text Encoding ---
    # You NEED to replace this with your actual text processing pipeline
    # The output dimension *must* match INSTRUCTION_DIM (1408 in this example)
    def get_instruction_features(question_text, batch_size, seq_len, dim, device, dtype):
        """Placeholder function to generate dummy instruction features"""
        print(f"Warning: Using DUMMY instruction features for '{question_text}'")
        # Replace with your actual tokenizer and text encoder (e.g., T5, BERT, etc.)
        # Ensure the output is (batch_size, seq_len, INSTRUCTION_DIM)
        instruct_states = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)
        instruct_masks = torch.ones(batch_size, seq_len, device=device, dtype=torch.long) # Example mask
        return instruct_states, instruct_masks
    # --- End Placeholder ---

    # --- Example Usage ---
    print("\n--- Example Usage ---")
    # Load a sample image
    url = "./000000039769.jpg" # Make sure this image exists
    try:
        image = Image.open(url).convert("RGB")
        print("Sample image loaded.")

        # Prepare image input (using a simplified transform - REPLACE with actual processor)
        # This transform should match what the model expects (e.g., normalization from EVA/BLIP2)
        # Example using typical ImageNet normalization:
        print(f"Loading Image Processor from: {IMAGE_PROCESSOR_ID}...")
        image_processor = AutoImageProcessor.from_pretrained(IMAGE_PROCESSOR_ID)
        pixel_values = image_processor(images=image, return_tensors="pt")['pixel_values'].to(device=device, dtype=PRECISION) # Add batch dim

        # Prepare question input (using placeholder function)
        question = "What color is the bus?"
        batch_size = pixel_values.shape[0]
        instr_seq_len = 16 # Dummy sequence length
        instruct_states, instruct_masks = get_instruction_features(
             question, batch_size, instr_seq_len, INSTRUCTION_DIM, device, PRECISION
        )

        # Forward pass through the QA-EVA-ViT model
        print("Performing forward pass...")
        enable_autocast = (device.type == 'cuda') and (PRECISION in ["fp16", "bf16"])
        with autocast(enabled=enable_autocast, dtype=PRECISION):
            outputs = qa_eva_model(
                x=pixel_values,
                instruct_states=instruct_states,
                instruct_masks=instruct_masks
            )
        # with torch.no_grad(): # Use no_grad for inference example
        #     outputs = qa_eva_model(
        #         x=pixel_values,
        #         # Pass the encoded question features
        #         instruct_states=instruct_states,
        #         instruct_masks=instruct_masks # Pass masks if your attention layer uses them
        #     )

        # Access outputs
        # Output is the sequence of token features (batch, num_tokens, embed_dim)
        # num_tokens = num_patches + 1 (CLS token)
        all_token_features = outputs
        cls_token_feature = outputs[:, 0] # Extract CLS token feature

        print(f"Forward pass successful.")
        print(f"  - All Token Features shape: {all_token_features.shape}")
        print(f"  - CLS Token Feature shape: {cls_token_feature.shape}")
        # You would typically feed these features (e.g., all_token_features or cls_token_feature)
        # into your decoder/LLM.

    except FileNotFoundError:
        print(f"Error: Sample image not found at {url}. Please provide a valid image path.")
    except Exception as e:
        print(f"Error during example usage: {e}")
        import traceback
        traceback.print_exc()

    TRAINABLE_WEIGHTS_PATH='./test/qvit_trainable_weight.pt'
    # --- 模拟训练和保存/加载可训练权重 ---
    print("\n--- 模拟训练和保存/加载可训练权重 ---")

    # 假设我们进行了一些训练，修改了可训练权重...
    # (这里我们仅作演示，实际中你需要运行训练循环)
    # 举例：手动修改一个可训练参数的值 (仅作演示)
    for name, param in qa_eva_model.named_parameters():
         if param.requires_grad and "parallel_gate" in name:
             with torch.no_grad():
                 param.data += 0.1 # 模拟训练更新
             print(f"模拟训练：修改了参数 {name}")
             break # 只修改一个作示例

    # 4. 保存可训练权重
    print(f"\n保存可训练权重到: {TRAINABLE_WEIGHTS_PATH}")
    qa_eva_model.save_trainable_weight(TRAINABLE_WEIGHTS_PATH)

    # --- 创建一个新模型实例来测试加载 ---
    print("\n--- 创建新模型实例并加载可训练权重 ---")
    try:
        new_qa_eva_model = create_qa_eva_vit_g(
            img_size=EVA_IMG_SIZE,
            drop_path_rate=DROP_PATH_RATE,
            use_checkpoint=False,
            precision=PRECISION,
            instruction_dim=INSTRUCTION_DIM,
            integration_point=QA_INTEGRATION_POINT,
        )
        new_qa_eva_model = new_qa_eva_model.to(device)
        if PRECISION == torch.float16: new_qa_eva_model.half()
        elif PRECISION == torch.bfloat16: new_qa_eva_model.bfloat16()

        # 同样冻结基础模型，确保可训练参数被正确识别
        new_qa_eva_model.freeze_base_model()
        print("新模型实例创建完成并冻结了基础参数。")

        # 5. 加载之前保存的可训练权重
        print(f"\n从 {TRAINABLE_WEIGHTS_PATH} 加载可训练权重到新模型...")
        new_qa_eva_model.load_trainable_weight(TRAINABLE_WEIGHTS_PATH, strict=True)

        # (可选) 验证加载是否成功 (例如，检查之前修改的那个参数)
        print("\n验证加载结果...")
        for name, param in new_qa_eva_model.named_parameters():
            if param.requires_grad and "parallel_gate" in name:
                print(f"新模型中参数 {name} 的值 (应包含模拟训练的修改): {param.data.item()}")
                break

    except Exception as e:
        print(f"错误：在加载或验证新模型时出错: {e}")
        import traceback
        traceback.print_exc()