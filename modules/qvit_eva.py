
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
        self.num_features = self.embed_dim = embed_dim

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

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None)
            for i in range(depth)])

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()


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

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
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


    def forward(self, x):
        x = self.forward_features(x)
        return x

    def get_intermediate_layers(self, x):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
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

class QAEVAMlp(Mlp):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__(in_features, hidden_features, out_features, act_layer, drop)

class QAEVAAttention(Attention):
    """
    Adapts EVA's Attention to concatenate instruction tokens (kv_states)
    before the self-attention mechanism.
    Skips relative position bias when instructions are provided for simplicity.
    """
    def forward(
        self,
        x: torch.Tensor,
        rel_pos_bias: Optional[torch.Tensor] = None,
        kv_states: Optional[torch.Tensor] = None,
        kv_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        B, N, C = x.shape
        num_visual_tokens = N

        if kv_states is None:
            return super().forward(x, rel_pos_bias=rel_pos_bias)

        _, num_instr_tokens, _ = kv_states.size()
        concat_states = torch.cat([kv_states, x], dim=1)
        concat_seq_len = concat_states.size(1)

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))

        qkv = F.linear(input=concat_states, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, concat_seq_len, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn_weights_vis_queries = attn[:, :, num_instr_tokens:, :]

        attn_probs = attn_weights_vis_queries.softmax(dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        attn_output = attn_probs @ v
        attn_output = attn_output.transpose(1, 2).reshape(B, num_visual_tokens, -1)
        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output)

        return attn_output


class QAEVABlock(Block):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None,
                 instruction_dim: int = 0
                ):

        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                         drop_path, init_values, act_layer, norm_layer, window_size, attn_head_dim)

        self.attn = QAEVAAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)

        if instruction_dim > 0:
             self.instruct_dim_reduce = nn.Sequential(
                nn.LayerNorm(instruction_dim),
                nn.Linear(instruction_dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim)
             )
        else:
            self.instruct_dim_reduce = None

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.parallel_mlp = QAEVAMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.parallel_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        rel_pos_bias: Optional[torch.Tensor] = None,
        instruct_states: Optional[torch.Tensor] = None,
        instruct_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        kv_states = None
        if instruct_states is not None and self.instruct_dim_reduce is not None:
             kv_states = self.instruct_dim_reduce(instruct_states)
        elif instruct_states is not None:
            kv_states = instruct_states

        attn_input = self.norm1(x)
        attn_output = self.attn(
            attn_input,
            rel_pos_bias=rel_pos_bias,
            kv_states=kv_states,
            kv_masks=instruct_masks
        )

        if self.gamma_1 is None:
            x = x + self.drop_path(attn_output)
        else:
            x = x + self.drop_path(self.gamma_1 * attn_output)


        mlp_input = self.norm2(x)

        mlp_output = self.mlp(mlp_input)

        parallel_output = self.parallel_mlp(mlp_input)
        gate_value = torch.tanh(self.parallel_gate)
        gated_parallel_output = parallel_output * gate_value

        if self.gamma_2 is None:
            x = x + self.drop_path(mlp_output + gated_parallel_output)
        else:
            x = x + self.drop_path(self.gamma_2 * (mlp_output + gated_parallel_output))

        return x

class QAEVAVisionTransformer(VisionTransformer):
    """
    Vision Transformer based on EVA-ViT-G, adapted with QA-ViT components.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12, # num_classes=0 for feature extraction
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 use_mean_pooling=False, init_scale=0.001, use_checkpoint=False,
                 instruction_dim: int = 0,
                 integration_point: str = 'none'
                ):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, embed_dim=embed_dim, depth=depth,
                         num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=0., norm_layer=norm_layer, init_values=init_values, # drop_path applied later
                         use_abs_pos_emb=use_abs_pos_emb, use_rel_pos_bias=use_rel_pos_bias, use_shared_rel_pos_bias=use_shared_rel_pos_bias,
                         use_mean_pooling=use_mean_pooling, init_scale=init_scale, use_checkpoint=use_checkpoint)

        self.instruction_dim = instruction_dim
        self.integration_point = integration_point

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList()
        for i in range(depth):
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
                    instruction_dim=instruction_dim
                )
            else:
                print(f"Layer {i}: Using standard Block")
                layer = Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None)
            self.blocks.append(layer)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()
        self.norm = norm_layer(embed_dim)


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
            if isinstance(blk, QAEVABlock):
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x, rel_pos_bias, instruct_states, instruct_masks,use_reentrant=False)
                else:
                    x = blk(x, rel_pos_bias=rel_pos_bias, instruct_states=instruct_states, instruct_masks=instruct_masks)
            else:
                if self.use_checkpoint:
                     x = checkpoint.checkpoint(blk, x, rel_pos_bias,use_reentrant=False)
                else:
                    x = blk(x, rel_pos_bias=rel_pos_bias)

        x = self.norm(x)
        return x


    def forward(self, x, instruct_states=None, instruct_masks=None):
        x = self.forward_features(x, instruct_states=instruct_states, instruct_masks=instruct_masks)
        return x


    def freeze_base_model(self):
        """Freezes all parameters except those specific to QA-ViT."""
        for name, param in self.named_parameters():
            is_qa_param = False
            if "instruct_dim_reduce" in name: is_qa_param = True
            if "parallel_mlp" in name: is_qa_param = True
            if "parallel_gate" in name: is_qa_param = True

            if not is_qa_param:
                param.requires_grad = False
            else:
                 print(f"  - Keeping trainable: {name}")

    def save_trainable_weight(self, save_path):
        """
                Only the parameters in the model that requires_grad=True are saved.

        Args:
                    save_path (str): The file path where the weights are stored.
        """
        trainable_state_dict = {}
        saved_keys = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_state_dict[name] = param.data.clone().cpu()
                saved_keys.append(name)

        if not saved_keys:
            print("Warning: No trainable weights (requires_grad=True) were found in the model, nothing was saved.")
            return

        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(trainable_state_dict, save_path)
            print(f"Successfully saved {len(saved_keys)} trainable parameters to {save_path}")
        except Exception as e:
            print(f"Error: Saving trainable weights to {save_path} Failed: {e}")

    def load_trainable_weight(self, load_path, strict=True):
        """
                Load the weight into the parameter of requires_grad=True in the model.

        Args:
                    load_path (str): The path to the weight file to be loaded.
                    strict (bool): If true, make sure all parameters requires_grad=True
                                   Find the corresponding key in the loaded file.
                                   If false, only keys that match are loaded.
        """
        if not os.path.exists(load_path):
            print(f"Error: Weight file {load_path} not found, unable to load.")
            return
        loaded_state_dict = torch.load(load_path, map_location='cpu')
        loaded_keys = set(loaded_state_dict.keys())
        current_trainable_keys = set()
        not_found_in_file = []
        loaded_count = 0

        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad:
                    current_trainable_keys.add(name)
                    if name in loaded_state_dict:
                        loaded_tensor = loaded_state_dict[name].to(dtype=param.dtype)
                        if loaded_tensor.shape == param.data.shape:
                             param.data.copy_(loaded_tensor)
                             loaded_count += 1
                        else:
                             print(f"warningShapeMismatchSkipLoadingName"
                                   f"Model required: {param.data.shape}, file provided: {loaded_tensor.shape}")
                    else:
                        not_found_in_file.append(name)

        print(f"权重加载完成。共加载了 {loaded_count} 个参数。")

        if strict and not_found_in_file:
            raise KeyError(f"Strict mode error: The following trainable parameters are not found in the file {load_path}: {not_found_in_file}")
        elif not_found_in_file:
            print(f"Warning: The following {len(not_found_in_file)} currently trainable parameters are not found in the load file: {not_found_in_file}")

        unused_keys = loaded_keys - current_trainable_keys
        if unused_keys:
            print(f"Tip: The following {len(unused_keys)} keys in the load file are not trainable parameters or do not exist in the current model: {unused_keys}")


def create_qa_eva_vit_g(
    img_size=224,
    drop_path_rate=0.4,
    use_checkpoint=False,
    instruction_dim: int = 1408,
    integration_point: str = 'late',
    cached_file: str = r'../models/eva_vit_g.pth'
    ):

    print(f"Instantiating QAEVAVisionTransformer with integration='{integration_point}'...")
    model = QAEVAVisionTransformer(
        img_size=img_size,
        patch_size=14,
        use_mean_pooling=False,
        embed_dim=1408,
        depth=39,
        num_heads=1408//88,
        mlp_ratio=4.3637,
        qkv_bias=True,
        drop_path_rate=drop_path_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_checkpoint=use_checkpoint,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        init_values=None,
        instruction_dim=instruction_dim,
        integration_point=integration_point
    )
    print("QAEVAVisionTransformer instantiated.")

    cached_file = cached_file
    try:
        state_dict = torch.load(cached_file, map_location="cpu")

        interpolate_pos_embed(model, state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        for name, param in model.named_parameters():
            if 'parallel_mlp' in name:
                 original_mlp_name = name.replace('parallel_mlp', 'mlp')
                 if original_mlp_name in state_dict:
                     with torch.no_grad():
                          param.copy_(state_dict[original_mlp_name])


    except Exception as e:
        print(f"Error loading EVA-ViT-G weights: {e}")
        raise e


    return model

def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model and model.pos_embed is not None:
        pos_embed_checkpoint = checkpoint_model['pos_embed'].float()
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches

        if pos_embed_checkpoint.shape[-2] == num_extra_tokens + num_patches:
             print("Position embedding size matches. No interpolation needed.")
             return

        print(f"Position embedding size mismatch: Checkpoint {pos_embed_checkpoint.shape} vs Model {model.pos_embed.shape}. Interpolating...")

        orig_num_patches = pos_embed_checkpoint.shape[-2] - num_extra_tokens
        if orig_num_patches <= 0:

             checkpoint_model['pos_embed'] = model.pos_embed.data
             return

        orig_size = int(math.sqrt(orig_num_patches))
        if orig_size * orig_size != orig_num_patches:
             print(f"Warning: Checkpoint position embedding patch count ({orig_num_patches}) is not a perfect square. Interpolation may be inaccurate.")
        new_size = int(math.sqrt(num_patches))
        if new_size * new_size != num_patches:
             raise ValueError("Model patch count not a perfect square.")


        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
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

    model.apply(_convert_weights_to_fp16)

