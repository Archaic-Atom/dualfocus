import torch,warnings
import torch.nn as nn
import os
import torch.nn.functional as F
import torch.nn.init as init
from functools import partial
from typing import Optional
from torch.utils.checkpoint import checkpoint

def weights_init_normal(m, std=0.02):
    """
    Apply weight initialization based on module type.
    - For nn. Linear layer: Weights are initialized with a normal distribution (mean=0, std=std) and bias is initialized to 0.
    - For nn. LayerNorm layer: Weights are initialized to 1, biases are initialized to 0.
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('LayerNorm') != -1:

        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)
class MLP(nn.Module):
    """Simple two-layer MLP with GELU activation"""
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            # Heuristic: Make hidden dim proportional to the larger of input/output
            # This can be adjusted based on empirical results
            hidden_dim = max(input_dim, output_dim) * 2
            # Or use the previous heuristic: hidden_dim = output_dim * 4
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        if input_dim == hidden_dim and hidden_dim == output_dim:
             print(f"Warning: MLP input, hidden, and output dims are all {input_dim}. This MLP might act like a simple linear layer if not for activation.")

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

# --- Feature Mapping Framework ---
class FeatureMappingFramework(nn.Module):
    """
    Maps spatiotemporal (t_feat) and semantic (s_feat) features
    through separate MLPs, concatenates them, and then maps the combined
    feature to a final output dimension (p_feat) aligned with the LLM embedding space.

    Process:
    t_feat -> MLP_t -> mapped_t
    s_feat -> MLP_s -> mapped_s
    concat([mapped_t, mapped_s]) -> MLP_final -> p_feat
    """
    def __init__(self,
                 t_feat_dim: int,
                 s_feat_dim: int,
                 output_dim: int, # Target dimension (e.g., LLM hidden dim for p_feat)
                 intermediate_proj_dim: Optional[int] = None, # Optional intermediate projection size
                 mlp_hidden_factor: int = 2 # Factor for MLP hidden layers (e.g., hidden = max(input, output)*factor)
                ):
        """
        Args:
            t_feat_dim (int): Dimension of the input spatiotemporal features (t_feat).
            s_feat_dim (int): Dimension of the input semantic features (s_feat).
            output_dim (int): The final desired output dimension for p_feat,
                              typically matching the LLM's hidden dimension.
            intermediate_proj_dim (Optional[int]): Dimension *after* the initial projection
                              of t_feat (mapped_t) and s_feat (mapped_s). If None,
                              it defaults to `output_dim`, meaning the first MLPs
                              project directly to the final target dimension.
            mlp_hidden_factor (int): Factor used in the MLP's hidden layer size heuristic.
        """
        super().__init__()

        # If intermediate_proj_dim is not specified, project directly to the target dim
        if intermediate_proj_dim is None:
            intermediate_proj_dim = output_dim

        self.intermediate_proj_dim = intermediate_proj_dim
        self.output_dim = output_dim
        self.activation = nn.GELU()

        print(f"Initializing FeatureMappingFramework:")
        print(f"  Input t_feat_dim: {t_feat_dim}")
        print(f"  Input s_feat_dim: {s_feat_dim}")
        print(f"  Intermediate projection dim (per feature): {self.intermediate_proj_dim}")
        print(f"  Final output_dim (p_feat): {self.output_dim}")

        # 1. MLP for Spatiotemporal features (t_feat)
        mlp_t_hidden = max(t_feat_dim, self.intermediate_proj_dim) * mlp_hidden_factor
        self.mlp_t = MLP(
            input_dim=t_feat_dim,
            output_dim=self.intermediate_proj_dim,
            hidden_dim=mlp_t_hidden
        )
        print(f"  mlp_t: Input={t_feat_dim}, Hidden={mlp_t_hidden}, Output={self.intermediate_proj_dim}")


        # 2. MLP for Semantic features (s_feat)
        mlp_s_hidden = max(s_feat_dim, self.intermediate_proj_dim) * mlp_hidden_factor
        self.mlp_s = MLP(
            input_dim=s_feat_dim,
            output_dim=self.intermediate_proj_dim,
            hidden_dim=mlp_s_hidden
        )
        print(f"  mlp_s: Input={s_feat_dim}, Hidden={mlp_s_hidden}, Output={self.intermediate_proj_dim}")

        # 3. Final MLP for concatenated features
        self.norm_final = nn.LayerNorm(self.intermediate_proj_dim)
        self.mlp_final = nn.Linear(self.intermediate_proj_dim,self.output_dim)

        init_std = 0.01
        self.apply(partial(weights_init_normal, std=init_std))


    def forward(self, t_feat: torch.Tensor, s_feat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the mapping framework.

        Args:
            t_feat (torch.Tensor): Spatiotemporal features.
                                   Expected shape: [batch_size, t_feat_dim] or [batch_size, 1, t_feat_dim].
                                   The code assumes the last dimension is the feature dim.
            s_feat (torch.Tensor): Semantic features.
                                   Expected shape: [batch_size, s_feat_dim] or [batch_size, n_query, s_feat_dim].
                                   The code assumes the last dimension is the feature dim.
                                   *IMPORTANT*: If s_feat has a sequence dimension (e.g., from Q-Former),
                                   you might need to average/pool it *before* passing it here, or adjust
                                   this framework. This implementation assumes s_feat is also projected
                                   to a single vector per batch item.

        Returns:
            torch.Tensor: The final mapped features (p_feat).
                          Shape: [batch_size, output_dim].
        """
        if len(t_feat.shape)==4:
            b, ft, st, _ = t_feat.shape  # Assume t_feat determines the structure
            _, fs, ss, _ = s_feat.shape
            print("t_feat shape:",t_feat.shape)
            print("s_feat shape:", s_feat.shape)
        else:
            ft, st, _ = t_feat.shape  # Assume t_feat determines the structure
            fs, ss, _ = s_feat.shape
            b=1
            print("t_feat shape:", t_feat.shape)
            print("s_feat shape:", s_feat.shape)
        # --- Feature Mapping ---
        # 1. Map t_feat
        # mapped_t = self.activation(self.mlp_t(t_feat)) # Shape: [batch_size, intermediate_proj_dim]
        mapped_t = self.mlp_t(t_feat)  # Shape: [batch_size, intermediate_proj_dim]


        # 2. Map s_feat
        # mapped_s = self.activation(self.mlp_s(s_feat)) # Shape: [batch_size, intermediate_proj_dim]
        mapped_s = self.mlp_s(s_feat) # Shape: [batch_size, intermediate_proj_dim]
        d_inter = self.intermediate_proj_dim
        mapped_t_reshaped = mapped_t.view(b, ft * st, d_inter)
        mapped_s_reshaped = mapped_s.view(b, fs * ss, d_inter)

        # 3. Concatenate
        concatenated_features = torch.cat([mapped_t_reshaped, mapped_s_reshaped], dim=1)
        # Shape: [batch_size, intermediate_proj_dim * 2]

        # 4. Final Mapping
        # p_feat = self.mlp_final(self.norm_final(concatenated_features)) # Shape: [batch_size, output_dim]
        p_feat = self.mlp_final(concatenated_features)  # Shape: [batch_size, output_dim]

        return p_feat

    def save_trainable_weight(self, save_path: str):
        """
                Saves all trainable parameters in the model (requires_grad=True).

        Args:
            save_path (str): The save path of the parameter file.
        """
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            print(f"Create directory: {save_dir}")

        trainable_state_dict = {}
        param_count = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_state_dict[name] = param.cpu()
                param_count += 1

        try:
            torch.save(trainable_state_dict, save_path)
            print(f"Successfully saved {param_count} trainable parameters to: {save_path}")
        except Exception as e:
            print(f"Error saving trainable parameters: {e}")

    def load_trainable_weight(self, save_path: str):
        """
        Load trainable parameters from the file to the current model.
        Only parameters with matching names that exist in the file and are also present in the current model will be loaded.

            Args:
            save_path (str): The file path that contains the state dictionary of trainable parameters.
        """
        if not os.path.exists(save_path):
            print(f"Error: Weight file does not exist in {save_path}")
            return

        try:
            saved_state_dict = torch.load(save_path, map_location='cpu')
            print(f"Load the status dictionary from {save_path}...")

            current_state_dict = self.state_dict()
            loaded_count = 0
            skipped_count = 0
            missing_in_model = []

            for name, param in saved_state_dict.items():
                if name in current_state_dict:

                    if current_state_dict[name].shape == param.shape:
                        current_state_dict[name] = param
                        loaded_count += 1
                    else:
                        print(f"Warning: Skip the parameter '{name}'. The shapes do not match. Model: {current_state_dict[name].shape}, File: {param.shape}")
                        skipped_count += 1
                else:
                    missing_in_model.append(name)
                    skipped_count += 1

            self.load_state_dict(current_state_dict, strict=False)

            if skipped_count > 0:
                print(f"{skipped_count} parameters were skipped (shapes don't match or don't exist in the current model).")
            if missing_in_model:
                print(f"The following parameters in the file do not exist in the current model: {', '.join(missing_in_model)}")

            model_trainable_names = {name for name, param in self.named_parameters() if param.requires_grad}
            loaded_names = set(saved_state_dict.keys())
            not_loaded_trainable = model_trainable_names - loaded_names
            if not_loaded_trainable:
                 print(f"Warning: The following trainable parameters in the model are not loaded from the file: {', '.join(not_loaded_trainable)}")


        except Exception as e:
            print(f"加载可训练参数时出错: {e}")

if hasattr(torch.backends.cuda, "sdp_kernel"):
    from torch.backends.cuda import sdp_kernel
else:
    class sdp_kernel:
        def __init__(self, **kwargs): pass
        def __enter__(self): pass
        def __exit__(self, exc_type, exc_val, exc_tb): pass

class MemoryEfficientMHA(nn.Module):
    """
    A memory-efficient multi-head attention module using torch.nn.functional.scaled_dot_product_attention.
    It functionally replaces nn. MultiheadAttention, but the memory footprint is reduced from O(L^2) to O(L).
    """
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: torch.Tensor = None):
        """
                Args:
                    query (torch. Tensor): query tensor, shape [B, L_q, D]
                    key (torch. Tensor): Key Tensor, Shape [B, L_k, D]
                    value (torch. Tensor): value tensor, shape [B, L_v, D]
                    attn_mask (torch. Tensor, optional): Attention mask.
                        In SDPA, a value of True indicates that the position *does not* participate in attention calculations.
                        The shape can be [B, H, L_q, L_k] or [L_q, L_k].

        Returns:
            Tuple[torch. Tensor, None]: (Attention Output, None).
            The output shape is [B, L_q, D]. None is returned to be consistent with the output format of nn.MultiheadAttention.
        """
        L_q, _ = query.shape
        L_k = key.shape[0]
        L_v = value.shape[0]

        # 1. 投影 Q, K, V
        # 如果 query, key, value 是同一个张量，可以一次性计算以提高效率
        if torch.equal(query, key) and torch.equal(key, value):
            q, k, v = self.in_proj(query).chunk(3, dim=-1)
        else:
            # 分别计算 Q, K, V 的投影
            w_q, w_k, w_v = self.in_proj.weight.chunk(3)
            if self.in_proj.bias is not None:
                b_q, b_k, b_v = self.in_proj.bias.chunk(3)
                q = F.linear(query, w_q, b_q)
                k = F.linear(key, w_k, b_k)
                v = F.linear(value, w_v, b_v)
            else:
                q = F.linear(query, w_q)
                k = F.linear(key, w_k)
                v = F.linear(value, w_v)


        q = q.view(1,L_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(1,L_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(1,L_v, self.num_heads, self.head_dim).transpose(1, 2)


        with sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view( L_q, self.embed_dim)
        output = self.out_proj(attn_output)

        return output, None

class FusionBlock(nn.Module):
    """
        Perform a complete feature fusion process (steps 2-5 corresponding to the original request).

    Process:
        1. t_feat Self-attention - > t_feat_s
        2. [s_feat, t_feat_s] Shared attention -> tal_feat_c
        3. Split s_feat_c, t_feat_c
        4. s_feat_c, t_feat_c Cross Attention -> cro_feat
        5. s_feat + gate * cro_feat -> new_s_feat
    """

    def __init__(self, proj_dim: int, nhead: int, mlp_hidden_factor: int):
        super().__init__()
        self.proj_dim = proj_dim

        self.self_attention_t = MemoryEfficientMHA(proj_dim, nhead)
        self.norm_t = nn.LayerNorm(proj_dim)

        self.co_attention = MemoryEfficientMHA(proj_dim, nhead)
        self.norm_co = nn.LayerNorm(proj_dim)
        ffn_hidden_dim = proj_dim * mlp_hidden_factor
        self.ffn_co = MLP(proj_dim, proj_dim, ffn_hidden_dim)
        self.norm_ffn_co = nn.LayerNorm(proj_dim)

        self.cross_attention = MemoryEfficientMHA(proj_dim, nhead)
        self.norm_cross = nn.LayerNorm(proj_dim)
        self.ffn_cross = MLP(proj_dim, proj_dim, ffn_hidden_dim)
        self.norm_ffn_cross = nn.LayerNorm(proj_dim)

        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, current_t_feat: torch.Tensor, current_s_feat: torch.Tensor):
        """
                Args:
                    current_t_feat (torch. Tensor): t_feat of the current round, [B*N, T_t, D_proj]
                    current_s_feat (torch. Tensor): s_feat of the current round, [B*N, T_s, D_proj]

        Returns:
            Tuple[torch. Tensor, torch. Tensor]: (t_feat for next round, s_feat for next round)
        """
        T_t = current_t_feat.shape[0]
        T_s = current_s_feat.shape[0]
        # print("1d:", current_t_feat.shape)
        # print("2d:", current_s_feat.shape)

        t_feat_attn, _ = self.self_attention_t(current_t_feat, current_t_feat, current_t_feat)
        t_feat_s = self.norm_t(current_t_feat + t_feat_attn)
        # print("1e:", t_feat_attn.shape)
        # print("2e:", t_feat_s.shape)

        tal_feat = torch.cat([current_s_feat, t_feat_s], dim=0)

        mask = torch.ones(T_s + T_t, T_s + T_t, device=current_t_feat.device, dtype=torch.bool)
        # mask[:T_s, :T_s] = False
        # mask[T_s:, :T_s] = False
        # mask[:T_s, T_s:] = False
        mask[T_s:, T_s:] = False
        # print("1f:", tal_feat.shape)
        # print("2f:", mask.shape)
        co_feat_attn, _ = self.co_attention(tal_feat, tal_feat, tal_feat, attn_mask=mask)
        co_feat = self.norm_co(tal_feat + co_feat_attn)
        # print("1:",co_feat.shape)
        co_feat_ffn = self.ffn_co(co_feat)
        # print("2:", co_feat_ffn.shape)
        tal_feat_c = self.norm_ffn_co(co_feat + co_feat_ffn)

        s_feat_c = tal_feat_c[:T_s, :]
        next_t_feat = tal_feat_c[T_s:, :]

        cross_feat_attn, _ = self.cross_attention(query=s_feat_c, key=next_t_feat, value=next_t_feat)
        cross_feat = self.norm_cross(s_feat_c + cross_feat_attn)
        cross_feat_ffn = self.ffn_cross(cross_feat)
        cro_feat = self.norm_ffn_cross(cross_feat + cross_feat_ffn)

        next_s_feat = current_s_feat + self.gate * cro_feat

        return next_t_feat, next_s_feat


class AttentionalFeatureFusionFramework(nn.Module):
    """
        A feature fusion network based on a multi-stage attention mechanism (looping version).

    Processing process:
        1. **Initial Mapping**:
            - t_feat -> MLP_t -> mapped_t
            - s_feat -> MLP_s -> mapped_s

    2. **Loop Fusion (num_cycles Executions)**:
            - Feed (mapped_t, mapped_s) into a FusionBlock.
            - FusionBlock internal execution (self-attention -> co-attention -> cross-attention -> gated fusion).
            - FusionBlock outputs the next round (mapped_t, mapped_s).

    3. **Output**:
            - The mapped_s obtained from the last round of the cycle serves as the final result p_feat.
    """

    def __init__(self,
                 t_feat_dim: int,
                 s_feat_dim: int,
                 proj_dim: int,
                 nhead: int = 8,
                 mlp_hidden_factor: int = 2,
                 num_cycles: int = 2 ,
                 use_checkpointing: bool = True
                 ):
        """
        Args:
            t_feat_dim (int): Enter the dimension of the spatiotemporal feature (t_feat).
            s_feat_dim (int): Enter the dimension of the semantic feature (s_feat).
            proj_dim (int): The feature dimension that is uniformly processed within the network.
            nhead (int): The number of attention heads used in all attention modules.
            mlp_hidden_factor (int): The calculation factor for the size of the MLP hidden layer.
            num_cycles (int): The number of cycles of the core fusion logic.
        """
        super().__init__()
        self.proj_dim = proj_dim
        self.num_cycles = num_cycles
        self.use_checkpointing = use_checkpointing


        mlp_t_hidden = max(t_feat_dim, proj_dim) * mlp_hidden_factor
        self.mlp_t = MLP(
            input_dim=t_feat_dim,
            output_dim=proj_dim,
            hidden_dim=mlp_t_hidden
        )
        print(f"  mlp_t: Input={t_feat_dim}, Hidden={mlp_t_hidden}, Output={proj_dim}")

        mlp_s_hidden = max(s_feat_dim, proj_dim) * mlp_hidden_factor
        self.mlp_s = MLP(
            input_dim=s_feat_dim,
            output_dim=proj_dim,
            hidden_dim=mlp_s_hidden
        )
        print(f"  mlp_s: Input={s_feat_dim}, Hidden={mlp_s_hidden}, Output={proj_dim}")


        self.fusion_blocks = nn.ModuleList(
            [FusionBlock(proj_dim, nhead, mlp_hidden_factor) for _ in range(num_cycles)]
        )

    def forward(self, t_feat: torch.Tensor, s_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t_feat (torch. Tensor): spatiotemporal features, shapes: [B, T_t, N, D_t]
            s_feat (torch. Tensor): semantic features, shapes: [B, T_s, N, D_s]
        Returns:
            torch. Tensor: Final fused features p_feat, shape: [B, T_s, N, proj_dim]
        """
        T_t, N_t, _ = t_feat.shape
        T_s, N_s, _ = s_feat.shape

        t_feat_flat = t_feat.reshape(T_t * N_t, -1)
        s_feat_flat = s_feat.reshape(T_s * N_s, -1)

        mapped_t = self.mlp_t(t_feat_flat)
        mapped_s = self.mlp_s(s_feat_flat)

        for i in range(self.num_cycles):
            block = self.fusion_blocks[i]
            if self.use_checkpointing :
                mapped_t, mapped_s = checkpoint(block, mapped_t, mapped_s, use_reentrant=False)
            else:
                mapped_t, mapped_s = block(mapped_t, mapped_s)

        p_feat = mapped_s.view(1,T_s * N_s,-1)

        return p_feat

    def save_trainable_weight(self, save_path: str):
        """
                Saves all trainable parameters in the model (requires_grad=True).

        Args:
            save_path (str): The save path of the parameter file.
        """
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            print(f"Create directory: {save_dir}")

        trainable_state_dict = {}
        param_count = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_state_dict[name] = param.cpu()
                param_count += 1

        try:
            torch.save(trainable_state_dict, save_path)
            print(f"Successfully saved {param_count} trainable parameters to: {save_path}")
        except Exception as e:
            print(f"Error saving trainable parameters: {e}")

    def load_trainable_weight(self, save_path: str):
        """
                Load trainable parameters from the file to the current model.
                Only parameters with matching names that exist in the file and are also present in the current model will be loaded.

        Args:
            save_path (str): The file path that contains the state dictionary of trainable parameters.
        """
        if not os.path.exists(save_path):
            return

        try:
            saved_state_dict = torch.load(save_path, map_location='cpu')

            current_state_dict = self.state_dict()
            loaded_count = 0
            skipped_count = 0
            missing_in_model = []

            for name, param in saved_state_dict.items():
                if name in current_state_dict:

                    if current_state_dict[name].shape == param.shape:
                        current_state_dict[name] = param
                        loaded_count += 1
                    else:
                        print(f"Warning: Skip the parameter '{name}'. The shapes do not match. Model: {current_state_dict[name].shape}, File: {param.shape}")
                        skipped_count += 1
                else:
                    missing_in_model.append(name)
                    skipped_count += 1

            self.load_state_dict(current_state_dict, strict=False)

            if skipped_count > 0:
                print(f"{skipped_count} parameters were skipped (shapes don't match or don't exist in the current model).")
            if missing_in_model:
                print(f"The following parameters in the file do not exist in the current model: {', '.join(missing_in_model)}")

            model_trainable_names = {name for name, param in self.named_parameters() if param.requires_grad}
            loaded_names = set(saved_state_dict.keys())
            not_loaded_trainable = model_trainable_names - loaded_names
            if not_loaded_trainable:
                 print(f"Warning: The following trainable parameters in the model are not loaded from the file: {', '.join(not_loaded_trainable)}")


        except Exception as e:
            print(f"Error loading trainable parameters: {e}")

