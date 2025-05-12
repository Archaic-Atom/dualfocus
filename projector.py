import torch,warnings
import torch.nn as nn
import os
import torch.nn.functional as F
import torch.nn.init as init
from functools import partial # 用于传递参数给 apply
from typing import Optional

def weights_init_normal(m, std=0.02):
    """
    根据模块类型应用权重初始化。
    - 对 nn.Linear 层: 权重使用正态分布 (mean=0, std=std) 初始化，偏置初始化为 0。
    - 对 nn.LayerNorm 层: 权重初始化为 1，偏置初始化为 0。
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # print(f"Initializing Linear layer: {m} with std={std}")
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
        # print("  Linear initialization done.")
    elif classname.find('LayerNorm') != -1:
        # print(f"Initializing LayerNorm layer: {m}")
        # 通常 LayerNorm 的 gamma (weight) 初始化为 1, beta (bias) 初始化为 0
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)
        # print("  LayerNorm initialization done.")
    # 你可以根据需要添加对其他类型层（如 Conv2d, Embedding）的初始化
    # elif classname.find('Conv') != -1:
    #     init.normal_(m.weight.data, 0.0, std)
    #     if m.bias is not None:
    #         init.constant_(m.bias.data, 0.0)

# --- Assuming MLP class is defined as provided previously ---
# Make sure this class definition is accessible in your script
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
        else:
            ft, st, _ = t_feat.shape  # Assume t_feat determines the structure
            fs, ss, _ = s_feat.shape
            b=1
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
        保存模型中所有可训练的参数 (requires_grad=True)。

        Args:
            save_path (str): 参数文件的保存路径。
        """
        # 确保保存目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            print(f"创建目录：{save_dir}")

        trainable_state_dict = {}
        param_count = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_state_dict[name] = param.cpu() # 保存到 CPU，避免设备问题
                param_count += 1

        try:
            torch.save(trainable_state_dict, save_path)
            print(f"成功将 {param_count} 个可训练参数保存到: {save_path}")
        except Exception as e:
            print(f"保存可训练参数时出错: {e}")

    def load_trainable_weight(self, save_path: str):
        """
        从文件加载可训练的参数到当前模型。
        只会加载文件中存在的、且当前模型中也存在的、名称匹配的参数。

        Args:
            save_path (str): 包含可训练参数的状态字典的文件路径。
        """
        if not os.path.exists(save_path):
            print(f"错误：权重文件不存在于 {save_path}")
            return

        try:
            # 加载保存的状态字典，映射到 CPU 以便后续处理
            saved_state_dict = torch.load(save_path, map_location='cpu')
            print(f"从 {save_path} 加载状态字典...")

            current_state_dict = self.state_dict()
            loaded_count = 0
            skipped_count = 0
            missing_in_model = []

            for name, param in saved_state_dict.items():
                if name in current_state_dict:
                    # 检查当前模型中的参数是否可训练，通常加载时我们不关心这个
                    # 但可以加一个检查，确保加载的参数确实是模型中可训练的部分
                    # model_param = self.get_parameter(name) # 需要辅助函数或直接访问
                    # is_trainable_in_model = any(p.requires_grad for n, p in self.named_parameters() if n == name)

                    # 检查形状是否匹配
                    if current_state_dict[name].shape == param.shape:
                        current_state_dict[name] = param # 使用加载的参数更新当前状态字典
                        loaded_count += 1
                    else:
                        print(f"警告: 跳过参数 '{name}'。形状不匹配。模型: {current_state_dict[name].shape}, 文件: {param.shape}")
                        skipped_count += 1
                else:
                    # 文件中的参数在当前模型中不存在
                    missing_in_model.append(name)
                    skipped_count += 1

            # 使用更新后的状态字典加载模型权重
            # strict=False 允许加载部分参数，忽略不在 saved_state_dict 中的模型参数
            self.load_state_dict(current_state_dict, strict=False)

            print(f"成功加载了 {loaded_count} 个参数。")
            if skipped_count > 0:
                print(f"跳过了 {skipped_count} 个参数（形状不匹配或在当前模型中不存在）。")
            if missing_in_model:
                print(f"  文件中的以下参数在当前模型中不存在: {', '.join(missing_in_model)}")

            # 检查是否有模型中的可训练参数没有被加载 (即不在文件中)
            model_trainable_names = {name for name, param in self.named_parameters() if param.requires_grad}
            loaded_names = set(saved_state_dict.keys())
            not_loaded_trainable = model_trainable_names - loaded_names
            if not_loaded_trainable:
                 print(f"警告: 模型中的以下可训练参数未从文件中加载: {', '.join(not_loaded_trainable)}")


        except Exception as e:
            print(f"加载可训练参数时出错: {e}")


# --- Feature Mapping Framework ---
class FeatureMappingFramework2(nn.Module):
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

        if s_feat_dim==intermediate_proj_dim:
            self.s_feat_proj_none=True
        else:
            self.s_feat_proj_none = False
            # 2. MLP for Semantic features (s_feat)
            mlp_s_hidden = max(s_feat_dim, self.intermediate_proj_dim) * mlp_hidden_factor
            self.mlp_s = MLP(
                input_dim=s_feat_dim,
                output_dim=self.intermediate_proj_dim,
                hidden_dim=mlp_s_hidden
            )
            print(f"  mlp_s: Input={s_feat_dim}, Hidden={mlp_s_hidden}, Output={self.intermediate_proj_dim}")

    def forward(self, t_feat: torch.Tensor, s_feat: torch.Tensor) -> tuple:
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

        # --- Feature Mapping ---
        # 1. Map t_feat
        mapped_t = self.mlp_t(t_feat) # Shape: [batch_size, frame_len, sqen_len, intermediate_proj_dim]

        if self.s_feat_proj_none:
            mapped_s=s_feat
        else:
            # 2. Map s_feat
            mapped_s = self.mlp_s(s_feat) # Shape: [batch_size, frame_len, sqen_len, intermediate_proj_dim]

        return mapped_t,mapped_s



# --- Example Usage ---
if __name__ == "__main__":
    print("\n--- Running Feature Mapping Framework Example ---")

    # Example Dimensions
    batch_size = 4
    frame1=16
    frame2=8
    sqen1=128
    sqen2=32
    # From DINOv2+Pool (assuming pooling happens before this module)
    t_feature_dimension = 1024
    # From SigLIP+QFormer (assuming QFormer output is pooled/averaged before this module)
    s_feature_dimension = 768 # Example Q-Former output dim might vary
    # Target LLM dimension
    llm_hidden_dimension = 1048 # Example Llama-3 8B

    # --- Test Case 1: Intermediate projection matches output ---
    print("\n--- Test Case 1: Intermediate Dim == Output Dim ---")
    mapper1 = FeatureMappingFramework(
        t_feat_dim=t_feature_dimension,
        s_feat_dim=s_feature_dimension,
        output_dim=llm_hidden_dimension,
        intermediate_proj_dim=None # Defaults to output_dim
    )

    # Dummy inputs (assuming already pooled/single vector per sample)
    dummy_t_feat_pooled = torch.randn(frame1,sqen1, t_feature_dimension)
    # Assume s_feat from Q-Former is averaged over queries
    dummy_s_feat_pooled = torch.randn(frame2,sqen2, s_feature_dimension)

    print(f"\nInput t_feat shape: {dummy_t_feat_pooled.shape}")
    print(f"Input s_feat shape: {dummy_s_feat_pooled.shape}")

    p_feat1 = mapper1(dummy_t_feat_pooled, dummy_s_feat_pooled)
    print(f"Output p_feat shape (Test Case 1): {p_feat1.shape}")

    # # --- 保存和加载可训练权重 ---
    # print("\n--- 测试保存和加载可训练权重 ---")
    # save_file_path = "./test/feature_mapper_trainable_weights.pth"
    #
    # # 1. 保存权重
    # mapper1.save_trainable_weight(save_file_path)
    #
    # # 2. 创建一个新的模型实例 (模拟重新开始训练或推理)
    # print("\n创建一个新的模型实例用于加载...")
    # mapper_new = FeatureMappingFramework(
    #     t_feat_dim=t_feature_dimension,
    #     s_feat_dim=s_feature_dimension,
    #     output_dim=llm_hidden_dimension,
    #     intermediate_proj_dim=llm_hidden_dimension
    # )
    #
    # # 3. 加载权重到新模型
    # mapper_new.load_trainable_weight(save_file_path)
    #
    # # (可选) 验证权重是否加载成功
    # # 比较原始模型和新模型的可训练参数
    # print("\n验证加载的权重...")
    # match = True
    # for (name1, param1), (name2, param2) in zip(mapper1.named_parameters(), mapper_new.named_parameters()):
    #     if param1.requires_grad:  # 只比较可训练的
    #         if name1 != name2:
    #             print(f"参数名称不匹配: {name1} vs {name2}")
    #             match = False
    #             break
    #         if not torch.equal(param1.cpu(), param2.cpu()):
    #             print(f"参数 '{name1}' 的值不匹配!")
    #             # print("Original:", param1)
    #             # print("Loaded:", param2)
    #             match = False
    #             # break # 可以取消注释以在第一个不匹配处停止
    #
    # if match:
    #     print("所有可训练参数已成功加载并验证匹配！")
    # else:
    #     print("加载的参数验证失败。")

    # # --- Test Case 2: new 划分 projection matches output ---
    # print("\n--- Test Case 4.1: s_featuren Dim == llm_hidden Dim ---")
    # s_feature_dimension1=llm_hidden_dimension
    # mapper4 = FeatureMappingFramework2(
    #     t_feat_dim=t_feature_dimension,
    #     s_feat_dim=s_feature_dimension1,
    #     output_dim=llm_hidden_dimension,
    #     intermediate_proj_dim=None  # Defaults to output_dim
    # )
    #
    # # Dummy inputs (assuming already pooled/single vector per sample)
    # dummy_t_feat_pooled = torch.randn(batch_size, t_feature_dimension)
    # # Assume s_feat from Q-Former is averaged over queries
    # dummy_s_feat_pooled = torch.randn(batch_size, s_feature_dimension1)
    #
    # print(f"\nInput t_feat shape: {dummy_t_feat_pooled.shape}")
    # print(f"Input s_feat shape: {dummy_s_feat_pooled.shape}")
    #
    # t_feat,s_feat = mapper4(dummy_t_feat_pooled, dummy_s_feat_pooled)
    # print(f"Output t_feat shape (Test Case 4.1): {t_feat.shape}")
    # print(f"Output s_feat shape (Test Case 4.1): {s_feat.shape}")
    #
    #
    # print("\n--- Test Case 4.2: s_featuren Dim != llm_hidden Dim ---")
    # mapper4 = FeatureMappingFramework2(
    #     t_feat_dim=t_feature_dimension,
    #     s_feat_dim=s_feature_dimension,
    #     output_dim=llm_hidden_dimension,
    #     intermediate_proj_dim=None  # Defaults to output_dim
    # )
    #
    # # Dummy inputs (assuming already pooled/single vector per sample)
    # dummy_t_feat_pooled = torch.randn(batch_size, t_feature_dimension)
    # # Assume s_feat from Q-Former is averaged over queries
    # dummy_s_feat_pooled = torch.randn(batch_size, s_feature_dimension)
    #
    # print(f"\nInput t_feat shape: {dummy_t_feat_pooled.shape}")
    # print(f"Input s_feat shape: {dummy_s_feat_pooled.shape}")
    #
    # t_feat, s_feat = mapper4(dummy_t_feat_pooled, dummy_s_feat_pooled)
    # print(f"Output t_feat shape (Test Case 4.2): {t_feat.shape}")
    # print(f"Output s_feat shape (Test Case 4.2): {s_feat.shape}")
    #
    #
    #
    # print("\n--- Mapping Framework Example Finished ---")