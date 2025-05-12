import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor # Keep using Auto* for flexibility
from PIL import Image
import math
import os
from typing import List, Optional, Union

# --- 1. Positional Encoding Module (Unchanged) ---
class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding"""
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Args:
            d_model (int): The dimension of the embeddings.
            max_len (int): The maximum sequence length.
        """
        super().__init__()
        self.d_model = d_model
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model) # Modified shape to [1, max_len, d_model] for easier broadcasting
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # shape [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim] or [seq_len, embedding_dim]

        Returns:
            Tensor: Input tensor with added positional encoding, same shape as x.
        """
        seq_len = x.size(-2) # Get sequence length dimension (works for both 2D and 3D)
        if seq_len > self.pe.size(1):
             raise ValueError(f"Sequence length {seq_len} exceeds maximum positional encoding length {self.pe.size(1)}")

        # Select positional embeddings for the sequence length
        # self.pe shape is [1, max_len, d_model] -> select [1, seq_len, d_model]
        pos_enc = self.pe[:, :seq_len, :]

        if x.dim() == 3:
            # Input shape [batch_size, seq_len, embedding_dim]
            x = x + pos_enc # Broadcasting adds it to each item in the batch
        elif x.dim() == 2:
             # Input shape [seq_len, embedding_dim] -> Unsqueeze to [1, seq_len, embedding_dim] for calculation
             x = x.unsqueeze(0)
             x = x + pos_enc
             x = x.squeeze(0) # Squeeze back to [seq_len, embedding_dim]
        else:
            raise ValueError(f"Input tensor must have 2 or 3 dimensions, but got {x.dim()}")
        return x

class PositionalEncodingTimestamp(nn.Module):
    """
    将基于离散化时间戳的学习位置嵌入添加到帧特征中。
    这模拟了将规范化时间戳 （0-1） 离散化为
    indices 的 Import，然后用于查找添加到
    视觉特征。
    """
    def __init__(self, hidden_dim: int, num_indices: int = 100):
        """
        Args:
            hidden_dim (int): 视觉特征的特征维度。
                              位置嵌入也将具有此维度。
            num_indices (int): 要除以标准化时间戳范围 [0， 1] 转换为。
                               默认为建议的 1000。
        """
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_indices <= 0:
            raise ValueError(f"num_indices must be positive, got {num_indices}")

        self.hidden_dim = hidden_dim
        self.num_indices = num_indices
        # The embedding layer that maps a discrete time index to a vector
        self.temporal_embedding = nn.Embedding(num_indices, hidden_dim)

        # Optional: Initialize weights (e.g., normal distribution)
        self.temporal_embedding.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        向输入要素添加临时位置编码。

        Args:
            features (torch.Tensor): The input visual features. Expected shape:
                                     (N, SeqLen, Dim) or (N, Dim), where N is
                                     the total number of frames (e.g., Batch*Frames),
                                     SeqLen is the sequence length per frame (e.g., num patches),
                                     and Dim must match self.hidden_dim.
            temporal_pos (torch.Tensor): A tensor containing the normalized timestamps (0 to 1)
                                         for each frame corresponding to the features.
                                         Expected shape: (N,). *Note: Now generated internally*

        Returns:
            torch.Tensor: Features with added positional encoding. Same shape as input features.
        """
        # Generate temporal_pos internally based on the number of features (frames)
        temporal_pos = torch.linspace(0, 1, features.shape[0], device=features.device, dtype=features.dtype)

        if features.shape[-1] != self.hidden_dim:
             raise ValueError(f"Feature dimension ({features.shape[-1]}) must match "
                              f"model hidden_dim ({self.hidden_dim})")
        # Removed check for external temporal_pos

        # 1. Discretize timestamps into indices (0 to num_indices-1)
        temporal_indices = (temporal_pos * self.num_indices).clamp(0, self.num_indices - 1).long()

        # 2. Look up embeddings for these indices
        # Shape: (N, hidden_dim)
        pos_embedding = self.temporal_embedding(temporal_indices)

        # 3. Reshape embedding to be added to features
        if features.ndim == 3: # Shape (N, SeqLen, Dim)
            pos_embedding = pos_embedding.unsqueeze(1) # Reshape to (N, 1, Dim) for broadcasting
        elif features.ndim != 2: # Shape (N, Dim)
             raise ValueError(f"Features must have 2 or 3 dimensions (N, Dim) or (N, SeqLen, Dim), "
                              f"got {features.ndim}")

        # 4. Add positional embedding to features
        output_features = features + pos_embedding

        return output_features

# --- 2. SigLIP Feature Extractor Module ---
class SiglipFeatureExtractor(nn.Module): # Renamed class
    """
    Extracts visual features from video frames using SigLIP, adds positional
    encoding, and performs temporal pooling.
    """
    def __init__(self,
                 model_id: str = "google/siglip-so400m-patch14-384", # Updated default model ID
                 output_layer_index: int = -1, # Default to last layer
                 max_frames: int = 32, # Max frames anticipated for positional encoding
                 feature_type: str = 'patch',  # select vit output feature type ['patch','cls_patch','cls']
                 embedding_type: str = 'num', # select position embedding type ['num','sin']
                 device: Optional[Union[str, torch.device]] = None,
                 dtype: Optional[torch.dtype] = None):
        """
        Args:
            model_id (str): Hugging Face model ID for SigLIP (or other compatible ViT).
            output_layer_index (int): Index of the encoder layer to output features from.
                                       -1 means the last layer, -2 the second to last, etc.
                                        0 is the embedding layer output.
            max_frames (int): Maximum number of frames expected in a video for PE sizing.
            feature_type (str): Type of feature to extract: 'cls', 'patch', or 'cls_patch'.
            embedding_type (str): Type of positional encoding: 'num' (learned timestamp) or 'sin' (sinusoidal).
            device (Optional[Union[str, torch.device]]): Device to run the model on. Auto-detects if None.
            dtype (Optional[torch.dtype]): Data type for model computations. Auto-detects if None.
        """
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.model_id = model_id
        self.output_layer_index = output_layer_index
        self.feature_type = feature_type
        self.embedding_type = embedding_type

        print(f"Initializing SiglipFeatureExtractor on device: {self.device} with dtype: {self.dtype}") # Updated print
        print(f"Loading SigLIP model: {self.model_id}") # Updated print

        try:
            # Use Auto* classes to load the specified model and processor
            self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            # Load only the vision_model part if it's a multimodal model like SigLIP
            # Check if the model config has 'vision_config'
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_id)
            if hasattr(config, 'vision_config'):
                 print("Loading vision backbone from the SigLIP model.")
                 self.model = AutoModel.from_pretrained(
                    self.model_id,
                    torch_dtype=self.dtype
                 ).vision_model # Access the vision_model directly
                 # self.model = AutoModel.from_pretrained(
                 #     self.model_id,
                 #     torch_dtype=self.dtype
                 # ).vision_model.to(self.device)  # Access the vision_model directly
            else:
                 # Assume it's a vision-only model if no vision_config
                 print("Loading the model directly (assuming vision-only).")
                 self.model = AutoModel.from_pretrained(
                    self.model_id,
                    torch_dtype=self.dtype
                 )
                # self.model = AutoModel.from_pretrained(
                #     self.model_id,
                #     torch_dtype=self.dtype
                # ).to(self.device)


            # --- Freeze Model ---
            # self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            print(f"SigLIP model '{self.model_id}' vision backbone loaded and frozen.") # Updated print

        except Exception as e:
            print(f"Error loading model '{self.model_id}': {e}")
            raise

        # Get hidden dimension from the loaded model's config
        self.hidden_dim = self.model.config.hidden_size
        print(f"Model hidden dimension: {self.hidden_dim}")

        # --- Positional Encoding ---
        if self.embedding_type == 'sin':
            self.pos_encoder = PositionalEncoding(self.hidden_dim, max_len=max_frames)
        elif self.embedding_type == 'num':
            self.pos_encoder = PositionalEncodingTimestamp(self.hidden_dim)
        else:
             raise ValueError(f"Invalid embedding_type: {self.embedding_type}. Choose 'sin' or 'num'.")

        self.pos_encoder = self.pos_encoder.to(self.dtype)
        print(f"Positional Encoder ({self.embedding_type}) initialized with max_len/indices appropriate for {max_frames} frames, d_model={self.hidden_dim}.")

        # --- Temporal Pooling --- (Remains the same)
        self.avg_pool = nn.AvgPool1d(kernel_size=4, stride=4) # Example pooling

    def forward(self, video_frames: List[Image.Image],device=None) -> Optional[torch.Tensor]:
        """
        Extracts features from a list of video frames.

        Args:
            video_frames (List[Image.Image]): A list of PIL Image objects representing the video frames.

        Returns:
            torch.Tensor: The temporally pooled spatiotemporal feature tensor (t_feat),
                          shape [Pooled_T, hidden_dim] or [Pooled_T, NumPatches, hidden_dim] depending on pooling and feature_type.
                          Returns None if no frames are provided or processing fails.
        """
        if not video_frames:
            print("Warning: Received empty list of video frames.")
            return None

        with torch.no_grad():

            # Process each frame using the model's processor
            # The processor handles resizing, normalization specific to the model
            inputs = self.processor(images=video_frames, return_tensors="pt").to(device=device,dtype=self.dtype)

            # Get hidden states from the specified layer
            outputs = self.model(**inputs, output_hidden_states=True)

            all_hidden_states = outputs.hidden_states

            # --- Layer Selection Logic (same as before) ---
            num_encoder_layers = len(all_hidden_states) - 1 # Exclude embedding layer
            if self.output_layer_index < -num_encoder_layers or self.output_layer_index >= num_encoder_layers:
                 raise ValueError(f"Invalid output_layer_index: {self.output_layer_index}. "
                                  f"Model has {num_encoder_layers} encoder layers (indices 0 to {num_encoder_layers-1}, or -1 to {-num_encoder_layers}).")

            if self.output_layer_index < 0:
                layer_output = all_hidden_states[self.output_layer_index] # e.g., -1 gets last element
            else:
                # User index 0..N-1 maps to python index 1..N
                if self.output_layer_index + 1 >= len(all_hidden_states):
                     raise ValueError(f"Invalid positive output_layer_index: {self.output_layer_index}. "
                                      f"Maps to Python index {self.output_layer_index + 1}, but tuple length is {len(all_hidden_states)}.")
                layer_output = all_hidden_states[self.output_layer_index + 1]
            # --- End Layer Selection ---

            # --- Feature Type Selection Logic (same as before) ---
            # Shape of layer_output: [batch_size=1, sequence_length=num_patches+1, hidden_dim]
            # CLS token is typically the first token in ViT outputs
            if self.feature_type == 'cls':
                # Select only the CLS token feature
                frame_feature = layer_output[:, 0:1, :] # Shape: [1, 1, hidden_dim]
            elif self.feature_type == 'patch':
                # Select only the patch token features
                frame_feature = layer_output[:, 1:, :]  # Shape: [1, num_patches, hidden_dim]
            elif self.feature_type == 'cls_patch':
                # Select all tokens (CLS + patches)
                frame_feature = layer_output[:, :, :]  # Shape: [1, 1+num_patches, hidden_dim]
            else:
                 raise ValueError(f"Invalid feature_type: {self.feature_type}. Choose 'cls', 'patch', or 'cls_patch'.")

            frame_feature = frame_feature / frame_feature.norm(p=2, dim=-1, keepdim=True)



        # Add positional encoding
        # Input shape: [T, SeqLen, Dim] or [T, 1, Dim] if feature_type='cls'
        # Output shape: [T, SeqLen, Dim] or [T, 1, Dim]
        pos_encoded_features = self.pos_encoder(frame_feature)

        # Temporal Pooling (Average Pooling, according to SeqLen dimension)
        pooled_features = self.avg_pool(pos_encoded_features.transpose(1, 2)).transpose(1, 2)

        return pooled_features

    # --- 新增函数：保存可训练权重 ---
    def save_trainable_weight(self, save_path: str):
        """
        仅保存模型中可训练的参数（requires_grad=True）到指定路径。

        Args:
            save_path (str): 保存权重的文件路径 (例如 'trainable_weights.pth')。
        """
        trainable_state_dict = {}
        # 遍历整个 SiglipFeatureExtractor 模块及其子模块的命名参数
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_state_dict[name] = param.cpu().data # 建议保存到 CPU，避免设备问题
                # 使用 .data 避免保存计算图， .cpu() 确保与设备无关

        if not trainable_state_dict:
            print(f"警告: 在 SiglipFeatureExtractor 中未找到可训练的权重。将保存一个空的 state_dict 到 '{save_path}'。")
        else:
            print(f"找到并准备保存以下可训练权重到 '{save_path}': {list(trainable_state_dict.keys())}")

        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(trainable_state_dict, save_path)
            print(f"可训练权重成功保存到: {save_path}")
        except Exception as e:
            print(f"错误：保存权重到 '{save_path}' 失败: {e}")

    # --- 新增函数：加载可训练权重 ---
    def load_trainable_weight(self, load_path: str):
        """
        从指定路径加载权重，并仅更新模型中当前标记为可训练的参数。

        Args:
            load_path (str): 包含可训练权重的文件的路径。
        """
        if not os.path.exists(load_path):
            print(f"错误: 权重文件 '{load_path}' 不存在。无法加载。")
            return

        # 加载保存的字典，确保映射到当前模型的设备
        loaded_state_dict = torch.load(load_path, map_location=self.device)
        print(f"从 '{load_path}' 加载权重。加载的键: {list(loaded_state_dict.keys())}")

        if not loaded_state_dict:
            print(f"警告: 加载的权重文件 '{load_path}' 为空。")
            return

        # 获取当前模型中标记为可训练的参数的名称集合
        current_trainable_keys = {name for name, param in self.named_parameters() if param.requires_grad}
        print(f"当前模型中可训练的参数: {list(current_trainable_keys)}")

        # --- 关键步骤: 使用 load_state_dict 加载，strict=False ---
        # strict=False 会忽略:
        # 1. missing_keys: 当前模型中有，但加载字典中没有的键 (例如，冻结的骨干网络参数)
        # 2. unexpected_keys: 加载字典中有，但当前模型中没有的键 (如果保存和加载的模型结构不匹配)
        load_result = self.load_state_dict(loaded_state_dict, strict=False)

        # --- 分析加载结果 (可选但有用) ---
        loaded_successfully = []
        missing_in_file = [] # 当前可训练但在文件中缺失
        ignored_from_file = [] # 文件中有但模型中没有或不可训练

        for key in loaded_state_dict.keys():
            if key in load_result.unexpected_keys:
                ignored_from_file.append(key)
            else:
                # 检查加载的键是否对应当前模型的可训练参数
                is_trainable_in_current_model = False
                try:
                    param = self.get_parameter(key)
                    if param.requires_grad:
                        is_trainable_in_current_model = True
                except AttributeError:
                    pass # Key might be for a buffer or non-parameter state

                if is_trainable_in_current_model:
                    loaded_successfully.append(key)
                else:
                     # 文件中的键存在于模型中，但模型中对应参数不是可训练的
                     # load_state_dict(strict=False) 仍然会加载它，这可能不是期望的行为
                     # 如果只想严格加载到可训练参数，需要手动过滤 loaded_state_dict
                     print(f"警告: 权重 '{key}' 从文件加载，但它在当前模型中不是可训练参数。")
                     # loaded_successfully.append(key) # 取决于是否认为这种情况算成功

        for key in current_trainable_keys:
            if key not in loaded_state_dict:
                missing_in_file.append(key)

        print(f"成功将权重加载到以下参数: {loaded_successfully}")
        if missing_in_file:
             print(f"警告: 当前模型中的以下可训练参数未在加载的文件 '{load_path}' 中找到，它们将保持不变: {missing_in_file}")
        if ignored_from_file:
             print(f"警告: 加载的文件 '{load_path}' 中包含当前模型不存在的键，已被忽略: {ignored_from_file}")
        if load_result.missing_keys and not all(k in current_trainable_keys for k in load_result.missing_keys):
             # 打印那些非预期的 missing keys (即不是因为我们只保存训练参数而缺少的键)
             unexpected_missing = [k for k in load_result.missing_keys if k not in current_trainable_keys and k not in loaded_state_dict.keys()]
             # if unexpected_missing:
             #     print(f"加载状态提示 (非预期): 模型中缺少文件或可训练参数列表之外的键: {unexpected_missing}")


        print(f"可训练权重加载操作完成 (来自 '{load_path}')。")

        return True


# --- 3. Example Usage ---
if __name__ == "__main__":
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # For better CUDA error messages
    os.environ["CUDA_VISIBLE_DEVICES"] = "5" # Select GPU

    print("--- Running SigLIP Feature Extractor Example ---") # Updated print

    # Configuration
    # SIGLIP_MODEL_ID = "google/siglip-base-patch16-224" # Smaller SigLIP for testing
    # SIGLIP_MODEL_ID = "google/siglip-large-patch16-384"
    # SIGLIP_MODEL_ID = "google/siglip-so400m-patch14-384" # The requested model
    SIGLIP_MODEL_ID = "/data1/sunmingyu/xiaoxia_laboratory/my_study/models/siglip" # if local

    NUM_FRAMES_TO_SAMPLE = 4 # Increased frame count
    OUTPUT_LAYER = -1 # Use the last layer's output
    FEATURE_TYPE = 'patch' # Extract CLS token features
    # FEATURE_TYPE = 'patch' # Extract patch token features
    EMBEDDING_TYPE = 'num' # Use learned timestamp embedding
    # EMBEDDING_TYPE = 'sin' # Use sinusoidal embedding

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # SigLIP models often work well with bfloat16 if available, otherwise float16 or float32
    compute_dtype = torch.float16

    print(f"Using device: {device}")
    print(f"Using compute dtype: {compute_dtype}")

    # --- Instantiate the extractor ---
    try:
        siglip_extractor = SiglipFeatureExtractor( # Instantiate new class
            model_id=SIGLIP_MODEL_ID,
            output_layer_index=OUTPUT_LAYER,
            max_frames=NUM_FRAMES_TO_SAMPLE + 10, # Buffer for PE
            feature_type=FEATURE_TYPE,
            embedding_type=EMBEDDING_TYPE,
            device=device,
            dtype=compute_dtype
        ).to(device)
        # Determine expected input size from the processor
        # This handles cases where the model name doesn't specify size clearly
        if hasattr(siglip_extractor.processor, 'size'):
             # Standard image processors have 'size' attribute
            if isinstance(siglip_extractor.processor.size, dict):
                 # Newer processors might use {'shortest_edge': size} or {'height': H, 'width': W}
                if 'shortest_edge' in siglip_extractor.processor.size:
                     img_size = siglip_extractor.processor.size['shortest_edge']
                     image_resolution = (img_size, img_size)
                elif 'height' in siglip_extractor.processor.size and 'width' in siglip_extractor.processor.size:
                    image_resolution = (siglip_extractor.processor.size['height'], siglip_extractor.processor.size['width'])
                else: # Fallback if dict format is unexpected
                    print("Warning: Processor size format unclear, defaulting to 384x384.")
                    image_resolution = (384, 384)
            else: # Assume it's an int or tuple
                size = siglip_extractor.processor.size
                image_resolution = (size, size) if isinstance(size, int) else size[:2]
        else: # Fallback if processor has no 'size'
            print("Warning: Cannot determine processor size automatically, defaulting to 384x384.")
            image_resolution = (384, 384) # Default for the specified SigLIP model

        print(f"Processor expects input size: {image_resolution}")

    except Exception as e:
        print(f"Failed to initialize extractor: {e}")
        import traceback
        traceback.print_exc()
        exit()

    # --- Create Dummy Video Data ---
    dummy_frames = []
    for i in range(NUM_FRAMES_TO_SAMPLE):
        # Vary color slightly per frame for visual distinction if saved
        color_val = ( (i * 10) % 256, (i * 5) % 256, (i * 20) % 256)
        dummy_image = Image.new('RGB', image_resolution, color=color_val) # Use determined resolution
        dummy_frames.append(dummy_image)

    print(f"\nProcessing {len(dummy_frames)} dummy frames ({image_resolution[0]}x{image_resolution[1]})...")

    # --- Extract Features ---
    t_feat = siglip_extractor(dummy_frames,device)

    # --- Check Output ---
    if t_feat is not None:
        print(f"\nSuccessfully extracted pooled features (t_feat).")
        print(f"Shape of t_feat: {t_feat.shape}")
        # Expected shape: [Pooled_T, SeqLen, hidden_dim]
        # Pooled_T = NUM_FRAMES_TO_SAMPLE / 4 = 16 / 4 = 4
        # SeqLen = 1 (for cls), num_patches (for patch), 1+num_patches (for cls_patch)
        # hidden_dim depends on the SigLIP model (e.g., 1152 for so400m)
        print(f"Device of t_feat: {t_feat.device}")
        print(f"Dtype of t_feat: {t_feat.dtype}")

        # Example: Check magnitude
        print(f"t_feat norm: {torch.linalg.norm(t_feat).item()}")
    else:
        print("\nFeature extraction failed.")

    # --- 保存和加载可训练权重示例 ---
    print("\n--- Testing Save/Load Trainable Weights ---")
    # save_file_path = "./test/vit_trainable_weights.pth"
    save_file_path='/data1/sunmingyu/xiaoxia_laboratory/my_study/dualfocus/training_output/dualfocus_msrvtt_exp01/checkpoints/vit_step_40.pth'

    # 1. 保存可训练权重
    print(f"\nAttempting to save trainable weights to: {save_file_path}")
    siglip_extractor.save_trainable_weight(save_file_path)

    # (可选) 模拟修改可训练权重，以便验证加载效果
    if EMBEDDING_TYPE == 'num':
        print("\n(Optional) Modifying trainable weights before loading...")
        try:
            with torch.no_grad():
                # 假设 pos_encoder.temporal_embedding.weight 是唯一可训练的
                param_to_modify = siglip_extractor.pos_encoder.temporal_embedding.weight
                original_norm = torch.linalg.norm(param_to_modify).item()
                param_to_modify.data.fill_(0.0)  # 将权重置零
                print(
                    f"Trainable weight 'pos_encoder.temporal_embedding.weight' zeroed out. Original norm: {original_norm:.4f}, New norm: {torch.linalg.norm(param_to_modify).item():.4f}")
        except AttributeError:
            print("Could not find 'pos_encoder.temporal_embedding.weight' to modify.")
        except Exception as e:
            print(f"Error modifying weights: {e}")

    # 2. 加载可训练权重
    print(f"\nAttempting to load trainable weights from: {save_file_path}")
    # siglip_extractor.load_trainable_weight(save_file_path)
    if not siglip_extractor.load_trainable_weight(save_file_path):
        print('aaaa')

    # (可选) 验证加载后的权重是否恢复
    if EMBEDDING_TYPE == 'num':
        print("\n(Optional) Verifying weights after loading...")
        try:
            param_after_load = siglip_extractor.pos_encoder.temporal_embedding.weight
            loaded_norm = torch.linalg.norm(param_after_load).item()
            print(f"Norm of 'pos_encoder.temporal_embedding.weight' after loading: {loaded_norm:.4f}")
            # 在这个简单例子中，如果之前修改了权重，这里的 norm 应该接近原始 norm (如果不完全相同，可能是因为保存/加载的精度问题)
            # 注意：如果之前没有修改，这里的 norm 就等于加载前的 norm
        except AttributeError:
            print("Could not find 'pos_encoder.temporal_embedding.weight' to verify.")
        except Exception as e:
            print(f"Error verifying weights after load: {e}")

    print("\n--- Example Finished ---")