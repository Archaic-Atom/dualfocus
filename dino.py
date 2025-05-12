import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import math
from typing import List, Optional, Union

# --- 1. Positional Encoding Module ---
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
    def __init__(self, hidden_dim: int, num_indices: int = 1000):
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
        # self.temporal_embedding.weight.data.normal_(mean=0.0, std=0.02)

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
                                         Expected shape: (N,).

        Returns:
            torch.Tensor: Features with added positional encoding. Same shape as input features.
        """
        temporal_pos=torch.linspace(0, 1, features.shape[0], device=features.device, dtype=features.dtype)
        # if features.shape[0] != temporal_pos.shape[0]:
        #     raise ValueError(f"Number of features ({features.shape[0]}) must match "
        #                      f"number of temporal positions ({temporal_pos.shape[0]})")
        if features.shape[-1] != self.hidden_dim:
             raise ValueError(f"Feature dimension ({features.shape[-1]}) must match "
                              f"model hidden_dim ({self.hidden_dim})")
        if temporal_pos.ndim != 1:
             raise ValueError(f"temporal_pos should be a 1D tensor, got shape {temporal_pos.shape}")

        # 1. Discretize timestamps into indices (0 to num_indices-1)
        # Multiply by num_indices (instead of 1000)
        # Clamp ensures indices are within [0, num_indices - 1]
        temporal_indices = (temporal_pos * self.num_indices).clamp(0, self.num_indices - 1).long()
        # temporal_indices = (temporal_pos * self.num_indices).clamp(0, self.num_indices - 2).long()

        # 2. Look up embeddings for these indices
        # Shape: (N, hidden_dim)
        pos_embedding = self.temporal_embedding(temporal_indices)

        # 3. Reshape embedding to be added to features
        # Original features might be (N, SeqLen, Dim) or (N, Dim)
        # We need to add (N, 1, Dim) or (N, Dim) respectively.
        # Add unsqueeze(1) if features have a SeqLen dimension > 1
        if features.ndim == 3: # Shape (N, SeqLen, Dim)
            pos_embedding = pos_embedding.unsqueeze(1) # Reshape to (N, 1, Dim) for broadcasting
        elif features.ndim != 2: # Shape (N, Dim)
             raise ValueError(f"Features must have 2 or 3 dimensions (N, Dim) or (N, SeqLen, Dim), "
                              f"got {features.ndim}")
        # If features.ndim == 2, pos_embedding shape (N, Dim) is already correct.

        # 4. Add positional embedding to features
        output_features = features + pos_embedding

        return output_features

# --- 2. DINOv2 Feature Extractor Module ---
class Dinov2FeatureExtractor(nn.Module):
    """
    Extracts visual features from video frames using DINOv2, adds positional
    encoding, and performs temporal pooling.
    """
    def __init__(self,
                 model_id: str = "facebook/dinov2-base",
                 output_layer_index: int = -1, # Default to last layer
                 max_frames: int = 32, # Max frames anticipated for positional encoding
                 feature_type: str = 'patch',  # select vit output feature type ['patch','cls_patch','cls']
                 embedding_type: str = 'num', # select position embedding type ['num','sin']
                 device: Optional[Union[str, torch.device]] = None,
                 dtype: Optional[torch.dtype] = None):
        """
        Args:
            model_id (str): Hugging Face model ID for DINOv2.
            output_layer_index (int): Index of the encoder layer to output features from.
                                       -1 means the last layer, -2 the second to last, etc.
                                        0 is the embedding layer output.
            max_frames (int): Maximum number of frames expected in a video for PE sizing.
            device (Optional[Union[str, torch.device]]): Device to run the model on. Auto-detects if None.
            dtype (Optional[torch.dtype]): Data type for model computations. Auto-detects if None.
        """
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.model_id = model_id
        self.output_layer_index = output_layer_index
        self.feature_type=feature_type
        self.embedding_type=embedding_type

        print(f"Initializing Dinov2FeatureExtractor on device: {self.device} with dtype: {self.dtype}")
        print(f"Loading DINOv2 model: {self.model_id}")

        # from transformers import Dinov2Model, Dinov2ImageProcessor
        #
        # self.processor = Dinov2ImageProcessor.from_pretrained(self.model_id)
        # self.model = Dinov2Model.from_pretrained(
        #     self.model_id,
        #     torch_dtype=self.dtype
        # ).to(self.device)

        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            # self.model = AutoModel.from_pretrained(
            #     self.model_id,
            #     torch_dtype=self.dtype
            # ).to(self.device)
            self.model = AutoModel.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype
            )

            # --- Freeze DINOv2 ---
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            print(f"DINOv2 model '{self.model_id}' loaded and frozen.")

        except Exception as e:
            print(f"Error loading DINOv2 model '{self.model_id}': {e}")
            raise

        self.hidden_dim = self.model.config.hidden_size
        print(f"DINOv2 hidden dimension: {self.hidden_dim}")

        # --- Positional Encoding ---
        # max_len should be large enough for the max number of frames
        if self.embedding_type == 'sin':
            self.pos_encoder = PositionalEncoding(self.hidden_dim, max_len=max_frames)
            self.pos_encoder = self.pos_encoder.to(self.dtype)
            # self.pos_encoder = self.pos_encoder.to(self.device).to(self.dtype)
        elif self.embedding_type == 'num':
            self.pos_encoder = PositionalEncodingTimestamp(self.hidden_dim)
            self.pos_encoder = self.pos_encoder.to(self.dtype)
        print(f"Positional Encoder initialized with max_len={max_frames}, d_model={self.hidden_dim}.")
        self.avg_pool = nn.AvgPool1d(kernel_size=4, stride=4)

    def forward(self, video_frames: List[Image.Image]) -> torch.Tensor:
        """
        Extracts features from a list of video frames.

        Args:
            video_frames (List[Image.Image]): A list of PIL Image objects representing the video frames.

        Returns:
            torch.Tensor: The temporally pooled spatiotemporal feature tensor (t_feat),
                          shape [1, hidden_dim]. Returns None if no frames are provided or processing fails.
        """
        if not video_frames:
            print("Warning: Received empty list of video frames.")
            return None

        frame_features_list = []
        try:
            with torch.no_grad():
                for frame_pil in video_frames:
                    # Process each frame
                    inputs = self.processor(images=frame_pil, return_tensors="pt").to(self.device, dtype=self.dtype)

                    # Get hidden states from the specified layer
                    # Note: output_hidden_states=True returns embeddings + output of each layer
                    outputs = self.model(**inputs, output_hidden_states=True)

                    # hidden_states is a tuple: (embeddings, layer1_out, layer2_out, ...)
                    all_hidden_states = outputs.hidden_states

                    # Adjust index: 0=embeddings, 1=layer1, ..., N=last_layer
                    # User input: -1=last, -2=second-last, etc. 0..N-1 for specific layers
                    num_encoder_layers = len(all_hidden_states) - 1 # Exclude embedding layer
                    if self.output_layer_index < -num_encoder_layers or self.output_layer_index >= num_encoder_layers:
                         raise ValueError(f"Invalid output_layer_index: {self.output_layer_index}. "
                                          f"Model has {num_encoder_layers} encoder layers (indices 0 to {num_encoder_layers-1}, or -1 to {-num_encoder_layers}).")

                    # Python's negative indexing works directly if layer_index is negative.
                    # For positive indices, we need layer_index + 1 because hidden_states[0] is embeddings.
                    # Let's simplify: use python's list indexing:
                    # -1 is last element, -2 second last.
                    # 0 is embeddings, 1 is layer 1 output, etc.
                    # So, user's -1 maps to python's -1. User's -2 maps to python's -2.
                    # User's 0 (layer 1 output) maps to python's 1. User's N-1 (last layer output) maps to python's N.
                    # Let's redefine user index slightly for clarity:
                    # -1 means output of the *last encoder layer*.
                    # 0 means output of the *first encoder layer*.
                    # N-1 means output of the *last encoder layer*.

                    # Let's adjust the interpretation:
                    # User's -1 means the last hidden state (output of final layer).
                    # User's -2 means the second to last hidden state.
                    # User's 0 means the output of the *embedding* layer. (Less common)
                    # User's 1 means the output of the *first encoder layer*.
                    # Let's map user index to python index:
                    # N layers -> hidden_states tuple has N+1 elements (emb + N layer outputs)
                    # User -1 (last layer) -> Python index -1
                    # User -2 (second last) -> Python index -2
                    # User 0 (first layer) -> Python index 1
                    # User L (layer L+1) -> Python index L+1
                    if self.output_layer_index < 0:
                        layer_output = all_hidden_states[self.output_layer_index] # e.g., -1 gets last element
                    else:
                        # User index 0..N-1 maps to python index 1..N
                        if self.output_layer_index + 1 >= len(all_hidden_states):
                             raise ValueError(f"Invalid positive output_layer_index: {self.output_layer_index}. "
                                              f"Maps to Python index {self.output_layer_index + 1}, but tuple length is {len(all_hidden_states)}.")
                        layer_output = all_hidden_states[self.output_layer_index + 1]

                    # Extract the CLS token feature from the selected layer's output
                    # Shape of layer_output: [batch_size=1, sequence_length=num_patches+1, hidden_dim]
                    # CLS token is typically the first token
                    if self.feature_type=='cls':
                        frame_cls_feature = layer_output[:, 0:1, :] # Shape: [1,1, hidden_dim]
                    elif self.feature_type=='patch':
                        frame_cls_feature = layer_output[:, 1:, :]  # Shape: [1,1, hidden_dim]
                    elif self.feature_type == 'cls_patch':
                        frame_cls_feature = layer_output[:, :, :]  # Shape: [1,1, hidden_dim]
                    frame_features_list.append(frame_cls_feature)

        except Exception as e:
            print(f"Error during DINOv2 feature extraction: {e}")
            return None

        if not frame_features_list:
            print("Warning: No features extracted from frames.")
            return None

        # Stack features along the time dimension: [T, hidden_dim]
        # Need shape [batch_size, T, hidden_dim] or [T, hidden_dim] for PE
        stacked_features = torch.cat(frame_features_list, dim=0) # Shape: [T, hidden_dim]

        # Add positional encoding ##sd
        pos_encoded_features = self.pos_encoder(stacked_features) # Shape: [T, hidden_dim]

        # Temporal Pooling (Average Pooling)
        pooled_features = self.avg_pool(pos_encoded_features.transpose(1, 2)).transpose(1, 2)
        # Pool along the time dimension (dim=0)
        # pooled_features = torch.mean(pos_encoded_features, dim=0, keepdim=True) # Shape: [1, hidden_dim]

        return pooled_features

# --- 3. Example Usage ---
if __name__ == "__main__":
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 这会让程序只看到4号GPU
    print("--- Running DINOv2 Feature Extractor Example ---")

    # Configuration (adjust as needed)
    # DINOV2_MODEL_ID = "..models/dino"
    DINOV2_MODEL_ID = "/data1/sunmingyu/xiaoxia_laboratory/my_study/models/siglip"
    NUM_FRAMES_TO_SAMPLE = 8
    OUTPUT_LAYER = -1 # Use the last layer's output
    # OUTPUT_LAYER = -2 # Use the second to last layer's output
    # OUTPUT_LAYER = 0 # Use the first encoder layer's output

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    compute_dtype = torch.float32
    print(f"Using device: {device}")
    print(f"Using compute dtype: {compute_dtype}")

    # --- Instantiate the extractor ---
    try:
        dinov2_extractor = Dinov2FeatureExtractor(
            model_id=DINOV2_MODEL_ID,
            output_layer_index=OUTPUT_LAYER,
            max_frames=NUM_FRAMES_TO_SAMPLE + 10, # Provide some buffer
            device=device,
            dtype=compute_dtype
        )
    except Exception as e:
        print(f"Failed to initialize extractor: {e}")
        exit()

    # --- Create Dummy Video Data ---
    # Create a list of dummy PIL images
    dummy_frames = []
    for _ in range(NUM_FRAMES_TO_SAMPLE):
        dummy_image = Image.new('RGB', (224, 224), color = (128, 64, 192))
        dummy_frames.append(dummy_image)

    print(f"\nProcessing {len(dummy_frames)} dummy frames...")

    # --- Extract Features ---
    t_feat = dinov2_extractor(dummy_frames)

    # --- Check Output ---
    if t_feat is not None:
        print(f"\nSuccessfully extracted t_feat.")
        print(f"Shape of t_feat: {t_feat.shape}") # Expected: [1, hidden_dim]
        print(f"Device of t_feat: {t_feat.device}")
        print(f"Dtype of t_feat: {t_feat.dtype}")

        # Example: Check magnitude
        print(f"t_feat norm: {torch.linalg.norm(t_feat).item()}")
    else:
        print("\nFeature extraction failed.")

    print("\n--- Example Finished ---")