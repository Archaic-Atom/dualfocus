# --- Step 4: 第二视觉路径 (VIT) - BATCHED ---
all_m_frames_pil = []
m_video_lens = [] # Store the number of frames *per video*
for video_frames in m_video_frames: # Iterate through videos in the local batch
    if video_frames: # Handle potentially empty video lists
        all_m_frames_pil.extend(video_frames)
        m_video_lens.append(len(video_frames))
    else:
        m_video_lens.append(0)

t_feat_all_frames = None
if all_m_frames_pil: # Only process if there are frames
    with torch.cuda.amp.autocast(enabled=(self.dtype == torch.float16 or self.dtype == torch.bfloat16)):
        # *** Call the MODIFIED extractor ONCE with ALL frames ***
        # Pass the flattened list of all PIL images from this rank's batch
        # The extractor's forward method now handles batching internally
        t_feat_all_frames = self.vit_feature_extractor(all_m_frames_pil)
        # t_feat_all_frames shape: [TotalFramesOnRank, SeqLen, Dim]

# --- Split features back and apply PE/Pooling per video ---
t_feat_list = []
if t_feat_all_frames is not None:
    # Use torch.split to divide the features based on video lengths
    t_feat_split_by_video = list(torch.split(t_feat_all_frames, m_video_lens, dim=0))

    # Now apply PE and Pooling individually to each video's features
    with torch.cuda.amp.autocast(enabled=(self.dtype == torch.float16 or self.dtype == torch.bfloat16)):
        for features_one_video in t_feat_split_by_video:
            if features_one_video.shape[0] == 0: # Handle videos that had 0 frames
                 # Need to decide what to output here. Maybe a zero tensor of expected pooled shape?
                 # Placeholder: append None or skip? Let's append a placeholder tensor
                 # Determine expected pooled dim and seqlen based on feature_type and pooling
                 pooled_dim = self.vit_hidden_dim
                 # This calculation is tricky without knowing the pooling kernel/stride and feature_type output len
                 # For simplicity, let's assume we need a [1, pooled_dim] tensor
                 # You might need to adjust this based on your actual feature_mapper input requirements
                 t_feat_list.append(torch.zeros((1, pooled_dim), device=accelerator.device, dtype=self.dtype)) # Adjust shape as needed
                 continue

            # Apply Positional Encoding (ensure pos_encoder is on correct device)
            pos_encoded = self.vit_feature_extractor.pos_encoder(features_one_video)

            # Apply Temporal Pooling (ensure avg_pool is usable)
            # Pooling expects [N, C, L], features are [T, SeqLen, Dim]
            # If SeqLen > 1 (patch features), pooling might be tricky here.
            # Assuming pooling happens along the Time (T) dimension if SeqLen=1 (CLS)
            # Or maybe pooling should average the SeqLen dimension?
            # --> Let's assume the original avg_pool was meant for the Time dimension.
            # Input to AvgPool1d should be (N, C, L) = (Batch=1, Dim, Time)
            pos_encoded_permuted = pos_encoded.permute(0, 2, 1) # [T, SeqLen, Dim] -> [T, Dim, SeqLen] ??? This needs review based on pooling intent

            # --- Revisit Pooling Logic ---
            # The original avg_pool = nn.AvgPool1d(kernel_size=4, stride=4) likely assumed
            # input shape [Batch, Channels, Length] = [Batch, HiddenDim, NumFrames].
            # Our current `pos_encoded` has shape [NumFrames, SeqLen, HiddenDim].
            # Let's assume we want to average pool along the NumFrames dimension.
            # If SeqLen > 1, we might need to average patches first, then pool time.
            # If SeqLen == 1 (CLS token): pos_encoded shape is [NumFrames, 1, HiddenDim]
            #   1. Squeeze: [NumFrames, HiddenDim]
            #   2. Permute for AvgPool1d: [HiddenDim, NumFrames]
            #   3. Unsqueeze for Batch dim: [1, HiddenDim, NumFrames]
            #   4. Pool: [1, HiddenDim, PooledNumFrames]
            #   5. Permute back & Squeeze/Reshape: [PooledNumFrames, HiddenDim] or [1, HiddenDim] if global pool?

            # --> SIMPLIFICATION: Let's do simple mean pooling over time for now <--
            # This replaces the AvgPool1d layer for simplicity, adjust if needed.
            pooled_feat = torch.mean(pos_encoded, dim=0) # Average over the frames dim -> [SeqLen, Dim]
            # If you always want a single vector per video:
            # pooled_feat = torch.mean(pos_encoded, dim=(0, 1)) # Average over frames and patches -> [Dim]
            # pooled_feat = pooled_feat.unsqueeze(0) # Make it [1, Dim]

            # Keep the SeqLen for now, maybe mapper handles it:
            pooled_feat = pooled_feat.unsqueeze(0) # -> [1, SeqLen, Dim] to match expected list element dims? Check mapper input.
            # --> Adjust pooling strategy based on what feature_mapper expects! <--

            t_feat_list.append(pooled_feat.to(accelerator.device)) # Ensure final feature is on correct device
else:
     # Handle case where no frames were processed on this rank at all
     t_feat_list = [torch.zeros((1, self.vit_hidden_dim), device=accelerator.device, dtype=self.dtype) # Adjust shape
                   for _ in range(len(m_video_frames))] # Create placeholders

# Ensure t_feat_list has the correct length matching batch_size for this rank
if len(t_feat_list) != len(m_video_frames):
    raise RuntimeError(f"Rank {accelerator.process_index}: Mismatch in t_feat_list length ({len(t_feat_list)}) and m_video_frames length ({len(m_video_frames)}) after processing.")

print(f"Rank {accelerator.process_index} finished Step 4") # Use rank info
# --- End Step 4 Modification ---