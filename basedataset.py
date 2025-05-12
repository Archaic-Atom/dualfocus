# base_dataset.py (New File)
import os
import json
import numpy as np # 需要 numpy 来处理帧数据
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from decord import VideoReader, cpu
import warnings
from PIL import Image # To return PIL images if needed later, or process tensors
from typing import Optional, List, Tuple, Dict,Any # 导入

# Note: The collate_fn remains the same as it handles None values generally
def collate_fn(batch):
    # Filter out items where video loading might have failed (returned None)
    batch = [item for item in batch if item is not None]
    if not batch:
        # If the whole batch failed, return an empty structure or handle as needed
        return {"video_r": [], "video_m": [], "question": [], "answer": [], "video_id": [], "question_id": []}

    # Collect data from valid items
    # Videos are now lists of tensors, handle appropriately downstream
    video_r_batch = [item["video_r"] for item in batch]
    video_m_batch = [item["video_m"] for item in batch]
    question_batch = [item["question"] for item in batch]
    answer_batch = [item["answer"] for item in batch]
    video_id_batch = [item["video_id"] for item in batch]
    question_id_batch = [item["question_id"] for item in batch]

    return {
        "video_r": video_r_batch,
        "video_m": video_m_batch,
        "question": question_batch,
        "answer": answer_batch,
        "video_id": video_id_batch,
        "question_id": question_id_batch,
    }

# --- 辅助函数：计算采样索引 ---
def _calculate_indices(num_frames: int, total_frames: int, video_fps: float) -> Optional[torch.Tensor]:
    """
    根据采样规则计算帧索引。

    Args:
        num_frames (int): 采样规则 (-1, -2, -3, >0)。
        total_frames (int): 视频总帧数。
        video_fps (float): 视频帧率 (仅当 num_frames 为 -2 或 -3 时需要)。

    Returns:
        Optional[torch.Tensor]: 计算得到的帧索引 Tensor，如果规则无效或无法计算则返回 None。
                                返回的 Tensor 中的索引不保证唯一。
    """
    indices = None
    if num_frames == -1: # 加载所有帧
        if total_frames > 0:
            indices = torch.arange(total_frames)
        else:
            return None # 没有帧可加载
    elif num_frames == -2 or num_frames == -3: # 基于 FPS 的采样
        if video_fps <= 0 or total_frames <= 0:
            print(f"警告: FPS ({video_fps}) 或总帧数 ({total_frames}) 无效，无法进行基于时间的采样。")
            return None # 无法计算

        target_fps = 1 if num_frames == -2 else 2
        duration_sec = total_frames / video_fps
        step_sec = 1.0 / target_fps
        # 处理边界情况：如果时长非常短，arange 可能返回空
        if duration_sec <= 0:
             return torch.tensor([], dtype=torch.long) # 返回空 Tensor

        # 生成目标时间戳 [0, step, 2*step, ..., < duration)
        target_timestamps = torch.arange(0, duration_sec, step_sec)
        if len(target_timestamps) == 0 and total_frames > 0:
             # 如果步长大于总时长，至少取第一帧
             target_timestamps = torch.tensor([0.0])
        elif len(target_timestamps) == 0:
             return torch.tensor([], dtype=torch.long) # 确实没有时间点

        indices = torch.round(target_timestamps * video_fps).long()
        indices = torch.clamp(indices, 0, total_frames - 1)

    elif num_frames > 0: # 采样固定数量
         if total_frames <= 0:
              print(f"警告: 视频总帧数为 {total_frames}，无法采样 {num_frames} 帧。")
              return None # 没有帧可采样

         # 均匀采样指定数量的帧
         indices = torch.linspace(0, total_frames - 1, num_frames)
         indices = torch.clamp(torch.round(indices), 0, total_frames - 1).long()
    else: # num_frames 为 0 或其他无效负数
        print(f"警告: 无效的 num_frames={num_frames}。")
        return None # 无效规则

    return indices


class BaseVideoQADataset(Dataset, ABC):
    """
    Abstract Base Class for Video Question Answering Datasets.

    Handles common functionalities like video loading, frame sampling,
    and basic data structure. Subclasses must implement methods for
    loading annotations specific to their format and determining video paths.

    Args:
        annotation_path (str): Path to the annotation file (e.g., JSON).
        video_dir (str): Directory containing the video files.
        num_frames (int): Number of frames to sample per video.
                          If -1, load all frames (use with caution).
        video_reader_backend (str): Backend for reading videos ('decord').
        transform (callable, optional): A transform to apply to the sampled
                                        video tensor ([T, C, H, W]). Defaults to None.
    """
    def __init__(self,
                 annotation_path: str,
                 video_dir: str,
                 num_frames_r: int = 16,
                 num_frames_m: int = 16,
                 video_reader_backend: str = 'decord',
                 transform: Optional[callable] = None):

        super().__init__()
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file not found at {annotation_path}")
        if not os.path.isdir(video_dir):
            warnings.warn(f"Video directory not found at {video_dir}")

        self.video_dir = video_dir
        self.num_frames_r = num_frames_r
        self.num_frames_m = num_frames_m
        self.video_reader_backend = video_reader_backend
        self.transform = transform # Store transform if needed for later application

        print(f"Loading annotations from: {annotation_path}")
        self.qa_data = self._load_annotations(annotation_path)
        if not self.qa_data:
             raise ValueError(f"No QA data loaded from {annotation_path}. Please check the file format and content.")
        print(f"Loaded {len(self.qa_data)} QA pairs.")

    @abstractmethod
    def _load_annotations(self, annotation_path: str) -> List[Dict[str, Any]]:
        """
        Loads annotations from the specific file format.

        Args:
            annotation_path (str): Path to the annotation file.

        Returns:
            List[Dict[str, Any]]: A list where each dictionary represents a
                                  single QA item with necessary keys
                                  (e.g., 'video_id', 'question', 'answer', 'id').
        """
        pass

    @abstractmethod
    def _get_video_path(self, item: Dict[str, Any]) -> str:
        """
        Constructs the full path to the video file based on the annotation item.

        Args:
            item (Dict[str, Any]): A dictionary representing one QA item
                                   from self.qa_data.

        Returns:
            str: The full path to the corresponding video file.
        """
        pass

    def _load_and_sample_video_old(self, video_path: str) -> Optional[torch.Tensor]:
        """
        Loads a video and samples frames using the specified backend.

        Args:
            video_path (str): The full path to the video file.

        Returns:
            Optional[torch.Tensor]: A tensor of shape [T, C, H, W] with sampled
                                    frames (values 0-1), or None if loading fails.
        """
        if not os.path.exists(video_path):
            raise ValueError(f"Warning: Video file not found: {video_path}. Skipping.")

        try:
            if self.video_reader_backend == 'decord':
                vr = VideoReader(video_path, ctx=cpu(0))
                total_frames = len(vr)

                if total_frames <= 0:
                    print(f"Warning: Video has {total_frames} frames: {video_path}. Skipping.")
                    raise ValueError(f"Warning: Video file not found: {video_path}. Skipping.")
                    del vr
                    return None

                # 1. Determine frame indices
                if self.num_frames == -1: # Load all frames
                    indices = torch.arange(total_frames)
                elif self.num_frames > 0: # Sample fixed number
                    # Ensure indices are within bounds [0, total_frames - 1]
                    indices = torch.linspace(0, total_frames - 1, self.num_frames)
                    indices = torch.clamp(torch.round(indices), 0, total_frames - 1).long()
                else: # num_frames is 0 or invalid negative
                    print(f"Warning: Invalid num_frames={self.num_frames}. Skipping {video_path}.")
                    raise ValueError(f"Warning: Video file not found: {video_path}. Skipping.")
                    del vr
                    return None

                # Ensure indices are unique if num_frames is large, though linspace usually handles this
                indices = torch.unique(indices)

                # 2. Read frames using decord
                frames = vr.get_batch(indices.tolist()).asnumpy() # [T, H, W, C] uint8
                del vr

                # 3. Convert to tensor [T, C, H, W], float range [0, 1]
                video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

                # 4. Apply transform if provided (expects tensor input)
                #    Note: HF processors usually want PIL, so transform might be None here.
                if self.transform:
                     video_tensor = self.transform(video_tensor)

            # elif self.video_reader_backend == 'pytorchvideo':
            #     # Placeholder for PyTorchVideo implementation if needed later
            #     # This would involve EncodedVideo, get_clip, and potentially different transform logic
            #     warnings.warn("PyTorchVideo backend not fully implemented in Base class yet.")
            #     return None
            else:
                raise ValueError(f"Unsupported video reader backend: {self.video_reader_backend}")

        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            return None

        # Final check on frame count if num_frames was specified > 0
        if self.num_frames > 0 and video_tensor is not None and video_tensor.shape[0] != len(indices):
             # This might happen if sampling resulted in fewer unique frames than requested
             print(f"Warning: Video {os.path.basename(video_path)} resulted in {video_tensor.shape[0]} frames after sampling, expected {self.num_frames} (requested {len(indices)} unique indices).")
             # Decide on behavior: pad, error, or allow? For now, allow.
             # If strict frame count is needed, add padding/error logic here.

        return video_tensor

    def _load_and_sample_video(self, video_path: str) -> Optional[Tuple[List[Image.Image], List[Image.Image]]]:
        """
        加载视频，根据 self.num_frames_r 和 self.num_frames_m 分别采样帧，
        并以两个 PIL Image 列表的形式返回。

        采样逻辑 (分别应用于 _r 和 _m):
        - num_frames == -1: 加载所有帧。
        - num_frames == -2: 每秒采样 1 帧。
        - num_frames == -3: 每秒采样 2 帧。
        - num_frames > 0:  均匀采样指定数量的帧。

        Args:
            video_path (str): 视频文件的完整路径。

        Returns:
            Optional[Tuple[List[Image.Image], List[Image.Image]]]:
                一个包含两个列表的元组 (pil_image_list_r, pil_image_list_m)。
                每个列表包含对应规则采样得到的 PIL Image 对象。
                如果加载或处理失败，则返回 None。
        """
        if not os.path.exists(video_path):
            print(f"警告: 视频文件未找到: {video_path}。跳过。")
            return None

        pil_image_list_r: List[Image.Image] = []
        pil_image_list_m: List[Image.Image] = []
        vr = None  # 初始化 vr 以便在 finally 中清理

        try:
            if self.video_reader_backend == 'decord':
                vr = VideoReader(video_path, ctx=cpu(0))
                total_frames = len(vr)

                if total_frames <= 0:
                    print(f"警告: 视频包含 {total_frames} 帧: {video_path}。跳过。")
                    # del vr # vr 会在 finally 中处理
                    return None

                video_fps = vr.get_avg_fps()
                # 对于 FPS 采样，需要有效的 FPS
                needs_fps = self.num_frames_r in [-2, -3] or self.num_frames_m in [-2, -3]
                if needs_fps and video_fps <= 0:
                    print(f"警告: 无法获取有效的 FPS ({video_fps})，但采样规则需要它。视频: {video_path}。跳过。")
                    # del vr
                    return None

                # 1. 计算两组索引
                indices_r_raw = _calculate_indices(self.num_frames_r, total_frames, video_fps)
                indices_m_raw = _calculate_indices(self.num_frames_m, total_frames, video_fps)

                # 如果任一索引计算失败，则无法继续
                if indices_r_raw is None or indices_m_raw is None:
                    print(f"错误: 无法为视频 {video_path} 计算一组或两组帧索引。跳过。")
                    # del vr
                    return None

                # 2. 合并并获取所有需要读取的唯一索引
                # 保留原始顺序以便后续分配
                # 注意：即使原始索引列表为空，torch.cat也能处理
                all_indices_to_read = torch.cat((indices_r_raw, indices_m_raw))
                # unique() 返回排序后的唯一值，这对于 get_batch 可能更高效
                unique_indices_to_read, inverse_indices = torch.unique(all_indices_to_read, return_inverse=True)

                # 如果没有索引需要读取（例如，两个 num_frames 都无效或视频太短）
                if len(unique_indices_to_read) == 0:
                    print(f"信息: 根据采样规则，视频 {video_path} 无需或无法采样任何帧。返回空列表。")
                    # del vr
                    return [], []  # 返回两个空列表是合理的

                unique_indices_list = unique_indices_to_read.tolist()

                # 3. 使用 decord 一次性读取所有需要的帧
                # .asnumpy() 返回 [T_unique, H, W, C] 形状的 uint8 NumPy 数组
                frames_np_unique = vr.get_batch(unique_indices_list).asnumpy()

                # 检查读取到的帧数是否符合预期
                if frames_np_unique.shape[0] != len(unique_indices_list):
                    print(f"警告: get_batch 读取到的帧数 ({frames_np_unique.shape[0]}) "
                          f"与请求的唯一索引数 ({len(unique_indices_list)}) 不匹配。视频: {video_path}")
                    # 决定如何处理？可以尝试继续，或者直接失败
                    # 为简单起见，这里选择继续，但可能会导致后续索引错误

                # 4. 创建从唯一索引到对应帧数据的映射
                # 使用 NumPy 数组，避免多次转换 PIL
                index_to_frame_map: Dict[int, np.ndarray] = {
                    idx: frames_np_unique[i] for i, idx in enumerate(unique_indices_list)
                    if i < frames_np_unique.shape[0]  # 确保索引在读取到的帧范围内
                }

                # 5. 根据原始索引列表 (indices_r_raw, indices_m_raw) 分配帧到结果列表
                # 处理 R 列表
                for index in indices_r_raw.tolist():
                    frame_np = index_to_frame_map.get(index)
                    if frame_np is not None:
                        pil_image = Image.fromarray(frame_np, mode='RGB')  # 假设 RGB
                        if self.transform:
                            pil_image = self.transform(pil_image)
                        pil_image_list_r.append(pil_image)
                    else:
                        # 如果映射中找不到索引（可能因为 get_batch 警告），记录日志或忽略
                        print(f"警告: 未能在读取的帧中找到索引 {index} (来自 R 规则)。视频: {video_path}")

                # 处理 M 列表
                for index in indices_m_raw.tolist():
                    frame_np = index_to_frame_map.get(index)
                    if frame_np is not None:
                        pil_image = Image.fromarray(frame_np, mode='RGB')  # 假设 RGB
                        if self.transform:
                            pil_image = self.transform(pil_image)
                        pil_image_list_m.append(pil_image)
                    else:
                        print(f"警告: 未能在读取的帧中找到索引 {index} (来自 M 规则)。视频: {video_path}")

            # elif self.video_reader_backend == 'pytorchvideo':
            #     warnings.warn("PyTorchVideo 后端尚未完全实现。")
            #     return None
            else:
                raise ValueError(f"不支持的视频读取后端: {self.video_reader_backend}")

        except Exception as e:
            print(f"处理视频 {video_path} 时发生严重错误: {e}")
            import traceback
            traceback.print_exc()  # 打印详细错误堆栈
            return None  # 出错时返回 None
        finally:
            # 确保 VideoReader 被释放
            if vr is not None:
                try:
                    del vr
                except Exception as del_e:
                    print(f"清理 VideoReader 时出错: {del_e}")

        # 最终帧数检查（可选，但有助于调试）
        if indices_r_raw is not None and len(pil_image_list_r) != len(indices_r_raw):
            print(f"最终检查警告 (R): 视频 {os.path.basename(video_path)} "
                  f"请求了 {len(indices_r_raw)} 帧 (原始索引), 实际获得 {len(pil_image_list_r)} 帧。")
        if indices_m_raw is not None and len(pil_image_list_m) != len(indices_m_raw):
            print(f"最终检查警告 (M): 视频 {os.path.basename(video_path)} "
                  f"请求了 {len(indices_m_raw)} 帧 (原始索引), 实际获得 {len(pil_image_list_m)} 帧。")

        return pil_image_list_r, pil_image_list_m  # <--- 返回包含两个列表的元组


    def __len__(self) -> int:
        return len(self.qa_data)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        if idx >= len(self.qa_data):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self.qa_data)}")

        qa_item = self.qa_data[idx]

        # --- Get Video Path ---
        video_path = self._get_video_path(qa_item)

        # --- Load and Sample Video ---
        video_image_list_r,video_image_list_m = self._load_and_sample_video(video_path)

        if video_image_list_r is None or video_image_list_m is None:
            # Video loading failed, return None to be filtered by collate_fn
            raise ValueError(f"视频帧读取出错")

        # --- Prepare Output Dictionary ---
        # Subclasses might need to override this if key names differ
        print('data len:',len(video_image_list_r))
        try:
            output = {
                "video_r": video_image_list_r, # [f_r, C, H, W] tensor
                "video_m": video_image_list_m,  # [f_r, C, H, W] tensor
                "question": qa_item['question'],
                "answer": qa_item['answer'],
                "video_id": qa_item.get('video_id', os.path.basename(video_path).split('.')[0]), # Default if key missing
                "question_id": qa_item.get('id', qa_item.get('question_id', idx)) # Try common keys or use index
            }
        except KeyError as e:
            print(f"Error accessing expected key '{e}' in annotation item at index {idx}. Item: {qa_item}. Skipping.")
            return None

        return output