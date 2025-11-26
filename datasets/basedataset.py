import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from decord import VideoReader, cpu
import warnings
from PIL import Image
from typing import Optional, List, Tuple, Dict,Any # 导入
import math

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

def _sample_by_fps(total_frames: int, video_fps: float, target_fps: int) -> Optional[torch.Tensor]:
    if video_fps <= 0 or total_frames <= 0 or target_fps <= 0:
        return None

    duration_sec = total_frames / video_fps
    if duration_sec <= 0:
        return torch.tensor([], dtype=torch.long)

    step_sec = 1.0 / target_fps
    target_timestamps = torch.arange(0, duration_sec, step_sec)

    if len(target_timestamps) == 0:
        target_timestamps = torch.tensor([0.0])

    indices = torch.round(target_timestamps * video_fps).long()
    indices = torch.clamp(indices, 0, total_frames - 1)
    return indices


def _calculate_indices(num_frames: int, total_frames: int, video_fps: float) -> Optional[torch.Tensor]:
    """
    Frame indexes are calculated based on sampling rules (optimized and refactored).
    Args:
            num_frames (int): Sampling rules (-1, -2, -3, >0).
            total_frames (int): The total number of frames in the video.
            video_fps (float): Video frame rate.

    Returns:
            Optional[torch. Tensor]: The computed frame index Tensor. The index in the returned Tensor is not guaranteed to be unique.
                                    If the rule is invalid or cannot be calculated, None is returned.
    """
    if total_frames <= 0:
        if num_frames > 0:
            print(f"警告: 视频总帧数为 {total_frames}，无法采样 {num_frames} 帧。")
        return None if num_frames != -1 else torch.tensor([], dtype=torch.long)

    if num_frames == -1:  # all
        return torch.arange(total_frames)

    elif num_frames == -2:  # 1 FPS
        return _sample_by_fps(total_frames, video_fps, 1)

    elif num_frames == -3:  # 2 FPS
        return _sample_by_fps(total_frames, video_fps, 2)

    elif num_frames > 0:
        fps1_frame_count = 0
        if video_fps > 0:
            duration_sec = total_frames / video_fps
            fps1_frame_count = math.ceil(duration_sec) if duration_sec > 0 else 0

        if video_fps > 0 and fps1_frame_count <= num_frames:
            return _sample_by_fps(total_frames, video_fps, 1)
        else:
            indices = torch.linspace(0, total_frames - 1, num_frames)
            return torch.round(indices).long()

    else:  # num_frames 为 0 或其他无效负数
        print(f"警告: 无效的 num_frames={num_frames}。")
        return None


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
        self.transform = transform

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

    def _load_and_sample_video(self, video_path: str) -> Optional[
        Tuple[List[Image.Image], List[Image.Image]]]:
        """
            Load videos and sample frames efficiently (optimized).

        Core Optimization:
            1. Use the 'return_inverse=True' parameter of 'torch.unique' to reconstruct the frame sequence with a vectorized index operation,
               Avoiding the use of dictionaries and loop lookups.
            2. Defer the transformation and image transformation ('transform') of NumPy -> PIL to the end, executed on the detached frame list,
               Make sure that each unique frame is processed only once.

        Args:
                video_path (str): The full path to the video file.

        Returns:
                Optional[Tuple[List, List]]:
                    A tuple containing two lists (list_r, list_m).
                    The list content depends on whether self.transform exists:
                    - If present, the transformed image (usually Tensor).
                    - If it doesn't exist, use the PIL. Image object.
                    If loading or processing fails, None is returned.
        """
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}. Skip. ")
            return None

        vr = None
        try:
            if self.video_reader_backend == 'decord':
                vr = VideoReader(video_path, ctx=cpu(0))
                total_frames = len(vr)

                if total_frames <= 0:
                    print(f"Warning: Video contains {total_frames} frame: {video_path}. Skip. ")
                    return None

                video_fps = vr.get_avg_fps()

                indices_r = _calculate_indices(self.num_frames_r, total_frames, video_fps)
                indices_m = _calculate_indices(self.num_frames_m, total_frames, video_fps)

                if indices_r is None or indices_m is None:
                    print(f"Error: Unable to calculate frame index for video {video_path}. Skip. ")
                    return None


                len_r = len(indices_r)
                all_indices = torch.cat((indices_r, indices_m))

                if len(all_indices) == 0:

                    return [], []

                unique_indices, inverse_map = torch.unique(all_indices, return_inverse=True)

                unique_frames_np = vr.get_batch(unique_indices.tolist()).asnumpy().copy()

                if unique_frames_np.shape[0] != len(unique_indices):
                    raise RuntimeError(
                        f"Decord get_batch error: {len(unique_indices)} frame requested,"
                        f"But only {unique_frames_np.shape[0]} frame was received. Video: {video_path}"
                    )

                all_frames_np_in_order = unique_frames_np[inverse_map]

                frames_np_r = all_frames_np_in_order[:len_r]
                frames_np_m = all_frames_np_in_order[len_r:]

                if self.transform:
                    list_r = [self.transform(Image.fromarray(frame)) for frame in frames_np_r]
                    list_m = [self.transform(Image.fromarray(frame)) for frame in frames_np_m]
                else:
                    list_r = [Image.fromarray(frame, mode='RGB') for frame in frames_np_r]
                    list_m = [Image.fromarray(frame, mode='RGB') for frame in frames_np_m]

                del unique_frames_np
                del all_frames_np_in_order
                del frames_np_r
                del frames_np_m


            else:
                raise ValueError(f"Unsupported video reading backend: {self.video_reader_backend}")

        except Exception as e:
            print(f"Critical error when processing video {video_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            if vr is not None:
                del vr

        return list_r, list_m


    def __len__(self) -> int:
        return len(self.qa_data)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        if idx >= len(self.qa_data):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self.qa_data)}")

        qa_item = self.qa_data[idx]

        video_path = self._get_video_path(qa_item)

        video_image_list_r,video_image_list_m = self._load_and_sample_video(video_path)

        if video_image_list_r is None or video_image_list_m is None:
            raise ValueError(f"videoFrameReadingError")

        print('data len:',len(video_image_list_r))
        try:
            output = {
                "video_r": video_image_list_r,
                "video_m": video_image_list_m,
                "question": qa_item['question'],
                "answer": qa_item['answer'],
                "video_id": qa_item.get('video_id', os.path.basename(video_path).split('.')[0]),
                "question_id": qa_item.get('id', qa_item.get('question_id', idx))
            }
        except KeyError as e:
            print(f"Error accessing expected key '{e}' in annotation item at index {idx}. Item: {qa_item}. Skipping.")
            return None

        return output


import torch
from torch.utils.data import Sampler, Dataset
from typing import Dict, Any, List, Optional


class StatefulSampler(Sampler):
    """
        A Sampler that can save and restore state for stand-alone training.

    It works by saving the complete index order and the number of samples processed.

    Args:
            data_source (Dataset): The dataset used for sampling.
            shuffle (bool): Whether to scramble the data.
            Seed (int, optional): A random seed used to shuffle the deck, ensuring reproducibility.
    """

    def __init__(self, data_source: Dataset, shuffle: bool = False, seed: Optional[int] = None):
        super().__init__(data_source)
        self.data_source = data_source
        self.shuffle = shuffle

        if seed is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        self.seed = seed

        g = torch.Generator()
        g.manual_seed(self.seed)

        if self.shuffle:
            self.indices = torch.randperm(len(self.data_source), generator=g).tolist()
        else:
            self.indices = list(range(len(self.data_source)))

        self.num_consumed_samples = 0

    def __iter__(self):
        return iter(self.indices[self.num_consumed_samples:])

    def __len__(self) -> int:
        return len(self.indices) - self.num_consumed_samples

    def state_dict(self) -> Dict[str, Any]:
        return {
            'seed': self.seed,
            'indices': self.indices,
            'num_consumed_samples': self.num_consumed_samples,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.seed = state_dict['seed']
        self.indices = state_dict['indices']
        self.num_consumed_samples = state_dict['num_consumed_samples']
        print(f"[StatefulSampler] status loaded. Will start with sample {self.num_consumed_samples}.")

    def set_num_consumed_samples(self, num_samples: int):
        self.num_consumed_samples = num_samples