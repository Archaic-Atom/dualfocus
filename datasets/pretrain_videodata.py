import os
import json
import warnings
from typing import Dict, Any, List, Optional, Tuple
from basedataset2 import BaseVideoQADataset, collate_fn


class VideoDataset(BaseVideoQADataset):
    """
    PyTorch Dataset class for the VideoQA dataset, inheriting from BaseVideoQADataset.

    Handles the specific annotation format of VideoQA

    Args:
        qa_path (str): Path to the questions answers JSON file (e.g., 'train_q.json').
        video_dir (str): Directory containing ActivityNet video files (v_*.mp4 or v_*.mkv).
        num_frames_r (int): Number of frames to sample for video_r.
        num_frames_m (int): Number of frames to sample for video_m.
        transform (callable, optional): Transform to apply to PIL images.
        video_reader_backend (str): Video reading backend ('decord').
    """

    def __init__(self,
                 qa_path: str,
                 video_dir: str,
                 num_frames_r: int = 16,
                 num_frames_m: int = 16,
                 transform: Optional[callable] = None,
                 video_reader_backend: str = 'decord'):

        self.qa_path = qa_path

        super().__init__(
            annotation_path=".",
            video_dir=video_dir,
            num_frames_r=num_frames_r,
            num_frames_m=num_frames_m,
            video_reader_backend=video_reader_backend,
            transform=transform
        )

        print(f"Dataset initialized. Loaded {len(self.qa_data)} QA pairs.")
        if len(self.qa_data) > 0:
            print("Sample Video QA item:", self.qa_data[0])

    def _load_annotations(self, annotation_path: str) -> List[Dict[str, Any]]:
        """Loads and combines annotations from separate Q and A JSON files."""
        try:
            if not os.path.exists(self.qa_path):
                raise FileNotFoundError(f"Question file not found: {self.qa_path}")
            with open(self.qa_path, 'r') as f:
                qa_data = json.load(f)

            qa_list = []
            for i,qa_item in enumerate(qa_data):
                qid = i

                merged_item = {
                    'video_name': qa_item['video_id'],
                    'question': qa_item['q'],
                    'question_id': qid,
                    'answer': qa_item['a'],
                }
                qa_list.append(merged_item)

            return qa_list

        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {self.q_path} or {self.a_path}")
            return []
        except Exception as e:
            print(f"An unexpected error occurred during annotation loading: {e}")
            return []

    def _get_video_path(self, item: Dict[str, Any]) -> str:
        """Constructs the video path for an Video item."""
        video_name = item['video_name']
        video_extensions = ['.mp4', '.mkv', '.avi', '.webm']

        for ext in video_extensions:
            candidate_path = os.path.join(self.video_dir, video_name + ext)
            if os.path.exists(candidate_path):
                return candidate_path

        raise FileNotFoundError(
            f"Video file for {video_name} not found in any of the search directories"
        )


