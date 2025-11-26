import os
import json
import warnings
from typing import Dict, Any, List, Optional, Tuple
from basedataset import BaseVideoQADataset, collate_fn


class ActivityNetQADataset(BaseVideoQADataset):
    """
    PyTorch Dataset class for the ActivityNet-QA dataset, inheriting from BaseVideoQADataset.

    Handles the specific annotation format of ActivityNet-QA where questions and answers
    are stored in separate JSON files (*_q.json and *_a.json).

    Args:
        q_path (str): Path to the questions JSON file (e.g., 'train_q.json').
        a_path (str): Path to the answers JSON file (e.g., 'train_a.json').
        video_dir (str): Directory containing ActivityNet video files (v_*.mp4 or v_*.mkv).
        num_frames_r (int): Number of frames to sample for video_r.
        num_frames_m (int): Number of frames to sample for video_m.
        transform (callable, optional): Transform to apply to PIL images.
        video_reader_backend (str): Video reading backend ('decord').
    """

    def __init__(self,
                 q_path: str,
                 a_path: str,
                 video_dir: str,
                 num_frames_r: int = 16,
                 num_frames_m: int = 16,
                 transform: Optional[callable] = None,
                 video_reader_backend: str = 'decord'):

        self.q_path = q_path
        self.a_path = a_path

        super().__init__(
            annotation_path=".",
            video_dir=video_dir,
            num_frames_r=num_frames_r,
            num_frames_m=num_frames_m,
            video_reader_backend=video_reader_backend,
            transform=transform
        )

        print(f"ActivityNetQADataset initialized. Loaded {len(self.qa_data)} QA pairs.")
        if len(self.qa_data) > 0:
            print("Sample ActivityNet QA item:", self.qa_data[0])

    def _load_annotations(self, annotation_path: str) -> List[Dict[str, Any]]:
        """Loads and combines annotations from separate Q and A JSON files."""
        try:
            if not os.path.exists(self.q_path):
                raise FileNotFoundError(f"Question file not found: {self.q_path}")
            with open(self.q_path, 'r') as f:
                q_data = json.load(f)

            if not os.path.exists(self.a_path):
                raise FileNotFoundError(f"Answer file not found: {self.a_path}")
            with open(self.a_path, 'r') as f:
                a_data = json.load(f)

            a_dict = {item['question_id']: item for item in a_data}

            qa_list = []
            for q_item in q_data:
                qid = q_item['question_id']
                if qid not in a_dict:
                    warnings.warn(f"Question ID {qid} not found in answer file. Skipping.")
                    continue

                merged_item = {
                    'video_name': q_item['video_name'],
                    'question': q_item['question']+'?',
                    'question_id': qid,
                    'answer': a_dict[qid]['answer'],
                    'type': a_dict[qid].get('type', -1)
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
        """Constructs the video path for an ActivityNet item."""
        video_name = item['video_name']
        base_filename = f"v_{video_name}"
        print('video_name:',base_filename)

        video_extensions = ['.mp4', '.mkv', '.avi', '.webm', '.mov', '.flv']
        for ext in video_extensions:
            candidate_path = os.path.join(self.video_dir, base_filename + ext)
            if os.path.exists(candidate_path):
                return candidate_path

        raise FileNotFoundError(f"Video file for '{video_name}' not found in {self.video_dir} with common extensions.")



if __name__ == "__main__":
    Q_PATH = '/data/datasets/activitynet/activitynet-qa/val_q.json'
    A_PATH = '/data/datasets/activitynet/activitynet-qa/val_a.json'
    VIDEO_DIR = '/data/datasets/activitynet/videos'
    NUM_FRAMES_R = 140
    NUM_FRAMES_M = 140

    print("\n--- Testing ActivityNetQADataset ---")

    try:
        anet_dataset = ActivityNetQADataset(
            q_path=Q_PATH,
            a_path=A_PATH,
            video_dir=VIDEO_DIR,
            num_frames_r=NUM_FRAMES_R,
            num_frames_m=NUM_FRAMES_M,
            transform=None
        )

        if len(anet_dataset) == 0:
            print("Dataset loaded but is empty. Exiting test.")
            exit()

        print(f"\nDataset Size: {len(anet_dataset)}")

        print("\nTesting __getitem__...")
        for i in range(min(5, len(anet_dataset))):
            try:
                item = anet_dataset[i]
                if item is None:
                    print(f"  Item {i} returned None (likely video loading failed)")
                    continue

                print(f"  Item {i}:")
                print(f"    Video_R PIL Frames: {len(item['video_r'])}")
                print(f"    Video_M PIL Frames: {len(item['video_m'])}")
                print(f"    Question: {item['question']}")
                print(f"    Answer: {item['answer']}")
                print(f"    Video ID: {item['video_id']}")
                print(f"    Question ID: {item['question_id']}")

            except Exception as e:
                print(f"  Error getting item {i}: {str(e)}")

        print("\nTesting DataLoader...")
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            anet_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn
        )

        for i, batch in enumerate(dataloader):
            print(f"\nBatch {i + 1}:")
            print(f"  Questions: {batch['question']}")
            print(f"  Answers: {batch['answer']}")
            print(f"  Video IDs: {batch['video_id']}")
            print(f"  Question IDs: {batch['question_id']}")

            if batch['video_r']:
                print(f"  Video_R frame counts: {[len(v) for v in batch['video_r']]}")
            if batch['video_m']:
                print(f"  Video_M frame counts: {[len(v) for v in batch['video_m']]}")

            if i >= 2:
                break

        print("\n--- ActivityNetQADataset Test Completed Successfully ---")

    except Exception as e:
        print(f"\n!!! Test Failed: {str(e)}")
        import traceback

        traceback.print_exc()