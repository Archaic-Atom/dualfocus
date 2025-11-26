
import os,warnings
import json
import torch
from basedataset import BaseVideoQADataset,collate_fn # Import the base class
from typing import Dict, Any, List, Optional

class MSRVTTQADataset(BaseVideoQADataset):
    """
    PyTorch Dataset class for the MSRVTT-QA dataset, inheriting from BaseVideoQADataset.

    Args:
        json_path (str): Path to the MSRVTT-QA JSON file (e.g., 'train_qa.json').
        video_dir (str): Directory containing MSRVTT video files (video*.mp4).
        num_frames (int): Number of frames to sample per video.
        transform (callable, optional): Transform to apply to the video tensor.
                                        (Often None if using external processors like HF's).
        video_reader_backend (str): Video reading backend ('decord').
    """
    def __init__(self,
                 json_path: str,
                 video_dir: str,
                 num_frames_r: int = 16,
                 num_frames_m: int = 16,
                 transform: Optional[callable] = None,
                 video_reader_backend: str = 'decord'):

        super().__init__(
            annotation_path=json_path,
            video_dir=video_dir,
            num_frames_r=num_frames_r,
            num_frames_m=num_frames_m,
            video_reader_backend=video_reader_backend,
            transform=transform # Pass transform to base class
        )


        self.video_template = "video{}.mp4"

        print(f"MSRVTTQADataset initialized using base class logic.")
        if len(self.qa_data) > 0:
             print("First MSRVTT QA item:", self.qa_data[0])


    def _load_annotations(self, annotation_path: str) -> List[Dict[str, Any]]:
        """Loads annotations from the MSRVTT-QA JSON file."""
        try:
            with open(annotation_path, 'r') as f:
                qa_data = json.load(f)
            if not isinstance(qa_data, list):
                raise TypeError(f"Expected a list of QA pairs, but got {type(qa_data)}")
            if qa_data and not all(k in qa_data[0] for k in ['video_id', 'question', 'answer', 'id']):
                 warnings.warn(f"First QA item in {annotation_path} might be missing expected keys ('video_id', 'question', 'answer', 'id').")
            return qa_data
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {annotation_path}")
            return []
        except Exception as e:
             print(f"An unexpected error occurred during annotation loading: {e}")
             return []

    def _get_video_path(self, item: Dict[str, Any]) -> str:
        """Constructs the video path for an MSRVTT item."""
        try:
            video_id = item['video_id']
            video_filename = self.video_template.format(video_id)
            return os.path.join(self.video_dir, video_filename)
        except KeyError:
             raise ValueError(f"Annotation item is missing the 'video_id' key: {item}")



if __name__ == "__main__":
    # Adjust paths as needed for your setup
    TRAIN_JSON_PATH = '../video_data/MSRVTT/MSRVTT-QA/val_qa.json'
    VIDEO_DIR = '../video_data/MSRVTT/MSRVTT/MSRVTT_Videos'
    NUM_FRAMES_R = -1
    NUM_FRAMES_M = -2
    BATCH_SIZE = 4

    print("--- Testing Refactored MSRVTTQADataset ---")

    try:
        # Instantiate the specific dataset
        msrvtt_dataset = MSRVTTQADataset(
            json_path=TRAIN_JSON_PATH,
            video_dir=VIDEO_DIR,
            num_frames_r=NUM_FRAMES_R,
            num_frames_m=NUM_FRAMES_M,
            transform=None # Assuming HF processors will handle transforms later
        )

        if len(msrvtt_dataset) == 0:
            print("Dataset loaded but is empty. Exiting test.")
            exit()

        print(f"\nDataset Size: {len(msrvtt_dataset)}")

        # Test getting a single item
        print("\nTesting __getitem__...")
        first_valid_item = None
        for i in range(len(msrvtt_dataset)):
            try:
                item = msrvtt_dataset[i]
                if item is not None:
                    first_valid_item = item
                    print(f"Successfully loaded item at index {i}:")
                    print(f"  Video_R PIL Len: {len(item['video_r'])}")
                    print(f"  Video_M PIL Len: {len(item['video_m'])}")
                    print(f"  Question: {item['question'][:50]}...") # Print snippet
                    print(f"  Answer: {item['answer']}")
                    print(f"  Video ID: {item['video_id']}")
                    print(f"  Question ID: {item['question_id']}")
                    break
            except IndexError:
                 print(f"Index {i} out of bounds.")
                 break
            except Exception as e:
                 print(f"Error getting item at index {i}: {e}")
        if first_valid_item is None:
            print("Could not retrieve any valid item from the dataset.")
            exit()

        # Test with DataLoader
        print("\nTesting DataLoader...")
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            msrvtt_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False, # Set True for training
            num_workers=2, # Adjust based on system
            collate_fn=collate_fn # Use the custom collate function
        )

        num_batches_to_test = 3
        for i, batch in enumerate(dataloader):
            print(f"\n--- Batch {i+1} ---")
            if not batch["video_r"]:
                print("  Batch is empty after collation (all items failed?).")
                continue

            print(f"  Batch Size (effective): {len(batch['video_r'])}")
            print(f"  Type of batch['video']: {type(batch['video_r'])}") # Should be list
            # Print shape of the first video tensor in the batch
            if batch['video_r']:
                print(f"  Video_R PIL Len: {len(batch['video_r'])}")
                print(f"  Video_M PIL Len: {len(batch['video_m'])}")
            print(f"  Questions (first {BATCH_SIZE}): {batch['question']}")
            print(f"  Video IDs (first {BATCH_SIZE}): {batch['video_id']}")
            print(f"  Answer (first {BATCH_SIZE}): {batch['answer']}")

            if i >= num_batches_to_test - 1:
                break

        print("\nDataLoader test finished.")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please check your JSON and Video paths.")
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")

    print("\n--- Refactored Dataset Test Complete ---")