import os
import warnings
import json
from typing import Dict, Any, List, Optional
from basedataset import BaseVideoQADataset, collate_fn


class MSVDQADataset(BaseVideoQADataset):
    """
    PyTorch Dataset class for the MSVD-QA dataset, inheriting from BaseVideoQADataset.

    Args:
        annotation_path (str): Path to the MSVD-QA JSON file (e.g., 'train_qa.json').
        video_dir (str): Directory containing MSVD video files (YouTubeClips directory).
        mapping_path (str): Path to the mapping file (youtube_mapping.txt).
        num_frames_r (int): Number of frames to sample for video_r.
        num_frames_m (int): Number of frames to sample for video_m.
        transform (callable, optional): Transform to apply to the video frames.
        video_reader_backend (str): Video reading backend ('decord').
    """

    def __init__(self,
                 annotation_path: str,
                 video_dir: str,
                 mapping_path: str,
                 num_frames_r: int = 16,
                 num_frames_m: int = 16,
                 transform: Optional[callable] = None,
                 video_reader_backend: str = 'decord'):

        super().__init__(
            annotation_path=annotation_path,
            video_dir=video_dir,
            num_frames_r=num_frames_r,
            num_frames_m=num_frames_m,
            video_reader_backend=video_reader_backend,
            transform=transform
        )

        self.mapping_path = mapping_path
        self.video_id_to_filename = self._load_mapping(mapping_path)

        print(f"MSVDQADataset initialized using base class logic.")
        if len(self.qa_data) > 0:
            print("First MSVD QA item:", self.qa_data[0])

    def _load_mapping(self, mapping_path: str) -> Dict[str, str]:
        """
        Load the mapping from video ID to video filename.

        Args:
            mapping_path (str): Path to the mapping file.

        Returns:
            Dict[str, str]: A dictionary mapping video IDs to filenames.
        """
        mapping = {}
        if not os.path.exists(mapping_path):
            warnings.warn(f"Mapping file not found: {mapping_path}")
            return mapping

        try:
            with open(mapping_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    filename = parts[0] + ".avi"  # Append the .avi extension
                    video_id = parts[1][3:]
                    mapping[video_id] = filename
        except Exception as e:
            print(f"Error loading mapping file: {e}")

        # print("mapping:",mapping)
        return mapping

    def _load_annotations(self, annotation_path: str) -> List[Dict[str, Any]]:
        """Load annotations from the MSVD-QA JSON file."""
        try:
            with open(annotation_path, 'r') as f:
                qa_data = json.load(f)
            if not isinstance(qa_data, list):
                raise TypeError(f"Expected a list of QA pairs, but got {type(qa_data)}")
            # Basic check for expected keys in the first item
            if qa_data and not all(k in qa_data[0] for k in ['video_id', 'question', 'answer', 'id']):
                warnings.warn(f"First QA item in {annotation_path} might be missing expected keys.")
            return qa_data
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {annotation_path}")
            return []
        except Exception as e:
            print(f"An unexpected error occurred during annotation loading: {e}")
            return []

    def _get_video_path(self, item: Dict[str, Any]) -> str:
        """Construct the video path for an MSVD item using the mapping."""
        try:
            video_id = item['video_id']
            video_filename = self.video_id_to_filename.get(str(video_id))
            if not video_filename:
                raise ValueError(f"Video ID {video_id} not found in mapping.")
            return os.path.join(self.video_dir, video_filename)
        except KeyError:
            raise ValueError(f"Annotation item is missing the 'video_id' key: {item}")


# Example usage for testing
if __name__ == "__main__":
    # Adjust paths as needed
    ANNOTATION_PATH = '../video_data/MSVD/MSVD-QA/val_qa.json'
    VIDEO_DIR = '../video_data/MSVD/YouTubeClips'
    MAPPING_PATH = '../video_data/MSVD/youtube_mapping.txt'
    NUM_FRAMES_R = -1
    NUM_FRAMES_M = -2
    BATCH_SIZE = 4

    print("--- Testing MSVDQADataset ---")
    try:
        # Instantiate the dataset
        msvd_dataset = MSVDQADataset(
            annotation_path=ANNOTATION_PATH,
            video_dir=VIDEO_DIR,
            mapping_path=MAPPING_PATH,
            num_frames_r=NUM_FRAMES_R,
            num_frames_m=NUM_FRAMES_M,
            transform=None
        )

        if len(msvd_dataset) == 0:
            print("Dataset loaded but is empty. Exiting test.")
            exit()

        print(f"\nDataset Size: {len(msvd_dataset)}")

        # Test getting a single item
        print("\nTesting __getitem__...")
        first_valid_item = None
        for i in range(len(msvd_dataset)):
            try:
                item = msvd_dataset[i]
                if item is not None:
                    first_valid_item = item
                    print(f"Successfully loaded item at index {i}:")
                    print(f"  Video_R PIL Len: {len(item['video_r'])}")
                    print(f"  Video_M PIL Len: {len(item['video_m'])}")
                    print(f"  Question: {item['question'][:50]}...")
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
            msvd_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn
        )

        num_batches_to_test = 3
        for i, batch in enumerate(dataloader):
            print(f"\n--- Batch {i + 1} ---")
            if not batch["video_r"]:
                print("  Batch is empty after collation (all items failed?).")
                continue

            print(f"  Batch Size (effective): {len(batch['video_r'])}")
            if batch['video_r']:
                print(f"  Video_R PIL Len: {len(batch['video_r'][0])}")
                print(f"  Video_M PIL Len: {len(batch['video_m'][0])}")
            print(f"  Questions (first {BATCH_SIZE}): {batch['question']}")
            print(f"  Video IDs (first {BATCH_SIZE}): {batch['video_id']}")
            print(f"  Answer (first {BATCH_SIZE}): {batch['answer']}")

            if i >= num_batches_to_test - 1:
                break

        print("\nDataLoader test finished.")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please check your paths.")
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")

    print("\n--- Dataset Test Complete ---")