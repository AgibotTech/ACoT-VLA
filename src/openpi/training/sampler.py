
import torch
import random
from tqdm import tqdm
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

def get_base_dataset(ds):
    if hasattr(ds, "_dataset"):
        return get_base_dataset(ds._dataset)
    return ds

def sample_subtask(dataset):
    valid_intervals = []
    base_ds = get_base_dataset(dataset)
    
    sub_datasets = []
    
    if isinstance(base_ds, lerobot_dataset.MultiLeRobotDataset):
        for sub_ds in base_ds._datasets:
            sub_datasets.append(sub_ds)
    else:
        sub_datasets.append(dataset)

    current_global_offset = 0
    total_episodes_processed = 0

    print(f"Processing {len(sub_datasets)} sub-datasets...")

    for sub_ds in sub_datasets:
        inner_ds = get_base_dataset(sub_ds)
        
        instruction_segment = inner_ds.meta.info.get('instruction_segments', {})
        episode_data_index = inner_ds.episode_data_index
        num_episodes = len(episode_data_index['from'])
        
        for ep_idx in range(num_episodes):
            local_episode_start = episode_data_index['from'][ep_idx].item()
            
            if str(ep_idx) not in instruction_segment:
                continue

            tasks = instruction_segment[str(ep_idx)]
            for subtask in tasks:
                local_start = subtask["start_frame_index"] + local_episode_start
                end_frame_key = "success_frame_index" if "success_frame_index" in subtask else "end_frame_index"
                local_end = subtask[end_frame_key] + local_episode_start
                
                instruction = subtask["instruction"].lower()
                is_reset = any(k in instruction for k in ['reset', 'return', 'default'])
                
                if is_reset:
                    if local_end - local_start > 90:
                        local_end = local_start + 45
                
                global_start = local_start + current_global_offset
                global_end = local_end + current_global_offset
                
                valid_intervals.append((global_start, global_end))
        
        current_global_offset += len(sub_ds)
        total_episodes_processed += num_episodes

    print(f"Total {len(valid_intervals)} valid intervals from {total_episodes_processed} episodes.")
    return valid_intervals


class FrameSampler(torch.utils.data.Sampler):
    """
    Custom sampler that only samples data indices falling within specified intervals
    """
    def __init__(self, dataset, sampler_type):
        valid_intervals = self.parse_dataset(dataset, sampler_type)
        self.sample_frames(valid_intervals, len(dataset))

    def parse_dataset(self, dataset, sampler_type):
        """
        Args:
            intervals: List of (start_index, end_index) tuples
        """
        if sampler_type == 'subtask':
            return sample_subtask(dataset)
        else:
            raise ValueError(f"Invalid sampler type: {sampler_type}")

    def sample_frames(self, intervals, dataset_size):
        """
        Args:
            intervals: List of (start_index, end_index) tuples
            dataset_size: Total size of the dataset
        """
        self.intervals = intervals
        self.dataset_size = dataset_size
        
        # Pre-compute all valid indices
        self.valid_indices = []
        for start_idx, end_idx in intervals:
            # Ensure indices are within dataset bounds
            start_idx = max(0, start_idx)
            end_idx = min(dataset_size - 1, end_idx)
            
            # Add all indices within the interval
            self.valid_indices.extend(range(start_idx, end_idx + 1))
        
        # Remove duplicates and sort
        self.valid_indices = sorted(list(set(self.valid_indices)))
        print(f"Total {len(self.valid_indices)} valid indices,", 'original:', dataset_size)

        import random
        random.shuffle(self.valid_indices)
    
    def __iter__(self):
        return iter(self.valid_indices)
    
    def __len__(self):
        return len(self.valid_indices)
