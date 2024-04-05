import torch

def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    
    dataset = worker_info.dataset
    worker_id = worker_info.id
    split_size = len(dataset.data) // worker_info.num_workers
    
    dataset.data = dataset.data[worker_id * split_size: (worker_id + 1) * split_size]
    
    