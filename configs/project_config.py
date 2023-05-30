import torch
from pathlib import Path


# Project root, computing resources
class ProjectConfig:
    project_root = Path(__file__).absolute().parent.parent
    single_gpu = 1  # TODO: which gpu to run
    device = torch.device(f"cuda:{single_gpu}")
    # device = torch.device(f"cpu")
    torch.cuda.set_device(device)
    num_workers = 0  # Number of threads for loading data
    run_root = Path("./runs")  # Save each run
    final_data_root = Path("./final_data")  # Cache preprocessed data
