import os
from huggingface_hub import HfApi


def upload_file(file_path, name):
    api = HfApi()
    api.upload_file(
        path_or_fileobj=name,
        path_in_repo="README.md",
        repo_id="username/test-dataset",
        repo_type="dataset",
    )


def upload_foler(folder):
    api = HfApi()
    api.upload_folder(
        folder_path="/mnt/data/jiabao/2025-Projects/MultiRobotR1/data_cache/robot",
        repo_id="jiabao/custom_data",
        repo_type="dataset",
    )
