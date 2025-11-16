from huggingface_hub import HfApi
import os

def upload_space():
    api = HfApi()
    space_repo_id = "labhara/predictive-maintenance-app"

    folder_path = os.path.dirname(os.path.abspath(__file__))

    api.upload_folder(
        folder_path=folder_path,
        repo_id=space_repo_id,
        repo_type="space"
    )

if __name__ == "__main__":
    upload_space()
