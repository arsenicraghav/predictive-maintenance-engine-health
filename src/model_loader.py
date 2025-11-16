from huggingface_hub import hf_hub_download
import joblib

def load_model():
    # Download model file from Hugging Face Model Hub
    model_path = hf_hub_download(
        repo_id="labhara/predictive-maintenance-bestmodel", 
        filename="GradientBoosting_model.pkl"
    )
    model = joblib.load(model_path)
    return model
