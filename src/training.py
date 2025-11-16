import pandas as pd
from datasets import load_dataset
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from huggingface_hub import HfApi, HfFolder
import joblib
import os

def load_data():
    repo_id = "labhara/predictive-maintenance-dataset-splits"
    dataset = load_dataset(repo_id)
    train_df = dataset['train'].to_pandas()
    test_df = dataset['test'].to_pandas()
    return train_df, test_df

def train_model(train_df, test_df):
    X_train = train_df.drop(columns=["Engine_Condition"])
    y_train = train_df["Engine_Condition"]
    X_test = test_df.drop(columns=["Engine_Condition"])
    y_test = test_df["Engine_Condition"]

    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"GradientBoosting Accuracy: {acc:.4f}")
    print(f"GradientBoosting F1-score: {f1:.4f}")

    return model

def save_model(model, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "GradientBoosting_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    return model_path

def upload_model_to_hf(model_path):
    api = HfApi()
    repo_id = "labhara/predictive-maintenance-bestmodel"

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="GradientBoosting_model.pkl",
        repo_id=repo_id,
        repo_type="model"
    )

    print(f"Uploaded model to https://huggingface.co/{repo_id}")

def main():
    print("Loading data from Hugging Face...")
    train_df, test_df = load_data()

    print("Training GradientBoosting model...")
    model = train_model(train_df, test_df)

    print("Saving model locally...")
    model_path = save_model(model, save_dir="models")

    print("Uploading model to Hugging Face model hub...")
    upload_model_to_hf(model_path)

if __name__ == "__main__":
    main()
