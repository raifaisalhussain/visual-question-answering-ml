import torch
from torch.utils.data import DataLoader
from model_for_camparison import load_sentence_embeddings, ITM_Dataset, ITM_Model, evaluate_model
import sys


# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === File Paths ===
IMAGES_PATH = "./ITM_Classifier-baselines/visual7w-images"
TEST_FILE_PATH = "./ITM_Classifier-baselines/visual7w-text/v7w.TestImages.itm.txt"
EMBEDDING_FILE_PATH = "./ITM_Classifier-baselines/v7w.sentence_embeddings-gtr-t5-large.pkl"
MODEL_PATH = "./cnn_trained_model.pth"  # Make sure this path is correct

# === Load Sentence Embeddings ===
print("Loading sentence embeddings...")
sentence_embeddings = load_sentence_embeddings(EMBEDDING_FILE_PATH)

# === Load Dataset and DataLoader ===
print("Preparing test dataset and dataloader...")
test_dataset = ITM_Dataset(IMAGES_PATH, TEST_FILE_PATH, sentence_embeddings, data_split="test")
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# === Load Model ===
MODEL_ARCHITECTURE = "CNN"  # Change to "ViT" or "CUSTOM" if needed
print(f"Loading model architecture: {MODEL_ARCHITECTURE}")
model = ITM_Model(num_classes=2, ARCHITECTURE=MODEL_ARCHITECTURE, PRETRAINED=True).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
print(f"Model loaded from '{MODEL_PATH}'")

# === Evaluate Model ===
print("Evaluating model...")
accuracy, balanced_accuracy, confusion_matrix = evaluate_model(model, test_loader, device)

# === Print Results ===
print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")
print(f"Final Balanced Accuracy: {balanced_accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix)
