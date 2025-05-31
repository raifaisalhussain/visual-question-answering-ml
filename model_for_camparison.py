import os
import time
import pickle
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import vit_b_32, resnet18
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
import seaborn as sns

# Custom Dataset
class ITM_Dataset(Dataset):
    def __init__(self, images_path, data_file, sentence_embeddings, data_split, train_ratio=1.0):
        self.images_path = images_path
        self.data_file = data_file
        self.sentence_embeddings = sentence_embeddings
        self.data_split = data_split.lower()
        self.train_ratio = train_ratio if self.data_split == "train" else 1.0

        self.image_data = []
        self.question_data = []
        self.answer_data = []
        self.question_embeddings_data = []
        self.answer_embeddings_data = []
        self.label_data = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard for pretrained models on ImageNet
        ])

        self.load_data()

    def load_data(self):
        print("LOADING data from "+str(self.data_file))
        print("=========================================")

        random.seed(42)

        with open(self.data_file) as f:
            lines = f.readlines()

            # Apply train_ratio only for training data
            if self.data_split == "train":
                random.shuffle(lines)  # Shuffle before selecting
                num_samples = int(len(lines) * self.train_ratio)
                lines = lines[:num_samples]

            for line in lines:
                line = line.rstrip("\n")
                img_name, text, raw_label = line.split("\t")  
                img_path = os.path.join(self.images_path, img_name.strip())

                question_answer_text = text.split("?")
                question_text = question_answer_text[0].strip() + '?'
                answer_text = question_answer_text[1].strip()

                # Get binary labels from match/no-match answers
                label = 1 if raw_label == "match" else 0
                self.image_data.append(img_path)
                self.question_data.append(question_text)
                self.answer_data.append(answer_text)
                self.question_embeddings_data.append(self.sentence_embeddings[question_text])
                self.answer_embeddings_data.append(self.sentence_embeddings[answer_text])
                self.label_data.append(label)

        print("|image_data|="+str(len(self.image_data)))
        print("|question_data|="+str(len(self.question_data)))
        print("|answer_data|="+str(len(self.answer_data)))
        print("done loading data...")

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        img_path = self.image_data[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)  
        question_embedding = torch.tensor(self.question_embeddings_data[idx], dtype=torch.float32)
        answer_embedding = torch.tensor(self.answer_embeddings_data[idx], dtype=torch.float32)
        label = torch.tensor(self.label_data[idx], dtype=torch.long)
        return img, question_embedding, answer_embedding, label

# Load sentence embeddings from an existing file
def load_sentence_embeddings(file_path):
    print("READING sentence embeddings...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# Pre-trained ViT model
class Transformer_VisionEncoder(nn.Module):
    def __init__(self, pretrained=None):
        super(Transformer_VisionEncoder, self).__init__()

        if pretrained:
            self.vision_model = vit_b_32(weights="IMAGENET1K_V1")
            # Freeze all layers initially
            for param in self.vision_model.parameters():
                param.requires_grad = False

            # Unfreeze the last two layers
            for param in list(self.vision_model.heads.parameters())[-2:]:
                param.requires_grad = True
        else:
            self.vision_model = vit_b_32(weights=None)  # Initialize without pretrained weights
    
        self.num_features = self.vision_model.heads[0].in_features
        self.vision_model.heads = nn.Identity()  # Remove the final classification head

    def forward(self, x):
        features = self.vision_model(x)
        return features

# Image-Text Matching Model
class ITM_Model(nn.Module):
    def __init__(self, num_classes=2, ARCHITECTURE=None, PRETRAINED=None):
        print(f'BUILDING {ARCHITECTURE} model, pretrained={PRETRAINED}')
        super(ITM_Model, self).__init__()
        self.ARCHITECTURE = ARCHITECTURE

        if self.ARCHITECTURE == "CNN":
            self.vision_model = resnet18(pretrained=PRETRAINED)
            if PRETRAINED:
                for param in self.vision_model.parameters():
                    param.requires_grad = False
                for param in list(self.vision_model.children())[-2:]:
                    for p in param.parameters():
                        p.requires_grad = True
            else:
                for param in self.vision_model.parameters():
                    param.requires_grad = True
            self.vision_model.fc = nn.Linear(self.vision_model.fc.in_features, 128)

        elif self.ARCHITECTURE == "ViT":
            self.vision_model = Transformer_VisionEncoder(pretrained=PRETRAINED)
            self.fc_vit = nn.Linear(self.vision_model.num_features, 128)

        elif self.ARCHITECTURE == "CustomCNN":
            self.vision_model = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(128 * 56 * 56, 512),
                nn.ReLU(),
                nn.Linear(512, 128)
            )

        self.question_embedding_layer = nn.Linear(768, 128)
        self.answer_embedding_layer = nn.Linear(768, 128)
        self.fc = nn.Linear(128 + 128 + 128, num_classes)

    def forward(self, img, question_embedding, answer_embedding):
        img_features = self.vision_model(img)
        if self.ARCHITECTURE == "ViT":
            img_features = self.fc_vit(img_features)
        question_features = self.question_embedding_layer(question_embedding)
        answer_features = self.answer_embedding_layer(answer_embedding)
        combined_features = torch.cat((img_features, question_features, answer_features), dim=1)
        output = self.fc(combined_features)
        return output

def train_model(model, train_loader, criterion, optimiser, num_epochs=10):
    print(f'TRAINING model')
    model.train()
    epoch_loss = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        total_batches = len(train_loader)
        start_time = time.time()

        for batch_idx, (images, question_embeddings, answer_embeddings, labels) in enumerate(train_loader):
            images = images.to(device)
            question_embeddings = question_embeddings.to(device)
            answer_embeddings = answer_embeddings.to(device)
            labels = labels.to(device)

            outputs = model(images, question_embeddings, answer_embeddings)
            loss = criterion(outputs, labels)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running_loss += loss.item()

        epoch_loss.append(running_loss / total_batches)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / total_batches:.4f}')
    
    return epoch_loss

def evaluate_model(model, test_loader, device):
    print('Evaluating model...')
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, question_embeddings, answer_embeddings, labels in test_loader:
            images = images.to(device)
            question_embeddings = question_embeddings.to(device)
            answer_embeddings = answer_embeddings.to(device)
            labels = labels.to(device)

            outputs = model(images, question_embeddings, answer_embeddings)
            predicted_class = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted_class.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    balanced_accuracy = balanced_accuracy_score(all_labels, all_predictions)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")

    cm = confusion_matrix(all_labels, all_predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Match", "Match"], yticklabels=["No Match", "Match"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    return accuracy, balanced_accuracy, cm


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    IMAGES_PATH = "./ITM_Classifier-baselines/visual7w-images"
    train_data_file = "./ITM_Classifier-baselines/visual7w-text/v7w.TrainImages.itm.txt"
    test_data_file = "./ITM_Classifier-baselines/visual7w-text/v7w.TestImages.itm.txt"
    sentence_embeddings_file = "./ITM_Classifier-baselines/v7w.sentence_embeddings-gtr-t5-large.pkl"

    sentence_embeddings = load_sentence_embeddings(sentence_embeddings_file)

    train_dataset = ITM_Dataset(IMAGES_PATH, train_data_file, sentence_embeddings, data_split="train", train_ratio=0.2)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    test_dataset = ITM_Dataset(IMAGES_PATH, test_data_file, sentence_embeddings, data_split="test")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    models_to_train = ["CNN", "ViT", "CustomCNN"]
    accuracies = {}

    for model_name in models_to_train:
        model = ITM_Model(num_classes=2, ARCHITECTURE=model_name, PRETRAINED=True).to(device)
        criterion = nn.CrossEntropyLoss()
        optimiser = optim.Adam(model.parameters(), lr=0.0001)

        print(f"\nTraining {model_name} model")
        train_loss = train_model(model, train_loader, criterion, optimiser, num_epochs=10)

        accuracy, balanced_accuracy, cm = evaluate_model(model, test_loader)
        accuracies[model_name] = accuracy

    # Plot Accuracy Comparison
    plt.bar(accuracies.keys(), accuracies.values())
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison of Models')
    plt.show()
