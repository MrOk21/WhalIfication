import numpy as np
import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk, UnidentifiedImageError
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import cv2
import os
import json

# Define U-Net model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2)

        self.encoder4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        out = self.final_conv(dec1)
        return self.sigmoid(out)

class IdentificationApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()

        self.title("Whale Identification")
        self.geometry("800x600")
        self.configure(bg="lightblue")
        self.subtitle_label = tk.Label(self, text="You want to know the name of the whale just seen? Drag its picture here!", font=("Times New Roman", 13), bg="lightblue")
        self.subtitle_label.pack(padx=1, pady=10, anchor="w")
        self.label_frame = tk.Frame(self, bg="lightblue")
        self.label_frame.pack(fill=tk.BOTH, expand=True)

        self.label = tk.Label(
            self.label_frame, 
            text="Drag the image of the whale you want to identify here!", 
            bg="#87CEFA", 
            font=("Arial", 24), 
            width=60, 
            height=20
        )
        self.label.pack(side=tk.LEFT, padx=10, pady=10, expand=True, fill=tk.BOTH)

        self.label.drop_target_register(DND_FILES)
        self.label.dnd_bind('<<Drop>>', self.drop)

        self.message_label = tk.Label(self, text="", font=("Arial", 14), bg="lightblue")
        self.message_label.pack(padx=10, pady=10)

        self.preprocess_btn = tk.Button(self, text="Preprocess Image", command=self.preprocess_image, bg="lightblue")
        self.preprocess_btn.pack(pady=10)

        self.identify_btn = tk.Button(self, text="Identify Whale", command=self.identify_image, bg="lightblue")
        self.identify_btn.pack(pady=10)

        self.image_path = None
        self.model = self.load_model()
        self.classifier = RandomForestClassifier()
        self.load_classifier()
        self.load_label_mappings()

    def load_model(self):
        model_path = os.path.join(os.path.dirname(__file__), 'Utils', 'unet_final_model.pth')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model

    def load_classifier(self):
        train_features_path = os.path.join(os.path.dirname(__file__), 'Utils', 'train_features.npy')
        train_labels_path = os.path.join(os.path.dirname(__file__), 'Utils', 'train_labels.npy')
        train_features = np.load(train_features_path)
        train_labels = np.load(train_labels_path)
        self.classifier.fit(train_features, train_labels)

    def load_label_mappings(self):
        label_mappings_path = os.path.join(os.path.dirname(__file__), 'Utils', 'train_label_to_class.json')
        with open(label_mappings_path, 'r') as f:
            self.label_to_class = json.load(f)

    def drop(self, event):
        file_path = event.data.strip('{}')
        self.image_path = file_path
        self.load_image(file_path)

    def load_image(self, file_path):
        try:
            image = Image.open(file_path)
            image.thumbnail((800, 600))
            photo = ImageTk.PhotoImage(image)

            self.label.config(image=photo)
            self.label.image = photo
            self.label.config(text="")
        except UnidentifiedImageError:
            self.label.config(text="Failed to load image: Unsupported format")
        except Exception as e:
            self.label.config(text="Failed to load image")
            print(f"Error loading image: {e}")

    def preprocess_image(self):
        if not self.image_path:
            self.message_label.config(text="No image loaded.")
            return

        transform = transforms.Compose([
            transforms.Resize((128, 128)),  
            transforms.ToTensor()
        ])
        
        image = Image.open(self.image_path).convert("RGB")
        image = transform(image).unsqueeze(0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image = image.to(device)

        with torch.no_grad():
            output = self.model(image)
            output = (output > 0.5).float().cpu().numpy()[0, 0] * 255

        mask_path = self.image_path.replace('.jpg', '_mask.png')
        cv2.imwrite(mask_path, output)
        self.crop_image(self.image_path, mask_path)
        self.display_image(mask_path.replace('.png', '_cropped.jpg'))

    def display_image(self, path):
        img = Image.open(path)
        img = img.resize((400, 400))
        self.img_tk = ImageTk.PhotoImage(img)
        self.label.config(image=self.img_tk)
        self.label.image = self.img_tk

    def crop_image(self, image_path, mask_path):
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Error loading image: {image_path}")
            return
        if mask is None:
            print(f"Error loading mask: {mask_path}")
            return

        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        masked_image = cv2.bitwise_and(image, image, mask=mask)
        x, y, w, h = cv2.boundingRect(mask)
        cropped_image = masked_image[y:y+h, x:x+w]

        if cropped_image.size == 0:
            print(f"Cropped image is empty for: {image_path}")
            return

        cropped_path = mask_path.replace('.png', '_cropped.jpg')
        cv2.imwrite(cropped_path, cropped_image)
        print(f"Cropped image saved at: {cropped_path}")

    def extract_features(self, image_path):
        model_path = os.path.join(os.path.dirname(__file__), 'Utils', 'complex_euclidean_siamese.keras')
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'contrastive_loss': self.contrastive_loss, 'compute_euclidean_distance': self.compute_euclidean_distance}
        )
        base_model = model.layers[2]

        img = Image.open(image_path).convert('L')
        img = img.resize((105, 105))
        img = np.array(img) / 255.0
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        features = base_model.predict(img)
        return features[0]

    @staticmethod
    def compute_euclidean_distance(vectors):
        x, y = vectors
        return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))

    @staticmethod
    def contrastive_loss(y_true, y_pred):
        margin = 1
        return y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0))

    def identify_image(self):
        if not self.image_path:
            self.message_label.config(text="No image loaded.")
            return

        cropped_image_path = self.image_path.replace('.jpg', '_mask_cropped.jpg')
        if not os.path.exists(cropped_image_path):
            self.message_label.config(text="Image not preprocessed.")
            return

        feature = self.extract_features(cropped_image_path)
        top_k_indices, predictions = self.identify_image_top_k(self.classifier, feature)
        
        real_label = os.path.basename(os.path.dirname(self.image_path))
        predicted_label = self.label_to_class[str(predictions[0])]

        result_text = f"Real Label: {real_label}\nPredicted Label: {predicted_label}"
        self.message_label.config(text=result_text)

        print(f"Top-{len(predictions)} Predictions: {predictions}")

    def identify_image_top_k(self, model, feature, k=5):
        probabilities = model.predict_proba([feature])[0]
        top_k_indices = np.argsort(probabilities)[-k:][::-1]
        predictions = model.classes_[top_k_indices]
        return top_k_indices, predictions

if __name__ == "__main__":
    app = IdentificationApp()
    app.mainloop()
