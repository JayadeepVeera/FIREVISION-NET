import base64
import cv2
import numpy as np
import torch
import torch.nn as nn


class FireVisionNet(nn.Module):
    def __init__(self, num_classes=1, threshold=0.5, input_size=224):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )
        self.threshold = threshold
        self.input_size = input_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.eval()

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return self.head(x)

    def preprocess_frame(self, frame):
        if frame is None:
            raise ValueError("Input frame is None")
        if not isinstance(frame, np.ndarray):
            raise TypeError("Frame must be a numpy.ndarray")
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (self.input_size, self.input_size))
        tensor = torch.from_numpy(frame_rgb).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def infer_score(self, frame):
        x = self.preprocess_frame(frame)
        with torch.no_grad():
            logits = self.forward(x)
            score = torch.sigmoid(logits).item()
        return score

    def predict_status(self, frame):
        score = self.infer_score(frame)
        status = "REAL FIRE" if score >= self.threshold else "NO FIRE"
        return status, score

    def process(self, frame):
        status, score = self.predict_status(frame)
        out = frame.copy()

        label = f"{status} ({score:.2f})"
        color = (0, 0, 255) if status == "REAL FIRE" else (0, 255, 0)

        cv2.rectangle(out, (10, 10), (420, 70), (0, 0, 0), -1)
        cv2.putText(
            out,
            label,
            (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3,
            cv2.LINE_AA,
        )
        return out, status, score

    def encode_frame_to_base64(self, frame, quality=85):
        if frame is None:
            raise ValueError("Frame is None")
        if not isinstance(frame, np.ndarray):
            raise TypeError("Frame must be a numpy.ndarray")

        ok, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
        )
        if not ok:
            raise ValueError("Failed to encode frame")

        return base64.b64encode(buffer).decode("utf-8")