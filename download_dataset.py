# download_dataset.py
from roboflow import Roboflow
import os

# FREE - signup at roboflow.com (30 seconds)
rf = Roboflow(api_key="CeOYwoJXPx5zoVJ1MLkM")  
project = rf.workspace("fire-detection-v2").project("fire-smoke-final")
dataset = project.version(1).download("yolov8")

print("✅ Dataset downloaded! Check data/ folder")