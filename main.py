from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the path to the model
MODEL_PATH = os.getenv('MODEL_PATH')

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.conv4 = torch.nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(512)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(512 * 14 * 14, 512)
        self.fc2 = torch.nn.Linear(512, 32)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.dropout(torch.nn.functional.max_pool2d(self.relu(self.bn1(self.conv1(x))), 2))
        x = self.dropout(torch.nn.functional.max_pool2d(self.relu(self.bn2(self.conv2(x))), 2))
        x = self.dropout(torch.nn.functional.max_pool2d(self.relu(self.bn3(self.conv3(x))), 2))
        x = self.dropout(torch.nn.functional.max_pool2d(self.relu(self.bn4(self.conv4(x))), 2))
        x = torch.flatten(x, 1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Load the trained model
net = Net().to(device)
if os.path.exists(MODEL_PATH):
    try:
        net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        net.eval()  # Set the model to evaluation mode
        print("Model loaded successfully.")
    except RuntimeError as e:
        print(f"Error loading model: {e}")

# Define class names
class_names = [
    "Banana Good", "Banana Rotten", "Banana Mild",
    "Cucumber Good", "Cucumber Mild", "Cucumber Rotten",
    "Grape Good", "Grape Mild", "Grape Rotten",
    "Kaki Good", "Kaki Mild", "Kaki Rotten",
    "Papaya Good", "Papaya Mild", "Papaya Rotten",
    "Peach Good", "Peach Mild", "Peach Rotten",
    "Pear Good", "Pear Mild", "Pear Rotten",
    "Pepper Good", "Pepper Mild", "Pepper Rotten",
    "Strawberry Mild", "Strawberry Rotten",
    "Tomato Good", "Tomato Mild", "Tomato Rotten",
    "Watermelon Good", "Watermelon Mild", "Watermelon Rotten"
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with the actual frontend URL for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define image transformations
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.409], [0.229, 0.224, 0.225])
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Load the image
        image = Image.open(file.file).convert("RGB")
        image = data_transform(image).unsqueeze(0).to(device)

        # Make predictions
        with torch.no_grad():
            output = net(image)
            _, predicted = torch.max(output, 1)

        predicted_class = class_names[predicted.item()]
        return JSONResponse(content={"class": predicted_class})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

# Health check endpoint
@app.get("/healthcheck")
def healthcheck():
    return {"status": "healthy"}

# Classes endpoint
@app.get("/classes")
def get_classes():
    return {"classes": class_names}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
