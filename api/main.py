from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
# from model import class_names

VERSION = 1

# Load the model
MODEL = tf.keras.models.load_model(f"model/{VERSION}.keras")
CLASS_NAMES = ['HandGun', 'Knife', 'Rifle', 'ShotGun', 'SMG']

app = FastAPI()

# Add CORS middleware to allow requests from your React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow your React app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
async def ping():
    return {"message": "hello, I am alive"}

def read_file_as_image(data) -> np.ndarray:
    """Convert image file data to a NumPy array."""
    image = Image.open(BytesIO(data))
    image = image.convert("RGB")  # Ensure the image is in RGB format
    return np.array(image)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict the class of an uploaded image."""
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)  # Add batch dimension

    # Make prediction
    predictions = MODEL.predict(img_batch)

    # Get class with highest probability
    predict_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        'class': predict_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
