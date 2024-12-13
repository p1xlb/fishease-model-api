import io
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from tensorflow.keras.applications.resnet50 import preprocess_input
import requests
import tempfile
import os

# Load the pre-trained model
model = tf.keras.models.load_model('fishease_model(1).h5', compile=False)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Define the class names (from the original training)
CLASS_NAMES = [
    'Bacterial diseases - Aeromoniasis',
    'Bacterial gill disease',
    'Bacterial Red disease',
    'Fungal diseases Saprolegniasis',
    'Healthy Fish',
    'Not Fish',
    'Parasitic diseases',
    'Viral diseases White tail disease'
]

# Image preprocessing parameters (from the notebook)
IMAGE_SIZE = 224
CHANNELS = 3

class PredictionResponse(BaseModel):
    """Response model for prediction results"""
    class_name: str
    confidence: float

app = FastAPI(
    title="Fish Disease Classification API",
    description="API for predicting fish diseases using a ResNet50-based model",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess the input image for model prediction
    
    Args:
        image (PIL.Image): Input image
    
    Returns:
        np.ndarray: Preprocessed image array
    """
    # Resize image
    img = image.convert('RGB')
    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    
    # Convert to numpy array
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # Ensure the image has 3 channels
    if img_array.shape[-1] != CHANNELS:
        raise HTTPException(status_code=400, detail="Image must have 3 color channels")
    
    # Expand dimensions to create a batch
    img_array = tf.expand_dims(img_array, 0)
    
    return img_array

@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """
    Endpoint for predicting fish disease from an uploaded image
    
    Args:
        file (UploadFile): Uploaded image file
    
    Returns:
        dict: Prediction results with class name and confidence
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read the image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        
        # Get the predicted class
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_index]
        confidence = round(100 * np.max(predictions[0]), 2)
        
        return {
            "class_name": predicted_class,
            "confidence": confidence
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/classes")
def get_classes() -> List[str]:
    """
    Endpoint to retrieve available disease classes
    
    Returns:
        list: List of fish disease classes
    """
    return CLASS_NAMES

# Optional: Health check endpoint
@app.get("/health")
def health_check():
    """
    Simple health check endpoint
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 3500)))