import cv2
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import Annotated
from PIL import Image
from io import BytesIO
import numpy as np
import json
from pydantic import BaseModel
import tensorflow as tf

# Cargar el modelo previamente entrenado y guardado
model = tf.keras.models.load_model('model_trained')

# Crear una instancia de la aplicación FastAPI
app = FastAPI()

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocesar la imagen: normalización y cambio de forma.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(image.shape)
    image = cv2.resize(image, (28, 28))  # Redimensionar a 28x28 píxeles
    print(image.shape)
    image = image / 255.0  # Normalizar los valores de los píxeles
    image = image.reshape(1, 28, 28)  # Ajustar la forma del array
    return image

@app.post("/predict")
async def predict(input_image: Annotated[UploadFile, File()]):
    #try:
    # Leer la imagen subida
    image_data = await input_image.read()
    image = np.array(Image.open(BytesIO(image_data)))

    # Preprocesar la imagen
    image_p = preprocess_image(image)

    # Realizar la predicción
    prediction = model.predict(image_p).argmax(axis=1)

    return {"prediction": int(prediction[0])}
    """except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar la imagen: {str(e)}")
    """