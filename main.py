from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import pickle
import tensorflow as tf
import shutil
import os

tflite = tf.lite
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()

# Globals for model and preprocessing
interpreter = None
tokenizer = None
label_encoder = None
input_details = None
output_details = None

# Function to load the model
def load_model():
    global interpreter, tokenizer, label_encoder, input_details, output_details
    # Load tokenizer
    with open("model/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    # Load label encoder
    with open("model/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    # Load TFLite model
    interpreter = tflite.Interpreter(model_path="model/behavior_classifier.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

# Try to load the model at startup (optional: protect if missing)
try:
    load_model()
except Exception as e:
    print(f"Model not loaded at startup: {e}")

# Define FastAPI input model
class TextInput(BaseModel):
    text: str

# Preprocess input
def preprocess(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=50)  # Adjust maxlen to match your model
    return np.array(padded, dtype=np.float32)

# Decode output
def decode_prediction(output):
    predicted_class = np.argmax(output)
    return label_encoder.inverse_transform([predicted_class])[0]

# Route to classify text
@app.post("/classify/")
async def classify_text(data: TextInput):
    try:
        input_data = preprocess(data.text)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = decode_prediction(output_data)
        return {"result": prediction}
    except Exception as e:
        return {"error": str(e)}

# Route to upload a new model
@app.post("/upload_model/")
async def upload_model(
    tflite_file: UploadFile = File(...),
    tokenizer_file: UploadFile = File(...),
    label_encoder_file: UploadFile = File(...)
):
    try:
        os.makedirs("model", exist_ok=True)

        # Save uploaded files
        with open("model/behavior_classifier.tflite", "wb") as buffer:
            shutil.copyfileobj(tflite_file.file, buffer)

        with open("model/tokenizer.pkl", "wb") as buffer:
            shutil.copyfileobj(tokenizer_file.file, buffer)

        with open("model/label_encoder.pkl", "wb") as buffer:
            shutil.copyfileobj(label_encoder_file.file, buffer)

        # Reload the models
        load_model()

        return {"message": "Model, tokenizer, and label encoder uploaded and loaded successfully"}
    except Exception as e:
        return {"error": str(e)}

# ✅ Simple homepage so Render doesn’t show 404
@app.get("/")
async def root():
    return {"message": "FastAPI Text Classifier is running!"}

# ✅ Download the TFLite model file
@app.get("/download_model/")
async def download_model():
    tflite_path = "model/behavior_classifier.tflite"
    if os.path.exists(tflite_path):
        return FileResponse(path=tflite_path, filename="behavior_classifier.tflite", media_type='application/octet-stream')
    else:
        raise HTTPException(status_code=404, detail="TFLite model not found")
