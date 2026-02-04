import gradio as gr
import numpy as np
import pickle
import cv2

# Load saved files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
pca = pickle.load(open("pca.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

IMG_SIZE = 32

def predict(image):
    image = np.array(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype("float32") / 255.0
    image = image.flatten().reshape(1, -1)

    image_scaled = scaler.transform(image)
    image_pca = pca.transform(image_scaled)

    pred = model.predict(image_pca)[0]
    return classes[pred]

app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Satellite Image Classification",
    description="Cloudy | Desert | Green Area | Water"
)

app.launch()
