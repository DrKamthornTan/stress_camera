import cv2
import streamlit as st
from PIL import Image
import numpy as np
import tempfile
from deepface import DeepFace
import tensorflow as tf

# JavaScript code
javascript_code = """
<script>
if (!navigator.mediaDevices?.enumerateDevices) {
  console.log("enumerateDevices() not supported.");
} else {
  // List cameras and microphones.
  navigator.mediaDevices
    .enumerateDevices()
    .then((devices) => {
      let audioSource = null;
      let videoSource = null;

      devices.forEach((device) => {
        if (device.kind === "audioinput") {
          audioSource = device.deviceId;
        } else if (device.kind === "videoinput") {
          videoSource = device.deviceId;
        }
      });
      sourceSelected(audioSource, videoSource);
    })
    .catch((err) => {
      console.error(`${err.name}: ${err.message}`);
    });
}

async function sourceSelected(audioSource, videoSource) {
  const constraints = {
    audio: { deviceId: audioSource },
    video: { deviceId: videoSource },
  };
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
}
</script>
"""

# Display the JavaScript code as a Streamlit component
st.components.v1.html(javascript_code)

# Load the model
model = tf.keras.models.load_model("keras_model.h5", compile=False)
class_names = ["stressless", "stressful"]

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

st.title("DHV AI Startup Stress Detection Demo")
st.subheader("Stress Assessment from Facial Expression using Camera or Image")
st.write("Please stay about 2 feet away from the camera, look at the camera, stay still, and press the button.")

# Define a function to get the prediction from an uploaded image
def predict_image(image_file):
    # Load the image
    image = Image.open(image_file).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, method=Image.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predict using the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Display the image and prediction
    caption = f"<h1 style='font-size: 24px; color: {'blue' if class_name == 'stressless' else 'red'}'>Class: {class_name} ({class_name.replace('stress', '')})\nConfidence Score: {confidence_score:.2f}</h1>"
    st.image(image, caption=None)
    st.markdown(caption, unsafe_allow_html=True)
    if class_name == "stressless":
        st.markdown("<h1 style='font-size: 24px;'>You've done good so far.</h1>", unsafe_allow_html=True)
    else:
        st.markdown("<h1 style='font-size: 24px;'>Take a deep breath! Sit back and relax. Try exercise, vacation, entertainment, or consult your physician.</h1>", unsafe_allow_html=True)

    # Perform stress detection using DeepFace
    results = DeepFace.analyze(image_file, actions=['age'])
    # Iterate over the list and find the dictionary with 'age' information
    predicted_age = None

    if results:
    # Assume the first element in the list has the 'age' information
        first_result = results[0]

        # Check if the first_result is a dictionary and contains the 'age' key
        if isinstance(first_result, dict) and 'age' in first_result:
            predicted_age = first_result['age']
import os
import tempfile

def predict_camera():
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(rgb_frame, (224, 224))
    normalized_frame = (resized_frame.astype(np.float32) / 127.5) - 1
    data[0] = normalized_frame

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    st.image(rgb_frame, channels="RGB", caption=f"{class_name} ({class_name.replace('stress', '')})\nConfidence Score: {confidence_score:.2f}")
    
    caption = f"<h1 style='color: {'blue' if class_name == 'stressless' else 'red'}'>Class: {class_name} ({class_name.replace('stress', '')})</h1>\n<h1>Confidence Score: {confidence_score:.2f}</h1>"
    st.markdown(caption, unsafe_allow_html=True)

    if class_name == "stressless":
        st.markdown("<h1 style='font-size: 24px;'>You've done good so far.</h1>", unsafe_allow_html=True)
    else:
        st.markdown("<h1 style='font-size: 24px;'>Take a deep breath! Sit back and relax. Try exercise, vacation, entertainment, or consult your physician.</h1>", unsafe_allow_html=True)

    pil_image = Image.fromarray(rgb_frame)

    _, temp_path = tempfile.mkstemp(suffix=".jpg", dir=tempfile.gettempdir())
    pil_image.save(temp_path)

    results = DeepFace.analyze(img_path=temp_path, actions=['age'])
    predicted_age = results[0]['age']
    st.markdown(f"<h1>Predicted Age: {predicted_age}</h1>", unsafe_allow_html=True)

    cap.release()  # Release the camera capture object after use

if st.button("Start Camera"):
    predict_camera()
