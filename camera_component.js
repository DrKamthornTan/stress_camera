import streamlit as st
from streamlit.components.v1 import html

camera_component = """
<video id="videoElement" width="100%" height="auto" autoplay playsinline></video>
<select id="videoSource"></select>

<script>
    const videoElement = document.querySelector('#videoElement');
    const videoSelect = document.querySelector('#videoSource');

    videoSelect.onchange = getStream;

    getStream().then(getDevices).then(gotDevices);

    function getDevices() {
      return navigator.mediaDevices.enumerateDevices();
    }

    function gotDevices(deviceInfos) {
      window.deviceInfos = deviceInfos;
      console.log('Available input devices:', deviceInfos);
      for (const deviceInfo of deviceInfos) {
        if (deviceInfo.kind === 'videoinput') {
          const option = document.createElement('option');
          option.value = deviceInfo.deviceId;
          option.text = deviceInfo.label || `Camera ${videoSelect.length + 1}`;
          videoSelect.appendChild(option);
        }
      }
    }

    function getStream() {
      if (window.stream) {
        window.stream.getTracks().forEach(track => {
          track.stop();
        });
      }
      const videoSource = videoSelect.value;
      const constraints = {
        video: { deviceId: videoSource ? { exact: videoSource } : undefined }
      };
      return navigator.mediaDevices.getUserMedia(constraints)
        .then(gotStream)
        .catch(handleError);
    }

    function gotStream(stream) {
      window.stream = stream;
      videoElement.srcObject = stream;
    }

    function handleError(error) {
      console.error('Error: ', error);
    }
</script>
"""

def camera_component():
    html(camera_component)