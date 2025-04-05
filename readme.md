# Text and Audio Streaming Chat Bot

This project is a system that performs real-time audio streaming and transcription via WebSocket connection. In the project, microphone data is received from the client side and sent to the server; the server processes the incoming audio data, converts speech to text, and then produces a response using a chatbot model (T5).

## Features

- **Real Time Audio Streaming:**  
  Audio data received using the microphone on the client side is sent to the server via WebSocket..
- **Audio Transcription:**  
  The server converts the incoming voice data into text using the Faster Whisper model.
- **Chat Bot Integration:**  
  The transcribed text is sent to the T5 model to generate the answer for special tasks such as code generation..
- **Voice Activity Detection (VAD):**  
  With WebRTC VAD, sections containing audio are detected and unnecessary data processing is prevented..
- **Dynamic WS Connection Management:**  
  On the client side, the first click on the microphone opens the connection, the second click sends the `"end_of_string"` message to process the remaining audio data and closes the WS connection..
- **Fine-Tuned Model Usage:**
  A trained (fine-tuned) data set is used.

## Setup

### Requirements

- Python 3.8+
- Node.js and a modern web browser (to run client HTML)
- The following Python packages:
  - `fastapi`
  - `uvicorn`
  - `faster-whisper`
  - `webrtcvad`
  - `pyttsx3`
  - `transformers`
  - `numpy`
- Other helper libraries: `asyncio`, `queue`, `threading`, `uuid`, `json`

### Steps

1. **Creating a Python Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/MacOS
   venv\Scripts\activate     # Windows

   ```

2. **Installing Necessary Packages:**

   ```bash
   pip install fastapi uvicorn faster-whisper webrtcvad pyttsx3 transformers numpy

   ```

3. **Download Model Files:**

   ```bash
   Make sure that the relevant files (e.g. ./ft_model folder) are available for the T5 model and tokenizer used in the project.

   ```

4. **Running the Server Code:**

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000

   ```

5. **Running Client Side:**

   ```bash
   Open the HTML file in a web browser.
   ```

![image alt](https://github.com/kadirdemirkaya/Trained_Code_ChatBot/blob/678cc68135e0fd692c3dee5af93a08f19fe9f807/Images/cb1.png)
![image alt](https://github.com/kadirdemirkaya/Trained_Code_ChatBot/blob/678cc68135e0fd692c3dee5af93a08f19fe9f807/Images/cb2.png)
