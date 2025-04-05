from contextlib import asynccontextmanager
import time
import asyncio
import queue
import threading
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
from starlette.websockets import WebSocketState
import webrtcvad
from collections import deque
from threading import Lock
import pyttsx3
import uuid
from textblob import TextBlob
from transformers import pipeline
import pandas as pd
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
import re
import json

MODEL_SIZE = "tiny"
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000) * 2
t5_model = T5ForConditionalGeneration.from_pretrained("./ft_model")
t5_tokenizer = T5Tokenizer.from_pretrained("./ft_model")
device = t5_model.device

# Global components
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
vad = webrtcvad.Vad(2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    for client_id in list(clients.keys()):
        try:
            await clients[client_id].websocket.close()
        except Exception as e:
            print(f"Connection close error: {e}")
        del clients[client_id]


class ClientState:
    def __init__(self, loop, client_id, websocket):
        self.client_id = client_id
        self.websocket = websocket
        self.buffer = bytearray()
        self.speech_buffer = bytearray()
        self.in_speech = False
        self.last_activity = time.monotonic()
        self.transcription = ""
        self.lock = Lock()
        self.pending_tasks = 0
        self.all_tasks_done = asyncio.Event()
        self.loop = loop
        self.loop.call_soon_threadsafe(self.all_tasks_done.set)


app = FastAPI(lifespan=lifespan)
clients = {}  # Key: client_id, Value: ClientState
audio_queue = queue.Queue()
client_cleanup_interval = 5


def worker():
    while True:
        item = audio_queue.get()
        if item is None:
            break
        audio_data, client_id = item

        try:
            client_state = clients.get(client_id)
            if client_state is None:
                continue

            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )
            segments, _ = model.transcribe(audio_np, beam_size=2, language="en")
            transcription = " ".join(segment.text for segment in segments).strip()

            if transcription:
                with client_state.lock:
                    client_state.transcription += " " + transcription

                # Check WebSocket state before sending
                if client_state.websocket.client_state == WebSocketState.CONNECTED:
                    asyncio.run_coroutine_threadsafe(
                        client_state.websocket.send_json({"partial": transcription}),
                        client_state.loop,
                    )
        except Exception as e:
            print(f"Processing error: {e}")
        finally:
            if client_id in clients:
                with clients[client_id].lock:
                    clients[client_id].pending_tasks -= 1
                    if clients[client_id].pending_tasks == 0:
                        clients[client_id].loop.call_soon_threadsafe(
                            clients[client_id].all_tasks_done.set
                        )
            audio_queue.task_done()


def process_audio(client_id, data):
    client_state = clients.get(client_id)
    if client_state is None:
        return

    with client_state.lock:
        client_state.buffer.extend(data)
        client_state.last_activity = time.monotonic()

        while len(client_state.buffer) >= FRAME_SIZE:
            frame = client_state.buffer[:FRAME_SIZE]
            client_state.buffer = client_state.buffer[FRAME_SIZE:]

            try:
                is_speech = vad.is_speech(frame, SAMPLE_RATE)
            except:
                is_speech = False

            if is_speech:
                client_state.speech_buffer.extend(frame)
                client_state.in_speech = True
            elif client_state.in_speech:
                audio_queue.put((bytes(client_state.speech_buffer), client_id))
                client_state.pending_tasks += 1
                if client_state.pending_tasks == 1:
                    client_state.loop.call_soon_threadsafe(
                        client_state.all_tasks_done.clear
                    )
                client_state.speech_buffer.clear()
                client_state.in_speech = False


def cleanup_inactive_clients():
    while True:
        now = time.monotonic()
        to_remove = []

        for client_id, state in list(clients.items()):
            with state.lock:
                if (now - state.last_activity) > client_cleanup_interval:
                    if state.speech_buffer:
                        audio_queue.put((bytes(state.speech_buffer), client_id))
                        state.pending_tasks += 1
                        if state.pending_tasks == 1:
                            state.loop.call_soon_threadsafe(state.all_tasks_done.clear)
                    to_remove.append(client_id)

        for client_id in to_remove:
            if client_id in clients:
                del clients[client_id]

        time.sleep(client_cleanup_interval)


@app.websocket("/text")
async def text_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "text_message":
                response = chatbot(data["content"])
                await websocket.send_json({"success": True, "message": response})
                break
    except Exception as e:
        print(f"Text socket error: {e}")
    finally:
        await websocket.close()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = str(uuid.uuid4())
    loop = asyncio.get_running_loop()
    clients[client_id] = ClientState(loop, client_id, websocket)

    try:
        while True:
            data = await websocket.receive()

            if data["type"] == "websocket.receive":
                if "bytes" in data:
                    process_audio(client_id, data["bytes"])
                elif "text" in data:
                    message = json.loads(data["text"])
                    if message.get("action") == "END_OF_STREAM":
                        break

        with clients[client_id].lock:
            if clients[client_id].speech_buffer:
                audio_queue.put((bytes(clients[client_id].speech_buffer), client_id))

        await clients[client_id].all_tasks_done.wait()

        print(f"Original Text: {clients[client_id].transcription}")

        response = chatbot(clients[client_id].transcription)

        await websocket.send_json(
            {"success": True, "message": response, "is_final": True}
        )

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close()
        except Exception as e:
            print(f"Close error: {e}")
        finally:
            if client_id in clients:
                del clients[client_id]


def chatbot(query):
    query = query.lower()
    input_text = "generate code: " + query
    input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    outputs = t5_model.generate(
        input_ids, max_length=512, num_beams=4, early_stopping=True
    )
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)


for _ in range(4):
    threading.Thread(target=worker, daemon=True).start()

threading.Thread(target=cleanup_inactive_clients, daemon=True).start()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
