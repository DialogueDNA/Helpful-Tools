# Realtime transcription using Vosk (offline, lightweight)
# Install first: pip install vosk sounddevice

import sounddevice as sd
import queue
import sys
import json
from vosk import Model, KaldiRecognizer

# ====== Config ======
samplerate = 16000  # Microphone rate
model_path = r"C:\Users\Yarden Daniel\PycharmProjects\Helpful-Tools\model\vosk-model-small-en-us-0.15"  # Path to vosk model folder

# ====== Audio Queue ======
audio_queue = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(bytes(indata))

# ====== Load Model ======
print("Loading model...")
model = Model(model_path)
recognizer = KaldiRecognizer(model, samplerate)
recognizer.SetWords(True)

# ====== Open Mic ======
with sd.RawInputStream(samplerate=samplerate, blocksize=8000, dtype='int16',
                       channels=1, callback=callback):
    try:
        current_partial = ""
        while True:
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                final_text = result.get("text", "")
                print(f"\r📝 {final_text:<80}", end="")
                current_partial = ""
            else:
                partial = json.loads(recognizer.PartialResult())
                new_partial = partial.get("partial", "")
                if new_partial != current_partial:
                    current_partial = new_partial
                    print(f"\r⌛ {current_partial:<80}", end="")
    except KeyboardInterrupt:
        print("\n[Finished]")