import tempfile
import sounddevice as sd
import numpy as np
import queue
import time
import wave
import os

from tools.modols.speaker_recognition_model import SpeakerRecognizer


class SmartRecorder:
    """
    SmartRecorder automatically records voice input and stops recording
    either after a period of silence or when a new speaker is detected.
    """

    def __init__(self, sample_rate=16000, silence_threshold=500, silence_duration=0.5, output_dir="outputs/audio_clips"):
        """
        Initialize the SmartRecorder.

        Args:
            sample_rate (int): Recording sample rate in Hz.
            silence_threshold (int): RMS threshold to consider audio as silence.
            silence_duration (float): Duration (in seconds) of silence to trigger stop.
            output_dir (str): Directory to save recorded audio.
        """
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.buffer = queue.Queue()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.chunk_duration = 1  # seconds for speaker detection check
        self.speaker_recognizer = SpeakerRecognizer()
        self._speaker_check_buffer = b""  # Accumulate audio for speaker comparison

    def _is_silent(self, audio_chunk):
        """
        Check if audio is considered silent.

        Args:
            audio_chunk (np.array): Raw audio data as int16 array.

        Returns:
            bool: True if audio is below the silence threshold.
        """
        rms = np.sqrt(np.mean(np.square(audio_chunk)))
        return rms < self.silence_threshold

    def _write_wav(self, frames, filename):
        """
        Write audio frames to a .wav file.

        Args:
            frames (list): List of audio chunks (bytes).
            filename (str): Name of the output file.

        Returns:
            str: Full path to the saved file.
        """
        filepath = os.path.join(self.output_dir, filename)
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
        return filepath

    def _is_different_speaker(self, audio_bytes):
        """
        Determine if the speaker in the given audio differs from known speakers.

        Args:
            audio_bytes (bytes): Audio data in raw bytes.

        Returns:
            bool: True if a new speaker is detected.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            with wave.open(tmp_wav.name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_bytes)

            embedding = self.speaker_recognizer.extract_embedding(tmp_wav.name)
            speaker_id, is_known = self.speaker_recognizer.is_same_speaker(embedding)
            print(f"🧠 Speaker detected: {speaker_id} ({'known' if is_known else 'new'})")
            return not is_known

    def record(self, max_record_time=60):
        """
        Start real-time recording. Stops after silence or speaker change.

        Args:
            max_record_time (int): Maximum duration in seconds.

        Returns:
            str: Path to saved .wav file.
        """
        print("🎙 Start speaking... (recording will stop after silence or speaker change)")
        recording = []
        start_time = time.time()
        silence_start = None
        chunk_for_checking = b""
        bytes_per_second = self.sample_rate * 2  # 2 bytes per sample (16-bit mono)

        def callback(indata, frames, time_info, status):
            self.buffer.put(bytes(indata))

        with sd.RawInputStream(samplerate=self.sample_rate, blocksize=8000, dtype='int16',
                               channels=1, callback=callback):
            while True:
                try:
                    chunk = self.buffer.get()
                    recording.append(chunk)
                    chunk_for_checking += chunk

                    audio_array = np.frombuffer(chunk, dtype=np.int16)

                    # Silence detection
                    if self._is_silent(audio_array):
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > self.silence_duration:
                            print("🛑 Silence detected. Stopping recording.")
                            break
                    else:
                        silence_start = None

                    # Speaker change detection every 1 second of audio
                    if len(chunk_for_checking) >= bytes_per_second:
                        if self._is_different_speaker(chunk_for_checking):
                            print("🔄 Speaker change detected. Stopping recording.")
                            break
                        chunk_for_checking = b""  # Reset buffer for next check

                    # Max duration safety limit
                    if time.time() - start_time > max_record_time:
                        print("⏱ Max recording time reached.")
                        break

                except KeyboardInterrupt:
                    print("🛑 Interrupted by user.")
                    break

        filename = f"smart_record_{int(time.time())}.wav"
        return self._write_wav(recording, filename)
