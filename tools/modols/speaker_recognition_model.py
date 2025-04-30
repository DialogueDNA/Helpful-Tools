import torchaudio
import numpy as np
from speechbrain.pretrained import SpeakerRecognition

class SpeakerRecognizer:
    def __init__(self):
        self.model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="tools/models_cache/spkrec"
        )
        self.voiceprints = {}  # Mapping: speaker_label -> embedding tensor

    def compare_files(self, file1: str, file2: str) -> dict:
        """
        Compare two files to determine speaker similarity.
        """
        score, prediction = self.model.verify_files(file1, file2)
        return {"score": float(score), "same_speaker": bool(prediction)}

    def extract_embedding(self, file_path: str):
        """
        Extract speaker embedding from an audio file.
        """
        signal, fs = torchaudio.load(file_path)
        embedding = self.model.encode_batch(signal)
        return embedding.squeeze().detach().cpu().numpy()

    def is_same_speaker(self, new_embedding: np.ndarray, threshold=0.75) -> (str, bool):
        """
        Check if the given embedding matches any known speaker.
        If a match is found above threshold → return (speaker_id, True).
        If not → return (new_speaker_id, False).
        """
        for label, known_embedding in self.voiceprints.items():
            sim = self._cosine_similarity(known_embedding, new_embedding)
            if sim > threshold:
                return label, True

        # New speaker
        new_label = f"Speaker_{len(self.voiceprints) + 1}"
        self.voiceprints[new_label] = new_embedding
        return new_label, False

    def _cosine_similarity(self, vec1, vec2):
        """
        Compute cosine similarity between two vectors.
        """
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot / (norm1 * norm2 + 1e-8)
