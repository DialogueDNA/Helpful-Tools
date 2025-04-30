from speechbrain.pretrained import SpeakerRecognition

class SpeakerRecognizer:
    def __init__(self):
        # Load the speaker verification model from SpeechBrain
        self.verifier = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="tools/models_cache/spkrec"
        )

    def compare(self, file1: str, file2: str) -> dict:
        """"
        Compares two audio files and determines if they are from the same speaker.
        :param file1: Path to the first audio file
        :param file2: Path to the second audio file
        :return: Dictionary with similarity score and boolean prediction
        """
        score, prediction = self.verifier.verify_files(file1, file2)
        return {"score": float(score), "same_speaker": bool(prediction)}
