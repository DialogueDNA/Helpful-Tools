import assemblyai as aai

class AssemblyTranscriber:
    def __init__(self, api_key: str):
        # Set the API key and initialize the transcriber
        aai.settings.api_key = api_key
        self.transcriber = aai.Transcriber()

    def transcribe_file(self, file_path: str, speaker_labels=True) -> str:
        """
        Transcribes an audio file using AssemblyAI.
        :param file_path: Path to the audio file
        :param speaker_labels: Whether to include speaker diarization
        :return: Transcribed text as string
        """
        config = aai.TranscriptionConfig(speaker_labels=speaker_labels)
        transcript = self.transcriber.transcribe(file_path, config=config)
        return transcript.text


