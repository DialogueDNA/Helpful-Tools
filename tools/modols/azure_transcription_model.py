import azure.cognitiveservices.speech as speechsdk

class AzureTranscriber:
    def __init__(self, api_key: str, region: str):
        # Initialize the Azure Speech service with your key and region
        self.speech_config = speechsdk.SpeechConfig(subscription=api_key, region=region)

    def transcribe_file(self, filepath: str) -> str:
        """
        Transcribes a local audio file using Azure's Speech-to-Text API.
        :param filepath: Path to the audio file (e.g. .wav)
        :return: Transcribed text or raises error if failed
        """
        audio_config = speechsdk.AudioConfig(filename=filepath)
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=audio_config
        )
        result = speech_recognizer.recognize_once()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text
        else:
            raise RuntimeError("Speech not recognized or failed.")
