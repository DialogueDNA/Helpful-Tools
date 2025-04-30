from transformers import pipeline

class EmotionTextModel:
    def __init__(self):
        # Load a pre-trained emotion classification model from HuggingFace
        self.model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=3)

    def analyze(self, text: str):
        """
        Analyzes a given text and returns the top 3 predicted emotions with confidence scores.
        :param text: The input sentence to analyze
        :return: List of emotion predictions
        """
        return self.model(text)
