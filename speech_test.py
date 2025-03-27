import azure.cognitiveservices.speech as speechsdk

# הכניסי כאן את ה-Key ואת ה-Region שלך מ-Azure
speech_key = "4Wpm839JNUGK79VhiQEOn2Tk7VGgirgdO1B7w2wnMATIbNxzJiw9JQQJ99BCAC5RqLJXJ3w3AAAYACOGVw8L"
service_region = "westeurope"  # או לפי מה שבחרת כשיצרת את השירות

# הגדרות השירות
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
audio_config = speechsdk.AudioConfig(filename="conversation.wav")

speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
result = speech_recognizer.recognize_once()

# תוצאה
if result.reason == speechsdk.ResultReason.RecognizedSpeech:
    print("🎤 Recognized text:", result.text)
else:
    print("❌ Speech not recognized.")
