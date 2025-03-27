import requests

# פרטי Azure שלך
speech_key = "4Wpm839JNUGK79VhiQEOn2Tk7VGgirgdO1B7w2wnMATIbNxzJiw9JQQJ99BCAC5RqLJXJ3w3AAAYACOGVw8LY"
region = "westeurope"
filename = "conversation.wav"

# כתובת API עם דיאריזציה מופעלת
url = f"https://{region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
params = {
    "language": "en-US",
    "diarizationEnabled": "true",  # הפעלת זיהוי דוברים
    "format": "detailed"
}
headers = {
    "Ocp-Apim-Subscription-Key": speech_key,
    "Content-Type": "audio/wav"  # או audio/mp3 לפי סוג הקובץ שלך
}

# קריאת הקובץ ושליחה ל-Azure
with open(filename, "rb") as audio_file:
    response = requests.post(url, params=params, headers=headers, data=audio_file)

# הדפסת התוצאה
if response.status_code == 200:
    result = response.json()
    print("📄 Full JSON Result:")
    print(result)
    if "NBest" in result["DisplayText"]:
        for phrase in result["NBest"]:
            print("👤 Speaker:", phrase["SpeakerId"])
            print("🗣 Text:", phrase["DisplayText"])
else:
    print("❌ Error:", response.status_code)
    print(response.text)
