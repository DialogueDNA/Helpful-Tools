import assemblyai as aai
import os


# 1. הכניסי כאן את ה-API KEY שלך
aai.settings.api_key = "590fdc1b0db146f49a266497d6d75572"

# 📁 הנתיב לקובץ האודיו
file_path = os.path.join("..", "recordings", "Gal Gadot's Daughter is Proud She's Wonder Woman.wav")

# 🧠 יצירת מופע תמלול עם זיהוי דוברים
transcriber = aai.Transcriber()
config = aai.TranscriptionConfig(speaker_labels=True)

# 🎙️ ביצוע התמלול
transcript = transcriber.transcribe(file_path, config=config)

# 🖨️ הדפסת התוצאה למסך
for utterance in transcript.utterances:
    print(f"🧍 Speaker {utterance.speaker}: {utterance.text}")

# 💾 שמירה לקובץ טקסט בתיקיית text
output_path = os.path.join("..", "text", "gal_gadot_transcript.txt")
with open(output_path, "w", encoding="utf-8") as f:
    for utterance in transcript.utterances:
        f.write(f"Speaker {utterance.speaker}: {utterance.text}\n")