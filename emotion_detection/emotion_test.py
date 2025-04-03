from transformers import pipeline
import os
from colorama import Fore, Style, init

# 🟢 מאפשר הדפסת צבעים בקונסול
init(autoreset=True)

# 🎭 טעינת מודל רגשות מ-HuggingFace
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)

# 🎨 פונקציה שמחזירה צבע + אימוג'י לפי הרגש
def format_emotion(emotion, score):
    emotions = {
        "joy":     (Fore.YELLOW, "😊"),
        "anger":   (Fore.RED, "😡"),
        "sadness": (Fore.BLUE, "😢"),
        "fear":    (Fore.MAGENTA, "😨"),
        "disgust": (Fore.GREEN, "🤢"),
        "surprise":(Fore.CYAN, "😲"),
        "neutral": (Fore.WHITE, "😐")
    }
    color, emoji = emotions.get(emotion.lower(), (Fore.WHITE, "❓"))
    return f"{color}{emoji} Emotion: {emotion} ({score}%)"

# 📂 נתיבי קלט ופלט
input_path = os.path.join("..", "text", "gal_gadot_transcript.txt")
output_dir = os.path.join("..", "emotion_detection", "results")
output_path = os.path.join(output_dir, "emotion_results.txt")

# 📁 יצירת תיקיית תוצאה אם לא קיימת
os.makedirs(output_dir, exist_ok=True)

# 🔁 ניתוח רגשות שורה אחר שורה
with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        if not line.strip():
            continue  # מדלג על שורות ריקות
        try:
            speaker, text = line.split(":", 1)
            prediction = emotion_classifier(text.strip())[0][0]
            emotion = prediction["label"]
            score = f"{prediction['score'] * 100:.1f}"  # 🔥 עיגול לאחוז מדויק

            # 👀 הצגה בקונסול עם צבעים ואימוג'ים
            print(f"{Fore.LIGHTWHITE_EX}{speaker.strip()}: {text.strip()}")
            print(format_emotion(emotion, score) + "\n")

            # 💾 כתיבה לקובץ פלט (ללא צבעים)
            outfile.write(f"{speaker.strip()}: {text.strip()}\n")
            outfile.write(f"→ Emotion: {emotion} ({score}%)\n\n")

        except Exception as e:
            print(f"{Fore.RED}❌ שגיאה בעיבוד השורה: {line.strip()}")
            print(str(e))
