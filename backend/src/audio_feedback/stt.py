import speech_recognition as sr
from openai import OpenAI
import os

load_dotenv()

client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

def transcribe_audio():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print('Be quiet waiting for calibration...')
        recognizer.adjust_for_ambient_noise(source, duration=1) #calibration for background noise
        print('Please speak now...')
        audio = recognizer.listen(source)                       #actually starts listening

    temp_file = "temp_audio.wav"                                #need to store the saved audio here
    with open(temp_file, 'wb') as f:
        f.write(audio.get_wav_data())

    print("[Processing...] Transcribing with Whisper...")

    try:
        with open(temp_file, 'rb') as audio_file:
            user_text = client.audio.transcriptions.create(
                model= "whisper-1",
                file= audio_file,
                response_format= "text"
            )
        print("Your said: ", user_text)
        return user_text
    except Exception as e:
        print("Error during the transcription: ", str(e))
        return None
    
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    text = transcribe_audio()
    

