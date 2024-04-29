import streamlit as st
import os
from dotenv import load_dotenv
import speech_recognition as sr
import openai
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import sys
import elevenlabs

load_dotenv()

# Set API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Initialize ElevenLabs client
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

recognizer = sr.Recognizer()

def handle_conversation():
    st.title("Speech-to-Text and Text-to-Speech Demo")

    while True:
        try:
            st.write("Listening...")
            with sr.Microphone() as source:
                audio = recognizer.listen(source, timeout=3)  # Adjust timeout as needed (e.g., 3 seconds)

            # Convert speech to text
            transcript = recognizer.recognize_google(audio)
            st.write("You said:", transcript)

            # Generate response using OpenAI GPT-3.5
            prompt = """you are expert in all fields in research. generate response maximum 1000 characters only"""
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": transcript},
                ],
            )

            text = response['choices'][0]['message']['content']

            # Convert text to speech using ElevenLabs
            audio = client.text_to_speech.convert(
                voice_id="pNInz6obpgDQGcFmaJgB",  # Adam pre-made voice
                optimize_streaming_latency="1",
                output_format="mp3_22050_32",
                text=text,
                model_id="eleven_turbo_v2",  # Use the turbo model for low latency
                voice_settings=VoiceSettings(
                    stability=0.0,
                    similarity_boost=1.0,
                    style=0.0,
                    use_speaker_boost=True,
                ),
            )

            # Play the audio stream
            elevenlabs.play(audio)

            # Display AI response
            st.write("AI:", text)

        except sr.UnknownValueError:
            st.write("Could not understand audio")
        except sr.RequestError as e:
            st.write("Error with the request; {0}".format(e))
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    handle_conversation()
