import os
import torch
import re
import spacy
from TTS.api import TTS
from pydub import AudioSegment
import pdfplumber
import warnings

warnings.filterwarnings("ignore", category=UserWarning)  # Suppress TTS warnings

# Load spaCy transformer model (this may take a moment)
nlp = spacy.load("en_core_web_trf")

class BookToSpeech:
    def __init__(self, speed=1.0):
        self.speed = speed
        self.model_name = "tts_models/en/vctk/vits"
        self.tts = TTS(self.model_name).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.available_voices = ["p225", "p226", "p227", "p228", "p229",
                                 "p230", "p231", "p232", "p233", "p234"]
        self.character_voices = {"narrator": "p225"}
        self.next_voice_index = 1
        os.makedirs("output_audio", exist_ok=True)

    # Extract text from only the first 3 pages of the PDF
    def extract_text_from_pdf(self, pdf_path):
        try:
            with pdfplumber.open(pdf_path) as pdf:
                full_text = []
                for page in pdf.pages[:3]:
                    text = page.extract_text(x_tolerance=2, y_tolerance=2)
                    # Convert smart quotes to standard quotes and fix punctuation spacing
                    text = re.sub(r"‘|’", '"', text)
                    text = re.sub(r'\s+([.,!?])', r'\1', text)
                    # Replace newlines with a space so dialogue isn't broken
                    text = re.sub(r'\n+', ' ', text)
                    full_text.append(text)
                return " ".join(full_text)
        except Exception as e:
            print(f"PDF Extraction Error: {str(e)}")
            return None

    # ML-based speaker detection using spaCy dependency parsing and NER
    def detect_speaker_ml(self, sentence):
        # Process the sentence with spaCy
        doc = nlp(sentence)
        dialogue = None

        # First, extract dialogue text if it's enclosed in quotes
        quote_matches = re.findall(r'["\'](.*?)["\']', sentence)
        if quote_matches:
            dialogue = quote_matches[0].strip()

        # Define a set of common speech verbs
        speech_verbs = {"say", "said", "exclaim", "exclaimed", "shout", "shouted",
                        "whisper", "whispered", "scream", "screamed", "yell", "yelled",
                        "ask", "asked", "reply", "replied"}

        speaker = None
        # Look through tokens for a speech verb; then check its subject
        for token in doc:
            if token.lemma_.lower() in speech_verbs:
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        speaker = child.text
                        break
                if speaker:
                    break

        # If no speaker is found, but the sentence contains quotes, label it as dialogue.
        if not speaker:
            if '"' in sentence or "'" in sentence:
                speaker = "dialogue"
            else:
                speaker = "narrator"
        # If no dialogue text was found, treat the whole sentence as dialogue
        if not dialogue:
            dialogue = sentence
        return speaker, f'"{dialogue}"'

    # Use the ML-based approach for speaker detection
    def detect_speaker(self, current_index, sentences):
        sentence = sentences[current_index].strip()
        return self.detect_speaker_ml(sentence)

    # Generate audio by processing text sentence-by-sentence
    def generate_audio(self, text, output_filename="output.wav"):
        # Split text into sentences (this basic split may need refinement)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        output_audio = AudioSegment.silent(duration=100)
        temp_files = []

        try:
            for idx, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue

                speaker, dialogue = self.detect_speaker(idx, sentences)
                voice = self.assign_voice(speaker)
                temp_path = f"output_audio/temp_{idx}.wav"
                print(f"Processing: {speaker} => {dialogue[:50]}...")

                self.tts.tts_to_file(
                    text=dialogue,
                    speaker=voice,
                    file_path=temp_path,
                    speed=self.speed
                )

                segment = AudioSegment.from_wav(temp_path)
                if self.speed != 1.0:
                    segment = segment._spawn(segment.raw_data, 
                                               overrides={"frame_rate": int(segment.frame_rate * self.speed)}
                                              ).set_frame_rate(segment.frame_rate)

                output_audio += segment + AudioSegment.silent(duration=200)
                temp_files.append(temp_path)

            output_audio.export(f"output_audio/{output_filename}", format="wav")
            print(f"Successfully generated: output_audio/{output_filename}")
        except Exception as e:
            print(f"Generation Error: {str(e)}")
        finally:
            for f in temp_files:
                if os.path.exists(f):
                    os.remove(f)

    # Assign voices to speakers; new speakers get the next available voice.
    def assign_voice(self, character):
        if character not in self.character_voices:
            try:
                self.character_voices[character] = self.available_voices[self.next_voice_index]
                self.next_voice_index = (self.next_voice_index + 1) % len(self.available_voices)
                print(f"Assigned {self.character_voices[character]} to {character}")
            except IndexError:
                self.character_voices[character] = "p225"
        return self.character_voices[character]

# Usage example:
if __name__ == "__main__":
    # Initialize with speed=0.8 for 20% slower speech
    converter = BookToSpeech(speed=0.8)
    # Extract text from only the first 3 pages for faster testing
    text = converter.extract_text_from_pdf("ch_1.pdf")

    if text:
        converter.generate_audio(text, "chapter_1.wav")
        print("\nCharacter Voices:")
        for char, voice in converter.character_voices.items():
            print(f"{char.ljust(15)} => {voice}")
    else:
        print("Failed to process PDF")
