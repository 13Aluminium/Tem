import pdfplumber
import requests
import json
import re
from typing import List, Dict
import os

class PDFToScript:
    def __init__(self):
        self.url = "http://localhost:11434/api/generate"
        self.headers = {
            "Content-Type": "application/json"
        }

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    # Extract text with better formatting preservation
                    page_text = page.extract_text(x_tolerance=2, y_tolerance=2)
                    # Clean up the text
                    page_text = re.sub(r'\n(?!\n)', ' ', page_text)
                    page_text = re.sub(r'\s+', ' ', page_text)
                    text += page_text + "\n\n"
                return text.strip()
        except Exception as e:
            print(f"Error extracting PDF: {str(e)}")
            return ""

    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into smaller chunks for processing."""
        # Split on sentences to avoid cutting in middle of dialogue
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def process_chunk(self, chunk: str) -> str:
        """Process a single chunk of text using Llama model."""
        prompt = f"""
        Convert the following text into a drama script format. 
        Include speaker names in CAPS followed by their dialogue.
        Include important action descriptions in (parentheses).
        If there's narration, mark it as NARRATOR.
        Format it clearly with one speaker per line.
        
        Text to convert:
        {chunk}
        
        Output the script in this format:
        SPEAKER: Dialogue
        (Action description)
        NARRATOR: Narration
        
        Only output the formatted script, no additional text.
        """

        data = {
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(
                self.url, 
                headers=self.headers, 
                data=json.dumps(data)
            )
            
            if response.status_code == 200:
                response_data = json.loads(response.text)
                return response_data["response"].strip()
            else:
                print(f"Error: Response status code {response.status_code}")
                return ""
        except Exception as e:
            print(f"Error processing chunk: {str(e)}")
            return ""

    def clean_script(self, script: str) -> str:
        """Clean and format the generated script."""
        # Remove any markdown or code block markers
        script = re.sub(r'```.*?```', '', script, flags=re.DOTALL)
        script = re.sub(r'`.*?`', '', script)
        
        # Ensure consistent formatting
        script = re.sub(r'\n{3,}', '\n\n', script)
        
        # Ensure speaker names are in caps
        def capitalize_speaker(match):
            return match.group(1).upper() + match.group(2)
        
        script = re.sub(r'([A-Za-z]+)(:)', capitalize_speaker, script)
        
        return script.strip()

    def generate_script(self, pdf_path: str, output_path: str) -> None:
        """Generate a complete drama script from PDF and save to file."""
        print("Extracting text from PDF...")
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            print("Failed to extract text from PDF")
            return

        print("Converting to script format...")
        chunks = self.chunk_text(text)
        full_script = []
        
        for i, chunk in enumerate(chunks, 1):
            print(f"Processing chunk {i}/{len(chunks)}...")
            script_chunk = self.process_chunk(chunk)
            if script_chunk:
                full_script.append(script_chunk)

        # Combine and clean the full script
        final_script = "\n\n".join(full_script)
        final_script = self.clean_script(final_script)

        # Add header to the script
        header = f"""DRAMA SCRIPT
Generated from: {os.path.basename(pdf_path)}
Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
        final_script = header + final_script

        # Save to file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(final_script)
            print(f"\nScript successfully generated and saved to: {output_path}")
        except Exception as e:
            print(f"Error saving script: {str(e)}")

def main():
    # Initialize converter
    converter = PDFToScript()
    
    # Get input and output paths
    pdf_path = "WHOLE_BOOK_MELUHA_1.pdf"
    output_path = input("Enter the path for the output script (default: script.txt): ").strip()
    
    if not output_path:
        output_path = "script_full.txt"
    
    # Generate script
    converter.generate_script(pdf_path, output_path)

if __name__ == "__main__":
    main()