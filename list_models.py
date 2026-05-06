import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load your API key from the .env file
load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

print("Here are the models available for text generation (generateContent):\n")

# Loop through and print the models that support chat/text generation
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"- {model.name}")