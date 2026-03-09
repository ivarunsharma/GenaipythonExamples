import os
from dotenv import load_dotenv
from google import genai

# Load the variables from .env into the system environment
load_dotenv()


client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Ask user to type a question
question = input("❓ Ask me anything: ")

try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=question
    )
    print("\n🤖 Answer:")
    print(response.text)
except Exception as e:
    print(f"Error: {e}")