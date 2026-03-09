import os
from dotenv import load_dotenv
from google import genai

# Load shared .env from root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("🤖 Chatbot Ready! Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        print("Goodbye! 👋")
        break

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_input       # ← only sends current message, no memory
    )

    print(f"\n🤖 AI: {response.text}\n")