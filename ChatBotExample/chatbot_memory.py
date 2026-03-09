import os
from dotenv import load_dotenv
from google import genai

# Load shared .env from root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("🤖 Chatbot with Memory Ready! Type 'quit' to exit.\n")

# 👇 This is the memory — stores full conversation history
history = []

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        print("Goodbye! 👋")
        break

    # 👇 Add user message to history
    history.append({
        "role": "user",
        "parts": [{"text": user_input}]
    })

    # 👇 Send full history to AI every time
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=history
    )

    ai_reply = response.text

    # 👇 Add AI reply to history too
    history.append({
        "role": "model",
        "parts": [{"text": ai_reply}]
    })

    print(f"\n🤖 AI: {ai_reply}\n")
