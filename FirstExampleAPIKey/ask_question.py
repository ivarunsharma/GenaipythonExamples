import os
from dotenv import load_dotenv
from google import genai

# Load the variables from .env into the system environment
##load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# Access the variable
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize your client
client = genai.Client(api_key=api_key)

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