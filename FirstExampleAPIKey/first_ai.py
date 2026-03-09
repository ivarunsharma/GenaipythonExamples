import os
from dotenv import load_dotenv
from google import genai

# Load the variables from .env into the system environment
load_dotenv()

# Access the variable
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize your client
client = genai.Client(api_key=api_key)

# Test call
try:
    response = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents="Hello, world!"
    )
    print("Success:", response.text)
except Exception as e:
    print(f"Error: {e}")