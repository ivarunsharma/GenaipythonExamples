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

# Read from a .txt file
#filename = input("📂 Enter the path to your .txt file: ")

# Automatically point to shared datasets folder
filename = os.path.join(os.path.dirname(__file__), "..", "datasets", "summarization_dataset.txt")


try:
    with open(filename, "r") as f:
        text = f.read()

    print(f"\n✅ File loaded! ({len(text)} characters)")
    print("⏳ Summarizing...\n")

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"Summarize the following document in a clear and concise paragraph:\n\n{text}"
    )

    print("📄 Summary:")
    print(response.text)

except FileNotFoundError:
    print("❌ File not found. Please check the path and try again.")
except Exception as e:
    print(f"Error: {e}")