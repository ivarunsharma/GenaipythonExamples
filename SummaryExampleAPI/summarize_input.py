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

print("📄 AI Text Summarizer")
print("Paste your text below. When done, type END on a new line and press Enter.\n")

# Collect multiple lines of input
lines = []
while True:
    line = input()
    if line.strip().upper() == "END":
        break
    lines.append(line)

text = "\n".join(lines)

# Ask how long the summary should be
print("\nHow long should the summary be?")
print("1. One sentence")
print("2. Three sentences")
print("3. One paragraph")
choice = input("\nEnter 1, 2 or 3: ")

length_map = {
    "1": "one sentence",
    "2": "three sentences",
    "3": "one paragraph"
}
length = length_map.get(choice, "three sentences")

try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"Summarize the following text in {length}:\n\n{text}"
    )
    print(f"\n📄 Summary ({length}):")
    print(response.text)
except Exception as e:
    print(f"Error: {e}")