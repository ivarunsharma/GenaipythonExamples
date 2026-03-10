import os
import base64
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load shared .env from root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# ── STEP 1: Load the PDF ──────────────────────────────────────────
pdf_path = os.path.join(os.path.dirname(__file__), "..", "datasets", "sample.pdf")

try:
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    print("✅ PDF loaded successfully!\n")

except FileNotFoundError:
    print("❌ PDF not found. Make sure sample.pdf exists in datasets folder.")
    exit()

# ── STEP 2: Choose Summary Length ────────────────────────────────
print("How detailed should the summary be?")
print("  1. Short     — 1 sentence")
print("  2. Medium    — 1 paragraph")
print("  3. Detailed  — key points in depth\n")

choice = input("Enter 1, 2 or 3: ").strip()

length_map = {
    "1": "in exactly one sentence",
    "2": "in one clear paragraph",
    "3": "in detail covering all key points"
}

# Default to medium if invalid choice
length = length_map.get(choice, "in one clear paragraph")
print(f"\n⏳ Generating summary...\n")

# ── STEP 3: Generate Summary ──────────────────────────────────────
pdf_part = types.Part.from_bytes(
    data=pdf_bytes,
    mime_type="application/pdf"
)

try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            pdf_part,
            f"Summarize this PDF document {length}."
        ]
    )

    print("📄 Summary:")
    print(response.text)

except Exception as e:
    print(f"Error generating summary: {e}")
    exit()

# ── STEP 4: Q&A Loop ─────────────────────────────────────────────
print("\n" + "─" * 50)
print("❓ You can now ask questions about this PDF!")
print("   Type 'quit' to exit.\n")

while True:
    question = input("You: ").strip()

    if question.lower() == "quit":
        print("Goodbye! 👋")
        break

    if not question:
        print("Please enter a question!\n")
        continue

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                pdf_part,
                f"Based on this PDF document, answer the following question:\n{question}"
            ]
        )

        print(f"\n🤖 Answer: {response.text}\n")

    except Exception as e:
        print(f"Error: {e}\n")