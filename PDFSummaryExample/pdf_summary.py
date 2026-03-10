import os
import base64
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load shared .env from root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Path to your PDF in shared datasets folder
pdf_path = os.path.join(os.path.dirname(__file__), "..", "datasets", "sample.pdf")

try:
    # Read and encode the PDF as base64
    with open(pdf_path, "rb") as f:
        pdf_data = base64.standard_b64encode(f.read()).decode("utf-8")

    print(f"✅ PDF loaded successfully!")
    print("⏳ Summarizing...\n")

    # Send PDF directly to Gemini
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(
                data=base64.b64decode(pdf_data),
                mime_type="application/pdf"
            ),
            "Summarize this PDF document in a clear and concise paragraph."
        ]
    )

    print("📄 Summary:")
    print(response.text)

except FileNotFoundError:
    print("❌ PDF not found. Make sure sample.pdf exists in the datasets folder.")
except Exception as e:
    print(f"Error: {e}")