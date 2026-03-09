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

# Your text to summarize
text = """
Artificial intelligence (AI) is intelligence demonstrated by machines, 
as opposed to the natural intelligence displayed by animals including humans. 
AI research has been defined as the field of study of intelligent agents, 
which refers to any system that perceives its environment and takes actions 
that maximize its chance of achieving its goals. The term "artificial intelligence" 
had previously been used to describe machines that mimic and display human cognitive 
skills associated with the human mind, such as learning and problem-solving. 
This definition has since been rejected by major AI researchers who now describe 
AI in terms of rationality and acting rationally, which does not limit how 
intelligence can be articulated.
"""

try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"Summarize the following text in 2 sentences:\n\n{text}"
    )
    print("📄 Summary:")
    print(response.text)
except Exception as e:
    print(f"Error: {e}")