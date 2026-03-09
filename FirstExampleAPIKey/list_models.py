from google import genai

API_KEY = "AIzaSyD7kl1ksne6kbsfN3HbIJnEnzJmUNTweKE"

client = genai.Client(api_key=API_KEY)

print("📋 Available models on your API key:\n")
for model in client.models.list():
    print(model.name)