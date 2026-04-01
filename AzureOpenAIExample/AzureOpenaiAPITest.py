import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()  # loads .env if present

AZURE_API_KEY      = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT     = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_VERSION  = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_DEPLOYMENT   = os.getenv("AZURE_OPENAI_DEPLOYMENT")  # your deployment name

print("Endpoint :", AZURE_ENDPOINT)
print("Deployment:", AZURE_DEPLOYMENT)
print("API Key  :", "*" * 8 + AZURE_API_KEY[-4:] if len(AZURE_API_KEY) > 4 else "(not set)")




client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
)

print("Client created successfully.")


response = client.chat.completions.create(
    model=AZURE_DEPLOYMENT,
    messages=[
        {"role": "user", "content": "Say hello and confirm you are working fine."}
    ],
    max_tokens=100,
)

print(response.choices[0].message.content)


print("Model      :", response.model)
print("Usage      :", response.usage)
print("Finish reason:", response.choices[0].finish_reason)