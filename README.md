# FirstGenAI: Python Implementation Suite

A collection of Generative AI tools and conversational agents focusing on content processing, summarization, and stateful interaction models.

---

## 🚀 Features

### 1. Content Summarization & Generation
* **Summary Module:** Extracts key insights and high-level overviews from large text inputs.
* **Content Generation:** Synthesizes new material or expands on existing summaries using structured prompts.

### 2. Conversational Agents
* **Chatbot (Without Memory):** * **Stateless:** Processes each user input independently.
    * **Use Case:** Ideal for single-turn tasks like classification or direct Q&A.
* **Chatbot (With Memory):** * **Stateful:** Retains conversation history to provide contextually aware responses.
    * **Use Case:** Supports natural follow-up questions and multi-turn dialogue.

---

## 📁 Project Structure

GENAIPYTHONEXAMPLES/
├── .env                                 
├── datasets/
├── FirstExampleAPIKey/
├── SummaryExampleAPI/
├── ChatBotExample/
└── README.md


##Getting Started
Prerequisites
Python 3.8+

An API Key for your preferred LLM provider (OpenAI, Google Gemini, or Anthropic).


🔧 Technical Details
Memory Management: Implements a conversation buffer to maintain context across turns.

Prompt Engineering: Utilizes f-string formatting for dynamic prompt construction.

Architecture: Modular design allowing for easy integration with different Transformer-based models.
