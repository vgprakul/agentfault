import os

from langchain_google_genai import ChatGoogleGenerativeAI


class GeminiLLM:
    def __init__(self):
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY is not set")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
        )

    def invoke(self, prompt):
        return self.llm.invoke(prompt)
