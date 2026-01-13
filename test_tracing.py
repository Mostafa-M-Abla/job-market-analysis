import os
import dotenv
from langsmith import traceable
from langchain_openai import ChatOpenAI

# Load environment variables
dotenv.load_dotenv()

@traceable(name="LangSmith Minimal Trace Test")
def say_hello():
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    response = llm.invoke("Say a short welcome message.")
    return response.content

if __name__ == "__main__":
    print("LANGCHAIN_API_KEY:", os.getenv("LANGCHAIN_API_KEY"))
    print("LANGSMITH_PROJECT:", os.getenv("LANGSMITH_PROJECT"))
    print("LANGSMITH_TRACING:", os.getenv("LANGSMITH_TRACING"))

    result = say_hello()
    print("LLM Response:", result)
