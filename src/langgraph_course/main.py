from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from langgraph_course.graph.graph import app


if __name__ == "__main__":
    print("Hello advanced RAG")
    print(app.invoke(input={"question": "what is agent memory?"}))