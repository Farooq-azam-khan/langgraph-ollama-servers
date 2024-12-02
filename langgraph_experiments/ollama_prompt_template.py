from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM


template = ""
with open("./system_msg.txt") as f:
    template = f.read()

prompt = ChatPromptTemplate.from_template(template)

llm = OllamaLLM(model="llama3.2")

qa_chain = prompt | llm

if __name__ == "__main__":
    print(qa_chain.invoke({"question": "What is LangChain?"}))
