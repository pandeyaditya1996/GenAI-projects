from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever


model = OllamaLLM(model="llama3.2")

template = """
    You are an expert in answering questions about a pizza restaurant

    Here are some relevant reviews: {reviews}

    Here is the question: {question}
"""

prompt = ChatPromptTemplate.from_template(template, model = model)

chain = prompt | model

while True:

    question = input("Enter a question: ")
    if question == "exit":
        break

    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)