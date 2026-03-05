from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

llm = ChatOllama(
    model="minimax-m2.5:cloud",
    temperature=0.7
)

#message = [ SystemMessage(content="?."), HumanMessage(content="What is RAG?")]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

chain = prompt | llm | StrOutputParser()

chat_history = []
max_turns = 10


def chat(question):
    current_turn = len(chat_history) // 2

    if current_turn >= max_turns:
        return (
            "Context window is full! "
            "The AI may not follow the previous thread properly. "
            "Please type 'clear' to clear the context and start a new thread."
        )

    response = chain.invoke({
        "question": question,
        "chat_history": chat_history
    })

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response))

    remaining = max_turns - (current_turn + 1)

    if remaining <= 2:
        response += f"\n\n[Warning!: {remaining} turns left before context window is full]"

    return response


def main():
    print("Langchain Chatbot with RAG. Type 'quit' to exit, 'clear' to clear history.")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            break

        if user_input.lower() == "clear":
            chat_history.clear()
            print("Chat history cleared. Starting a new thread.")
            continue

        print(f"AI: {chat(user_input)}")


main()
#print(chat("What is RAG?"))
#print(chat("give python example of it?"))
#print(chat("Now explain it?"))



 
#response = chain.invoke({"question": "What is rag?"}) 
#print(response)


#for chunk in chain.stream({"question": "What is RAG?"}):
    #print(chunk, end="", flush=True)