from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import List, Union
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

chat_history: List[Union[SystemMessage, HumanMessage, AIMessage]] = [
    SystemMessage(content='You are a helpful assistant')
]  

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() == 'q':
        break
    model_response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=model_response.content))
    print("Bot: ", model_response.content)