from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    model="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser
# prompt1 = template1.invoke({'topic': 'The impact of climate change on global agriculture'})

# result = model.invoke(prompt1)

# prompt2 = template2.invoke({'text':result.content})

# result2 = model.invoke(prompt2)

res = chain.invoke({'topic': 'The impact of heavy water consumption of AI infrastructure on climate change'})

print(res)
