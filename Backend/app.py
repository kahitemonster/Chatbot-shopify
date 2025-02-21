from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
import uvicorn
import sys

os.environ["OPENAI_API_KEY"] = "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"
chat_history = []
# Create fastapi object app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

documents = []

def load_files(folder):
  for file in os.listdir(folder):
    full_path = os.path.join(folder, file)
    if os.path.isdir(full_path):
      load_files(full_path) 
    elif file.endswith(".txt"):
      loader = TextLoader(full_path, encoding='utf-8')
      documents.extend(loader.load())

load_files("datas")

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
documents = text_splitter.split_documents(documents)

vectordb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(), persist_directory="./data")
vectordb.persist()

pdf_qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo"),
    vectordb.as_retriever(search_kwargs={'k': 6}),
    return_source_documents=True,
    verbose=False
)

@app.get("/api/response")
async def get_response(message: str, request: Request):
  result = pdf_qa(
    {"question": message, "chat_history": chat_history})
  chat_history.append((message, result["answer"]))
  return result

if __name__ == "__main__":
  uvicorn.run("app:app",port = 8000,reload=True)
