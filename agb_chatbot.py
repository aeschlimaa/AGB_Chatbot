from operator import itemgetter
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch

# For OpenAI usage
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model Selection
# MODEL = "gpt-3.5-turbo"
# MODEL = "mistral"
# MODEL = "deepseek-r1:7b"
# MODEL = "llama3.1:8b"
# MODEL = "llama3.2"
MODEL = "tinyllama"

# conditional Model and embeddings selection
if MODEL in ["mistral", "deepseek-r1:7b", "llama3.1:8b", "llama3.2", "tinyllama"]: #  Ollama LLM and Embeddings
    model = OllamaLLM(temperature= 0.3, model=MODEL)
    embeddings = OllamaEmbeddings(model=MODEL)
elif MODEL.startswith("gpt-"):
    # Use OpenAI LLM and Embeddings
    if MODEL.startswith("gpt-") and not OPENAI_API_KEY:
        raise ValueError("OpenAI API Key is missing, check .env file.")
    model = ChatOpenAI(temperature=0.3, model=MODEL, api_key=OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
else:
    raise ValueError(f"Model not supported in this script: {MODEL}")

# Output Parser
parser = StrOutputParser()

# PDF Loader single document
#loader = PyPDFLoader("agb-fuer-den-erwerb-und-die-nutzung-des-generalabonnements.pdf")
#pages = loader.load_and_split()

# PDF Loader Directory
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

directory = "B2B"

pdf_files = [f for f in os.listdir(directory) if f.endswith(".pdf")]

documents = []
for pdf_file in pdf_files:
    pdf_path = os.path.join(directory, pdf_file)
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    documents.extend(pages)

if not documents:
    raise ValueError(f" No PDFs found in: {directory}")

print(f"{len(documents)} pages loaded from {len(pdf_files)} PDFs.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", ". ", "? ", "! "])
pages = text_splitter.split_documents(documents)

print(f"{len(pages)} text chunks created.")

# Prompt Template
template = """ \
"You are an AI assistant specialized in analyzing terms and conditions. Your task is to extract precise and concise legal information.

- Only use the provided context to answer the question.
- If the context does not contain enough information, respond with: "I don't know."
- Provide a structured answer with numbered points if applicable.

If you can't answer the question, reply "I don't know"." \
"" \
"Context: {context}" \
"" \
"Question: {question}" \
"""
prompt = PromptTemplate.from_template(template)

# Vector Store
vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
retriever = vectorstore.as_retriever()

# Chain
chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question")
    }
    | prompt
    | model
    | parser
)

# Invoke
print(chain.invoke({"question": "In welchem Fall erhalte ich einen Übergans-SwissPass?"}), end="", flush=True)