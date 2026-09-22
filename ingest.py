from pathlib import Path
from pypdf import PdfReader
from docx import Document
from openai import OpenAI
from dotenv import load_dotenv
import chromadb

load_dotenv()

client = OpenAI()

DOCUMENTS_FOLDER = Path("documents")

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="personal_knowledge"
)


def load_pdf(file_path):
    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text

def load_txt(file_path):

    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def load_docx(file_path):
    document = Document(file_path)

    text = ""
    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"

    return text


def create_chunks(text, chunk_size=1000, overlap=200):
    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def create_embedding(text):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


print("Building Personal Knowledge Base")
print("---------------------------------")

for file in DOCUMENTS_FOLDER.iterdir():
    extension = file.suffix.lower()

    if extension == ".pdf":

        print(f"\nReading PDF: {file.name}")

        text = load_pdf(file)

    elif extension == ".txt":

        print(f"\nReading TXT: {file.name}")

        text = load_txt(file)

    elif extension == ".docx":

        print(f"\nReading DOCX: {file.name}")

        text = load_docx(file)

    else:

        continue


print("\nKnowledge base ready!")