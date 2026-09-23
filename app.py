import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import chromadb

load_dotenv()

try:
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)
except Exception:
    client = OpenAI()

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name="personal_knowledge"
)


def load_pdf(file_path):
    from pypdf import PdfReader

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text


def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def load_docx(file_path):
    from docx import Document

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


def build_knowledge_base():

    import os

    document_info = []

    files = [
        file for file in os.listdir("documents")
        if file.lower().endswith((".pdf", ".txt", ".docx"))
    ]

    for file_name in files:

        file_path = os.path.join("documents", file_name)

        try:

            # -------------------------
            # Read document
            # -------------------------

            if file_name.lower().endswith(".pdf"):

                text = load_pdf(file_path)

            elif file_name.lower().endswith(".txt"):

                text = load_txt(file_path)

            elif file_name.lower().endswith(".docx"):

                text = load_docx(file_path)

            else:
                continue

            # -------------------------
            # Create chunks
            # -------------------------

            chunks = create_chunks(text)

            # -------------------------
            # Add chunks to Chroma
            # -------------------------

            for i, chunk in enumerate(chunks):

                chunk_id = f"{file_name}-{i}"

                embedding = create_embedding(chunk)

                collection.upsert(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        "source": file_name
                    }]
                )

            # -------------------------
            # Save document information
            # -------------------------

            document_info.append({
                "name": file_name,
                "characters": len(text),
                "chunks": len(chunks),
                "status": "✅ Loaded"
            })

        except Exception as e:

            document_info.append({
                "name": file_name,
                "characters": 0,
                "chunks": 0,
                "status": f"❌ Error: {e}"
            })

    return document_info


def create_embedding(text):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


def search_knowledge(query, number_of_results=3):

    query_embedding = create_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=number_of_results
    )

    return results


def generate_answer(query, context, conversation_history):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a personal knowledge assistant. "
                "Answer questions using the provided context and conversation history. "
                "If the answer is not in the context, say you don't know."
            )
        }
    ]

    messages.extend(conversation_history)

    messages.append({
        "role": "user",
        "content": f"""
Context from the knowledge base:

{context}

Current question:

{query}
"""
    })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return response.choices[0].message.content


# -------------------------
# Build knowledge base
# -------------------------

document_info = build_knowledge_base()


# -------------------------
# Streamlit user interface
# -------------------------

st.title("🧠 Personal Knowledge AI")

st.write(
    "Ask questions about the documents stored in your knowledge base."
)


# -------------------------
# Knowledge Base Information
# -------------------------

with st.expander("📚 Knowledge Base"):

    st.write(
        "Documents currently loaded into the knowledge base:"
    )

    for document in document_info:

        st.write(f"### 📄 {document['name']}")

        st.write(
            f"Characters extracted: **{document['characters']:,}**"
        )

        st.write(
            f"Chunks created: **{document['chunks']:,}**"
        )

        st.write(
            f"Status: **{document['status']}**"
        )

        st.divider()


# -------------------------
# Conversation history
# -------------------------

if "conversation_history" not in st.session_state:

    st.session_state.conversation_history = []


query = st.chat_input(
    "Ask a question about your documents..."
)


if query:

    st.session_state.conversation_history.append({
        "role": "user",
        "content": query
    })

    results = search_knowledge(query)

    context = "\n\n".join(
        results["documents"][0]
    )

    sources = results["metadatas"][0]

    answer = generate_answer(
        query,
        context,
        st.session_state.conversation_history
    )

    st.session_state.conversation_history.append({
        "role": "assistant",
        "content": answer
    })


# -------------------------
# Display conversation
# -------------------------

for message in st.session_state.conversation_history:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# -------------------------
# Display sources
# -------------------------

if query:

    st.subheader("Sources")

    unique_sources = []

    for source in sources:

        if source is not None and source["source"] not in unique_sources:

            unique_sources.append(
                source["source"]
            )

    for source in unique_sources:

        st.write(f"📄 {source}")
