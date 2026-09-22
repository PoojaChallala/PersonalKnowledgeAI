from openai import OpenAI
from dotenv import load_dotenv
import chromadb

load_dotenv()

client = OpenAI()

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="personal_knowledge"
)


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


print("Personal Knowledge AI")
print("---------------------")

conversation_history = []

while True:

    query = input("\nYou: ")

    if query.lower() == "exit":

        print("Goodbye!")

        break

    conversation_history.append({
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
        conversation_history
    )

    conversation_history.append({
        "role": "assistant",
        "content": answer
    })

    print("\nAI:")
    print(answer)

    print("\nSources:")
    unique_sources = []

    for source in sources:

        if source is not None and source["source"] not in unique_sources:
            unique_sources.append(source["source"])

    for source in unique_sources:
        print(f" {source}")