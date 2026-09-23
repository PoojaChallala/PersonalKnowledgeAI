PERSONAL KNOWLEDGEAI ASSISTANT

Personal KnowledgeAI is a python-based AI knowledge assistant that allows users to add their own documents and ask questions about their content using natural language.

The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from user-provided documents before generating an answer with an OpenAI language model. 

OVERVIEW

Instead of asking an AI model to answer a question using only its general knowledge, PersonalKnowledgeAI first searches the user's documents for relevant information. 

The applcation currently supports: 
- PDF documents 
- TXT files
- DOCX documents 

HOW IT WORKS

RAG workflow: 
- User Documents
--> Document Text Extraction --> Text Chunking --> OpenAI Embeddings --> ChromaDB Vector Database --> Semantic Similarity Search --> Relevant Document Chunks --> OPENAI Language Model --> Generated Answer

TECHNOLOGY STACK

Language 
- Python 

AI / Machine Learning
- OpenAI API
- OPENAI text-embedding-3-small
- Retrieval-Augmented Generation

Vector Database
- ChromaDB

Document Processing
- pypdf
- python-docx

Application Framework
- Streamlit

Development Tools
- PyCharm
- Git
- GitHub
- Python virtual environment

EXAMPLE USE CASE    

A user can place a collection of personal, academic, or technical documents into the application's document directory

Then the user can ask questions such as: What are the main concepts discussed in these documents? or: Explain the section about machine learning in simple terms.

The system retrieves relevant sections from the documents and uses them as context when generating the response. 

