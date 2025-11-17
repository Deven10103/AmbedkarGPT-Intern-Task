from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA

# Above we have imported all the langchain models and tools required to 
# implement the Retrieval-Augmented Generation (RAG) system.

#1. Setup the Embedding Model: all-MiniLM-L6-v2 (as instructed in the assignment)

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embedding = HuggingFaceEmbeddings(model_name=embedding_model)
print("Embedding model initialized.")


#2. Load and store the speech.txt file

loader = TextLoader("speech.txt")
documents = loader.load()


# Here we added some extra logic to dynamically calculate chunk size based on average sentence length. 
# This would help in more readable and logical chunks.

sentences = [s.strip() for s in documents[0].page_content.split('.') if s.strip()]

total_characters = sum(len(sentence) for sentence in sentences)
average_length = total_characters / len(sentences)

chunk_size = int(average_length * 1.25)
chunk_overlap = int(average_length / 5)


#3. Initialize the text splitter with the above calculated chunk size. Split the content of speech.txt into chunks.

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, 
    chunk_overlap=chunk_overlap
)
texts = text_splitter.split_documents(documents)
print(f"Document split into {len(texts)} chunks.")


#4. Creating the vector store in memory using Chroma and the embedding model. 
# We are also persisting it to disk for future use.

vector_store_db = Chroma.from_documents(
    documents=texts,
    embedding=embedding,
    persist_directory="./chroma_db" 
)
print("Vector store created and stored in './chroma_db'.")


#5. Setting up the retriever to fetch relevant chunks from the vector store based on similarity search.

retriever = vector_store_db.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 5}
)


#6. Setting up the Ollama LLM and the RetrievalQA chain.

llm = OllamaLLM(model="mistral")
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever
)

print("Q&A chain is ready. You can now ask questions.")
print("Type 'exit' or 'quit' to stop.")
print("---------")


# Here we implement a loop to continuously accept user queries until the users choose to exit.

while True:
    try:
        query = input("Enter your question: ")

        if query.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
        
        answer = qa.invoke(query)
        print("\nAnswer:", answer['result'])
        print("---------")

    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
        break