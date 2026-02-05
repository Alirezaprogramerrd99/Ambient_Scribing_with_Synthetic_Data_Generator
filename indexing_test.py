# from src.knowledge_base import RAGFactory, RAGConfig, RAGBackend

# config = RAGConfig(
#     backend=RAGBackend.MANUAL,
#     persist_dir='./data/chroma_db'
# )
# factory = RAGFactory(config)

# # REBUILD with new documents
# factory.build_knowledge_base(
#     './medical_knowledge/sample', 
#     clear_existing=True  # Important: clear old index
# )

# # Verify
# retriever = factory.get_retriever()
# response = retriever.retrieve('migraine treatment triptans', top_k=3)
# for r in response.results:
#     print(f"Score: {r.score:.3f} - {r.source_file}")




from src.knowledge_base import RAGFactory, RAGConfig, RAGBackend

# Configure for LlamaIndex
config = RAGConfig(
    backend=RAGBackend.LLAMA_INDEX,
    vector_store="chroma",
    persist_dir="./data/llama_index_chroma_db",  # Different dir from manual
    embedding_model="BAAI/bge-base-en-v1.5",
    chunk_size=512,
    chunk_overlap=50,
)

factory = RAGFactory(config)

# Build knowledge base
stats = factory.build_knowledge_base(
    "./medical_knowledge/sample",
    clear_existing=True,
)
print(f"Built LlamaIndex KB: {stats}")

# Test retrieval
retriever = factory.get_retriever()
response = retriever.retrieve("migraine treatment with triptans", top_k=3)
for r in response.results:
    print(f"Score: {r.score:.3f} - {r.source_file}")