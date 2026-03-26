# RAG Ablation Study: Component Descriptions

| Config | Components | Description |
|---|---|---|
| Dense | Dense retrieval (BGE) | BGE embeddings + ChromaDB cosine similarity |
| +Rerank | Dense retrieval (BGE) + Cross-encoder reranker | + Cross-encoder reranking (ms-marco-MiniLM-L-6-v2) |
| +Rerank+QE | Dense retrieval (BGE) + Cross-encoder reranker + Medical query expansion | + Medical query expansion (synonym injection) |
| Full Medical | Dense retrieval (BGE) + Cross-encoder reranker + Medical query expansion + Clinical relevance filter | + Clinical relevance post-filtering (entity matching) |

## Scientific References

- Dense retrieval: Karpukhin et al. (2020) 'Dense Passage Retrieval'
- Cross-encoder reranking: Nogueira & Cho (2019) 'Passage Re-ranking with BERT'
- Query expansion: Jagerman et al. (2023) 'Query Expansion by Prompting LLMs'
- RAG for clinical NLP: Lewis et al. (2020) 'Retrieval-Augmented Generation'