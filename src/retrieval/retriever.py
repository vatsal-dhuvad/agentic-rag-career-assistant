from langchain_classic.chains import RetrievalQA

from src.config import RETRIEVAL_K


def retrieve_context(vector_store, query: str) -> str:
    docs = vector_store.similarity_search(query, k=RETRIEVAL_K)

    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in docs
    )


def ask_rag_question(llm, vector_store, question: str) -> str:
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
    )

    result = qa_chain.invoke(
        {
            "query": (
                question
                + "\nAnswer as a positive agentic RAG career assistant. "
                + "Use resume/job context when available. Do not fake experience. "
                + "Focus only on strengths, matching skills, strong projects, suitable roles, and helpful next actions. "
                + "Do not give scores, weak points, negative comments, or discouraging wording."
            )
        }
    )
    return result["result"].strip()
