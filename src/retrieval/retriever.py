from langchain_classic.chains import RetrievalQA

from src.config import RETRIEVAL_K


def retrieve_context(vector_store, query: str) -> str:
    docs = vector_store.similarity_search(query, k=RETRIEVAL_K)

    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in docs
    )


def ask_rag_question(llm, vector_store, question: str) -> str:
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})
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
                + "\nAnswer as an agentic RAG career assistant. "
                + "Use only resume, portfolio, job, and skill context when available. Do not fake experience. "
                + "Answer the exact question directly. "
                + "If the user asks for portfolio projects, list all portfolio projects found in the context with short descriptions. "
                + "Do not add helpful next actions, recommendations, or improvement advice unless the user asks for advice. "
                + "Focus on strengths, matching skills, strong projects, and suitable roles when relevant. "
                + "Do not give scores, weak points, negative comments, or discouraging wording. "
                + "Do not mention that the answer is positive, supportive, or strength-focused."
            )
        }
    )
    return result["result"].strip()
