import streamlit as st
from vector import load_vector_db
from langchain_ollama import ChatOllama

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Gemma 3 Career AI",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Career Guidance AI (Gemma 3:1B)")
st.caption("Ask about universities, courses, fees, and career paths.")


# ---------------- LOAD VECTOR DB ---------------- #
if "vector_store" not in st.session_state:
    with st.spinner("🔄 Loading knowledge base..."):
        st.session_state.vector_store = load_vector_db()


# ---------------- LLM SETUP ---------------- #
llm = ChatOllama(
    model="gemma3:1b",   # Change if your ollama model name differs
    temperature=0.1,
    num_predict=512
)


# ---------------- PROMPT ---------------- #
system_prompt = (
    "You are an expert career guidance counselor.\n"
    "Answer ONLY from the provided context.\n"
    "If the answer is not available, say "
    "'I don't have that information.'\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])


# ---------------- CHAINS ---------------- #
combine_docs_chain = create_stuff_documents_chain(
    llm,
    prompt
)

retriever = st.session_state.vector_store.as_retriever(
    search_kwargs={"k": 4}
)

retrieval_chain = create_retrieval_chain(
    retriever,
    combine_docs_chain
)


# ---------------- CHAT MEMORY ---------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# ---------------- USER INPUT ---------------- #
if user_input := st.chat_input("Ask about a course, university, or fees..."):

    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    st.chat_message("user").write(user_input)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):

            response = retrieval_chain.invoke({
                "input": user_input
            })

            answer = response["answer"]

            st.write(answer)

            # Save assistant reply
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

            # ---------- SOURCE DOCUMENTS (Optional) ---------- #
            if "context" in response:
                with st.expander("📚 Source Context"):
                    for doc in response["context"]:
                        st.write(doc.page_content)
