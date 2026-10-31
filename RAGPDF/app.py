import os
import streamlit as st
import pathlib
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

if "OPENAI_API_KEY" not in os.environ and "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

CAMINHO_DB = "db"

st.set_page_config(
    page_title="RAG - Consulta de PDFs",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Assistente de Consulta de PDFs")
st.write("Faça perguntas sobre os documentos da base de conhecimento.")

@st.cache_resource
def carregar_recursos():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    db = Chroma(
        persist_directory=CAMINHO_DB,
        embedding_function=embeddings
    )

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0
    )

    return db, llm


prompt_template = """
Você é um assistente que responde perguntas com base
exclusivamente no contexto fornecido.

Regras:
- Responda diretamente à pergunta.
- Utilize apenas informações do contexto.
- Não invente informações.
- Se não encontrar a resposta, informe isso claramente.
- Justifique sua resposta com base nos documentos.

Contexto:
{context}

Pergunta:
{question}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

if "historico" not in st.session_state:
    st.session_state.historico = []

question = st.text_input(
    "Digite sua pergunta:",
    placeholder="Ex.: O que é uma célula?"
)

if st.button("Consultar", type="primary"):

    if not question.strip():
        st.warning("Digite uma pergunta antes de consultar.")

    else:
        try:
            db, llm = carregar_recursos()

            with st.spinner("Consultando documentos e gerando resposta..."):

                results = db.similarity_search_with_relevance_scores(
                    question,
                    k=3
                )

                if not results or results[0][1] < 0.3:
                    resposta = (
                        "Não encontrei informações relevantes "
                        "nos documentos para responder."
                    )
                    fontes = []

                else:
                    context = "\n\n---\n\n".join(
                        doc.page_content
                        for doc, score in results
                    )

                    chain = prompt | llm

                    response = chain.invoke({
                        "question": question,
                        "context": context
                    })

                    resposta = response.content
                    fontes = results

            st.session_state.historico.append({
                "pergunta": question,
                "resposta": resposta,
                "fontes": fontes
            })

        except Exception as e:
            st.error(f"Erro ao consultar: {e}")


for item in reversed(st.session_state.historico):

    st.divider()

    st.markdown("### 🧑 Pergunta")
    st.write(item["pergunta"])

    st.markdown("### 🤖 Resposta")
    st.write(item["resposta"])

    if item["fontes"]:
        with st.expander("Ver trechos encontrados nos PDFs"):

            for i, (doc, score) in enumerate(item["fontes"], 1):

                st.markdown(f"**Trecho {i}**")
                st.caption(f"Relevância: {score:.3f}")

                st.write(doc.page_content)

                st.caption(
                    f"Origem: {doc.metadata.get('source', 'Não identificada')}"
                )

                st.divider()


if st.button("Limpar histórico"):
    st.session_state.historico = []
    st.rerun()
