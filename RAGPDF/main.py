from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

CAMINHO_DB = "db"

prompt_template = """
Você é um assistente que responde perguntas com base
exclusivamente no contexto dos documentos fornecidos.

Regras:
- Responda diretamente à pergunta.
- Use somente informações presentes no contexto.
- Não invente horários, regras ou exceções.
- Se não houver informações suficientes, informe isso claramente.
- Justifique a resposta com base no contexto.

Contexto:
{context}

Pergunta:
{question}
"""

def ask():
    question = input("Digite sua pergunta: ")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    db = Chroma(
        persist_directory=CAMINHO_DB,
        embedding_function=embeddings
    )

    results = db.similarity_search_with_relevance_scores(
        question,
        k=3
    )

    if not results or results[0][1] < 0.3:
        print("Não encontrei contexto relevante nos documentos.")
        return

    context = "\n\n---\n\n".join(
        doc.page_content for doc, score in results
    )

    prompt = ChatPromptTemplate.from_template(prompt_template)

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0
    )

    chain = prompt | llm

    response = chain.invoke({
        "question": question,
        "context": context
    })

    print("\nResposta:\n")
    print(response.content)

if __name__ == "__main__":
    ask()
