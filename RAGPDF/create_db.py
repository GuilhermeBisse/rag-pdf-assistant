from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

BASE_PATH = "base"
CAMINHO_DB = "db"


def load_documents():
    if not Path(BASE_PATH).exists():
        raise FileNotFoundError(
            f"A pasta '{BASE_PATH}' não foi encontrada."
        )

    loader = PyPDFDirectoryLoader(BASE_PATH, glob="*.pdf")
    documents = loader.load()

    if not documents:
        raise ValueError(
            f"Nenhum PDF encontrado na pasta '{BASE_PATH}'."
        )

    print(f"{len(documents)} páginas carregadas.")
    return documents


def split_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True
    )

    chunks = text_splitter.split_documents(documents)
    print(f"{len(chunks)} chunks criados.")

    return chunks


def vetorize(chunks):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CAMINHO_DB
    )

    print("Database created successfully!")


def create_db():
    documents = load_documents()
    chunks = split_chunks(documents)
    vetorize(chunks)


if __name__ == "__main__":
    create_db()
