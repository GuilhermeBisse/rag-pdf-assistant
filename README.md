# RAG PDF Assistant

Assistente inteligente desenvolvido em Python para consulta de documentos PDF utilizando a arquitetura RAG (Retrieval-Augmented Generation).

**[Acesse a aplicação rodando online no Streamlit Cloud](https://rag-pdf-assistant0.streamlit.app/)**

## Tecnologias utilizadas

- Python
- Streamlit
- LangChain
- OpenAI API
- ChromaDB
- PyPDF

## Funcionalidades

- Carregamento de documentos PDF.
- Extração e divisão de textos em chunks.
- Geração de embeddings utilizando OpenAI.
- Armazenamento vetorial com ChromaDB.
- Recuperação de informações por similaridade semântica.
- Geração de respostas utilizando GPT-4.1-mini.
- Interface interativa desenvolvida com Streamlit.

## Arquitetura

1. Os documentos PDF são carregados.
2. O conteúdo é dividido em pequenos fragmentos (chunks).
3. Os embeddings são gerados pela OpenAI.
4. Os vetores são armazenados no ChromaDB.
5. O usuário realiza uma pergunta pela interface.
6. O sistema recupera os trechos mais relevantes.
7. O modelo da OpenAI gera uma resposta baseada no contexto recuperado.

# Base de Dados

Sinta-se livre para adicionar/remover/modificar arquivos na sua base de dados. O único procedimento que deve ser mantido é a criação de um novo banco de embeddings ao atualizar os documentos .pdf usados.
Para o teste do projeto, foram usados dois documentos:

1. Biologia Celular e Fundamentos de Microscopia.pdf
2. Genética Molecular e Princípios de Hereditariedade.pdf

## Instalação

Clone o repositório:

```bash
git clone https://github.com/SEU-USUARIO/RAG-PDF-Assistant.git
```

Entre na pasta:

```bash
cd RAG-PDF-Assistant
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Configure sua chave da OpenAI no arquivo `.env`:

```text
OPENAI_API_KEY=sua_chave_aqui
```

Coloque os documentos PDF na pasta `base`.

Crie o banco vetorial:

```bash
python create_database.py
```

Execute a aplicação:

```bash
streamlit run app.py
```
