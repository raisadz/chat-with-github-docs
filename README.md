# Chat with Polars docs

The project contains an implementation of a conversational application that allows you to chat with [Polars documentation](https://github.com/pola-rs/polars). Polars is a modern dataframe library that has multiple advantages over pandas in terms of performance, memory-efficiency, supported data types, and expressiveness of the API. However, because Polars is a relatively new library, many foundational LLM models do not have access to its syntax. In addition, Polars improves its API by adding and modifying its functions. In order to get access to the latest API, this project retrieves this information from the official GitHub page. Apart from the official Polars documentation, [Modern Polars book](https://github.com/kevinheavey/modern-polars.git) and the top 50 most-voted stackoverflow questions and answers were used to create the RAG. It is implemented using [Langchain LCEL](https://python.langchain.com/docs/expression_language/) that utilises a chat memory and streams the response to a user.

## Installation

Install python 3.11. Clone the repo and inside the folder create a virtual env and install the requirements:
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -U pip
pip install uv
uv pip install -r requirements.txt
```

### Saving secrets

The project uses OPENAI gpt-3.5-turbo model and Pinecone as a vector database. You need to create a file `.streamlit/secrets.toml` and add your API keys to it:
```
OPENAI_API_KEY="YOUR OPENAI API KEY"
PINECONE_API_KEY="YOUR PINECONE API KEY"
```

## Running the project

To create GitHub embeddings and save them to the Pinecode index `polars-docs`:
```bash
python github_embeddings.py
```
To add stackoverflow embeddings to the index:
```bash
python stackoverflow_embeddings.py
```
To run the streamlit app:
```bash
streamlit run streamlit_app.py
```
