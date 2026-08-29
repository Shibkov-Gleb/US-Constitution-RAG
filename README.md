# U.S. Constitution RAG CLI

A deliberately small first RAG project using **LlamaIndex**, **Chroma**, and an OpenAI model.

It has two clear steps:

1. **Retrieve**: turn a question into an embedding and find the most relevant Constitution chunks in Chroma.
2. **Answer**: send only those chunks and the question to the LLM. If no chunk is sufficiently relevant, the app declines to answer.

## What you need

- Python 3.10 or newer (3.11 recommended)
- An OpenAI API key

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and replace `your_openai_api_key_here` with your key. Do not commit this file.

## Build the vector database

```powershell
python app.py index
```

This reads `data/us_constitution.txt`, splits it into small chunks, creates embeddings, and saves them under `chroma_db/`. Run it again after changing the source document.

## Chat with the Constitution

```powershell
python app.py chat
```

Try:

```text
What are the requirements to be President?
What does the First Amendment protect?
Who won the 2024 World Series?
```

The last question should be declined because it is not supported by the document.

## Project map

```text
app.py                       CLI and the two RAG steps
data/us_constitution.txt     Local source document
chroma_db/                   Created locally after indexing (not committed)
```

## A useful first experiment

`MIN_RELEVANCE_SCORE` in `app.py` controls when a retrieved chunk is considered relevant. Start with the included value, then try unrelated and Constitution questions. Increase it if the app answers unrelated questions; lower it if it rejects good questions too easily.

This is an educational demo, not legal advice. The included local transcript is adapted from the public-domain U.S. Constitution; for research, check the official [National Archives transcription](https://www.archives.gov/founding-docs/constitution-transcript) and its [Amendments 11–27 page](https://www.archives.gov/founding-docs/amendments-11-27).
