# scrap_linkedin_llm

Experiments for turning job-post content (for example LinkedIn-style HTML or markdown exports) into structured fields using **LangChain**, **Chroma** for retrieval, and **OpenAI** models.

## What it does

The `test/` pipeline loads a markdown document, splits it for embedding, retrieves the chunks most relevant to each question, answers with an LLM, then formats specific goals (company name, location, task list, employee-range bounds, etc.) using guided examples.

## Requirements

- Python 3.10+ recommended  
- An [OpenAI API key](https://platform.openai.com/)

## Setup

```bash
cd test
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Configure your API key where the code reads it (currently `test/constant.py` in the `OPENAI_API_KEY` variable). Prefer environment variables or a local untracked config file in your own fork; do not commit real keys.

## Run

The demo loads the file named in `test/constant.py` as `PATH_TO_EXAMPLE` (by default a sample markdown file under `test/`).

```bash
cd test
python main.py
```

Each goal in `GOALS` is queried in order; after **Company name** is resolved, follow-up queries substitute `this company` with that name.

## Repository layout

| Path | Role |
|------|------|
| `test/main.py` | End-to-end flow: load → retrieve → answer → format per goal |
| `test/loader.py` | Loads markdown via LangChain `TextLoader` |
| `test/preprocessing.py` | Splitting helpers used by retrieval |
| `test/constant.py` | Paths, goals, query templates, examples |
| `get_info.py` (repo root) | Earlier `WebsiteHandler` experiment: HTML → markdown-ish text, Chroma retrieval, `StuffDocumentsChain` |
| `test/*.md`, `*.html` | Sample inputs for development |

## Notes

- **Terms of use**: Scraping or automating LinkedIn may violate their terms. This repo is for research and local experimentation with sample files, not for production scraping.
- **Dependencies**: Pin versions live in `test/requirements.txt` (LangChain 0.1.x, Chroma, OpenAI SDK, etc.).

## Credits

Original experiment notes: Robin & Hernan (`test/README.md`).
