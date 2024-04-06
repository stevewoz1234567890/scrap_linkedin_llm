from langchain_openai import OpenAIEmbeddings

def embed(texts:list[str]):
    embeddings_model = OpenAIEmbeddings()
    embeddings = embeddings_model.embed_documents(texts)
    return embeddings