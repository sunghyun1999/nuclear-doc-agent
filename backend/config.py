from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    gemini_api_key: str
    chroma_persist_dir: str = "./chroma_data"
    chunk_size: int = 800
    chunk_overlap: int = 200
    top_k: int = 5
    embedding_model: str = "all-MiniLM-L6-v2"
    gemini_model: str = "gemini-3.8-flash"

    model_config = {"env_file": ".env"}


settings = Settings()
