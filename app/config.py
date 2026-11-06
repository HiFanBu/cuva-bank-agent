from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "fake")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1")
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    bedrock_model_id: str = os.getenv(
        "BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"
    )
    db_path: str = os.getenv("DB_PATH", "./data/cuva.db")
    pending_action_ttl_seconds: int = int(
        os.getenv("PENDING_ACTION_TTL_SECONDS", "600")
    )
    max_agent_steps: int = int(os.getenv("MAX_AGENT_STEPS", "6"))
    max_json_repair_retries: int = 2
    max_payment_amount_cents: int = 10_000_000  # $100,000 safety ceiling
    semantic_cache_enabled_intents: tuple[str, ...] = ("policy_question",)


settings = Settings()
