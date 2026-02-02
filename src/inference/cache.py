"""SQLite-based response caching for resumption and efficiency."""

import hashlib
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from src.models.base import GenerationOutput, UsageStats


class ResponseCache:
    """
    SQLite-based cache for model responses.
    
    Enables:
    - Resumption after crashes
    - Avoiding redundant API calls
    - Reproducibility
    
    Key is hash of (prompt, model_id, generation_params).
    """

    def __init__(
        self,
        cache_path: str | Path,
        enabled: bool = True,
    ):
        self.cache_path = Path(cache_path)
        self.enabled = enabled
        self._conn: Optional[sqlite3.Connection] = None

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.cache_path))
            self._create_tables()
        return self._conn

    def _create_tables(self) -> None:
        """Create cache tables if they don't exist."""
        conn = self._conn
        conn.execute("""
            CREATE TABLE IF NOT EXISTS responses (
                cache_key TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                model_id TEXT NOT NULL,
                params_json TEXT NOT NULL,
                response_text TEXT NOT NULL,
                tokens_json TEXT,
                logprobs_json TEXT,
                usage_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_id ON responses(model_id)
        """)
        conn.commit()

    def _compute_key(
        self,
        prompt: str,
        model_id: str,
        params: dict[str, Any],
    ) -> str:
        """Compute cache key from prompt, model, and params."""
        key_data = {
            "prompt": prompt,
            "model_id": model_id,
            "params": params,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(
        self,
        prompt: str,
        model_id: str,
        params: dict[str, Any],
    ) -> Optional[GenerationOutput]:
        """
        Get cached response if available.
        
        Args:
            prompt: The input prompt.
            model_id: Model identifier.
            params: Generation parameters.
        
        Returns:
            Cached GenerationOutput or None if not found.
        """
        if not self.enabled:
            return None
        
        cache_key = self._compute_key(prompt, model_id, params)
        conn = self._get_conn()
        
        cursor = conn.execute(
            "SELECT response_text, tokens_json, logprobs_json, usage_json FROM responses WHERE cache_key = ?",
            (cache_key,),
        )
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        response_text, tokens_json, logprobs_json, usage_json = row
        
        tokens = json.loads(tokens_json) if tokens_json else []
        logprobs = json.loads(logprobs_json) if logprobs_json else []
        usage_dict = json.loads(usage_json) if usage_json else {}
        
        return GenerationOutput(
            text=response_text,
            tokens=tokens,
            token_logprobs=logprobs,
            hidden_states=None,  # Hidden states are not cached
            usage=UsageStats(**usage_dict) if usage_dict else UsageStats(),
        )

    def put(
        self,
        prompt: str,
        model_id: str,
        params: dict[str, Any],
        output: GenerationOutput,
    ) -> None:
        """
        Store response in cache.
        
        Args:
            prompt: The input prompt.
            model_id: Model identifier.
            params: Generation parameters.
            output: The generation output to cache.
        """
        if not self.enabled:
            return
        
        cache_key = self._compute_key(prompt, model_id, params)
        conn = self._get_conn()
        
        conn.execute(
            """
            INSERT OR REPLACE INTO responses 
            (cache_key, prompt, model_id, params_json, response_text, tokens_json, logprobs_json, usage_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cache_key,
                prompt,
                model_id,
                json.dumps(params, sort_keys=True),
                output.text,
                json.dumps(output.tokens),
                json.dumps(output.token_logprobs),
                json.dumps(asdict(output.usage)),
            ),
        )
        conn.commit()

    def contains(
        self,
        prompt: str,
        model_id: str,
        params: dict[str, Any],
    ) -> bool:
        """Check if a response is cached."""
        if not self.enabled:
            return False
        
        cache_key = self._compute_key(prompt, model_id, params)
        conn = self._get_conn()
        
        cursor = conn.execute(
            "SELECT 1 FROM responses WHERE cache_key = ?",
            (cache_key,),
        )
        return cursor.fetchone() is not None

    def clear(self) -> None:
        """Clear all cached responses."""
        if self._conn is not None:
            self._conn.execute("DELETE FROM responses")
            self._conn.commit()

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        conn = self._get_conn()
        
        cursor = conn.execute("SELECT COUNT(*) FROM responses")
        total_entries = cursor.fetchone()[0]
        
        cursor = conn.execute("SELECT COUNT(DISTINCT model_id) FROM responses")
        unique_models = cursor.fetchone()[0]
        
        return {
            "total_entries": total_entries,
            "unique_models": unique_models,
            "cache_path": str(self.cache_path),
            "enabled": self.enabled,
        }

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "ResponseCache":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


