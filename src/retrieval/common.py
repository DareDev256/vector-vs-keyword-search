from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RetrievalResult:
    doc_id: str
    score: float


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        ...

