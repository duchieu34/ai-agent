from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..models import DatasetContext


class DomainAdapter(ABC):
    mode: str
    id_label: str
    entity_label: str

    def __init__(self, max_candidate_pool: int) -> None:
        self.max_candidate_pool = max_candidate_pool

    @abstractmethod
    def load(self, dataset_path: Path, dataset_key: str) -> DatasetContext:
        raise NotImplementedError

    @abstractmethod
    def is_valid_id(self, value: str) -> bool:
        raise NotImplementedError
