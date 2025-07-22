from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class SampleParams:
    max_tokens: int = 100
    temperature: float = 0.5
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: List[str] = field(default_factory=list)
    n: int = 1

    def to_dict(self) -> dict:
        return asdict(self)
