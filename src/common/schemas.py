from dataclasses import dataclass, field
from typing import Any

@dataclass
class AnalysisResult:
    """Standard structured container for analytical results."""
    analysis_name: str
    status: str
    parameters: dict[str, Any]
    results: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class AnalysisProfile:
    """Configuration describing a reproducible analysis."""
    target: str
    features: list[str]
    confidence_level: float = 0.95

@dataclass
class Experiment:
    """Metadata describing a reproducible modeling experiment."""
    experiment_id: str
    model_name: str
    feature_set: list[str]
    training_strategy: str
    threshold: float
    dataset_version: str
    random_state: int
    metrics: dict[str, Any]
    timestamp: str
