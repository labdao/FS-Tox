from dataclasses import dataclass, field


@dataclass
class Paths:
    raw: str
    assay: str
    assay_id: str
    feature: str
    model: str
    prediction: str
    score: str

@dataclass
class Files:
    identifier: str

@dataclass
class Params:
    dataset: str
    assay_size: int
    support_set_size: int
    feature: str
    model: str
    test_prob: float

@dataclass
class XGBoostParams:
    eta: list[float] = field(default_factory=list)
    gamma: list[float] = field(default_factory=list)
    max_depth: list[int] = field(default_factory=list)

@dataclass
class AssayConfig:
    paths: Paths
    files: Files
    params: Params
    xgboost: XGBoostParams