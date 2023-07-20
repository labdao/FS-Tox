from dataclasses import dataclass


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
class AssayConfig:
    paths: Paths
    files: Files
    params: Params