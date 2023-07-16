from dataclasses import dataclass

@dataclass
class Paths:
    raw: str
    assay: str
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