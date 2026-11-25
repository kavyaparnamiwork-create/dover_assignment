from dataclasses import dataclass


@dataclass
class Word:
    text: str
    conf: float  # 0..1
    x: int
    y: int
    w: int
    h: int  # box in working-image coordinates

    @property
    def cy(self):
        return self.y + self.h / 2


@dataclass
class Proposal:
    key: str
    value: str
