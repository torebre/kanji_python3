from dataclasses import dataclass


@dataclass
class Line:
    start_x: int
    start_y: int
    stop_x: int
    stop_y: int
