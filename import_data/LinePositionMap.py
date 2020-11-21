# from typing import TypedDict
from typing_extensions import TypedDict
from import_data import LineData


class LinePositionMap(TypedDict):
    line_id: int
    line_data: LineData.LineData
