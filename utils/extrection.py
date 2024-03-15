from typing import List, Any


def extract_values_by_index(data: List[List[Any]], index_to_extract: int) -> List[Any]:
    return [sublist[index_to_extract] for sublist in data]
