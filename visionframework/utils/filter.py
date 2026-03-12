"""
Class-filter utilities shared across detection algorithms.
"""
from __future__ import annotations
from typing import List, Optional, Set, Union


def resolve_filter_ids(
    filter_classes: Optional[List[Union[int, str]]],
    class_names: Optional[Union[List[str], dict]],
) -> Optional[Set[int]]:
    """将 filter_classes（int/str 混合列表）统一解析为 class_id 集合。

    Parameters
    ----------
    filter_classes : list[int | str] | None
        要保留的类别，可以是 id 或名称。
    class_names : list[str] | dict | None
        id → name 映射（list 或 {id: name} dict）。

    Returns
    -------
    set[int] | None
        解析后的 class_id 集合，若无过滤则返回 None。
    """
    if not filter_classes:
        return None
    ids: Set[int] = set()
    name_to_id: dict = {}
    if class_names:
        if isinstance(class_names, dict):
            name_to_id = {v.lower(): k for k, v in class_names.items()}
        else:
            name_to_id = {n.lower(): i for i, n in enumerate(class_names) if n}
    for c in filter_classes:
        if isinstance(c, int):
            ids.add(c)
        elif isinstance(c, str):
            cid = name_to_id.get(c.lower())
            if cid is not None:
                ids.add(cid)
    return ids if ids else None
