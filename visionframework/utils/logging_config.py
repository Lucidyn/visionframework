"""
VisionFramework 日志：默认安静（WARNING），可通过环境变量打开 INFO/DEBUG。

环境变量（任选其一，优先 ``VISIONFRAMEWORK_LOG_LEVEL``）::

    VISIONFRAMEWORK_LOG_LEVEL=INFO
    VF_LOG_LEVEL=DEBUG

有效取值：``DEBUG``、``INFO``、``WARNING``、``ERROR``、``CRITICAL``（大小写不敏感）。
非法取值时回退为 ``WARNING``。

仅作用于名为 ``visionframework`` 及其子 logger 的层级，不会修改根 logger。
"""

from __future__ import annotations

import logging
import os
from typing import Optional

_CONFIGURED = False


def configure_visionframework_logging(level: Optional[int] = None) -> int:
    """将 ``visionframework`` 根 logger 设为指定级别；未传则从环境变量解析。

    若未显式传入 *level*，且此前已配置过，则不再重复读取环境变量（幂等）。

    Returns
    -------
    int
        生效的 logging 级别数值。
    """
    global _CONFIGURED
    root = logging.getLogger("visionframework")

    if level is None and _CONFIGURED:
        return root.getEffectiveLevel()

    if level is None:
        raw = os.environ.get("VISIONFRAMEWORK_LOG_LEVEL") or os.environ.get("VF_LOG_LEVEL")
        if raw is None or not str(raw).strip():
            level = logging.WARNING
        else:
            name = str(raw).strip().upper()
            resolved = getattr(logging, name, None)
            level = resolved if isinstance(resolved, int) else logging.WARNING
    root.setLevel(level)
    _CONFIGURED = True
    return level


def reset_visionframework_logging() -> None:
    """测试用：清除「已配置」标记，并将包 logger 恢复为 NOTSET。"""
    global _CONFIGURED
    _CONFIGURED = False
    logging.getLogger("visionframework").setLevel(logging.NOTSET)
