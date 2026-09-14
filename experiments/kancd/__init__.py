# coding: utf-8
"""
KaNCD models with channel-wise extensions.

默认导出：
    - KaNCD : Channel-wise Gating 版本（KaNCD_channel.py）

可选导出：
    - BaseKaNCD : 原始 KaNCD（无 channel）
"""

# ===== Channel-wise 版本（主推）=====
from .KaNCD_channel import KaNCD

# ===== 原始版本（用于消融/对比）=====
from .KaNCD import KaNCD as BaseKaNCD

__all__ = [
    "KaNCD",
    "BaseKaNCD",
]
