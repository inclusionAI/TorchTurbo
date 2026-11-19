"""Vendored Huawei CSA2 compressor operator (PyPTO, customer release V3).

Forward and backward fused kernels for the DeepSeek v4.1 flash packed-KV
compressor. The three implementation files and README are carried verbatim
from the vendor delivery; the package-internal relative imports fall back to
flat top-level imports, so both this package layout and the vendor flat
layout work. V3 adds the TH (BS-merged) kernels: pass a 2D ``(t, hidden)``
token stream plus an int32 ``cu_seqlens`` prefix sum and every packed segment
is handled inside one kernel launch, forward and backward.

Environment: pto-isa run package + ``pypto>=0.3.0`` wheel; optionally
``ASCEND_PTO_ISA_PATH=/usr/local/Ascend/cann/pto_isa``. ``TILE_FWK_DEVICE_ID``
selects the device (default 0).
"""

from .CSA2_compressor import CSA2Compressor, CSA2CompressorFunction, csa2_compressor

__all__ = ["CSA2Compressor", "CSA2CompressorFunction", "csa2_compressor"]
