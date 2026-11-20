"""Windowed full-chain MSA features for the frozen v3 projection control."""

WINDOW_COLUMNS = 512
WINDOW_STRIDE = 256
WINDOW_POLICY = "aligned_columns_512_stride_256_right_aligned_tail_equal_mean_v1"

__all__ = ["WINDOW_COLUMNS", "WINDOW_STRIDE", "WINDOW_POLICY"]
