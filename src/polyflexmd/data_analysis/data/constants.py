import numpy as np

RAW_TRAJECTORY_DF_COLUMN_TYPES = {
    "t": np.uint64,  # From 0 to 18_446_744_073_709_551_615
    "id": np.ushort,  # From 0 to 65_535
    "type": np.ubyte,  # From 0 to 255
    "x": float,
    "y": float,
    "z": float,
    "ix": np.short,
    "iy": np.short,
    "iz": np.short
}
