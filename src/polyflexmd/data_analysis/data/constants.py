import numpy as np

RAW_TRAJECTORY_DF_COLUMN_TYPES = {
    "t": np.uint64,  # From 0 to 18_446_744_073_709_551_615
    "id": np.ushort,  # From 0 to 65_535
    "type": np.ubyte,  # From 0 to 255
    "x": np.float32,
    "y": np.float32,
    "z": np.float32,
    "ix": np.short,
    "iy": np.short,
    "iz": np.short
}
