from typing import Optional

import numpy as np


def get_figure_size(width: float, n_rows: int = 1, n_cols: int = 1, height: Optional[float] = None,
                    column_width: float = 479.17036) -> tuple[float, float]:
    figure_width_pt = column_width * width
    inches_per_pt = 1.0 / 72.27  # Convert pt to inches
    golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
    fig_width = figure_width_pt * inches_per_pt  # width in inches
    if height is not None:
        fig_height = height * column_width * inches_per_pt
    else:
        effective_ax_width = fig_width / n_cols
        fig_height = effective_ax_width * golden_mean * n_rows

    return (fig_width, fig_height)
