import typing as t
import warnings
from numbers import Number

import pandas as pd

from metricker.meter import Meter


def apply_weights(
    df: pd.DataFrame, min_weight: t.Union[int, Number] = -3
) -> None:
    """
    For `min_weight`, see the documentation of Meter.

    >>> df = pd.DataFrame({
    ...     "pitch": [0, float("nan"), 60, 64, 67, 72],
    ...     "onset": [0, 0, 0, 1.0, 1.5, 2.25],
    ...     "release": [3, float("nan"), 1, 1.5, 2.0, 3.0],
    ...     "other": [float("nan"),
    ...               {"numerator": 3, "denominator": 4}] + [float("nan")] * 4,
    ...     "type": ["bar", "time_signature"] + ["note"] * 4,
    ... })
    >>> apply_weights(df)
    >>> df
       pitch  onset  release                               other            type  weight
    0    0.0   0.00      3.0                                 NaN             bar     NaN
    1    NaN   0.00      NaN  {'numerator': 3, 'denominator': 4}  time_signature     NaN
    2   60.0   0.00      1.0                                 NaN            note     1.0
    3   64.0   1.00      1.5                                 NaN            note     0.0
    4   67.0   1.50      2.0                                 NaN            note    -1.0
    5   72.0   2.25      3.0                                 NaN            note    -2.0

    Incomplete measures containing time-signature changes are interpreted as
    pickup measures.

    >>> df = pd.DataFrame({
    ...     "pitch": [0, float("nan"), 60, 0, 64, 67, 0, 72, 0, 67, 64],
    ...     "onset": [0, 0, 0, 1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0],
    ...     "release": [1, float("nan"), 1, 3, 2, 3, 4, 4, 6, 5, 6],
    ...     "other": [float("nan"),
    ...               {"numerator": 3, "denominator": 4}] + [float("nan")] * 9,
    ...     "type": ["bar", "time_signature", "note", "bar", "note", "note", "bar", "note", "bar", "note", "note"],
    ... })
    >>> apply_weights(df)
    >>> df
        pitch  onset  release                               other            type  weight
    0     0.0    0.0      1.0                                 NaN             bar     NaN
    1     NaN    0.0      NaN  {'numerator': 3, 'denominator': 4}  time_signature     NaN
    2    60.0    0.0      1.0                                 NaN            note     0.0
    3     0.0    1.0      3.0                                 NaN             bar     NaN
    4    64.0    1.0      2.0                                 NaN            note     1.0
    5    67.0    2.0      3.0                                 NaN            note     0.0
    6     0.0    3.0      4.0                                 NaN             bar     NaN
    7    72.0    3.0      4.0                                 NaN            note     0.0
    8     0.0    4.0      6.0                                 NaN             bar     NaN
    9    67.0    4.0      5.0                                 NaN            note     1.0
    10   64.0    5.0      6.0                                 NaN            note     0.0
    """
    offset = 0
    meter = None
    bar_onset = None
    weights = []
    for _, row in df.iterrows():
        if row.type == "bar":
            bar_onset = row.onset
            bar_dur = row.release - row.onset
        elif row.type == "time_signature":
            if row.onset != bar_onset:
                warnings.warn(
                    "mid-measure time-signature changes are not supported, skipping"
                )
                continue
            ts_str = f"{row.other['numerator']}/{row.other['denominator']}"
            candidate_meter = Meter(ts_str, min_weight)
            offset = candidate_meter.bar_dur - bar_dur
            if offset < 0:
                warnings.warn(
                    "bar is longer than time signature, skipping time signature"
                )
                continue
            meter = candidate_meter
        elif row.type == "note":
            weight = meter(offset + row.onset)
            weights.append(weight)
    df["weight"] = float("nan")
    df.loc[df.type == "note", "weight"] = weights
