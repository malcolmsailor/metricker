import ast
import logging
import typing as t
import warnings
from numbers import Number

import pandas as pd

from metricker.meter import Meter

LOGGER = logging.getLogger(__name__)


def apply_weights(df: pd.DataFrame, min_weight: t.Union[int, Number] = -3) -> None:
    """

    df is modified in place.

    The highest weight is 2.

    Required columns of df are "type", "onset", "release", and "other".

    For `min_weight`, see the documentation of Meter.

    Normally we expect "time_signatures" to precede "bar" events.

    >>> df = pd.DataFrame(
    ...     {
    ...         "pitch": [float("nan"), 0, 60, 64, 67, 72],
    ...         "onset": [0, 0, 0, 1.0, 1.5, 2.25],
    ...         "release": [float("nan"), 3, 1, 1.5, 2.0, 3.0],
    ...         "other": [{"numerator": 3, "denominator": 4}] + [float("nan")] * 5,
    ...         "type": ["time_signature", "bar"] + ["note"] * 4,
    ...     }
    ... )
    >>> apply_weights(df)
    >>> df[["pitch", "onset", "release", "other", "type", "weight"]]
       pitch  onset  release                               other            type  weight
    0    NaN   0.00      NaN  {'numerator': 3, 'denominator': 4}  time_signature     NaN
    1    0.0   0.00      3.0                                 NaN             bar     NaN
    2   60.0   0.00      1.0                                 NaN            note     1.0
    3   64.0   1.00      1.5                                 NaN            note     0.0
    4   67.0   1.50      2.0                                 NaN            note    -1.0
    5   72.0   2.25      3.0                                 NaN            note    -2.0

    Incomplete measures containing time-signature changes are interpreted as
    pickup measures.

    >>> df = pd.DataFrame(
    ...     {
    ...         "pitch": [float("nan"), 0, 60, 0, 64, 67, 0, 72, 0, 67, 64],
    ...         "onset": [0, 0, 0, 1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0],
    ...         "release": [float("nan"), 1, 1, 3, 2, 3, 4, 4, 6, 5, 6],
    ...         "other": [{"numerator": 3, "denominator": 4}] + [float("nan")] * 10,
    ...         "type": [
    ...             "time_signature",
    ...             "bar",
    ...             "note",
    ...             "bar",
    ...             "note",
    ...             "note",
    ...             "bar",
    ...             "note",
    ...             "bar",
    ...             "note",
    ...             "note",
    ...         ],
    ...     }
    ... )
    >>> apply_weights(df)
    >>> df[["pitch", "onset", "release", "other", "type", "weight"]]
        pitch  onset  release                               other            type  weight
    0     NaN    0.0      NaN  {'numerator': 3, 'denominator': 4}  time_signature     NaN
    1     0.0    0.0      1.0                                 NaN             bar     NaN
    2    60.0    0.0      1.0                                 NaN            note     0.0
    3     0.0    1.0      3.0                                 NaN             bar     NaN
    4    64.0    1.0      2.0                                 NaN            note     1.0
    5    67.0    2.0      3.0                                 NaN            note     0.0
    6     0.0    3.0      4.0                                 NaN             bar     NaN
    7    72.0    3.0      4.0                                 NaN            note     0.0
    8     0.0    4.0      6.0                                 NaN             bar     NaN
    9    67.0    4.0      5.0                                 NaN            note     1.0
    10   64.0    5.0      6.0                                 NaN            note     0.0

    The score doesn't have to start with a barline but if it doesn't then we
    assume that it begins with a full measure (rather than a pickup) which
    could lead to the entire subsequent piece being misaligned.
    """

    # Find mid-measure time signatures (time signatures that don't co-occur
    #   with a barline)
    bars = df[df["type"] == "bar"]
    time_sigs = df[df["type"] == "time_signature"]
    merged = time_sigs.merge(bars[["onset"]], on="onset", how="left", indicator=True)
    time_sigs_wo_bars = merged[merged["_merge"] == "left_only"]

    # drop _merge_ column from time_sigs_wo_bars
    time_sigs_wo_bars = time_sigs_wo_bars.drop(columns="_merge")
    for _, time_sig in time_sigs_wo_bars.iterrows():
        breakpoint()
        LOGGER.warning(
            "mid-measure time-signature not supported, time signature will take "
            f"effect at next measure: {time_sig}"
        )

    offset = 0
    next_meter = None
    meter = None
    # Previously I initialized bar_onset to None, but we want to handle the case
    # where the initial barline is omitted
    bar_dur = None
    weights = []
    for _, row in df.iterrows():
        if row.type == "bar":
            bar_dur = row.release - row.onset
            if next_meter is not None:
                offset = next_meter.bar_dur - bar_dur
                if offset < 0:
                    warnings.warn(
                        "bar is longer than time signature, skipping time signature"
                    )
                else:
                    meter = next_meter
                next_meter = None
        elif row.type == "time_signature":
            ts = row.other
            if isinstance(ts, str):
                ts = ast.literal_eval(ts)
            ts_str = f"{ts['numerator']}/{ts['denominator']}"
            next_meter = Meter(ts_str, min_weight)
            # meter = candidate_meter
        elif row.type == "note":
            assert meter is not None
            weight = meter(offset + row.onset)
            weights.append(weight)
    df["weight"] = float("nan")
    df.loc[df.type == "note", "weight"] = weights
