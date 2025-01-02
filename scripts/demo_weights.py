from fractions import Fraction as F

from metricker.meter import Meter, TimeClass

# NOTE_VALUES = {
#     "Brev": 2.0,
#     "Whole": 1.0,
#     "Half": 0.5,
#     "Quarter": 0.25,
#     "Eighth": 0.125,
#     "Sixteenth": 0.0625,
# }

if __name__ == "__main__":

    for ts in TimeClass.available_time_signatures:
        meter = Meter(ts, bar_has_max_weight=True, max_weight=1)
        # TODO: (Malcolm 2024-06-20)
        print(f"Meter: {ts}")
        print(
            f"{'Super-beat weight':>20}: {meter.superbeat_weight:<5}  {'Super-beat dur':>20}: {str(meter.superbeat_dur):<5}"
        )
        print(
            f"{'Beat weight':>20}: {meter.beat_weight:<5}  {'Beat dur':>20}: {str(meter.beat_dur):<5}"
        )
        print(
            f"{'Semi-beat weight':>20}: {meter.semibeat_weight:<5}  {'Semi-beat dur':>20}: {str(meter.semibeat_dur):<5}"
        )
        demisemibeat = meter.weight_to_grid[-2]
        weights_result = meter.weights_between(demisemibeat, 0, meter.bar_dur)
        onsets = [x["onset"] for x in weights_result]  # type:ignore
        weights = [x["weight"] for x in weights_result]  # type:ignore
        print("".join([f"{str(o):>8}" for o in onsets]))
        print("".join([f"{str(w):>8}" for w in weights]))
