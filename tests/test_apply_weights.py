import os
import random

from xml_to_note_table.parser import parse as xml_parse
from metricker.apply_weights import apply_weights

ABC_DIR = os.getenv("ABC_DIR")


def test_apply_weights(slow):
    # This test doesn't really "test" anything other than that apply_weights
    #   runs without error on these files.
    paths = [
        os.path.join(ABC_DIR, p)
        for p in os.listdir(ABC_DIR)
        if p.endswith(".mscx")
    ]
    if not slow:
        random.seed(42)
        random.shuffle(paths)
        paths = paths[:10]
    for i, p in enumerate(paths):
        print(f"{i + 1}/{len(paths)}: {p}")
        df = xml_parse(p)
        apply_weights(df)
