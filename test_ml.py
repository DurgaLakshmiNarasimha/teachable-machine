import os
import tempfile

import numpy as np
from PIL import Image

from ml import TeachableMachine, load_folder


def make(color, seed, stripes=False):
    rng = np.random.default_rng(seed)
    a = np.clip(np.array(color) + rng.integers(-25, 25, (64, 64, 3)), 0, 255).astype("uint8")
    if stripes:
        a[::8] = 255
    return Image.fromarray(a)


def test_end_to_end():
    d = tempfile.mkdtemp()
    for cls, col, st in [("red", (220, 30, 30), False), ("blue", (30, 30, 220), False), ("striped", (120, 120, 120), True)]:
        os.makedirs(f"{d}/{cls}")
        for i in range(6):
            make(col, i, st).save(f"{d}/{cls}/{i}.png")
    tm = TeachableMachine()
    stats = tm.train(load_folder(d))
    assert stats["cv_accuracy"] >= 0.9, stats
    assert list(tm.predict(make((210, 40, 40), 99)))[0] == "red"
    assert list(tm.predict(make((40, 40, 230), 98)))[0] == "blue"
    assert list(tm.predict(make((120, 120, 120), 97, True)))[0] == "striped"
    p = f"{d}/m.joblib"; tm.save(p)
    assert list(TeachableMachine.load(p).predict(make((210, 40, 40), 5)))[0] == "red"


if __name__ == "__main__":
    test_end_to_end(); print("OK")
