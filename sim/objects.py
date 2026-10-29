"""Object sets. Appearance (what the camera sees) vs physics (what the fingers feel).

Embedding = fixed random projection of [log volume, half-sizes, material one-hot] + per-instance
noise: a stand-in for DINOv2 that, like DINOv2, encodes how things LOOK, not what they weigh.
Training objects are honest (density follows material). Test set adds deceptive objects where
appearance lies: the size-weight pair, identical bottles empty vs full, hollow metal, lead-filled plastic.
"""
from __future__ import annotations

import numpy as np

from psg.types import Percept
from .env import PhysObject

# looks-like -> (density kg/m^3 of the box, friction, crush limit N)
MATERIALS = {
    "wood":    (600, 0.50, 60),
    "plastic": (450, 0.40, 22),
    "metal":   (2500, 0.35, 80),
    "foam":    (60, 0.80, 5),
    "ceramic": (1400, 0.60, 30),
    "rubber":  (1100, 0.90, 40),
}
MAT_LIST = list(MATERIALS)
EMB_DIM = 64
_PROJ = np.random.default_rng(1234).normal(0, 1, (4 + len(MAT_LIST), EMB_DIM))


def embed(obj: PhysObject, rng) -> np.ndarray:
    sx, sy, sz = obj.half_size
    onehot = np.eye(len(MAT_LIST))[MAT_LIST.index(obj.material)]
    feat = np.concatenate([[np.log(8 * sx * sy * sz) / 3, sx * 30, sy * 30, sz * 30], onehot * 1.5])
    return feat @ _PROJ + rng.normal(0, 0.15, EMB_DIM)


def percept(obj: PhysObject, rng) -> Percept:
    return Percept(obj.name, embed(obj, rng), np.array([obj.half_size[0] * 30]))


def make(name, material, half_size, rng, density=None, mu=None, crush=None):
    d0, mu0, c0 = MATERIALS[material]
    d = (density if density is not None else d0) * rng.uniform(0.85, 1.15)
    vol = 8 * np.prod(half_size)
    return PhysObject(name, tuple(half_size), mass=float(d * vol),
                      mu=float(mu if mu is not None else mu0 * rng.uniform(0.9, 1.1)),
                      f_crush=float(crush if crush is not None else c0), material=material)


def random_size(rng):
    return tuple(rng.uniform([0.015, 0.015, 0.02], [0.035, 0.035, 0.045]))


def train_set(rng, n=48):
    return [make(f"train_{i}_{m}", m, random_size(rng), rng)
            for i, m in enumerate(rng.choice(MAT_LIST, n))]


def heldout_set(rng, n=30):
    """Unseen but honest objects: Exp-Force's 'unseen objects' metric."""
    return [make(f"heldout_{i}_{m}", m, random_size(rng), rng)
            for i, m in enumerate(rng.choice(MAT_LIST, n))]


def deceptive_set(rng):
    """Appearance lies. Each tuple: (object, what the trick is)."""
    big, small = (0.035, 0.035, 0.045), (0.016, 0.016, 0.022)
    bottle = (0.025, 0.025, 0.045)
    return [
        (make("sw_big_hollow", "wood", big, rng, density=90), "size-weight: big but light"),
        (make("sw_small_dense", "wood", small, rng, density=7000, mu=0.5), "size-weight: small but heavy"),
        (make("bottle_empty", "plastic", bottle, rng, density=80), "identical look, empty"),
        (make("bottle_full", "plastic", bottle, rng, density=1000, crush=40), "identical look, full"),
        (make("hollow_metal", "metal", (0.03, 0.03, 0.04), rng, density=250, crush=6), "looks heavy, light + dents"),
        (make("lead_plastic", "plastic", (0.025, 0.025, 0.03), rng, density=3500, crush=60), "looks light, very heavy"),
    ]
