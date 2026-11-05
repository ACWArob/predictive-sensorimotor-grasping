"""Object sets. Appearance (what the camera sees) vs physics (what the fingers feel).

Embedding = fixed random projection of [log volume, half-sizes, material one-hot] + per-instance
noise: a stand-in for DINOv2 that, like DINOv2, encodes how things LOOK, not what they weigh.
Training objects are honest (density follows material). Test set adds deceptive objects where
appearance lies: the size-weight pair, identical bottles empty vs full, hollow metal, lead-filled plastic.
"""
from __future__ import annotations

import os

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
SHAPES = ["box", "cylinder", "sphere", "capsule", "ellipsoid", "mug"]
_PROJ_SHAPE = np.random.default_rng(4321).normal(0, 1, (len(SHAPES), EMB_DIM))
USE_SHAPES = bool(os.environ.get("PSG_SHAPES"))


def embed(obj: PhysObject, rng) -> np.ndarray:
    sx, sy, sz = obj.half_size
    onehot = np.eye(len(MAT_LIST))[MAT_LIST.index(obj.material)]
    feat = np.concatenate([[np.log(8 * sx * sy * sz) / 3, sx * 30, sy * 30, sz * 30], onehot * 1.5])
    e = feat @ _PROJ
    if obj.shape != "box":                              # the camera sees the shape too (boxes unchanged)
        e = e + 1.5 * _PROJ_SHAPE[SHAPES.index(obj.shape)]
    return e + rng.normal(0, 0.15, EMB_DIM)


def percept(obj: PhysObject, rng) -> Percept:
    return Percept(obj.name, embed(obj, rng), np.array([obj.half_size[0] * 30]))


def make(name, material, half_size, rng, density=None, mu=None, crush=None, shape="box", slosh=0.0):
    from .env import shape_volume
    d0, mu0, c0 = MATERIALS[material]
    d = (density if density is not None else d0) * rng.uniform(0.85, 1.15)
    vol = shape_volume(shape, half_size)
    return PhysObject(name, tuple(half_size), mass=float(d * vol),
                      mu=float(mu if mu is not None else mu0 * rng.uniform(0.9, 1.1)),
                      f_crush=float(crush if crush is not None else c0), material=material,
                      shape=shape, slosh=slosh)


def random_shape_size(rng, shape):
    if shape in ("cylinder", "mug"):
        r = rng.uniform(0.018, 0.034)
        return (r, r, rng.uniform(0.03, 0.055))
    if shape == "sphere":
        r = rng.uniform(0.02, 0.034)
        return (r, r, r)
    if shape == "capsule":
        r = rng.uniform(0.016, 0.03)
        return (r, r, rng.uniform(r + 0.015, 0.055))
    if shape == "ellipsoid":
        return tuple(rng.uniform([0.02, 0.018, 0.025], [0.035, 0.032, 0.05]))
    return random_size(rng)


def _shape_set(prefix, rng, n):
    out = []
    for i in range(n):
        shape = SHAPES[i % len(SHAPES)]
        m = str(rng.choice(MAT_LIST))
        if shape == "mug":
            m = str(rng.choice(["ceramic", "plastic", "metal"]))
        out.append(make(f"{prefix}_{i}_{m}_{shape}", m, random_shape_size(rng, shape), rng, shape=shape))
    return out


def random_size(rng):
    return tuple(rng.uniform([0.015, 0.015, 0.02], [0.035, 0.035, 0.045]))


def train_set(rng, n=48):
    if USE_SHAPES:
        return _shape_set("train", rng, n)
    return [make(f"train_{i}_{m}", m, random_size(rng), rng)
            for i, m in enumerate(rng.choice(MAT_LIST, n))]


def heldout_set(rng, n=30):
    """Unseen but honest objects: Exp-Force's 'unseen objects' metric."""
    if USE_SHAPES:
        return _shape_set("heldout", rng, n)
    return [make(f"heldout_{i}_{m}", m, random_size(rng), rng)
            for i, m in enumerate(rng.choice(MAT_LIST, n))]


def deceptive_set(rng):
    """Appearance lies. Each tuple: (object, what the trick is)."""
    if USE_SHAPES:
        can = (0.033, 0.033, 0.06)
        return [
            (make("can_empty", "metal", can, rng, density=60, shape="cylinder"), "identical cans: empty"),
            (make("can_full_slosh", "metal", can, rng, density=1000, crush=60, shape="cylinder", slosh=0.85),
             "identical cans: full of liquid that sloshes"),
            (make("bottle_half_slosh", "plastic", (0.028, 0.028, 0.05), rng, density=550, crush=30, shape="capsule",
                  slosh=0.8), "half-full bottle, liquid moves"),
            (make("lead_ball", "plastic", (0.026, 0.026, 0.026), rng, density=4000, crush=80, shape="sphere"),
             "looks like a plastic ball, very heavy"),
            (make("mug_of_coins", "ceramic", (0.03, 0.03, 0.045), rng, density=3600, crush=40, shape="mug"),
             "mug full of coins, handle shifts the weight"),
            (make("hollow_metal_egg", "metal", (0.028, 0.024, 0.04), rng, density=200, crush=6, shape="ellipsoid"),
             "looks heavy, light + dents"),
        ]
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
