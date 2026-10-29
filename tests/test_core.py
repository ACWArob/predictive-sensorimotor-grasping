"""Fast checks of the core logic (no MuJoCo). Run: python -m pytest tests -q"""
import numpy as np

from psg import ForwardModel, ObjectResidualStore, Percept, PredictiveGraspAgent
from psg.types import LOAD


def _toy(n=200, d=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    Y = np.stack([X[:, 0] * 0.5 + 1.0, -0.5 + 0.1 * X[:, 1]], 1) + rng.normal(0, 0.05, (n, 2))
    return X, Y


def test_predictor_fits_and_std_grows_off_data():
    X, Y = _toy()
    m = ForwardModel(n_features=256).fit(X, Y)
    mu, sd = m.predict(X[:5])
    assert np.abs(mu - Y[:5]).max() < 0.3
    _, sd_far = m.predict(X[:5] * 6)
    assert sd_far.mean() > sd.mean()


def test_one_shot_update_when_uncertain():
    s = ObjectResidualStore()
    s.get("a", np.array([0.5, 0.5]))
    k = s.update("a", LOAD, 1.0)
    assert k > 0.95 and abs(s.entries["a"].r[LOAD] - 1.0) < 0.06


def test_no_leak_between_objects():
    X, Y = _toy()
    agent = PredictiveGraspAgent(ForwardModel().fit(X, Y), mode="predictor+update")
    x = X[0]
    pa, ca = agent.plan(Percept("a", x[:7], x[7:]))
    ca("liftoff", {"load_N": pa.load * 3, "grip_N": 1.0, "t": 0.5})
    pb, _ = agent.plan(Percept("b", x[:7], x[7:]))
    pa2, _ = agent.plan(Percept("a", x[:7], x[7:]))
    assert abs(pb.load - pa.load) / pa.load < 0.05          # b untouched
    assert pa2.load > 2.5 * pa.load                         # a corrected in one grasp
