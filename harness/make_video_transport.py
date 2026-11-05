"""After the grasp: v1 vs v4 vs v5 carrying and placing on the MAGPIE model (real transport_test trials).

    MUJOCO_GL=osmesa python -m harness.make_video_transport  ->  harness/out_transport/psg_after_the_grasp.mp4
"""
from __future__ import annotations

import os
import subprocess
import sys
import zlib

import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import harness.compare_autograsp as H                                    # noqa: E402
import harness.transport_test as TT                                       # noqa: E402
from harness.make_video import FPS, encode, font                         # noqa: E402
from psg import ForwardModel, PredictiveGraspAgent                       # noqa: E402
from sim import env_magpie as E                                          # noqa: E402
from sim.baselines import DeliGraspLike                                  # noqa: E402
from sim.run_experiments import collect                                  # noqa: E402

OUT = os.path.join(ROOT, "harness", "out_transport")
TMP = os.path.join(OUT, "_video_tmp")
W, HH, HEAD = 640, 600, 130
EVERY = 16
SYS = [("v1_hold", "v1  today: grip fixed"), ("v4_hold", "v4  PSG grasp, grip fixed"),
       ("v5_feedforward", "v5  grip follows the planned move")]
SCENES = [(s.split("|")[0], int(s.split("|")[1]), s.split("|")[2], s.split("|")[3]) for s in os.environ.get(
    "SCENES", "S2_deceptive|7|fast|Small dense block, 2nd grasp, fast carry-and-place;"
              "S1_honest_unseen|0|fast|Ordinary metal block, fast carry-and-place").split(";")]


def record():
    H.use_gripper("magpie")
    X, Y, vrows, _ = collect(np.random.default_rng(0))
    model = ForwardModel(seed=0).fit(X, Y)
    clips = {}
    real = E.run_grasp
    for suite, t_want, speed, _ in SCENES:
        trials = H.suites(0)[suite]
        for name, _ in SYS:
            system = DeliGraspLike() if name.startswith("v1") else PredictiveGraspAgent(model, mode="predictor+update", couple=True)
            rng = np.random.default_rng(0 * 7 + zlib.crc32(suite.encode()) % 1000)
            for i, (obj, k) in enumerate(trials):
                if i > t_want:
                    break
                buf, st = [], {}
                if i == t_want:
                    def hook(m, d, info, buf=buf, st=st):
                        if "r" not in st:
                            m.vis.global_.offwidth, m.vis.global_.offheight = W, HH
                            st["r"] = mujoco.Renderer(m, HH, W)
                            c = mujoco.MjvCamera()
                            c.lookat[:] = (0.15, 0, 0.17)
                            c.distance, c.azimuth, c.elevation = 0.78, 140, -14
                            st["c"] = c
                            m.vis.headlight.ambient[:] = (0.45,) * 3
                            m.vis.headlight.diffuse[:] = (0.65,) * 3
                            m.geom_rgba[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "obj")] = (0.35, 0.6, 0.9, 1)
                        if info["k"] < 0:
                            return
                        if info["crushed"]:
                            m.geom_rgba[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "obj")] = (0.9, 0.2, 0.2, 1)
                        if info["k"] % EVERY == 0:
                            st["r"].update_scene(d, st["c"])
                            buf.append((st["r"].render().copy(), info["grip"], info.get("need"), info["k"] >= 10_000))
                    E.run_grasp = lambda *a, hook=hook, **kw: real(*a, **{**kw, "on_step": _chain(kw.get("on_step"), hook)})
                else:
                    E.run_grasp = real
                row = TT.run_trial(name, system, obj, rng, TT.SPEEDS[speed])
                E.run_grasp = real
                if i == t_want:
                    clips[(suite, t_want, name)] = (buf, row, obj)
    return clips


def _chain(a, b):
    def f(m, d, info):
        if a:
            a(m, d, info)
        b(m, d, info)
    return f


def panel(img, label, grip, need, moving, row, final):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 44], fill=(20, 24, 32))
    d.text((14, 9), label, font=font(23, True), fill="white")
    d.rectangle([0, HH - 46, W, HH], fill=(20, 24, 32))
    txt = f"grip {grip:5.1f} N" + (f"   needs {need:4.1f} N now" if (moving and need) else "")
    d.text((14, HH - 38), txt, font=font(22), fill=(230, 230, 230))
    if final:
        if row["task_ok"]:
            tag, col = "PLACED", (40, 170, 90)
        elif row["crushed"]:
            tag, col = "CRUSHED", (220, 50, 50)
        elif not row["held"]:
            tag, col = "DROPPED", (230, 140, 30)
        else:
            tag, col = f"SHIFTED {row['shift_mm']:.0f} mm", (230, 140, 30)
        f = font(28, True)
        w = d.textlength(tag, font=f)
        d.rounded_rectangle([W - w - 34, 58, W - 14, 100], 8, fill=col)
        d.text((W - w - 24, 62), tag, font=f, fill="white")
    return np.asarray(im)


def card(lines, path, secs=4.0):
    im = Image.new("RGB", (W * 3, HH + HEAD), (20, 24, 32))
    d = ImageDraw.Draw(im)
    y = 170
    for txt, sz, bold, col in lines:
        d.text((90, y), txt, font=font(sz, bold), fill=col)
        y += int(sz * 1.6)
    encode([np.asarray(im)] * int(secs * FPS), path)


def main():
    os.makedirs(TMP, exist_ok=True)
    clips = record()
    parts = [os.path.join(TMP, "00.mp4")]
    card([("After the grasp", 64, True, "white"),
          ("The policy carries the object 30 cm and places it. Does the grip survive, and does the object", 30, False, (200, 200, 200)),
          ("stay where the next step expects it? MAGPIE gripper in MuJoCo, real trials, half speed.", 30, False, (200, 200, 200)),
          ("v5 sets grip from the move it is about to make (like people do) and learns per object.", 30, False, (170, 170, 170))],
         parts[0], secs=5)
    for n, (suite, t, speed, cap) in enumerate(SCENES, 1):
        cl = [clips[(suite, t, s)] for s, _ in SYS]
        L = max(len(c[0]) for c in cl)
        frames = []
        for fi in range(L + FPS):
            canvas = Image.new("RGB", (W * 3, HH + HEAD), (245, 245, 242))
            d = ImageDraw.Draw(canvas)
            d.text((24, 18), cap, font=font(40, True), fill=(20, 24, 32))
            d.text((24, 74), f"{suite.split('_', 1)[1]} suite, trial {t + 1}, {speed} move. Same object, start and move for all three.",
                   font=font(25), fill=(80, 80, 80))
            for j, ((buf, row, obj), (_, lab)) in enumerate(zip(cl, SYS)):
                img, g, need, moving = buf[min(fi, len(buf) - 1)]
                canvas.paste(Image.fromarray(panel(img, lab, g, need, moving, row, fi >= len(buf) - 1)), (j * W, HEAD))
            frames.append(np.asarray(canvas))
        if "--check" in sys.argv:
            for q in (0.5, 0.8, 1.0):
                Image.fromarray(frames[int(q * (len(frames) - 1))]).save(os.path.join(TMP, f"check_{n}_{q}.png"))
            continue
        p = os.path.join(TMP, f"{n:02d}.mp4")
        encode(frames, p)
        parts.append(p)
        for (buf, row, obj), (s, _) in zip(cl, SYS):
            print(suite, t, s, {k: row[k] for k in ("task_ok", "held", "crushed", "shift_mm")})
    if "--check" in sys.argv:
        return
    lst = os.path.join(TMP, "list.txt")
    open(lst, "w").write("".join(f"file '{p}'\n" for p in parts))
    final = os.path.join(OUT, "psg_after_the_grasp.mp4")
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                           final[:-4] + ".part.mp4"])
    os.replace(final[:-4] + ".part.mp4", final)
    print("wrote", final)


if __name__ == "__main__":
    main()
