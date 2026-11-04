"""Render v1 / v2 / v3 side by side on the harness's own trials (seed 0), replayed exactly.

    MUJOCO_GL=osmesa python -m harness.make_video            ->  harness/out/psg_v1_v2_v3.mp4
    MUJOCO_GL=osmesa python -m harness.make_video --magpie   ->  harness/out_magpie/psg_v1_v2_v3_magpie.mp4
"""
from __future__ import annotations

import os
import subprocess
import sys
import zlib

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import harness.compare_autograsp as H                                    # noqa: E402
from psg import ForwardModel                                             # noqa: E402
from sim import env                                                      # noqa: E402
from sim.run_experiments import collect                                  # noqa: E402

W, HH, HEAD, FPS, EVERY = 640, 600, 120, 30, 16        # 16 ms sim per frame -> ~half speed
SYS = [("v1_deligrasp", "v1  DeliGrasp (current)"), ("v2_psg", "v2  PSG"), ("v3_psg_couple", "v3  PSG + coupling")]
# (suite, trial, caption)
SCENES = [("S2_deceptive", 1, "Small but dense block, grasp 1: looks light, is heavy"),
          ("S2_deceptive", 4, "Hollow metal, grasp 1: looks heavy, is light and fragile"),
          ("S2_deceptive", 10, "Hollow metal, grasp 2: PSG has corrected its prediction"),
          ("S4_fragile", 0, "Thin cup that looks sturdy, grasp 1")]
GRIPPER = "magpie" if "--magpie" in sys.argv else "box"
OUT = os.path.join(ROOT, "harness", "out_magpie" if GRIPPER == "magpie" else "out")
TMP = os.path.join(OUT, "_video_tmp")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(sz, bold=False):
    try:
        return ImageFont.truetype(FONTB if bold else FONT, sz)
    except OSError:
        return ImageFont.load_default()


def record():
    """Replay seed 0 exactly as the harness does; keep frames for the chosen trials."""
    H.use_gripper(GRIPPER)
    want = {(s, t) for s, t, _ in SCENES}
    frames = {}                                     # (suite, trial, system) -> (list[np.ndarray], row)
    X, Y, vrows, _ = collect(np.random.default_rng(0))
    model = ForwardModel(seed=0).fit(X, Y)
    real = H.run_grasp
    for suite, trials in H.suites(0).items():
        if suite not in {s for s, _ in want}:
            continue
        for name, _ in SYS:
            system = H.new_system(name, model, vrows)
            rng = np.random.default_rng(0 * 7 + zlib.crc32(suite.encode()) % 1000)
            for i, (obj, k) in enumerate(trials):
                buf = []
                if (suite, i) in want:
                    state = {}

                    def hook(m, d, info, buf=buf, state=state):
                        if "r" not in state:
                            m.vis.global_.offwidth, m.vis.global_.offheight = W, HH
                            state["r"] = mujoco.Renderer(m, HH, W)
                            cam = mujoco.MjvCamera()
                            if GRIPPER == "magpie":
                                cam.lookat[:] = (0, 0, 0.15)
                                cam.distance, cam.azimuth, cam.elevation = 0.55, 200, -15
                            else:
                                cam.lookat[:] = (0, 0, 0.10)
                                cam.distance, cam.azimuth, cam.elevation = 0.48, 120, -18
                            m.vis.headlight.ambient[:] = (0.45, 0.45, 0.45)
                            m.vis.headlight.diffuse[:] = (0.65, 0.65, 0.65)
                            state["cam"] = cam
                            m.geom_rgba[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "obj")] = (0.35, 0.6, 0.9, 1)
                            m.geom_rgba[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table")] = (0.85, 0.85, 0.82, 1)
                        if info["crushed"]:
                            m.geom_rgba[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "obj")] = (0.9, 0.2, 0.2, 1)
                        if info["k"] % EVERY == 0:
                            state["r"].update_scene(d, state["cam"])
                            buf.append((state["r"].render().copy(), info["grip"], info["crushed"], info["slipped"]))

                    H.run_grasp = lambda o, f, cb, hook=hook: real(o, f, cb, on_step=hook)
                else:
                    H.run_grasp = real
                row = H.do_grasp(system, obj, rng)
                if buf:
                    frames[(suite, i, name)] = (buf, row)
    H.run_grasp = real
    return frames


def panel(img, label, grip, need, crushed, slipped, final=None):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 44], fill=(20, 24, 32))
    d.text((14, 9), label, font=font(24, True), fill="white")
    txt = f"grip {grip:5.1f} N   ({grip / need:4.1f}× needed)"
    d.rectangle([0, HH - 46, W, HH], fill=(20, 24, 32))
    d.text((14, HH - 38), txt, font=font(22), fill=(230, 230, 230))
    tag = None
    if crushed:
        tag, col = "CRUSHED", (220, 50, 50)
    elif final == "dropped":
        tag, col = "DROPPED", (230, 140, 30)
    elif final == "ok":
        tag, col = "HELD", (40, 170, 90)
    elif slipped:
        tag, col = "slip → reflex", (230, 140, 30)
    if tag:
        f = font(30, True)
        w = d.textlength(tag, font=f)
        d.rounded_rectangle([W - w - 34, 58, W - 14, 102], 8, fill=col)
        d.text((W - w - 24, 63), tag, font=f, fill="white")
    return np.asarray(im)


def compose(panels, caption, sub):
    canvas = Image.new("RGB", (W * 3, HH + HEAD), (245, 245, 242))
    d = ImageDraw.Draw(canvas)
    d.text((24, 18), caption, font=font(40, True), fill=(20, 24, 32))
    d.text((24, 72), sub, font=font(26), fill=(80, 80, 80))
    for j, p in enumerate(panels):
        canvas.paste(Image.fromarray(p), (j * W, HEAD))
    for j in (1, 2):
        d.line([(j * W, HEAD), (j * W, HEAD + HH)], fill=(245, 245, 242), width=4)
    return np.asarray(canvas)


def encode(frames, path):
    tmp = path[:-4] + ".part.mp4"
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
                          "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    assert p.wait() == 0
    os.replace(tmp, path)


def card(lines, path, secs=4.0):
    im = Image.new("RGB", (W * 3, HH + HEAD), (20, 24, 32))
    d = ImageDraw.Draw(im)
    y = 150
    for txt, sz, bold, col in lines:
        d.text((90, y), txt, font=font(sz, bold), fill=col)
        y += int(sz * 1.6)
    encode([np.asarray(im)] * int(secs * FPS), path)


def main(check_only=False):
    os.makedirs(TMP, exist_ok=True)
    frames = record()
    parts = []
    card([("Predictive Sensorimotor Grasping", 64, True, "white"),
          ("v1 current AutoGrasp  vs  v2 PSG  vs  v3 PSG + grip-load coupling", 34, False, (200, 200, 200)),
          ("MuJoCo, " + ("MAGPIE gripper model (HAMS CL_Assets), " if GRIPPER == "magpie" else "") + "same object, seed and percept noise for all three", 30, False, (160, 160, 160)),
          ("Real harness trials (seed 0), replayed. Shown at half speed.", 30, False, (160, 160, 160))],
         os.path.join(TMP, "00.mp4"))
    parts.append(os.path.join(TMP, "00.mp4"))
    for n, (suite, t, cap) in enumerate(SCENES, 1):
        clips = [frames[(suite, t, s)] for s, _ in SYS]
        need = clips[0][1]["f_need"]
        sub = f"{suite.split('_', 1)[1]} suite, trial {t + 1}.  Grip it actually needs: {need:.2f} N"
        L = max(len(c[0]) for c in clips)
        out = []
        for fi in range(L + FPS):                                   # hold last frame 1 s with the verdict
            ps = []
            for (buf, row), (_, lab) in zip(clips, SYS):
                img, g, cr, sl = buf[min(fi, len(buf) - 1)]
                final = None
                if fi >= len(buf) - 1:
                    final = "ok" if row["success"] else ("crushed" if row["crushed"] else "dropped")
                ps.append(panel(img, lab, g, need, cr, sl, final))
            out.append(compose(ps, cap, sub))
        if check_only:
            for q in (0.35, 0.7, 1.0):
                Image.fromarray(out[int(q * (len(out) - 1))]).save(os.path.join(TMP, f"check_{n}_{q}.png"))
            continue
        p = os.path.join(TMP, f"{n:02d}.mp4")
        encode(out, p)
        parts.append(p)
        for (buf, row), (s, _) in zip(clips, SYS):
            print(suite, t, s, "success", row["success"], "crushed", row["crushed"], "peak", row["peak_excess"])
    if check_only:
        return
    import pandas as pd
    t = pd.read_csv(os.path.join(OUT, "trials.csv"))
    sr = lambda su, sy: t[(t.suite == su) & (t.system == sy)].success.mean()          # noqa: E731
    cr = lambda su, sy: int(t[(t.suite == su) & (t.system == sy)].crushed.sum())     # noqa: E731
    v = [s for s, _ in SYS]
    row = lambda lab, f, fmt: (lab + "    ".join(f"{s.split('_')[0]} {fmt(f(s))}" for s in v), 36, False, (220, 220, 220))  # noqa: E731
    card([(f"{t.seed.nunique()} seeds, all suites (paired)" + (", MAGPIE gripper" if GRIPPER == "magpie" else ""), 54, True, "white"),
          row("Deceptive objects success:   ", lambda s: sr("S2_deceptive", s), lambda x: f"{x:.0%}"),
          row("Fragile objects success:      ", lambda s: sr("S4_fragile", s), lambda x: f"{x:.0%}"),
          row("Fragile crushes:                 ", lambda s: cr("S4_fragile", s), str),
          row("Honest objects success:       ", lambda s: sr("S1_honest_unseen", s), lambda x: f"{x:.0%}")],
         os.path.join(TMP, "99.mp4"), secs=5)
    parts.append(os.path.join(TMP, "99.mp4"))
    lst = os.path.join(TMP, "list.txt")
    open(lst, "w").write("".join(f"file '{p}'\n" for p in parts))
    final = os.path.join(OUT, "psg_v1_v2_v3" + ("_magpie" if GRIPPER == "magpie" else "") + ".mp4")
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                           "-c", "copy", final[:-4] + ".part.mp4"])
    os.replace(final[:-4] + ".part.mp4", final)
    print("wrote", final)


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
