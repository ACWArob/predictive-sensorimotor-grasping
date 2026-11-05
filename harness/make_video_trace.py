"""Inside one grasp: v1 vs v3 on the MAGPIE model, with the live force trace beside each render.

Each row: the MuJoCo render (left) and, drawn as the grasp plays, the grip the controller commands,
the grip the object actually needs at that instant (felt load / 2*mu), and the crush limit. Real
harness trials (seed 0), replayed exactly.

    MUJOCO_GL=osmesa python -m harness.make_video_trace   ->  harness/out_magpie/psg_inside_one_grasp.mp4
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
from harness.make_video import FPS, encode, font                   # noqa: E402
from psg import ForwardModel, PredictiveGraspAgent                       # noqa: E402
from sim.run_experiments import collect                                  # noqa: E402

OUT = os.path.join(ROOT, "harness", "out_magpie")
TMP = os.path.join(OUT, "_trace_tmp")
W, ROWH, HEAD, RW = 1920, 470, 140, 620          # canvas width, row height, header, render width
EVERY = 16
SYS = [("v1_deligrasp", "v1  DeliGrasp (current)"), ("v3_psg_couple", "v3  PSG + coupling")]
SCENES = [("S2_deceptive", 5, "Lead-filled plastic, grasp 1: looks light, is very heavy",
           "v1 sets one grip from the looks and it's far too low. v3 closes gently, feels the load climb past its prediction, and raises grip with it."),
          ("S4_fragile", 0, "Thin cup that looks sturdy, grasp 1",
           "v1 squeezes 7x harder than needed and crushes it. v3 closes at a third of its plan and adds grip only as the load asks for it."),
          ("S2_deceptive", 10, "Hollow metal, grasp 2 (after one try)",
           "v1 makes the same mistake every time. v3 corrected this object's prediction after one grasp, so it starts right.")]
INK, MUTED, GRID = (20, 24, 32), (110, 110, 110), (225, 225, 220)
BLUE, RED, GREEN, ORANGE = (42, 122, 226), (214, 60, 50), (40, 160, 90), (230, 140, 30)


def card(lines, path, secs=4.0):
    im = Image.new("RGB", (W, HEAD + 2 * ROWH), INK)
    d = ImageDraw.Draw(im)
    y = 330
    for txt, sz, bold, col in lines:
        d.text((110, y), txt, font=font(sz, bold), fill=col)
        y += int(sz * 1.7)
    encode([np.asarray(im)] * int(secs * FPS), path)


def record():
    H.use_gripper("magpie")
    want = {(s, t) for s, t, *_ in SCENES}
    X, Y, vrows, _ = collect(np.random.default_rng(0))
    model = ForwardModel(seed=0).fit(X, Y)
    real = H.run_grasp
    clips = {}
    for suite, trials in H.suites(0).items():
        if suite not in {s for s, _ in want}:
            continue
        for name, _ in SYS:
            system = H.new_system(name, model, vrows)
            rng = np.random.default_rng(0 * 7 + zlib.crc32(suite.encode()) % 1000)
            for i, (obj, k) in enumerate(trials):
                if (suite, i) not in want:
                    H.run_grasp = real
                    H.do_grasp(system, obj, rng)
                    continue
                buf, tr, st = [], [], {}

                def hook(m, d, info, buf=buf, tr=tr, st=st, obj=obj):
                    if "r" not in st:
                        m.vis.global_.offwidth, m.vis.global_.offheight = RW, ROWH
                        st["r"] = mujoco.Renderer(m, ROWH, RW)
                        c = mujoco.MjvCamera()
                        c.lookat[:] = (0, 0, 0.15)
                        c.distance, c.azimuth, c.elevation = 0.55, 200, -15
                        st["c"] = c
                        m.vis.headlight.ambient[:] = (0.45,) * 3
                        m.vis.headlight.diffuse[:] = (0.65,) * 3
                        m.geom_rgba[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "obj")] = (0.35, 0.6, 0.9, 1)
                    if info["k"] < 0:
                        return
                    if info["crushed"]:
                        m.geom_rgba[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "obj")] = (0.9, 0.2, 0.2, 1)
                    tr.append((info["k"] / 1000.0, info["grip"], max(info["load"], 0.0) / (2 * obj.mu)))
                    if info["k"] % EVERY == 0:
                        st["r"].update_scene(d, st["c"])
                        buf.append((st["r"].render().copy(), len(tr), info["crushed"], info["slipped"]))

                H.run_grasp = lambda o, f, cb, hook=hook: real(o, f, cb, on_step=hook)
                pred, box = None, {}
                if isinstance(system, PredictiveGraspAgent):
                    orig = system.plan

                    def plan(pc, orig=orig, box=box):
                        p, c = orig(pc)
                        box["p"] = p
                        return p, c
                    system.plan = plan
                row = H.do_grasp(system, obj, rng)
                if isinstance(system, PredictiveGraspAgent):
                    pred = box.get("p")
                    del system.plan
                tr = np.array(tr)
                tr[:, 2] = np.convolve(tr[:, 2], np.ones(25) / 25, mode="same")      # 25 ms smoothing of the felt load
                clips[(suite, i, name)] = dict(frames=buf, trace=tr, row=row, obj=obj, pred=pred)
    H.run_grasp = real
    return clips


def plot(draw, x0, y0, w, h, tr, upto, obj, ymax, pred):
    L, R, T, B = x0 + 70, x0 + w - 30, y0 + 50, y0 + h - 50
    t_end = 2.3
    X = lambda t: L + (R - L) * t / t_end                                # noqa: E731
    Y = lambda v: B - (B - T) * min(v, ymax) / ymax                      # noqa: E731
    step = next(s for s in (0.5, 1, 2, 5, 10, 20) if ymax / s <= 6)
    for v in np.arange(0, ymax + 1e-9, step):
        draw.line([(L, Y(v)), (R, Y(v))], fill=GRID, width=1)
        draw.text((L - 12, Y(v)), f"{v:g}", font=font(17), fill=MUTED, anchor="rm")
    draw.text((L - 55, T - 30), "N", font=font(17), fill=MUTED)
    for t, lab in ((0.0, "close"), (0.3, "lift"), (1.3, "hold")):
        draw.line([(X(t), T), (X(t), B)], fill=(200, 200, 195), width=1)
        draw.text((X(t) + 6, B + 8), lab, font=font(17), fill=MUTED)
    # crush limit
    if obj.f_crush < ymax:
        draw.line([(L, Y(obj.f_crush)), (R, Y(obj.f_crush))], fill=RED, width=2)
        draw.text((R - 6, Y(obj.f_crush) - 6), f"crush limit {obj.f_crush:.0f} N", font=font(17), fill=RED, anchor="rb")
    seg = tr[:upto]
    if len(seg) > 1:
        need = [(X(t), Y(n)) for t, _, n in seg[::4]]
        grip = [(X(t), Y(g)) for t, g, _ in seg[::4]]
        draw.line(need, fill=(90, 90, 90), width=3)
        draw.line(grip, fill=BLUE, width=4)
        t, g, n = seg[-1]
        draw.ellipse([X(t) - 6, Y(g) - 6, X(t) + 6, Y(g) + 6], fill=BLUE)
    # legend
    lx = L + 10
    for col, lab in ((BLUE, "grip commanded"), ((90, 90, 90), "grip the object needs right now")):
        draw.line([(lx, T - 22), (lx + 30, T - 22)], fill=col, width=4)
        draw.text((lx + 38, T - 22), lab, font=font(18), fill=INK, anchor="lm")
        lx += 60 + draw.textlength(lab, font=font(18))
    if pred is not None:
        pn = float(np.exp(pred.mean[0] + pred.mean[1]))
        draw.line([(X(0.3), Y(pn)), (R, Y(pn))], fill=BLUE, width=1)
        draw.text((R - 6, Y(pn) + 6), f"v3's predicted need {pn:.1f} N", font=font(17), fill=BLUE, anchor="rt")


def scene_frames(clips, suite, t, cap, sub):
    rows = [clips[(suite, t, s)] for s, _ in SYS]
    obj = rows[0]["obj"]
    ymax = max(max(r["trace"][:, 1].max(), r["trace"][:, 2].max()) for r in rows) * 1.15
    if obj.f_crush < 1.5 * ymax:
        ymax = max(ymax, obj.f_crush * 1.15)
    ymax = max(ymax, 2.0)
    L = max(len(r["frames"]) for r in rows)
    out = []
    for fi in range(L + FPS):
        im = Image.new("RGB", (W, HEAD + 2 * ROWH), (247, 246, 243))
        d = ImageDraw.Draw(im)
        d.text((28, 22), cap, font=font(40, True), fill=INK)
        d.text((28, 80), sub, font=font(24), fill=(70, 70, 70))
        for j, ((sname, lab), r) in enumerate(zip(SYS, rows)):
            y0 = HEAD + j * ROWH
            img, upto, crushed, slipped = r["frames"][min(fi, len(r["frames"]) - 1)]
            im.paste(Image.fromarray(img), (0, y0))
            d.rectangle([0, y0, RW, y0 + 40], fill=INK)
            d.text((14, y0 + 7), lab, font=font(22, True), fill="white")
            if fi >= len(r["frames"]) - 1:
                ok = r["row"]["success"]
                tag, col = ("HELD", GREEN) if ok else (("CRUSHED", RED) if r["row"]["crushed"] else ("DROPPED", ORANGE))
                f = font(28, True)
                w = d.textlength(tag, font=f)
                d.rounded_rectangle([RW - w - 34, y0 + 54, RW - 14, y0 + 96], 8, fill=col)
                d.text((RW - w - 24, y0 + 58), tag, font=f, fill="white")
            plot(d, RW, y0, W - RW, ROWH, r["trace"], upto, obj, ymax, r["pred"])
            d.line([(0, y0 + ROWH - 1), (W, y0 + ROWH - 1)], fill=(220, 220, 215), width=2)
        out.append(np.asarray(im))
    return out


def main():
    os.makedirs(TMP, exist_ok=True)
    clips = record()
    parts = [os.path.join(TMP, "00.mp4")]
    card([("Inside one grasp", 64, True, "white"),
          ("Current AutoGrasp (v1) vs PSG + grip-load coupling (v3), MAGPIE gripper in MuJoCo", 32, False, (200, 200, 200)),
          ("Blue = grip the controller commands.  Grey = grip the object needs at that instant.", 30, False, (170, 170, 170)),
          ("Below grey: it slips.  Above the red line: it crushes.  Real harness trials, half speed.", 30, False, (170, 170, 170))],
         parts[0], secs=5)
    for n, (suite, t, cap, sub) in enumerate(SCENES, 1):
        fr = scene_frames(clips, suite, t, cap, sub)
        if "--check" in sys.argv:
            for q in (0.3, 0.6, 1.0):
                Image.fromarray(fr[int(q * (len(fr) - 1))]).save(os.path.join(TMP, f"check_{n}_{q}.png"))
            continue
        p = os.path.join(TMP, f"{n:02d}.mp4")
        encode(fr, p)
        parts.append(p)
    if "--check" in sys.argv:
        return
    lst = os.path.join(TMP, "list.txt")
    open(lst, "w").write("".join(f"file '{p}'\n" for p in parts))
    final = os.path.join(OUT, "psg_inside_one_grasp.mp4")
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                           final[:-4] + ".part.mp4"])
    os.replace(final[:-4] + ".part.mp4", final)
    print("wrote", final)


if __name__ == "__main__":
    main()
