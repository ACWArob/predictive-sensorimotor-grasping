import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
INK,INK2,EDGE,SURF="#0b0b0b","#52514e","#9a9893","#fcfcfb"
BLUE,ORANGE,AQUA,GREY="#2a78d6","#eb6834","#1baf7a","#e4e3df"
fig,ax=plt.subplots(figsize=(15,8.6),dpi=150); fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
ax.set_xlim(0,15); ax.set_ylim(0,8.6); ax.axis("off")
def box(x,y,w,h,title,lines,c=EDGE,fill=None,tc=INK):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.12",fc=fill or SURF,ec=c,lw=2 if fill else 1.3,alpha=1))
    ax.text(x+w/2,y+h-0.28,title,ha="center",va="top",fontsize=11,fontweight="bold",color=tc)
    for i,l in enumerate(lines): ax.text(x+w/2,y+h-0.68-i*0.3,l,ha="center",va="top",fontsize=8.8,color=INK2)
def arr(a,b,c=INK2,ls="-",rad=0):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=14,lw=1.4,color=c,linestyle=ls,connectionstyle=f"arc3,rad={rad}"))
ax.text(0.2,8.4,"Predictive sensorimotor grasping inside AutoGrasp: predict the contact, compare at control points, learn value from success",fontsize=13,fontweight="bold",color=INK,va="top")
ax.text(0.2,7.98,"Grey = existing MAGPIE (unchanged).  Coloured = new.  Solid = runs every grasp.  Dashed = learning.",fontsize=9.5,color=INK2,va="top")
# existing row
box(0.2,5.6,2.3,1.6,"Perception",["Gemini detect","SAM3 mask","DINOv2 embedding"],fill="#f2f1ed")
box(2.9,5.6,2.4,1.6,"Grasp planner",["GraspGenX + PCA","Gemini arbitration","-> pose + width"],fill="#f2f1ed")
box(11.9,5.6,2.9,1.6,"DeliGrasp loop",["closes at grip_cmd","emits contact / load /","liftoff / slip events"],fill="#f2f1ed")
# new
box(5.8,5.6,2.6,1.6,"Predictor",["forward model (cerebellum)","-> load, slip ratio","+ std per output"],c=BLUE,fill="#eaf2fb",tc=BLUE)
box(8.8,5.6,2.7,1.6,"Command",["grip = load x ratio x margin","margin = f(std) or value head","capped by fragility ceiling"],c=INK2)
box(11.9,3.3,2.9,1.6,"Comparator",["actual vs predicted at","each control point","-> reflex now / update"],c=ORANGE,fill="#fdeee7",tc=ORANGE)
box(8.8,3.3,2.7,1.6,"Object residual",["per-object offset r, var P","Kalman: 1 grasp fixes it","+ crush ceiling"],c=ORANGE,fill="#fdeee7",tc=ORANGE)
box(5.8,3.3,2.6,1.6,"Value head",["RL-Token-style residual","reward = success only","acts only if familiar"],c=AQUA,fill="#e6f6f0",tc=AQUA)
box(0.2,3.3,5.1,1.6,"LeRobot v3 log",["percept, prediction, every control point,","grip trace, outcome  (= next training set)"])
arr((2.5,6.4),(2.9,6.4)); arr((5.3,6.4),(5.8,6.4)); arr((8.4,6.4),(8.8,6.4)); arr((11.5,6.4),(11.9,6.4))
arr((13.0,5.6),(13.0,4.9),c=ORANGE); ax.text(12.9,5.25,"events",fontsize=8.5,color=ORANGE,ha="right")
arr((13.7,4.9),(13.7,5.6),c=ORANGE); ax.text(13.8,5.25,"reflex:\nnew grip",fontsize=8.5,color=ORANGE,va="center")
arr((11.9,4.1),(11.5,4.1),c=ORANGE,ls="--"); ax.text(11.7,3.9,"update",fontsize=8,color=ORANGE,ha="center",va="top")
arr((10.15,4.9),(10.15,5.6),c=ORANGE); ax.text(10.25,5.25,"r corrects\nthe prediction",fontsize=8.5,color=ORANGE,va="center")
arr((7.1,5.6),(7.1,4.9),c=AQUA,ls="--"); ax.text(7.0,5.25,"std = gate",fontsize=8.5,color=AQUA,va="center",ha="right")
arr((8.4,4.6),(8.95,5.6),c=AQUA); ax.text(8.8,5.0,"margin",fontsize=8.5,color=AQUA,ha="left",va="center")
# timescales band
ax.add_patch(FancyBboxPatch((0.2,0.25),14.6,2.65,boxstyle="round,pad=0.02,rounding_size=0.12",fc="#f6f5f2",ec=GREY))
ax.text(0.45,2.7,"Three timescales, three error signals (Doya 2000; McClelland et al. 1995)",fontsize=11,fontweight="bold",color=INK,va="top")
rows=[("Within a grasp (ms)","Comparator reflex","predicted vs felt contact","no learning: grip-load coupling",ORANGE),
      ("Across grasps (1 grasp)","Object residual + ceiling","same mismatch, once","this object only, no weights",ORANGE),
      ("Offline / overnight","Predictor (supervised) + value head (RL)","logged labels / success","VLA stays frozen",BLUE)]
for i,(a,b,c,d,col) in enumerate(rows):
    y=2.15-i*0.6
    ax.text(0.45,y,a,fontsize=9.5,color=INK,fontweight="bold",va="top")
    ax.text(3.4,y,b,fontsize=9.5,color=col,fontweight="bold",va="top")
    ax.text(7.6,y,"learns from: "+c,fontsize=9,color=INK2,va="top")
    ax.text(11.4,y,d,fontsize=9,color=INK2,va="top")
plt.savefig("results/fig_architecture.png",bbox_inches="tight",facecolor=SURF)
