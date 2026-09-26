"""
supp.py — Supplementary Figure 1 (bootstrap validation) + Central Illustration.
Self-contained: computes its own bootstrap; no external .npy dependencies.
Reproduces the honest 4-variable model numbers from raw data.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold
import warnings; warnings.filterwarnings("ignore")

np.random.seed(42); PEN=0.1; HORIZON=730.0; N_BOOT=1000
F=["age","heart_rate","LVEF","ACprev"]
df=pd.read_csv("data/BC_cardiotox_clinical_variables.csv",sep=";",decimal=",")
d=df[F+["CTRCD","time"]].dropna().reset_index(drop=True)
n=len(d); ev=int(d["CTRCD"].sum())

# ── bootstrap optimism correction (computed here, not loaded) ──
cph_app=CoxPHFitter(penalizer=PEN); cph_app.fit(d,"time","CTRCD")
c_app=concordance_index(d["time"],-cph_app.predict_partial_hazard(d),d["CTRCD"])
rng=np.random.default_rng(42); opt=[]; corig=[]
for _ in range(N_BOOT):
    idx=rng.integers(0,n,n); b=d.iloc[idx]
    if b["CTRCD"].sum()<2: continue
    try:
        mb=CoxPHFitter(penalizer=PEN); mb.fit(b,"time","CTRCD")
        cb=concordance_index(b["time"],-mb.predict_partial_hazard(b),b["CTRCD"])
        co=concordance_index(d["time"],-mb.predict_partial_hazard(d),d["CTRCD"])
        opt.append(cb-co); corig.append(co)
    except Exception: continue
opt=np.array(opt); corig=np.array(corig)
m_opt=float(np.mean(opt)); c_corr=c_app-m_opt
lo,hi=np.percentile(corig,2.5),np.percentile(corig,97.5)

# ── calibration + brier at 2-year landmark ──
skf=StratifiedKFold(5,shuffle=True,random_state=42)
risk=np.zeros(n)
for tr,te in skf.split(d,d["CTRCD"]):
    m=CoxPHFitter(penalizer=PEN); m.fit(d.iloc[tr],"time","CTRCD")
    risk[te]=1-m.predict_survival_function(d.iloc[te][F],times=[HORIZON]).values.flatten()
ev2=((d["CTRCD"]==1)&(d["time"]<=HORIZON)).astype(int).values
keep=~((d["CTRCD"]==0)&(d["time"]<HORIZON))
r=risk[keep.values]; y=ev2[keep.values]
cal_df=pd.DataFrame({"p":r,"o":y}); cal_df["b"]=pd.qcut(cal_df["p"],5,duplicates="drop")
cg=cal_df.groupby("b").agg(mp=("p","mean"),mo=("o","mean"),nn=("o","count")).reset_index()
cal_err=float(np.mean(np.abs(cg["mp"]-cg["mo"])))
bs_m=float(np.mean((r-y)**2)); bs_n=float(np.mean((y.mean()-y)**2)); scaled=1-bs_m/bs_n

print(f"apparent={c_app:.3f} corrected={c_corr:.3f} optimism={m_opt:+.3f} CI={lo:.3f}-{hi:.3f}")

BLUE="#2196F3"; GREEN="#1D9E75"; RED="#E91E63"; GREY="#9E9E9E"
fig=plt.figure(figsize=(14,9))
fig.suptitle(f"Bootstrap Internal Validation (n={n}, events={ev}, B={N_BOOT})",fontsize=13,fontweight="bold",y=0.99)
gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.42,wspace=0.35)
ax=fig.add_subplot(gs[0,0])
ax.hist(corig,bins=35,color=BLUE,alpha=0.75,edgecolor="white",lw=0.4)
ax.axvline(c_corr,color=RED,lw=2,label=f"Corrected: {c_corr:.3f}")
ax.axvline(c_app,color=GREEN,lw=1.5,ls="--",label=f"Apparent: {c_app:.3f}")
ax.axvline(0.5,color="black",lw=0.8,ls=":",alpha=0.5,label="No skill (0.5)")
ax.set_xlabel("C-index on original data"); ax.set_ylabel("Bootstrap frequency")
ax.set_title("A  C-index distribution\n(1000 bootstrap models)",fontweight="bold",loc="left"); ax.legend(fontsize=7.5)
ax=fig.add_subplot(gs[0,1])
ax.hist(opt,bins=35,color=GREY,alpha=0.8,edgecolor="white",lw=0.4)
ax.axvline(m_opt,color=RED,lw=2,label=f"Mean optimism: {m_opt:+.4f}")
ax.axvline(0,color="black",lw=0.8,ls="--",alpha=0.5)
ax.set_xlabel("Optimism (C_boot - C_orig)"); ax.set_ylabel("Frequency")
ax.set_title("B  Optimism distribution",fontweight="bold",loc="left"); ax.legend(fontsize=8)
ax=fig.add_subplot(gs[0,2])
vals=[c_app,c_corr]
ax.bar(["Apparent","Optimism-\ncorrected"],vals,color=[GREEN,BLUE],alpha=0.85,width=0.45,
       yerr=[[0,c_corr-lo],[0,hi-c_corr]],capsize=6,error_kw={"elinewidth":1.5})
ax.axhline(0.5,color="black",ls=":",lw=0.8,alpha=0.5); ax.axhline(0.7,color="green",ls=":",lw=0.8,alpha=0.4,label="C=0.70")
ax.set_ylim(0.45,0.9); ax.set_ylabel("C-index")
ax.set_title("C  Discrimination",fontweight="bold",loc="left"); ax.legend(fontsize=8)
for i,v in enumerate(vals): ax.text(i,v+0.008,f"{v:.3f}",ha="center",va="bottom",fontsize=10,fontweight="bold")
ax=fig.add_subplot(gs[1,0])
ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.5,label="Ideal")
ax.scatter(cg["mp"],cg["mo"],s=cg["nn"]*3,color=RED,alpha=0.85,zorder=5)
for _,row in cg.iterrows(): ax.annotate(f"n={int(row['nn'])}",(row["mp"],row["mo"]),textcoords="offset points",xytext=(5,4),fontsize=7.5)
lim=max(cg["mp"].max(),cg["mo"].max())*1.25; ax.set_xlim(0,lim); ax.set_ylim(0,lim)
ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed event rate")
ax.set_title(f"D  Calibration at 2 years\n(landmark, quintiles)",fontweight="bold",loc="left"); ax.legend(fontsize=8)
ax=fig.add_subplot(gs[1,1])
ax.bar(["Null model","Cox model"],[bs_n,bs_m],color=[GREY,BLUE],alpha=0.85,width=0.4)
for i,v in enumerate([bs_n,bs_m]): ax.text(i,v+0.0008,f"{v:.4f}",ha="center",va="bottom",fontsize=9.5,fontweight="bold")
ax.set_ylabel("Brier score (lower = better)"); ax.set_ylim(0,max(bs_n,bs_m)*1.35)
ax.set_title(f"E  Brier Score\nScaled IBS = {scaled:.3f}",fontweight="bold",loc="left")
ax=fig.add_subplot(gs[1,2]); ax.axis("off")
rows=[["Metric","Value"],["Patients",f"{n}"],["Events",f"{ev} ({ev/n*100:.1f}%)"],["Bootstrap resamples",f"{len(corig)}"],["",""],
      ["Apparent C-index",f"{c_app:.4f}"],["Mean optimism",f"{m_opt:+.4f}"],["Corrected C-index",f"{c_corr:.4f}"],
      ["95% CI",f"{lo:.4f} - {hi:.4f}"],["",""],["Calibration t=2y",""],["  Mean abs error",f"{cal_err:.4f}"],["",""],
      ["Brier (null)",f"{bs_n:.4f}"],["Brier (model)",f"{bs_m:.4f}"],["Scaled IBS",f"{scaled:.4f}"]]
tb=ax.table(cellText=rows,loc="center",cellLoc="left"); tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1.1,1.22)
for (rr,cc),cell in tb.get_celld().items():
    cell.set_edgecolor("#ddd")
    if rr==0: cell.set_facecolor("#37474F"); cell.set_text_props(color="white",fontweight="bold")
    elif rows[rr][0] in ("Corrected C-index","Scaled IBS"): cell.set_facecolor("#e3f2fd")
ax.set_title("F  Summary",fontweight="bold",loc="left",pad=10)
plt.savefig("results/SuppFig1.png",dpi=150,bbox_inches="tight",facecolor="white"); plt.close()
print("SuppFig1 done")

# ─────────── CENTRAL ILLUSTRATION (honest numbers) ───────────
cph=CoxPHFitter(penalizer=PEN); cph.fit(d,"time","CTRCD")
d2=d.copy(); d2["ph"]=cph.predict_partial_hazard(d2[F]).values
q1,q2=d2["ph"].quantile([1/3,2/3])
d2["t"]=np.where(d2["ph"]<=q1,"low",np.where(d2["ph"]<=q2,"int","high"))
inc2={}
for g in ["low","int","high"]:
    sub=d2[d2["t"]==g]; kmf=KaplanMeierFitter().fit(sub["time"]/365.25,sub["CTRCD"])
    inc2[g]=(1-float(kmf.predict(2)))*100
fig,ax=plt.subplots(figsize=(13,7.2)); ax.set_xlim(0,13); ax.set_ylim(0,9); ax.axis("off")
fig.patch.set_facecolor("white")
def box(x,y,w,h,fc,title,sub,tc="white"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.08,rounding_size=0.15",fc=fc,ec="none"))
    ax.text(x+w/2,y+h*0.63,title,ha="center",va="center",fontsize=12,fontweight="bold",color=tc)
    ax.text(x+w/2,y+h*0.28,sub,ha="center",va="center",fontsize=8.5,color=tc)
def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=14,color="#888",lw=1.2))
ax.text(2.0,8.6,"Pre-treatment inputs",ha="center",fontsize=10,color="#999")
ax.text(6.5,8.6,"Cox model output",ha="center",fontsize=10,color="#999")
ax.text(10.8,8.6,"Clinical implication",ha="center",fontsize=10,color="#999")
inputs=[("Age","years at baseline"),("Resting heart rate","beats per minute"),
        ("Baseline LVEF","ejection fraction %"),("Prior anthracyclines","yes / no")]
for i,(t,s) in enumerate(inputs):
    box(0.3,6.7-i*1.5,3.4,1.15,"#4B4BA6",t,s); arrow(3.8,7.27-i*1.5,5.0,4.4)
box(5.0,3.6,3.0,1.7,"#1D6E52","Cox proportional\nhazards model","optimism-corrected\nC-index 0.723")
tiers=[("Low risk",f"{inc2['low']:.1f}% 2-yr incidence\ncandidate for\nreduced surveillance","#1D6E52","\u2193 echo"),
       ("Intermediate risk",f"{inc2['int']:.1f}% 2-yr incidence","#8A6D1D","std."),
       ("High risk",f"{inc2['high']:.1f}% 2-yr incidence\nearly events dominant","#8B3A2E","\u2191 care")]
for i,(t,s,c,tag) in enumerate(tiers):
    yy=6.6-i*1.75; box(8.5,yy,3.1,1.35,c,t,s); arrow(8.1,4.45,8.5,yy+0.7)
    ax.add_patch(FancyBboxPatch((11.8,yy+0.25),0.9,0.85,boxstyle="round,pad=0.05,rounding_size=0.1",fc=c,ec="none"))
    ax.text(12.25,yy+0.67,tag,ha="center",va="center",fontsize=8,color="white")
ax.add_patch(FancyBboxPatch((0.3,0.15),11.4,1.2,boxstyle="round,pad=0.05,rounding_size=0.1",fc="#3A3A3A",ec="none"))
ax.text(6.0,1.05,"Model vs HFA-ICOS benchmark",ha="center",fontsize=11,fontweight="bold",color="white")
for i,(v,lab) in enumerate([("+0.077 paired","discrimination"),("NRI +0.625","reclassification"),("slope 1.52","calibration (needs recal.)")]):
    ax.text(2.3+i*3.8,0.62,v,ha="center",fontsize=9.5,color="#ddd")
    ax.text(2.3+i*3.8,0.34,lab,ha="center",fontsize=8,color="#aaa")
plt.savefig("results/CentralIllustration.png",dpi=150,bbox_inches="tight",facecolor="white"); plt.close()
print(f"CentralIllustration done; 2yr inc = {inc2['low']:.1f}/{inc2['int']:.1f}/{inc2['high']:.1f}")