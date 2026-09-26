"""
make_figures.py — regenerate manuscript figures from the honest model.
Fig 1: KM by risk tertile WITH numbers-at-risk table
Fig 2: calibration (honest slope/intercept) + DCA + risk distribution
Fig 3: subgroup forest + KM panels, overall C-index = 0.723 (not 0.760)
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold
import statsmodels.api as sm
import warnings; warnings.filterwarnings("ignore")

SEED=42; PEN=0.1; HORIZON=730.0
FEATURES=["age","heart_rate","LVEF","ACprev"]
np.random.seed(SEED)
df=pd.read_csv("data/BC_cardiotox_clinical_variables.csv",sep=";",decimal=",")
d=df[FEATURES+["CTRCD","time"]].dropna().reset_index(drop=True)
skf=StratifiedKFold(5,shuffle=True,random_state=SEED)

GREEN="#1D9E75"; ORANGE="#EF9F27"; RED="#D85A30"; GREY="#B4B2A9"; BLUE="#4E79C7"

# tertiles from full-model partial hazard
cph=CoxPHFitter(penalizer=PEN); cph.fit(d,"time","CTRCD")
d=d.copy(); d["ph"]=cph.predict_partial_hazard(d[FEATURES]).values
q1,q2=d["ph"].quantile([1/3,2/3])
d["tert"]=np.where(d["ph"]<=q1,"Low",np.where(d["ph"]<=q2,"Intermediate","High"))
tcol={"Low":BLUE,"Intermediate":ORANGE,"High":RED}

# ── FIG 1 ────────────────────────────────────────────────────────────────────
fig,ax=plt.subplots(figsize=(8,6)); fig.patch.set_facecolor("#fafaf8")
risk_years=[0,1,2,3,4,5,6]
atrisk={}
for grp in ["Low","Intermediate","High"]:
    sub=d[d["tert"]==grp]; yrs=sub["time"]/365.25
    kmf=KaplanMeierFitter().fit(yrs,sub["CTRCD"],label=grp)
    ax.plot(kmf.timeline,1-kmf.survival_function_.values.flatten(),color=tcol[grp],
            lw=1.8,label=f"{grp} risk (n={len(sub)}, ev={int(sub['CTRCD'].sum())})")
    ci=kmf.confidence_interval_
    ax.fill_between(kmf.timeline,1-ci.iloc[:,1],1-ci.iloc[:,0],color=tcol[grp],alpha=0.12)
    atrisk[grp]=[int((yrs>=y).sum()) for y in risk_years]
ax.set_xlim(0,6); ax.set_ylim(0,0.55)
ax.set_xlabel("Time (years)",labelpad=2); ax.set_ylabel("Cumulative CTRCD incidence")
ax.set_title("CTRCD cumulative incidence by predicted risk tertile",fontweight="500")
ax.legend(loc="upper left",fontsize=9); ax.spines[["top","right"]].set_visible(False)
# numbers-at-risk table under axis
ax.set_position([0.12,0.30,0.83,0.60])
tax=fig.add_axes([0.12,0.02,0.83,0.15]); tax.axis("off")
tax.text(-0.02,1.0,"Numbers at risk",transform=tax.transAxes,fontsize=8.5,fontweight="600")
for i,grp in enumerate(["Low","Intermediate","High"]):
    tax.text(-0.02,0.72-i*0.26,grp,transform=tax.transAxes,fontsize=8,color=tcol[grp],fontweight="600")
    for j,y in enumerate(risk_years):
        tax.text(0.16+j*0.135,0.72-i*0.26,str(atrisk[grp][j]),transform=tax.transAxes,fontsize=8,ha="center")
for j,y in enumerate(risk_years):
    tax.text(0.16+j*0.135,1.0,str(y),transform=tax.transAxes,fontsize=8,ha="center",color="#555")
plt.savefig("results/Fig1.png",dpi=150,bbox_inches="tight",facecolor="#fafaf8"); plt.close()
print("Fig1 done; numbers at risk:",atrisk)

# ── FIG 2: calibration + DCA + distribution ─────────────────────────────────
# out-of-fold 2yr risk
risk=np.zeros(len(d))
for tr,te in skf.split(d,d["CTRCD"]):
    m=CoxPHFitter(penalizer=PEN); m.fit(d.iloc[tr][FEATURES+["CTRCD","time"]],"time","CTRCD")
    risk[te]=1-m.predict_survival_function(d.iloc[te][FEATURES],times=[HORIZON]).values.flatten()
event2=((d["CTRCD"]==1)&(d["time"]<=HORIZON)).astype(int).values
keep=~((d["CTRCD"]==0)&(d["time"]<HORIZON))
r=np.clip(risk[keep.values],1e-6,1-1e-6); y=event2[keep.values]
logit=np.log(r/(1-r)); platt=sm.Logit(y,sm.add_constant(logit)).fit(disp=0)
inter,slope=platt.params; cal=platt.predict(sm.add_constant(logit))

fig,axes=plt.subplots(1,3,figsize=(16,5)); fig.patch.set_facecolor("#fafaf8")
# 2A calibration (raw vs recalibrated, quintiles)
ax=axes[0]; ax.plot([0,0.5],[0,0.5],"k--",lw=1,alpha=0.4,label="Perfect calibration")
for probs,lab,c,mk in [(r,"Before recalibration",RED,"o"),(cal,"After Platt",GREEN,"s")]:
    edges=np.percentile(probs,np.linspace(0,100,6)); pm,ob=[],[]
    for i in range(5):
        lo,hi=edges[i],edges[i+1]
        m=(probs>=lo)&(probs<=hi if i==4 else probs<hi)
        if m.sum()>0: pm.append(probs[m].mean()); ob.append(y[m].mean())
    ax.plot(pm,ob,f"{mk}-",color=c,lw=1.5,markersize=7,label=lab)
ax.set_xlim(-0.01,0.45); ax.set_ylim(-0.01,0.45)
ax.set_xlabel("Mean predicted 2-year risk"); ax.set_ylabel("Observed 2-year event rate")
ax.set_title("Calibration plot (quintiles, 2-year landmark)",fontweight="500")
ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)
ax.text(0.03,0.40,f"slope {slope:.2f}, intercept {inter:+.2f}\nE/O 1.33→1.00 (Platt)\nn=208, 33 events",
        fontsize=8,color="#444")
# 2B DCA
ax=axes[1]
def nb(yt,yp,pt): n=len(yt); tp=np.sum((yp>=pt)&(yt==1)); fp=np.sum((yp>=pt)&(yt==0)); return tp/n-fp/n*(pt/(1-pt+1e-8))
def nba(yt,pt): n=len(yt); return (yt==1).sum()/n-(yt==0).sum()/n*(pt/(1-pt+1e-8))
th=np.linspace(0.01,0.40,100)
ax.plot(th*100,[nb(y,cal,t) for t in th],color=GREEN,lw=2,label="Cox model")
ax.plot(th*100,[nba(y,t) for t in th],color=RED,lw=1.3,ls=":",label="Treat all")
ax.axhline(0,color="#444",lw=0.8,label="Treat none")
ax.set_xlim(0,40); ax.set_ylim(-0.02,0.12)
ax.set_xlabel("Risk threshold (%)"); ax.set_ylabel("Net benefit")
ax.set_title("Decision curve analysis (2-year CTRCD)",fontweight="500")
ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)
# 2C distribution
ax=axes[2]
bins=np.linspace(0,max(risk)*1.05,20)
ax.hist(risk[event2==0],bins=bins,alpha=0.6,color=GREEN,density=True,label=f"No CTRCD (n={(event2==0).sum()})")
ax.hist(risk[event2==1],bins=bins,alpha=0.6,color=RED,density=True,label=f"CTRCD (n={(event2==1).sum()})")
ax.axvline(np.median(risk),color="#444",ls="--",lw=1,label=f"Median {np.median(risk):.2f}")
ax.set_xlabel("Predicted 2-year risk"); ax.set_ylabel("Density")
ax.set_title("Predicted risk distribution by outcome",fontweight="500")
ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig("results/Fig2.png",dpi=150,bbox_inches="tight",facecolor="#fafaf8"); plt.close()
print(f"Fig2 done; slope={slope:.3f} intercept={inter:.3f}")

# ── FIG 3: subgroup forest + overall C-index 0.723 ──────────────────────────
df["BMI"]=df["weight"]/(df["height"]/100)**2
def cindex_sub(mask):
    sub=d[mask].reset_index(drop=True)
    if sub["CTRCD"].sum()<4: return None
    c=concordance_index(sub["time"],-sub["ph"],sub["CTRCD"]); return c,len(sub),int(sub["CTRCD"].sum())
subs=[("Age <50",d["age"]<50),("Age 50-65",(d["age"]>=50)&(d["age"]<=65)),("Age >65",d["age"]>65),
      ("LVEF <60",d["LVEF"]<60),("LVEF 60-70",(d["LVEF"]>=60)&(d["LVEF"]<=70)),("LVEF >70",d["LVEF"]>70),
      ("Prior AC",d["ACprev"]==1),("No prior AC",d["ACprev"]==0)]
fig,ax=plt.subplots(figsize=(8,6)); fig.patch.set_facecolor("#fafaf8")
rng=np.random.default_rng(SEED); rows=[]
for lab,mask in subs:
    res=cindex_sub(mask)
    if res is None: continue
    c,ns,evs=res
    # bootstrap CI
    sub=d[mask].reset_index(drop=True); bc=[]
    for _ in range(500):
        idx=rng.integers(0,len(sub),len(sub))
        if sub["CTRCD"].values[idx].sum()<2: continue
        bc.append(concordance_index(sub["time"].values[idx],-sub["ph"].values[idx],sub["CTRCD"].values[idx]))
    rows.append((lab,c,np.percentile(bc,2.5),np.percentile(bc,97.5),ns,evs))
rows=rows[::-1]
for i,(lab,c,lo,hi,ns,evs) in enumerate(rows):
    ax.plot([lo,hi],[i,i],color=GREEN,lw=1.3); ax.plot(c,i,"o",color=GREEN,markersize=7)
    ax.text(1.02,i,f"n={ns}, ev={evs}, C={c:.2f}",fontsize=8,va="center",color="#555")
ax.axvline(0.723,color="#444",ls="--",lw=1,label="Overall C-index 0.723")
ax.axvline(0.5,color="#999",ls=":",lw=0.8,label="No discrimination (0.5)")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows],fontsize=9)
ax.set_xlim(0.3,1.0); ax.set_xlabel("C-index (95% bootstrap CI)")
ax.set_title("Model C-index by subgroup",fontweight="500")
ax.legend(fontsize=8,loc="lower right"); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig("results/Fig3.png",dpi=150,bbox_inches="tight",facecolor="#fafaf8"); plt.close()
print("Fig3 done; overall C-index label = 0.723")
