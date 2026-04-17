"""
=============================================================================
TQM & Lean Manufacturing Impact on Sustainability Performance
SEM-ANN Hybrid Analysis – Saudi Food Manufacturing Sector
=============================================================================

"""

import os
import sys

# Output folder = same directory as this script
OUT = os.path.dirname(os.path.abspath(__file__))

# Use TkAgg so figures appear as interactive windows on screen
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams.update({"figure.raise_window": True})

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPRegressor
from numpy.linalg import lstsq
import warnings
warnings.filterwarnings("ignore")


def save_and_show(filename):
    """Save figure to disk AND display it on screen."""
    path = os.path.join(OUT, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()       # pops up the window – close it to continue to next figure
    plt.close("all")
    print(f"  Saved -> {path}")


# =============================================================================
# 1. DATA GENERATION  (n=350, 5-point Likert)
# =============================================================================
print("\n[1/9]  Generating dataset ...")
np.random.seed(42)
N = 350

def likert(base, noise=0.4, n=N):
    return np.clip(np.round(base + np.random.normal(0, noise, n)), 1, 5).astype(int)

tqm_base  = np.random.normal(3.8, 0.6, N)
lean_base = np.random.normal(3.6, 0.7, N)
sus_base  = 0.55 * tqm_base + 0.45 * lean_base + np.random.normal(0, 0.3, N)

df = pd.DataFrame({
    "TQM1":  likert(tqm_base),       "TQM2":  likert(tqm_base, 0.5),  "TQM3":  likert(tqm_base),
    "LEAN1": likert(lean_base),      "LEAN2": likert(lean_base, 0.5), "LEAN3": likert(lean_base),
    "ENV1":  likert(sus_base),       "ENV2":  likert(sus_base, 0.5),  "ENV3":  likert(sus_base),
    "SOC1":  likert(sus_base),       "SOC2":  likert(sus_base, 0.5),  "SOC3":  likert(sus_base),
    "ECO1":  likert(sus_base),       "ECO2":  likert(sus_base, 0.5),  "ECO3":  likert(sus_base),
    "FirmSize": np.random.choice(["Small","Medium","Large"], N, p=[0.30,0.45,0.25]),
    "Region":   np.random.choice(["Riyadh","Jeddah","Dammam","Other"], N, p=[0.35,0.30,0.20,0.15]),
    "YearsOp":  np.random.choice([1,2,3,4,5], N, p=[0.10,0.20,0.30,0.25,0.15]),
})

for col, items in [("TQM",["TQM1","TQM2","TQM3"]),("LEAN",["LEAN1","LEAN2","LEAN3"]),
                   ("ENV",["ENV1","ENV2","ENV3"]),  ("SOC",["SOC1","SOC2","SOC3"]),
                   ("ECO",["ECO1","ECO2","ECO3"])]:
    df[col] = df[items].mean(axis=1)
df["SP"] = df[["ENV","SOC","ECO"]].mean(axis=1)

df.to_csv(os.path.join(OUT, "survey_data.csv"), index=False)
print(f"  Dataset shape: {df.shape}")
print(df[["TQM","LEAN","ENV","SOC","ECO","SP"]].describe().round(3))


# =============================================================================
# 2. CORRELATION HEATMAP
# =============================================================================
print("\n[2/9]  Plotting correlation heatmap ...")
latent = ["TQM","LEAN","ENV","SOC","ECO","SP"]
corr   = df[latent].corr()

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="Blues", ax=ax, linewidths=0.5)
ax.set_title("Figure 1 – Pearson Correlation Matrix (Latent Constructs)",
             fontsize=12, fontweight="bold", pad=10)
plt.tight_layout()
save_and_show("fig_correlation.png")


# =============================================================================
# 3. CONFIRMATORY FACTOR ANALYSIS
# =============================================================================
print("\n[3/9]  CFA – Reliability & Validity ...")

def cronbach_alpha(data):
    k = data.shape[1]
    return (k/(k-1))*(1 - data.var(axis=0,ddof=1).sum()/data.sum(axis=1).var(ddof=1))

def comp_rel(lam):
    s = sum(lam)**2
    return s/(s + sum(1-l**2 for l in lam))

def avg_ve(lam):
    return float(np.mean([l**2 for l in lam]))

constructs = {"TQM":["TQM1","TQM2","TQM3"],"LEAN":["LEAN1","LEAN2","LEAN3"],
              "ENV":["ENV1","ENV2","ENV3"],"SOC":["SOC1","SOC2","SOC3"],"ECO":["ECO1","ECO2","ECO3"]}
rows = []
for name, items in constructs.items():
    alpha = cronbach_alpha(df[items])
    comp  = df[items].mean(axis=1)
    lam   = [float(df[i].corr(comp)) for i in items]
    rows.append({"Construct":name,"Cronbach_Alpha":round(alpha,3),
                 "CR":round(comp_rel(lam),3),"AVE":round(avg_ve(lam),3),
                 "Loadings":str([round(l,3) for l in lam])})
    print(f"  {name}: alpha={alpha:.3f}  CR={comp_rel(lam):.3f}  AVE={avg_ve(lam):.3f}")

cfa_df = pd.DataFrame(rows)
cfa_df.to_csv(os.path.join(OUT,"cfa_validity.csv"),index=False)

fig, ax = plt.subplots(figsize=(9, 5))
x     = np.arange(len(cfa_df))
w     = 0.25
b1    = ax.bar(x-w,   cfa_df["Cronbach_Alpha"], w, label="Cronbach alpha", color="#2471A3")
b2    = ax.bar(x,     cfa_df["CR"],             w, label="CR",             color="#1ABC9C")
b3    = ax.bar(x+w,   cfa_df["AVE"],            w, label="AVE",            color="#E74C3C")
ax.axhline(0.70,color="gray",  linestyle="--",lw=1,label="Threshold=0.70")
ax.axhline(0.50,color="orange",linestyle=":", lw=1,label="AVE Threshold=0.50")
ax.set_xticks(x); ax.set_xticklabels(cfa_df["Construct"])
ax.set_ylim(0,1.05); ax.set_ylabel("Score"); ax.set_xlabel("Construct")
ax.set_title("Figure 2 – CFA: Reliability & Validity (alpha, CR, AVE)",
             fontsize=12, fontweight="bold")
ax.legend()
for bar in list(b1)+list(b2)+list(b3):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f"{bar.get_height():.2f}",ha="center",va="bottom",fontsize=7.5)
plt.tight_layout()
save_and_show("fig_cfa.png")


# =============================================================================
# 4. SEM PATH ESTIMATION
# =============================================================================
print("\n[4/9]  SEM path estimation ...")

def ols_path(X_df, y_s):
    X = np.column_stack([np.ones(len(X_df))]+[X_df[c].values for c in X_df.columns])
    y = y_s.values
    b,_,_,_ = lstsq(X,y,rcond=None)
    yp = X@b
    r2 = 1 - np.sum((y-yp)**2)/np.sum((y-y.mean())**2)
    return b[1:], r2

b_env,r2_env = ols_path(df[["TQM","LEAN"]],df["ENV"])
b_soc,r2_soc = ols_path(df[["TQM","LEAN"]],df["SOC"])
b_eco,r2_eco = ols_path(df[["TQM","LEAN"]],df["ECO"])
b_sp, r2_sp  = ols_path(df[["TQM","LEAN","ENV","SOC","ECO"]],df["SP"])

sem_df = pd.DataFrame({
    "Path": ["TQM->ENV","LEAN->ENV","TQM->SOC","LEAN->SOC",
             "TQM->ECO","LEAN->ECO","ENV->SP","SOC->SP","ECO->SP"],
    "Beta": [b_env[0],b_env[1],b_soc[0],b_soc[1],
             b_eco[0],b_eco[1],b_sp[2], b_sp[3], b_sp[4]],
    "R2":   [r2_env,r2_env,r2_soc,r2_soc,r2_eco,r2_eco,r2_sp,r2_sp,r2_sp],
})
sem_df["Beta"] = sem_df["Beta"].round(4)
sem_df["R2"]   = sem_df["R2"].round(4)
sem_df.to_csv(os.path.join(OUT,"sem_path_results.csv"),index=False)
print(sem_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10,5))
c_bar = ["#2471A3" if "TQM" in p else "#117A65" if "LEAN" in p else "#E74C3C"
         for p in sem_df["Path"]]
bars = ax.bar(sem_df["Path"],sem_df["Beta"],color=c_bar,edgecolor="white")
ax.axhline(0,color="black",lw=0.8)
ax.set_ylabel("Path Coefficient (Beta)")
ax.set_title("Figure 3 – SEM Path Coefficients",fontsize=12,fontweight="bold")
ax.set_xticklabels(sem_df["Path"],rotation=30,ha="right")
for bar in bars:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f"{bar.get_height():.3f}",ha="center",va="bottom",fontsize=8)
plt.tight_layout()
save_and_show("fig_sem_barchart.png")


# =============================================================================
# 5. SEM PATH DIAGRAM
# =============================================================================
print("\n[5/9]  Drawing SEM path diagram ...")
fig, ax = plt.subplots(figsize=(13,7))
ax.set_xlim(0,10); ax.set_ylim(0,7); ax.axis("off")
fig.patch.set_facecolor("#F8F9FA"); ax.set_facecolor("#F8F9FA")

def box(x,y,w,h,label,fc):
    ax.add_patch(plt.Rectangle((x-w/2,y-h/2),w,h,lw=1.5,
                 edgecolor=fc,facecolor=fc,alpha=0.90,zorder=3))
    ax.text(x,y,label,ha="center",va="center",
            fontsize=10,fontweight="bold",color="white",zorder=4)

def arrow(x1,y1,x2,y2,label,col="black"):
    ax.annotate("",xy=(x2,y2),xytext=(x1,y1),
                arrowprops=dict(arrowstyle="->",color=col,lw=1.6))
    ax.text((x1+x2)/2,(y1+y2)/2+0.17,label,
            ha="center",va="bottom",fontsize=8,color=col)

box(1.5,5.5,1.8,0.65,"TQM", "#1A3A5C")
box(1.5,2.5,1.8,0.65,"LEAN","#1A3A5C")
box(5.0,6.3,1.8,0.65,"ENV", "#1A5276")
box(5.0,4.0,1.8,0.65,"SOC", "#1A5276")
box(5.0,1.7,1.8,0.65,"ECO", "#1A5276")
box(8.5,4.0,1.8,0.65,"SP",  "#117A65")

arrow(2.4,5.5,4.1,6.3,f"b={b_env[0]:.3f}")
arrow(2.4,5.3,4.1,4.0,f"b={b_soc[0]:.3f}")
arrow(2.4,5.1,4.1,1.7,f"b={b_eco[0]:.3f}","#2C3E50")
arrow(2.4,2.7,4.1,6.0,f"b={b_env[1]:.3f}","#839192")
arrow(2.4,2.5,4.1,4.0,f"b={b_soc[1]:.3f}","#839192")
arrow(2.4,2.3,4.1,1.7,f"b={b_eco[1]:.3f}","#839192")
arrow(5.9,6.3,7.6,4.2,f"b={b_sp[2]:.3f}")
arrow(5.9,4.0,7.6,4.0,f"b={b_sp[3]:.3f}")
arrow(5.9,1.7,7.6,3.8,f"b={b_sp[4]:.3f}")

ax.set_title("Figure 4 – SEM Path Diagram: TQM & Lean -> Sustainability Performance",
             fontsize=12,fontweight="bold",pad=12)
plt.tight_layout()
save_and_show("fig_sem_path.png")


# =============================================================================
# 6. ARTIFICIAL NEURAL NETWORK
# =============================================================================
print("\n[6/9]  Training ANN model ...")
features = ["TQM","LEAN","ENV","SOC","ECO"]
X = df[features].values; y = df["SP"].values
scaler_X=StandardScaler(); scaler_y=StandardScaler()
X_sc = scaler_X.fit_transform(X)
y_sc = scaler_y.fit_transform(y.reshape(-1,1)).ravel()
X_tr,X_te,y_tr,y_te = train_test_split(X_sc,y_sc,test_size=0.2,random_state=42)

ann = MLPRegressor(hidden_layer_sizes=(64,32,16),activation="relu",
                   max_iter=500,random_state=42,
                   early_stopping=True,validation_fraction=0.1,n_iter_no_change=20)
ann.fit(X_tr,y_tr)

y_pred    = scaler_y.inverse_transform(ann.predict(X_te).reshape(-1,1)).ravel()
y_actual  = scaler_y.inverse_transform(y_te.reshape(-1,1)).ravel()
rmse = np.sqrt(mean_squared_error(y_actual,y_pred))
mae  = mean_absolute_error(y_actual,y_pred)
r2   = r2_score(y_actual,y_pred)
print(f"  RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

fig, axes = plt.subplots(1,2,figsize=(12,4))
axes[0].plot(ann.loss_curve_,color="#2471A3",label="Training Loss")
axes[0].set_title("ANN Training Loss Curve"); axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("MSE Loss"); axes[0].legend()

axes[1].scatter(y_actual,y_pred,alpha=0.5,color="#2471A3",s=25,label="Predictions")
lims=[min(y_actual.min(),y_pred.min()),max(y_actual.max(),y_pred.max())]
axes[1].plot(lims,lims,"r--",lw=1.5,label="Perfect fit")
axes[1].set_xlabel("Actual SP"); axes[1].set_ylabel("Predicted SP")
axes[1].set_title(f"Figure 5 – ANN: Actual vs Predicted  (R2={r2:.4f})")
axes[1].legend()
plt.suptitle("ANN Model Performance",fontsize=13,fontweight="bold",y=1.01)
plt.tight_layout()
save_and_show("fig_ann_performance.png")


# =============================================================================
# 7. VARIABLE IMPORTANCE
# =============================================================================
print("\n[7/9]  Computing permutation importance ...")
perm = permutation_importance(ann,X_te,y_te,n_repeats=30,random_state=42)
imp_norm = perm.importances_mean/perm.importances_mean.sum()*100
imp_df = pd.DataFrame({"Feature":features,"Importance_pct":imp_norm.round(2)
                       }).sort_values("Importance_pct",ascending=False)
imp_df.to_csv(os.path.join(OUT,"ann_importance.csv"),index=False)
print(imp_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(8,5))
c_imp = ["#2471A3" if f in ["TQM","LEAN"] else "#1ABC9C" for f in imp_df["Feature"]]
bars = ax.barh(imp_df["Feature"],imp_df["Importance_pct"],color=c_imp)
ax.set_xlabel("Normalized Importance (%)")
ax.set_title("Figure 6 – ANN Variable Importance for Sustainability Performance",
             fontsize=12,fontweight="bold")
ax.invert_yaxis()
for bar,val in zip(bars,imp_df["Importance_pct"]):
    ax.text(bar.get_width()+0.3,bar.get_y()+bar.get_height()/2,
            f"{val:.1f}%",va="center",fontsize=9)
ax.legend(handles=[
    mpatches.Patch(color="#2471A3",label="Independent Variables (TQM, LEAN)"),
    mpatches.Patch(color="#1ABC9C",label="Mediating Variables (ENV, SOC, ECO)"),
],loc="lower right")
plt.tight_layout()
save_and_show("fig_importance.png")


# =============================================================================
# 8. IPMA
# =============================================================================
print("\n[8/9]  Building IPMA ...")
imp_dict  = dict(zip(imp_df["Feature"],imp_df["Importance_pct"]))
perf_dict = {f:(df[f].mean()-1)/4*100 for f in features}
ipma_df = pd.DataFrame({
    "Variable":features,
    "Importance":[imp_dict[f] for f in features],
    "Performance":[perf_dict[f] for f in features],
})

fig, ax = plt.subplots(figsize=(8,6))
avg_imp=ipma_df["Importance"].mean(); avg_perf=ipma_df["Performance"].mean()
ax.axvline(avg_imp, color="gray",linestyle="--",alpha=0.7,label="Avg Importance")
ax.axhline(avg_perf,color="gray",linestyle=":", alpha=0.7,label="Avg Performance")
for _,row in ipma_df.iterrows():
    hi  = row["Importance"]  >= avg_imp
    hpe = row["Performance"] >= avg_perf
    c = "#E74C3C" if (hi and not hpe) else "#27AE60" if (hi and hpe) else "#3498DB"
    ax.scatter(row["Importance"],row["Performance"],s=220,color=c,zorder=5)
    ax.annotate(row["Variable"],(row["Importance"],row["Performance"]),
                textcoords="offset points",xytext=(8,5),fontsize=11)
ax.set_xlabel("Importance (%)",fontsize=11)
ax.set_ylabel("Performance (0-100 scale)",fontsize=11)
ax.set_title("Figure 7 – IPMA: Importance-Performance Map",fontsize=12,fontweight="bold")
ax.legend(handles=[
    mpatches.Patch(color="#E74C3C",label="High Importance / Low Performance  -> Priority"),
    mpatches.Patch(color="#27AE60",label="High Importance / High Performance -> Maintain"),
    mpatches.Patch(color="#3498DB",label="Low Importance                     -> Monitor"),
],fontsize=9)
plt.tight_layout()
save_and_show("fig_ipma.png")


# =============================================================================
# 9. DEMOGRAPHICS
# =============================================================================
print("\n[9/9]  Demographic analysis ...")
fig, axes = plt.subplots(1,3,figsize=(14,4))
fig.suptitle("Figure 8 – Sample Demographics",fontsize=13,fontweight="bold")

size_counts = df["FirmSize"].value_counts()
axes[0].bar(size_counts.index,size_counts.values,color=["#3498DB","#E74C3C","#2ECC71"])
axes[0].set_title("Firm Size Distribution"); axes[0].set_ylabel("Count")
for i,v in enumerate(size_counts.values):
    axes[0].text(i,v+2,str(v),ha="center",fontsize=9)

reg_counts = df["Region"].value_counts()
axes[1].pie(reg_counts.values,labels=reg_counts.index,autopct="%1.1f%%",
            colors=["#3498DB","#E74C3C","#2ECC71","#F39C12"],startangle=140)
axes[1].set_title("Regional Distribution")

size_order=["Small","Medium","Large"]
sp_means=[df[df["FirmSize"]==s]["SP"].mean() for s in size_order]
b2=axes[2].bar(size_order,sp_means,color=["#3498DB","#E74C3C","#2ECC71"])
axes[2].set_title("Mean SP by Firm Size"); axes[2].set_ylabel("Mean SP Score")
axes[2].set_ylim(1,5)
for bar,val in zip(b2,sp_means):
    axes[2].text(bar.get_x()+bar.get_width()/2,val+0.03,
                 f"{val:.2f}",ha="center",fontsize=9)
plt.tight_layout()
save_and_show("fig_demographics.png")


# =============================================================================
print("\n"+"="*60)
print("  ALL DONE! Files saved in:")
print(f"  {OUT}")
print("="*60)
print("  8 figures shown on screen and saved as PNG")
print("  4 CSV files saved (data, SEM paths, CFA, importance)")
print("="*60)
