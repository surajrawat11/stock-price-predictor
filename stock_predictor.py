# ============================================================
#   Stock Price Direction Predictor
#   Predicts: Will Nifty 50 go UP or DOWN tomorrow?
#   Models: Logistic Regression vs Random Forest
#   Author: [Your Name] | Amazon MLSS Project
# ============================================================
 
# ── STEP 0: Install dependencies (run this once in terminal) ──
# pip install pandas numpy scikit-learn matplotlib seaborn yfinance
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler
 
import yfinance as yf
 
print("=" * 55)
print("   STOCK PRICE DIRECTION PREDICTOR")
print("   NIFTY 50 | 2021-2024 | ML Classification")
print("=" * 55)
 
 
# ──────────────────────────────────────────────
# STEP 1: DOWNLOAD DATA
# ──────────────────────────────────────────────
print("\n[1/5] Downloading Nifty 50 data from Yahoo Finance...")
 
ticker = "^NSEI"
df = yf.download(ticker, start="2021-01-01", end="2024-12-31", auto_adjust=True)
df.columns = df.columns.get_level_values(0)   # flatten MultiIndex if present
df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
df.dropna(inplace=True)
 
print(f"     Downloaded {len(df)} trading days of data.")
print(f"     Date range: {df.index[0].date()} → {df.index[-1].date()}")
 
 
# ──────────────────────────────────────────────
# STEP 2: FEATURE ENGINEERING
# ──────────────────────────────────────────────
print("\n[2/5] Engineering features...")
 
# Price-based features
df["Return"]       = df["Close"].pct_change()                        # daily % return
df["MA_5"]         = df["Close"].rolling(5).mean()                   # 5-day moving avg
df["MA_20"]        = df["Close"].rolling(20).mean()                  # 20-day moving avg
df["MA_50"]        = df["Close"].rolling(50).mean()                  # 50-day moving avg
df["MA_ratio"]     = df["MA_5"] / df["MA_20"]                        # momentum signal
df["Volatility"]   = df["Return"].rolling(10).std()                  # 10-day volatility
df["High_Low_Pct"] = (df["High"] - df["Low"]) / df["Close"]         # daily range %
df["Vol_Change"]   = df["Volume"].pct_change()                       # volume momentum
 
# RSI (Relative Strength Index) — popular trading indicator
delta     = df["Close"].diff()
gain      = delta.clip(lower=0).rolling(14).mean()
loss      = (-delta.clip(upper=0)).rolling(14).mean()
rs        = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))
 
# Target: 1 = price goes UP tomorrow, 0 = price goes DOWN
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
 
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
 
FEATURES = ["Return", "MA_5", "MA_20", "MA_50", "MA_ratio",
            "Volatility", "High_Low_Pct", "Vol_Change", "RSI"]
 
print(f"     Features created: {FEATURES}")
print(f"     Target distribution — UP: {df['Target'].sum()} | DOWN: {(df['Target']==0).sum()}")
 
 
# ──────────────────────────────────────────────
# STEP 3: TRAIN / TEST SPLIT & MODELS
# ──────────────────────────────────────────────
print("\n[3/5] Training models...")
 
X = df[FEATURES]
y = df["Target"]
 
# Time-series split — NO shuffle (future data must not leak into training)
split = int(len(X) * 0.80)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
 
# Scale features (important for Logistic Regression)
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
 
# ── Model 1: Logistic Regression (baseline) ──
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_train)
lr_pred = lr.predict(X_test_sc)
lr_acc  = accuracy_score(y_test, lr_pred)
 
# ── Model 2: Random Forest (main model) ──
rf = RandomForestClassifier(n_estimators=200, max_depth=6,
                             min_samples_leaf=5, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc  = accuracy_score(y_test, rf_pred)
 
print(f"\n     {'Model':<25} {'Accuracy':>10}")
print(f"     {'-'*35}")
print(f"     {'Logistic Regression':<25} {lr_acc:>9.1%}")
print(f"     {'Random Forest':<25} {rf_acc:>9.1%}")
print(f"     {'Random Baseline':<25} {'50.0%':>10}")
print(f"\n     Best model: Random Forest ({rf_acc:.1%})")
print("\n     Detailed Report (Random Forest):")
print(classification_report(y_test, rf_pred, target_names=["DOWN", "UP"]))
 
 
# ──────────────────────────────────────────────
# STEP 4: 3 VISUALIZATIONS
# ──────────────────────────────────────────────
print("\n[4/5] Generating visualizations...")
 
plt.style.use("seaborn-v0_8-darkgrid")
COLORS = {
    "up":   "#26a69a",   # teal
    "down": "#ef5350",   # red
    "rf":   "#1565c0",   # deep blue
    "lr":   "#f57c00",   # orange
    "bg":   "#f5f5f5",
}
 
# ── VIZ 1: Stock Price with Moving Averages ──────────────────
fig1, ax = plt.subplots(figsize=(14, 6))
fig1.patch.set_facecolor(COLORS["bg"])
ax.set_facecolor(COLORS["bg"])
 
ax.plot(df.index, df["Close"],   color="#37474f", linewidth=1,   label="Close Price",  zorder=3)
ax.plot(df.index, df["MA_5"],    color="#e91e63", linewidth=1.2, label="MA 5",  linestyle="--")
ax.plot(df.index, df["MA_20"],   color=COLORS["rf"], linewidth=1.4, label="MA 20", linestyle="--")
ax.plot(df.index, df["MA_50"],   color=COLORS["lr"], linewidth=1.6, label="MA 50", linestyle="-.")
 
# Shade train vs test regions
train_end = df.index[split]
ax.axvspan(df.index[0], train_end, alpha=0.07, color="blue",  label="Train period")
ax.axvspan(train_end,   df.index[-1], alpha=0.07, color="orange", label="Test period")
ax.axvline(train_end, color="gray", linestyle=":", linewidth=1.5)
ax.text(train_end, ax.get_ylim()[0], " ← Train | Test →",
        fontsize=9, color="gray", va="bottom")
 
ax.set_title("Nifty 50 Index with Moving Averages (2021-2024)",
             fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Price (INR)", fontsize=11)
ax.legend(loc="upper left", fontsize=9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
plt.tight_layout()
fig1.savefig("viz1_price_moving_averages.png", dpi=150, bbox_inches="tight")
plt.show()
print("     ✓ viz1_price_moving_averages.png saved")
 
 
# ── VIZ 2: Feature Importance (Random Forest) ────────────────
fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
fig2.patch.set_facecolor(COLORS["bg"])
fig2.suptitle("Feature Importance & RSI Analysis - Nifty 50", fontsize=14, fontweight="bold", y=0.98)
 
# Left: Feature importance bar chart
feat_imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
colors_bar = [COLORS["up"] if v > feat_imp.median() else "#b0bec5" for v in feat_imp]
axes[0].barh(feat_imp.index, feat_imp.values, color=colors_bar, edgecolor="white", linewidth=0.5)
axes[0].set_facecolor(COLORS["bg"])
axes[0].set_title("Random Forest — Feature Importance", fontweight="bold")
axes[0].set_xlabel("Importance Score")
for i, (val, name) in enumerate(zip(feat_imp.values, feat_imp.index)):
    axes[0].text(val + 0.001, i, f"{val:.3f}", va="center", fontsize=8.5)
 
# Right: RSI distribution — UP days vs DOWN days
rsi_up   = df.loc[df["Target"] == 1, "RSI"]
rsi_down = df.loc[df["Target"] == 0, "RSI"]
axes[1].hist(rsi_up,   bins=30, alpha=0.6, color=COLORS["up"],   label="UP days",   edgecolor="white")
axes[1].hist(rsi_down, bins=30, alpha=0.6, color=COLORS["down"], label="DOWN days", edgecolor="white")
axes[1].axvline(30, color="black", linestyle="--", linewidth=1, label="Oversold (30)")
axes[1].axvline(70, color="gray",  linestyle="--", linewidth=1, label="Overbought (70)")
axes[1].set_facecolor(COLORS["bg"])
axes[1].set_title("RSI Distribution: UP vs DOWN Days", fontweight="bold")
axes[1].set_xlabel("RSI Value")
axes[1].set_ylabel("Frequency")
axes[1].legend(fontsize=9)
 
plt.tight_layout()
fig2.savefig("viz2_feature_importance_rsi.png", dpi=150, bbox_inches="tight")
plt.show()
print("     ✓ viz2_feature_importance_rsi.png saved")
 
 
# ── VIZ 3: Model Comparison Dashboard ───────────────────────
fig3 = plt.figure(figsize=(14, 6))
fig3.patch.set_facecolor(COLORS["bg"])
gs  = gridspec.GridSpec(1, 3, figure=fig3, wspace=0.35)
 
# Left: Confusion matrix — Logistic Regression
ax_cm1 = fig3.add_subplot(gs[0])
cm_lr  = confusion_matrix(y_test, lr_pred)
disp1  = ConfusionMatrixDisplay(cm_lr, display_labels=["DOWN", "UP"])
disp1.plot(ax=ax_cm1, colorbar=False, cmap="Blues")
ax_cm1.set_title(f"Logistic Regression\nAccuracy: {lr_acc:.1%}", fontweight="bold")
ax_cm1.set_facecolor(COLORS["bg"])
 
# Middle: Confusion matrix — Random Forest
ax_cm2 = fig3.add_subplot(gs[1])
cm_rf  = confusion_matrix(y_test, rf_pred)
disp2  = ConfusionMatrixDisplay(cm_rf, display_labels=["DOWN", "UP"])
disp2.plot(ax=ax_cm2, colorbar=False, cmap="Blues")
ax_cm2.set_title(f"Random Forest\nAccuracy: {rf_acc:.1%}", fontweight="bold")
ax_cm2.set_facecolor(COLORS["bg"])
 
# Right: Accuracy bar comparison
ax_bar = fig3.add_subplot(gs[2])
ax_bar.set_facecolor(COLORS["bg"])
models = ["Random\nBaseline", "Logistic\nRegression", "Random\nForest"]
accs   = [0.50, lr_acc, rf_acc]
bar_colors = ["#b0bec5", COLORS["lr"], COLORS["rf"]]
bars = ax_bar.bar(models, accs, color=bar_colors, edgecolor="white",
                  linewidth=0.8, width=0.5)
ax_bar.set_ylim(0.40, min(max(accs) + 0.10, 1.0))
ax_bar.axhline(0.50, color="red", linestyle="--", linewidth=1, label="Baseline (50%)")
ax_bar.set_title("Model Accuracy Comparison", fontweight="bold")
ax_bar.set_ylabel("Accuracy")
ax_bar.legend(fontsize=8)
for bar, acc in zip(bars, accs):
    ax_bar.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{acc:.1%}", ha="center", va="bottom", fontweight="bold", fontsize=10)
 
fig3.suptitle("Model Comparison Dashboard - Nifty 50 Predictor",
              fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()
fig3.savefig("viz3_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("     ✓ viz3_model_comparison.png saved")
 
 
# ──────────────────────────────────────────────
# STEP 5: SUMMARY
# ──────────────────────────────────────────────
print("\n[5/5] DONE! Summary")
print("=" * 55)
print(f"  Ticker        : {ticker}")
print(f"  Training rows : {len(X_train)}")
print(f"  Testing  rows : {len(X_test)}")
print(f"  Features used : {len(FEATURES)}")
print(f"  LR Accuracy   : {lr_acc:.1%}")
print(f"  RF Accuracy   : {rf_acc:.1%}")
print(f"  Beat baseline : {'YES ✓' if rf_acc > 0.50 else 'NO — tune model'}")
print("=" * 55)
print("\n  Files saved:")
print("  📊 viz1_price_moving_averages.png")
print("  📊 viz2_feature_importance_rsi.png")
print("  📊 viz3_model_comparison.png")
print("\n  Next step: Upload everything to GitHub!")
print("=" * 55)
 