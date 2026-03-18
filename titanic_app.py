import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, auc,
    classification_report, precision_score, recall_score, f1_score
)
from sklearn.model_selection import cross_val_score
from sklearn.tree import plot_tree
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic ML Dashboard",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Dark nautical theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1525 50%, #0a1020 100%);
    color: #e8e4d8;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; max-width: 1400px; }

/* Hero header */
.hero {
    background: linear-gradient(135deg, rgba(201,168,76,0.12), rgba(139,58,47,0.08));
    border: 1px solid rgba(201,168,76,0.3);
    border-radius: 4px;
    padding: 36px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 80% 50%, rgba(26,39,68,0.4), transparent 70%);
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem; font-weight: 900;
    color: #f0e8d0; margin: 0 0 8px; line-height: 1.1;
}
.hero h1 span { color: #c9a84c; }
.hero p { color: #8899aa; font-size: 1rem; margin: 0; font-weight: 300; }

/* Metric cards */
.metric-row { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 140px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(201,168,76,0.2);
    border-radius: 4px; padding: 20px 24px;
    text-align: center;
}
.metric-card .val {
    font-family: 'Playfair Display', serif;
    font-size: 2rem; font-weight: 700; color: #c9a84c;
}
.metric-card .lbl { font-size: 0.72rem; letter-spacing: 0.15em; text-transform: uppercase; color: #6b7fa3; margin-top: 4px; }

/* Result cards */
.survived-card {
    background: linear-gradient(135deg, rgba(46,90,60,0.25), rgba(46,90,60,0.1));
    border: 2px solid rgba(46,150,80,0.5);
    border-radius: 4px; padding: 28px 32px; text-align: center;
}
.perished-card {
    background: linear-gradient(135deg, rgba(139,58,47,0.25), rgba(139,58,47,0.1));
    border: 2px solid rgba(180,60,50,0.5);
    border-radius: 4px; padding: 28px 32px; text-align: center;
}
.result-icon { font-size: 3rem; margin-bottom: 8px; }
.result-verdict { font-family: 'Playfair Display', serif; font-size: 1.8rem; font-weight: 700; }
.survived-card .result-verdict { color: #5adb7a; }
.perished-card .result-verdict { color: #e06050; }
.result-prob { color: #8899aa; font-size: 0.95rem; margin-top: 6px; }

/* Section headers */
.section-hdr {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem; font-weight: 700; color: #c9a84c;
    border-bottom: 1px solid rgba(201,168,76,0.2);
    padding-bottom: 10px; margin-bottom: 20px; margin-top: 8px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 4px; padding: 4px; gap: 4px;
    border: 1px solid rgba(201,168,76,0.15);
}
.stTabs [data-baseweb="tab"] {
    color: #8899aa !important; border-radius: 3px !important;
    font-size: 0.85rem; letter-spacing: 0.05em; font-weight: 500;
    padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(201,168,76,0.15) !important;
    color: #c9a84c !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1525 0%, #0a1020 100%) !important;
    border-right: 1px solid rgba(201,168,76,0.15);
}
section[data-testid="stSidebar"] .block-container { padding: 1.5rem; }

/* Buttons */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #1a2744, #243360) !important;
    color: #f0e8d0 !important;
    border: 1px solid rgba(201,168,76,0.4) !important;
    border-radius: 3px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important; letter-spacing: 0.08em;
    padding: 12px 24px !important; font-size: 0.9rem !important;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #243360, #2e4080) !important;
    border-color: rgba(201,168,76,0.7) !important;
}

/* Selectbox / slider labels */
label { color: #8899aa !important; font-size: 0.8rem !important; letter-spacing: 0.1em; text-transform: uppercase; }
.stSlider > div > div > div { background: #c9a84c !important; }
.stSelectbox > div > div { background: rgba(255,255,255,0.05) !important; border-color: rgba(201,168,76,0.25) !important; color: #e8e4d8 !important; }
.stDataFrame { border: 1px solid rgba(201,168,76,0.15) !important; border-radius: 4px; }

/* Divider */
hr { border-color: rgba(201,168,76,0.15) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("titanic.csv")
    return df

@st.cache_resource
def load_model():
    return joblib.load("model_v2.pkl")

@st.cache_data
def engineer_features(df):
    data = df.copy()
    data['title'] = data['name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    data['title'] = data['title'].replace(
        ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
    data['title'] = data['title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
    data['family_size'] = data['sibsp'] + data['parch'] + 1
    data['is_alone'] = (data['family_size'] == 1).astype(int)
    data['has_cabin'] = data['cabin'].notna().astype(int)
    data['cabin_letter'] = data['cabin'].str[0].fillna('U')
    data['age'] = data.groupby(['pclass','sex','title'])['age'].transform(lambda x: x.fillna(x.median()))
    data['age'] = data['age'].fillna(data['age'].median())
    data['fare'] = data.groupby('pclass')['fare'].transform(lambda x: x.fillna(x.median()))
    data['embarked'] = data['embarked'].fillna('S')
    data['age_bin'] = pd.cut(data['age'], bins=[0,12,18,35,60,100], labels=[0,1,2,3,4]).astype(int)
    data['fare_bin'] = pd.qcut(data['fare'], q=4, labels=[0,1,2,3]).astype(int)
    bundle = load_model()
    data['sex_enc'] = bundle['le_sex'].transform(data['sex'])
    data['embarked_enc'] = bundle['le_emb'].transform(data['embarked'])
    data['title_enc'] = bundle['le_title'].transform(data['title'].where(
        data['title'].isin(bundle['le_title'].classes_), 'Rare'))
    data['cabin_enc'] = bundle['le_cabin'].transform(data['cabin_letter'].where(
        data['cabin_letter'].isin(bundle['le_cabin'].classes_), 'U'))
    return data

# ─────────────────────────────────────────────
# MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────
def set_dark_style():
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#0d1525',
        'axes.facecolor': '#111a2e',
        'axes.edgecolor': '#2a3a5a',
        'axes.labelcolor': '#8899aa',
        'xtick.color': '#8899aa',
        'ytick.color': '#8899aa',
        'grid.color': '#1e2d45',
        'grid.alpha': 0.5,
        'text.color': '#e8e4d8',
        'font.family': 'sans-serif',
    })

GOLD    = '#c9a84c'
TEAL    = '#2eb8a0'
RUST    = '#e06050'
BLUE    = '#4a90d9'
PURPLE  = '#9b6dd6'
GREEN   = '#5adb7a'
PALETTE = [GOLD, TEAL, RUST, BLUE, PURPLE, GREEN]

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
df_raw  = load_data()
bundle  = load_model()
df      = engineer_features(df_raw)
FEATURES = bundle['features']
FEATURE_LABELS = bundle['feature_labels']
X = df[FEATURES]
y = df['survived']

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🚢 Titanic <span>Survival</span> ML Dashboard</h1>
  <p>Advanced machine learning analysis · Random Forest · Gradient Boosting · Decision Tree · 14 features</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR — ALL passenger inputs
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Passenger Profile")
    st.markdown("---")

    st.markdown("**Demographics**")
    sex = st.selectbox("Sex", ["female", "male"], index=1)
    age = st.slider("Age", min_value=0, max_value=80, value=29, step=1)
    title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"], index=0)

    st.markdown("---")
    st.markdown("**Journey Details**")
    pclass = st.selectbox("Passenger Class", [1, 2, 3],
                          format_func=lambda x: f"{'First' if x==1 else 'Second' if x==2 else 'Third'} Class (Class {x})",
                          index=2)
    fare = st.slider("Fare (£)", min_value=0.0, max_value=515.0, value=32.5, step=0.5)
    embarked = st.selectbox("Port of Embarkation",
                            ["S","C","Q"],
                            format_func=lambda x: {"S":"Southampton","C":"Cherbourg","Q":"Queenstown"}[x])
    has_cabin = st.selectbox("Has Cabin Info?", [0,1], format_func=lambda x: "Yes" if x else "No")
    cabin_letter = st.selectbox("Cabin Deck", ["A","B","C","D","E","F","G","T","U"],
                                format_func=lambda x: f"Deck {x}" if x != "U" else "Unknown")

    st.markdown("---")
    st.markdown("**Family Aboard**")
    sibsp = st.slider("Siblings / Spouses", 0, 8, 0)
    parch = st.slider("Parents / Children", 0, 9, 0)

    st.markdown("---")
    st.markdown("**Model Selection**")
    model_choice = st.selectbox("Classifier", ["Random Forest", "Gradient Boosting", "Decision Tree"])

    st.markdown("---")
    predict_btn = st.button("⚓  Predict Survival", use_container_width=True)

# ─────────────────────────────────────────────
# BUILD PREDICTION INPUT
# ─────────────────────────────────────────────
def build_input():
    family_size = sibsp + parch + 1
    is_alone    = int(family_size == 1)
    age_bin     = int(pd.cut([age], bins=[0,12,18,35,60,100], labels=[0,1,2,3,4])[0])
    fare_bin    = int(min(3, max(0, int(fare / 130))))

    sex_enc  = bundle['le_sex'].transform([sex])[0]
    emb_enc  = bundle['le_emb'].transform([embarked])[0]
    title_v  = title if title in bundle['le_title'].classes_ else 'Rare'
    title_enc = bundle['le_title'].transform([title_v])[0]
    cabin_v   = cabin_letter if cabin_letter in bundle['le_cabin'].classes_ else 'U'
    cabin_enc = bundle['le_cabin'].transform([cabin_v])[0]

    return [[pclass, sex_enc, age, sibsp, parch, fare, emb_enc,
             title_enc, family_size, is_alone, has_cabin, cabin_enc,
             age_bin, fare_bin]]

model_map = {
    "Random Forest": bundle['rf'],
    "Gradient Boosting": bundle['gb'],
    "Decision Tree": bundle['dt'],
}
selected_model = model_map[model_choice]

# ─────────────────────────────────────────────
# TOP METRICS
# ─────────────────────────────────────────────
y_pred   = selected_model.predict(X)
y_proba  = selected_model.predict_proba(X)[:,1]
acc      = accuracy_score(y, y_pred)
prec     = precision_score(y, y_pred)
rec      = recall_score(y, y_pred)
f1       = f1_score(y, y_pred)
surv_pct = y.mean() * 100

st.markdown(f"""
<div class="metric-row">
  <div class="metric-card"><div class="val">{acc:.1%}</div><div class="lbl">Accuracy</div></div>
  <div class="metric-card"><div class="val">{prec:.1%}</div><div class="lbl">Precision</div></div>
  <div class="metric-card"><div class="val">{rec:.1%}</div><div class="lbl">Recall</div></div>
  <div class="metric-card"><div class="val">{f1:.1%}</div><div class="lbl">F1 Score</div></div>
  <div class="metric-card"><div class="val">{len(df)}</div><div class="lbl">Passengers</div></div>
  <div class="metric-card"><div class="val">{surv_pct:.0f}%</div><div class="lbl">Survived</div></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯  Prediction",
    "📊  Data Analysis",
    "🔬  Model Performance",
    "🌲  Feature Insights",
    "📋  Dataset"
])

# ══════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════
with tab1:
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown('<div class="section-hdr">Passenger Summary</div>', unsafe_allow_html=True)
        summary_data = {
            "Attribute": ["Sex","Age","Title","Class","Fare","Embarked","Siblings/Spouses",
                          "Parents/Children","Family Size","Has Cabin","Cabin Deck"],
            "Value": [sex.capitalize(), age, title,
                      f"{'First' if pclass==1 else 'Second' if pclass==2 else 'Third'} Class",
                      f"£{fare:.2f}", {"S":"Southampton","C":"Cherbourg","Q":"Queenstown"}[embarked],
                      sibsp, parch, sibsp+parch+1, "Yes" if has_cabin else "No",
                      f"Deck {cabin_letter}" if cabin_letter != "U" else "Unknown"]
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

        # Similar passengers survival rate
        mask = (
            (df['pclass'] == pclass) &
            (df['sex'] == sex) &
            (df['age'].between(max(0, age-10), age+10))
        )
        similar = df[mask]
        if len(similar) > 0:
            sim_rate = similar['survived'].mean() * 100
            st.info(f"📌 Among **{len(similar)}** similar passengers (same class, sex, ±10 yrs age): **{sim_rate:.0f}%** survived.")

    with col_r:
        st.markdown('<div class="section-hdr">Prediction Result</div>', unsafe_allow_html=True)
        if predict_btn:
            inp = build_input()
            pred   = selected_model.predict(inp)[0]
            proba  = selected_model.predict_proba(inp)[0]
            surv_p = proba[1] * 100

            if pred == 1:
                st.markdown(f"""
                <div class="survived-card">
                  <div class="result-icon">🛟</div>
                  <div class="result-verdict">SURVIVED</div>
                  <div class="result-prob">Survival probability: <strong>{surv_p:.1f}%</strong></div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="perished-card">
                  <div class="result-icon">🌊</div>
                  <div class="result-verdict">DID NOT SURVIVE</div>
                  <div class="result-prob">Survival probability: <strong>{surv_p:.1f}%</strong></div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Probability gauge
            set_dark_style()
            fig, ax = plt.subplots(figsize=(6, 1.2))
            ax.barh(0, 100, color='#1e2d45', height=0.5)
            bar_color = GREEN if pred == 1 else RUST
            ax.barh(0, surv_p, color=bar_color, height=0.5)
            ax.axvline(50, color='#c9a84c', linewidth=1.5, linestyle='--', alpha=0.7)
            ax.set_xlim(0, 100)
            ax.set_yticks([])
            ax.set_xlabel("Survival Probability %", fontsize=9)
            ax.set_title(f"{surv_p:.1f}% Survival Probability", fontsize=10, color='#e8e4d8')
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Prediction with all 3 models
            st.markdown("**Predictions across all models:**")
            model_results = []
            for mname, mobj in model_map.items():
                mp = mobj.predict(inp)[0]
                mprob = mobj.predict_proba(inp)[0][1] * 100
                model_results.append({"Model": mname,
                                       "Prediction": "✅ Survived" if mp == 1 else "❌ Not Survived",
                                       "Survival %": f"{mprob:.1f}%"})
            st.dataframe(pd.DataFrame(model_results), use_container_width=True, hide_index=True)
        else:
            st.markdown("""
            <div style="text-align:center; padding:60px 20px; color:#4a5a7a;">
              <div style="font-size:3rem">⚓</div>
              <div style="font-size:1rem; margin-top:12px;">Set passenger details in the sidebar<br>and click <strong>Predict Survival</strong></div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════
# TAB 2 — DATA ANALYSIS
# ══════════════════════════════
with tab2:
    set_dark_style()

    # Row 1
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="section-hdr">Survival Overview</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4,3.5))
        counts = df['survived'].value_counts()
        wedges, texts, autotexts = ax.pie(
            counts, labels=["Perished","Survived"],
            colors=[RUST, GREEN], autopct='%1.1f%%',
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(width=0.55, edgecolor='#0d1525', linewidth=2)
        )
        for t in texts: t.set_color('#8899aa'); t.set_fontsize(9)
        for t in autotexts: t.set_color('#e8e4d8'); t.set_fontsize(9); t.set_fontweight('bold')
        ax.set_title("Overall Survival Rate", fontsize=10, color='#c9a84c', pad=10)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    with c2:
        st.markdown('<div class="section-hdr">Survival by Class</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4,3.5))
        ct = df.groupby('pclass')['survived'].mean() * 100
        bars = ax.bar(['1st','2nd','3rd'], ct.values, color=[GREEN, GOLD, RUST], width=0.55, edgecolor='#0d1525', linewidth=1.5)
        for bar, val in zip(bars, ct.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{val:.0f}%',
                    ha='center', va='bottom', fontsize=9, color='#e8e4d8')
        ax.set_ylabel("Survival Rate (%)", fontsize=9)
        ax.set_title("By Passenger Class", fontsize=10, color='#c9a84c', pad=10)
        ax.grid(axis='y', alpha=0.3); ax.set_ylim(0, max(ct.values)+15)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    with c3:
        st.markdown('<div class="section-hdr">Survival by Sex</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4,3.5))
        ct = df.groupby('sex')['survived'].mean() * 100
        bars = ax.bar(['Female','Male'], ct.values, color=[TEAL, BLUE], width=0.45, edgecolor='#0d1525', linewidth=1.5)
        for bar, val in zip(bars, ct.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{val:.0f}%',
                    ha='center', va='bottom', fontsize=9, color='#e8e4d8')
        ax.set_ylabel("Survival Rate (%)", fontsize=9)
        ax.set_title("By Sex", fontsize=10, color='#c9a84c', pad=10)
        ax.grid(axis='y', alpha=0.3); ax.set_ylim(0, max(ct.values)+15)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    # Row 2
    c4, c5 = st.columns(2)

    with c4:
        st.markdown('<div class="section-hdr">Age Distribution by Survival</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        survived  = df[df['survived']==1]['age'].dropna()
        perished  = df[df['survived']==0]['age'].dropna()
        ax.hist(perished, bins=25, alpha=0.65, color=RUST,  label='Perished',  edgecolor='#0d1525')
        ax.hist(survived, bins=25, alpha=0.65, color=GREEN, label='Survived',  edgecolor='#0d1525')
        ax.axvline(survived.mean(),  color=GREEN, linewidth=1.5, linestyle='--', alpha=0.8)
        ax.axvline(perished.mean(),  color=RUST,  linewidth=1.5, linestyle='--', alpha=0.8)
        ax.set_xlabel("Age"); ax.set_ylabel("Count")
        ax.set_title("Age Distribution", fontsize=10, color='#c9a84c', pad=10)
        ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with c5:
        st.markdown('<div class="section-hdr">Fare Distribution by Survival</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        df_fare = df[df['fare'] < 300]
        survived_f = df_fare[df_fare['survived']==1]['fare']
        perished_f = df_fare[df_fare['survived']==0]['fare']
        ax.hist(perished_f, bins=30, alpha=0.65, color=RUST,  label='Perished', edgecolor='#0d1525')
        ax.hist(survived_f, bins=30, alpha=0.65, color=GREEN, label='Survived', edgecolor='#0d1525')
        ax.set_xlabel("Fare (£)"); ax.set_ylabel("Count")
        ax.set_title("Fare Distribution (<£300)", fontsize=10, color='#c9a84c', pad=10)
        ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    # Row 3
    c6, c7 = st.columns(2)

    with c6:
        st.markdown('<div class="section-hdr">Survival by Title</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        title_surv = df.groupby('title')['survived'].agg(['mean','count']).reset_index()
        title_surv = title_surv[title_surv['count'] >= 5].sort_values('mean', ascending=True)
        colors = [GREEN if v > 0.5 else RUST for v in title_surv['mean']]
        bars = ax.barh(title_surv['title'], title_surv['mean']*100, color=colors, edgecolor='#0d1525', height=0.6)
        ax.axvline(50, color=GOLD, linewidth=1, linestyle='--', alpha=0.6)
        ax.set_xlabel("Survival Rate (%)"); ax.grid(axis='x', alpha=0.3)
        ax.set_title("By Passenger Title", fontsize=10, color='#c9a84c', pad=10)
        for bar, row in zip(bars, title_surv.itertuples()):
            ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                    f'{row.mean*100:.0f}% (n={row.count})', va='center', fontsize=8, color='#8899aa')
        ax.set_xlim(0, 120)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with c7:
        st.markdown('<div class="section-hdr">Family Size vs Survival</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        fam_surv = df.groupby('family_size')['survived'].mean() * 100
        fam_count = df.groupby('family_size')['survived'].count()
        colors = [GREEN if v > 50 else RUST for v in fam_surv.values]
        bars = ax.bar(fam_surv.index, fam_surv.values, color=colors, edgecolor='#0d1525', width=0.6)
        ax.axhline(50, color=GOLD, linewidth=1, linestyle='--', alpha=0.6)
        ax.set_xlabel("Family Size"); ax.set_ylabel("Survival Rate (%)")
        ax.set_title("Survival by Family Size", fontsize=10, color='#c9a84c', pad=10)
        ax.grid(axis='y', alpha=0.3)
        for bar, (fs, cnt) in zip(bars, fam_count.items()):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    f'n={cnt}', ha='center', va='bottom', fontsize=7, color='#8899aa')
        fig.tight_layout(); st.pyplot(fig); plt.close()

    # Row 4 - Heatmap
    st.markdown('<div class="section-hdr">Survival Rate Heatmap: Class × Embarkation</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 3))
    pivot = df.pivot_table('survived', index='pclass', columns='embarked', aggfunc='mean') * 100
    pivot.index = ['1st','2nd','3rd']
    pivot.columns = ['Cherbourg','Queenstown','Southampton']
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn',
                linewidths=1, linecolor='#0d1525',
                ax=ax, vmin=0, vmax=100,
                annot_kws={'fontsize':11, 'fontweight':'bold'})
    ax.set_title("Survival Rate % by Class & Port", fontsize=10, color='#c9a84c', pad=12)
    ax.set_ylabel("Passenger Class"); ax.set_xlabel("")
    fig.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════
with tab3:
    set_dark_style()
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-hdr">Confusion Matrix</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Dead','Predicted Survived'],
                    yticklabels=['Actually Dead','Actually Survived'],
                    ax=ax, linewidths=2, linecolor='#0d1525',
                    annot_kws={'fontsize':13, 'fontweight':'bold'})
        ax.set_title(f"Confusion Matrix — {model_choice}", fontsize=10, color='#c9a84c', pad=10)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        st.markdown('<div class="section-hdr">ROC Curve</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        for mname, mobj in model_map.items():
            mp = mobj.predict_proba(X)[:,1]
            fpr, tpr, _ = roc_curve(y, mp)
            roc_auc_val = auc(fpr, tpr)
            color = {'Random Forest':GREEN, 'Gradient Boosting':GOLD, 'Decision Tree':BLUE}[mname]
            ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{mname} (AUC={roc_auc_val:.3f})')
        ax.plot([0,1],[0,1], color='#4a5a7a', linewidth=1, linestyle='--', label='Random')
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — All Models", fontsize=10, color='#c9a84c', pad=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    # Model comparison table
    st.markdown('<div class="section-hdr">Model Comparison</div>', unsafe_allow_html=True)
    rows = []
    for mname, mobj in model_map.items():
        mp = mobj.predict(X)
        mprob = mobj.predict_proba(X)[:,1]
        fpr, tpr, _ = roc_curve(y, mprob)
        rows.append({
            "Model": mname,
            "Accuracy": f"{accuracy_score(y,mp):.4f}",
            "Precision": f"{precision_score(y,mp):.4f}",
            "Recall": f"{recall_score(y,mp):.4f}",
            "F1 Score": f"{f1_score(y,mp):.4f}",
            "AUC-ROC": f"{auc(fpr,tpr):.4f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Learning-style: survival probability distribution
    st.markdown('<div class="section-hdr">Predicted Probability Distribution</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.hist(y_proba[y==0], bins=40, alpha=0.65, color=RUST,  label='Actually Perished', edgecolor='#0d1525')
    ax.hist(y_proba[y==1], bins=40, alpha=0.65, color=GREEN, label='Actually Survived', edgecolor='#0d1525')
    ax.axvline(0.5, color=GOLD, linewidth=2, linestyle='--', label='Decision Boundary (0.5)')
    ax.set_xlabel("Predicted Survival Probability"); ax.set_ylabel("Count")
    ax.set_title(f"Probability Distribution — {model_choice}", fontsize=10, color='#c9a84c', pad=10)
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════
# TAB 4 — FEATURE INSIGHTS
# ══════════════════════════════
with tab4:
    set_dark_style()
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-hdr">Feature Importance</div>', unsafe_allow_html=True)
        importances = selected_model.feature_importances_
        feat_df = pd.DataFrame({'Feature': FEATURE_LABELS, 'Importance': importances})
        feat_df = feat_df.sort_values('Importance', ascending=True)

        fig, ax = plt.subplots(figsize=(6, 6))
        colors = [GOLD if i > feat_df['Importance'].median() else TEAL for i in feat_df['Importance']]
        bars = ax.barh(feat_df['Feature'], feat_df['Importance']*100, color=colors, edgecolor='#0d1525', height=0.65)
        ax.set_xlabel("Importance (%)")
        ax.set_title(f"Feature Importance — {model_choice}", fontsize=10, color='#c9a84c', pad=10)
        ax.grid(axis='x', alpha=0.3)
        for bar, val in zip(bars, feat_df['Importance']):
            ax.text(bar.get_width()+0.2, bar.get_y()+bar.get_height()/2,
                    f'{val*100:.1f}%', va='center', fontsize=8, color='#8899aa')
        ax.set_xlim(0, feat_df['Importance'].max()*100 + 8)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        st.markdown('<div class="section-hdr">Survival Rate by Feature</div>', unsafe_allow_html=True)
        feature_pick = st.selectbox("Select feature to explore", [
            "pclass", "sex", "embarked", "title", "family_size", "is_alone",
            "has_cabin", "age_bin", "fare_bin", "cabin_letter"
        ], format_func=lambda x: {
            "pclass":"Passenger Class","sex":"Sex","embarked":"Embarkation",
            "title":"Title","family_size":"Family Size","is_alone":"Traveling Alone",
            "has_cabin":"Has Cabin","age_bin":"Age Group","fare_bin":"Fare Group",
            "cabin_letter":"Cabin Deck"
        }.get(x, x))

        label_map = {
            "age_bin":  {0:"Child",1:"Teen",2:"Young Adult",3:"Middle Aged",4:"Senior"},
            "fare_bin": {0:"Low",1:"Medium",2:"High",3:"Very High"},
            "is_alone": {0:"With Family",1:"Alone"},
            "has_cabin":{0:"No Cabin",1:"Has Cabin"},
            "embarked": {"S":"Southampton","C":"Cherbourg","Q":"Queenstown"},
        }

        grp = df.groupby(feature_pick)['survived'].agg(['mean','count']).reset_index()
        grp.columns = ['value','rate','count']
        grp = grp[grp['count'] >= 5].sort_values('rate', ascending=True)

        if feature_pick in label_map:
            grp['label'] = grp['value'].map(label_map[feature_pick]).fillna(grp['value'].astype(str))
        else:
            grp['label'] = grp['value'].astype(str)

        fig, ax = plt.subplots(figsize=(6, max(3.5, len(grp)*0.6)))
        colors = [GREEN if v > 0.5 else RUST for v in grp['rate']]
        bars = ax.barh(grp['label'], grp['rate']*100, color=colors, edgecolor='#0d1525', height=0.6)
        ax.axvline(50, color=GOLD, linewidth=1, linestyle='--', alpha=0.6)
        ax.set_xlabel("Survival Rate (%)")
        ax.set_title(f"Survival by {feature_pick.replace('_',' ').title()}", fontsize=10, color='#c9a84c', pad=10)
        ax.grid(axis='x', alpha=0.3)
        for bar, row in zip(bars, grp.itertuples()):
            ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                    f'{row.rate*100:.0f}% (n={row.count})', va='center', fontsize=8, color='#8899aa')
        ax.set_xlim(0, 120)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    # Decision tree visualization
    if model_choice == "Decision Tree":
        st.markdown('<div class="section-hdr">Decision Tree Visualization</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(20, 8))
        plot_tree(
            bundle['dt'],
            feature_names=FEATURE_LABELS,
            class_names=["Perished","Survived"],
            filled=True, rounded=True, fontsize=7, ax=ax,
            impurity=False, proportion=True
        )
        ax.set_title("Decision Tree Structure (max_depth=6)", fontsize=12, color='#c9a84c', pad=10)
        fig.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════
# TAB 5 — DATASET
# ══════════════════════════════
with tab5:
    st.markdown('<div class="section-hdr">Full Dataset Explorer</div>', unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        f_class  = st.multiselect("Filter by Class", [1,2,3], default=[1,2,3])
    with col_f2:
        f_sex    = st.multiselect("Filter by Sex", ["male","female"], default=["male","female"])
    with col_f3:
        f_surv   = st.multiselect("Filter by Survived", [0,1],
                                   format_func=lambda x: "Survived" if x==1 else "Perished",
                                   default=[0,1])

    filtered = df_raw[
        (df_raw['pclass'].isin(f_class)) &
        (df_raw['sex'].isin(f_sex)) &
        (df_raw['survived'].isin(f_surv))
    ]

    display_cols = ['name','survived','pclass','sex','age','fare','embarked','sibsp','parch','cabin']
    st.markdown(f"**{len(filtered):,} passengers** match your filters")
    st.dataframe(
        filtered[display_cols].rename(columns={
            'name':'Name','survived':'Survived','pclass':'Class','sex':'Sex',
            'age':'Age','fare':'Fare (£)','embarked':'Embarked',
            'sibsp':'Siblings/Spouses','parch':'Parents/Children','cabin':'Cabin'
        }),
        use_container_width=True, height=420
    )

    # Quick stats
    st.markdown('<div class="section-hdr">Filtered Statistics</div>', unsafe_allow_html=True)
    if len(filtered) > 0:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Survival Rate", f"{filtered['survived'].mean():.1%}")
        c2.metric("Avg Age", f"{filtered['age'].mean():.1f} yrs")
        c3.metric("Avg Fare", f"£{filtered['fare'].mean():.2f}")
        c4.metric("Avg Family Size", f"{(filtered['sibsp']+filtered['parch']+1).mean():.1f}")
