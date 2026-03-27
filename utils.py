"""utils.py - Book DNA shared helpers. No cache decorators (avoids Streamlit Cloud hashing errors)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix,
    mean_squared_error, r2_score)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

CLUSTER_COLORS = {
    "Midnight Escapist":     "#5C3D8F",
    "Productivity Achiever": "#C8922A",
    "Emotional Reader":      "#D45379",
    "Curious Explorer":      "#1A6B5A",
    "Non-Reader":            "#888780",
}
PALETTE = list(CLUSTER_COLORS.values())

AGE_LABELS  = {2:"13-17",3:"18-22",4:"23-28",5:"29-35",6:"36-45",7:"46+"}
CITY_LABELS = {1:"Metro",2:"Tier 1",3:"Tier 2",4:"Tier 3",5:"Rural"}
GENDER_LABELS = {1:"Male",2:"Female",3:"Non-binary",4:"Prefer not say"}
OCC_LABELS  = {1:"School student",2:"College student",3:"Working professional",
               4:"Freelancer",5:"Homemaker",6:"Other"}
FORMAT_CLASS = {1:"Physical",2:"Mixed",3:"Digital"}
RH_LABELS   = {1:"Active reader",2:"Want to return",3:"Building habit",4:"Occasional",5:"Not my thing"}
PAY_LABELS  = {1:"UPI",2:"Debit card",3:"Credit card",4:"COD",5:"BNPL"}
DISC_LABELS = {1:"Flat % off",2:"Bundle deal",3:"Free shipping",
               4:"Loyalty pts",5:"Festival sale",6:"First-buyer offer"}

PRODUCT_COLS  = ["interest_books","interest_journal","interest_bookmark","interest_candle",
                 "interest_tote","interest_apparel","interest_decor","interest_annotation","interest_pin"]
PRODUCT_NAMES = ["Book bundles","Journals","Bookmarks","Candles",
                 "Tote bags","Apparel","Shelf decor","Annotation kits","Enamel pins"]

CLUSTER_FEATURES = ["openness_score","conscientiousness_score","extraversion_score",
    "agreeableness_score","neuroticism_score","stress_score","books_per_month",
    "social_sharing_level","lifestyle_spend","genre_fantasy","genre_selfhelp",
    "genre_literary","genre_romance","genre_thriller","genre_biography","genre_business",
    "reading_motivation","reader_identity","personalization_importance",
    "nps_proxy","monthly_income_midpoint"]

CLF_FEATURES = ["age_group","city_tier","monthly_income_midpoint","stress_score",
    "openness_score","conscientiousness_score","neuroticism_score","agreeableness_score",
    "books_per_month","social_sharing_level","nps_proxy","switching_tendency",
    "personalization_importance","barrier_price","barrier_trust","barrier_delivery",
    "subscription_interest","lifestyle_spend","impulse_buying",
    "discovery_social","shops_instagram","platform_interest"]

REG_FEATURES = ["monthly_income_midpoint","lifestyle_spend","stress_score","nps_proxy",
    "conscientiousness_score","openness_score","city_tier","payment_method",
    "purchase_intent","books_per_month","personalization_importance",
    "subscription_interest","impulse_buying","barrier_price"]

ARM_COLS = PRODUCT_COLS + ["barrier_price","barrier_quality","barrier_trust",
    "barrier_delivery","barrier_privacy","barrier_platform",
    "shops_instagram","shops_amazon","shops_offline"]

PRESCRIPTIVE = {
    "Midnight Escapist":{"priority":"Primary","offer":"Free shipping + 10% off first order",
        "bundle":"Scented candle + dark journal + 1 fantasy book","channel":"Instagram Reels / BookTok",
        "timing":"Post after 9 PM","churn_risk":"Medium","ltv":"Rs6,000-12,000/yr"},
    "Productivity Achiever":{"priority":"High","offer":"Annual plan 25% saving",
        "bundle":"Self-help book + Annotation kit","channel":"LinkedIn / YouTube / Google Search",
        "timing":"Morning 6-9 AM","churn_risk":"Low","ltv":"Rs10,000-20,000/yr"},
    "Emotional Reader":{"priority":"Secondary","offer":"Buy 2 get 1 free",
        "bundle":"Literary fiction + Romance + Bookmarks","channel":"Instagram Stories / WhatsApp",
        "timing":"Evening 7-10 PM","churn_risk":"Low-medium","ltv":"Rs6,000-10,000/yr"},
    "Curious Explorer":{"priority":"Growth engine","offer":"Free DNA quiz + 15% first purchase",
        "bundle":"3-genre mystery box + Enamel pin","channel":"Instagram / Discord",
        "timing":"Any time","churn_risk":"High","ltv":"Rs4,000-8,000/yr"},
    "Non-Reader":{"priority":"Deprioritize","offer":"Free DNA quiz only",
        "bundle":"N/A","channel":"Organic only","timing":"N/A","churn_risk":"Very high","ltv":"<Rs1,000/yr"},
}


def load_data():
    df = pd.read_csv("book_dna_data.csv")
    df["psm_bargain"] = df["psm_bargain"].fillna(df["psm_bargain"].median())
    return df


def get_clean(df):
    if "data_quality_flag" in df.columns:
        c = df[df["data_quality_flag"] == "clean"].copy()
    else:
        c = df.copy()
    c["psm_bargain"] = c["psm_bargain"].fillna(c["psm_bargain"].median())
    return c


def run_kmeans(df, k=5):
    feats = [c for c in CLUSTER_FEATURES if c in df.columns]
    Xs = StandardScaler().fit_transform(df[feats].fillna(0))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    return km, Xs, labels, feats


def compute_elbow_sil(df, k_max=9):
    feats = [c for c in CLUSTER_FEATURES if c in df.columns]
    Xs = StandardScaler().fit_transform(df[feats].fillna(0))
    ks, inertias, sils = [], [], []
    for k in range(2, k_max+1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(Xs)
        ks.append(k); inertias.append(km.inertia_); sils.append(silhouette_score(Xs, lbl))
    return ks, inertias, sils


def compute_pca(df, labels):
    feats = [c for c in CLUSTER_FEATURES if c in df.columns]
    Xs = StandardScaler().fit_transform(df[feats].fillna(0))
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xs)
    out = pd.DataFrame(coords, columns=["PC1","PC2"])
    out["cluster"] = labels
    out["segment"] = df["dna_segment"].values if "dna_segment" in df.columns else labels.astype(str)
    return out, pca.explained_variance_ratio_


def cluster_segment_map(df, labels):
    tmp = df.copy(); tmp["_k"] = labels
    return {int(c): tmp[tmp["_k"]==c]["dna_segment"].value_counts().idxmax()
            for c in np.unique(labels) if "dna_segment" in tmp.columns}


def train_classifiers(df):
    feats = [c for c in CLF_FEATURES if c in df.columns]
    X = df[feats].fillna(0); y = df["will_buy"].astype(int)
    Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    sc = StandardScaler(); Xtr_s=sc.fit_transform(Xtr); Xte_s=sc.transform(Xte)
    out, rf_m, dt_m = {}, None, None
    for name, model, xtr, xte in [
        ("Random Forest",       RandomForestClassifier(n_estimators=100,random_state=42,class_weight="balanced"),Xtr,Xte),
        ("Logistic Regression", LogisticRegression(max_iter=500,random_state=42,class_weight="balanced"),Xtr_s,Xte_s),
        ("Decision Tree",       DecisionTreeClassifier(max_depth=5,random_state=42,class_weight="balanced"),Xtr,Xte),
    ]:
        model.fit(xtr,ytr); yp=model.predict(xte); yprob=model.predict_proba(xte)[:,1]
        fpr,tpr,_ = roc_curve(yte,yprob)
        out[name] = {"model":model,"y_test":yte,"y_pred":yp,"y_prob":yprob,
            "accuracy":round(accuracy_score(yte,yp),4),
            "precision":round(precision_score(yte,yp,zero_division=0),4),
            "recall":round(recall_score(yte,yp,zero_division=0),4),
            "f1":round(f1_score(yte,yp,zero_division=0),4),
            "roc_auc":round(roc_auc_score(yte,yprob),4),
            "fpr":fpr,"tpr":tpr,"cm":confusion_matrix(yte,yp)}
        if name=="Random Forest": rf_m=model
        if name=="Decision Tree": dt_m=model
    return out, feats, sc, rf_m, dt_m


def train_format_clf(df):
    feats = [c for c in CLF_FEATURES if c in df.columns]
    X = df[feats].fillna(0); y = df["format_class"].astype(int)
    Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    m = RandomForestClassifier(n_estimators=100,random_state=42); m.fit(Xtr,ytr); yp=m.predict(Xte)
    return m, feats, {"accuracy":round(accuracy_score(yte,yp),4),
        "precision":round(precision_score(yte,yp,average="weighted",zero_division=0),4),
        "recall":round(recall_score(yte,yp,average="weighted",zero_division=0),4),
        "f1":round(f1_score(yte,yp,average="weighted",zero_division=0),4),
        "cm":confusion_matrix(yte,yp),
        "classes":[FORMAT_CLASS.get(c,str(c)) for c in sorted(y.unique())]}


def train_regressors(df):
    feats = [c for c in REG_FEATURES if c in df.columns]
    X = df[feats].fillna(0); y = df["max_single_spend"].astype(float)
    Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.25, random_state=42)
    sc = StandardScaler(); Xtr_s=sc.fit_transform(Xtr); Xte_s=sc.transform(Xte)
    rf = RandomForestRegressor(n_estimators=100,random_state=42); rf.fit(Xtr,ytr)
    ri = Ridge(); ri.fit(Xtr_s,ytr)
    out = {}
    for name,m,xte in [("Random Forest",rf,Xte),("Ridge Regression",ri,Xte_s)]:
        yp=m.predict(xte)
        out[name]={"model":m,"y_test":yte.values,"y_pred":yp,
            "r2":round(r2_score(yte,yp),4),"rmse":round(float(np.sqrt(mean_squared_error(yte,yp))),2)}
    return out, feats, sc, rf, dict(zip(feats,ri.coef_))


def run_arm(df, min_sup=0.08, min_conf=0.40, min_lift=1.0):
    from mlxtend.frequent_patterns import apriori, association_rules
    cols = [c for c in ARM_COLS if c in df.columns]
    basket = df[cols].astype(bool)
    freq = apriori(basket, min_support=min_sup, use_colnames=True)
    if freq.empty: return pd.DataFrame()
    try:
        rules = association_rules(freq, metric="confidence", min_threshold=min_conf, num_itemsets=len(freq))
    except TypeError:
        rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    if rules.empty: return pd.DataFrame()
    rules = rules[rules["lift"] >= min_lift].copy()
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
    return rules[["antecedents","consequents","support","confidence","lift"]].round(4).sort_values("lift",ascending=False).reset_index(drop=True)


def psm_chart(df):
    need = ["psm_too_cheap","psm_bargain","psm_expensive_ok","psm_too_expensive"]
    sub = df.dropna(subset=need)
    if len(sub) < 20: return None, None, None, None
    prices = np.linspace(50, 3000, 300)
    tc   = np.array([np.mean(sub["psm_too_cheap"].values    >= p) for p in prices])*100
    barg = np.array([np.mean(sub["psm_bargain"].values      <= p) for p in prices])*100
    exok = np.array([np.mean(sub["psm_expensive_ok"].values >= p) for p in prices])*100
    toex = np.array([np.mean(sub["psm_too_expensive"].values>= p) for p in prices])*100
    fig = go.Figure()
    for name,y,col in [("Too cheap",tc,"#E24B4A"),("Bargain",barg,"#1D9E75"),
                        ("Expensive but OK",exok,"#EF9F27"),("Too expensive",toex,"#5C3D8F")]:
        fig.add_trace(go.Scatter(x=prices,y=y,name=name,mode="lines",line=dict(color=col,width=2.5)))
    pmc=int(prices[np.argmin(np.abs(tc-barg))])
    pme=int(prices[np.argmin(np.abs(exok-toex))])
    opp=int(prices[np.argmin(tc+toex)])
    for xv,lbl,c in [(pmc,"PMC","#1D9E75"),(opp,"OPP","#222"),(pme,"PME","#5C3D8F")]:
        fig.add_vline(x=xv,line_dash="dash",line_color=c,line_width=2,
                      annotation_text=f"{lbl}: Rs{xv}",annotation_position="top right")
    fig.update_layout(title="Van Westendorp Price Sensitivity",xaxis_title="Price (Rs)",
        yaxis_title="Cumulative %",height=400,legend=dict(orientation="h",y=1.08),
        plot_bgcolor="white",paper_bgcolor="white",margin=dict(t=80,b=20))
    fig.update_xaxes(showgrid=True,gridcolor="#eee"); fig.update_yaxes(showgrid=True,gridcolor="#eee",range=[0,105])
    return fig, pmc, opp, pme
