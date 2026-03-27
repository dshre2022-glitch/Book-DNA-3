import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils import (load_data, get_clean, CLUSTER_COLORS, PALETTE,
                   CITY_LABELS, PRODUCT_COLS, PRODUCT_NAMES, PAY_LABELS)

st.set_page_config(page_title="Book DNA Analytics", page_icon="📚", layout="wide")

st.markdown("""<style>
[data-testid="stSidebar"]{background:#1a1612;}
[data-testid="stSidebar"] *{color:#faf8f4 !important;}
.kpi{background:#faf8f4;border:1px solid #e0dcd4;border-radius:10px;
     padding:16px;text-align:center;margin-bottom:4px;}
.kv{font-size:1.8rem;font-weight:700;color:#1a1612;}
.kl{font-size:.7rem;color:#7a736b;text-transform:uppercase;letter-spacing:.06em;margin-top:4px;}
.ins{background:#f3f0ea;border-left:4px solid #c8922a;border-radius:0 8px 8px 0;
     padding:12px 16px;font-size:.87rem;color:#4a4540;line-height:1.7;margin:8px 0;}
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 📚 Book DNA")
    st.markdown("*Founder Analytics Dashboard*")
    st.divider()
    st.page_link("app.py",                        label="🏠  Home")
    st.page_link("pages/1_Descriptive.py",         label="📊  Descriptive")
    st.page_link("pages/2_Clustering.py",          label="🔵  Clustering")
    st.page_link("pages/3_ARM.py",                 label="🔗  Association Rules")
    st.page_link("pages/4_Predictive.py",          label="🔮  Predictive")
    st.page_link("pages/5_Prescriptive_Upload.py", label="🎯  Prescriptive & Upload")
    st.divider()
    up = st.file_uploader("Upload survey CSV", type=["csv"])
    if up: st.session_state["upload"] = up
    st.caption("Book DNA v3.0")

# ── DATA ─────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state["df"] = load_data()
if "upload" in st.session_state and "df_up" not in st.session_state:
    import pandas as _pd
    st.session_state["df_up"] = _pd.read_csv(st.session_state["upload"])

df    = st.session_state.get("df_up", st.session_state["df"])
clean = get_clean(df)

# ── HEADER ────────────────────────────────────────────────────────────
st.markdown("# 📚 Book DNA — Founder Analytics Dashboard")
st.markdown("**Descriptive · Diagnostic · Predictive · Prescriptive**")
st.markdown("---")

# ── KPIs ─────────────────────────────────────────────────────────────
k1,k2,k3,k4,k5,k6 = st.columns(6)
buy_pct = clean["will_buy"].mean()*100
for col,val,lbl in [
    (k1, f"{len(df):,}",               "Respondents"),
    (k2, f"{buy_pct:.1f}%",            "Will buy"),
    (k3, f"Rs{clean['max_single_spend'].mean():,.0f}", "Avg max spend"),
    (k4, f"{clean['nps_proxy'].mean():.1f}/10",         "NPS proxy"),
    (k5, clean["dna_segment"].value_counts().idxmax().split()[0], "Top segment"),
    (k6, f"Rs{clean['psm_bargain'].median():,.0f}",    "PSM optimal price"),
]:
    with col:
        st.markdown(f'<div class="kpi"><div class="kv">{val}</div><div class="kl">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ── SEGMENT BAR ───────────────────────────────────────────────────────
l,r = st.columns([1.3,1])
with l:
    st.subheader("Segment distribution")
    sd = clean["dna_segment"].value_counts().reset_index()
    sd.columns=["Segment","Count"]; sd["Pct"]=(sd["Count"]/len(clean)*100).round(1)
    fig=go.Figure(go.Bar(x=sd["Segment"],y=sd["Count"],
        marker_color=[CLUSTER_COLORS.get(s,"#888") for s in sd["Segment"]],
        text=sd["Pct"].astype(str)+"%",textposition="outside"))
    fig.update_layout(height=300,xaxis_title=None,
        yaxis=dict(showgrid=True,gridcolor="#eee"),
        plot_bgcolor="white",paper_bgcolor="white",margin=dict(t=10,b=10))
    st.plotly_chart(fig,use_container_width=True)
with r:
    st.subheader("Quick stats")
    for _,row in sd.iterrows():
        seg=row["Segment"]; sub=clean[clean["dna_segment"]==seg]
        color=CLUSTER_COLORS.get(seg,"#888")
        st.markdown(f"""<div style="border-left:4px solid {color};background:#faf8f4;
            border-radius:0 8px 8px 0;padding:9px 14px;margin-bottom:5px;">
            <strong style="color:{color}">{seg}</strong> · n={row['Count']} ({row['Pct']}%)
            · Buy: <strong>{sub['will_buy'].mean()*100:.0f}%</strong>
            · Spend: <strong>Rs{sub['max_single_spend'].mean():,.0f}</strong>
            </div>""",unsafe_allow_html=True)

st.markdown("---")

# ── BUY INTENT ────────────────────────────────────────────────────────
st.subheader("Purchase intent by segment")
bi=clean.groupby("dna_segment")["will_buy"].mean().reset_index()
bi["Buy%"]=(bi["will_buy"]*100).round(1); bi=bi.sort_values("Buy%",ascending=False)
fig2=go.Figure(go.Bar(x=bi["dna_segment"],y=bi["Buy%"],
    marker_color=[CLUSTER_COLORS.get(s,"#888") for s in bi["dna_segment"]],
    text=bi["Buy%"].astype(str)+"%",textposition="outside"))
fig2.update_layout(height=280,xaxis_title=None,
    yaxis=dict(title="% Will Buy",range=[0,100],showgrid=True,gridcolor="#eee"),
    plot_bgcolor="white",paper_bgcolor="white",margin=dict(t=10,b=10))
st.plotly_chart(fig2,use_container_width=True)
st.markdown("---")

# ── INSIGHTS ─────────────────────────────────────────────────────────
st.subheader("Key founder insights")
i1,i2,i3=st.columns(3)
with i1:
    prod=sorted([(n,clean[c].mean()*100) for c,n in zip(PRODUCT_COLS,PRODUCT_NAMES) if c in clean.columns],key=lambda x:-x[1])
    lines="<br>".join([f"• {n}: {v:.0f}%" for n,v in prod[:5]])
    st.markdown(f'<div class="ins"><strong>Top product demand</strong><br>{lines}</div>',unsafe_allow_html=True)
with i2:
    cb=clean.groupby("city_tier")["will_buy"].mean()*100
    st.markdown(f'<div class="ins"><strong>Geographic priority</strong><br>'
        f'Best city: <strong>{CITY_LABELS.get(cb.idxmax(),str(cb.idxmax()))} ({cb.max():.0f}%)</strong><br>'
        f'Metro+Tier1 = {(clean["city_tier"]<=2).mean()*100:.0f}% of high-intent</div>',unsafe_allow_html=True)
with i3:
    upi=(clean["payment_method"]==1).mean()*100
    cc=(clean["payment_method"]==3).mean()*100
    psm=int(clean["psm_bargain"].median())
    st.markdown(f'<div class="ins"><strong>Payment & pricing</strong><br>'
        f'UPI: <strong>{upi:.0f}%</strong> · Credit card: <strong>{cc:.0f}%</strong><br>'
        f'PSM optimal: <strong>Rs{psm}</strong></div>',unsafe_allow_html=True)
