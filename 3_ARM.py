import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils import load_data, get_clean, run_arm, CLUSTER_COLORS, PRODUCT_COLS, PRODUCT_NAMES

st.set_page_config(page_title="ARM - Book DNA", layout="wide")
st.markdown("""<style>
[data-testid="stSidebar"]{background:#1a1612;}
[data-testid="stSidebar"] *{color:#faf8f4 !important;}
.rc{background:#faf8f4;border:1px solid #ddd8ce;border-radius:10px;padding:12px 15px;margin-bottom:6px;}
.ins{background:#f3f0ea;border-left:4px solid #c8922a;border-radius:0 8px 8px 0;
     padding:12px 16px;font-size:.87rem;color:#4a4540;line-height:1.7;margin:8px 0;}
</style>""", unsafe_allow_html=True)

if "df" not in st.session_state: st.session_state["df"] = load_data()
df=st.session_state.get("df_up",st.session_state["df"]); clean=get_clean(df)

st.title("🔗 Association Rule Mining")
st.markdown("---")

c1,c2,c3,c4=st.columns(4)
min_sup =c1.slider("Min Support",   0.04,0.30,0.08,0.01)
min_conf=c2.slider("Min Confidence",0.20,0.90,0.40,0.05)
min_lift=c3.slider("Min Lift",      1.0, 5.0, 1.2, 0.1)
seg_arm =c4.selectbox("Segment:",["All"]+sorted(clean["dna_segment"].unique().tolist()))

arm_data=clean if seg_arm=="All" else clean[clean["dna_segment"]==seg_arm]

with st.spinner("Mining rules..."):
    try:
        rules=run_arm(arm_data,min_sup,min_conf,min_lift)
    except Exception as e:
        st.error(f"ARM error: {e}"); rules=pd.DataFrame()

if rules.empty:
    st.warning("No rules found. Lower Support or Confidence."); st.stop()

st.success(f"Found **{len(rules)}** rules for: *{seg_arm}*")
st.markdown("---")

k1,k2,k3,k4=st.columns(4)
k1.metric("Rules",len(rules)); k2.metric("Avg confidence",f"{rules['confidence'].mean():.3f}")
k3.metric("Avg lift",f"{rules['lift'].mean():.3f}"); k4.metric("Max lift",f"{rules['lift'].max():.3f}")
st.markdown("---")

# 1. Scatter Support x Confidence
st.subheader("1. Support × Confidence  (bubble = Lift)")
fs=go.Figure(go.Scatter(x=rules["support"],y=rules["confidence"],mode="markers",
    marker=dict(size=rules["lift"]*12,color=rules["lift"],colorscale="Purples",
        showscale=True,colorbar=dict(title="Lift",thickness=14),
        opacity=0.72,line=dict(color="white",width=0.5)),
    text=rules["antecedents"]+" -> "+rules["consequents"],
    hovertemplate="<b>%{text}</b><br>Sup:%{x:.4f} Conf:%{y:.4f}<extra></extra>"))
fs.update_layout(height=420,xaxis=dict(title="Support",showgrid=True,gridcolor="#eee"),
    yaxis=dict(title="Confidence",showgrid=True,gridcolor="#eee"),
    plot_bgcolor="white",paper_bgcolor="white",margin=dict(t=10,b=10))
st.plotly_chart(fs,use_container_width=True)

# 2. Bar top 15 by Lift
st.markdown("---"); st.subheader("2. Top 15 rules by Lift")
t15=rules.head(15).copy()
t15["rule"]=t15["antecedents"]+"  →  "+t15["consequents"]
t15["lbl"]=(t15["lift"].round(2).astype(str)+" | conf="+t15["confidence"].round(2).astype(str))
t15=t15.sort_values("lift",ascending=True)
fb=go.Figure(go.Bar(y=t15["rule"],x=t15["lift"],orientation="h",
    marker=dict(color=t15["lift"],colorscale="Purples",showscale=True,colorbar=dict(title="Lift",thickness=14)),
    text=t15["lbl"],textposition="outside"))
fb.update_layout(height=max(360,len(t15)*30+60),
    xaxis=dict(title="Lift",showgrid=True,gridcolor="#eee"),
    yaxis_title=None,plot_bgcolor="white",paper_bgcolor="white",margin=dict(t=10,b=10,l=10,r=180))
st.plotly_chart(fb,use_container_width=True)

# 3. Confidence vs Lift
st.markdown("---"); st.subheader("3. Confidence vs Lift  (bubble = Support)")
fc=go.Figure(go.Scatter(x=rules["confidence"],y=rules["lift"],mode="markers",
    marker=dict(size=rules["support"]*120,color=rules["support"],colorscale="Teal",
        showscale=True,colorbar=dict(title="Support",thickness=14),opacity=0.68),
    text=rules["antecedents"]+" -> "+rules["consequents"],
    hovertemplate="<b>%{text}</b><br>Conf:%{x:.4f} Lift:%{y:.4f}<extra></extra>"))
fc.update_layout(height=380,xaxis=dict(title="Confidence",showgrid=True,gridcolor="#eee"),
    yaxis=dict(title="Lift",showgrid=True,gridcolor="#eee"),
    plot_bgcolor="white",paper_bgcolor="white",margin=dict(t=10,b=10))
st.plotly_chart(fc,use_container_width=True)

# 4. Table
st.markdown("---"); st.subheader("4. Full rules table")
srt=st.selectbox("Sort by:",["lift","confidence","support"])
disp=rules.sort_values(srt,ascending=False).reset_index(drop=True); disp.index+=1
st.dataframe(disp.style
    .background_gradient(subset=["lift"],cmap="Purples")
    .background_gradient(subset=["confidence"],cmap="Blues")
    .background_gradient(subset=["support"],cmap="Greens")
    .format({"support":"{:.4f}","confidence":"{:.4f}","lift":"{:.4f}"}),
    use_container_width=True,height=380)

# 5. Business actions
st.markdown("---"); st.subheader("5. Business actions")
for _,row in rules.head(5).iterrows():
    st.markdown(f'<div class="rc"><strong>Rule:</strong> <code>{row["antecedents"]}</code> → <code>{row["consequents"]}</code><br>'
        f'<strong>Lift: {row["lift"]:.2f}</strong> · Confidence: {row["confidence"]:.2f} · Support: {row["support"]:.3f}<br>'
        f'<strong>Action:</strong> Customers with <em>{row["antecedents"]}</em> are <strong>{row["lift"]:.1f}x</strong> '
        f'more likely to also want <em>{row["consequents"]}</em>. Bundle or upsell. Covers {row["support"]*100:.1f}% of respondents.</div>',
        unsafe_allow_html=True)

# 6. Co-interest heatmap
st.markdown("---"); st.subheader("6. Product co-interest heatmap  P(col | row)")
pp=[c for c in PRODUCT_COLS if c in arm_data.columns]
pn=[PRODUCT_NAMES[PRODUCT_COLS.index(c)] for c in pp]
mat=np.zeros((len(pp),len(pp)))
for i,ci in enumerate(pp):
    for j,cj in enumerate(pp):
        if i!=j:
            m=arm_data[ci]==1
            mat[i][j]=arm_data.loc[m,cj].mean() if m.sum()>0 else 0
fhm=go.Figure(go.Heatmap(z=mat,x=pn,y=pn,colorscale="Purples",zmin=0,zmax=1,
    text=[[f"{v:.2f}" for v in row] for row in mat],texttemplate="%{text}",
    colorbar=dict(title="P(col|row)",thickness=14)))
fhm.update_layout(height=400,margin=dict(t=10,b=10),
    xaxis_title=None,yaxis_title=None,plot_bgcolor="white",paper_bgcolor="white")
st.plotly_chart(fhm,use_container_width=True)
