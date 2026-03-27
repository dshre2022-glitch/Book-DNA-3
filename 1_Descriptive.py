import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils import (load_data, get_clean, psm_chart, AGE_LABELS, GENDER_LABELS,
                   CITY_LABELS, OCC_LABELS, CLUSTER_COLORS, PALETTE,
                   PRODUCT_COLS, PRODUCT_NAMES, RH_LABELS)

st.set_page_config(page_title="Descriptive - Book DNA", layout="wide")
st.markdown("""<style>
[data-testid="stSidebar"]{background:#1a1612;}
[data-testid="stSidebar"] *{color:#faf8f4 !important;}
.ins{background:#f3f0ea;border-left:4px solid #c8922a;border-radius:0 8px 8px 0;
     padding:12px 16px;font-size:.87rem;color:#4a4540;line-height:1.7;margin:8px 0;}
</style>""", unsafe_allow_html=True)

if "df" not in st.session_state: st.session_state["df"] = load_data()
df=st.session_state.get("df_up",st.session_state["df"]); clean=get_clean(df)

st.title("📊 Descriptive Analysis")
st.caption(f"{len(clean):,} clean respondents · {len(df):,} total")
st.markdown("---")

with st.expander("Filters", expanded=False):
    c1,c2,c3=st.columns(3)
    seg_sel =c1.multiselect("Segment",sorted(clean["dna_segment"].unique()),default=sorted(clean["dna_segment"].unique()))
    city_sel=c2.multiselect("City tier",sorted(clean["city_tier"].unique()),default=sorted(clean["city_tier"].unique()),format_func=lambda x:CITY_LABELS.get(x,x))
    age_sel =c3.multiselect("Age group",sorted(clean["age_group"].unique()),default=sorted(clean["age_group"].unique()),format_func=lambda x:AGE_LABELS.get(x,x))
filt=clean[clean["dna_segment"].isin(seg_sel)&clean["city_tier"].isin(city_sel)&clean["age_group"].isin(age_sel)].copy()
st.caption(f"Filtered: {len(filt):,}")

def barchart(col,lmap,title,color):
    c=filt[col].value_counts().reset_index(); c.columns=["k","n"]
    c["label"]=c["k"].map(lmap).fillna(c["k"].astype(str)); c=c.sort_values("k")
    f=go.Figure(go.Bar(x=c["label"],y=c["n"],marker_color=color,text=c["n"],textposition="outside"))
    f.update_layout(title=title,height=280,margin=dict(t=40,b=8),xaxis=dict(title=None,tickangle=-30),
        yaxis=dict(title=None,showgrid=True,gridcolor="#eee"),plot_bgcolor="white",paper_bgcolor="white")
    return f

st.subheader("1. Demographics")
d1,d2,d3,d4=st.columns(4)
d1.plotly_chart(barchart("age_group",AGE_LABELS,"Age groups","#5C3D8F"),use_container_width=True)
d2.plotly_chart(barchart("gender",GENDER_LABELS,"Gender","#C8922A"),use_container_width=True)
d3.plotly_chart(barchart("city_tier",CITY_LABELS,"City tier","#1A6B5A"),use_container_width=True)
d4.plotly_chart(barchart("occupation",OCC_LABELS,"Occupation","#D45379"),use_container_width=True)

inc_map={5000:"<Rs10K",15000:"Rs10-20K",30000:"Rs20-40K",60000:"Rs40-80K",100000:">Rs80K"}
ic=filt["monthly_income_midpoint"].value_counts().sort_index().reset_index(); ic.columns=["k","n"]
ic["label"]=ic["k"].map(inc_map).fillna(ic["k"].astype(str))
fig_inc=go.Figure(go.Bar(x=ic["label"],y=ic["n"],marker_color="#378ADD",text=ic["n"],textposition="outside"))
fig_inc.update_layout(title="Income distribution",height=260,margin=dict(t=40,b=8),xaxis_title=None,
    yaxis=dict(showgrid=True,gridcolor="#eee"),plot_bgcolor="white",paper_bgcolor="white")
st.plotly_chart(fig_inc,use_container_width=True)

st.markdown("---")
st.subheader("2. Van Westendorp Price Sensitivity")
pa,pb=st.columns([3,1])
with pa:
    seg_opt=["All"]+sorted(clean["dna_segment"].unique().tolist())
    psm_seg=st.selectbox("Segment:",seg_opt)
psm_data=filt if psm_seg=="All" else filt[filt["dna_segment"]==psm_seg]
with pb: st.metric("Respondents",len(psm_data))
fig_psm,pmc,opp,pme=psm_chart(psm_data)
if fig_psm:
    st.plotly_chart(fig_psm,use_container_width=True)
    m1,m2,m3=st.columns(3)
    m1.metric("Floor (PMC)",f"Rs{pmc}"); m2.metric("Optimal (OPP)",f"Rs{opp}"); m3.metric("Ceiling (PME)",f"Rs{pme}")
    st.markdown(f'<div class="ins">Price MVP box at <strong>Rs{opp}</strong>. Acceptable range Rs{pmc}–Rs{pme}.</div>',unsafe_allow_html=True)
else:
    st.info("Not enough data. Try a broader filter.")

st.markdown("---")
st.subheader("3. Product interest heatmap by segment")
pp=[c for c in PRODUCT_COLS if c in filt.columns]; pn=[PRODUCT_NAMES[PRODUCT_COLS.index(c)] for c in pp]
segs=sorted(filt["dna_segment"].unique())
z=[[filt[filt["dna_segment"]==s][c].mean()*100 for c in pp] for s in segs]
fig_h=go.Figure(go.Heatmap(z=z,x=pn,y=segs,colorscale="Purples",
    text=[[f"{v:.0f}%" for v in row] for row in z],texttemplate="%{text}",
    colorbar=dict(title="% interest")))
fig_h.update_layout(height=320,margin=dict(t=10,b=10),xaxis_title=None,yaxis_title=None,
    plot_bgcolor="white",paper_bgcolor="white")
st.plotly_chart(fig_h,use_container_width=True)

st.markdown("---")
st.subheader("4. Genre popularity")
gcols=[c for c in filt.columns if c.startswith("genre_")]
gdf=pd.DataFrame({"Genre":[c.replace("genre_","").title() for c in gcols],
    "Pct":[filt[c].mean()*100 for c in gcols]}).sort_values("Pct",ascending=True)
fig_g=go.Figure(go.Bar(y=gdf["Genre"],x=gdf["Pct"],orientation="h",marker_color="#C8922A",
    text=[f"{v:.1f}%" for v in gdf["Pct"]],textposition="outside"))
fig_g.update_layout(height=340,margin=dict(t=10,b=10,r=60),
    xaxis=dict(title="% respondents",range=[0,90],showgrid=True,gridcolor="#eee"),
    yaxis_title=None,plot_bgcolor="white",paper_bgcolor="white")
st.plotly_chart(fig_g,use_container_width=True)

st.markdown("---")
st.subheader("5. Reading habits")
h1,h2,h3=st.columns(3)
with h1:
    bpm=filt["books_per_month"].value_counts().sort_index().reset_index(); bpm.columns=["b","n"]
    f=go.Figure(go.Bar(x=bpm["b"].astype(str),y=bpm["n"],marker_color="#5DCAA5",text=bpm["n"],textposition="outside"))
    f.update_layout(title="Books per month",height=270,margin=dict(t=40,b=8),xaxis_title=None,
        yaxis=dict(showgrid=True,gridcolor="#eee"),plot_bgcolor="white",paper_bgcolor="white")
    st.plotly_chart(f,use_container_width=True)
with h2:
    tm={"reads_morning":"Morning","reads_commute":"Commute","reads_afternoon":"Afternoon",
        "reads_evening":"Evening","reads_latenight":"Late night","reads_weekend":"Weekend"}
    tl=[v for k,v in tm.items() if k in filt.columns]; tp=[filt[k].mean()*100 for k in tm if k in filt.columns]
    f=go.Figure(go.Bar(x=tl,y=tp,marker_color="#378ADD",text=[f"{v:.0f}%" for v in tp],textposition="outside"))
    f.update_layout(title="Reading time",height=270,margin=dict(t=40,b=8),xaxis_title=None,
        yaxis=dict(range=[0,88],showgrid=True,gridcolor="#eee"),plot_bgcolor="white",paper_bgcolor="white")
    st.plotly_chart(f,use_container_width=True)
with h3:
    rhs=filt["reading_habit_status"].value_counts().reset_index(); rhs.columns=["k","n"]
    rhs["label"]=rhs["k"].map(RH_LABELS).fillna(rhs["k"].astype(str))
    f=go.Figure(go.Pie(labels=rhs["label"],values=rhs["n"],hole=0.42,marker_colors=PALETTE[:len(rhs)]))
    f.update_layout(title="Reading habit",height=270,margin=dict(t=40,l=10,r=10,b=10))
    st.plotly_chart(f,use_container_width=True)
st.markdown('<div class="ins">Late-night is the peak reading window. Schedule posts 9–11 PM IST.</div>',unsafe_allow_html=True)
