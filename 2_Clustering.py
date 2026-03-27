import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils import (load_data, get_clean, run_kmeans, compute_elbow_sil,
                   compute_pca, cluster_segment_map,
                   CLUSTER_COLORS, PALETTE, PRODUCT_COLS, PRODUCT_NAMES)

st.set_page_config(page_title="Clustering - Book DNA", layout="wide")
st.markdown("""<style>
[data-testid="stSidebar"]{background:#1a1612;}
[data-testid="stSidebar"] *{color:#faf8f4 !important;}
.ins{background:#f3f0ea;border-left:4px solid #c8922a;border-radius:0 8px 8px 0;
     padding:12px 16px;font-size:.87rem;color:#4a4540;line-height:1.7;margin:8px 0;}
</style>""", unsafe_allow_html=True)

if "df" not in st.session_state: st.session_state["df"] = load_data()
df=st.session_state.get("df_up",st.session_state["df"]); clean=get_clean(df)

st.title("🔵 Clustering & Customer Personas")
st.markdown("---")

# 1. Elbow + Silhouette
st.subheader("1. Finding optimal k")
k_max=st.slider("Max k to explore:",4,12,9)
if "elbow_data" not in st.session_state or st.session_state.get("elbow_kmax")!=k_max:
    with st.spinner("Computing..."):
        ks,inertias,sils = compute_elbow_sil(clean,k_max=k_max)
    st.session_state["elbow_data"]=(ks,inertias,sils); st.session_state["elbow_kmax"]=k_max

ks,inertias,sils=st.session_state["elbow_data"]
best_k=ks[int(np.argmax(sils))]
ce,cs=st.columns(2)
with ce:
    fe=go.Figure()
    fe.add_trace(go.Scatter(x=ks,y=inertias,mode="lines+markers",
        line=dict(color="#5C3D8F",width=2.5),marker=dict(size=8)))
    fe.add_vline(x=5,line_dash="dash",line_color="#C8922A",line_width=2,
        annotation_text="k=5",annotation_position="top right")
    fe.update_layout(title="Elbow Chart",height=340,
        xaxis=dict(title="k",dtick=1,showgrid=True,gridcolor="#eee"),
        yaxis=dict(title="Inertia",showgrid=True,gridcolor="#eee"),
        plot_bgcolor="white",paper_bgcolor="white",margin=dict(t=45,b=15))
    st.plotly_chart(fe,use_container_width=True)
with cs:
    bc=["#C8922A" if k==5 else "#5C3D8F" for k in ks]
    fs=go.Figure(go.Bar(x=ks,y=sils,marker_color=bc,text=[f"{s:.3f}" for s in sils],textposition="outside"))
    fs.add_hline(y=max(sils),line_dash="dot",line_color="#1A6B5A",
        annotation_text=f"Best={max(sils):.3f} k={best_k}")
    fs.update_layout(title="Silhouette Score",height=340,
        xaxis=dict(title="k",dtick=1,showgrid=True,gridcolor="#eee"),
        yaxis=dict(title="Silhouette",range=[0,max(sils)*1.25],showgrid=True,gridcolor="#eee"),
        plot_bgcolor="white",paper_bgcolor="white",margin=dict(t=45,b=15))
    st.plotly_chart(fs,use_container_width=True)
st.markdown(f'<div class="ins">Elbow bends at k=5. Silhouette best at k={best_k} (score={max(sils):.3f}). 5 clusters = 5 Book DNA personas.</div>',unsafe_allow_html=True)

# 2. K-Means k=5
st.markdown("---"); st.subheader("2. K-Means (k=5) results")
if "km5" not in st.session_state:
    with st.spinner("Training K-Means..."):
        km,Xs,labels,feats=run_kmeans(clean,k=5)
        cmap=cluster_segment_map(clean,labels)
        pca_df,ev=compute_pca(clean,labels)
        pca_df["persona"]=[cmap.get(int(l),f"C{l}") for l in pca_df["cluster"]]
    st.session_state["km5"]=(km,labels,cmap,pca_df,ev)

km,labels,cmap,pca_df,ev=st.session_state["km5"]

cp,cs2=st.columns([2.2,1])
with cp:
    fpca=px.scatter(pca_df,x="PC1",y="PC2",color="persona",
        color_discrete_map=CLUSTER_COLORS,opacity=0.6,height=380,
        title=f"PCA 2-D  (PC1={ev[0]:.1%} · PC2={ev[1]:.1%})")
    fpca.update_traces(marker=dict(size=5))
    fpca.update_layout(plot_bgcolor="white",paper_bgcolor="white",margin=dict(t=50,b=10),
        xaxis=dict(showgrid=True,gridcolor="#eee"),yaxis=dict(showgrid=True,gridcolor="#eee"))
    st.plotly_chart(fpca,use_container_width=True)
with cs2:
    st.markdown("**Cluster sizes**")
    for c in sorted(cmap):
        seg=cmap[c]; cnt=int((labels==c).sum()); pct=cnt/len(labels)*100
        color=CLUSTER_COLORS.get(seg,"#888")
        st.markdown(f'<div style="border-left:4px solid {color};background:#faf8f4;border-radius:0 8px 8px 0;padding:9px 13px;margin-bottom:5px;"><strong style="color:{color};font-size:.85rem">{seg}</strong><br><span style="color:#7a736b;font-size:.8rem">n={cnt} ({pct:.1f}%)</span></div>',unsafe_allow_html=True)

# 3. Persona tabs
st.markdown("---"); st.subheader("3. Persona deep-dive")
tabs=st.tabs(list(CLUSTER_COLORS.keys()))
for tab,seg in zip(tabs,CLUSTER_COLORS.keys()):
    with tab:
        sub=clean[clean["dna_segment"]==seg]
        if len(sub)==0: st.info("No data."); continue
        k1,k2,k3,k4,k5c=st.columns(5)
        k1.metric("Count",len(sub)); k2.metric("Will buy",f"{sub['will_buy'].mean()*100:.0f}%")
        k3.metric("Avg spend",f"Rs{sub['max_single_spend'].mean():,.0f}")
        k4.metric("NPS",f"{sub['nps_proxy'].mean():.1f}/10")
        k5c.metric("Stress",f"{sub['stress_score'].mean():.1f}/16")
        la,ra=st.columns(2)
        with la:
            traits=["openness_score","conscientiousness_score","extraversion_score","agreeableness_score","neuroticism_score"]
            tlbls=["Openness","Conscient.","Extraversion","Agreeableness","Neuroticism"]
            means=[sub[t].mean() if t in sub.columns else 3 for t in traits]
            mc=means+[means[0]]; lc=tlbls+[tlbls[0]]
            color=CLUSTER_COLORS.get(seg,"#5C3D8F")
            fr=go.Figure(go.Scatterpolar(r=mc,theta=lc,fill="toself",
                line_color=color,fillcolor=color+"30"))
            fr.update_layout(polar=dict(radialaxis=dict(visible=True,range=[1,5])),
                title="OCEAN Profile",height=300,margin=dict(t=50,b=10),
                showlegend=False,paper_bgcolor="white")
            st.plotly_chart(fr,use_container_width=True)
        with ra:
            ppd=[(PRODUCT_NAMES[PRODUCT_COLS.index(c)],sub[c].mean()*100) for c in PRODUCT_COLS if c in sub.columns]
            ppdf=pd.DataFrame(ppd,columns=["Product","Pct"]).sort_values("Pct",ascending=True)
            fp=go.Figure(go.Bar(y=ppdf["Product"],x=ppdf["Pct"],orientation="h",
                marker_color=CLUSTER_COLORS.get(seg,"#888"),
                text=[f"{v:.0f}%" for v in ppdf["Pct"]],textposition="outside"))
            fp.update_layout(title="Product interest",height=300,margin=dict(t=50,b=10,r=55),
                xaxis=dict(range=[0,102],showgrid=True,gridcolor="#eee"),
                yaxis_title=None,plot_bgcolor="white",paper_bgcolor="white")
            st.plotly_chart(fp,use_container_width=True)
        s1,s2,s3,s4=st.columns(4)
        s1.metric("Books/month",f"{sub['books_per_month'].mean():.1f}")
        s2.metric("PSM bargain",f"Rs{sub['psm_bargain'].median():,.0f}")
        s3.metric("Churn risk",f"{sub['switching_tendency'].mean():.1f}/4")
        s4.metric("Sharing",f"{sub['social_sharing_level'].mean():.1f}/4")
