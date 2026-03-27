import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils import (load_data, get_clean, train_classifiers, train_regressors,
                   run_kmeans, cluster_segment_map,
                   CLUSTER_COLORS, PRESCRIPTIVE, PRODUCT_COLS, PRODUCT_NAMES,
                   CLF_FEATURES, REG_FEATURES, PAY_LABELS, CITY_LABELS, DISC_LABELS)

st.set_page_config(page_title="Prescriptive - Book DNA", layout="wide")
st.markdown("""<style>
[data-testid="stSidebar"]{background:#1a1612;}
[data-testid="stSidebar"] *{color:#faf8f4 !important;}
.pc{background:#faf8f4;border:1px solid #ddd8ce;border-radius:12px;padding:15px 17px;margin-bottom:8px;}
.ins{background:#f3f0ea;border-left:4px solid #c8922a;border-radius:0 8px 8px 0;
     padding:12px 16px;font-size:.87rem;color:#4a4540;line-height:1.7;margin:8px 0;}
</style>""", unsafe_allow_html=True)

if "df" not in st.session_state: st.session_state["df"] = load_data()
df=st.session_state.get("df_up",st.session_state["df"]); clean=get_clean(df)

st.title("🎯 Prescriptive Analysis & Upload")
st.markdown("---")

t1,t2,t3=st.tabs(["🎯 Prescriptive Strategy","📍 Focus Customer","📤 Upload & Predict"])

# ── TAB 1 ─────────────────────────────────────────────────────────────
with t1:
    st.subheader("Segment-by-segment recommendations")
    rows_p=[]
    for seg in CLUSTER_COLORS:
        sub=clean[clean["dna_segment"]==seg]; p=PRESCRIPTIVE[seg]
        rows_p.append({"Segment":seg,"Size":len(sub),"Buy%":f"{sub['will_buy'].mean()*100:.0f}%",
            "Avg spend":f"Rs{sub['max_single_spend'].mean():,.0f}",
            "NPS":f"{sub['nps_proxy'].mean():.1f}","Priority":p["priority"],"LTV":p["ltv"]})
    st.dataframe(pd.DataFrame(rows_p).set_index("Segment"),use_container_width=True)
    st.markdown("---")

    for seg,color in CLUSTER_COLORS.items():
        p=PRESCRIPTIVE[seg]
        with st.expander(f"{p['priority']}  {seg}",expanded=(seg=="Midnight Escapist")):
            cx1,cx2,cx3=st.columns(3)
            with cx1: st.markdown(f'<div class="pc" style="border-top:3px solid {color}"><strong style="color:{color}">Offer</strong><br>{p["offer"]}<br><br><strong style="color:{color}">LTV</strong><br>{p["ltv"]}</div>',unsafe_allow_html=True)
            with cx2: st.markdown(f'<div class="pc" style="border-top:3px solid {color}"><strong style="color:{color}">Bundle</strong><br>{p["bundle"]}<br><br><strong style="color:{color}">Churn risk</strong><br>{p["churn_risk"]}</div>',unsafe_allow_html=True)
            with cx3: st.markdown(f'<div class="pc" style="border-top:3px solid {color}"><strong style="color:{color}">Channel</strong><br>{p["channel"]}<br><br><strong style="color:{color}">Timing</strong><br>{p["timing"]}</div>',unsafe_allow_html=True)

    st.markdown("---"); st.markdown("#### Discount preference by segment")
    disc_data={}
    for seg in sorted(clean["dna_segment"].unique()):
        sub=clean[clean["dna_segment"]==seg]; vc=sub["discount_preference"].value_counts(normalize=True)
        disc_data[seg]={DISC_LABELS.get(k,str(k)):round(v*100,1) for k,v in vc.items()}
    disc_df=pd.DataFrame(disc_data).fillna(0).T
    fig_disc=px.bar(disc_df.reset_index(),x="index",y=disc_df.columns.tolist(),barmode="group",height=340,
        color_discrete_sequence=px.colors.qualitative.Pastel,
        labels={"index":"Segment","value":"% preference","variable":"Offer type"})
    fig_disc.update_layout(xaxis_title=None,yaxis=dict(showgrid=True,gridcolor="#eee"),
        plot_bgcolor="white",paper_bgcolor="white",margin=dict(t=10,b=10))
    st.plotly_chart(fig_disc,use_container_width=True)

    st.markdown("---"); st.markdown("#### High churn-risk customers")
    churn=clean[(clean["switching_tendency"]>=3)&(clean["nps_proxy"]<=5)&(clean["will_buy"]==1)]
    st.metric("High churn-risk buyers",len(churn),f"{len(churn)/max(len(clean),1)*100:.1f}% of dataset")
    if len(churn)>0:
        cs=churn["dna_segment"].value_counts().reset_index(); cs.columns=["Segment","Count"]
        fch=go.Figure(go.Bar(x=cs["Segment"],y=cs["Count"],
            marker_color=[CLUSTER_COLORS.get(s,"#888") for s in cs["Segment"]],
            text=cs["Count"],textposition="outside"))
        fch.update_layout(height=280,xaxis_title=None,yaxis=dict(showgrid=True,gridcolor="#eee"),
            plot_bgcolor="white",paper_bgcolor="white",margin=dict(t=10,b=10))
        st.plotly_chart(fch,use_container_width=True)
        st.markdown('<div class="ins">Offer locking incentive before subscription: annual plan at 30% saving, loyalty bonus, or money-back guarantee.</div>',unsafe_allow_html=True)

# ── TAB 2 ─────────────────────────────────────────────────────────────
with t2:
    st.subheader("The Focus Customer — highest-value acquisition target")
    focus=clean[(clean["purchase_intent"]<=2)&(clean["max_single_spend"]>=750)&
                (clean["switching_tendency"]<=2)&(clean["social_sharing_level"]<=2)&
                (clean["nps_proxy"]>=7)]
    st.metric("Focus customers",len(focus),f"{len(focus)/max(len(clean),1)*100:.1f}% of dataset")
    if len(focus)>0:
        fa,fb=st.columns(2)
        with fa:
            top_seg_f=focus["dna_segment"].value_counts().idxmax()
            st.markdown(f"""| Attribute | Profile |
|---|---|
| **Top segment** | {top_seg_f} |
| **Avg NPS** | {focus['nps_proxy'].mean():.1f}/10 |
| **Avg stress** | {focus['stress_score'].mean():.1f}/16 |
| **Avg max spend** | Rs{focus['max_single_spend'].mean():,.0f} |
| **City** | {CITY_LABELS.get(int(focus['city_tier'].mode()[0]),'Metro')} |""")
        with fb:
            fc_s=focus["dna_segment"].value_counts().reset_index(); fc_s.columns=["Segment","Count"]
            ffc=go.Figure(go.Bar(x=fc_s["Segment"],y=fc_s["Count"],
                marker_color=[CLUSTER_COLORS.get(s,"#888") for s in fc_s["Segment"]],
                text=fc_s["Count"],textposition="outside"))
            ffc.update_layout(height=280,xaxis_title=None,yaxis=dict(showgrid=True,gridcolor="#eee"),
                plot_bgcolor="white",paper_bgcolor="white",margin=dict(t=10,b=10))
            st.plotly_chart(ffc,use_container_width=True)

        pp_fc=sorted([(PRODUCT_NAMES[PRODUCT_COLS.index(c)],focus[c].mean()*100) for c in PRODUCT_COLS if c in focus.columns],key=lambda x:-x[1])
        fpf=go.Figure(go.Bar(x=[x[0] for x in pp_fc],y=[x[1] for x in pp_fc],marker_color="#5C3D8F",
            text=[f"{x[1]:.0f}%" for x in pp_fc],textposition="outside"))
        fpf.update_layout(height=270,xaxis_title=None,yaxis=dict(title="% interested",range=[0,100],showgrid=True,gridcolor="#eee"),
            plot_bgcolor="white",paper_bgcolor="white",margin=dict(t=10,b=10))
        st.plotly_chart(fpf,use_container_width=True)
        st.markdown('<div class="ins">Put 60-70% of Year 1 paid marketing budget here. Lead with candle + journal at Rs599. Upsell to subscription in 30 days.</div>',unsafe_allow_html=True)

# ── TAB 3 ─────────────────────────────────────────────────────────────
with t3:
    st.subheader("Upload new survey data — predict segment, buy intent & spend")
    sample_cols=(["respondent_id"] if "respondent_id" in clean.columns else [])+\
        [c for c in CLF_FEATURES if c in clean.columns]+\
        [c for c in REG_FEATURES if c in clean.columns and c not in CLF_FEATURES]
    sample_cols=list(dict.fromkeys(sample_cols))
    st.download_button("Download template CSV",clean[sample_cols].head(3).to_csv(index=False),
        file_name="book_dna_template.csv",mime="text/csv")
    st.markdown("---")

    new_file=st.file_uploader("Upload new survey CSV",type=["csv"],key="new_csv")
    if new_file is not None:
        new_df=pd.read_csv(new_file)
        st.success(f"Loaded {len(new_df):,} rows · {new_df.shape[1]} cols")
        st.dataframe(new_df.head(3),use_container_width=True)

        with st.spinner("Predicting..."):
            if "clf" not in st.session_state:
                st.session_state["clf"]=train_classifiers(clean)
            if "reg" not in st.session_state:
                st.session_state["reg"]=train_regressors(clean)
            res_u,clf_fu,_,rf_u,_=st.session_state["clf"]
            reg_u,reg_fu,_,rfr_u,_=st.session_state["reg"]
            km_u,_,lbl_u,km_fu=run_kmeans(clean,k=5)
            cmap_u=cluster_segment_map(clean,lbl_u)
            out=new_df.copy()

            # Cluster
            cc=[c for c in km_fu if c in out.columns]
            if len(cc)>=5:
                from sklearn.preprocessing import StandardScaler
                Xc=out[cc].fillna(0)
                for mc in km_fu:
                    if mc not in Xc.columns: Xc[mc]=0
                Xc=Xc[km_fu]
                sc_tmp=StandardScaler().fit(clean[[c for c in km_fu if c in clean.columns]].fillna(0).reindex(columns=km_fu,fill_value=0))
                Xc_s=sc_tmp.transform(Xc)
                out["predicted_dna_segment"]=[cmap_u.get(int(l),f"C{l}") for l in km_u.predict(Xc_s)]
            else:
                out["predicted_dna_segment"]="Insufficient features"

            # Buy intent
            cfc=[c for c in clf_fu if c in out.columns]
            if len(cfc)>=5:
                Xcl=out[cfc].fillna(0)
                for mc in clf_fu:
                    if mc not in Xcl.columns: Xcl[mc]=0
                Xcl=Xcl[clf_fu]; prob=rf_u.predict_proba(Xcl)[:,1]
                out["buy_probability"]=np.round(prob,4); out["buy_prediction"]=(prob>=0.5).astype(int)
                out["priority_lead"]=(prob>=0.65).astype(bool)
            else:
                out["buy_probability"]=np.nan; out["buy_prediction"]=np.nan; out["priority_lead"]=False

            # Spend
            rfc=[c for c in reg_fu if c in out.columns]
            if len(rfc)>=4:
                Xrg=out[rfc].fillna(0)
                for mc in reg_fu:
                    if mc not in Xrg.columns: Xrg[mc]=0
                Xrg=Xrg[reg_fu]; out["predicted_spend"]=np.round(rfr_u.predict(Xrg),0).astype(int)
            else:
                out["predicted_spend"]=np.nan

            out["recommended_offer"]=out["predicted_dna_segment"].map(lambda s: PRESCRIPTIVE.get(str(s),{}).get("offer","Standard offer"))
            out["recommended_bundle"]=out["predicted_dna_segment"].map(lambda s: PRESCRIPTIVE.get(str(s),{}).get("bundle","Curated bundle"))

        st.markdown("---"); s1,s2,s3,s4=st.columns(4)
        s1.metric("Respondents",len(out))
        if out["buy_probability"].notna().any():
            s2.metric("Will buy",f"{out['buy_prediction'].mean()*100:.1f}%")
            s3.metric("Priority leads",int(out["priority_lead"].sum()))
        if out["predicted_spend"].notna().any():
            s4.metric("Avg predicted spend",f"Rs{out['predicted_spend'].mean():,.0f}")

        sn=out["predicted_dna_segment"].value_counts().reset_index(); sn.columns=["Segment","Count"]
        fsn=go.Figure(go.Bar(x=sn["Segment"],y=sn["Count"],
            marker_color=[CLUSTER_COLORS.get(s,"#888") for s in sn["Segment"]],
            text=sn["Count"],textposition="outside"))
        fsn.update_layout(height=270,xaxis_title=None,yaxis=dict(showgrid=True,gridcolor="#eee"),
            plot_bgcolor="white",paper_bgcolor="white",margin=dict(t=10,b=10))
        st.plotly_chart(fsn,use_container_width=True)

        disp_c=(["respondent_id"] if "respondent_id" in out.columns else [])+\
            [c for c in ["predicted_dna_segment","buy_probability","buy_prediction",
                "priority_lead","predicted_spend","recommended_offer","recommended_bundle"] if c in out.columns]
        st.dataframe(out[disp_c].head(25),use_container_width=True)
        st.download_button("Download enriched CSV",out.to_csv(index=False),
            file_name="book_dna_predictions.csv",mime="text/csv")
        st.markdown('<div class="ins">Filter priority_lead=True for highest-ROI outreach. Use recommended_offer for personalised messaging.</div>',unsafe_allow_html=True)
    else:
        st.info("Upload a CSV using the button above. Download the template first to see the required columns.")
