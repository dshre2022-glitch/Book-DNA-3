import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.tree import export_text
from utils import (load_data, get_clean, train_classifiers, train_format_clf,
                   train_regressors, CLF_FEATURES, REG_FEATURES,
                   CLUSTER_COLORS, FORMAT_CLASS, PAY_LABELS)

st.set_page_config(page_title="Predictive - Book DNA", layout="wide")
st.markdown("""<style>
[data-testid="stSidebar"]{background:#1a1612;}
[data-testid="stSidebar"] *{color:#faf8f4 !important;}
.mc{background:#faf8f4;border:1px solid #ddd8ce;border-radius:10px;padding:14px;text-align:center;}
.mv{font-size:1.6rem;font-weight:700;color:#1a1612;}
.ml{font-size:.7rem;color:#7a736b;text-transform:uppercase;letter-spacing:.06em;margin-top:4px;}
.ins{background:#f3f0ea;border-left:4px solid #c8922a;border-radius:0 8px 8px 0;
     padding:12px 16px;font-size:.87rem;color:#4a4540;line-height:1.7;margin:8px 0;}
</style>""", unsafe_allow_html=True)

if "df" not in st.session_state: st.session_state["df"] = load_data()
df=st.session_state.get("df_up",st.session_state["df"]); clean=get_clean(df)

st.title("🔮 Predictive Models")
st.markdown("---")

t1,t2,t3,t4=st.tabs(["🎯 Buy Classification","📖 Format Classification","💰 Spend Regression","🔬 Live Predict"])

# ── TAB 1 ─────────────────────────────────────────────────────────────
with t1:
    st.subheader("Will the customer buy? — Binary classification")
    if "clf" not in st.session_state:
        with st.spinner("Training classifiers..."):
            st.session_state["clf"]=train_classifiers(clean)
    res,feats,sc,rf_m,dt_m=st.session_state["clf"]

    sel=st.selectbox("Model:",list(res.keys()))
    r=res[sel]

    st.markdown("#### Performance metrics")
    m1,m2,m3,m4,m5=st.columns(5)
    for col,lbl,val in [(m1,"Accuracy",f"{r['accuracy']*100:.2f}%"),(m2,"Precision",f"{r['precision']*100:.2f}%"),
        (m3,"Recall",f"{r['recall']*100:.2f}%"),(m4,"F1-Score",f"{r['f1']*100:.2f}%"),(m5,"ROC-AUC",f"{r['roc_auc']:.4f}")]:
        with col: st.markdown(f'<div class="mc"><div class="mv">{val}</div><div class="ml">{lbl}</div></div>',unsafe_allow_html=True)

    st.markdown("---")
    cc,cr=st.columns(2)
    with cc:
        st.markdown("#### Confusion matrix")
        cm=r["cm"]; lbls=["Won't buy","Will buy"]
        fcm=go.Figure(go.Heatmap(z=cm,x=lbls,y=lbls,colorscale=[[0,"#EEEDFE"],[1,"#5C3D8F"]],
            text=cm,texttemplate="<b>%{text}</b>",textfont=dict(size=20),showscale=False))
        fcm.update_layout(height=310,margin=dict(t=10,b=20),
            xaxis_title="Predicted",yaxis_title="Actual",plot_bgcolor="white",paper_bgcolor="white")
        st.plotly_chart(fcm,use_container_width=True)
        tn,fp,fn,tp=cm.ravel(); st.caption(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    with cr:
        st.markdown("#### ROC curve — all models")
        roc_c={"Random Forest":"#5C3D8F","Logistic Regression":"#C8922A","Decision Tree":"#1A6B5A"}
        froc=go.Figure()
        froc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",line=dict(dash="dash",color="#ccc",width=1.5),name="Random"))
        for nm,ri in res.items():
            froc.add_trace(go.Scatter(x=ri["fpr"],y=ri["tpr"],mode="lines",
                line=dict(color=roc_c.get(nm,"#888"),width=3 if nm==sel else 1.5),
                name=f"{nm} (AUC={ri['roc_auc']:.3f})"))
        froc.update_layout(height=310,margin=dict(t=10,b=20),
            xaxis=dict(title="FPR",range=[0,1],showgrid=True,gridcolor="#eee"),
            yaxis=dict(title="TPR",range=[0,1],showgrid=True,gridcolor="#eee"),
            legend=dict(x=0.3,y=0.1,font=dict(size=10)),plot_bgcolor="white",paper_bgcolor="white")
        st.plotly_chart(froc,use_container_width=True)

    st.markdown("---"); st.markdown("#### Feature importance — Random Forest")
    fi=pd.DataFrame({"Feature":feats,"Importance":rf_m.feature_importances_}).sort_values("Importance",ascending=True).tail(20)
    ffi=go.Figure(go.Bar(y=fi["Feature"],x=fi["Importance"],orientation="h",
        marker=dict(color=fi["Importance"],colorscale="Purples",showscale=False),
        text=[f"{v:.4f}" for v in fi["Importance"]],textposition="outside"))
    ffi.update_layout(height=500,margin=dict(t=10,b=10,l=10,r=80),
        xaxis=dict(title="Importance",showgrid=True,gridcolor="#eee"),
        yaxis_title=None,plot_bgcolor="white",paper_bgcolor="white")
    st.plotly_chart(ffi,use_container_width=True)

    st.markdown("---"); st.markdown("#### Decision tree rules")
    feats_dt=[c for c in CLF_FEATURES if c in clean.columns]
    txt=export_text(dt_m,feature_names=feats_dt,max_depth=4)
    if len(txt)>3500: txt=txt[:3500]+"\n... [truncated]"
    with st.expander("View rules"): st.code(txt)

    st.markdown("---"); st.markdown("#### Model comparison")
    st.dataframe(pd.DataFrame([{"Model":n,"Accuracy":f"{ri['accuracy']*100:.2f}%",
        "Precision":f"{ri['precision']*100:.2f}%","Recall":f"{ri['recall']*100:.2f}%",
        "F1":f"{ri['f1']*100:.2f}%","AUC":f"{ri['roc_auc']:.4f}"} for n,ri in res.items()]).set_index("Model"),
        use_container_width=True)
    st.markdown('<div class="ins">Top predictors: nps_proxy, stress_score, personalization_importance, switching_tendency.</div>',unsafe_allow_html=True)

# ── TAB 2 ─────────────────────────────────────────────────────────────
with t2:
    st.subheader("Format preference — 3-class classification")
    if "fmt" not in st.session_state:
        with st.spinner("Training..."):
            st.session_state["fmt"]=train_format_clf(clean)
    fm,ff,fm_=st.session_state["fmt"]
    f1c,f2c,f3c,f4c=st.columns(4)
    for col,lbl,val in [(f1c,"Accuracy",f"{fm_['accuracy']*100:.2f}%"),(f2c,"Precision(wtd)",f"{fm_['precision']*100:.2f}%"),
        (f3c,"Recall(wtd)",f"{fm_['recall']*100:.2f}%"),(f4c,"F1(wtd)",f"{fm_['f1']*100:.2f}%")]:
        with col: st.markdown(f'<div class="mc"><div class="mv">{val}</div><div class="ml">{lbl}</div></div>',unsafe_allow_html=True)
    st.markdown("---")
    fa2,fb2=st.columns(2)
    with fa2:
        fcm2=go.Figure(go.Heatmap(z=fm_["cm"],x=fm_["classes"],y=fm_["classes"],
            colorscale=[[0,"#E1F5EE"],[1,"#0F6E56"]],
            text=fm_["cm"],texttemplate="<b>%{text}</b>",textfont=dict(size=18),showscale=False))
        fcm2.update_layout(height=300,margin=dict(t=10,b=20),
            xaxis_title="Predicted",yaxis_title="Actual",plot_bgcolor="white",paper_bgcolor="white")
        st.plotly_chart(fcm2,use_container_width=True)
    with fb2:
        fd=clean["format_class"].map(FORMAT_CLASS).value_counts().reset_index(); fd.columns=["Format","Count"]
        ffd=go.Figure(go.Pie(labels=fd["Format"],values=fd["Count"],hole=0.42,
            marker_colors=["#5C3D8F","#C8922A","#1A6B5A"]))
        ffd.update_layout(height=300,margin=dict(t=10,b=20))
        st.plotly_chart(ffd,use_container_width=True)

# ── TAB 3 ─────────────────────────────────────────────────────────────
with t3:
    st.subheader("Predicting max single spend — Regression")
    if "reg" not in st.session_state:
        with st.spinner("Training regressors..."):
            st.session_state["reg"]=train_regressors(clean)
    reg_res,reg_f,reg_sc,rf_reg,reg_coef=st.session_state["reg"]

    rsel=st.selectbox("Model:",list(reg_res.keys()))
    rr=reg_res[rsel]
    ra2,rb2=st.columns(2)
    ra2.markdown(f'<div class="mc"><div class="mv">{rr["r2"]:.4f}</div><div class="ml">R² Score</div></div>',unsafe_allow_html=True)
    rb2.markdown(f'<div class="mc"><div class="mv">Rs{rr["rmse"]:,.0f}</div><div class="ml">RMSE</div></div>',unsafe_allow_html=True)
    st.markdown("---")
    rc,rd=st.columns(2)
    with rc:
        maxv=float(max(rr["y_test"].max(),rr["y_pred"].max()))
        fpa=go.Figure()
        fpa.add_trace(go.Scatter(x=rr["y_test"],y=rr["y_pred"],mode="markers",
            marker=dict(color="#5C3D8F",size=5,opacity=0.45),name="Predictions"))
        fpa.add_trace(go.Scatter(x=[0,maxv],y=[0,maxv],mode="lines",
            line=dict(dash="dash",color="#C8922A",width=2),name="Perfect fit"))
        fpa.update_layout(title="Predicted vs Actual",height=360,
            xaxis=dict(title="Actual (Rs)",showgrid=True,gridcolor="#eee"),
            yaxis=dict(title="Predicted (Rs)",showgrid=True,gridcolor="#eee"),
            plot_bgcolor="white",paper_bgcolor="white",margin=dict(t=40,b=20))
        st.plotly_chart(fpa,use_container_width=True)
    with rd:
        resid=rr["y_test"]-rr["y_pred"]
        fres=go.Figure(go.Histogram(x=resid,nbinsx=40,marker_color="#5C3D8F",opacity=0.78))
        fres.add_vline(x=0,line_dash="dash",line_color="#C8922A",line_width=2)
        fres.update_layout(title="Residual distribution",height=360,
            xaxis=dict(title="Residual",showgrid=True,gridcolor="#eee"),
            yaxis=dict(title="Count",showgrid=True,gridcolor="#eee"),
            plot_bgcolor="white",paper_bgcolor="white",margin=dict(t=40,b=20))
        st.plotly_chart(fres,use_container_width=True)

    st.markdown("---"); st.markdown("#### Feature importance — RF Regressor")
    rfi=pd.DataFrame({"Feature":reg_f,"Importance":rf_reg.feature_importances_}).sort_values("Importance",ascending=True)
    frfi=go.Figure(go.Bar(y=rfi["Feature"],x=rfi["Importance"],orientation="h",
        marker=dict(color=rfi["Importance"],colorscale="Teal",showscale=False),
        text=[f"{v:.4f}" for v in rfi["Importance"]],textposition="outside"))
    frfi.update_layout(height=380,margin=dict(t=10,b=10,l=10,r=80),
        xaxis=dict(title="Importance",showgrid=True,gridcolor="#eee"),
        yaxis_title=None,plot_bgcolor="white",paper_bgcolor="white")
    st.plotly_chart(frfi,use_container_width=True)

    st.markdown("---"); st.markdown("#### Ridge coefficients")
    cdf=pd.DataFrame({"Feature":list(reg_coef.keys()),"Coef":list(reg_coef.values())}).sort_values("Coef",key=abs,ascending=True)
    fcf=go.Figure(go.Bar(y=cdf["Feature"],x=cdf["Coef"],orientation="h",
        marker_color=["#1A6B5A" if v>=0 else "#D45379" for v in cdf["Coef"]],
        text=[f"{v:+.1f}" for v in cdf["Coef"]],textposition="outside"))
    fcf.add_vline(x=0,line_color="#ccc",line_width=1)
    fcf.update_layout(height=380,margin=dict(t=10,b=10,l=10,r=80),
        xaxis=dict(title="Coefficient",showgrid=True,gridcolor="#eee"),
        yaxis_title=None,plot_bgcolor="white",paper_bgcolor="white")
    st.plotly_chart(fcf,use_container_width=True)
    st.markdown('<div class="ins">Income and lifestyle spend are the strongest predictors (R²=0.86). NPS proxy is also significant.</div>',unsafe_allow_html=True)

# ── TAB 4 ─────────────────────────────────────────────────────────────
with t4:
    st.subheader("Live prediction — enter a customer profile")
    lc1,lc2,lc3=st.columns(3)
    with lc1:
        inp_age   =st.select_slider("Age",[2,3,4,5,6,7],value=3,format_func=lambda x:{2:"13-17",3:"18-22",4:"23-28",5:"29-35",6:"36-45",7:"46+"}[x])
        inp_city  =st.select_slider("City tier",[1,2,3,4,5],format_func=lambda x:{1:"Metro",2:"Tier1",3:"Tier2",4:"Tier3",5:"Rural"}[x])
        inp_income=st.select_slider("Income (Rs)",[5000,15000,30000,60000,100000],format_func=lambda x:f"Rs{x:,}")
    with lc2:
        inp_stress=st.slider("Stress (0-16)",0,16,8)
        inp_open  =st.slider("Openness (1-5)",1,5,3)
        inp_nps   =st.slider("NPS (0-10)",0,10,6)
    with lc3:
        inp_books =st.select_slider("Books/month",[0,1,2.5,5,8])
        inp_switch=st.slider("Switching (1-4)",1,4,2)
        inp_perso =st.slider("Personalisation (1-5)",1,5,4)

    if st.button("Predict", type="primary"):
        if "clf" not in st.session_state:
            with st.spinner("Training..."): st.session_state["clf"]=train_classifiers(clean)
        if "reg" not in st.session_state:
            with st.spinner("Training..."): st.session_state["reg"]=train_regressors(clean)
        res2,clf_f2,_,rf2,_=st.session_state["clf"]
        reg2,reg_f2,_,rfr2,_=st.session_state["reg"]

        raw_lifestyle = inp_income * 0.06 * {"Midnight Escapist":0.8,"Productivity Achiever":1.3,
            "Emotional Reader":0.9,"Curious Explorer":0.7,"Non-Reader":0.3}.get(
            "Midnight Escapist" if inp_stress>=10 else "Productivity Achiever" if inp_income>=45000 else "Curious Explorer" if inp_open>=4 else "Emotional Reader", 0.9)

        base={f:0 for f in clf_f2}
        base.update({"age_group":inp_age,"city_tier":inp_city,"monthly_income_midpoint":inp_income,
            "stress_score":inp_stress,"openness_score":inp_open,"nps_proxy":inp_nps,
            "books_per_month":inp_books,"switching_tendency":inp_switch,
            "personalization_importance":inp_perso,"conscientiousness_score":3,
            "neuroticism_score":max(1,inp_stress//3),"agreeableness_score":3,
            "lifestyle_spend":int(raw_lifestyle),"purchase_intent":2,"platform_interest":2,
            "barrier_price":int(inp_income<15000),"barrier_trust":int(inp_nps<5),
            "barrier_delivery":int(inp_city>=3),"subscription_interest":2,"impulse_buying":2,
            "discovery_social":int(inp_open>=4),"shops_instagram":int(inp_open>=4)})
        Xcl=pd.DataFrame([base])[clf_f2].fillna(0)
        prob=float(rf2.predict_proba(Xcl)[0][1])

        base_r={f:0 for f in reg_f2}; base_r.update({k:v for k,v in base.items() if k in reg_f2})
        Xrg=pd.DataFrame([base_r])[reg_f2].fillna(0)
        pred_spend=float(rfr2.predict(Xrg)[0])

        pr1,pr2,pr3=st.columns(3)
        bc="#1A6B5A" if prob>=0.5 else "#D45379"; bl="Will buy" if prob>=0.5 else "Won't buy"
        pr1.markdown(f'<div class="mc" style="border-top:4px solid {bc}"><div class="mv" style="color:{bc}">{bl}</div><div class="ml">Purchase intent</div></div>',unsafe_allow_html=True)
        pr2.markdown(f'<div class="mc"><div class="mv">{prob*100:.1f}%</div><div class="ml">Buy probability</div></div>',unsafe_allow_html=True)
        pr3.markdown(f'<div class="mc"><div class="mv">Rs{pred_spend:,.0f}</div><div class="ml">Predicted max spend</div></div>',unsafe_allow_html=True)

        est=("Midnight Escapist" if inp_stress>=10 else "Productivity Achiever" if inp_income>=45000 and inp_open<=3 else "Curious Explorer" if inp_open>=4 else "Emotional Reader")
        ec=CLUSTER_COLORS.get(est,"#888")
        st.markdown(f'<div style="margin-top:12px;border-left:4px solid {ec};background:#faf8f4;border-radius:0 8px 8px 0;padding:12px 16px;"><strong>Estimated DNA segment:</strong> <span style="color:{ec};font-weight:600;margin-left:8px">{est}</span></div>',unsafe_allow_html=True)
