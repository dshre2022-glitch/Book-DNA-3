"""Book DNA - Synthetic Dataset Generator. Run once: python generate_data.py"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 2000

SEGMENTS = {
    "Midnight Escapist":     0.30,
    "Productivity Achiever": 0.22,
    "Emotional Reader":      0.18,
    "Curious Explorer":      0.18,
    "Non-Reader":            0.12,
}
segs = np.random.choice(list(SEGMENTS.keys()), size=N, p=list(SEGMENTS.values()))

rows = []
for i, seg in enumerate(segs):
    r = {}
    r["respondent_id"] = f"BDNA{i+1:04d}"
    r["dna_segment"]   = seg

    inc_opts = [5000, 15000, 30000, 60000, 100000]
    inc_p = {
        "Midnight Escapist":     [0.30, 0.35, 0.22, 0.10, 0.03],
        "Productivity Achiever": [0.05, 0.18, 0.35, 0.30, 0.12],
        "Emotional Reader":      [0.15, 0.32, 0.30, 0.16, 0.07],
        "Curious Explorer":      [0.22, 0.40, 0.25, 0.10, 0.03],
        "Non-Reader":            [0.25, 0.35, 0.25, 0.12, 0.03],
    }
    income = int(np.random.choice(inc_opts, p=inc_p[seg]))
    r["monthly_income_midpoint"] = income

    r["age_group"] = int(np.random.choice([2,3,4,5,6,7], p={
        "Midnight Escapist":[0.18,0.38,0.22,0.12,0.06,0.04],
        "Productivity Achiever":[0.08,0.22,0.30,0.24,0.11,0.05],
        "Emotional Reader":[0.10,0.28,0.26,0.20,0.11,0.05],
        "Curious Explorer":[0.20,0.42,0.22,0.10,0.04,0.02],
        "Non-Reader":[0.10,0.20,0.24,0.22,0.14,0.10]}[seg]))
    r["city_tier"] = int(np.random.choice([1,2,3,4,5], p={
        "Midnight Escapist":[0.22,0.25,0.28,0.18,0.07],
        "Productivity Achiever":[0.38,0.30,0.20,0.09,0.03],
        "Emotional Reader":[0.28,0.28,0.24,0.14,0.06],
        "Curious Explorer":[0.30,0.26,0.24,0.15,0.05],
        "Non-Reader":[0.20,0.22,0.28,0.20,0.10]}[seg]))
    r["gender"]     = int(np.random.choice([1,2,3,4], p=[0.43,0.52,0.03,0.02]))
    r["occupation"] = int(np.random.choice([1,2,3,4,5,6], p={
        "Midnight Escapist":[0.20,0.42,0.20,0.08,0.05,0.05],
        "Productivity Achiever":[0.05,0.25,0.48,0.14,0.04,0.04],
        "Emotional Reader":[0.12,0.38,0.28,0.08,0.10,0.04],
        "Curious Explorer":[0.18,0.48,0.18,0.08,0.04,0.04],
        "Non-Reader":[0.15,0.28,0.32,0.10,0.10,0.05]}[seg]))

    ocean_mu = {"Midnight Escapist":[2.8,2.6,2.5,3.8,4.2],
                "Productivity Achiever":[3.4,4.5,3.2,3.0,2.4],
                "Emotional Reader":[3.5,3.2,2.8,4.6,3.2],
                "Curious Explorer":[4.6,2.9,4.0,3.4,2.8],
                "Non-Reader":[2.5,2.8,3.0,3.0,2.5]}
    for idx, trait in enumerate(["openness","conscientiousness","extraversion","agreeableness","neuroticism"]):
        r[f"{trait}_score"] = int(np.clip(round(np.random.normal(ocean_mu[seg][idx], 0.7)), 1, 5))

    stress_mu = {"Midnight Escapist":10.5,"Productivity Achiever":6.0,
                 "Emotional Reader":7.8,"Curious Explorer":5.5,"Non-Reader":5.0}
    r["stress_score"] = int(np.clip(round(np.random.normal(stress_mu[seg], 2.2)), 0, 16))

    r["life_stage"] = int(np.random.choice([1,2,3,4,5,6,7], p={
        "Midnight Escapist":[0.35,0.20,0.15,0.10,0.05,0.10,0.05],
        "Productivity Achiever":[0.10,0.12,0.42,0.08,0.08,0.05,0.15],
        "Emotional Reader":[0.08,0.20,0.18,0.22,0.10,0.08,0.14],
        "Curious Explorer":[0.22,0.38,0.18,0.08,0.04,0.04,0.06],
        "Non-Reader":[0.12,0.15,0.25,0.10,0.12,0.08,0.18]}[seg]))
    r["current_mood"] = int(np.random.choice([1,2,3,4,5,6], p={
        "Midnight Escapist":[0.40,0.12,0.22,0.10,0.08,0.08],
        "Productivity Achiever":[0.08,0.48,0.10,0.16,0.12,0.06],
        "Emotional Reader":[0.15,0.10,0.35,0.16,0.14,0.10],
        "Curious Explorer":[0.08,0.18,0.12,0.40,0.14,0.08],
        "Non-Reader":[0.18,0.12,0.16,0.14,0.22,0.18]}[seg]))
    r["reader_identity"] = int(np.random.choice([1,2,3,4,5,6], p={
        "Midnight Escapist":[0.50,0.10,0.12,0.14,0.08,0.06],
        "Productivity Achiever":[0.06,0.50,0.06,0.14,0.16,0.08],
        "Emotional Reader":[0.12,0.08,0.48,0.10,0.12,0.10],
        "Curious Explorer":[0.08,0.10,0.10,0.48,0.14,0.10],
        "Non-Reader":[0.06,0.06,0.08,0.12,0.12,0.56]}[seg]))
    r["personalization_importance"] = int(np.clip(round(np.random.normal(
        {"Midnight Escapist":4.2,"Productivity Achiever":4.0,"Emotional Reader":3.8,
         "Curious Explorer":4.4,"Non-Reader":2.5}[seg], 0.8)), 1, 5))

    r["books_per_month"] = float(np.random.choice([0,1,2.5,5,8], p={
        "Midnight Escapist":[0.05,0.20,0.38,0.25,0.12],
        "Productivity Achiever":[0.04,0.18,0.35,0.28,0.15],
        "Emotional Reader":[0.06,0.22,0.40,0.22,0.10],
        "Curious Explorer":[0.08,0.25,0.38,0.20,0.09],
        "Non-Reader":[0.50,0.28,0.15,0.05,0.02]}[seg]))

    for col, tp in {
        "reads_morning":   {"Midnight Escapist":0.15,"Productivity Achiever":0.55,"Emotional Reader":0.30,"Curious Explorer":0.35,"Non-Reader":0.15},
        "reads_commute":   {"Midnight Escapist":0.30,"Productivity Achiever":0.40,"Emotional Reader":0.28,"Curious Explorer":0.32,"Non-Reader":0.12},
        "reads_afternoon": {"Midnight Escapist":0.20,"Productivity Achiever":0.25,"Emotional Reader":0.25,"Curious Explorer":0.28,"Non-Reader":0.10},
        "reads_evening":   {"Midnight Escapist":0.45,"Productivity Achiever":0.38,"Emotional Reader":0.45,"Curious Explorer":0.42,"Non-Reader":0.18},
        "reads_latenight": {"Midnight Escapist":0.72,"Productivity Achiever":0.22,"Emotional Reader":0.38,"Curious Explorer":0.35,"Non-Reader":0.08},
        "reads_weekend":   {"Midnight Escapist":0.40,"Productivity Achiever":0.30,"Emotional Reader":0.38,"Curious Explorer":0.35,"Non-Reader":0.20},
    }.items():
        r[col] = int(np.random.random() < tp[seg])

    r["reading_motivation"] = int(np.random.choice([1,2,3,4,5,6], p={
        "Midnight Escapist":[0.55,0.08,0.18,0.06,0.08,0.05],
        "Productivity Achiever":[0.06,0.55,0.10,0.18,0.06,0.05],
        "Emotional Reader":[0.18,0.10,0.38,0.08,0.18,0.08],
        "Curious Explorer":[0.10,0.22,0.30,0.10,0.18,0.10],
        "Non-Reader":[0.10,0.08,0.14,0.16,0.08,0.44]}[seg]))

    for g, gp in {
        "genre_fantasy":  {"Midnight Escapist":0.75,"Productivity Achiever":0.18,"Emotional Reader":0.28,"Curious Explorer":0.50,"Non-Reader":0.10},
        "genre_selfhelp": {"Midnight Escapist":0.22,"Productivity Achiever":0.80,"Emotional Reader":0.28,"Curious Explorer":0.38,"Non-Reader":0.12},
        "genre_literary": {"Midnight Escapist":0.30,"Productivity Achiever":0.20,"Emotional Reader":0.70,"Curious Explorer":0.40,"Non-Reader":0.08},
        "genre_romance":  {"Midnight Escapist":0.35,"Productivity Achiever":0.12,"Emotional Reader":0.62,"Curious Explorer":0.22,"Non-Reader":0.08},
        "genre_thriller": {"Midnight Escapist":0.48,"Productivity Achiever":0.30,"Emotional Reader":0.32,"Curious Explorer":0.38,"Non-Reader":0.10},
        "genre_biography":{"Midnight Escapist":0.15,"Productivity Achiever":0.55,"Emotional Reader":0.28,"Curious Explorer":0.30,"Non-Reader":0.08},
        "genre_business": {"Midnight Escapist":0.08,"Productivity Achiever":0.68,"Emotional Reader":0.10,"Curious Explorer":0.22,"Non-Reader":0.06},
    }.items():
        r[g] = int(np.random.random() < gp[seg])

    for d, dp in {
        "discovery_social":  {"Midnight Escapist":0.60,"Productivity Achiever":0.35,"Emotional Reader":0.48,"Curious Explorer":0.78,"Non-Reader":0.25},
        "discovery_friends": {"Midnight Escapist":0.45,"Productivity Achiever":0.38,"Emotional Reader":0.52,"Curious Explorer":0.42,"Non-Reader":0.28},
        "discovery_apps":    {"Midnight Escapist":0.30,"Productivity Achiever":0.38,"Emotional Reader":0.28,"Curious Explorer":0.42,"Non-Reader":0.12},
    }.items():
        r[d] = int(np.random.random() < dp[seg])

    r["social_sharing_level"] = int(np.clip(round(np.random.normal(
        {"Midnight Escapist":2.8,"Productivity Achiever":2.5,"Emotional Reader":2.6,
         "Curious Explorer":1.8,"Non-Reader":3.5}[seg], 0.8)), 1, 4))
    r["format_preference"] = int(np.random.choice([1,2,3,4,5], p={
        "Midnight Escapist":[0.48,0.28,0.14,0.07,0.03],
        "Productivity Achiever":[0.22,0.26,0.22,0.20,0.10],
        "Emotional Reader":[0.40,0.30,0.18,0.08,0.04],
        "Curious Explorer":[0.28,0.24,0.22,0.16,0.10],
        "Non-Reader":[0.20,0.22,0.28,0.20,0.10]}[seg]))

    for p, pp in {
        "interest_books":      {"Midnight Escapist":0.78,"Productivity Achiever":0.70,"Emotional Reader":0.72,"Curious Explorer":0.68,"Non-Reader":0.20},
        "interest_journal":    {"Midnight Escapist":0.65,"Productivity Achiever":0.55,"Emotional Reader":0.60,"Curious Explorer":0.50,"Non-Reader":0.15},
        "interest_bookmark":   {"Midnight Escapist":0.58,"Productivity Achiever":0.42,"Emotional Reader":0.52,"Curious Explorer":0.55,"Non-Reader":0.12},
        "interest_candle":     {"Midnight Escapist":0.72,"Productivity Achiever":0.28,"Emotional Reader":0.60,"Curious Explorer":0.38,"Non-Reader":0.10},
        "interest_tote":       {"Midnight Escapist":0.50,"Productivity Achiever":0.38,"Emotional Reader":0.45,"Curious Explorer":0.60,"Non-Reader":0.12},
        "interest_apparel":    {"Midnight Escapist":0.42,"Productivity Achiever":0.30,"Emotional Reader":0.35,"Curious Explorer":0.65,"Non-Reader":0.08},
        "interest_decor":      {"Midnight Escapist":0.60,"Productivity Achiever":0.25,"Emotional Reader":0.48,"Curious Explorer":0.40,"Non-Reader":0.08},
        "interest_annotation": {"Midnight Escapist":0.28,"Productivity Achiever":0.65,"Emotional Reader":0.32,"Curious Explorer":0.38,"Non-Reader":0.05},
        "interest_pin":        {"Midnight Escapist":0.35,"Productivity Achiever":0.20,"Emotional Reader":0.28,"Curious Explorer":0.55,"Non-Reader":0.06},
    }.items():
        r[p] = int(np.random.random() < pp[seg])

    r["subscription_interest"] = int(np.clip(round(np.random.normal(
        {"Midnight Escapist":1.8,"Productivity Achiever":2.2,"Emotional Reader":2.0,
         "Curious Explorer":2.0,"Non-Reader":3.5}[seg], 0.8)), 1, 4))
    r["eco_importance"]  = int(np.clip(round(np.random.normal(2.5, 0.9)), 1, 4))
    r["gifting_behaviour"] = int(np.clip(round(np.random.normal(
        {"Midnight Escapist":2.2,"Productivity Achiever":2.0,"Emotional Reader":1.8,
         "Curious Explorer":2.2,"Non-Reader":3.0}[seg], 0.8)), 1, 4))

    for b, bp in {
        "barrier_price":    {"Midnight Escapist":0.45,"Productivity Achiever":0.22,"Emotional Reader":0.35,"Curious Explorer":0.38,"Non-Reader":0.40},
        "barrier_quality":  {"Midnight Escapist":0.32,"Productivity Achiever":0.25,"Emotional Reader":0.28,"Curious Explorer":0.22,"Non-Reader":0.30},
        "barrier_trust":    {"Midnight Escapist":0.28,"Productivity Achiever":0.18,"Emotional Reader":0.24,"Curious Explorer":0.20,"Non-Reader":0.35},
        "barrier_delivery": {"Midnight Escapist":0.22,"Productivity Achiever":0.12,"Emotional Reader":0.18,"Curious Explorer":0.15,"Non-Reader":0.28},
        "barrier_privacy":  {"Midnight Escapist":0.15,"Productivity Achiever":0.20,"Emotional Reader":0.12,"Curious Explorer":0.18,"Non-Reader":0.15},
        "barrier_platform": {"Midnight Escapist":0.20,"Productivity Achiever":0.25,"Emotional Reader":0.18,"Curious Explorer":0.15,"Non-Reader":0.30},
    }.items():
        r[b] = int(np.random.random() < bp[seg])

    r["trust_driver"] = int(np.random.choice([1,2,3,4,5,6], p={
        "Midnight Escapist":[0.30,0.28,0.18,0.08,0.12,0.04],
        "Productivity Achiever":[0.15,0.25,0.20,0.18,0.12,0.10],
        "Emotional Reader":[0.20,0.22,0.18,0.10,0.22,0.08],
        "Curious Explorer":[0.38,0.30,0.12,0.06,0.10,0.04],
        "Non-Reader":[0.14,0.18,0.25,0.12,0.18,0.13]}[seg]))
    r["discount_preference"] = int(np.random.choice([1,2,3,4,5,6], p={
        "Midnight Escapist":[0.22,0.35,0.20,0.12,0.08,0.03],
        "Productivity Achiever":[0.18,0.20,0.15,0.28,0.12,0.07],
        "Emotional Reader":[0.20,0.28,0.18,0.16,0.12,0.06],
        "Curious Explorer":[0.25,0.28,0.18,0.10,0.12,0.07],
        "Non-Reader":[0.28,0.22,0.20,0.10,0.14,0.06]}[seg]))

    base_b = {"Midnight Escapist":320,"Productivity Achiever":550,"Emotional Reader":380,
               "Curious Explorer":280,"Non-Reader":180}[seg]
    psm_b  = max(50,  int(np.random.normal(base_b + income/1000*8, 70)))
    psm_tc = max(30,  int(np.random.normal(psm_b * 0.45, 50)))
    psm_eo = max(psm_b+50, int(np.random.normal(psm_b * 1.60, 80)))
    psm_te = max(psm_eo+80, int(np.random.normal(psm_b * 2.30, 120)))
    r["psm_too_cheap"] = psm_tc; r["psm_bargain"] = psm_b
    r["psm_expensive_ok"] = psm_eo; r["psm_too_expensive"] = psm_te

    seg_mult = {"Midnight Escapist":0.80,"Productivity Achiever":1.30,
                "Emotional Reader":0.90,"Curious Explorer":0.70,"Non-Reader":0.30}
    raw = income * seg_mult[seg] * 0.04 + np.random.normal(0, income * 0.01)
    raw = max(150, raw)
    r["max_single_spend"] = int(150 if raw<250 else 350 if raw<500 else 750 if raw<900 else 1500 if raw<1800 else 2500)

    raw_ls = income * seg_mult[seg] * 0.06 + np.random.normal(0, income * 0.015)
    raw_ls = max(250, raw_ls)
    r["lifestyle_spend"] = int(250 if raw_ls<600 else 1000 if raw_ls<1500 else 2250 if raw_ls<3000 else 4500 if raw_ls<6000 else 7500)

    raw_bs = income * 0.015 + np.random.normal(0, income * 0.005)
    raw_bs = max(0, raw_bs)
    r["monthly_book_spend"] = int(0 if raw_bs<50 else 100 if raw_bs<200 else 350 if raw_bs<500 else 750 if raw_bs<900 else 1500)

    for s, sp in {
        "shops_amazon":    {"Midnight Escapist":0.65,"Productivity Achiever":0.70,"Emotional Reader":0.62,"Curious Explorer":0.58,"Non-Reader":0.55},
        "shops_flipkart":  {"Midnight Escapist":0.45,"Productivity Achiever":0.42,"Emotional Reader":0.40,"Curious Explorer":0.38,"Non-Reader":0.35},
        "shops_d2c":       {"Midnight Escapist":0.28,"Productivity Achiever":0.38,"Emotional Reader":0.30,"Curious Explorer":0.32,"Non-Reader":0.10},
        "shops_instagram": {"Midnight Escapist":0.38,"Productivity Achiever":0.22,"Emotional Reader":0.32,"Curious Explorer":0.62,"Non-Reader":0.12},
        "shops_offline":   {"Midnight Escapist":0.28,"Productivity Achiever":0.22,"Emotional Reader":0.30,"Curious Explorer":0.22,"Non-Reader":0.25},
    }.items():
        r[s] = int(np.random.random() < sp[seg])

    r["payment_method"] = int(np.random.choice([1,2,3,4,5], p={
        "Midnight Escapist":[0.60,0.18,0.06,0.14,0.02],
        "Productivity Achiever":[0.48,0.18,0.20,0.08,0.06],
        "Emotional Reader":[0.58,0.18,0.08,0.12,0.04],
        "Curious Explorer":[0.62,0.14,0.06,0.12,0.06],
        "Non-Reader":[0.55,0.20,0.05,0.18,0.02]}[seg]))
    r["impulse_buying"] = int(np.clip(round(np.random.normal(
        {"Midnight Escapist":2.2,"Productivity Achiever":3.0,"Emotional Reader":2.5,
         "Curious Explorer":1.8,"Non-Reader":3.2}[seg], 0.8)), 1, 4))
    r["nps_proxy"] = int(np.clip(round(np.random.normal(
        {"Midnight Escapist":7.2,"Productivity Achiever":6.8,"Emotional Reader":7.0,
         "Curious Explorer":7.8,"Non-Reader":3.5}[seg], 1.8)), 0, 10))
    r["switching_tendency"] = int(np.random.choice([1,2,3,4], p={
        "Midnight Escapist":[0.35,0.30,0.22,0.13],
        "Productivity Achiever":[0.40,0.30,0.20,0.10],
        "Emotional Reader":[0.38,0.30,0.22,0.10],
        "Curious Explorer":[0.28,0.28,0.28,0.16],
        "Non-Reader":[0.20,0.25,0.30,0.25]}[seg]))
    r["platform_interest"] = int(np.random.choice([1,2,3,4,5], p={
        "Midnight Escapist":[0.35,0.38,0.15,0.08,0.04],
        "Productivity Achiever":[0.28,0.35,0.22,0.10,0.05],
        "Emotional Reader":[0.30,0.36,0.20,0.10,0.04],
        "Curious Explorer":[0.30,0.35,0.20,0.10,0.05],
        "Non-Reader":[0.04,0.08,0.20,0.35,0.33]}[seg]))
    r["share_intent"] = int(np.random.choice([1,2,3,4,5], p={
        "Midnight Escapist":[0.30,0.32,0.20,0.12,0.06],
        "Productivity Achiever":[0.20,0.28,0.25,0.18,0.09],
        "Emotional Reader":[0.22,0.28,0.25,0.16,0.09],
        "Curious Explorer":[0.40,0.30,0.16,0.10,0.04],
        "Non-Reader":[0.05,0.10,0.20,0.32,0.33]}[seg]))
    r["reading_habit_status"] = int(np.random.choice([1,2,3,4,5], p={
        "Midnight Escapist":[0.40,0.25,0.15,0.12,0.08],
        "Productivity Achiever":[0.45,0.22,0.14,0.12,0.07],
        "Emotional Reader":[0.38,0.28,0.16,0.12,0.06],
        "Curious Explorer":[0.38,0.20,0.22,0.14,0.06],
        "Non-Reader":[0.05,0.20,0.22,0.25,0.28]}[seg]))
    r["purchase_intent"] = int(np.random.choice([1,2,3,4,5], p={
        "Midnight Escapist":[0.35,0.38,0.15,0.08,0.04],
        "Productivity Achiever":[0.28,0.34,0.22,0.12,0.04],
        "Emotional Reader":[0.30,0.36,0.20,0.10,0.04],
        "Curious Explorer":[0.28,0.35,0.22,0.10,0.05],
        "Non-Reader":[0.03,0.07,0.18,0.36,0.36]}[seg]))

    r["will_buy"]     = int(r["purchase_intent"] <= 2)
    r["format_class"] = 1 if r["format_preference"] <= 2 else (2 if r["format_preference"] == 3 else 3)
    rows.append(r)

df = pd.DataFrame(rows)
miss = np.random.choice(df.index, size=int(N*0.03), replace=False)
df.loc[miss, "psm_bargain"] = np.nan
df["data_quality_flag"] = "clean"
df.loc[miss, "data_quality_flag"] = "missing_psm"
df.to_csv("book_dna_data.csv", index=False)
print(f"Saved book_dna_data.csv  {df.shape[0]} rows x {df.shape[1]} cols")
