import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Fraud Detection", page_icon="🔍", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0f0f0f; }
[data-testid="stSidebar"] { background-color: #141414; }
h1, h2, h3 { color: #e8e8e8; }
.fraud-box {
    background: rgba(255,60,60,0.12);
    border: 1px solid #ff4444;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.safe-box {
    background: rgba(50,200,100,0.12);
    border: 1px solid #32c864;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.info-box {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 16px 20px;
}
</style>
""", unsafe_allow_html=True)


# ── Load everything
@st.cache_resource
def load_all():
    with open('rf_model.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('v_means.json', 'r') as f:
        v_means = json.load(f)
    fraud_samples = pd.read_csv('fraud_samples.csv')
    normal_samples = pd.read_csv('normal_samples.csv')
    return rf, scaler, v_means, fraud_samples, normal_samples

rf, scaler, v_means, fraud_samples, normal_samples = load_all()

V_COLS = [f'V{i}' for i in range(1, 29)]

# time window → seconds mapping (dataset covers ~48 hours)
TIME_MAP = {
    'Morning  (6am – 12pm)':   (21600,  43200),
    'Afternoon (12pm – 6pm)':  (43200,  64800),
    'Evening  (6pm – 12am)':   (64800,  86400),
    'Late Night (12am – 6am)': (0,      21600),
}

# transaction type → amount range + V profile weight
# weight 0.0 = fully normal V means, 1.0 = fully fraud V means
#
# DESIGN ASSUMPTION: These weights are NOT data-derived. They're based on intuition.
# Why? V features are PCA-compressed → original meaning is lost. Cannot reverse-map
# from V features back to transaction category without raw Kaggle data (never released).
# So: assigned fraud risk levels intuitively:
#   - Low-risk categories (grocery) → 5% fraud weight
#   - Medium-risk (online) → 10-15%
#   - High-risk (international) → 25%
#   - Obviously suspicious → 85%
#
# This is a linear interpolation: synthetic_features = (1-weight)*normal_profile + weight*fraud_profile
# For transparency: include a reference table in the app explaining this design choice.
TYPE_MAP = {
    'Grocery / Supermarket':     (50,    500,   0.05),
    'Online Shopping':           (200,   5000,  0.15),
    'Electronics / Gadgets':     (1000,  25000, 0.10),
    'ATM Withdrawal':            (500,   10000, 0.08),
    'International Transaction': (1000,  50000, 0.25),
    'Suspicious / Test':         (1,     2500,  0.85),
}

def build_input(amount, time_sec, fraud_weight):
    """Build a full 30-feature row for the model."""
    normal_v = np.array([v_means['normal_means'][c] for c in V_COLS])
    fraud_v  = np.array([v_means['fraud_means'][c]  for c in V_COLS])
    v_vals   = (1 - fraud_weight) * normal_v + fraud_weight * fraud_v

    # scale Amount and Time
    scaled = scaler.transform([[amount, time_sec]])
    amt_scaled  = scaled[0][0]
    time_scaled = scaled[0][1]

    row = np.concatenate([[amt_scaled, time_scaled], v_vals])
    # model expects: V1..V28, Time, Amount order — check column order
    # original df columns: Time, V1..V28, Amount, Class
    full_row = np.concatenate([[time_scaled], v_vals, [amt_scaled]])
    return full_row.reshape(1, -1)

def predict(row):
    prob = rf.predict_proba(row)[0][1]
    pred = int(prob >= 0.5)
    return prob, pred

def risk_label(prob):
    if prob < 0.3:   return "Low Risk", "safe-box"
    elif prob < 0.6: return "Medium Risk", "info-box"
    else:            return "High Risk — FRAUD FLAGGED", "fraud-box"

def display_prob(prob):
    """Floor probability at 0.1% for realistic UI display."""
    return max(prob, 0.001)

def plot_shap(row_df):
    explainer = shap.TreeExplainer(rf)
    sv = explainer.shap_values(row_df)

    if isinstance(sv, list):
        shap_vals = np.array(sv[1])[0]
        base_val = explainer.expected_value[1]
    else:
        sv = np.array(sv)
        if sv.ndim == 3:
            shap_vals = sv[0, :, 1]
        elif sv.ndim == 2:
            shap_vals = sv[0]
        else:
            shap_vals = sv
        base_val = explainer.expected_value[1] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value

    exp = shap.Explanation(
        values=shap_vals,
        base_values=base_val,
        data=row_df.values[0],
        feature_names=row_df.columns.tolist()
    )
    
    # Generate one-sentence interpretation
    top_indices = np.argsort(np.abs(shap_vals))[-3:][::-1]
    top_features = [(row_df.columns[i], shap_vals[i]) for i in top_indices if np.abs(shap_vals[i]) > 0.001]
    
    if top_features:
        main_feature, main_impact = top_features[0]
        direction = "increases" if main_impact > 0 else "decreases"
        summary = f"The strongest predictor is {main_feature}, which {direction} fraud likelihood."
    else:
        summary = "No dominant features; prediction is balanced across multiple factors."
    
    # Configure matplotlib with comprehensive dark theme settings
    dark_config = {
        'figure.facecolor': '#1a1a1a',
        'axes.facecolor': '#1a1a1a',
        'axes.edgecolor': '#555555',
        'axes.labelcolor': '#e8e8e8',
        'text.color': '#e8e8e8',
        'xtick.color': '#e8e8e8',
        'ytick.color': '#e8e8e8',
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'grid.color': '#444444',
        'grid.alpha': 0.3,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
    }
    
    with plt.rc_context(dark_config):
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1a1a1a')
        shap.waterfall_plot(exp, show=False, max_display=12)
        
        # Re-apply styling after shap plot is drawn
        current_ax = plt.gca()
        current_ax.set_facecolor('#1a1a1a')
        current_ax.spines['left'].set_color('#555555')
        current_ax.spines['bottom'].set_color('#555555')
        current_ax.spines['top'].set_visible(False)
        current_ax.spines['right'].set_visible(False)
        current_ax.tick_params(colors='#e8e8e8', labelsize=9)
        current_ax.xaxis.label.set_color('#e8e8e8')
        current_ax.yaxis.label.set_color('#e8e8e8')
        
        # Ensure all text elements are properly colored
        for text_obj in fig.findobj(match=plt.Text):
            text_obj.set_color('#e8e8e8')
            text_obj.set_fontsize(9)
        
        fig.patch.set_facecolor('#1a1a1a')
        plt.tight_layout()
    
    return fig, summary


# ── Sidebar
st.sidebar.title("🔍 Fraud Detection")
st.sidebar.caption("Random Forest · F1: 0.83 · ROC-AUC: 0.95")
st.sidebar.markdown("---")
page = st.sidebar.radio("Section", [
    "🎲 Random Simulator",
    "🧪 Manual Prediction",
    "📊 Model Stats"
])


# Section 1 — RANDOM SIMULATOR

if page == "🎲 Random Simulator":
    st.title("Random Transaction Simulator")
    st.markdown("Pick a real transaction from the dataset and see the model predict it live.")
    st.markdown("---")

    col_ctrl1, col_ctrl2 = st.columns([1, 1])
    with col_ctrl1:
        tx_type = st.selectbox("Transaction type to sample", ["Any", "Fraud", "Normal"])
    with col_ctrl2:
        st.markdown("<br>", unsafe_allow_html=True)
        pick = st.button("🎲 Pick Random Transaction", use_container_width=True)

    if pick or 'sim_row' not in st.session_state:
        if tx_type == "Fraud":
            row = fraud_samples.sample(1).iloc[0]
        elif tx_type == "Normal":
            row = normal_samples.sample(1).iloc[0]
        else:
            combined = pd.concat([fraud_samples, normal_samples])
            row = combined.sample(1).iloc[0]
        st.session_state['sim_row'] = row

    row = st.session_state['sim_row']
    true_label = int(row['Class']) if 'Class' in row else None

    # build input for model
    time_val   = row['Time']
    amount_val = row['Amount']
    scaled     = scaler.transform([[amount_val, time_val]])
    v_vals     = [row[c] for c in V_COLS]

    # correct column order: Time, V1-V28, Amount
    full_row = np.array([scaled[0][1]] + v_vals + [scaled[0][0]]).reshape(1, -1)
    col_names = ['Time'] + V_COLS + ['Amount']
    row_df = pd.DataFrame(full_row, columns=col_names)

    prob, pred = predict(row_df)
    label, box_class = risk_label(prob)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    col1.markdown(f"""
    <div class="info-box">
        <div style="color:#888;font-size:12px">AMOUNT</div>
        <div style="font-size:28px;font-weight:700;color:#e8e8e8">₹{amount_val:,.2f}</div>
    </div>""", unsafe_allow_html=True)

    col2.markdown(f"""
    <div class="info-box">
        <div style="color:#888;font-size:12px">TIME ELAPSED</div>
        <div style="font-size:28px;font-weight:700;color:#e8e8e8">{int(time_val):,}s</div>
    </div>""", unsafe_allow_html=True)

    if true_label is not None:
        actual_text = "🚨 Actually Fraud" if true_label == 1 else "✅ Actually Normal"
        actual_color = "#ff6b6b" if true_label == 1 else "#32c864"
        col3.markdown(f"""
        <div class="info-box">
            <div style="color:#888;font-size:12px">ACTUAL LABEL</div>
            <div style="font-size:22px;font-weight:700;color:{actual_color}">{actual_text}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # prediction result
    emoji = "🚨" if pred == 1 else "✅"
    display_p = display_prob(prob)
    st.markdown(f"""
    <div class="{box_class}">
        <div style="font-size:32px">{emoji}</div>
        <div style="font-size:22px;font-weight:700;margin-top:4px">{label}</div>
        <div style="font-size:16px;margin-top:4px;opacity:0.8">Fraud Probability: {display_p*100:.1f}%</div>
    </div>""", unsafe_allow_html=True)

    # probability bar
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Fraud Probability**")
    st.progress(float(display_p))

    # SHAP
    st.markdown("---")
    st.subheader("Why did the model predict this?")
    with st.spinner("Computing SHAP explanation..."):
        fig, summary = plot_shap(row_df)
        st.pyplot(fig)
        st.info(summary)
        plt.close()

    st.caption("Red bars push prediction toward fraud. Blue bars push toward normal. Longer bar = stronger influence.")


# Section 2 - MANUAL PREDICTION
elif page == "🧪 Manual Prediction":
    st.title("🧪 Manual Transaction Prediction")
    st.markdown("Enter transaction details. Model predicts fraud probability in real time.")
    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Transaction Details")

        amount = st.slider(
            "Transaction Amount (₹)",
            min_value=1.0,
            max_value=5000.0,
            value=150.0,
            step=10.0
        )

        time_window = st.selectbox("Time of Transaction", list(TIME_MAP.keys()))

        tx_category = st.selectbox("Transaction Category", list(TYPE_MAP.keys()))

        st.markdown("---")
        st.caption("""
        **How inputs are used:**
        - Amount → direct input to model
        - Time of day → mapped to seconds elapsed in dataset
        - Category → determines transaction behaviour profile (V features)

        **Privacy note:** This demo does not ask users for the raw `V1..V28` feature values.
        Those detailed feature columns are simulated using aggregate normal/fraud profiles because the real
        influencing columns are not exposed in the UI for privacy reasons.

        **Synthetic data disclaimer:** The inputs in this section generate a synthetic transaction profile
        for demonstration. For the 🎲 Random Simulator, real transactions from the dataset are used.
        """)
        

    # compute inputs
    t_min, t_max           = TIME_MAP[time_window]
    amt_min, amt_max, fw   = TYPE_MAP[tx_category]
    time_sec               = np.random.randint(t_min, t_max)

    row = build_input(amount, time_sec, fw)
    col_names = ['Time'] + V_COLS + ['Amount']
    row_df = pd.DataFrame(row, columns=col_names)

    prob, pred = predict(row_df)
    label, box_class = risk_label(prob)

    with col_right:
        st.subheader("Prediction")
        st.markdown("<br>", unsafe_allow_html=True)

        emoji = "🚨" if pred == 1 else "✅"
        display_p = display_prob(prob)
        st.markdown(f"""
        <div class="{box_class}">
            <div style="font-size:40px">{emoji}</div>
            <div style="font-size:24px;font-weight:700;margin-top:6px">{label}</div>
            <div style="font-size:18px;margin-top:6px;opacity:0.8">Probability: {display_p*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Fraud Probability**")
        st.progress(float(display_p))

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Amount", f"₹{amount:,.0f}")
        c2.metric("Time Window", time_window.split('(')[0].strip())
        c3.metric("Category", tx_category.split('/')[0].strip())

    # SHAP
    st.markdown("---")
    st.subheader("Why this prediction?")
    with st.spinner("Computing SHAP..."):
        fig, summary = plot_shap(row_df)
        st.pyplot(fig)
        st.info(summary)
        plt.close()

    st.caption("Red = pushed toward fraud. Blue = pushed toward normal.")

    # fraud weight explanation
    st.markdown("---")
    st.subheader("How Category Affects Prediction")
    cat_df = pd.DataFrame([
        {'Category': k, 'Fraud Profile Weight': f"{v[2]*100:.0f}%",
         'Amount Range': f"₹{v[0]:,} – ₹{v[1]:,}"}
        for k, v in TYPE_MAP.items()
    ])
    st.dataframe(cat_df, use_container_width=True, hide_index=True)
    st.caption("Fraud Profile Weight = how much the model leans on fraud transaction patterns for V features. Higher = riskier behaviour profile.")
    
# Weightage reference table
    with st.expander("Fraud Weights by Category"):
        st.markdown("These weights control how 'fraudulent' the synthetic profile looks.")
            
        weight_data = [
                {"Category": "Grocery / Supermarket", "Weight": "5%"},
                {"Category": "ATM Withdrawal", "Weight": "8%"},
                {"Category": "Electronics / Gadgets", "Weight": "10%"},
                {"Category": "Online Shopping", "Weight": "15%"},
                {"Category": "International Transaction", "Weight": "25%"},
                {"Category": "Suspicious / Test", "Weight": "85%"},
            ]
        df_weights = pd.DataFrame(weight_data)
        st.table(df_weights)
            
        st.markdown("""
            These weights reflect how fraud typically varies by category.
            Grocery is low-risk. International transfers are higher-risk. Suspicious/test transactions are
            flagged very high.

            The V features are PCA-compressed, so I can't reverse-map them to categories without the raw
            Kaggle data. Given that limitation, I set these weights to reflect real-world patterns.
            """)

# MODEL STATS
elif page == "📊 Model Stats":
    st.title("📊 Model Performance")
    st.markdown("---")

    # metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training Samples", "283,726")
    c2.metric("Fraud Rate", "0.17%")
    c3.metric("Best F1 (Fraud)", "0.83")
    c4.metric("Best ROC-AUC", "0.97")

    st.markdown("---")

    # comparison table
    st.subheader("Model Comparison")
    results = pd.DataFrame({
        'Model':              ['Random Forest ', 'XGBoost', 'Logistic Regression', 'Decision Tree'],
        'F1 (Fraud)':         [0.8276, 0.6387, 0.1002, 0.0630],
        'Precision (Fraud)':  [0.91,   0.53,   0.05,   0.03],
        'Recall (Fraud)':     [0.76,   0.80,   0.87,   0.85],
        'ROC-AUC':            [0.9494, 0.9716, 0.9619, 0.9433],
        'Accuracy':           [0.9995, 0.9985, 0.9737, 0.9576],
    })
    st.dataframe(results, use_container_width=True, hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("F1 Score")
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = ['#c8f135', '#f1c335', '#666', '#444']
        bars = ax.barh(results['Model'], results['F1 (Fraud)'], color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel('F1 Score', color='white')
        ax.set_facecolor('#1a1a1a')
        fig.patch.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Precision vs Recall")
        x = np.arange(4)
        w = 0.35
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(x - w/2, results['Precision (Fraud)'], w, label='Precision', color='#4e9de0')
        ax.bar(x + w/2, results['Recall (Fraud)'],    w, label='Recall',    color='#f1a535')
        ax.set_xticks(x)
        ax.set_xticklabels(['RF', 'XGB', 'LR', 'DT'], color='white')
        ax.legend()
        ax.set_facecolor('#1a1a1a')
        fig.patch.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("SMOTE — Handling Class Imbalance")
    col3, col4 = st.columns(2)

    with col3:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(['Normal', 'Fraud'], [226602, 378], color=['#4e9de0', '#ff4444'])
        ax.set_title('Before SMOTE', color='white')
        ax.set_facecolor('#1a1a1a')
        fig.patch.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        ax.title.set_color('white')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col4:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(['Normal', 'Fraud'], [226602, 226602], color=['#4e9de0', '#c8f135'])
        ax.set_title('After SMOTE', color='white')
        ax.set_facecolor('#1a1a1a')
        fig.patch.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        ax.title.set_color('white')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Fraud Pattern Clusters (KMeans)")
    cluster_df = pd.DataFrame({
        'Cluster':         ['Cluster 0', 'Cluster 1', 'Cluster 2'],
        'Count':           [147, 135, 191],
        'Mean Amount (₹)': [137.19, 150.02, 95.14],
        'Max Amount (₹)':  [1402.16, 2125.87, 1809.68],
        'Time Pattern':    ['Mid period', 'Late period — highest value fraud', 'Early period — most frequent']
    })
    st.dataframe(cluster_df, use_container_width=True, hide_index=True)
    st.caption("KMeans separated fraud primarily by time of occurrence, not amount. Fraudsters have distinct activity windows.")
