import streamlit as st
import requests
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and professional styling
st.markdown("""
<style>
    /* Dark theme overrides */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Metric cards styling */
    div[data-testid="metric-container"] {
        background-color: #1e2130;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    div[data-testid="metric-container"] label {
        color: #a0aec0 !important;
    }
    
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #a0aec0;
        border-radius: 8px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2d3748 !important;
        color: #fff !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1a1d29;
        border-right: 1px solid #2d3748;
    }
    
    section[data-testid="stSidebar"] .stMarkdown h1 {
        color: #63b3ed;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #3182ce;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #2c5282;
        transform: translateY(-2px);
    }
    
    /* Text area styling */
    .stTextArea textarea {
        background-color: #1e2130;
        border: 1px solid #2d3748;
        color: #e2e8f0;
        border-radius: 8px;
    }
    
    /* Success/Error boxes */
    .fraud-alert {
        background-color: #742a2a;
        border: 2px solid #fc8181;
        border-radius: 10px;
        padding: 20px;
        color: #feb2b2;
    }
    
    .normal-alert {
        background-color: #22543d;
        border: 2px solid #68d391;
        border-radius: 10px;
        padding: 20px;
        color: #9ae6b4;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #63b3ed;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #a0aec0;
        margin-bottom: 2rem;
    }
    
    /* GLOBAL TEXT VISIBILITY FIX */
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #e2e8f0 !important;
    }
    
    /* Tab panel content */
    .stTabs [data-baseweb="tab-panel"] {
        color: #e2e8f0;
    }
    
    /* Labels */
    .stTextInput label, .stTextArea label, .stSelectbox label {
        color: #a0aec0 !important;
    }
    
    /* Info boxes */
    .stAlert p {
        color: inherit !important;
    }
    
    /* Dividers */
    hr {
        border-color: #2d3748;
    }
    
    /* Column text */
    [data-testid="column"] {
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# API Base URL - uses environment variable on Render, localhost for local dev
def get_api_base():
    # Try Streamlit Cloud secrets first
    try:
        return st.secrets["API_BASE_URL"]
    except (KeyError, FileNotFoundError):
        pass
    # Then try environment variable (Render)
    return os.getenv("API_BASE_URL", "http://localhost:8000")

API_BASE = get_api_base()

# Hardcoded examples
FRAUD_EXAMPLE = [406.0, -2.3122265423263, 1.95199201064158, -1.60985073229769, 3.9979055875468,
                 -0.522187864667764, -1.42654531920595, -2.53738730624579, 1.39165724829804,
                 -2.77008927719433, -2.77227214465915, 3.20203320709635, -2.89990738849473,
                 -0.595221881324605, -4.28925378244217, 0.389724120274487, -1.14074717980657,
                 -2.83005567450437, -0.0168224681808257, 0.416955705037907, 0.126910559061474,
                 0.517232370861764, -0.0350493686052974, -0.465211076182388, 0.320198198514526,
                 0.0445191674731724, 0.177839798284401, 0.261145002567677, -0.143275874698919, 0.0]

NORMAL_EXAMPLE = [0.0, -1.3598071336738, -0.0727811733098497, 2.53634673796914, 1.37815522427443,
                  -0.338320769942518, 0.462387777762292, 0.239598554061257, 0.0986979012610507,
                  0.363786969611213, 0.0907941719789316, -0.551599533260813, -0.617800855762348,
                  -0.991389847235408, -0.311169353699879, 1.46817697209427, -0.470400525259478,
                  0.207971241929242, 0.0257905801985591, 0.403992960255733, 0.251412098239705,
                  -0.018306777944153, 0.277837575558899, -0.110473910188767, 0.0669280749146731,
                  0.128539358273528, -0.189114843888824, 0.133558376740387, -0.0210530534538215, 149.62]


def fetch_dashboard():
    """Fetch dashboard statistics from API."""
    try:
        response = requests.get(f"{API_BASE}/dashboard", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch dashboard: {e}")
        return None


def fetch_history():
    """Fetch transaction history from API."""
    try:
        response = requests.get(f"{API_BASE}/history", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch history: {e}")
        return None


def predict_transaction(features: list):
    """Send prediction request to API."""
    try:
        response = requests.post(
            f"{API_BASE}/predict-json",
            json={"features": features},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None


# Sidebar
with st.sidebar:
    st.markdown("# 🛡️ Fraud Detection")
    st.markdown("### Dashboard")
    st.divider()
    
    # Current date/time
    st.markdown("**Current Time**")
    st.markdown(f"📅 {datetime.now().strftime('%B %d, %Y')}")
    st.markdown(f"🕐 {datetime.now().strftime('%H:%M:%S')}")
    st.divider()
    
    # Connection status
    st.markdown("**Connection Status**")
    st.markdown(f"🔗 Connected to: `localhost:8000`")
    
    # Quick stats in sidebar
    st.divider()
    dashboard_data = fetch_dashboard()
    if dashboard_data:
        st.markdown("**Quick Stats**")
        st.metric("Total Transactions", f"{dashboard_data['total_transactions']:,}")
        st.metric("Fraud Rate", f"{dashboard_data['fraud_rate_percent']}%")


# Main content
st.markdown('<p class="main-header">🛡️ Fraud Detection Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time monitoring and analysis of transaction fraud detection</p>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab5, tab4, tab6 = st.tabs(["📊 Live Monitor", "🔍 SHAP Explorer", "📈 Model Confidence", "🔬 Drift Monitor", "💰 Business Impact", "🔄 Auto Retrain"])

# TAB 1 - Live Monitor
with tab1:
    st.markdown("### Real-Time Monitoring")
    st.markdown("Auto-refreshes every 10 seconds")
    
    # Fetch data
    dashboard_data = fetch_dashboard()
    history_data = fetch_history()
    
    if dashboard_data:
        # Metric cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Total Transactions",
                value=f"{dashboard_data['total_transactions']:,}",
                delta="Live"
            )
        
        with col2:
            st.metric(
                label="Fraud Detected",
                value=f"{dashboard_data['fraud_transactions']:,}",
                delta=f"{dashboard_data['fraud_rate_percent']}% rate"
            )
        
        with col3:
            st.metric(
                label="Fraud Rate",
                value=f"{dashboard_data['fraud_rate_percent']}%",
                delta="Current"
            )
    
    st.divider()
    
    if history_data:
        df = pd.DataFrame(history_data)
        
        # Charts side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Fraud Probability Over Transactions")
            fig_line = px.line(
                df,
                x='id',
                y='fraud_probability',
                title='Fraud Probability Trend',
                labels={'id': 'Transaction ID', 'fraud_probability': 'Fraud Probability'},
                template='plotly_dark'
            )
            fig_line.update_traces(line_color='#63b3ed', line_width=2)
            fig_line.update_layout(
                plot_bgcolor='rgba(30,33,48,1)',
                paper_bgcolor='rgba(30,33,48,1)',
                font_color='#a0aec0',
                title_font_color='#e2e8f0'
            )
            st.plotly_chart(fig_line, use_container_width=True)
        
        with col2:
            st.markdown("#### Fraud vs Normal Distribution")
            fraud_count = df['is_fraud'].sum()
            normal_count = len(df) - fraud_count
            
            fig_pie = px.pie(
                values=[normal_count, fraud_count],
                names=['Normal', 'Fraud'],
                title='Transaction Classification',
                color_discrete_sequence=['#48bb78', '#fc8181'],
                template='plotly_dark'
            )
            fig_pie.update_layout(
                plot_bgcolor='rgba(30,33,48,1)',
                paper_bgcolor='rgba(30,33,48,1)',
                font_color='#a0aec0',
                title_font_color='#e2e8f0'
            )
            st.plotly_chart(fig_pie, use_container_width=True)

# TAB 2 - SHAP Explorer
with tab2:
    st.markdown("### SHAP Feature Explainer")
    st.markdown("Understand which features contribute most to fraud detection")
    
    # Example buttons - side by side
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔴 Load Fraud Example", key="fraud_btn", use_container_width=True):
            st.session_state['features_input'] = ','.join(map(str, FRAUD_EXAMPLE))
            st.rerun()
    
    with col2:
        if st.button("🟢 Load Normal Example", key="normal_btn", use_container_width=True):
            st.session_state['features_input'] = ','.join(map(str, NORMAL_EXAMPLE))
            st.rerun()
    
    # Feature input
    default_input = st.session_state.get('features_input', '')
    features_text = st.text_area(
        "Enter 30 comma-separated feature values (Time, V1-V28, Amount):",
        value=default_input,
        height=100,
        placeholder="e.g., 0.0, -1.35, 0.07, 2.53, ..."
    )
    
    # Prominent Analyze button
    if st.button("🔍 Analyze Transaction", key="analyze_btn", type="primary", use_container_width=True):
        if features_text:
            try:
                features = [float(x.strip()) for x in features_text.split(',')]
                if len(features) != 30:
                    st.error(f"Expected 30 features, got {len(features)}")
                else:
                    with st.spinner("Analyzing transaction..."):
                        result = predict_transaction(features)
                    
                    if result:
                        st.divider()
                        
                        # Prediction result with native Streamlit components
                        if result['is_fraud']:
                            st.error("🚨 **FRAUD DETECTED** - This transaction shows high fraud indicators")
                        else:
                            st.success("✅ **NORMAL TRANSACTION** - This transaction appears legitimate")
                        
                        # Metrics row
                        mcol1, mcol2 = st.columns(2)
                        with mcol1:
                            st.metric("Fraud Probability", f"{result['fraud_probability']:.4f}")
                        with mcol2:
                            st.metric("Fraud Score", f"{result['fraud_score']}/100")
                        
                        # SHAP horizontal bar chart
                        st.markdown("#### Top Contributing Features")
                        
                        top_features = result.get('top_features', [])
                        if top_features:
                            features_df = pd.DataFrame(top_features)
                            features_df = features_df.sort_values('shap_value', key=abs)
                            
                            colors = ['#fc8181' if v > 0 else '#63b3ed' for v in features_df['shap_value']]
                            
                            fig_bar = go.Figure(go.Bar(
                                x=features_df['shap_value'],
                                y=features_df['feature'],
                                orientation='h',
                                marker_color=colors
                            ))
                            
                            fig_bar.update_layout(
                                title='SHAP Values (Red = Fraud Risk, Blue = Safe)',
                                xaxis_title='SHAP Value',
                                yaxis_title='Feature',
                                template='plotly_dark',
                                plot_bgcolor='rgba(30,33,48,1)',
                                paper_bgcolor='rgba(30,33,48,1)',
                                font_color='#a0aec0',
                                title_font_color='#e2e8f0',
                                height=300
                            )
                            
                            st.plotly_chart(fig_bar, use_container_width=True)
                            st.markdown("🔴 **Red bars** = Increases fraud probability | 🔵 **Blue bars** = Decreases fraud probability")
                        else:
                            st.info("SHAP data not available - showing prediction result only")
                        
                        # Dashboard link button
                        st.divider()
                        st.link_button(
                            "📊 View API Dashboard",
                            f"{API_BASE}/",
                            use_container_width=True
                        )
                    else:
                        st.warning("API returned no data. Check if the server is running.")
                        
            except ValueError as e:
                st.error(f"Invalid input: {e}. Please enter valid numbers separated by commas.")
        else:
            st.warning("Please enter feature values or load an example")

# TAB 3 - Model Confidence
with tab3:
    st.markdown("### Model Confidence Analysis")
    st.markdown("Distribution of fraud probability scores across all predictions")
    
    history_data = fetch_history()
    
    if history_data:
        df = pd.DataFrame(history_data)
        
        # Histogram
        fig_hist = px.histogram(
            df,
            x='fraud_probability',
            nbins=20,
            title='Distribution of Fraud Probability Scores',
            labels={'fraud_probability': 'Fraud Probability', 'count': 'Number of Transactions'},
            template='plotly_dark',
            color_discrete_sequence=['#63b3ed']
        )
        
        fig_hist.update_layout(
            plot_bgcolor='rgba(30,33,48,1)',
            paper_bgcolor='rgba(30,33,48,1)',
            font_color='#a0aec0',
            title_font_color='#e2e8f0'
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.divider()
        
        # Risk categories
        st.markdown("#### Risk Category Breakdown")
        
        total = len(df)
        low_risk = len(df[df['fraud_probability'] <= 0.3])
        medium_risk = len(df[(df['fraud_probability'] > 0.3) & (df['fraud_probability'] <= 0.7)])
        high_risk = len(df[df['fraud_probability'] > 0.7])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            low_pct = (low_risk / total * 100) if total > 0 else 0
            st.markdown("""
            <div style="background-color: #22543d; border-radius: 10px; padding: 20px; text-align: center;">
                <h3 style="color: #68d391; margin: 0;">🟢 Low Risk</h3>
                <p style="color: #9ae6b4; font-size: 2rem; margin: 10px 0;">{:.1f}%</p>
                <p style="color: #a0aec0;">(0 - 0.3 probability)</p>
                <p style="color: #68d391;">{:,} transactions</p>
            </div>
            """.format(low_pct, low_risk), unsafe_allow_html=True)
        
        with col2:
            med_pct = (medium_risk / total * 100) if total > 0 else 0
            st.markdown("""
            <div style="background-color: #744210; border-radius: 10px; padding: 20px; text-align: center;">
                <h3 style="color: #f6ad55; margin: 0;">🟡 Medium Risk</h3>
                <p style="color: #fbd38d; font-size: 2rem; margin: 10px 0;">{:.1f}%</p>
                <p style="color: #a0aec0;">(0.3 - 0.7 probability)</p>
                <p style="color: #f6ad55;">{:,} transactions</p>
            </div>
            """.format(med_pct, medium_risk), unsafe_allow_html=True)
        
        with col3:
            high_pct = (high_risk / total * 100) if total > 0 else 0
            st.markdown("""
            <div style="background-color: #742a2a; border-radius: 10px; padding: 20px; text-align: center;">
                <h3 style="color: #fc8181; margin: 0;">🔴 High Risk</h3>
                <p style="color: #feb2b2; font-size: 2rem; margin: 10px 0;">{:.1f}%</p>
                <p style="color: #a0aec0;">(0.7 - 1.0 probability)</p>
                <p style="color: #fc8181;">{:,} transactions</p>
            </div>
            """.format(high_pct, high_risk), unsafe_allow_html=True)

# TAB 5 - Drift Monitor
with tab5:
    st.markdown("### Data Drift Detection")
    st.markdown("Monitor for distribution changes between training and production data")
    
    # Run Drift Check button
    if st.button("🔬 Run Drift Check", type="primary", use_container_width=True, key="drift_btn"):
        with st.spinner("Analyzing data distribution..."):
            try:
                drift_response = requests.get(f"{API_BASE}/drift", timeout=30)
                drift_response.raise_for_status()
                drift_result = drift_response.json()
                st.session_state['drift_result'] = drift_result
            except Exception as e:
                st.error(f"Drift check failed: {e}")
                st.session_state['drift_result'] = None
    
    # Display results if available
    if 'drift_result' in st.session_state and st.session_state['drift_result']:
        drift_result = st.session_state['drift_result']
        
        st.divider()
        
        # Alert/Success message
        if drift_result.get('error'):
            st.warning(f"⚠️ {drift_result['error']}")
        elif drift_result.get('alert', False):
            st.error("⚠️ DATA DRIFT DETECTED — Consider retraining the model")
        else:
            st.success("✅ Data distribution is stable — No action needed")
        
        # Metric cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            drift_pct = drift_result.get('drift_score', 0) * 100
            st.metric("Drift Score", f"{drift_pct:.1f}%")
        
        with col2:
            n_drifted = drift_result.get('n_drifted_features', 0)
            st.metric("Drifted Features", f"{n_drifted} / 30")
        
        with col3:
            status = "🔴 ALERT" if drift_result.get('alert', False) else "🟢 STABLE"
            st.metric("Status", status)
        
        # Drifted features list
        drifted_features = drift_result.get('drifted_features', [])
        if drifted_features:
            st.warning(f"Drifted Features: {', '.join(drifted_features)}")
    
    st.divider()
    
    # Drift history chart
    st.markdown("#### Drift Score Over Time")
    
    try:
        history_response = requests.get(f"{API_BASE}/drift/history", timeout=10)
        history_response.raise_for_status()
        drift_history = history_response.json()
    except Exception:
        drift_history = []
    
    if drift_history:
        hist_df = pd.DataFrame(drift_history)
        hist_df['drift_pct'] = hist_df['drift_score'] * 100
        
        fig_drift = go.Figure()
        
        # Drift score line
        fig_drift.add_trace(go.Scatter(
            x=hist_df['timestamp'],
            y=hist_df['drift_score'],
            mode='lines+markers',
            name='Drift Score',
            line=dict(color='#63b3ed', width=2),
            marker=dict(size=8)
        ))
        
        # Alert threshold line
        fig_drift.add_hline(
            y=0.3, 
            line_dash="dash", 
            line_color="red",
            annotation_text="Alert Threshold (30%)",
            annotation_position="top right"
        )
        
        fig_drift.update_layout(
            title='Drift Score Over Time',
            xaxis_title='Timestamp',
            yaxis_title='Drift Score',
            yaxis=dict(range=[0, 1]),
            template='plotly_dark',
            plot_bgcolor='rgba(30,33,48,1)',
            paper_bgcolor='rgba(30,33,48,1)',
            font_color='#a0aec0',
            title_font_color='#e2e8f0',
            height=400
        )
        
        st.plotly_chart(fig_drift, use_container_width=True)
    else:
        st.info("No drift history yet. Click 'Run Drift Check' to start monitoring.")

# TAB 4 - Business Impact
with tab4:
    st.markdown("### Business Impact Analysis")
    st.markdown("Financial metrics and ROI of the fraud detection model")
    
    # Business assumptions
    AVG_FRAUD_AMOUNT = 847
    FALSE_POSITIVE_COST = 15
    
    # Fetch data with error handling
    try:
        history_response = requests.get(f"{API_BASE}/history", timeout=5)
        history_response.raise_for_status()
        history_data = history_response.json()
    except Exception:
        st.warning("API unavailable - showing cached/empty data")
        history_data = []
    
    # Handle empty data case
    if not history_data:
        st.warning("No transactions yet. Use the SHAP Explorer tab to run some predictions first.")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Fraud Transactions Caught", "0")
            st.metric("Estimated Fraud Prevented", "$0")
        with col2:
            st.metric("False Positive Cost", "$0")
            st.metric("Net Savings", "$0")
        st.info(f"Assumptions: Avg fraud transaction = ${AVG_FRAUD_AMOUNT:,} | False positive review cost = ${FALSE_POSITIVE_COST} | Based on 0 total transactions analyzed")
    else:
        df = pd.DataFrame(history_data)
        total_transactions = len(df)
        
        # Calculate metrics
        fraud_caught = int(df['is_fraud'].sum())
        false_positives = len(df[(df['is_fraud'] == False) & (df['fraud_score'] > 30)])
        
        fraud_prevented = fraud_caught * AVG_FRAUD_AMOUNT
        fp_cost = false_positives * FALSE_POSITIVE_COST
        net_savings = fraud_prevented - fp_cost
        
        # 2x2 Metric cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Fraud Transactions Caught", f"{fraud_caught:,}")
            st.metric("Estimated Fraud Prevented", f"${fraud_prevented:,}")
        
        with col2:
            st.metric("False Positive Cost", f"${fp_cost:,}")
            st.metric("Net Savings", f"${net_savings:,}")
        
        # Assumptions info box
        st.info(f"Assumptions: Avg fraud transaction = ${AVG_FRAUD_AMOUNT:,} | False positive review cost = ${FALSE_POSITIVE_COST} | Based on {total_transactions} total transactions analyzed")
        
        st.divider()
        
        # Last 10 transactions table
        st.markdown("#### Recent Transactions")
        
        recent_df = df.head(10).copy()
        recent_df['Status'] = recent_df['is_fraud'].apply(lambda x: '🔴 FRAUD' if x else '🟢 NORMAL')
        recent_df['Risk Score'] = recent_df['fraud_score']
        recent_df['Transaction ID'] = recent_df['id']
        recent_df['Time'] = recent_df['created_at']
        
        display_df = recent_df[['Transaction ID', 'Risk Score', 'Status', 'Time']]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

# TAB 6 - Auto Retrain
with tab6:
    st.markdown("### Automated Model Retraining")
    st.markdown("Monitor model health and trigger retraining when needed")
    
    # Section 1: Current Model Health
    st.markdown("#### Current Model Health")
    
    try:
        metrics_response = requests.get(f"{API_BASE}/retrain/current-metrics", timeout=30)
        metrics_response.raise_for_status()
        current_metrics = metrics_response.json()
    except Exception as e:
        st.warning(f"Could not fetch current metrics: {e}")
        current_metrics = {}
    
    if current_metrics and "error" not in current_metrics:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        def get_metric_color(value):
            if value > 0.95:
                return "🟢"
            elif value >= 0.90:
                return "🟡"
            else:
                return "🔴"
        
        with col1:
            acc = current_metrics.get('accuracy', 0)
            st.metric(f"{get_metric_color(acc)} Accuracy", f"{acc:.2%}")
        with col2:
            prec = current_metrics.get('precision', 0)
            st.metric(f"{get_metric_color(prec)} Precision", f"{prec:.2%}")
        with col3:
            rec = current_metrics.get('recall', 0)
            st.metric(f"{get_metric_color(rec)} Recall", f"{rec:.2%}")
        with col4:
            f1 = current_metrics.get('f1', 0)
            st.metric(f"{get_metric_color(f1)} F1 Score", f"{f1:.2%}")
        with col5:
            roc = current_metrics.get('roc_auc', 0)
            st.metric(f"{get_metric_color(roc)} ROC-AUC", f"{roc:.2%}")
    else:
        st.warning("Could not load current model metrics")
    
    st.divider()
    
    # Section 2: Trigger Controls
    st.markdown("#### Trigger Retraining")
    st.info("Retraining compares new model vs current. Only promotes if new model wins on F1 score.")
    
    if st.button("🚀 Trigger Retraining", type="primary", use_container_width=True, key="retrain_btn"):
        with st.spinner("Training new model... this takes 2-5 minutes"):
            try:
                retrain_response = requests.post(f"{API_BASE}/retrain", timeout=600)
                retrain_response.raise_for_status()
                retrain_result = retrain_response.json()
                st.session_state['retrain_result'] = retrain_result
            except Exception as e:
                st.error(f"Retraining failed: {e}")
                st.session_state['retrain_result'] = None
    
    # Display retraining result
    if 'retrain_result' in st.session_state and st.session_state['retrain_result']:
        result = st.session_state['retrain_result']
        status = result.get('status', 'unknown')
        
        st.divider()
        
        if status == "promoted":
            st.success("✅ New model promoted! F1 score improved.")
        elif status == "rejected":
            st.warning("⚠️ New model rejected — current model performs better.")
        elif status == "skipped":
            st.info(f"ℹ️ Retraining skipped — {result.get('reason', 'model is healthy')}")
        else:
            st.error(f"❌ Error: {result.get('reason', 'Unknown error')}")
        
        # Comparison table if new_metrics exists
        new_metrics = result.get('new_metrics')
        curr_metrics = result.get('current_metrics')
        
        if new_metrics and curr_metrics:
            st.markdown("##### Model Comparison")
            
            comparison_data = []
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                curr_val = curr_metrics.get(metric, 0)
                new_val = new_metrics.get(metric, 0)
                winner = "🏆 New" if new_val > curr_val else ("🏆 Current" if curr_val > new_val else "Tie")
                comparison_data.append({
                    "Metric": metric.upper().replace('_', '-'),
                    "Current": f"{curr_val:.4f}",
                    "New Model": f"{new_val:.4f}",
                    "Winner": winner
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Training duration
        duration = result.get('training_duration_seconds', 0)
        st.caption(f"Training completed in {duration:.1f} seconds")
    
    st.divider()
    
    # Section 3: Retraining History
    st.markdown("#### Retraining History")
    
    try:
        history_response = requests.get(f"{API_BASE}/retrain/status", timeout=10)
        history_response.raise_for_status()
        retrain_history = history_response.json()
    except Exception:
        retrain_history = []
    
    if retrain_history:
        history_data = []
        for item in retrain_history[-5:][::-1]:  # Last 5, newest first
            status = item.get('status', 'unknown')
            status_icon = {
                'promoted': '🟢 promoted',
                'rejected': '🟡 rejected',
                'skipped': '⚪ skipped',
                'error': '🔴 error'
            }.get(status, status)
            
            curr_f1 = item.get('current_metrics', {}).get('f1', 0) if item.get('current_metrics') else 0
            new_f1 = item.get('new_metrics', {}).get('f1', 0) if item.get('new_metrics') else '-'
            
            history_data.append({
                "Timestamp": item.get('timestamp', '')[:19],
                "Status": status_icon,
                "Drift Score": f"{item.get('drift_score', 0):.1%}",
                "F1 (Current)": f"{curr_f1:.4f}" if curr_f1 else "-",
                "F1 (New)": f"{new_f1:.4f}" if isinstance(new_f1, float) else new_f1,
                "Promoted": "✅" if item.get('promoted') else "❌"
            })
        
        history_df = pd.DataFrame(history_data)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    else:
        st.info("No retraining history yet. Click 'Trigger Retraining' to start.")
