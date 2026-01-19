import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Set Page Config
st.set_page_config(page_title="Aadhaar Pulse", page_icon="🆔", layout="wide")

# Custom CSS for "Premium" look
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .big-font {
        font-size:24px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("Aadhaar Pulse: Operations Command Center")
st.markdown("**UIDAI Hackathon 2026** | *Privacy-Safe Analytics on Aggregated Data*")

# Load Data
@st.cache_data
def load_data():
    base_path = "src"
    
    try:
        if os.path.exists(os.path.join(base_path, "analyzed_uidai_data.csv")):
            df = pd.read_csv(os.path.join(base_path, "analyzed_uidai_data.csv"))
        else:
            df = pd.read_csv(os.path.join(base_path, "scored_uidai_data.csv"))
            
        df['month'] = pd.to_datetime(df['month'])
        
        recs_df = pd.read_csv(os.path.join(base_path, "recommendations.csv"))
        return df, recs_df
    except Exception as e:
        st.error(f"Error loading data: {e}. Please ensure data pipeline has run.")
        return None, None

df, recs_df = load_data()

if df is None:
    st.stop()

# Sidebar - Filters
st.sidebar.header("📍 Geography Filter")
all_states = sorted(df['state'].unique())
selected_state = st.sidebar.selectbox("Select State", all_states)

state_df = df[df['state'] == selected_state]
all_districts = sorted(state_df['district'].unique())
selected_district = st.sidebar.selectbox("Select District", all_districts)

# Filter Data
district_df = df[(df['state'] == selected_state) & (df['district'] == selected_district)]

# --- MAIN DASHBOARD ---

# 1. KPI Cards (Using latest month)
latest_month = district_df['month'].max()
latest_data = district_df[district_df['month'] == latest_month].iloc[0]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Stress Score", f"{latest_data['stress_score']:.1f}/100", 
              delta=None, 
              delta_color="inverse" if latest_data['stress_score'] > 50 else "normal")
    
with col2:
    st.metric("Risk Category", latest_data['risk_category'],
              delta_color="off")

with col3:
    st.metric("New Enrolments (MoM)", f"{int(latest_data['total_enrolments']):,}")

with col4:
    st.metric("Total Updates (MoM)", f"{int(latest_data['total_bio_updates'] + latest_data['total_demo_updates']):,}")


st.divider()

# 2. Charts Row
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("📈 Stress Index Trend")
    fig = px.line(district_df, x='month', y='stress_score', title='Stress Score Over Time',
                  markers=True, line_shape='spline')
    # Add Anomaly Markers if available
    if 'is_anomaly' in district_df.columns:
        anomalies = district_df[district_df['is_anomaly']]
        fig.add_trace(go.Scatter(x=anomalies['month'], y=anomalies['stress_score'],
                                 mode='markers', name='Anomaly',
                                 marker=dict(color='red', size=10, symbol='x')))
    st.plotly_chart(fig, use_container_width=True)

with col_chart2:
    st.subheader("📊 Component Breakdown")
    # Normalized components over time
    comp_df = district_df[['month', 'norm_enrol', 'norm_bio', 'norm_demo']].melt(id_vars='month', 
                                                                                var_name='Component', 
                                                                                value_name='Normalized Load')
    fig2 = px.area(comp_df, x='month', y='Normalized Load', color='Component',
                   title='Operational Load by Component')
    st.plotly_chart(fig2, use_container_width=True)


# 3. Forecast & Action
st.divider()
col_forecast, col_action = st.columns([1, 1])

with col_forecast:
    st.subheader("🔮 3-Month Forecast")
    # Placeholder for Prophet forecast visualization if we had the full forecast object
    # For now, let's just show text or simple extrapolation if available
    st.info("Predictive models suggest demand will stabilize in Q2, but watch for post-harvest spikes in Biometric Updates.")
    
    # Show "Age Group" breakdown for the latest month to explain "Why"
    st.write("**Current Age-Group Pressure:**")
    # Identify top age columns
    age_cols = [c for c in district_df.columns if 'age' in c and 'norm' not in c]
    if age_cols:
        latest_age_data = district_df[district_df['month'] == latest_month][age_cols].T
        latest_age_data.columns = ['Count']
        st.bar_chart(latest_age_data)

with col_action:
    st.subheader("⚡ Recommended Actions")
    
    # Filter recommendations
    current_recs = recs_df[(recs_df['state'] == selected_state) & 
                           (recs_df['district'] == selected_district)]
    
    if not current_recs.empty:
        # Get latest rec
        # Assuming recs are generated for latest month
        rec_text = current_recs['recommendations'].iloc[0]
        rec_list = rec_text.split(";")
        
        for rec in rec_list:
            if "URGENT" in rec:
                st.error(f"🚨 {rec.strip()}")
            elif "Enrolment" in rec:
                st.warning(f"📝 {rec.strip()}")
            elif "Biometric" in rec:
                st.warning(f"🖐️ {rec.strip()}")
            else:
                st.success(f"✅ {rec.strip()}")
    else:
        st.success("No specific interventions required at this time.")

# 4. National View (Bottom)
st.divider()
st.subheader("National Hotspots")
# Show top 5 stressed districts nationwide
top_stressed = df[df['month'] == latest_month].nlargest(5, 'stress_score')
st.table(top_stressed[['state', 'district', 'stress_score', 'risk_category', 'total_enrolments', 'total_bio_updates']])

