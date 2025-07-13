import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="AI Lead Scoring Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean and Readable CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background-color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Remove default padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header Styles */
    .dashboard-header {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.15);
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: white;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 400;
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background-color: #1f2937;
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .sidebar-section {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .sidebar-section h3 {
        color: #374151;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 0.5rem;
    }
    
    /* Form Elements */
    .stSelectbox label, .stSlider label, .stNumberInput label, .stTextArea label {
        color: #374151 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }
    
    .stSelectbox > div > div {
        background-color: #f9fafb;
        border: 2px solid #e5e7eb;
        border-radius: 8px;
        color: #374151;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Score Display */
    .score-display {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.2);
    }
    
    .score-value {
        font-size: 4rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: white;
    }
    
    .score-label {
        font-size: 1.3rem;
        color: rgba(255, 255, 255, 0.95);
        font-weight: 500;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
    }
    
    .metric-title {
        color: #6b7280;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        color: #111827;
        font-size: 2rem;
        font-weight: 700;
        line-height: 1;
    }
    
    /* Status Badges */
    .status-high {
        background: #ef4444;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .status-medium {
        background: #f59e0b;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .status-low {
        background: #10b981;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4);
    }
    
    /* Progress Bar */
    .progress-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%);
        border-radius: 8px;
        height: 12px;
    }
    
    /* Chart Container */
    .chart-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
        border: 1px solid #e5e7eb;
    }
    
    .chart-title {
        color: #374151;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Section Headers */
    .section-header {
        color: #111827;
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        margin: 3rem 0 2rem 0;
        padding-bottom: 1rem;
        border-bottom: 3px solid #e5e7eb;
    }
    
    /* Alert Styles */
    .stAlert {
        border-radius: 8px;
        border: none;
    }
    
    /* Checkbox */
    .stCheckbox label {
        color: #374151 !important;
        font-weight: 500 !important;
    }
    
    /* Text Area */
    .stTextArea textarea {
        background-color: #f9fafb;
        border: 2px solid #e5e7eb;
        border-radius: 8px;
        color: #374151;
    }
    
    .stTextArea textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Number Input */
    .stNumberInput input {
        background-color: #f9fafb;
        border: 2px solid #e5e7eb;
        border-radius: 8px;
        color: #374151;
    }
    
    .stNumberInput input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Create a simple mock model for demonstration
def create_mock_model():
    """Create a mock model structure for demonstration"""
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    
    # Create mock encoders
    encoders = {
        'age_group': LabelEncoder(),
        'family_background': LabelEncoder(),
        'education_level': LabelEncoder(),
        'employment_type': LabelEncoder(),
        'scaler': StandardScaler()
    }
    
    # Fit encoders with sample data
    encoders['age_group'].fit(["18-25","26-35","36-45","46-55","56-65","65+"])
    encoders['family_background'].fit(["Single","Married","Divorced","Widowed","Other"])
    encoders['education_level'].fit(["High School","Bachelor's","Master's","PhD","Other"])
    encoders['employment_type'].fit(["Full-time","Part-time","Self-employed","Unemployed","Retired","Student"])
    
    # Create mock scaler
    import numpy as np
    mock_data = np.array([[650, 50000, 50], [700, 75000, 60], [600, 40000, 40]])
    encoders['scaler'].fit(mock_data)
    
    # Create mock model
    model = RandomForestClassifier(random_state=42)
    # Fit with dummy data
    X_dummy = np.random.rand(100, 7)
    y_dummy = np.random.randint(0, 2, 100)
    model.fit(X_dummy, y_dummy)
    
    feature_names = ['credit_score', 'age_group', 'family_background', 'income', 
                    'education_level', 'employment_type', 'website_engagement_score']
    
    return {
        'model': model,
        'encoders': encoders,
        'feature_names': feature_names
    }

# Load or create model package
@st.cache_resource
def load_package():
    try:
        return joblib.load('model.pkl')
    except FileNotFoundError:
        st.warning("⚠️ Model file not found. Using demo mode with mock predictions.")
        return create_mock_model()

pkg = load_package()
feature_names = pkg['feature_names']

# Header
st.markdown("""
<div class="dashboard-header">
    <div class="main-title">🎯 AI Lead Scoring Dashboard</div>
    <div class="subtitle">Predict conversion intent with ML-powered intelligent scoring</div>
</div>
""", unsafe_allow_html=True)

# Sidebar with Clean Design
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2 style="margin: 0; font-size: 1.5rem;">📋 Lead Information</h2>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Enter prospect details below</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form(key="lead_form", clear_on_submit=True):
        st.markdown('<div class="sidebar-section"><h3>💳 Financial Profile</h3>', unsafe_allow_html=True)
        credit_score = st.slider(
            "Credit Score", 
            300, 850, 650,
            help="Credit score range from 300 to 850"
        )
        income = st.number_input(
            "Annual Income ($)", 
            0, 1_000_000, 50_000, 
            step=1000,
            help="Annual income in USD"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section"><h3>👤 Demographics</h3>', unsafe_allow_html=True)
        age_group = st.selectbox(
            "Age Group", 
            ["18-25","26-35","36-45","46-55","56+"]
        )
        family_background = st.selectbox(
            "Family Background", 
            ["Single","Married","Divorced","Widowed","Other"]
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section"><h3>🎓 Professional Profile</h3>', unsafe_allow_html=True)
        education_level = st.selectbox(
            "Education Level", 
            ["High School","Bachelor's","Master's","PhD","Other"]
        )
        employment_type = st.selectbox(
            "Employment Type", 
            ["Full-time","Part-time","Self-employed","Unemployed","Retired","Student"]
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section"><h3>🌐 Digital Engagement</h3>', unsafe_allow_html=True)
        website_engagement_score = st.slider(
            "Website Engagement Score", 
            0, 100, 50,
            help="Engagement score based on website interactions"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section"><h3>📝 Additional Information</h3>', unsafe_allow_html=True)
        comments = st.text_area(
            "Notes / Comments",
            placeholder="Add any additional notes about this lead..."
        )
        
        consent = st.checkbox("✅ I consent to process this lead information")
        st.markdown('</div>', unsafe_allow_html=True)
        
        submit = st.form_submit_button(
            label="🔍 Calculate Lead Score",
            use_container_width=True
        )

# Main Content Area
if submit:
    if not consent:
        st.error("⚠️ Please provide consent to proceed with lead scoring.")
    else:
        # Prepare input data
        data = {
            'credit_score': credit_score,
            'age_group': age_group,
            'family_background': family_background,
            'income': income,
            'education_level': education_level,
            'employment_type': employment_type,
            'website_engagement_score': website_engagement_score
        }
        
        # Predict
        model = pkg['model']
        enc = pkg['encoders']
        X = pd.DataFrame([data], columns=feature_names)
        
        # Encode & scale
        for c in ["age_group","family_background","education_level","employment_type"]:
            X[c] = enc[c].transform(X[c].astype(str))
        X[["credit_score","income","website_engagement_score"]] = \
            enc['scaler'].transform(X[["credit_score","income","website_engagement_score"]])
        
        # Calculate score with some business logic for demo
        base_score = model.predict_proba(X)[0,1] * 100
        
        # Add some realistic scoring logic
        score_adjustments = 0
        if credit_score > 750:
            score_adjustments += 10
        elif credit_score < 600:
            score_adjustments -= 15
            
        if income > 75000:
            score_adjustments += 8
        elif income < 30000:
            score_adjustments -= 10
            
        if website_engagement_score > 70:
            score_adjustments += 12
        elif website_engagement_score < 30:
            score_adjustments -= 8
            
        score = max(0, min(100, base_score + score_adjustments))
        
        # Score Display
        st.markdown(f"""
        <div class="score-display">
            <div class="score-value">{score:.1f}%</div>
            <div class="score-label">Lead Conversion Probability</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics Display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">🎯 Lead Score</div>
                <div class="metric-value">{score:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if score >= 70:
                priority_class = "status-high"
                priority_text = "High Priority"
                priority_icon = "🔴"
            elif score >= 40:
                priority_class = "status-medium"
                priority_text = "Medium Priority"
                priority_icon = "🟡"
            else:
                priority_class = "status-low"
                priority_text = "Low Priority"
                priority_icon = "🟢"
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">📊 Priority Level</div>
                <div class="{priority_class}">{priority_icon} {priority_text}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">🌐 Engagement</div>
                <div class="metric-value">{website_engagement_score}/100</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            recommendation = "Follow up now!" if score >= 70 else "Schedule follow-up" if score >= 40 else "Nurture lead"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">💡 Recommendation</div>
                <div style="color: #374151; font-weight: 600; font-size: 1.1rem;">{recommendation}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Progress Bar
        st.markdown('<div class="progress-container">', unsafe_allow_html=True)
        st.markdown("#### 📈 Score Visualization")
        
        progress_col1, progress_col2 = st.columns([4, 1])
        with progress_col1:
            st.progress(int(score))
        with progress_col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 0.5rem;">
                <div style="font-size: 1.5rem; font-weight: 700; color: #3b82f6;">{score:.1f}%</div>
                <div style="color: #6b7280; font-size: 0.9rem;">Score</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Feedback Messages
        if score >= 70:
            st.balloons()
            st.success("🎉 Excellent! This is a high-quality lead with strong conversion potential.")
        elif score >= 40:
            st.info("📈 Good potential! Consider targeted nurturing strategies for this lead.")
        else:
            st.warning("📋 Lower conversion probability. Focus on qualification and nurturing.")
        
        # Save to history
        if 'history' not in st.session_state:
            st.session_state.history = []
        
        st.session_state.history.append({
            **data, 
            'score': score, 
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'priority': priority_text
        })

# History & Analytics
if 'history' in st.session_state and st.session_state.history:
    st.markdown('<div class="section-header">📊 Lead Analytics & History</div>', unsafe_allow_html=True)
    
    df = pd.DataFrame(st.session_state.history)
    
    # KPI Cards
    avg_score = df['score'].mean()
    high_pct = ((df['score'] >= 70).mean() * 100)
    total_leads = len(df)
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">👥 Total Leads</div>
            <div class="metric-value">{total_leads}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📊 Average Score</div>
            <div class="metric-value">{avg_score:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🎯 High Priority</div>
            <div class="metric-value">{high_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi_col4:
        latest_score = df['score'].iloc[-1]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📈 Latest Score</div>
            <div class="metric-value">{latest_score:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">📈 Score Distribution</div>', unsafe_allow_html=True)
        
        fig_hist = px.histogram(
            df, x='score', nbins=10,
            color_discrete_sequence=['#3b82f6'],
            title=""
        )
        fig_hist.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", color="#374151"),
            showlegend=False,
            xaxis_title="Lead Score (%)",
            yaxis_title="Number of Leads"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with chart_col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">🎯 Priority Breakdown</div>', unsafe_allow_html=True)
        
        priority_counts = df['priority'].value_counts()
        colors = {'High Priority': '#ef4444', 'Medium Priority': '#f59e0b', 'Low Priority': '#10b981'}
        fig_pie = px.pie(
            values=priority_counts.values,
            names=priority_counts.index,
            color=priority_counts.index,
            color_discrete_map=colors,
            title=""
        )
        fig_pie.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", color="#374151"),
            showlegend=True
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Data Table
    with st.expander("📋 View Detailed Lead History", expanded=False):
        st.markdown("#### Recent Leads")
        
        display_df = df.tail(10)[['timestamp', 'score', 'priority', 'income', 'credit_score', 'website_engagement_score']].copy()
        display_df.columns = ['Timestamp', 'Score (%)', 'Priority', 'Income ($)', 'Credit Score', 'Engagement']
        
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "Score (%)": st.column_config.ProgressColumn(
                    "Score (%)",
                    help="Lead conversion score",
                    min_value=0,
                    max_value=100,
                ),
                "Income ($)": st.column_config.NumberColumn(
                    "Income ($)",
                    help="Annual income",
                    format="$%d",
                ),
            }
        )
    
    # Clear History
    st.markdown("---")
    col_clear = st.columns([2, 1, 2])
    with col_clear[1]:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history.clear()
            st.success("✅ History cleared!")
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: white; border-radius: 12px; margin-top: 2rem;">
    <h3 style="color: #374151; margin-bottom: 0.5rem;">🎯 AI Lead Scoring Dashboard</h3>
    <p style="color: #6b7280; margin: 0;">Transform your sales process with intelligent lead scoring</p>
</div>
""", unsafe_allow_html=True)
