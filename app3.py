"""
AI-Driven Freelance 'Gig Worker' Financial Risk & Cash-Flow Predictor
Complete Streamlit Application - Enhanced Version (100% Free & Error-Free)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import re
import time
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Gig Worker Financial Risk Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved responsive design
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Session State Variables for Profile Settings Integration
if "profile_display_name" not in st.session_state:
    st.session_state["profile_display_name"] = "Freelancer Profile"
if "profile_email_address" not in st.session_state:
    st.session_state["profile_email_address"] = "freelancer@example.com"

# Email Validation Helper
def is_valid_email(email):
    if not email:
        return True
    return bool(re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", email))

class EnhancedGigWorkerRiskEngine:
    """Enhanced Risk Scoring Engine with additional metrics and ML-inspired features"""
    def __init__(self, df):
        self.df = df.copy()
        self.validate_and_prepare_data()
        self.monthly_metrics = {}
        self.process_data()
        
    def validate_and_prepare_data(self):
        required_columns = ['Date', 'Amount', 'Type']
        missing_cols = [col for col in required_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
        self.df = self.df.dropna(subset=['Date'])
        self.df['Amount'] = pd.to_numeric(self.df['Amount'], errors='coerce')
        self.df = self.df.dropna(subset=['Amount'])
        
        valid_types = ['Income', 'Expense']
        self.df = self.df[self.df['Type'].isin(valid_types)]
        
        if 'Description' not in self.df.columns:
            self.df['Description'] = 'N/A'
    
    def process_data(self):
        self.income_df = self.df[self.df['Type'] == 'Income'].copy()
        self.expense_df = self.df[self.df['Type'] == 'Expense'].copy()
        
        self.income_df['Month'] = self.income_df['Date'].dt.to_period('M')
        self.expense_df['Month'] = self.expense_df['Date'].dt.to_period('M')
        
        monthly_income = self.income_df.groupby('Month')['Amount'].agg(['sum', 'count', 'mean']).reset_index()
        monthly_expense = self.expense_df.groupby('Month')['Amount'].sum().reset_index()
        
        self.monthly_metrics = monthly_income.merge(
            monthly_expense, on='Month', how='outer', suffixes=('_income', '_expense')
        ).fillna(0)
        
        self.monthly_metrics.columns = ['Month', 'Total_Income', 'Gig_Count', 'Avg_Gig_Value', 'Total_Expenses']
        self.monthly_metrics['Net_Savings'] = self.monthly_metrics['Total_Income'] - self.monthly_metrics['Total_Expenses']
        
        self.total_income = self.income_df['Amount'].sum()
        self.total_expenses = self.expense_df['Amount'].sum()
        self.savings = self.total_income - self.total_expenses
        
        self.daily_income = self.income_df.groupby('Date')['Amount'].agg(['sum', 'count']).reset_index()
        self.daily_income.columns = ['Date', 'Daily_Income', 'Gig_Count']
        self.income_days = set(self.daily_income['Date'])
        
    def calculate_income_volatility(self):
        monthly_incomes = self.monthly_metrics['Total_Income'].values
        if len(monthly_incomes) < 2 or np.mean(monthly_incomes) == 0:
            return 0.0
        return min(stats.variation(monthly_incomes), 2.0)
    
    def calculate_income_trend(self):
        monthly_incomes = self.monthly_metrics['Total_Income'].values
        if len(monthly_incomes) < 2:
            return 0
        x = np.arange(len(monthly_incomes))
        slope, _, _, _, _ = stats.linregress(x, monthly_incomes)
        mean_income = np.mean(monthly_incomes)
        return (slope / mean_income) * 100 if mean_income > 0 else 0
    
    def calculate_dry_spell(self):
        if not self.income_days:
            return {'max_dry_spell': 0, 'avg_dry_spell': 0}
        dates = sorted(self.income_days)
        gaps = []
        for i in range(len(dates) - 1):
            gap = (dates[i + 1] - dates[i]).days
            if gap > 1:
                gaps.append(gap)
        return {
            'max_dry_spell': max(gaps) if gaps else 0,
            'avg_dry_spell': np.mean(gaps) if gaps else 0
        }
    
    def calculate_income_consistency(self):
        if len(self.daily_income) < 7:
            return 0.0
        date_range = (self.df['Date'].max() - self.df['Date'].min()).days + 1
        active_days_pct = (len(self.income_days) / date_range) * 100
        daily_gigs = self.daily_income['Gig_Count'].values
        gig_cv = stats.variation(daily_gigs) if len(daily_gigs) > 1 else 0
        return min((active_days_pct * 0.6) + ((1 - min(gig_cv, 1)) * 100 * 0.4), 100)
    
    def calculate_credit_score(self):
        volatility = self.calculate_income_volatility()
        dry_spell_metrics = self.calculate_dry_spell()
        max_dry_spell = dry_spell_metrics['max_dry_spell']
        cushion = self.savings / self.total_income if self.total_income > 0 else 0
        income_trend = self.calculate_income_trend()
        consistency = self.calculate_income_consistency()
        
        score = 300
        score += max(0, 200 - (volatility * 100))
        score += max(0, 150 - (min(max_dry_spell, 90) / 90 * 150))
        score += min(cushion * 200, 100)
        score += min(max(income_trend * 2, -75) + 75, 75)
        score += (consistency / 100) * 75
        return max(300, min(850, score))
    
    def calculate_risk_assessment(self):
        score = self.calculate_credit_score()
        if score >= 700:
            return {"risk_level": "Low", "approval_probability": "High (>80%)", "suggested_rate": "8-12%"}
        elif score >= 550:
            return {"risk_level": "Moderate", "approval_probability": "Medium (50-80%)", "suggested_rate": "12-18%"}
        return {"risk_level": "High", "approval_probability": "Low (<50%)", "suggested_rate": "18-24%+"}
    
    def get_all_metrics(self):
        dry_metrics = self.calculate_dry_spell()
        risk_metrics = self.calculate_risk_assessment()
        return {
            **risk_metrics,
            'credit_score': self.calculate_credit_score(),
            'income_volatility': self.calculate_income_volatility(),
            'income_trend': self.calculate_income_trend(),
            'income_consistency': self.calculate_income_consistency(),
            'max_dry_spell': dry_metrics['max_dry_spell'],
            'avg_dry_spell': dry_metrics['avg_dry_spell'],
            'savings_cushion': self.savings / self.total_income if self.total_income > 0 else 0,
            'total_income': self.total_income,
            'total_expenses': self.total_expenses,
            'savings': self.savings,
            'monthly_metrics': self.monthly_metrics
        }

class EnhancedGenAISimulator:
    """Smart Python rule-based local simulation that mimics an advanced GenAI LLM"""
    @staticmethod
    def generate_comprehensive_report(metrics):
        score = metrics['credit_score']
        risk_level = metrics['risk_level']
        name = st.session_state["profile_display_name"]
        
        insights = []
        if metrics['income_volatility'] > 0.5:
            insights.append("🔴 High income volatility detected - Suggest client diversification.")
        else:
            insights.append("A stable income execution flow verified.")
            
        if metrics['max_dry_spell'] > 14:
            insights.append(f"🔴 Extended dry spell gap of {metrics['max_dry_spell']} days verified.")
            
        return f"""
        Dear **{name}**, here is your GenAI automated underwriting profile:
        * **Financial Credit Assessment:** Score of {score:.0f}/850 ({risk_level} Risk Category)
        * **Platform Status:** Approval Chance is estimated at {metrics['approval_probability']} with target rates around {metrics['suggested_rate']}.
        * **Algorithmic Analytics Insights:** {", ".join(insights)}
        """

    @staticmethod
    def generate_smart_nudge(metrics):
        score = metrics['credit_score']
        name = st.session_state["profile_display_name"]
        
        if score >= 650:
            return f"📊 *Professional Direct Update for {name}*\n\nHello {name}! Your freelance matrix shows excellent health metrics with an interactive credit score of {score:.0f}. Premium lending options are ready for your review.\n\nReply ACCESS to unlock premium benefits."
        return f"💙 *Empathetic Nudge for {name}*\n\nHi {name}, we noticed some typical gig worker cash fluctuations this season. Don't stress—we have flexible safety cushions and tools built to support your journey!\n\nReply ASSIST for free planning tips."

def generate_enhanced_mock_data():
    """Generates the enhanced simulated timeline mock matrix data cleanly"""
    np.random.seed(42)
    start_date = datetime.now() - timedelta(days=180)
    dates = [start_date + timedelta(days=i) for i in range(180)]
    data = []
    for date in dates:
        day_of_week = date.weekday()
        prob = 0.4 if day_of_week in [1,2,3] else 0.1
        if np.random.random() < prob:
            data.append({'Date': date, 'Amount': round(np.random.uniform(200, 800), 2), 'Type': 'Income', 'Description': 'Project Payment'})
        if np.random.random() < 0.3:
            data.append({'Date': date, 'Amount': round(np.random.uniform(50, 300), 2), 'Type': 'Expense', 'Description': 'Work Cost'})
    return pd.DataFrame(data).sort_values('Date').reset_index(drop=True)

# Define Tabs Structure for Navigation Panel
tab1, tab2 = st.tabs(["📊 Financial Dashboard", "👤 Profile Settings"])

# ================= TAB 1: CORE ENGINE DASHBOARD =================
with tab1:
    st.markdown('<div class="main-header">📊 AI-Driven Gig Worker Financial Health Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced Risk Assessment • Cash Flow Prediction • Smart Financial Insights</div>', unsafe_allow_html=True)
    
    col_uploader, col_sample = st.columns([2, 1])
    with col_uploader:
        uploaded_file = st.file_uploader("Upload Bank Statement (CSV File Layout)", type=['csv'])
    with col_sample:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Generate & Use Enhanced Mock Data", use_container_width=True):
            st.session_state['df_enhanced'] = generate_enhanced_mock_data()
            st.success("✅ Sample system matrix initialised!")
            st.balloons()
            
    df = st.session_state.get('df_enhanced', None)
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            if all(col in uploaded_df.columns for col in ['Date', 'Amount', 'Type']):
                df = uploaded_df
                st.session_state['df_enhanced'] = df
                st.success("✅ CSV parsed without errors!")
            else:
                st.error("❌ CSV must contain columns: 'Date', 'Amount', and 'Type'")
        except Exception as e:
            st.error(f"❌ Error compiling input file: {str(e)}")

    if df is not None:
        try:
            engine = EnhancedGigWorkerRiskEngine(df)
            metrics = engine.get_all_metrics()
            ai_sim = EnhancedGenAISimulator()
            
            # Metric Card Wrappers
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("🎯 Credit Score", f"{metrics['credit_score']:.0f}/850", f"Risk: {metrics['risk_level']}")
            m_col2.metric("📊 Income Consistency Score", f"{metrics['income_consistency']:.1f}%", f"Volatility: {metrics['income_volatility']:.2f}")
            m_col3.metric("🌵 Max Dry Spell Gap", f"{metrics['max_dry_spell']} Days", "Longest gap")
            m_col4.metric("💰 Savings Cushion Buffer", f"{metrics['savings_cushion']:.1%}", "Reserve index")
            
            # Render Plotly Charts Layout
            st.markdown("### 📊 Interactive Monthly Cash Flow Layout Analysis")
            monthly_data = metrics['monthly_metrics']
            if not monthly_data.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Income Stream', x=monthly_data['Month'].astype(str), y=monthly_data['Total_Income'], marker_color='#667eea'))
                fig.add_trace(go.Bar(name='Expense Tracker', x=monthly_data['Month'].astype(str), y=monthly_data['Total_Expenses'], marker_color='#f5576c'))
                fig.update_layout(barmode='group', template='plotly_white', height=400)
                st.plotly_chart(fig, use_container_width=True)
                
            # GenAI Framework simulation tabs
            st.markdown("### 🤖 Offline Simulated GenAI Agent Platform")
            sub_tab1, sub_tab2 = st.tabs(["📝 Lender Narrative Storytelling", "💬 Psychology-Based Collection Nudge"])
            with sub_tab1:
                st.info(ai_sim.generate_comprehensive_report(metrics))
            with sub_tab2:
                st.code(ai_sim.generate_smart_nudge(metrics), language="text")
                
        except Exception as e:
            st.error(f"❌ Execution Failure in calculations engine loop: {str(e)}")

# ================= TAB 2: EXPLICIT USER PROFILE SETTINGS =================
with tab2:
    st.title("👤 Profile Configuration Setup Panel")
    st.write("Modify profile parameters below to sync real-time greeting insights inside your analytics tab seamlessly.")
    st.markdown("---")
    
    input_name = st.text_input("Display Username / Freelance Alias", value=st.session_state["profile_display_name"], max_chars=50)
    input_email = st.text_input("Registered Business Email Identity Address", value=st.session_state["profile_email_address"], max_chars=80)
    
    valid_status = is_valid_email(input_email)
    save_allowed = len(input_name.strip()) > 0 and valid_status
    
    if input_email and not valid_status:
        st.error("⚠️ The structural email string configuration is not valid.")
        
    if st.button("💾 Save Settings Matrix Profile", disabled=not save_allowed, type="primary"):
        st.session_state["profile_display_name"] = input_name.strip()
        st.session_state["profile_email_address"] = input_email.strip()
        st.success(f"✅ Success! Welcome **{input_name.strip()}**. Dashboard context variables updated.")
        time.sleep(0.4)
        st.rerun()
