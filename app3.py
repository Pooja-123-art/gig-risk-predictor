"""
AI-Driven Freelance 'Gig Worker' Financial Risk & Cash-Flow Predictor
Complete Streamlit Application - Enhanced Version
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        color: white;
    }
    .risk-low {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1rem;
        border-radius: 10px;
    }
    .risk-moderate {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        padding: 1rem;
        border-radius: 10px;
    }
    .risk-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

class EnhancedGigWorkerRiskEngine:
    """
    Enhanced Risk Scoring Engine with additional metrics and ML-inspired features
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.validate_and_prepare_data()
        self.monthly_metrics = {}
        self.process_data()
        
    def validate_and_prepare_data(self):
        """Validate and prepare the uploaded data"""
        required_columns = ['Date', 'Amount', 'Type']
        
        # Check required columns
        missing_cols = [col for col in required_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert date and ensure proper types
        self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
        self.df = self.df.dropna(subset=['Date'])
        self.df['Amount'] = pd.to_numeric(self.df['Amount'], errors='coerce')
        self.df = self.df.dropna(subset=['Amount'])
        
        # Validate Type values
        valid_types = ['Income', 'Expense']
        invalid_types = self.df[~self.df['Type'].isin(valid_types)]
        if not invalid_types.empty:
            st.warning(f"⚠️ Found {len(invalid_types)} rows with invalid Type. These will be excluded.")
            self.df = self.df[self.df['Type'].isin(valid_types)]
        
        if 'Description' not in self.df.columns:
            self.df['Description'] = 'N/A'
    
    def process_data(self):
        """Enhanced data processing with trend analysis"""
        # Separate income and expenses
        self.income_df = self.df[self.df['Type'] == 'Income'].copy()
        self.expense_df = self.df[self.df['Type'] == 'Expense'].copy()
        
        # Monthly aggregation
        self.income_df['Month'] = self.income_df['Date'].dt.to_period('M')
        self.expense_df['Month'] = self.expense_df['Date'].dt.to_period('M')
        
        monthly_income = self.income_df.groupby('Month')['Amount'].agg(['sum', 'count', 'mean']).reset_index()
        monthly_expense = self.expense_df.groupby('Month')['Amount'].sum().reset_index()
        
        # Merge monthly data
        self.monthly_metrics = monthly_income.merge(
            monthly_expense, 
            on='Month', 
            how='outer', 
            suffixes=('_income', '_expense')
        ).fillna(0)
        
        self.monthly_metrics.columns = ['Month', 'Total_Income', 'Gig_Count', 'Avg_Gig_Value', 'Total_Expenses']
        self.monthly_metrics['Net_Savings'] = self.monthly_metrics['Total_Income'] - self.monthly_metrics['Total_Expenses']
        self.monthly_metrics['Savings_Rate'] = np.where(
            self.monthly_metrics['Total_Income'] > 0,
            self.monthly_metrics['Net_Savings'] / self.monthly_metrics['Total_Income'],
            0
        )
        
        # Overall metrics
        self.total_income = self.income_df['Amount'].sum()
        self.total_expenses = self.expense_df['Amount'].sum()
        self.savings = self.total_income - self.total_expenses
        
        # Daily income patterns
        self.daily_income = self.income_df.groupby('Date')['Amount'].agg(['sum', 'count']).reset_index()
        self.daily_income.columns = ['Date', 'Daily_Income', 'Gig_Count']
        
        # Income days for dry spell calculation
        self.income_days = set(self.daily_income['Date'])
        
    def calculate_income_volatility(self):
        """Enhanced volatility calculation with coefficient of variation"""
        monthly_incomes = self.monthly_metrics['Total_Income'].values
        
        if len(monthly_incomes) < 2 or np.mean(monthly_incomes) == 0:
            return 0.0
        
        cv = stats.variation(monthly_incomes)  # Coefficient of variation
        return min(cv, 2.0)
    
    def calculate_income_trend(self):
        """Calculate income trend using linear regression"""
        monthly_incomes = self.monthly_metrics['Total_Income'].values
        if len(monthly_incomes) < 2:
            return 0
        
        x = np.arange(len(monthly_incomes))
        slope, _, _, _, _ = stats.linregress(x, monthly_incomes)
        
        # Normalize slope relative to mean income
        mean_income = np.mean(monthly_incomes)
        if mean_income > 0:
            return (slope / mean_income) * 100  # Percentage change per month
        return 0
    
    def calculate_dry_spell(self):
        """Calculate maximum and average dry spells"""
        if not self.income_days:
            return {'max_dry_spell': 0, 'avg_dry_spell': 0, 'dry_spell_count': 0}
        
        dates = sorted(self.income_days)
        gaps = []
        
        for i in range(len(dates) - 1):
            gap = (dates[i + 1] - dates[i]).days
            if gap > 1:  # Consecutive days without income
                gaps.append(gap)
        
        return {
            'max_dry_spell': max(gaps) if gaps else 0,
            'avg_dry_spell': np.mean(gaps) if gaps else 0,
            'dry_spell_count': len(gaps)
        }
    
    def calculate_income_consistency(self):
        """Calculate income consistency score (0-100)"""
        if len(self.daily_income) < 7:  # Need at least a week of data
            return 0
        
        # Calculate percentage of days with income
        date_range = (self.df['Date'].max() - self.df['Date'].min()).days + 1
        active_days_pct = (len(self.income_days) / date_range) * 100
        
        # Calculate gig frequency consistency
        daily_gigs = self.daily_income['Gig_Count'].values
        gig_cv = stats.variation(daily_gigs) if len(daily_gigs) > 1 else 0
        
        # Combine metrics (higher is better)
        consistency = (active_days_pct * 0.6) + ((1 - min(gig_cv, 1)) * 100 * 0.4)
        return min(consistency, 100)
    
    def calculate_credit_score(self):
        """
        Enhanced credit score calculation (300-850)
        Now includes income trend and consistency
        """
        # Get base metrics
        volatility = self.calculate_income_volatility()
        dry_spell_metrics = self.calculate_dry_spell()
        max_dry_spell = dry_spell_metrics['max_dry_spell']
        cushion = self.savings / self.total_income if self.total_income > 0 else 0
        income_trend = self.calculate_income_trend()
        consistency = self.calculate_income_consistency()
        
        # Start with base score
        score = 300
        
        # 1. Income Volatility (0-200 points)
        volatility_score = max(0, 200 - (volatility * 100))
        score += volatility_score
        
        # 2. Dry Spell Impact (0-150 points)
        dry_spell_score = max(0, 150 - (min(max_dry_spell, 90) / 90 * 150))
        score += dry_spell_score
        
        # 3. Savings Cushion (0-100 points)
        cushion_score = min(cushion * 200, 100)
        score += cushion_score
        
        # 4. Income Trend (0-75 points)
        trend_score = min(max(income_trend * 2, -75) + 75, 75)  # Range: -75 to 75, normalized to 0-75
        score += trend_score
        
        # 5. Consistency Bonus (0-75 points)
        consistency_score = (consistency / 100) * 75
        score += consistency_score
        
        return max(300, min(850, score))
    
    def calculate_risk_assessment(self):
        """Comprehensive risk assessment"""
        score = self.calculate_credit_score()
        
        if score >= 700:
            risk_level = "Low"
            approval_probability = "High (>80%)"
            suggested_rate = "8-12%"
        elif score >= 550:
            risk_level = "Moderate"
            approval_probability = "Medium (50-80%)"
            suggested_rate = "12-18%"
        else:
            risk_level = "High"
            approval_probability = "Low (<50%)"
            suggested_rate = "18-24%+"
        
        return {
            'risk_level': risk_level,
            'approval_probability': approval_probability,
            'suggested_rate': suggested_rate,
            'credit_score': score
        }
    
    def get_all_metrics(self):
        """Return comprehensive metrics dictionary"""
        dry_spell_metrics = self.calculate_dry_spell()
        risk_assessment = self.calculate_risk_assessment()
        
        return {
            **risk_assessment,
            'income_volatility': self.calculate_income_volatility(),
            'income_trend': self.calculate_income_trend(),
            'income_consistency': self.calculate_income_consistency(),
            'max_dry_spell': dry_spell_metrics['max_dry_spell'],
            'avg_dry_spell': dry_spell_metrics['avg_dry_spell'],
            'dry_spell_count': dry_spell_metrics['dry_spell_count'],
            'savings_cushion': self.savings / self.total_income if self.total_income > 0 else 0,
            'total_income': self.total_income,
            'total_expenses': self.total_expenses,
            'savings': self.savings,
            'monthly_metrics': self.monthly_metrics,
            'income_days_count': len(self.income_days)
        }


class EnhancedGenAISimulator:
    """Enhanced AI simulation with more sophisticated rule-based logic"""
    
    @staticmethod
    def generate_comprehensive_report(metrics):
        """Generate a comprehensive financial health report"""
        score = metrics['credit_score']
        risk_level = metrics['risk_level']
        trend = metrics['income_trend']
        consistency = metrics['income_consistency']
        
        # Financial health score (0-100)
        health_score = (score - 300) / 550 * 100
        
        # Generate insights based on metrics
        insights = []
        recommendations = []
        
        # Volatility insights
        if metrics['income_volatility'] > 0.5:
            insights.append("🔴 High income volatility detected - Consider diversifying income sources")
            recommendations.append("Explore multiple gig platforms to reduce dependency on single income stream")
        elif metrics['income_volatility'] > 0.3:
            insights.append("🟡 Moderate income volatility - Room for improvement in stability")
            recommendations.append("Build a retainer client base for more predictable income")
        else:
            insights.append("🟢 Stable income pattern - Well managed cash flow")
        
        # Dry spell insights
        if metrics['max_dry_spell'] > 14:
            insights.append("🔴 Extended dry spells (>2 weeks) - Emergency fund critical")
            recommendations.append("Aim to build emergency fund covering 3-6 months of expenses")
        elif metrics['max_dry_spell'] > 7:
            insights.append("🟡 Occasional income gaps - Monitor and plan accordingly")
        
        # Trend insights
        if trend > 5:
            insights.append("🟢 Positive income growth trend - Continue current strategies")
        elif trend < -5:
            insights.append("🔴 Declining income trend - Immediate action required")
            recommendations.append("Review pricing strategy and client acquisition methods")
        
        # Generate narrative
        narrative = f"""
        📊 **Financial Health Score: {health_score:.0f}/100** | Risk Level: {risk_level}
        
        **Key Insights:**
        {chr(10).join(f'• {insight}' for insight in insights)}
        
        **Recommendations:**
        {chr(10).join(f'• {rec}' for rec in recommendations[:3])}
        
        **Lending Assessment:**
        • Approval Probability: {metrics['approval_probability']}
        • Suggested Interest Rate: {metrics['suggested_rate']}
        • Credit Score: {score:.0f}/850
        """
        
        return narrative
    
    @staticmethod
    def generate_smart_nudge(metrics, channel='whatsapp'):
        """Generate smart nudges based on behavioral economics principles"""
        score = metrics['credit_score']
        consistency = metrics['income_consistency']
        trend = metrics['income_trend']
        
        # Determine nudge type based on user behavior patterns
        if score >= 700:
            nudge_type = "achievement"
        elif score >= 550:
            nudge_type = "progression"
        else:
            nudge_type = "support"
        
        nudges = {
            'whatsapp': {
                'achievement': f"""
🌟 **Financial Milestone Achieved!**

Congratulations on maintaining a strong financial profile! 
Your credit score of {score:.0f} puts you in the top tier of gig workers.

💎 **Exclusive Offer:** Pre-approved loan up to ₹50,000 at preferential rates
🎯 **Next Goal:** Reach 800+ score to unlock premium benefits

Tap to explore → [Link]
""",
                'progression': f"""
📈 **You're Making Progress!**

Your financial health is improving! Score: {score:.0f}/850

💡 **Quick Wins to Boost Your Score:**
• Reduce dry spells by 20% → +50 points
• Increase savings by 5% → +30 points

🎁 **Reward:** Access to financial planning tools

Keep going! 💪
""",
                'support': f"""
🤝 **We're Here to Help**

We noticed you're facing some financial challenges. 
Your current score: {score:.0f}

🆘 **Immediate Support Available:**
• Free financial counseling session
• Flexible payment options for existing loans
• Community support group access

You're not alone in this journey. Let us help you bounce back!

Reply HELP for assistance
"""
            }
        }
        
        return nudges[channel][nudge_type]


def create_interactive_charts(metrics):
    """Create interactive Plotly charts for better visualization"""
    
    # Monthly cash flow chart
    if not metrics['monthly_metrics'].empty:
        fig = go.Figure()
        
        monthly_data = metrics['monthly_metrics']
        
        # Add income bar
        fig.add_trace(go.Bar(
            name='Income',
            x=monthly_data['Month'].astype(str),
            y=monthly_data['Total_Income'],
            marker_color='#667eea'
        ))
        
        # Add expense bar
        fig.add_trace(go.Bar(
            name='Expenses',
            x=monthly_data['Month'].astype(str),
            y=monthly_data['Total_Expenses'],
            marker_color='#f5576c'
        ))
        
        # Add savings line
        fig.add_trace(go.Scatter(
            name='Net Savings',
            x=monthly_data['Month'].astype(str),
            y=monthly_data['Net_Savings'],
            mode='lines+markers',
            line=dict(color='#84fab0', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='Monthly Cash Flow Analysis',
            xaxis_title='Month',
            yaxis_title='Amount ($)',
            barmode='group',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    return None


def main():
    """Enhanced main application"""
    
    # Header with gradient styling
    st.markdown('<div class="main-header">📊 AI-Driven Gig Worker Financial Health Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced Risk Assessment • Cash Flow Prediction • Smart Financial Insights</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/freelance.png", width=80)
        st.markdown("### ⚙️ Configuration")
        
        freelancer_name = st.text_input("👤 Freelancer Name", value="Freelancer Profile")
        st.session_state['freelancer_name'] = freelancer_name
        
        st.markdown("---")
        st.markdown("### 📁 Data Source")
        
        uploaded_file = st.file_uploader(
            "Upload Bank Statement (CSV)",
            type=['csv'],
            help="Required columns: Date, Amount, Type (Income/Expense), Description (optional)"
        )
        
        st.markdown("---")
        if st.button("🚀 Generate Sample Data", use_container_width=True):
            with st.spinner("Generating realistic freelance data..."):
                mock_df = generate_enhanced_mock_data()
                st.session_state['df'] = mock_df
                st.success("✅ Sample data generated!")
                st.balloons()
        
        st.markdown("---")
        st.markdown("### 📊 Analysis Settings")
        show_advanced = st.checkbox("Show Advanced Metrics", value=False)
    
    # Main content
    df = None
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['df'] = df
            st.success("✅ Data uploaded successfully!")
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    elif 'df' in st.session_state:
        df = st.session_state['df']
        st.info("📊 Using previously loaded data")
    
    if df is not None:
        try:
            # Initialize enhanced engine
            engine = EnhancedGigWorkerRiskEngine(df)
            metrics = engine.get_all_metrics()
            ai_sim = EnhancedGenAISimulator()
            
            # Risk assessment dashboard
            st.markdown("---")
            st.markdown("### 📈 Financial Health Dashboard")
            
            # Top-level metrics
            col1, col2, col3, col4 = st.columns(4)
            
            risk_color = {
                'Low': '#84fab0',
                'Moderate': '#f6d365',
                'High': '#f5576c'
            }
            
            with col1:
                st.metric(
                    "🎯 Credit Score",
                    f"{metrics['credit_score']:.0f}/850",
                    delta=f"Risk: {metrics['risk_level']}",
                    delta_color="normal"
                )
            
            with col2:
                st.metric(
                    "📊 Income Stability",
                    f"{metrics['income_consistency']:.1f}%",
                    delta=f"Volatility: {metrics['income_volatility']:.2f}"
                )
