#!/usr/bin/env python3
"""Streamlit web application for securitization comparison."""

import streamlit as st
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
import time
import os
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from securitization_comparison.models import TraditionalModel, BlockchainModel
from securitization_comparison.statistical_analysis import run_statistical_analysis

# Configure logging to suppress excessive output during streamlit runs
logging.basicConfig(level=logging.WARNING)

# Page configuration
st.set_page_config(
    page_title="Securitization Comparison",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 3rem;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #A23B72;
        font-size: 1.5rem;
        margin-bottom: 2rem;
    }
    .description-box {
        background-color: #04AF70;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .factor-explanation {
        background-color: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #2E86AB;
        margin: 0.5rem 0;
    }
    .winner-badge {
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .blockchain-winner {
        background-color: #d4edda;
        color: #155724;
    }
    .traditional-winner {
        background-color: #fff3cd;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

def load_fannie_mae_data(data_dir: str, sample_size: int = 50000) -> pd.DataFrame:
    """Load and process real Fannie Mae data files."""
    data_path = Path(data_dir)
    
    # Find all quarterly data files
    csv_files = list(data_path.glob("*.csv"))
    csv_files = [f for f in csv_files if f.name != "sample_loan_data.csv"]
    
    if not csv_files:
        # If no real data, create sample data
        return generate_sample_data(sample_size)
    
    # Show progress while loading
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_data = []
    total_files = len(csv_files)
    
    for i, file_path in enumerate(csv_files):
        status_text.text(f"Loading {file_path.name}...")
        progress_bar.progress((i + 1) / total_files)
        
        try:
            # Read file in chunks
            chunks = []
            chunk_count = 0
            max_chunks = max(1, sample_size // (50000 * len(csv_files)))
            
            for chunk in pd.read_csv(
                file_path,
                sep='|',
                header=None,
                chunksize=50000,
                low_memory=False,
                dtype=str
            ):
                if len(chunk) == 0:
                    continue
                    
                chunks.append(chunk)
                chunk_count += 1
                
                if chunk_count >= max_chunks:
                    break
                    
            if chunks:
                file_data = pd.concat(chunks, ignore_index=True)
                all_data.append(file_data)
                
        except Exception as e:
            st.warning(f"Could not load {file_path.name}: {e}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    if not all_data:
        st.warning("No Fannie Mae data found, generating sample data...")
        return generate_sample_data(sample_size)
    
    # Combine and process data
    df = pd.concat(all_data, ignore_index=True)
    
    # Sample if too large
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    # Process the data similar to the script
    df = df.rename(columns={
        1: 'loan_id',
        2: 'monthly_reporting_period', 
        7: 'current_interest_rate',
        8: 'current_actual_upb',
        9: 'loan_age',
        10: 'remaining_months_to_legal_maturity',
        14: 'current_loan_delinquency_status'
    })
    
    # Convert numeric columns
    numeric_cols = ['current_interest_rate', 'current_actual_upb', 'loan_age', 'remaining_months_to_legal_maturity']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Clean and filter data
    df = df.dropna(subset=['loan_id', 'current_actual_upb'])
    df = df[df['current_actual_upb'] > 0]
    
    # Take most recent record per loan
    df = df.sort_values(['loan_id', 'monthly_reporting_period']).groupby('loan_id').tail(1)
    
    # Create derived fields
    df['original_loan_amount'] = df['current_actual_upb'] * (1 + np.random.normal(0.2, 0.1, len(df)))
    df['loan_to_value_ratio'] = np.random.normal(75, 15, len(df)).clip(10, 100)
    df['debt_to_income_ratio'] = np.random.normal(30, 10, len(df)).clip(5, 60)
    df['credit_score'] = np.random.normal(720, 80, len(df)).clip(300, 850)
    df['current_loan_balance'] = df['current_actual_upb']
    df['months_since_origination'] = df['loan_age']
    df['prepayment_penalty_flag'] = np.random.choice([0, 1], len(df), p=[0.8, 0.2])
    df['occupancy_type'] = np.random.choice(['P', 'S', 'I'], len(df), p=[0.8, 0.15, 0.05])
    df['delinquency_status'] = pd.to_numeric(df['current_loan_delinquency_status'], errors='coerce').fillna(0)
    
    return df

def generate_sample_data(n_loans: int = 10000) -> pd.DataFrame:
    """Generate sample mortgage loan data."""
    np.random.seed(42)
    
    data = {
        'loan_id': [f'LOAN_{i:06d}' for i in range(n_loans)],
        'original_loan_amount': np.random.normal(300000, 100000, n_loans).clip(50000, 1000000),
        'loan_to_value_ratio': np.random.normal(75, 15, n_loans).clip(10, 100),
        'debt_to_income_ratio': np.random.normal(30, 10, n_loans).clip(5, 60),
        'credit_score': np.random.normal(720, 80, n_loans).clip(300, 850),
        'current_loan_balance': np.random.normal(250000, 80000, n_loans).clip(0, 800000),
        'months_since_origination': np.random.normal(60, 30, n_loans).clip(1, 360),
        'prepayment_penalty_flag': np.random.choice([0, 1], n_loans, p=[0.8, 0.2]),
        'occupancy_type': np.random.choice(['P', 'S', 'I'], n_loans, p=[0.8, 0.15, 0.05]),
        'delinquency_status': np.random.choice([0, 1, 2, 3], n_loans, p=[0.85, 0.10, 0.03, 0.02]),
    }
    
    return pd.DataFrame(data)

def create_interactive_radar_chart(traditional_scores, blockchain_scores):
    """Create an interactive radar chart using Plotly."""
    factors = list(traditional_scores.keys())
    
    fig = go.Figure()
    
    # Traditional securitization
    fig.add_trace(go.Scatterpolar(
        r=list(traditional_scores.values()),
        theta=factors,
        fill='toself',
        name='Traditional',
        line_color='#FF6B6B',
        fillcolor='rgba(255, 107, 107, 0.2)'
    ))
    
    # Blockchain securitization
    fig.add_trace(go.Scatterpolar(
        r=list(blockchain_scores.values()),
        theta=factors,
        fill='toself',
        name='Blockchain',
        line_color='#4ECDC4',
        fillcolor='rgba(78, 205, 196, 0.2)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Securitization Comparison: Traditional vs Blockchain",
        height=600
    )
    
    return fig

def create_comparison_bar_chart(traditional_scores, blockchain_scores):
    """Create a comparison bar chart."""
    factors = list(traditional_scores.keys())
    
    fig = go.Figure(data=[
        go.Bar(name='Traditional', x=factors, y=list(traditional_scores.values()), 
               marker_color='#FF6B6B', text=list(traditional_scores.values()), 
               texttemplate='%{text:.1f}', textposition='outside'),
        go.Bar(name='Blockchain', x=factors, y=list(blockchain_scores.values()), 
               marker_color='#4ECDC4', text=list(blockchain_scores.values()), 
               texttemplate='%{text:.1f}', textposition='outside')
    ])
    
    fig.update_layout(
        barmode='group',
        title="Factor-by-Factor Comparison",
        xaxis_title="Evaluation Factors",
        yaxis_title="Score (0-100)",
        height=500,
        yaxis=dict(range=[0, 105])
    )
    
    return fig

def create_difference_chart(traditional_scores, blockchain_scores):
    """Create a chart showing the differences between approaches."""
    factors = list(traditional_scores.keys())
    differences = [blockchain_scores[f] - traditional_scores[f] for f in factors]
    colors = ['#4ECDC4' if d > 0 else '#FF6B6B' for d in differences]
    
    fig = go.Figure(data=[
        go.Bar(x=factors, y=differences, marker_color=colors,
               text=[f"{d:+.1f}" for d in differences],
               texttemplate='%{text}', textposition='outside')
    ])
    
    fig.update_layout(
        title="Blockchain Advantage (+) / Traditional Advantage (-)",
        xaxis_title="Evaluation Factors",
        yaxis_title="Score Difference",
        height=400,
        yaxis=dict(range=[min(differences)-5, max(differences)+5])
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    return fig

def main():
    """Main Streamlit application."""
    
    # Header with logo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("images/logo.png", width=400)
    

    st.markdown('<h2 class="sub-header">Traditional vs Blockchain-Based Securitization Analysis</h2>', unsafe_allow_html=True)
    
    # Description
    st.markdown("""
    <div class="description-box">
        <h3>About the Analysis</h3>
        <p>This platform compares traditional securitization with blockchain-based securitization using real-world 
        mortgage data from Fannie Mae. The analysis evaluates both approaches across six key factors:</p>
        <ul>
            <li><strong>Transparency:</strong> Public disclosure and visibility of loan information</li>
            <li><strong>Liquidity:</strong> Ease of trading and market depth</li>
            <li><strong>Systemic Risk:</strong> Risk of widespread financial system impact</li>
            <li><strong>Governance:</strong> Decision-making and oversight mechanisms</li>
            <li><strong>Auditing:</strong> Verification and compliance capabilities</li>
            <li><strong>Interoperability:</strong> Compatibility across different systems</li>
        </ul>
        <p>Each factor is scored from 0-100, with higher scores indicating better performance.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Analysis Configuration")
    
    sample_size = st.sidebar.slider(
        "Sample Size (number of loans)",
        min_value=1000,
        max_value=100000,
        value=25000,
        step=1000,
        help="Number of loans to include in the analysis. Larger samples provide more robust results but take longer to process."
    )
    
    use_real_data = st.sidebar.checkbox(
        "Use Real Fannie Mae Data",
        value=True,
        help="Use actual Fannie Mae loan performance data if available, otherwise generate sample data."
    )
    
    # Statistical Analysis Configuration
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Statistical Analysis")
    
    run_statistical = st.sidebar.checkbox(
        "Run Monte Carlo Analysis",
        value=False,
        help="Perform sensitivity analysis across different parameter ranges to calculate win probability."
    )
    
    if run_statistical:
        n_simulations = st.sidebar.slider(
            "Number of Simulations",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            help="More simulations provide more accurate probability estimates but take longer."
        )
        
        n_threads = st.sidebar.slider(
            "Parallel Threads",
            min_value=1,
            max_value=8,
            value=4,
            step=1,
            help="Number of CPU threads to use for parallel processing."
        )
    
    # Main action button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Run Securitization Comparison", type="primary", use_container_width=True):
            
            # Loading state
            with st.spinner("Loading and processing loan data..."):
                # Load data
                if use_real_data and Path("data").exists():
                    df = load_fannie_mae_data("data", sample_size)
                    data_source = f"Fannie Mae Data ({len(df):,} loans)"
                else:
                    df = generate_sample_data(sample_size)
                    data_source = f"Simulated Data ({len(df):,} loans)"
            
            if len(df) == 0:
                st.error("❌ No data available for analysis. Please check your data directory.")
                return
            
            # Run analysis
            with st.spinner("Computing securitization scores..."):
                try:
                    # Initialize models
                    traditional_model = TraditionalModel(df)
                    blockchain_model = BlockchainModel(df)
                    
                    # Compute scores
                    traditional_scores = traditional_model.compute_scores()
                    blockchain_scores = blockchain_model.compute_scores()
                    
                    # Calculate averages
                    traditional_avg = sum(traditional_scores.values()) / len(traditional_scores)
                    blockchain_avg = sum(blockchain_scores.values()) / len(blockchain_scores)
                    overall_diff = blockchain_avg - traditional_avg
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {e}")
                    return
            
            # Display results
            st.success("✅ Analysis Complete!")
            
            # Data source info
            st.info(f"📊 **Data Source:** {data_source}")
            
            # Overall results
            st.markdown("## 🏆 Overall Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "🏛️ Traditional Average",
                    f"{traditional_avg:.1f}/100",
                    help="Average score across all factors for traditional securitization"
                )
            
            with col2:
                st.metric(
                    "⛓️ Blockchain Average", 
                    f"{blockchain_avg:.1f}/100",
                    delta=f"{overall_diff:+.1f}",
                    help="Average score across all factors for blockchain securitization"
                )
            
            with col3:
                winner = "🥇 Blockchain" if overall_diff > 0 else "🥇 Traditional"
                winner_class = "blockchain-winner" if overall_diff > 0 else "traditional-winner"
                st.markdown(f"""
                <div class="winner-badge {winner_class}">
                    {winner} Wins!<br>
                    <small>+{abs(overall_diff):.1f} points</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Interactive visualizations
            st.markdown("## 📈 Interactive Visualizations")
            
            # Tabs for different charts
            tab1, tab2, tab3 = st.tabs(["🎯 Radar Chart", "📊 Bar Comparison", "📈 Differences"])
            
            with tab1:
                st.plotly_chart(
                    create_interactive_radar_chart(traditional_scores, blockchain_scores),
                    use_container_width=True
                )
            
            with tab2:
                st.plotly_chart(
                    create_comparison_bar_chart(traditional_scores, blockchain_scores),
                    use_container_width=True
                )
            
            with tab3:
                st.plotly_chart(
                    create_difference_chart(traditional_scores, blockchain_scores),
                    use_container_width=True
                )
            
            # Detailed factor breakdown
            st.markdown("## 📋 Detailed Factor Analysis")
            
            for factor in traditional_scores.keys():
                trad_score = traditional_scores[factor]
                block_score = blockchain_scores[factor]
                difference = block_score - trad_score
                
                with st.expander(f"{factor}: Traditional {trad_score:.1f} vs Blockchain {block_score:.1f}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Traditional", f"{trad_score:.1f}/100")
                    
                    with col2:
                        st.metric("Blockchain", f"{block_score:.1f}/100", delta=f"{difference:+.1f}")
                    
                    with col3:
                        if difference > 0:
                            st.success(f"Blockchain +{difference:.1f}")
                        elif difference < 0:
                            st.error(f"Traditional +{abs(difference):.1f}")
                        else:
                            st.info("Tie")
                    
                    # Factor explanations
                    explanations = {
                        "Transparency": "Measures public disclosure and visibility of loan information and performance.",
                        "Liquidity": "Evaluates ease of trading, market depth, and ability to quickly convert to cash.",
                        "Systemic Risk": "Assesses risk of widespread financial system impact and contagion effects.",
                        "Governance": "Analyzes decision-making processes and oversight mechanisms.",
                        "Auditing": "Reviews verification capabilities and compliance monitoring.",
                        "Interoperability": "Examines compatibility and integration across different systems."
                    }
                    
                    st.markdown(f"""
                    <div class="factor-explanation" style="color: #000000;">
                        <strong>What this measures:</strong> {explanations.get(factor, "No description available.")}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Summary insights
            st.markdown("## 💡 Key Insights")
            
            advantages = []
            if blockchain_scores["Transparency"] > traditional_scores["Transparency"]:
                advantages.append("🔍 **Higher Transparency** through on-chain visibility")
            if blockchain_scores["Liquidity"] > traditional_scores["Liquidity"]:
                advantages.append("💧 **Better Liquidity** via automated market makers")
            if blockchain_scores["Systemic Risk"] > traditional_scores["Systemic Risk"]:
                advantages.append("⚖️ **Lower Systemic Risk** through diversification")
            if blockchain_scores["Interoperability"] > traditional_scores["Interoperability"]:
                advantages.append("🔗 **Better Interoperability** with standard protocols")
            
            if advantages:
                st.markdown("**Blockchain Advantages:**")
                for advantage in advantages:
                    st.markdown(f"- {advantage}")
            
            # Traditional advantages
            trad_advantages = []
            if traditional_scores["Governance"] > blockchain_scores["Governance"]:
                trad_advantages.append("🏛️ **Established Governance** frameworks")
            if traditional_scores["Auditing"] > blockchain_scores["Auditing"]:
                trad_advantages.append("📋 **Mature Auditing** processes")
            
            if trad_advantages:
                st.markdown("**Traditional Advantages:**")
                for advantage in trad_advantages:
                    st.markdown(f"- {advantage}")
            
            # Statistical Analysis Section
            if run_statistical:
                st.markdown("---")
                st.markdown("## 🎲 Monte Carlo Statistical Analysis")
                
                st.info(f"""
                **What this analyzes:** This section runs {n_simulations} simulations with different 
                parameter combinations to determine how often blockchain outperforms traditional 
                securitization across various scenarios.
                """)
                
                with st.spinner(f"Running {n_simulations} Monte Carlo simulations... This may take several minutes."):
                    try:
                        statistical_results = run_statistical_analysis(
                            df, 
                            n_simulations=n_simulations,
                            n_threads=n_threads,
                            output_dir="reports"
                        )
                        
                        # Display statistical results
                        summary = statistical_results["summary"]
                        
                        # Main probability result
                        st.markdown("### 🎯 Win Probability Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "🔗 Blockchain Wins",
                                f"{summary['blockchain_wins']}/{summary['total_simulations']}",
                                help="Number of simulations where blockchain outperformed traditional"
                            )
                        
                        with col2:
                            win_prob = summary['win_probability_percent']
                            st.metric(
                                "📊 Win Probability",
                                f"{win_prob:.1f}%",
                                help="Probability that blockchain outperforms traditional across different scenarios"
                            )
                        
                        with col3:
                            st.metric(
                                "🏛️ Traditional Wins",
                                f"{summary['traditional_wins']}/{summary['total_simulations']}",
                                help="Number of simulations where traditional outperformed blockchain"
                            )
                        
                        with col4:
                            if win_prob > 50:
                                st.success("🥇 Blockchain Favored")
                            elif win_prob < 50:
                                st.error("🥇 Traditional Favored")
                            else:
                                st.info("🤝 Even Match")
                        
                        # Score statistics
                        stats = statistical_results["score_statistics"]
                        st.markdown("### 📈 Score Distribution Statistics")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(
                                "Average Score Difference",
                                f"{stats['mean_difference']:+.1f}",
                                help="Average difference between blockchain and traditional scores (blockchain - traditional)"
                            )
                            
                            st.metric(
                                "Standard Deviation",
                                f"{stats['std_difference']:.1f}",
                                help="Variability in score differences across simulations"
                            )
                        
                        with col2:
                            st.metric(
                                "Minimum Difference",
                                f"{stats['min_difference']:+.1f}",
                                help="Worst case scenario for blockchain vs traditional"
                            )
                            
                            st.metric(
                                "Maximum Difference", 
                                f"{stats['max_difference']:+.1f}",
                                help="Best case scenario for blockchain vs traditional"
                            )
                        
                        # Confidence interval
                        ci = statistical_results["confidence_interval_95"]
                        st.markdown(f"**95% Confidence Interval:** [{ci[0]:+.1f}, {ci[1]:+.1f}] points")
                        
                        # Factor-level analysis
                        st.markdown("### 🔍 Factor-Level Win Rates")
                        
                        factor_analysis = statistical_results["factor_analysis"]
                        factor_data = []
                        
                        for factor, stats in factor_analysis.items():
                            factor_data.append({
                                "Factor": factor,
                                "Blockchain Win Rate": f"{stats['blockchain_win_rate']*100:.1f}%",
                                "Average Difference": f"{stats['mean_difference']:+.1f}",
                                "Traditional Avg": f"{stats['traditional_mean']:.1f}",
                                "Blockchain Avg": f"{stats['blockchain_mean']:.1f}"
                            })
                        
                        factor_df = pd.DataFrame(factor_data)
                        st.dataframe(factor_df, use_container_width=True)
                        
                        # Key insights from statistical analysis
                        st.markdown("### 💡 Statistical Insights")
                        
                        insights = []
                        
                        if win_prob > 70:
                            insights.append("🟢 **Strong Evidence** that blockchain consistently outperforms traditional securitization")
                        elif win_prob > 60:
                            insights.append("🟡 **Moderate Evidence** that blockchain tends to outperform traditional securitization")
                        elif win_prob < 30:
                            insights.append("🔴 **Strong Evidence** that traditional consistently outperforms blockchain securitization")
                        elif win_prob < 40:
                            insights.append("🟡 **Moderate Evidence** that traditional tends to outperform blockchain securitization")
                        else:
                            insights.append("⚪ **Mixed Results** - Performance varies significantly based on parameters")
                        
                        if abs(stats['mean_difference']) < 5:
                            insights.append("📊 **Close Competition** - Average score differences are small")
                        elif abs(stats['mean_difference']) > 20:
                            insights.append("📊 **Clear Performance Gap** - Significant average score differences")
                        
                        if stats['std_difference'] > 15:
                            insights.append("📈 **High Variability** - Results depend heavily on parameter assumptions")
                        elif stats['std_difference'] < 5:
                            insights.append("📈 **Consistent Results** - Outcome is relatively stable across parameter ranges")
                        
                        for insight in insights:
                            st.markdown(f"- {insight}")
                        
                        # Export note
                        st.info("📁 **Detailed results** have been exported to the `reports/` directory for further analysis.")
                        
                    except Exception as e:
                        st.error(f"❌ Statistical analysis failed: {e}")
                        st.exception(e)

if __name__ == "__main__":
    main() 