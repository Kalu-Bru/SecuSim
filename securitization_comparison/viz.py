"""Visualization module for securitization comparison."""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

from .models.base import FACTORS

logger = logging.getLogger(__name__)

# Color scheme for consistent visualizations
COLORS = {
    "traditional": "#2E86AB",  # Blue
    "blockchain": "#A23B72",   # Purple
    "neutral": "#F18F01",      # Orange
    "background": "#F5F5F5"    # Light gray
}


def plot_scores(
    traditional_scores: Dict[str, float],
    blockchain_scores: Dict[str, float],
    output_dir: str = "reports/figures",
    show_plots: bool = True,
    save_plots: bool = True
) -> Tuple[plt.Figure, go.Figure]:
    """Create comprehensive visualization of securitization scores.
    
    Args:
        traditional_scores: Dictionary of traditional model scores.
        blockchain_scores: Dictionary of blockchain model scores.
        output_dir: Directory to save plots.
        show_plots: Whether to display plots inline.
        save_plots: Whether to save plots to files.
        
    Returns:
        Tuple of (matplotlib figure, plotly figure).
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create matplotlib radar chart
    fig_mpl = _create_radar_chart_matplotlib(traditional_scores, blockchain_scores)
    
    # Create plotly interactive charts
    fig_plotly = _create_interactive_charts(traditional_scores, blockchain_scores)
    
    # Save plots if requested
    if save_plots:
        radar_file = output_path / "securitization_comparison_radar.png"
        interactive_file = output_path / "securitization_comparison_interactive.html"
        
        fig_mpl.savefig(radar_file, dpi=300, bbox_inches="tight", facecolor="white")
        fig_plotly.write_html(str(interactive_file))
        
        logger.info(f"Saved radar chart to {radar_file}")
        logger.info(f"Saved interactive chart to {interactive_file}")
    
    # Show plots if requested
    if show_plots:
        plt.show()
        fig_plotly.show()
    
    return fig_mpl, fig_plotly


def _create_radar_chart_matplotlib(
    traditional_scores: Dict[str, float],
    blockchain_scores: Dict[str, float]
) -> plt.Figure:
    """Create a radar/spider chart using matplotlib.
    
    Args:
        traditional_scores: Traditional model scores.
        blockchain_scores: Blockchain model scores.
        
    Returns:
        Matplotlib figure object.
    """
    # Prepare data
    factors = FACTORS
    trad_values = [traditional_scores[factor] for factor in factors]
    chain_values = [blockchain_scores[factor] for factor in factors]
    
    # Number of variables
    N = len(factors)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Add first value to the end to close the radar chart
    trad_values += trad_values[:1]
    chain_values += chain_values[:1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    fig.patch.set_facecolor('white')
    
    # Plot traditional securitization
    ax.plot(angles, trad_values, 'o-', linewidth=2, 
            label='Traditional', color=COLORS["traditional"])
    ax.fill(angles, trad_values, alpha=0.25, color=COLORS["traditional"])
    
    # Plot blockchain securitization
    ax.plot(angles, chain_values, 'o-', linewidth=2,
            label='Blockchain', color=COLORS["blockchain"])
    ax.fill(angles, chain_values, alpha=0.25, color=COLORS["blockchain"])
    
    # Add factor labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(factors, fontsize=12)
    
    # Set y-axis limits and labels
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=10)
    ax.grid(True)
    
    # Add title and legend
    plt.title('Securitization Comparison: Traditional vs Blockchain\n', 
              fontsize=16, fontweight='bold', pad=30)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=12)
    
    # Add score annotations
    _add_score_annotations(ax, angles[:-1], trad_values[:-1], chain_values[:-1])
    
    plt.tight_layout()
    return fig


def _add_score_annotations(ax, angles, trad_values, chain_values):
    """Add score annotations to radar chart.
    
    Args:
        ax: Matplotlib polar axes.
        angles: Angle positions for each factor.
        trad_values: Traditional model values.
        chain_values: Blockchain model values.
    """
    for angle, trad_val, chain_val in zip(angles, trad_values, chain_values):
        # Traditional score annotation
        ax.annotate(f'{trad_val:.1f}', 
                   xy=(angle, trad_val), 
                   xytext=(10, 10), 
                   textcoords='offset points',
                   fontsize=9, 
                   color=COLORS["traditional"],
                   fontweight='bold')
        
        # Blockchain score annotation
        ax.annotate(f'{chain_val:.1f}', 
                   xy=(angle, chain_val), 
                   xytext=(-10, -10), 
                   textcoords='offset points',
                   fontsize=9, 
                   color=COLORS["blockchain"],
                   fontweight='bold')


def _create_interactive_charts(
    traditional_scores: Dict[str, float],
    blockchain_scores: Dict[str, float]
) -> go.Figure:
    """Create interactive charts using plotly.
    
    Args:
        traditional_scores: Traditional model scores.
        blockchain_scores: Blockchain model scores.
        
    Returns:
        Plotly figure object with multiple subplots.
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Radar Chart', 'Bar Chart Comparison', 
                       'Score Differences', 'Factor Rankings'),
        specs=[[{"type": "polar"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}]]
    )
    
    # 1. Interactive radar chart
    factors = FACTORS
    trad_values = [traditional_scores[factor] for factor in factors]
    chain_values = [blockchain_scores[factor] for factor in factors]
    
    # Traditional radar trace
    fig.add_trace(
        go.Scatterpolar(
            r=trad_values + [trad_values[0]],  # Close the polygon
            theta=factors + [factors[0]],
            fill='toself',
            name='Traditional',
            line_color=COLORS["traditional"],
            fillcolor=f'rgba(46, 134, 171, 0.3)',
            hovertemplate='<b>%{theta}</b><br>Score: %{r:.1f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Blockchain radar trace
    fig.add_trace(
        go.Scatterpolar(
            r=chain_values + [chain_values[0]],
            theta=factors + [factors[0]],
            fill='toself',
            name='Blockchain',
            line_color=COLORS["blockchain"],
            fillcolor=f'rgba(162, 59, 114, 0.3)',
            hovertemplate='<b>%{theta}</b><br>Score: %{r:.1f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Bar chart comparison
    fig.add_trace(
        go.Bar(
            x=factors,
            y=trad_values,
            name='Traditional',
            marker_color=COLORS["traditional"],
            hovertemplate='<b>%{x}</b><br>Score: %{y:.1f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(
            x=factors,
            y=chain_values,
            name='Blockchain',
            marker_color=COLORS["blockchain"],
            hovertemplate='<b>%{x}</b><br>Score: %{y:.1f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Score differences
    differences = [chain_values[i] - trad_values[i] for i in range(len(factors))]
    colors = [COLORS["blockchain"] if diff > 0 else COLORS["traditional"] for diff in differences]
    
    fig.add_trace(
        go.Bar(
            x=factors,
            y=differences,
            name='Blockchain Advantage',
            marker_color=colors,
            hovertemplate='<b>%{x}</b><br>Difference: %{y:+.1f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. Factor rankings
    # Create ranking data
    combined_scores = []
    for i, factor in enumerate(factors):
        combined_scores.append({'Factor': factor, 'Model': 'Traditional', 'Score': trad_values[i]})
        combined_scores.append({'Factor': factor, 'Model': 'Blockchain', 'Score': chain_values[i]})
    
    # Sort by score for ranking visualization
    trad_ranked = sorted(zip(factors, trad_values), key=lambda x: x[1], reverse=True)
    chain_ranked = sorted(zip(factors, chain_values), key=lambda x: x[1], reverse=True)
    
    # Create ranking comparison
    rank_factors = [item[0] for item in trad_ranked]
    rank_trad = [item[1] for item in trad_ranked]
    rank_chain = [blockchain_scores[factor] for factor in rank_factors]
    
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(factors)+1)),
            y=rank_trad,
            mode='markers+lines',
            name='Traditional (ranked)',
            marker=dict(size=8, color=COLORS["traditional"]),
            line=dict(color=COLORS["traditional"]),
            text=rank_factors,
            hovertemplate='<b>%{text}</b><br>Rank: %{x}<br>Score: %{y:.1f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(factors)+1)),
            y=rank_chain,
            mode='markers+lines',
            name='Blockchain (ranked)',
            marker=dict(size=8, color=COLORS["blockchain"]),
            line=dict(color=COLORS["blockchain"]),
            text=rank_factors,
            hovertemplate='<b>%{text}</b><br>Rank: %{x}<br>Score: %{y:.1f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="Securitization Comparison Dashboard",
        title_x=0.5,
        title_font_size=20,
        showlegend=True,
        height=800,
        template="plotly_white"
    )
    
    # Update polar subplot
    fig.update_polars(
        radialaxis=dict(range=[0, 100], tickvals=[20, 40, 60, 80, 100]),
        row=1, col=1
    )
    
    # Update bar chart layouts
    fig.update_xaxes(title_text="Factors", tickangle=45, row=1, col=2)
    fig.update_yaxes(title_text="Score", range=[0, 100], row=1, col=2)
    
    fig.update_xaxes(title_text="Factors", tickangle=45, row=2, col=1)
    fig.update_yaxes(title_text="Score Difference (Blockchain - Traditional)", row=2, col=1)
    
    fig.update_xaxes(title_text="Rank", row=2, col=2)
    fig.update_yaxes(title_text="Score", range=[0, 100], row=2, col=2)
    
    return fig


def create_summary_table(
    traditional_scores: Dict[str, float],
    blockchain_scores: Dict[str, float],
    output_dir: str = "reports/figures",
    save_table: bool = True
) -> str:
    """Create a summary table of the comparison results.
    
    Args:
        traditional_scores: Traditional model scores.
        blockchain_scores: Blockchain model scores.
        output_dir: Directory to save the table.
        save_table: Whether to save the table to a file.
        
    Returns:
        HTML string of the summary table.
    """
    # Calculate differences and winners
    table_data = []
    total_trad = 0
    total_chain = 0
    
    for factor in FACTORS:
        trad_score = traditional_scores[factor]
        chain_score = blockchain_scores[factor]
        difference = chain_score - trad_score
        winner = "Blockchain" if difference > 0 else "Traditional" if difference < 0 else "Tie"
        
        table_data.append({
            "Factor": factor,
            "Traditional": f"{trad_score:.1f}",
            "Blockchain": f"{chain_score:.1f}",
            "Difference": f"{difference:+.1f}",
            "Winner": winner
        })
        
        total_trad += trad_score
        total_chain += chain_score
    
    # Add totals row
    total_diff = total_chain - total_trad
    overall_winner = "Blockchain" if total_diff > 0 else "Traditional" if total_diff < 0 else "Tie"
    
    table_data.append({
        "Factor": "<b>TOTAL</b>",
        "Traditional": f"<b>{total_trad:.1f}</b>",
        "Blockchain": f"<b>{total_chain:.1f}</b>",
        "Difference": f"<b>{total_diff:+.1f}</b>",
        "Winner": f"<b>{overall_winner}</b>"
    })
    
    # Create HTML table
    html_table = """
    <table style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif;">
        <thead>
            <tr style="background-color: #f2f2f2;">
                <th style="border: 1px solid #ddd; padding: 12px; text-align: left;">Factor</th>
                <th style="border: 1px solid #ddd; padding: 12px; text-align: center;">Traditional</th>
                <th style="border: 1px solid #ddd; padding: 12px; text-align: center;">Blockchain</th>
                <th style="border: 1px solid #ddd; padding: 12px; text-align: center;">Difference</th>
                <th style="border: 1px solid #ddd; padding: 12px; text-align: center;">Winner</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for row in table_data:
        html_table += f"""
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">{row['Factor']}</td>
                <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{row['Traditional']}</td>
                <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{row['Blockchain']}</td>
                <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{row['Difference']}</td>
                <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{row['Winner']}</td>
            </tr>
        """
    
    html_table += """
        </tbody>
    </table>
    """
    
    # Save table if requested
    if save_table:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        table_file = output_path / "comparison_summary.html"
        
        with open(table_file, 'w') as f:
            f.write(html_table)
        
        logger.info(f"Saved summary table to {table_file}")
    
    return html_table 