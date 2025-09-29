"""
Plotly Interactive Visualization Utilities
For exporting to HTML with GitHub Pages support
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix


def create_interactive_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    title: str = "ROC Curve"
) -> go.Figure:
    """
    Create interactive ROC curve with Plotly

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        title: Plot title

    Returns:
        Plotly Figure object
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    from sklearn.metrics import auc
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()

    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC (AUC = {roc_auc:.3f})',
        line=dict(color='blue', width=2),
        hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<br>Threshold: %{text:.3f}',
        text=thresholds
    ))

    # Diagonal reference
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random (AUC = 0.5)',
        line=dict(color='red', width=2, dash='dash')
    ))

    fig.update_layout(
        title=title,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=700,
        height=600,
        hovermode='closest'
    )

    return fig


def create_interactive_pr_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    title: str = "Precision-Recall Curve"
) -> go.Figure:
    """
    Create interactive PR curve with Plotly

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        title: Plot title

    Returns:
        Plotly Figure object
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    from sklearn.metrics import auc
    pr_auc = auc(recall, precision)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name=f'PR (AUC = {pr_auc:.3f})',
        line=dict(color='green', width=2),
        hovertemplate='Recall: %{x:.3f}<br>Precision: %{y:.3f}'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Recall',
        yaxis_title='Precision',
        width=700,
        height=600,
        hovermode='closest'
    )

    return fig


def create_interactive_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None
) -> go.Figure:
    """
    Create interactive confusion matrix heatmap

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels (defaults to [0, 1])

    Returns:
        Plotly Figure object
    """
    if labels is None:
        labels = ['False Positive', 'True Exoplanet']

    cm = confusion_matrix(y_true, y_pred)

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=cm,
        texttemplate='%{text}',
        colorscale='Blues',
        showscale=True
    ))

    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        width=600,
        height=600
    )

    return fig


def create_interactive_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 15
) -> go.Figure:
    """
    Create interactive feature importance bar chart

    Args:
        feature_names: List of feature names
        importances: Feature importance values
        top_n: Number of top features to show

    Returns:
        Plotly Figure object
    """
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]

    fig = go.Figure(data=[
        go.Bar(
            x=importances[indices],
            y=[feature_names[i] for i in indices],
            orientation='h',
            marker=dict(
                color=importances[indices],
                colorscale='Viridis',
                showscale=True
            ),
            hovertemplate='Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>'
        )
    ])

    fig.update_layout(
        title=f'Top {top_n} Feature Importances',
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=max(400, top_n * 25),
        width=800
    )

    # Reverse y-axis to show highest at top
    fig.update_yaxis(autorange="reversed")

    return fig


def create_interactive_calibration_curve(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    n_bins: int = 10
) -> go.Figure:
    """
    Create interactive calibration curve comparison

    Args:
        y_true: True binary labels
        predictions: Dict mapping method names to predicted probabilities
        n_bins: Number of calibration bins

    Returns:
        Plotly Figure object
    """
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    fig = go.Figure()

    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='black', width=2, dash='dash')
    ))

    colors = px.colors.qualitative.Plotly

    for i, (method, y_pred) in enumerate(predictions.items()):
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins)
        brier = brier_score_loss(y_true, y_pred)

        fig.add_trace(go.Scatter(
            x=prob_pred, y=prob_true,
            mode='lines+markers',
            name=f'{method} (Brier: {brier:.4f})',
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=8),
            hovertemplate='Predicted: %{x:.3f}<br>Actual: %{y:.3f}<extra></extra>'
        ))

    fig.update_layout(
        title='Calibration Curves Comparison',
        xaxis_title='Mean Predicted Probability',
        yaxis_title='Fraction of Positives',
        width=800,
        height=600,
        hovermode='closest'
    )

    return fig


def create_metrics_dashboard(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred: np.ndarray,
    feature_names: List[str],
    feature_importances: np.ndarray
) -> go.Figure:
    """
    Create comprehensive metrics dashboard with subplots

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        y_pred: Predicted labels
        feature_names: Feature names
        feature_importances: Feature importance values

    Returns:
        Plotly Figure with subplots
    """
    # Create 2x2 subplot layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('ROC Curve', 'Precision-Recall Curve',
                       'Confusion Matrix', 'Top Features'),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "heatmap"}, {"type": "bar"}]
        ]
    )

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    fig.add_trace(
        go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC', line=dict(color='blue')),
        row=1, col=1
    )

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    fig.add_trace(
        go.Scatter(x=recall, y=precision, mode='lines', name='PR', line=dict(color='green')),
        row=1, col=2
    )

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig.add_trace(
        go.Heatmap(z=cm, colorscale='Blues', showscale=False),
        row=2, col=1
    )

    # Feature Importance (top 10)
    top_n = 10
    indices = np.argsort(feature_importances)[::-1][:top_n]
    fig.add_trace(
        go.Bar(
            x=feature_importances[indices],
            y=[feature_names[i] for i in indices],
            orientation='h',
            marker=dict(color='purple')
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=1000,
        width=1400,
        showlegend=False,
        title_text="Exoplanet Detection Metrics Dashboard"
    )

    return fig


def export_to_html(
    fig: go.Figure,
    output_path: Path,
    include_plotlyjs: str = 'cdn'
) -> None:
    """
    Export Plotly figure to HTML file

    Args:
        fig: Plotly Figure object
        output_path: Path to save HTML (e.g., docs/metrics.html)
        include_plotlyjs: 'cdn' (smaller file) or True (standalone)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.write_html(
        output_path,
        include_plotlyjs=include_plotlyjs,
        config={'displayModeBar': True, 'displaylogo': False}
    )

    print(f"✅ Interactive HTML saved to: {output_path}")
    print(f"   Open in browser or deploy to GitHub Pages")