"""
Interactive Plotly visualizations for model evaluation metrics
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    calibration_curve,
    roc_auc_score,
    average_precision_score
)
from typing import Dict, List, Any, Optional
from pathlib import Path


def create_interactive_roc_curve(
    y_true: np.ndarray,
    y_probs: Dict[str, np.ndarray],
    output_path: Optional[str] = None
) -> go.Figure:
    """
    Create interactive ROC curve with multiple models

    Args:
        y_true: True labels
        y_probs: Dictionary of {model_name: predicted_probabilities}
        output_path: Optional path to save HTML file

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    for idx, (model_name, y_prob) in enumerate(y_probs.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'{model_name} (AUC={auc:.3f})',
            line=dict(color=colors[idx % len(colors)], width=3),
            hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
        ))

    # Add diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random (AUC=0.500)',
        line=dict(color='gray', width=2, dash='dash'),
        showlegend=True,
        hoverinfo='skip'
    ))

    fig.update_layout(
        title={
            'text': '📈 Interactive ROC Curve',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        xaxis=dict(range=[0, 1], gridcolor='lightgray'),
        yaxis=dict(range=[0, 1], gridcolor='lightgray'),
        hovermode='closest',
        template='plotly_white',
        width=900,
        height=700,
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        )
    )

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_file))
        print(f"✅ ROC curve saved: {output_file}")

    return fig


def create_interactive_pr_curve(
    y_true: np.ndarray,
    y_probs: Dict[str, np.ndarray],
    output_path: Optional[str] = None
) -> go.Figure:
    """
    Create interactive Precision-Recall curve with multiple models

    Args:
        y_true: True labels
        y_probs: Dictionary of {model_name: predicted_probabilities}
        output_path: Optional path to save HTML file

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    # Calculate baseline (prevalence)
    baseline = y_true.mean()

    for idx, (model_name, y_prob) in enumerate(y_probs.items()):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)

        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name=f'{model_name} (AP={ap:.3f})',
            line=dict(color=colors[idx % len(colors)], width=3),
            hovertemplate='Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>'
        ))

    # Add baseline
    fig.add_hline(
        y=baseline,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Baseline (Prevalence={baseline:.3f})",
        annotation_position="right"
    )

    fig.update_layout(
        title={
            'text': '📊 Interactive Precision-Recall Curve',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title='Recall',
        yaxis_title='Precision',
        xaxis=dict(range=[0, 1], gridcolor='lightgray'),
        yaxis=dict(range=[0, 1], gridcolor='lightgray'),
        hovermode='closest',
        template='plotly_white',
        width=900,
        height=700,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_file))
        print(f"✅ PR curve saved: {output_file}")

    return fig


def create_interactive_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    output_path: Optional[str] = None
) -> go.Figure:
    """
    Create interactive confusion matrix heatmap

    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        output_path: Optional path to save HTML file

    Returns:
        Plotly figure object
    """
    cm = confusion_matrix(y_true, y_pred)

    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create annotations
    annotations = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=f"{cm[i, j]}<br>({cm_percent[i, j]:.1f}%)",
                    showarrow=False,
                    font=dict(color='white' if cm[i, j] > cm.max() / 2 else 'black', size=14)
                )
            )

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Negative', 'Predicted Positive'],
        y=['Actual Negative', 'Actual Positive'],
        colorscale='Blues',
        showscale=True,
        hovertemplate='%{y}<br>%{x}<br>Count: %{z}<extra></extra>'
    ))

    fig.update_layout(
        title={
            'text': f'🎯 Confusion Matrix - {model_name}',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        annotations=annotations,
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        template='plotly_white',
        width=700,
        height=700
    )

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_file))
        print(f"✅ Confusion matrix saved: {output_file}")

    return fig


def create_interactive_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    model_name: str = "Model",
    top_n: int = 20,
    output_path: Optional[str] = None
) -> go.Figure:
    """
    Create interactive feature importance bar chart

    Args:
        feature_names: List of feature names
        importances: Array of feature importance values
        model_name: Name of the model
        top_n: Number of top features to display
        output_path: Optional path to save HTML file

    Returns:
        Plotly figure object
    """
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    # Create color scale based on importance
    colors = ['rgba(52, 152, 219, 0.7)' if i < len(top_importances) // 2
              else 'rgba(231, 76, 60, 0.7)' for i in range(len(top_importances))]

    fig = go.Figure(data=[
        go.Bar(
            x=top_importances,
            y=top_features,
            orientation='h',
            marker_color=colors,
            hovertemplate='%{y}<br>Importance: %{x:.4f}<extra></extra>'
        )
    ])

    fig.update_layout(
        title={
            'text': f'⭐ Feature Importance - {model_name}',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title='Importance',
        yaxis_title='Feature',
        yaxis=dict(autorange="reversed"),
        template='plotly_white',
        width=900,
        height=max(600, top_n * 25),
        hovermode='closest'
    )

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_file))
        print(f"✅ Feature importance saved: {output_file}")

    return fig


def create_interactive_calibration_curve(
    y_true: np.ndarray,
    y_probs: Dict[str, np.ndarray],
    n_bins: int = 10,
    output_path: Optional[str] = None
) -> go.Figure:
    """
    Create interactive calibration curve with multiple models

    Args:
        y_true: True labels
        y_probs: Dictionary of {model_name: predicted_probabilities}
        n_bins: Number of bins for calibration
        output_path: Optional path to save HTML file

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    for idx, (model_name, y_prob) in enumerate(y_probs.items()):
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy='uniform'
        )

        fig.add_trace(go.Scatter(
            x=mean_predicted_value,
            y=fraction_of_positives,
            mode='lines+markers',
            name=model_name,
            line=dict(color=colors[idx % len(colors)], width=3),
            marker=dict(size=10),
            hovertemplate='Predicted: %{x:.3f}<br>Actual: %{y:.3f}<extra></extra>'
        ))

    # Add perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='gray', width=2, dash='dash'),
        showlegend=True,
        hoverinfo='skip'
    ))

    fig.update_layout(
        title={
            'text': '🎯 Calibration Curve (Reliability Diagram)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title='Mean Predicted Probability',
        yaxis_title='Fraction of Positives',
        xaxis=dict(range=[0, 1], gridcolor='lightgray'),
        yaxis=dict(range=[0, 1], gridcolor='lightgray'),
        hovermode='closest',
        template='plotly_white',
        width=900,
        height=700,
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        )
    )

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_file))
        print(f"✅ Calibration curve saved: {output_file}")

    return fig


def create_metrics_dashboard(
    y_true: np.ndarray,
    y_probs: Dict[str, np.ndarray],
    metrics: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None
) -> go.Figure:
    """
    Create comprehensive metrics dashboard with multiple subplots

    Args:
        y_true: True labels
        y_probs: Dictionary of {model_name: predicted_probabilities}
        metrics: Dictionary of {model_name: {metric_name: value}}
        output_path: Optional path to save HTML file

    Returns:
        Plotly figure object with subplots
    """
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'ROC Curves',
            'Precision-Recall Curves',
            'Calibration Curves',
            'Key Metrics Comparison'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "bar"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    # 1. ROC Curves (top-left)
    for idx, (model_name, y_prob) in enumerate(y_probs.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        fig.add_trace(
            go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{model_name}',
                line=dict(color=colors[idx % len(colors)], width=2),
                legendgroup=model_name,
                showlegend=True
            ),
            row=1, col=1
        )

    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                   line=dict(color='gray', dash='dash'),
                   showlegend=False),
        row=1, col=1
    )

    # 2. PR Curves (top-right)
    for idx, (model_name, y_prob) in enumerate(y_probs.items()):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)

        fig.add_trace(
            go.Scatter(
                x=recall, y=precision,
                mode='lines',
                name=model_name,
                line=dict(color=colors[idx % len(colors)], width=2),
                legendgroup=model_name,
                showlegend=False
            ),
            row=1, col=2
        )

    # 3. Calibration Curves (bottom-left)
    for idx, (model_name, y_prob) in enumerate(y_probs.items()):
        fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)

        fig.add_trace(
            go.Scatter(
                x=mean_pred, y=fraction_pos,
                mode='lines+markers',
                name=model_name,
                line=dict(color=colors[idx % len(colors)], width=2),
                marker=dict(size=8),
                legendgroup=model_name,
                showlegend=False
            ),
            row=2, col=1
        )

    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                   line=dict(color='gray', dash='dash'),
                   showlegend=False),
        row=2, col=1
    )

    # 4. Metrics Comparison (bottom-right)
    metric_names = ['PR-AUC', 'ROC-AUC', 'ECE']
    model_names = list(y_probs.keys())

    for idx, metric_name in enumerate(metric_names):
        values = [metrics[model][metric_name] for model in model_names]

        fig.add_trace(
            go.Bar(
                x=model_names,
                y=values,
                name=metric_name,
                marker_color=colors[idx],
                hovertemplate=f'{metric_name}: %{{y:.3f}}<extra></extra>'
            ),
            row=2, col=2
        )

    # Update axes labels
    fig.update_xaxes(title_text="FPR", row=1, col=1)
    fig.update_yaxes(title_text="TPR", row=1, col=1)

    fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text="Precision", row=1, col=2)

    fig.update_xaxes(title_text="Predicted Prob", row=2, col=1)
    fig.update_yaxes(title_text="Fraction of Positives", row=2, col=1)

    fig.update_xaxes(title_text="Model", row=2, col=2)
    fig.update_yaxes(title_text="Score", row=2, col=2)

    # Update layout
    fig.update_layout(
        title={
            'text': '📊 Comprehensive Metrics Dashboard',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24}
        },
        height=1000,
        width=1600,
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        )
    )

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_file))
        print(f"✅ Metrics dashboard saved: {output_file}")

    return fig