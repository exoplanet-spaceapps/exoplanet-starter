"""
行星候選報告產生模組
生成 HTML/PDF 報告卡片與視覺化
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
import base64
from io import BytesIO


class ExoplanetReportGenerator:
    """系外行星候選報告產生器"""

    def __init__(self, style: str = "default"):
        """
        初始化報告產生器

        Parameters:
        -----------
        style : str
            報告風格 ('default', 'compact', 'detailed')
        """
        self.style = style
        self.setup_plotting_style()

    def setup_plotting_style(self):
        """設定繪圖風格"""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10

    def generate_candidate_card(
        self,
        tic_id: str,
        probability: float,
        features: Dict[str, float],
        light_curve_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        bls_result: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        產生候選行星報告卡片

        Parameters:
        -----------
        tic_id : str
            TIC 識別碼
        probability : float
            行星機率
        features : Dict[str, float]
            特徵值字典
        light_curve_data : Tuple[np.ndarray, np.ndarray], optional
            時間與通量陣列
        bls_result : Dict[str, Any], optional
            BLS 搜尋結果
        metadata : Dict[str, Any], optional
            額外元資料

        Returns:
        --------
        html : str
            HTML 格式報告
        """
        # 產生視覺化
        plots = {}
        if light_curve_data is not None:
            plots['light_curve'] = self._plot_light_curve(light_curve_data, tic_id)
            if bls_result is not None:
                plots['folded'] = self._plot_folded_curve(
                    light_curve_data,
                    bls_result.get('period', 1.0),
                    tic_id
                )

        # 建立 HTML 報告
        html = self._build_html_report(
            tic_id=tic_id,
            probability=probability,
            features=features,
            plots=plots,
            bls_result=bls_result,
            metadata=metadata or {}
        )

        return html

    def _plot_light_curve(
        self,
        light_curve_data: Tuple[np.ndarray, np.ndarray],
        tic_id: str
    ) -> str:
        """繪製光變曲線"""
        time, flux = light_curve_data

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 原始光曲線
        ax1.scatter(time, flux, s=1, alpha=0.5, color='blue')
        ax1.set_xlabel('Time (days)')
        ax1.set_ylabel('Normalized Flux')
        ax1.set_title(f'Light Curve - {tic_id}')
        ax1.grid(True, alpha=0.3)

        # 去趨勢後的光曲線（簡單移動平均）
        window = min(101, len(flux) // 10)
        if window > 3:
            detrended = flux - pd.Series(flux).rolling(
                window=window, center=True
            ).median().fillna(flux.mean())
            ax2.scatter(time, detrended, s=1, alpha=0.5, color='green')
            ax2.set_xlabel('Time (days)')
            ax2.set_ylabel('Detrended Flux')
            ax2.set_title('Detrended Light Curve')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _plot_folded_curve(
        self,
        light_curve_data: Tuple[np.ndarray, np.ndarray],
        period: float,
        tic_id: str
    ) -> str:
        """繪製相位折疊曲線"""
        time, flux = light_curve_data

        # 計算相位
        phase = (time % period) / period

        fig, ax = plt.subplots(figsize=(10, 6))

        # 繪製折疊曲線
        ax.scatter(phase, flux, s=1, alpha=0.3, color='purple')
        ax.scatter(phase + 1, flux, s=1, alpha=0.3, color='purple')  # 重複週期

        # 分箱平均
        n_bins = min(50, len(flux) // 20)
        if n_bins > 5:
            bins = np.linspace(0, 1, n_bins)
            binned_flux = []
            bin_centers = []
            for i in range(len(bins) - 1):
                mask = (phase >= bins[i]) & (phase < bins[i+1])
                if mask.sum() > 0:
                    binned_flux.append(np.median(flux[mask]))
                    bin_centers.append((bins[i] + bins[i+1]) / 2)

            if binned_flux:
                ax.plot(bin_centers, binned_flux, 'ro-', markersize=4, linewidth=2,
                       label='Binned median')
                ax.plot(np.array(bin_centers) + 1, binned_flux, 'ro-',
                       markersize=4, linewidth=2)

        ax.set_xlabel('Phase')
        ax.set_ylabel('Normalized Flux')
        ax.set_title(f'Phase-Folded Light Curve - Period: {period:.4f} days')
        ax.set_xlim(-0.1, 2.1)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _fig_to_base64(self, fig: Figure) -> str:
        """將 matplotlib 圖形轉換為 base64 編碼"""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64

    def _build_html_report(
        self,
        tic_id: str,
        probability: float,
        features: Dict[str, float],
        plots: Dict[str, str],
        bls_result: Optional[Dict[str, Any]] = None,
        metadata: Dict[str, Any] = {}
    ) -> str:
        """建立 HTML 報告"""

        # 判斷候選等級
        if probability >= 0.8:
            status = "強候選"
            status_color = "#28a745"
        elif probability >= 0.5:
            status = "中等候選"
            status_color = "#ffc107"
        else:
            status = "弱候選"
            status_color = "#dc3545"

        # 特徵表格
        features_html = ""
        for name, value in features.items():
            features_html += f"""
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{name}</td>
                <td style="padding: 8px; border-bottom: 1px solid #dee2e6; text-align: right;">
                    {value:.4f if isinstance(value, float) else value}
                </td>
            </tr>
            """

        # BLS 結果
        bls_html = ""
        if bls_result:
            bls_html = f"""
            <div class="card">
                <h3>BLS/TLS 搜尋結果</h3>
                <table style="width: 100%;">
                    <tr>
                        <td>週期</td>
                        <td style="text-align: right;">{bls_result.get('period', 'N/A'):.4f} days</td>
                    </tr>
                    <tr>
                        <td>深度</td>
                        <td style="text-align: right;">{bls_result.get('depth', 0)*1000:.1f} ppt</td>
                    </tr>
                    <tr>
                        <td>持續時間</td>
                        <td style="text-align: right;">{bls_result.get('duration', 0)*24:.1f} hours</td>
                    </tr>
                    <tr>
                        <td>SNR</td>
                        <td style="text-align: right;">{bls_result.get('snr', 0):.1f}</td>
                    </tr>
                </table>
            </div>
            """

        # 圖片區塊
        plots_html = ""
        if 'light_curve' in plots:
            plots_html += f"""
            <div class="card">
                <h3>光變曲線</h3>
                <img src="data:image/png;base64,{plots['light_curve']}"
                     style="width: 100%; max-width: 800px;">
            </div>
            """
        if 'folded' in plots:
            plots_html += f"""
            <div class="card">
                <h3>相位折疊圖</h3>
                <img src="data:image/png;base64,{plots['folded']}"
                     style="width: 100%; max-width: 800px;">
            </div>
            """

        # 組合完整 HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Exoplanet Candidate Report - {tic_id}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 15px;
                    padding: 30px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                }}
                .header {{
                    border-bottom: 3px solid #667eea;
                    padding-bottom: 20px;
                    margin-bottom: 30px;
                }}
                h1 {{
                    color: #333;
                    margin: 0;
                    font-size: 2.5em;
                }}
                .status-badge {{
                    display: inline-block;
                    padding: 8px 20px;
                    border-radius: 25px;
                    color: white;
                    font-weight: bold;
                    margin-left: 20px;
                }}
                .probability {{
                    font-size: 3em;
                    color: #667eea;
                    text-align: center;
                    margin: 20px 0;
                }}
                .card {{
                    background: #f8f9fa;
                    border-radius: 10px;
                    padding: 20px;
                    margin: 20px 0;
                }}
                h3 {{
                    color: #495057;
                    margin-top: 0;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 1px solid #dee2e6;
                    color: #6c757d;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>
                        {tic_id}
                        <span class="status-badge" style="background: {status_color};">
                            {status}
                        </span>
                    </h1>
                    <p style="color: #6c757d; margin-top: 10px;">
                        生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    </p>
                </div>

                <div class="probability">
                    行星機率: {probability:.1%}
                </div>

                {bls_html}

                <div class="card">
                    <h3>特徵值</h3>
                    <table>
                        {features_html}
                    </table>
                </div>

                {plots_html}

                <div class="footer">
                    <p>NASA Space Apps 2025 - Exoplanet AI Detection Pipeline</p>
                    <p>Apache License 2.0</p>
                </div>
            </div>
        </body>
        </html>
        """

        return html

    def save_report(
        self,
        html_content: str,
        output_path: str,
        tic_id: str
    ) -> str:
        """
        儲存報告至檔案

        Parameters:
        -----------
        html_content : str
            HTML 內容
        output_path : str
            輸出目錄
        tic_id : str
            TIC 識別碼

        Returns:
        --------
        file_path : str
            儲存的檔案路徑
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{tic_id}_report_{timestamp}.html"
        file_path = output_dir / filename

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return str(file_path)

    def generate_batch_summary(
        self,
        results: List[Dict[str, Any]],
        output_path: str = "reports"
    ) -> str:
        """
        產生批次處理摘要報告

        Parameters:
        -----------
        results : List[Dict[str, Any]]
            批次處理結果列表
        output_path : str
            輸出路徑

        Returns:
        --------
        summary_path : str
            摘要報告路徑
        """
        # 轉換為 DataFrame
        df = pd.DataFrame(results)

        # 統計資訊
        stats = {
            'total_candidates': len(df),
            'strong_candidates': len(df[df['probability'] >= 0.8]),
            'medium_candidates': len(df[(df['probability'] >= 0.5) & (df['probability'] < 0.8)]),
            'weak_candidates': len(df[df['probability'] < 0.5]),
            'mean_probability': df['probability'].mean(),
            'median_probability': df['probability'].median()
        }

        # 產生摘要 HTML
        html = self._build_batch_summary_html(df, stats)

        # 儲存報告
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = output_dir / f"batch_summary_{timestamp}.html"

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(html)

        return str(summary_path)

    def _build_batch_summary_html(
        self,
        df: pd.DataFrame,
        stats: Dict[str, Any]
    ) -> str:
        """建立批次摘要 HTML"""

        # 候選表格
        candidates_html = ""
        for _, row in df.nlargest(20, 'probability').iterrows():
            prob = row['probability']
            if prob >= 0.8:
                badge_color = "#28a745"
            elif prob >= 0.5:
                badge_color = "#ffc107"
            else:
                badge_color = "#dc3545"

            candidates_html += f"""
            <tr>
                <td>{row['tic_id']}</td>
                <td>
                    <div style="background: {badge_color}; color: white;
                               padding: 3px 10px; border-radius: 15px;
                               display: inline-block;">
                        {prob:.1%}
                    </div>
                </td>
                <td>{row.get('period', 'N/A'):.2f}</td>
                <td>{row.get('snr', 'N/A'):.1f}</td>
            </tr>
            """

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Batch Processing Summary</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: #f0f2f5;
                }}
                .container {{
                    max-width: 1000px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 10px;
                    padding: 30px;
                    box-shadow: 0 5px 20px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    border-bottom: 2px solid #667eea;
                    padding-bottom: 10px;
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 20px;
                    margin: 30px 0;
                }}
                .stat-card {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                }}
                .stat-value {{
                    font-size: 2em;
                    color: #667eea;
                    font-weight: bold;
                }}
                .stat-label {{
                    color: #6c757d;
                    margin-top: 5px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }}
                th {{
                    background: #f8f9fa;
                    padding: 12px;
                    text-align: left;
                    border-bottom: 2px solid #dee2e6;
                }}
                td {{
                    padding: 10px;
                    border-bottom: 1px solid #dee2e6;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>批次處理摘要報告</h1>

                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{stats['total_candidates']}</div>
                        <div class="stat-label">總候選數</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{stats['strong_candidates']}</div>
                        <div class="stat-label">強候選 (≥80%)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{stats['mean_probability']:.1%}</div>
                        <div class="stat-label">平均機率</div>
                    </div>
                </div>

                <h2>Top 20 候選目標</h2>
                <table>
                    <thead>
                        <tr>
                            <th>TIC ID</th>
                            <th>機率</th>
                            <th>週期 (days)</th>
                            <th>SNR</th>
                        </tr>
                    </thead>
                    <tbody>
                        {candidates_html}
                    </tbody>
                </table>

                <div style="text-align: center; margin-top: 40px; color: #6c757d;">
                    <p>生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>NASA Space Apps 2025 - Exoplanet AI Detection Pipeline</p>
                </div>
            </div>
        </body>
        </html>
        """

        return html


if __name__ == "__main__":
    # 測試報告產生
    print("🧪 測試報告產生模組")

    generator = ExoplanetReportGenerator()

    # 產生測試資料
    np.random.seed(42)
    time = np.linspace(0, 30, 1000)
    flux = 1.0 + 0.001 * np.random.randn(1000)
    # 加入假凌日
    transit_mask = ((time % 3.5) < 0.2)
    flux[transit_mask] *= 0.99

    # 產生報告
    html = generator.generate_candidate_card(
        tic_id="TIC 123456789",
        probability=0.85,
        features={
            'bls_period': 3.5,
            'bls_depth': 0.01,
            'bls_snr': 12.5,
            'odd_even_diff': 0.002
        },
        light_curve_data=(time, flux),
        bls_result={'period': 3.5, 'depth': 0.01, 'duration': 0.2, 'snr': 12.5}
    )

    # 儲存報告
    output_path = generator.save_report(html, "test_reports", "TIC_123456789")
    print(f"✅ 報告已儲存至: {output_path}")