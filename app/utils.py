"""
工具函式：TAP/MAST/Lightkurve 便捷呼叫
"""
from typing import Optional, Tuple, Dict, Any, List
import requests
import pandas as pd

EXO_TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

def tap_query(sql: str, fmt: str = "json") -> pd.DataFrame:
    """簡易 TAP 同步查詢，回傳 pandas.DataFrame"""
    params = {"query": sql, "format": fmt}
    resp = requests.get(EXO_TAP, params=params, timeout=60)
    resp.raise_for_status()
    if fmt == "json":
        return pd.DataFrame(resp.json())
    else:
        from io import StringIO
        return pd.read_csv(StringIO(resp.text))

def toi_latest(n: int = 500) -> pd.DataFrame:
    sql = f"SELECT * FROM toi ORDER BY date TOI DESC"
    # 留空：使用者可自行補齊（欄位眾多）；這裡示意由 Notebook 直接用 astroquery 取
    return pd.DataFrame()
