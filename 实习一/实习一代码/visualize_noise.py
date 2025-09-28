# -*- coding: utf-8 -*-
"""
visualize_noise.py
用途：
1) 读取 GeoJSON/CSV（推荐为 DBSCAN 输出，含 cluster_id/Latitude/Longitude/Created Date）。
2) 生成投诉量的时间分布图（按小时、按星期）。
3) 生成 Folium 交互式簇地图（cluster_id >= 0 的记录，簇颜色各不相同）。
4) 额外生成一个可在浏览器直接打开的一体化报告 HTML（包含两张时间分布图 + 交互式地图）。

示例：
python visualize_noise.py --input dbscan_noise_clusters.geojson --time_col "Created Date" --out_prefix noise_viz
"""
from __future__ import annotations
import os
import io
import base64
import argparse
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
from matplotlib import cm, colors as mcolors


def read_any(input_path: str) -> gpd.GeoDataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"未找到输入文件：{input_path}")
    if input_path.lower().endswith(".geojson"):
        gdf = gpd.read_file(input_path)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    elif input_path.lower().endswith(".csv"):
        df = pd.read_csv(input_path)
        # 尝试构造 geometry
        lat_col = next((c for c in df.columns if c.lower() in ("latitude","lat","y")), None)
        lon_col = next((c for c in df.columns if c.lower() in ("longitude","lon","lng","x")), None)
        if not lat_col or not lon_col:
            raise KeyError("CSV 需包含纬度与经度列（Latitude/Longitude 或 lat/lon）")
        df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
        df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
        df = df.dropna(subset=[lat_col, lon_col])
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs="EPSG:4326",
        )
        # 标准化列名，便于后续使用
        if "Latitude" not in gdf.columns:
            gdf["Latitude"] = gdf.geometry.y
        if "Longitude" not in gdf.columns:
            gdf["Longitude"] = gdf.geometry.x
    else:
        raise ValueError("仅支持 .geojson 或 .csv")
    return gdf


def _save_fig_as_png() -> bytes:
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=150)
    plt.close()
    buf.seek(0)
    return buf.read()


def plot_time_distribution(gdf: gpd.GeoDataFrame, time_col: str | None, out_prefix: str) -> tuple[str | None, str | None, str | None, str | None]:
    """生成两张时间分布图，返回（小时图文件名, 星期图文件名, 小时图base64, 星期图base64）。任一缺失时返回 None。"""
    # 自动/手动识别时间列
    if time_col is None:
        for cand in ["Created Date","Created","Open Date","created_date","created","open_date"]:
            if cand in gdf.columns:
                time_col = cand
                break
    if time_col is None or time_col not in gdf.columns:
        print("未找到时间列，跳过时间分布绘图。")
        return None, None, None, None

    ts = pd.to_datetime(gdf[time_col], errors="coerce")
    ts = ts.dropna()
    if ts.empty:
        print("时间列为空，跳过时间分布绘图。")
        return None, None, None, None

    hour_png_b64 = dow_png_b64 = None
    # 按小时（上方折线 + 下方柱状）
    hour_counts = ts.dt.hour.value_counts().sort_index()
    hour_series = hour_counts.reindex(range(24), fill_value=0)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [1, 2]})
    # 上方折线图
    axes[0].plot(hour_series.index, hour_series.values, marker="o")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Complaints by Hour of Day (Line)")
    # 下方柱状图
    axes[1].bar(hour_series.index, hour_series.values)
    axes[1].set_xlabel("Hour (0-23)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Complaints by Hour of Day (Bar)")

    hour_png = _save_fig_as_png()
    hour_png_b64 = base64.b64encode(hour_png).decode('ascii')
    hour_path = f"{out_prefix}_time_by_hour.png"
    with open(hour_path, 'wb') as f:
        f.write(hour_png)

    # 按星期几（上方折线 + 下方柱状）（0=Mon, 6=Sun）
    dow_counts = ts.dt.dayofweek.value_counts().sort_index()
    dow_index = list(range(7))
    dow_series = dow_counts.reindex(dow_index, fill_value=0)
    dow_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [1, 2]})
    # 上方折线图
    axes[0].plot(dow_index, dow_series.values, marker="o")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Complaints by Day of Week (Line)")
    # 下方柱状图
    axes[1].bar(dow_index, dow_series.values)
    axes[1].set_xlabel("Day of Week")
    axes[1].set_ylabel("Count")
    axes[1].set_xticks(dow_index, dow_labels)
    axes[1].set_title("Complaints by Day of Week (Bar)")

    dow_png = _save_fig_as_png()
    dow_png_b64 = base64.b64encode(dow_png).decode('ascii')
    dow_path = f"{out_prefix}_time_by_dow.png"
    with open(dow_path, 'wb') as f:
        f.write(dow_png)

    print(f"已生成时间分布图：{hour_path}, {dow_path}")
    return hour_path, dow_path, hour_png_b64, dow_png_b64


def make_folium_map(gdf: gpd.GeoDataFrame, out_prefix: str) -> tuple[str | None, str | None]:
    """生成 Folium 地图并返回（HTML路径，HTML字符串）。若无法生成，返回 (None, None)。"""
    if "cluster_id" not in gdf.columns:
        print("未找到 cluster_id 列，跳过 Folium 地图制作。")
        return None, None
    sub = gdf[gdf["cluster_id"] >= 0].copy()
    if sub.empty:
        print("无有效簇（cluster_id >= 0），跳过 Folium 地图制作。")
        return None, None

    # 地图中心
    center_lat = float(sub["Latitude"].mean() if "Latitude" in sub.columns else sub.geometry.y.mean())
    center_lon = float(sub["Longitude"].mean() if "Longitude" in sub.columns else sub.geometry.x.mean())
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="cartodbpositron")

    # 为每个簇分配颜色
    unique_ids = sorted(sub["cluster_id"].unique())
    n = len(unique_ids)
    cmap = cm.get_cmap("tab20", n if n > 1 else 2)
    id_to_color = {cid: mcolors.to_hex(cmap(i % cmap.N)) for i, cid in enumerate(unique_ids)}

    # 按簇分层
    for cid in unique_ids:
        group = sub[sub["cluster_id"] == cid]
        layer = folium.FeatureGroup(name=f"Cluster {cid} (n={len(group)})", show=True)
        color = id_to_color[cid]
        for _, row in group.iterrows():
            lat = float(row["Latitude"]) if "Latitude" in row else float(row.geometry.y)
            lon = float(row["Longitude"]) if "Longitude" in row else float(row.geometry.x)
            folium.CircleMarker(
                location=[lat, lon],
                radius=2,
                color=color,
                fill=True,
                fill_opacity=0.6,
                weight=0,
            ).add_to(layer)
        layer.add_to(fmap)

    # 添加热力图图层（基于 cluster_id >= 0 的点）
    heat_layer = folium.FeatureGroup(name="Heatmap (cluster_id>=0)", show=False)
    # HeatMap 需要 [lat, lon] 列表
    heat_data = [
        [
            float(row["Latitude"]) if "Latitude" in sub.columns else float(row.geometry.y),
            float(row["Longitude"]) if "Longitude" in sub.columns else float(row.geometry.x),
        ]
        for _, row in sub.iterrows()
    ]
    if len(heat_data) > 0:
        HeatMap(heat_data, radius=10, blur=15, max_zoom=18).add_to(heat_layer)
        heat_layer.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)

    # 导出与内嵌内容
    out_html = f"{out_prefix}_map.html"
    fmap.save(out_html)
    map_html_str = fmap.get_root().render()
    print(f"已生成交互式地图：{out_html}")
    return out_html, map_html_str


def build_report_html(out_prefix: str,
                       hour_png_b64: str | None,
                       dow_png_b64: str | None,
                       map_html_str: str | None) -> str:
    """生成一体化 HTML 报告并返回路径。"""
    report_path = f"{out_prefix}_report.html"
    # 简单 CSS 布局
    style = """
    <style>
      body{font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:24px;}
      h1{margin:0 0 12px;}
      .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;}
      .card{border:1px solid #ddd;border-radius:8px;padding:12px;background:#fff;}
      img{max-width:100%;height:auto;}
      .map{height:700px;border:1px solid #ddd;border-radius:8px;overflow:hidden;}
    </style>
    """
    # 拼装内容
    hour_img = f'<img alt="Complaints by Hour of Day" src="data:image/png;base64,{hour_png_b64}">' if hour_png_b64 else '<p>无可用的小时分布图。</p>'
    dow_img = f'<img alt="Complaints by Day of Week" src="data:image/png;base64,{dow_png_b64}">' if dow_png_b64 else '<p>无可用的星期分布图。</p>'
    map_block = map_html_str if map_html_str else '<p>无可用的交互式地图。</p>'

    html = f"""
    <!doctype html>
    <html lang="zh-CN">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Noise-Residential 热点可视化报告</title>
      {style}
    </head>
    <body>
      <h1>Noise-Residential 热点可视化报告</h1>
      <div class="grid">
        <div class="card">{hour_img}</div>
        <div class="card">{dow_img}</div>
      </div>
      <div class="card map">{map_block}</div>
    </body>
    </html>
    """
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return report_path


def _find_default_input() -> str | None:
    """在当前目录自动查找可用的输入文件，优先含 cluster_id 的 DBSCAN 结果。"""
    candidates = [
        # 优先 DBSCAN 输出
        'dbscan_noise_clusters.geojson',
        'dbscan_noise_clusters.csv',
        'dbscan_noise_cluster_summary.csv',  # 仅摘要，通常不含点，不作为首选
        # 备选清洗结果
        'noise_all.geojson',
        'noise_all_clean.geojson',
        'noise_all_clean.csv',
        'noise_all.csv',
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description="Noise-Residential 可视化（时间分布 + 簇地图 + HTML 报告）")
    parser.add_argument("--input", required=False, default=None, help="输入文件（.geojson 或 .csv），留空则自动在当前目录搜索常见文件名")
    parser.add_argument("--time_col", default=None, help="时间列名（默认自动识别，如 'Created Date'）")
    parser.add_argument("--out_prefix", default="noise_viz", help="输出文件前缀（默认 noise_viz）")
    parser.add_argument("--no_time_plot", action="store_true", help="不生成时间分布图")
    parser.add_argument("--no_map", action="store_true", help="不生成 Folium 地图")
    parser.add_argument("--no_report", action="store_true", help="不生成一体化 HTML 报告")
    args = parser.parse_args()

    # 若未提供 --input，则在当前目录尝试自动识别
    if not args.input:
        auto_input = _find_default_input()
        if auto_input is None:
            print("未提供 --input，且在当前目录未找到常见数据文件。请使用 --input 指定 .geojson 或 .csv 输入。")
            return
        print(f"未提供 --input，已自动使用：{auto_input}")
        args.input = auto_input

    gdf = read_any(args.input)

    hour_path = dow_path = None
    hour_b64 = dow_b64 = None
    if not args.no_time_plot:
        hour_path, dow_path, hour_b64, dow_b64 = plot_time_distribution(gdf, args.time_col, args.out_prefix)

    map_path = None
    map_html = None
    if not args.no_map:
        map_path, map_html = make_folium_map(gdf, args.out_prefix)

    if not args.no_report:
        report_path = build_report_html(args.out_prefix, hour_b64, dow_b64, map_html)
        print(f"已生成报告：{report_path}")


if __name__ == "__main__":
    main()
