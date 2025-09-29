from jinja2 import Template
import json
import pandas as pd
import numpy as np
from datetime import datetime

def generate_stats_html(gdf, noise_count, valid_clusters, cluster_counts):
    """
    生成包含统计信息和交互式图表的HTML面板（使用折线图和柱状图）
    
    参数:
    gdf: GeoDataFrame, 包含投诉数据
    noise_count: 噪声点数量
    valid_clusters: 有效聚类ID列表
    cluster_counts: 每个聚类的点数字典
    
    返回:
    包含完整统计面板的HTML字符串
    """
    # 确保正确的时间格式
    gdf['Created Date'] = pd.to_datetime(gdf['Created Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    
    # 计算基本统计数据
    total_points = len(gdf)
    clustered_count = total_points - noise_count
    clustered_percentage = (clustered_count / total_points * 100) if total_points > 0 else 0
    noise_percentage = (noise_count / total_points * 100) if total_points > 0 else 0
    
    # 将百分比转换为 Python 原生浮点数
    clustered_percentage = float(clustered_percentage)
    noise_percentage = float(noise_percentage)
    
    # 计算聚类分布
    cluster_distribution = []
    for cluster_id in valid_clusters:
        count = cluster_counts[cluster_id]
        percentage = (count / clustered_count * 100) if clustered_count > 0 else 0
        cluster_distribution.append({
            'id': cluster_id,
            'count': int(count),
            'percentage': float(percentage)
        })
    
    # 按数量排序聚类分布
    cluster_distribution.sort(key=lambda x: x['count'], reverse=True)
    
    # 计算周期性投诉模式数据
    gdf['hour'] = gdf['Created Date'].dt.hour
    gdf['day_of_week'] = gdf['Created Date'].dt.dayofweek
    
    hourly_counts = gdf.groupby('hour').size()
    daily_counts = gdf.groupby('day_of_week').size()
    
    # 准备小时数据
    hours = list(range(24))
    hour_values = [int(hourly_counts.get(h, 0)) for h in hours]
    
    # 准备星期数据
    day_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    day_values = [int(daily_counts.get(d, 0)) for d in range(7)]
    
    # 计算热点区域
    hotspots = []
    for i, cluster in enumerate(cluster_distribution[:5]):
        hotspots.append({
            'name': f'热点区域 {i+1}',
            'count': cluster['count']
        })
    
    # 计算响应时间（使用 Created Date 和 Closed Date）
    if 'Closed Date' in gdf.columns:
        gdf['Closed Date'] = pd.to_datetime(gdf['Closed Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
        
        # 规范化时间为无时区（naive）
        gdf['Created Date'] = gdf['Created Date'].dt.tz_localize(None)
        gdf['Closed Date'] = gdf['Closed Date'].dt.tz_localize(None)
        
        # 计算响应时间
        gdf['Response Time'] = (gdf['Closed Date'] - gdf['Created Date']).dt.total_seconds() / 3600
        avg_response_time = gdf['Response Time'].mean()
        avg_response_time = f"{avg_response_time:.1f}小时" if not pd.isna(avg_response_time) else "N/A"
    else:
        avg_response_time = "N/A"
    
    # 计算解决率
    if 'Status' in gdf.columns:
        resolution_rate = (gdf['Status'].str.lower() == 'closed').sum() / len(gdf) * 100
        resolution_rate = f"{resolution_rate:.1f}%"
    else:
        resolution_rate = "N/A"
    
    # 格式化更新时间
    update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 创建模板上下文
    context = {
        'total_points': int(total_points),
        'clustered_count': int(clustered_count),
        'clustered_percentage': clustered_percentage,
        'noise_count': int(noise_count),
        'noise_percentage': noise_percentage,
        'cluster_count': len(valid_clusters),
        'cluster_distribution': cluster_distribution[:5],
        'update_time': update_time,
        'hours': hours,
        'hour_values': hour_values,
        'day_names': day_names,
        'day_values': day_values,
        'hours_json': json.dumps(hours),
        'hour_values_json': json.dumps(hour_values),
        'day_names_json': json.dumps(day_names),
        'day_values_json': json.dumps(day_values),
        'hotspots': hotspots,
        'avg_response_time': avg_response_time,
        'resolution_rate': resolution_rate
    }
    
    # Jinja2 模板（保持不变）
    template_str = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>投诉数据统计面板</title>
        <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
        <style>
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f8f9fa;
            }
            
            #stats-container {
                position: fixed;
                bottom: 20px;
                left: 20px;
                z-index: 1000;
                display: flex;
                transition: transform 0.4s cubic-bezier(0.25, 0.1, 0.25, 1);
                transform: translateX(0);
            }
            
            #stats-container.collapsed {
                transform: translateX(-420px);
            }
            
            #stats-panel {
                width: 420px;
                background: rgba(255, 255, 255, 0.97);
                border-radius: 8px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                overflow: hidden;
                display: flex;
                flex-direction: column;
                backdrop-filter: blur(5px);
                border: 1px solid #e0e0e0;
                position: relative;
            }
            
            #toggle-stats {
                position: absolute;
                top: 10px;
                right: 10px;
                background: white;
                border: none;
                border-radius: 4px;
                padding: 8px 12px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.2);
                z-index: 1001;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            
            #toggle-stats:hover {
                background: #f5f5f5;
                transform: scale(1.05);
            }
            
            .stats-header {
                background: linear-gradient(135deg, #2c3e50, #3498db);
                color: white;
                padding: 15px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .stats-header h2 {
                font-size: 20px;
                font-weight: 600;
                margin: 0;
                display: flex;
                align-items: center;
            }
            
            .stats-header h2:before {
                content: '📊';
                margin-right: 10px;
                font-size: 24px;
            }
            
            .stats-content {
                padding: 15px;
                overflow-y: auto;
                max-height: 80vh;
            }
            
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 15px;
                margin-bottom: 15px;
            }
            
            .stat-card {
                background: #ffffff;
                border-radius: 10px;
                padding: 15px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                transition: all 0.3s ease;
                border-left: 4px solid #3498db;
            }
            
            .stat-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            
            .stat-card.highlight {
                border-left-color: #e74c3c;
            }
            
            .stat-card.success {
                border-left-color: #2ecc71;
            }
            
            .stat-label {
                font-size: 14px;
                color: #6c757d;
                margin-bottom: 8px;
                display: flex;
                align-items: center;
            }
            
            .stat-label i {
                margin-right: 6px;
                font-size: 16px;
            }
            
            .stat-value {
                font-size: 24px;
                font-weight: 700;
                color: #2c3e50;
            }
            
            .stat-highlight {
                color: #3498db;
            }
            
            .stat-description {
                font-size: 12px;
                color: #6c757d;
                margin-top: 5px;
            }
            
            .section-title {
                font-size: 16px;
                font-weight: 600;
                color: #2c3e50;
                margin: 20px 0 12px;
                padding-bottom: 8px;
                border-bottom: 2px solid #eaeaea;
                display: flex;
                align-items: center;
            }
            
            .section-title:before {
                content: '▸';
                margin-right: 8px;
                color: #3498db;
            }
            
            .chart-container {
                height: 200px;
                margin-bottom: 20px;
                background: white;
                border-radius: 8px;
                padding: 10px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            }
            
            .cluster-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 14px;
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            }
            
            .cluster-table th {
                background-color: #f1f5f9;
                text-align: left;
                padding: 12px 15px;
                font-weight: 600;
                color: #2c3e50;
            }
            
            .cluster-table td {
                padding: 10px 15px;
                border-bottom: 1px solid #eee;
            }
            
            .cluster-table tr:hover td {
                background-color: #f8fafc;
            }
            
            .count-cell {
                text-align: right;
                font-weight: 600;
                color: #3498db;
            }
            
            .percentage-cell {
                text-align: right;
                color: #6c757d;
            }
            
            .hotspots {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 12px;
                margin-top: 15px;
            }
            
            .hotspot-card {
                background: white;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                transition: all 0.3s ease;
            }
            
            .hotspot-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            
            .hotspot-name {
                font-weight: 600;
                margin-bottom: 8px;
                color: #2c3e50;
                display: flex;
                align-items: center;
            }
            
            .hotspot-name:before {
                content: '📍';
                margin-right: 6px;
            }
            
            .hotspot-count {
                color: #3498db;
                font-weight: 700;
                font-size: 20px;
            }
            
            .hotspot-description {
                font-size: 12px;
                color: #6c757d;
                margin-top: 5px;
            }
            
            .kpi-container {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 15px;
                margin: 20px 0;
            }
            
            .kpi-card {
                background: white;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                text-align: center;
            }
            
            .kpi-value {
                font-size: 24px;
                font-weight: 700;
                color: #3498db;
                margin: 10px 0;
            }
            
            .kpi-label {
                font-size: 14px;
                color: #6c757d;
            }
            
            .update-time {
                text-align: center;
                font-size: 12px;
                color: #6c757d;
                margin-top: 20px;
                padding-top: 15px;
                border-top: 1px solid #eee;
            }
            
            .progress-bar {
                height: 8px;
                background: #e9ecef;
                border-radius: 4px;
                margin-top: 8px;
                overflow: hidden;
            }
            
            .progress {
                height: 100%;
                background: linear-gradient(90deg, #3498db, #2ecc71);
                border-radius: 4px;
                transition: width 0.5s ease;
            }
        </style>
    </head>
    <body>
        <div id="stats-container">
            <div id="stats-panel">
                <button id="toggle-stats" title="折叠/展开统计面板">X</button>
                <div class="stats-header">
                    <h2>投诉数据分析面板</h2>
                </div>
                
                <div class="stats-content">
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-label">📋 总投诉量</div>
                            <div class="stat-value stat-highlight">{{ total_points }}</div>
                            <div class="stat-description">涵盖所有区域和类型</div>
                        </div>
                        
                        <div class="stat-card highlight">
                            <div class="stat-label">⚠️ 未聚类数量</div>
                            <div class="stat-value">{{ noise_count }}</div>
                            <div class="progress-bar">
                                <div class="progress" style="width: {{ noise_percentage }}%"></div>
                            </div>
                            <div class="stat-description">占比 {{ noise_percentage | round(1) }}%</div>
                        </div>
                        
                        <div class="stat-card">
                            <div class="stat-label">🔍 已聚类数量</div>
                            <div class="stat-value">{{ clustered_count }}</div>
                            <div class="progress-bar">
                                <div class="progress" style="width: {{ clustered_percentage }}%"></div>
                            </div>
                            <div class="stat-description">占比 {{ clustered_percentage | round(1) }}%</div>
                        </div>
                        
                        <div class="stat-card success">
                            <div class="stat-label">🗺️ 聚类数量</div>
                            <div class="stat-value">{{ cluster_count }}</div>
                            <div class="stat-description">识别出的热点区域</div>
                        </div>
                    </div>
                    
                    <div class="kpi-container">
                        <div class="kpi-card">
                            <div class="kpi-label">⏱️ 平均响应时间</div>
                            <div class="kpi-value">{{ avg_response_time }}</div>
                        </div>
                        <div class="kpi-card">
                            <div class="kpi-label">✅ 问题解决率</div>
                            <div class="kpi-value">{{ resolution_rate }}</div>
                        </div>
                    </div>
                    
                    <div class="section-title">投诉时间分布（按小时）</div>
                    <div class="chart-container" id="hour-chart"></div>
                    
                    <div class="section-title">投诉时间分布（按星期）</div>
                    <div class="chart-container" id="day-chart"></div>
                    
                    <div class="section-title">聚类分布</div>
                    <table class="cluster-table">
                        <thead>
                            <tr>
                                <th>聚类ID</th>
                                <th>数量</th>
                                <th>占比</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for cluster in cluster_distribution %}
                            <tr>
                                <td>{{ cluster.id }}</td>
                                <td class="count-cell">{{ cluster.count }}</td>
                                <td class="percentage-cell">{{ cluster.percentage | round(1) }}%</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                    
                    <div class="section-title">热点区域</div>
                    <div class="hotspots">
                        {% for hotspot in hotspots %}
                        <div class="hotspot-card">
                            <div class="hotspot-name">{{ hotspot.name }}</div>
                            <div class="hotspot-count">{{ hotspot.count }} 起</div>
                            <div class="hotspot-description">需要重点关注区域</div>
                        </div>
                        {% endfor %}
                    </div>
                    
                    <div class="update-time">
                        数据更新时间: <span id="update-time">{{ update_time }}</span>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // 初始化图表
            function initCharts() {
                // 小时分布折线图
                const hourChart = echarts.init(document.getElementById('hour-chart'));
                hourChart.setOption({
                    title: {
                        text: '按小时投诉量',
                        left: 'center',
                        textStyle: {
                            fontSize: 14,
                            fontWeight: 'normal'
                        }
                    },
                    tooltip: {
                        trigger: 'axis',
                        formatter: '{b}时: {c} 起投诉'
                    },
                    grid: {
                        left: '3%',
                        right: '3%',
                        bottom: '15%',
                        top: '20%',
                        containLabel: true
                    },
                    xAxis: {
                        type: 'category',
                        data: {{ hours_json | safe }},
                        axisLine: {
                            lineStyle: {
                                color: '#ccc'
                            }
                        },
                        axisLabel: {
                            interval: 1,
                            fontSize: 10
                        }
                    },
                    yAxis: {
                        type: 'value',
                        axisLine: {
                            show: false
                        },
                        splitLine: {
                            lineStyle: {
                                color: '#f0f0f0'
                            }
                        }
                    },
                    series: [{
                        name: '投诉量',
                        type: 'line',
                        smooth: true,
                        symbol: 'circle',
                        symbolSize: 8,
                        data: {{ hour_values_json | safe }},
                        lineStyle: {
                            width: 3,
                            color: '#3498db'
                        },
                        itemStyle: {
                            color: '#3498db'
                        },
                        areaStyle: {
                            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                                {offset: 0, color: 'rgba(52, 152, 219, 0.3)'},
                                {offset: 1, color: 'rgba(52, 152, 219, 0.05)'}
                            ])
                        }
                    }]
                });
                
                // 星期分布柱状图
                const dayChart = echarts.init(document.getElementById('day-chart'));
                dayChart.setOption({
                    title: {
                        text: '按星期投诉量',
                        left: 'center',
                        textStyle: {
                            fontSize: 14,
                            fontWeight: 'normal'
                        }
                    },
                    tooltip: {
                        trigger: 'axis',
                        formatter: '{b}: {c} 起投诉'
                    },
                    grid: {
                        left: '3%',
                        right: '3%',
                        bottom: '15%',
                        top: '20%',
                        containLabel: true
                    },
                    xAxis: {
                        type: 'category',
                        data: {{ day_names_json | safe }},
                        axisLine: {
                            lineStyle: {
                                color: '#ccc'
                            }
                        },
                        axisLabel: {
                            fontSize: 11
                        }
                    },
                    yAxis: {
                        type: 'value',
                        axisLine: {
                            show: false
                        },
                        splitLine: {
                            lineStyle: {
                                color: '#f0f0f0'
                            }
                        }
                    },
                    series: [{
                        name: '投诉量',
                        type: 'bar',
                        barWidth: '60%',
                        data: {{ day_values_json | safe }},
                        itemStyle: {
                            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                                {offset: 0, color: '#2ecc71'},
                                {offset: 1, color: '#3498db'}
                            ])
                        }
                    }]
                });
                
                // 响应窗口大小变化
                window.addEventListener('resize', function() {
                    hourChart.resize();
                    dayChart.resize();
                });
            }
            
            // 添加折叠/展开功能
            const statsContainer = document.getElementById('stats-container');
            const toggleBtn = document.getElementById('toggle-stats');
            
            // 初始状态为展开
            let isCollapsed = false;
            
            function togglePanel() {
                isCollapsed = !isCollapsed;
                statsContainer.classList.toggle('collapsed', isCollapsed);
                
                if (isCollapsed) {
                    toggleBtn.title = "展开统计面板";
                } else {
                    toggleBtn.title = "折叠统计面板";
                }
            }
            
            toggleBtn.addEventListener('click', togglePanel);
            
            // 初始化图表
            document.addEventListener('DOMContentLoaded', function() {
                initCharts();
                
                // 添加进度条动画
                document.querySelectorAll('.progress').forEach(progress => {
                    const width = progress.style.width;
                    progress.style.width = '0';
                    setTimeout(() => {
                        progress.style.width = width;
                    }, 300);
                });
            });
        </script>
    </body>
    </html>
    """
    
    # 使用 Jinja2 模板渲染
    template = Template(template_str)
    return template.render(context)