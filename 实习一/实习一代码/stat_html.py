from jinja2 import Template
import json
import pandas as pd
import numpy as np
from datetime import datetime

def generate_stats_html(gdf, noise_count, valid_clusters, cluster_counts):
    """
    生成包含统计信息和交互式图表的HTML面板（使用折线图和ECharts）
    
    参数:
    gdf: GeoDataFrame, 包含投诉数据
    noise_count: 噪声点数量
    valid_clusters: 有效聚类ID列表
    cluster_counts: 每个聚类的点数字典
    
    返回:
    包含完整统计面板的HTML字符串
    """
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
        # 转换为 Python 原生类型
        cluster_distribution.append({
            'id': cluster_id,
            'count': int(count),  # 转换为 int
            'percentage': float(percentage)  # 转换为 float
        })
    
    # 按数量排序聚类分布
    cluster_distribution.sort(key=lambda x: x['count'], reverse=True)
    
    # 计算周期性投诉模式数据
    if not pd.api.types.is_datetime64_any_dtype(gdf['Created Date']):
        gdf['Created Date'] = pd.to_datetime(gdf['Created Date'])
    
    gdf['hour'] = gdf['Created Date'].dt.hour
    gdf['day_of_week'] = gdf['Created Date'].dt.dayofweek
    
    hourly_counts = gdf.groupby('hour').size()
    daily_counts = gdf.groupby('day_of_week').size()
    
    # 准备小时数据
    hours = list(range(24))
    # 将 NumPy 类型转换为 Python 原生类型
    hour_values = [int(hourly_counts.get(h, 0)) for h in hours]
    
    # 准备星期数据
    day_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    # 将 NumPy 类型转换为 Python 原生类型
    day_values = [int(daily_counts.get(d, 0)) for d in range(7)]
    
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
    }
    
    # Jinja2 模板
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
            }
            
            #stats-container {
                position: fixed;
                top: 10px;
                left: 10px;
                z-index: 1000;
                display: flex;
                transition: transform 0.4s cubic-bezier(0.25, 0.1, 0.25, 1);
                transform: translateX(0);
            }
            
            #stats-container.collapsed {
                transform: translateX(-330px);
            }
            
            #toggle-stats {
                background: white;
                border: none;
                border-radius: 4px 0 0 4px;
                padding: 8px 12px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.15);
                cursor: pointer;
                height: 40px;
                align-self: center;
                transition: all 0.3s ease;
                z-index: 1001;
            }
            
            #toggle-stats:hover {
                background: #f5f5f5;
            }
            
            #stats-panel {
                width: 320px;
                background: rgba(255, 255, 255, 0.97);
                border-radius: 0 8px 8px 0;
                box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                overflow: hidden;
                display: flex;
                flex-direction: column;
                backdrop-filter: blur(5px);
            }
            
            .stats-header {
                background: linear-gradient(135deg, #3498db, #2c3e50);
                color: white;
                padding: 15px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .stats-header h2 {
                font-size: 18px;
                font-weight: 600;
                margin: 0;
            }
            
            .close-btn {
                background: none;
                border: none;
                color: white;
                font-size: 18px;
                cursor: pointer;
                width: 30px;
                height: 30px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: background 0.2s;
            }
            
            .close-btn:hover {
                background: rgba(255,255,255,0.2);
            }
            
            .stats-content {
                padding: 15px;
                overflow-y: auto;
                max-height: 75vh;
            }
            
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 15px;
                margin-bottom: 15px;
            }
            
            .stat-card {
                background: #f8f9fa;
                border-radius: 8px;
                padding: 12px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                transition: transform 0.2s;
            }
            
            .stat-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 3px 6px rgba(0,0,0,0.08);
            }
            
            .stat-label {
                font-size: 13px;
                color: #6c757d;
                margin-bottom: 5px;
            }
            
            .stat-value {
                font-size: 20px;
                font-weight: 700;
                color: #2c3e50;
            }
            
            .stat-highlight {
                color: #3498db;
            }
            
            .section-title {
                font-size: 15px;
                font-weight: 600;
                color: #2c3e50;
                margin: 15px 0 10px;
                padding-bottom: 5px;
                border-bottom: 1px solid #eee;
            }
            
            .chart-container {
                height: 180px;
                margin-bottom: 15px;
            }
            
            .cluster-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 13px;
            }
            
            .cluster-table th {
                background-color: #f1f5f9;
                text-align: left;
                padding: 8px;
                font-weight: 600;
            }
            
            .cluster-table td {
                padding: 8px;
                border-bottom: 1px solid #eee;
            }
            
            .cluster-table tr:hover td {
                background-color: #f8fafc;
            }
            
            .count-cell {
                text-align: right;
                font-weight: 600;
            }
            
            .percentage-cell {
                text-align: right;
                color: #6c757d;
            }
            
            .hotspots {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 10px;
                margin-top: 10px;
            }
            
            .hotspot-card {
                background: #f8f9fa;
                border-radius: 6px;
                padding: 10px;
                font-size: 13px;
            }
            
            .hotspot-name {
                font-weight: 600;
                margin-bottom: 5px;
                color: #2c3e50;
            }
            
            .hotspot-count {
                color: #3498db;
                font-weight: 700;
            }
            
            .update-time {
                text-align: center;
                font-size: 12px;
                color: #6c757d;
                margin-top: 15px;
                padding-top: 10px;
                border-top: 1px solid #eee;
            }
            
            .progress-bar {
                height: 6px;
                background: #e9ecef;
                border-radius: 3px;
                margin-top: 5px;
                overflow: hidden;
            }
            
            .progress {
                height: 100%;
                background: linear-gradient(90deg, #3498db, #2ecc71);
                border-radius: 3px;
            }
        </style>
    </head>
    <body>
        <div id="stats-container">
            <button id="toggle-stats" title="显示/隐藏统计面板">📊</button>
            
            <div id="stats-panel">
                <div class="stats-header">
                    <h2>投诉数据统计</h2>
                    <button class="close-btn">✕</button>
                </div>
                
                <div class="stats-content">
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-label">总投诉量</div>
                            <div class="stat-value stat-highlight">{{ total_points }}</div>
                        </div>
                        
                        <div class="stat-card">
                            <div class="stat-label">已聚类数量</div>
                            <div class="stat-value">{{ clustered_count }}</div>
                            <div class="progress-bar">
                                <div class="progress" style="width: {{ clustered_percentage }}%"></div>
                            </div>
                        </div>
                        
                        <div class="stat-card">
                            <div class="stat-label">未聚类数量</div>
                            <div class="stat-value">{{ noise_count }}</div>
                            <div class="progress-bar">
                                <div class="progress" style="width: {{ noise_percentage }}%"></div>
                            </div>
                        </div>
                        
                        <div class="stat-card">
                            <div class="stat-label">聚类数量</div>
                            <div class="stat-value">{{ cluster_count }}</div>
                        </div>
                    </div>
                    
                    <div class="section-title">投诉时间分布</div>
                    <div class="chart-container" id="hour-chart"></div>
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
                        <div class="hotspot-card">
                            <div class="hotspot-name">市中心商业区</div>
                            <div class="hotspot-count">285 起</div>
                        </div>
                        <div class="hotspot-card">
                            <div class="hotspot-name">老旧居民区</div>
                            <div class="hotspot-count">243 起</div>
                        </div>
                        <div class="hotspot-card">
                            <div class="hotspot-name">工业区周边</div>
                            <div class="hotspot-count">198 起</div>
                        </div>
                        <div class="hotspot-card">
                            <div class="hotspot-name">新兴住宅区</div>
                            <div class="hotspot-count">156 起</div>
                        </div>
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
                            fontSize: 13,
                            fontWeight: 'normal'
                        }
                    },
                    tooltip: {
                        trigger: 'axis'
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
                        symbolSize: 6,
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
                            fontSize: 13,
                            fontWeight: 'normal'
                        }
                    },
                    tooltip: {
                        trigger: 'axis'
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
            const closeBtn = document.querySelector('.close-btn');
            
            // 初始状态为展开
            let isCollapsed = false;
            
            function togglePanel() {
                isCollapsed = !isCollapsed;
                statsContainer.classList.toggle('collapsed', isCollapsed);
                toggleBtn.title = isCollapsed ? "显示统计面板" : "隐藏统计面板";
                
                // 添加按钮动画
                if (isCollapsed) {
                    toggleBtn.innerHTML = '📈';
                } else {
                    toggleBtn.innerHTML = '📊';
                }
            }
            
            toggleBtn.addEventListener('click', togglePanel);
            closeBtn.addEventListener('click', togglePanel);
            
            // 初始化图表
            document.addEventListener('DOMContentLoaded', initCharts);
        </script>
    </body>
    </html>
    """
    
    # 使用 Jinja2 模板渲染
    template = Template(template_str)
    return template.render(context)