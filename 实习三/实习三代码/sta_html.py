from jinja2 import Template
import json
from datetime import datetime

def generate_road_network_html():
    """
    生成路网密度和可达性分析的HTML统计面板（左侧定位+折叠功能）
    """
    # 数据准备
    context = {
        # 全局分析 - 路网密度统计
        'total_grids': 4410,
        'valid_grids': 1450,
        'empty_grids': 2960,
        'empty_ratio': 67.1,
        
        'density_min': 0.02,
        'density_max': 87.21,
        'density_mean': 20.24,
        'density_median': 20.07,
        'density_std': 8.97,
        
        # 密度等级分布
        'density_levels': [
            {'level': '低(5-10)', 'count': 81, 'percentage': 5.6},
            {'level': '中(10-15)', 'count': 175, 'percentage': 12.1},
            {'level': '高(15-20)', 'count': 376, 'percentage': 25.9},
            {'level': '极高(>20)', 'count': 733, 'percentage': 50.6}
        ],
        
        # 节点连通性分析
        'total_nodes': 4619,
        'total_edges': 9900,
        'avg_degree': 4.29,
        'intersection_ratio': 97.2,
        
        # 局部分析 - Manhattan可达性
        'accessibility_data': {
            'hospital': {'total': 16, 'accessible': 13, 'rate': 81.2},
            'school': {'total': 279, 'accessible': 219, 'rate': 78.5},
            'park': {'total': 26, 'accessible': 21, 'rate': 80.8}
        },
        
        'update_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 准备图表数据
    context['density_level_names'] = json.dumps([level['level'] for level in context['density_levels']])
    context['density_level_counts'] = json.dumps([level['count'] for level in context['density_levels']])
    
    # POI类型数据
    poi_types = ['医院', '学校', '公园']
    context['poi_types'] = json.dumps(poi_types)
    context['poi_rates'] = json.dumps([
        context['accessibility_data']['hospital']['rate'],
        context['accessibility_data']['school']['rate'],
        context['accessibility_data']['park']['rate']
    ])
    
    # 分位数数据
    quantiles = [9.07, 15.49, 20.07, 25.14, 30.38]
    context['quantile_labels'] = json.dumps(['10%', '25%', '50%', '75%', '90%'])
    context['quantile_values'] = json.dumps(quantiles)

    # HTML模板 - 左侧定位 + 折叠功能
    template_str = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>纽约路网密度与可达性分析</title>
        <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
        <style>
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }
          
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: #f8f9fa;
                margin: 0;
                padding: 0;
                min-height: 100vh;
            }
          
            /* 统计面板容器 - 左侧定位 */
            #stats-container {
                position: fixed;
                top: 20px;
                left: 20px;
                z-index: 1000;
                display: flex;
                transition: transform 0.4s cubic-bezier(0.25, 0.1, 0.25, 1);
                transform: translateX(0);
            }
          
            #stats-container.collapsed {
                transform: translateX(-420px);
            }
          
            /* 统计面板主体 */
            #stats-panel {
                width: 420px;
                background: rgba(255, 255, 255, 0.98);
                border-radius: 12px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.15);
                overflow: hidden;
                display: flex;
                flex-direction: column;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                max-height: 95vh;
                overflow-y: auto;
            }
          
            /* 折叠按钮 */
            #toggle-stats {
                position: absolute;
                top: 15px;
                right: -50px;
                width: 40px;
                height: 40px;
                background: rgba(255, 255, 255, 0.95);
                border: none;
                border-radius: 50%;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                z-index: 1001;
                cursor: pointer;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 18px;
                font-weight: bold;
                color: #2c3e50;
            }
          
            #toggle-stats:hover {
                background: #ffffff;
                transform: scale(1.1);
                box-shadow: 0 6px 20px rgba(0,0,0,0.25);
            }
          
            .header {
                background: linear-gradient(135deg, #2c3e50, #3498db);
                color: white;
                padding: 20px;
                text-align: center;
            }
          
            .header h1 {
                font-size: 20px;
                margin-bottom: 5px;
                font-weight: 600;
            }
          
            .header p {
                font-size: 12px;
                opacity: 0.9;
            }
          
            .content {
                padding: 20px;
                flex: 1;
                overflow-y: auto;
            }
          
            .section {
                background: white;
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 15px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                border-left: 4px solid #3498db;
            }
          
            .section-title {
                font-size: 14px;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 15px;
                display: flex;
                align-items: center;
            }
          
            .section-title:before {
                content: '📊';
                margin-right: 8px;
                font-size: 16px;
            }
          
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 10px;
                margin-bottom: 15px;
            }
          
            .stat-card {
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                padding: 12px;
                border-radius: 8px;
                text-align: center;
                transition: transform 0.3s ease;
            }
          
            .stat-card:hover {
                transform: translateY(-2px);
            }
          
            .stat-value {
                font-size: 18px;
                font-weight: 700;
                margin: 5px 0;
            }
          
            .stat-label {
                font-size: 11px;
                opacity: 0.9;
            }
          
            .chart-container {
                height: 200px;
                margin: 15px 0;
            }
          
            .density-levels {
                display: grid;
                gap: 8px;
                margin-top: 10px;
            }
          
            .level-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 8px;
                background: #f8f9fa;
                border-radius: 6px;
                font-size: 12px;
            }
          
            .level-name {
                font-weight: 600;
                color: #2c3e50;
            }
          
            .level-stats {
                text-align: right;
            }
          
            .level-count {
                font-weight: 700;
                color: #3498db;
            }
          
            .level-percentage {
                font-size: 10px;
                color: #6c757d;
            }
          
            .accessibility-cards {
                display: grid;
                gap: 10px;
                margin-top: 15px;
            }
          
            .poi-card {
                background: white;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                padding: 12px;
                text-align: center;
                transition: all 0.3s ease;
            }
          
            .poi-card:hover {
                border-color: #3498db;
                transform: translateY(-2px);
            }
          
            .poi-icon {
                font-size: 24px;
                margin-bottom: 5px;
            }
          
            .poi-name {
                font-weight: 600;
                color: #2c3e50;
                font-size: 12px;
                margin-bottom: 5px;
            }
          
            .poi-stats {
                font-size: 16px;
                font-weight: 700;
                color: #27ae60;
            }
          
            .poi-detail {
                font-size: 10px;
                color: #6c757d;
                margin-top: 3px;
            }
          
            .progress-bar {
                height: 6px;
                background: #e9ecef;
                border-radius: 3px;
                margin: 8px 0;
                overflow: hidden;
            }
          
            .progress {
                height: 100%;
                background: linear-gradient(90deg, #3498db, #2ecc71);
                border-radius: 3px;
                transition: width 0.5s ease;
            }
          
            .update-time {
                text-align: center;
                color: #6c757d;
                font-size: 10px;
                margin-top: 15px;
                padding-top: 10px;
                border-top: 1px solid #e9ecef;
            }
          
            /* 滚动条样式 */
            #stats-panel::-webkit-scrollbar {
                width: 6px;
            }
          
            #stats-panel::-webkit-scrollbar-track {
                background: #f1f1f1;
                border-radius: 3px;
            }
          
            #stats-panel::-webkit-scrollbar-thumb {
                background: #c1c1c1;
                border-radius: 3px;
            }
          
            #stats-panel::-webkit-scrollbar-thumb:hover {
                background: #a8a8a8;
            }
        </style>
    </head>
    <body>
    
        <!-- 统计面板 -->
        <div id="stats-container">
            <div id="stats-panel">
                <button id="toggle-stats" title="折叠统计面板">◀</button>
                
                <div class="header">
                    <h1>🏙️ 纽约路网分析</h1>
                    <p>从全局到局部的综合分析</p>
                </div>
              
                <div class="content">
                    <!-- 全局分析 - 路网密度概览 -->
                    <div class="section">
                        <div class="section-title">🌐 全局路网密度</div>
                      
                        <div class="stats-grid">
                            <div class="stat-card">
                                <div class="stat-label">总网格</div>
                                <div class="stat-value">{{ total_grids }}</div>
                            </div>
                          
                            <div class="stat-card">
                                <div class="stat-label">有效网格</div>
                                <div class="stat-value">{{ valid_grids }}</div>
                                <div class="stat-label">{{ (valid_grids/total_grids*100)|round(1) }}%</div>
                            </div>
                          
                            <div class="stat-card">
                                <div class="stat-label">平均密度</div>
                                <div class="stat-value">{{ density_mean }}</div>
                                <div class="stat-label">km/km²</div>
                            </div>
                          
                            <div class="stat-card">
                                <div class="stat-label">标准差</div>
                                <div class="stat-value">{{ density_std }}</div>
                            </div>
                        </div>
                      
                        <div class="section-title" style="font-size: 12px; margin-bottom: 10px;">📈 密度等级分布</div>
                        <div class="chart-container" id="densityChart"></div>
                      
                        <div class="density-levels">
                            {% for level in density_levels %}
                            <div class="level-item">
                                <span class="level-name">{{ level.level }}</span>
                                <div class="level-stats">
                                    <div class="level-count">{{ level.count }}</div>
                                    <div class="level-percentage">{{ level.percentage }}%</div>
                                </div>
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                  
                    <!-- 节点连通性分析 -->
                    <div class="section">
                        <div class="section-title">🔗 节点连通性</div>
                      
                        <div class="stats-grid">
                            <div class="stat-card">
                                <div class="stat-label">节点数</div>
                                <div class="stat-value">{{ total_nodes }}</div>
                            </div>
                          
                            <div class="stat-card">
                                <div class="stat-label">边数</div>
                                <div class="stat-value">{{ total_edges }}</div>
                            </div>
                          
                            <div class="stat-card">
                                <div class="stat-label">平均度数</div>
                                <div class="stat-value">{{ avg_degree }}</div>
                            </div>
                          
                            <div class="stat-card">
                                <div class="stat-label">交叉口</div>
                                <div class="stat-value">{{ intersection_ratio }}%</div>
                            </div>
                        </div>
                      
                        <div class="chart-container" id="quantileChart"></div>
                    </div>
                  
                    <!-- 局部分析 - Manhattan可达性 -->
                    <div class="section">
                        <div class="section-title">🏃 15分钟可达性</div>
                      
                        <div class="accessibility-cards">
                            <div class="poi-card">
                                <div class="poi-icon">🏥</div>
                                <div class="poi-name">医院</div>
                                <div class="poi-stats">{{ accessibility_data.hospital.rate }}%</div>
                                <div class="poi-detail">{{ accessibility_data.hospital.accessible }}/{{ accessibility_data.hospital.total }}</div>
                                <div class="progress-bar">
                                    <div class="progress" style="width: {{ accessibility_data.hospital.rate }}%"></div>
                                </div>
                            </div>
                          
                            <div class="poi-card">
                                <div class="poi-icon">🏫</div>
                                <div class="poi-name">学校</div>
                                <div class="poi-stats">{{ accessibility_data.school.rate }}%</div>
                                <div class="poi-detail">{{ accessibility_data.school.accessible }}/{{ accessibility_data.school.total }}</div>
                                <div class="progress-bar">
                                    <div class="progress" style="width: {{ accessibility_data.school.rate }}%"></div>
                                </div>
                            </div>
                          
                            <div class="poi-card">
                                <div class="poi-icon">🌳</div>
                                <div class="poi-name">公园</div>
                                <div class="poi-stats">{{ accessibility_data.park.rate }}%</div>
                                <div class="poi-detail">{{ accessibility_data.park.accessible }}/{{ accessibility_data.park.total }}</div>
                                <div class="progress-bar">
                                    <div class="progress" style="width: {{ accessibility_data.park.rate }}%"></div>
                                </div>
                            </div>
                        </div>
                      
                        <div class="chart-container" id="accessibilityChart"></div>
                    </div>
                  
                    <div class="update-time">
                        更新时间: {{ update_time }}
                    </div>
                </div>
            </div>
        </div>
      
        <script>
            // 初始化所有图表
            function initCharts() {
                // 路网密度分布图
                const densityChart = echarts.init(document.getElementById('densityChart'));
                densityChart.setOption({
                    tooltip: {
                        trigger: 'axis',
                        formatter: '{b}: {c}个网格'
                    },
                    grid: {
                        left: '3%',
                        right: '3%',
                        bottom: '15%',
                        top: '10%',
                        containLabel: true
                    },
                    xAxis: {
                        type: 'category',
                        data: {{ density_level_names | safe }},
                        axisLabel: {
                            rotate: 45,
                            fontSize: 10
                        }
                    },
                    yAxis: {
                        type: 'value',
                        name: '数量',
                        nameTextStyle: { fontSize: 10 },
                        axisLabel: { fontSize: 9 }
                    },
                    series: [{
                        name: '网格数量',
                        type: 'bar',
                        data: {{ density_level_counts | safe }},
                        itemStyle: {
                            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                                {offset: 0, color: '#3498db'},
                                {offset: 1, color: '#2ecc71'}
                            ])
                        },
                        barWidth: '60%'
                    }]
                });
              
                // 分位数图表
                const quantileChart = echarts.init(document.getElementById('quantileChart'));
                quantileChart.setOption({
                    tooltip: {
                        trigger: 'axis',
                        formatter: '{b}: {c} km/km²'
                    },
                    grid: {
                        left: '3%',
                        right: '3%',
                        bottom: '15%',
                        top: '10%',
                        containLabel: true
                    },
                    xAxis: {
                        type: 'category',
                        data: {{ quantile_labels | safe }},
                        axisLabel: { fontSize: 10 }
                    },
                    yAxis: {
                        type: 'value',
                        name: '密度',
                        nameTextStyle: { fontSize: 10 },
                        axisLabel: { fontSize: 9 }
                    },
                    series: [{
                        name: '密度值',
                        type: 'line',
                        smooth: true,
                        data: {{ quantile_values | safe }},
                        lineStyle: { width: 2, color: '#e74c3c' },
                        itemStyle: { color: '#e74c3c' },
                        symbolSize: 6
                    }]
                });
              
                // 可达性分析图
                const accessibilityChart = echarts.init(document.getElementById('accessibilityChart'));
                accessibilityChart.setOption({
                    tooltip: {
                        trigger: 'axis',
                        formatter: '{b}: {c}%'
                    },
                    grid: {
                        left: '3%',
                        inRight: '3%',
                        bottom: '15%',
                        top: '10%',
                        containLabel: true
                    },
                    xAxis: {
                        type: 'category',
                        data: {{ poi_types | safe }},
                        axisLabel: { fontSize: 10 }
                    },
                    yAxis: {
                        type: 'value',
                        name: '可达率(%)',
                        max: 100,
                        nameTextStyle: { fontSize: 10 },
                        axisLabel: { fontSize: 9 }
                    },
                    series: [{
                        name: '可达率',
                        type: 'bar',
                        data: {{ poi_rates | safe }},
                        itemStyle: {
                            color: function(params) {
                                const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'];
                                return colors[params.dataIndex];
                            }
                        },
                        barWidth: '60%'
                    }]
                });
              
                // 响应窗口调整
                window.addEventListener('resize', function() {
                    densityChart.resize();
                    quantileChart.resize();
                    accessibilityChart.resize();
                });
            }
          
            // 折叠/展开功能
            const statsContainer = document.getElementById('stats-container');
            const toggleBtn = document.getElementById('toggle-stats');
            let isCollapsed = false;
          
            function togglePanel() {
                isCollapsed = !isCollapsed;
                statsContainer.classList.toggle('collapsed', isCollapsed);
                toggleBtn.textContent = isCollapsed ? '▶' : '◀';
                toggleBtn.title = isCollapsed ? '展开统计面板' : '折叠统计面板';
            }
          
            toggleBtn.addEventListener('click', togglePanel);
          
            // 页面加载完成后初始化
            document.addEventListener('DOMContentLoaded', function() {
                initCharts();
              
                // 添加进度条动画
                setTimeout(() => {
                    document.querySelectorAll('.progress').forEach(progress => {
                        const finalWidth = progress.style.width;
                        progress.style.width = '0';
                        setTimeout(() => {
                            progress.style.width = finalWidth;
                        }, 100);
                    });
                }, 500);
            });
        </script>
    </body>
    </html>
    """
  
    template = Template(template_str)
    return template.render(context)

if __name__ == '__main__':
    # 生成HTML并保存
    html_content = generate_road_network_html()
    with open('new_york_road_network_analysis.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
print("纽约路网密度与可达性分析面板已生成！支持左侧定位和折叠功能！")