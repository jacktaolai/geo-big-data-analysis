# 数据有偏性分析 - 评估数据质量并分析潜在偏倚
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap, MarkerCluster
from textblob import TextBlob
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import os

# 设置中文字体支持（解决matplotlib中文显示问题）
plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]  # 英文字体为新罗马，中文字体为宋体
plt.rcParams["font.serif"] = ["Times New Roman", "SimSun"]  # 衬线字体
plt.rcParams["font.sans-serif"] = ["Times New Roman", "SimSun", "Arial", "SimHei"]  # 无衬线字体，与Latex相关
plt.rcParams["mathtext.fontset"] = "custom"

plt.rcParams['axes.unicode_minus'] = False

# 定义数据路径
DATA_PATH = "D:/必须用电脑解决的作业/地理大数据分析/实习四/实习四数据/yelp/"
BUSINESS_FILE = os.path.join(DATA_PATH, "yelp_academic_dataset_business.json")
REVIEW_FILE = os.path.join(DATA_PATH, "yelp_academic_dataset_review.json")
USER_FILE = os.path.join(DATA_PATH, "yelp_academic_dataset_user.json")
TIP_FILE = os.path.join(DATA_PATH, "yelp_academic_dataset_tip.json")
CHECKIN_FILE = os.path.join(DATA_PATH, "yelp_academic_dataset_checkin.json")

# 目标城市和研究参数
TARGET_CITY = "New Orleans"
SAVE_INTERMEDIATE = True  # 是否保存中间结果
INTERMEDIATE_PATH = "./intermediate_data/"
os.makedirs(INTERMEDIATE_PATH, exist_ok=True)

# 定义餐厅相关类别关键词
RESTAURANT_KEYWORDS = ['restaurant', 'food', 'cafe', 'coffee', 'bar', 'pub', 'diner', 'grill', 'steakhouse', 
                       'pizza', 'italian', 'chinese', 'mexican', 'japanese', 'indian', 'thai', 'vietnamese',
                       'breakfast', 'brunch', 'lunch', 'dinner', 'bakery', 'deli', 'bistro', 'tavern']
def analyze_data_bias(business_df, review_df, merged_df):
    """
    分析数据的潜在偏倚，为后续分析结果的可信度提供支持
    """
    print("=" * 60)
    print("数据有偏性分析")
    print("=" * 60)
    
    bias_results = {}
    
    # 1. 数据完整性分析
    print("\n1. 数据完整性分析:")
    print("-" * 40)
    
    # 检查business数据完整性
    business_missing = {}
    for col in ['name', 'city', 'state', 'latitude', 'longitude', 'stars', 'review_count', 'categories']:
        missing_percent = (business_df[col].isna().sum() / len(business_df)) * 100
        business_missing[col] = missing_percent
        print(f"  {col} 缺失率: {missing_percent:.2f}%")
    
    bias_results['business_missing_rates'] = business_missing
    
    # 检查review数据完整性
    review_missing = {}
    for col in ['text', 'stars', 'date', 'useful', 'funny', 'cool']:
        if col in review_df.columns:
            missing_percent = (review_df[col].isna().sum() / len(review_df)) * 100
            review_missing[col] = missing_percent
            print(f"  {col} 缺失率: {missing_percent:.2f}%")
    
    bias_results['review_missing_rates'] = review_missing
    
    # 2. 时间分布偏倚分析
    print("\n2. 时间分布偏倚分析:")
    print("-" * 40)
    
    if 'date' in merged_df.columns:
        # 提取年份和月份
        merged_df['review_year'] = pd.to_datetime(merged_df['date']).dt.year
        merged_df['review_month'] = pd.to_datetime(merged_df['date']).dt.month
        
        # 按年份统计评论数量
        year_distribution = merged_df['review_year'].value_counts().sort_index()
        
        print("评论按年份分布:")
        for year, count in year_distribution.items():
            percentage = (count / len(merged_df)) * 100
            print(f"  {year}: {count} 条 ({percentage:.2f}%)")
        
        bias_results['year_distribution'] = year_distribution.to_dict()
        
        # 计算时间跨度
        if len(year_distribution) > 0:
            min_year = min(year_distribution.index)
            max_year = max(year_distribution.index)
            print(f"\n  数据时间跨度: {min_year} 年 - {max_year} 年")
            print(f"  覆盖年份数: {len(year_distribution)} 年")
            
            # 检查是否有集中趋势
            max_year_count = year_distribution.max()
            max_year = year_distribution.idxmax()
            max_year_percentage = (max_year_count / len(merged_df)) * 100
            print(f"  最集中年份: {max_year} 年 ({max_year_percentage:.2f}% 的评论)")
            
            if max_year_percentage > 50:
                print(f"  ⚠️ 警告: {max_year} 年的评论占比超过50%，可能存在时间偏倚")
        
        # 可视化时间分布
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        year_distribution.plot(kind='bar', color='skyblue', alpha=0.7)
        plt.title('评论按年份分布')
        plt.xlabel('年份')
        plt.ylabel('评论数量')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        # 按月份统计（所有年份合计）
        month_distribution = merged_df['review_month'].value_counts().sort_index()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_distribution.index = month_names[:len(month_distribution)]
        month_distribution.plot(kind='bar', color='lightcoral', alpha=0.7)
        plt.title('评论按月分布')
        plt.xlabel('月份')
        plt.ylabel('评论数量')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        bias_results['month_distribution'] = month_distribution.to_dict()
    
    # 3. 用户行为偏倚分析
    print("\n3. 用户行为偏倚分析:")
    print("-" * 40)
    
    if 'user_id' in merged_df.columns:
        # 计算每个用户的评论数量
        user_review_counts = merged_df['user_id'].value_counts()
        
        print(f"  总用户数: {len(user_review_counts)}")
        print(f"  总评论数: {len(merged_df)}")
        print(f"  平均每个用户的评论数: {len(merged_df) / len(user_review_counts):.2f}")
        
        # 计算帕累托分布
        total_reviews = len(merged_df)
        sorted_users = user_review_counts.sort_values(ascending=False)
        cumulative_percentage = sorted_users.cumsum() / total_reviews * 100
        
        # 找到20%的用户贡献了多少评论
        threshold_20 = int(len(user_review_counts) * 0.2)
        top_20_users_reviews = sorted_users.iloc[:threshold_20].sum()
        top_20_percentage = (top_20_users_reviews / total_reviews) * 100
        
        print(f"  前20%的用户贡献了 {top_20_percentage:.2f}% 的评论")
        
        # 找到最活跃的用户
        top_10_users = user_review_counts.head(10)
        print("\n  最活跃的10个用户:")
        for i, (user_id, count) in enumerate(top_10_users.items(), 1):
            percentage = (count / total_reviews) * 100
            print(f"    {i}. 用户 {user_id[:10]}...: {count} 条 ({percentage:.2f}%)")
        
        bias_results['user_distribution'] = {
            'total_users': len(user_review_counts),
            'avg_reviews_per_user': len(merged_df) / len(user_review_counts),
            'top_20_percentage': top_20_percentage
        }
        
        # 可视化用户分布
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        # 绘制用户评论数量的直方图（只显示前100名）
        user_review_counts_top100 = user_review_counts.head(100)
        plt.hist(user_review_counts_top100.values, bins=30, color='lightgreen', alpha=0.7, edgecolor='black')
        plt.title('用户评论数量分布 (前100名用户)')
        plt.xlabel('评论数量')
        plt.ylabel('用户数量')
        
        plt.subplot(1, 2, 2)
        # 绘制帕累托图
        plt.plot(range(1, len(cumulative_percentage) + 1), cumulative_percentage.values, 
                'b-', linewidth=2, label='累积百分比')
        plt.axhline(y=80, color='r', linestyle='--', alpha=0.5, label='80% 阈值')
        plt.axvline(x=threshold_20, color='g', linestyle='--', alpha=0.5, label='20% 用户')
        plt.title('用户评论的帕累托分布')
        plt.xlabel('用户数量 (按评论数排序)')
        plt.ylabel('累积评论百分比 (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # 4. 餐厅覆盖偏倚分析
    print("\n4. 餐厅覆盖偏倚分析:")
    print("-" * 40)
    
    # 计算每个餐厅的评论数量
    business_review_counts = merged_df['business_id'].value_counts()
    
    print(f"  有评论的餐厅数量: {len(business_review_counts)}")
    print(f"  总餐厅数量: {len(business_df)}")
    coverage_rate = (len(business_review_counts) / len(business_df)) * 100
    print(f"  餐厅评论覆盖率: {coverage_rate:.2f}%")
    
    if coverage_rate < 100:
        print(f"  ⚠️ 警告: {len(business_df) - len(business_review_counts)} 家餐厅没有评论数据")
    
    # 分析评论分布是否均匀
    avg_reviews_per_business = len(merged_df) / len(business_review_counts)
    print(f"  平均每个餐厅的评论数: {avg_reviews_per_business:.2f}")
    
    # 检查评论集中度
    top_10_businesses = business_review_counts.head(10)
    top_10_percentage = (top_10_businesses.sum() / len(merged_df)) * 100
    print(f"  评论最多的10家餐厅贡献了 {top_10_percentage:.2f}% 的评论")
    
    # 找到没有评论的餐厅
    businesses_without_reviews = set(business_df['business_id']) - set(business_review_counts.index)
    print(f"  没有评论的餐厅数量: {len(businesses_without_reviews)}")
    
    # 检查没有评论的餐厅是否有特殊特征
    if len(businesses_without_reviews) > 0:
        businesses_no_reviews = business_df[business_df['business_id'].isin(businesses_without_reviews)]
        
        # 检查星级分布
        if 'stars' in businesses_no_reviews.columns:
            avg_stars_no_reviews = businesses_no_reviews['stars'].mean()
            avg_stars_with_reviews = business_df[~business_df['business_id'].isin(businesses_without_reviews)]['stars'].mean()
            print(f"  无评论餐厅平均星级: {avg_stars_no_reviews:.2f}")
            print(f"  有评论餐厅平均星级: {avg_stars_with_reviews:.2f}")
            
            if abs(avg_stars_no_reviews - avg_stars_with_reviews) > 0.5:
                print(f"  ⚠️ 警告: 有评论和无评论餐厅的星级有明显差异")
        
        # 检查review_count字段
        if 'review_count' in businesses_no_reviews.columns:
            avg_review_count_no_reviews = businesses_no_reviews['review_count'].mean()
            print(f"  无评论餐厅的平均review_count字段值: {avg_review_count_no_reviews:.2f}")
            if avg_review_count_no_reviews > 0:
                print(f"  ⚠️ 警告: 无评论餐厅的review_count字段不为零，可能存在数据不一致")
    
    bias_results['business_coverage'] = {
        'total_businesses': len(business_df),
        'businesses_with_reviews': len(business_review_counts),
        'coverage_rate': coverage_rate,
        'avg_reviews_per_business': avg_reviews_per_business,
        'top_10_percentage': top_10_percentage,
        'businesses_without_reviews': len(businesses_without_reviews)
    }
    
    # 可视化餐厅评论分布
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    # 绘制餐厅评论数量的直方图
    plt.hist(business_review_counts.values, bins=50, color='lightblue', alpha=0.7, edgecolor='black')
    plt.title('餐厅评论数量分布')
    plt.xlabel('评论数量')
    plt.ylabel('餐厅数量')
    plt.yscale('log')  # 使用对数坐标，因为分布通常长尾
    
    plt.subplot(1, 2, 2)
    # 绘制餐厅评论数量的累积分布
    sorted_counts = np.sort(business_review_counts.values)[::-1]
    cumulative = np.cumsum(sorted_counts) / np.sum(sorted_counts) * 100
    plt.plot(range(1, len(cumulative) + 1), cumulative, 'r-', linewidth=2)
    plt.axhline(y=80, color='b', linestyle='--', alpha=0.5, label='80% 阈值')
    plt.title('餐厅评论累积分布')
    plt.xlabel('餐厅数量 (按评论数排序)')
    plt.ylabel('累积评论百分比 (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 5. 评分分布偏倚分析
    print("\n5. 评分分布偏倚分析:")
    print("-" * 40)
    
    if 'stars' in merged_df.columns:
        # 分析评论星级分布
        star_distribution = merged_df['stars'].value_counts().sort_index()
        
        print("评论星级分布:")
        for star, count in star_distribution.items():
            percentage = (count / len(merged_df)) * 100
            print(f"  {star} 星: {count} 条 ({percentage:.2f}%)")
        
        # 计算平均星级
        avg_star = merged_df['stars'].mean()
        median_star = merged_df['stars'].median()
        print(f"\n  平均星级: {avg_star:.2f}")
        print(f"  中位数星级: {median_star:.2f}")
        
        # 检查评分偏倚
        positive_bias = (star_distribution.get(5.0, 0) + star_distribution.get(4.0, 0)) / len(merged_df) * 100
        negative_bias = (star_distribution.get(1.0, 0) + star_distribution.get(2.0, 0)) / len(merged_df) * 100
        
        print(f"  积极评价比例 (4-5星): {positive_bias:.2f}%")
        print(f"  消极评价比例 (1-2星): {negative_bias:.2f}%")
        
        if positive_bias > 70:
            print(f"  ⚠️ 警告: 积极评价比例超过70%，可能存在积极偏倚")
        elif negative_bias > 40:
            print(f"  ⚠️ 警告: 消极评价比例超过40%，可能存在消极偏倚")
        
        bias_results['star_distribution'] = star_distribution.to_dict()
        bias_results['rating_bias'] = {
            'avg_star': avg_star,
            'median_star': median_star,
            'positive_bias': positive_bias,
            'negative_bias': negative_bias
        }
        
        # 可视化星级分布
        plt.figure(figsize=(10, 6))
        
        # 创建子图
        ax1 = plt.subplot(2, 1, 1)
        star_distribution.plot(kind='bar', color=['red', 'orangered', 'orange', 'yellowgreen', 'green'], alpha=0.7)
        plt.title('评论星级分布')
        plt.xlabel('星级')
        plt.ylabel('评论数量')
        
        ax2 = plt.subplot(2, 1, 2)
        # 绘制累积分布
        star_cumulative = star_distribution.sort_index().cumsum() / len(merged_df) * 100
        star_cumulative.plot(kind='line', marker='o', color='blue', linewidth=2)
        plt.title('评论星级累积分布')
        plt.xlabel('星级')
        plt.ylabel('累积百分比 (%)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # 6. 地理分布偏倚分析
    print("\n6. 地理分布偏倚分析:")
    print("-" * 40)
    
    if 'latitude' in merged_df.columns and 'longitude' in merged_df.columns:
        # 检查地理坐标覆盖范围
        lat_min, lat_max = merged_df['latitude'].min(), merged_df['latitude'].max()
        lon_min, lon_max = merged_df['longitude'].min(), merged_df['longitude'].max()
        
        print(f"  纬度范围: {lat_min:.4f}° - {lat_max:.4f}°")
        print(f"  经度范围: {lon_min:.4f}° - {lon_max:.4f}°")
        
        # 计算地理中心
        lat_center = merged_df['latitude'].mean()
        lon_center = merged_df['longitude'].mean()
        print(f"  地理中心: ({lat_center:.4f}°, {lon_center:.4f}°)")
        
        # 检查是否有地理聚类
        from scipy.stats import gaussian_kde
        
        # 计算核密度估计
        coords = merged_df[['latitude', 'longitude']].dropna()
        if len(coords) > 10:
            try:
                kde = gaussian_kde(coords.T)
                # 评估密度均匀性（简单方法：计算坐标的标准差）
                lat_std = coords['latitude'].std()
                lon_std = coords['longitude'].std()
                
                print(f"  纬度标准差: {lat_std:.4f}")
                print(f"  经度标准差: {lon_std:.4f}")
                
                # 如果标准差很小，说明数据点很集中
                if lat_std < 0.01 or lon_std < 0.01:
                    print(f"  ⚠️ 警告: 地理坐标分布过于集中，可能存在地理偏倚")
            except:
                print("  无法计算地理分布密度")
        
        bias_results['geographic_distribution'] = {
            'lat_range': [lat_min, lat_max],
            'lon_range': [lon_min, lon_max],
            'center': [lat_center, lon_center]
        }
        
        # 可视化地理分布
        plt.figure(figsize=(10, 8))
        
        plt.scatter(merged_df['longitude'], merged_df['latitude'], 
                   alpha=0.3, s=10, c='blue', marker='.')
        
        # 标记地理中心
        plt.scatter(lon_center, lat_center, s=200, c='red', marker='*', label='地理中心')
        
        plt.title(f'{TARGET_CITY}餐厅评论地理分布')
        plt.xlabel('经度')
        plt.ylabel('纬度')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # 7. 综合偏倚评估
    print("\n7. 综合偏倚评估:")
    print("-" * 40)
    
    # 计算总体偏倚分数（0-10，越高表示偏倚越严重）
    bias_score = 0
    max_score = 0
    warnings = []
    
    # 时间偏倚检查
    if 'year_distribution' in bias_results:
        if max(bias_results.get('rating_bias', {}).get('positive_bias', 0), 
               bias_results.get('rating_bias', {}).get('negative_bias', 0)) > 70:
            bias_score += 2
            warnings.append("评分分布存在明显偏倚")
        max_score += 2
    
    # 用户偏倚检查
    if 'user_distribution' in bias_results:
        if bias_results['user_distribution']['top_20_percentage'] > 80:
            bias_score += 2
            warnings.append("用户评论集中度过高")
        max_score += 2
    
    # 餐厅覆盖偏倚检查
    if 'business_coverage' in bias_results:
        if bias_results['business_coverage']['coverage_rate'] < 70:
            bias_score += 2
            warnings.append("餐厅评论覆盖率不足")
        
        if bias_results['business_coverage']['top_10_percentage'] > 30:
            bias_score += 1
            warnings.append("评论过度集中在少数餐厅")
        max_score += 3
    
    # 数据完整性检查
    if 'review_missing_rates' in bias_results:
        text_missing = bias_results['review_missing_rates'].get('text', 0)
        if text_missing > 10:
            bias_score += 2
            warnings.append(f"评论文本缺失率过高 ({text_missing:.1f}%)")
        max_score += 2
    
    # 计算偏倚程度
    if max_score > 0:
        bias_percentage = (bias_score / max_score) * 100
        
        print(f"  偏倚评估分数: {bias_score}/{max_score} ({bias_percentage:.1f}%)")
        
        if bias_percentage < 30:
            print("  偏倚程度: 低 - 数据质量较好")
        elif bias_percentage < 60:
            print("  偏倚程度: 中 - 数据存在一定偏倚，分析时需要谨慎")
        else:
            print("  偏倚程度: 高 - 数据存在明显偏倚，分析结果需谨慎解释")
        
        if warnings:
            print("\n  主要偏倚警告:")
            for i, warning in enumerate(warnings, 1):
                print(f"    {i}. {warning}")
    
    bias_results['bias_assessment'] = {
        'bias_score': bias_score,
        'max_score': max_score,
        'bias_percentage': bias_percentage if max_score > 0 else 0,
        'warnings': warnings
    }
    
    print("\n" + "=" * 60)
    print("数据有偏性分析完成")
    print("=" * 60)
    
    return bias_results

def save_bias_analysis_results(bias_results, output_path=None):
    """
    保存有偏性分析结果
    """
    if output_path is None:
        output_path = INTERMEDIATE_PATH
    
    # 保存为JSON文件
    import json
    
    output_file = os.path.join(output_path, "data_bias_analysis.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        # 转换numpy数组和pandas对象为Python原生类型
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(bias_results)
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    print(f"有偏性分析结果已保存到: {output_file}")
    
    # 保存摘要报告
    summary_file = os.path.join(output_path, "data_bias_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("数据有偏性分析摘要报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"分析城市: {TARGET_CITY}\n")
        f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 数据概览
        if 'business_coverage' in bias_results:
            f.write("1. 数据概览:\n")
            f.write(f"   餐厅总数: {bias_results['business_coverage']['total_businesses']}\n")
            f.write(f"   有评论的餐厅数: {bias_results['business_coverage']['businesses_with_reviews']}\n")
            f.write(f"   评论覆盖率: {bias_results['business_coverage']['coverage_rate']:.2f}%\n\n")
        
        # 偏倚评估
        if 'bias_assessment' in bias_results:
            f.write("2. 偏倚评估:\n")
            assessment = bias_results['bias_assessment']
            f.write(f"   偏倚分数: {assessment['bias_score']}/{assessment['max_score']}\n")
            f.write(f"   偏倚程度: {assessment['bias_percentage']:.1f}%\n\n")
            
            if assessment['warnings']:
                f.write("3. 主要偏倚警告:\n")
                for warning in assessment['warnings']:
                    f.write(f"   • {warning}\n")
                f.write("\n")
        
        # 建议
        f.write("4. 数据分析建议:\n")
        if 'bias_assessment' in bias_results:
            if bias_results['bias_assessment']['bias_percentage'] < 30:
                f.write("   • 数据质量良好，分析结果可信度较高\n")
                f.write("   • 可以直接进行情感分析和推荐\n")
            elif bias_results['bias_assessment']['bias_percentage'] < 60:
                f.write("   • 数据存在一定偏倚，建议在分析时考虑以下因素:\n")
                f.write("   • 对时间、用户和地理分布进行加权处理\n")
                f.write("   • 在结论中明确说明数据局限性\n")
                f.write("   • 考虑对热门餐厅的结果进行适当调整\n")
            else:
                f.write("   • 数据存在明显偏倚，分析结果需谨慎解释:\n")
                f.write("   • 建议进行数据重采样或加权处理\n")
                f.write("   • 结论中必须详细说明数据偏倚问题\n")
                f.write("   • 考虑补充其他数据源以验证结果\n")
    
    print(f"分析摘要已保存到: {summary_file}")
    
    return output_file, summary_file

def analyze_coordinate_cleaning_impact(business_df, merged_df=None):
    """
    专门分析坐标清洗对数据分析结果的影响
    """
    print("=" * 60)
    print("坐标清洗影响分析")
    print("=" * 60)
    
    impact_results = {
        'removed_percentage': 0,
        'stars_difference': 0,
        'reviews_difference': 0,
        'potential_bias': False,
        'warnings': []
    }
    
    # 1. 检查是否有被移除的数据文件
    removed_file = os.path.join(INTERMEDIATE_PATH, "business_coordinates_removed.csv")
    
    if os.path.exists(removed_file):
        print("\n1. 分析坐标清洗移除的数据:")
        print("-" * 40)
        
        removed_data = pd.read_csv(removed_file)
        total_before_cleaning = len(business_df) + len(removed_data)
        
        if total_before_cleaning > 0:
            removal_percentage = len(removed_data) / total_before_cleaning * 100
            impact_results['removed_percentage'] = removal_percentage
            
            print(f"  坐标清洗前餐厅总数: {total_before_cleaning}")
            print(f"  被移除的餐厅数量: {len(removed_data)}")
            print(f"  移除比例: {removal_percentage:.2f}%")
            
            if removal_percentage > 10:
                impact_results['warnings'].append(f"坐标清洗移除了 {removal_percentage:.1f}% 的数据，比例较高")
                print(f"  ⚠️  警告: 坐标清洗移除了 {removal_percentage:.2f}% 的数据，可能影响分析代表性")
            elif removal_percentage > 5:
                print(f"  ⚠️  注意: 坐标清洗移除了 {removal_percentage:.2f}% 的数据")
            else:
                print(f"  ✓ 坐标清洗移除比例较低 ({removal_percentage:.2f}%)，影响较小")
            
            # 分析被移除餐厅的特征
            if len(removed_data) > 0 and len(business_df) > 0:
                # 星级比较
                if 'stars' in removed_data.columns and 'stars' in business_df.columns:
                    avg_stars_removed = removed_data['stars'].mean()
                    avg_stars_kept = business_df['stars'].mean()
                    stars_diff = abs(avg_stars_removed - avg_stars_kept)
                    impact_results['stars_difference'] = stars_diff
                    
                    print(f"\n  星级分布比较:")
                    print(f"    被移除餐厅平均星级: {avg_stars_removed:.2f}")
                    print(f"    保留餐厅平均星级: {avg_stars_kept:.2f}")
                    print(f"    差异: {stars_diff:.2f}")
                    
                    if stars_diff > 0.5:
                        impact_results['potential_bias'] = True
                        impact_results['warnings'].append(f"被移除餐厅平均星级 ({avg_stars_removed:.2f}) 与保留餐厅 ({avg_stars_kept:.2f}) 差异明显")
                        print(f"    ⚠️  警告: 星级差异明显，可能导致星级分布偏倚")
                    else:
                        print(f"    ✓ 星级差异较小，影响不大")
                
                # 评论数比较
                if 'review_count' in removed_data.columns and 'review_count' in business_df.columns:
                    avg_reviews_removed = removed_data['review_count'].mean()
                    avg_reviews_kept = business_df['review_count'].mean()
                    reviews_diff = abs(avg_reviews_removed - avg_reviews_kept)
                    impact_results['reviews_difference'] = reviews_diff
                    
                    print(f"\n  评论数量比较:")
                    print(f"    被移除餐厅平均评论数: {avg_reviews_removed:.2f}")
                    print(f"    保留餐厅平均评论数: {avg_reviews_kept:.2f}")
                    print(f"    差异: {reviews_diff:.2f}")
                    
                    if reviews_diff > avg_reviews_kept * 0.5:  # 如果差异超过保留餐厅平均评论数的50%
                        impact_results['potential_bias'] = True
                        impact_results['warnings'].append(f"被移除餐厅评论数 ({avg_reviews_removed:.2f}) 与保留餐厅 ({avg_reviews_kept:.2f}) 差异明显")
                        print(f"    ⚠️  警告: 评论数差异明显，可能导致热门度分析偏倚")
                    else:
                        print(f"    ✓ 评论数差异较小，影响不大")
                
                # 类别分析（如果类别信息存在）
                if 'categories' in removed_data.columns:
                    # 提取被移除餐厅的主要类别
                    removed_categories = []
                    for cats in removed_data['categories'].dropna():
                        if isinstance(cats, str):
                            removed_categories.extend([cat.strip() for cat in cats.split(',')])
                    
                    if removed_categories:
                        from collections import Counter
                        removed_category_counts = Counter(removed_categories)
                        top_removed_categories = removed_category_counts.most_common(5)
                        
                        print(f"\n  被移除餐厅主要类别 (前5):")
                        for category, count in top_removed_categories:
                            print(f"    {category}: {count} 家")
        
        else:
            print("  无坐标清洗移除数据记录")
    else:
        print("  未找到坐标清洗移除数据文件")
        print("  注意: 这可能意味着坐标清洗没有移除数据，或者移除数据未被保存")
    
    # 2. 分析对地理分析的影响
    print("\n2. 对地理分析的影响评估:")
    print("-" * 40)
    
    if 'latitude' in business_df.columns and 'longitude' in business_df.columns:
        # 计算地理覆盖范围
        lat_min, lat_max = business_df['latitude'].min(), business_df['latitude'].max()
        lon_min, lon_max = business_df['longitude'].min(), business_df['longitude'].max()
        
        print(f"  餐厅地理覆盖范围:")
        print(f"    纬度: {lat_min:.4f}° - {lat_max:.4f}°")
        print(f"    经度: {lon_min:.4f}° - {lon_max:.4f}°")
        
        # 计算地理中心
        lat_center = business_df['latitude'].mean()
        lon_center = business_df['longitude'].mean()
        print(f"  地理中心: ({lat_center:.4f}°, {lon_center:.4f}°)")
        
        # 检查地理分布是否均匀
        lat_std = business_df['latitude'].std()
        lon_std = business_df['longitude'].std()
        print(f"  地理分布标准差: 纬度 {lat_std:.4f}°, 经度 {lon_std:.4f}°")
        
        if lat_std < 0.01 or lon_std < 0.01:
            impact_results['warnings'].append("餐厅地理分布过于集中")
            print(f"  ⚠️  警告: 地理分布过于集中，可能无法代表整个城市")
        else:
            print(f"  ✓ 地理分布相对分散，代表性较好")
    
    # 3. 分析对后续分析的影响
    print("\n3. 对后续分析的影响总结:")
    print("-" * 40)
    
    if impact_results['potential_bias']:
        print(f"  ⚠️  潜在偏倚: 坐标清洗可能引入了系统性偏倚")
        print(f"  建议:")
        print(f"    1. 在分析结论中说明坐标清洗的影响")
        print(f"    2. 对于星级和评论数的分析，考虑坐标清洗的潜在影响")
        print(f"    3. 地理分析可能无法代表被移除的餐厅")
    else:
        print(f"  ✓ 坐标清洗影响较小，对后续分析影响有限")
    
    if impact_results['warnings']:
        print(f"\n  主要警告:")
        for i, warning in enumerate(impact_results['warnings'], 1):
            print(f"    {i}. {warning}")
    
    # 4. 可视化影响分析
    if len(business_df) > 0 and os.path.exists(removed_file):
        removed_data = pd.read_csv(removed_file)
        if len(removed_data) > 0:
            print("\n4. 生成影响分析可视化...")
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. 星级分布对比
            if 'stars' in removed_data.columns and 'stars' in business_df.columns:
                axes[0, 0].hist(business_df['stars'].dropna(), bins=20, alpha=0.7, label='保留餐厅', color='blue')
                axes[0, 0].hist(removed_data['stars'].dropna(), bins=20, alpha=0.7, label='被移除餐厅', color='red')
                axes[0, 0].set_xlabel('星级')
                axes[0, 0].set_ylabel('餐厅数量')
                axes[0, 0].set_title('星级分布对比')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 评论数分布对比（对数坐标）
            if 'review_count' in removed_data.columns and 'review_count' in business_df.columns:
                kept_reviews = business_df['review_count'].dropna()
                removed_reviews = removed_data['review_count'].dropna()
                
                # 过滤掉0值，避免对数坐标问题
                kept_reviews = kept_reviews[kept_reviews > 0]
                removed_reviews = removed_reviews[removed_reviews > 0]
                
                if len(kept_reviews) > 0 and len(removed_reviews) > 0:
                    axes[0, 1].hist(np.log10(kept_reviews), bins=20, alpha=0.7, label='保留餐厅', color='blue')
                    axes[0, 1].hist(np.log10(removed_reviews), bins=20, alpha=0.7, label='被移除餐厅', color='red')
                    axes[0, 1].set_xlabel('评论数量 (对数)')
                    axes[0, 1].set_ylabel('餐厅数量')
                    axes[0, 1].set_title('评论数量分布对比 (对数)')
                    axes[0, 1].legend()
                    axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 地理分布图（只显示保留的餐厅）
            if 'latitude' in business_df.columns and 'longitude' in business_df.columns:
                axes[1, 0].scatter(business_df['longitude'], business_df['latitude'], 
                                 alpha=0.3, s=10, color='blue', label='保留餐厅')
                axes[1, 0].set_xlabel('经度')
                axes[1, 0].set_ylabel('纬度')
                axes[1, 0].set_title('保留餐厅地理分布')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].legend()
            
            # 4. 移除比例饼图
            labels = ['保留餐厅', '被移除餐厅']
            sizes = [len(business_df), len(removed_data)]
            colors = ['#66b3ff', '#ff9999']
            
            axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('坐标清洗移除比例')
            
            plt.suptitle('坐标清洗影响分析可视化', fontsize=16)
            plt.tight_layout()
            plt.show()
    
    print("\n" + "=" * 60)
    print("坐标清洗影响分析完成")
    print("=" * 60)
    
    # 保存分析结果
    if SAVE_INTERMEDIATE:
        import json
        impact_file = os.path.join(INTERMEDIATE_PATH, "coordinate_cleaning_impact.json")
        with open(impact_file, 'w', encoding='utf-8') as f:
            json.dump(impact_results, f, ensure_ascii=False, indent=2)
        print(f"坐标清洗影响分析结果已保存到: {impact_file}")
    
    return impact_results

