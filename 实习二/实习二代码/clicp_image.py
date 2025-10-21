# clicp_image.py

import rasterio
from rasterio.mask import mask
from shapely.geometry import box
import json
import matplotlib.pyplot as plt
def clip_image(input_file, output_file):
    # 1. 定义纽约市边界（经纬度）
    nyc_bounds = [
        -74.2591,  # 西边界（最小经度）
        40.4774,   # 南边界（最小纬度）
        -73.7004,  # 东边界（最大经度）
        40.9176    # 北边界（最大纬度）
    ]

    # 2. 将边界转换为rasterio需要的几何格式
    geoms = [box(*nyc_bounds)]  # 转换为shapely的box几何对象
    geoms_json = [json.loads(json.dumps(geom.__geo_interface__)) for geom in geoms]  # 转为GeoJSON格式

    # 3. 读取全球人口TIFF文件并裁剪
    with rasterio.open(input_file) as src:
        # 裁剪并获取裁剪后的元数据
        out_image, out_transform = mask(src, geoms_json, crop=True)
        out_meta = src.meta.copy()  # 复制原数据的元信息（坐标、投影等）

    # 4. 更新裁剪后的元数据（调整尺寸和变换参数）
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

    # 5. 保存裁剪后的纽约市人口TIFF（带坐标信息）
    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(out_image)

def show_image(show_file):
    with rasterio.open(show_file) as src:
        image = src.read(1)
        plt.imshow(image, cmap='viridis')
        plt.colorbar()
        plt.title("NYC Population Density (2016)")
        plt.show()

if __name__ == "__main__":
    input_file = r"D:\必须用电脑解决的作业\地理大数据分析\实习二\实习二数据\global_pop_2016_CN_1km_R2025A_UA_v1.tif"
    output_file = r"D:\必须用电脑解决的作业\地理大数据分析\实习二\实习二数据\nyc_population_2016.tif"
    # clip_image(input_file, output_file)
    show_image(output_file)
