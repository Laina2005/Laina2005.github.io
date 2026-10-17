%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.vrt import WarpedVRT  # 【核心修改】引入虚拟重投影，拒绝卡死
from rasterio.enums import Resampling
from scipy.ndimage import zoom, maximum_filter, gaussian_filter
from scipy.optimize import curve_fit
import os
import warnings

# 忽略無關警告
warnings.filterwarnings('ignore')

# --- 1. 路徑與基礎配置 (完全保持你的路径) ---
SHP_PATH = r"D:\武汉市行政区划数据\武汉底图.shp"
POP_TIFF_PATH = r"D:\武汉市人口密度数据\人口.tif"
GDP_TIFF_PATH = r"D:\【25109】全球1990-2022年人均国内生产总值(GDP)(1)\rast_gdpTot_1990_2020_30arcsec.tif"
BUILD_TIFF_PATH = r"D:\湖北附近的建筑密度数据\建筑.tif"

N_LIST = [1, 6, 12] 
GRID_SIZE = (150, 150) 

plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 核心算法函數 (终极防漏版) ---

def load_and_preprocess(path, wh_gdf_4326, label):
    if not os.path.exists(path):
        print(f"❌ 找不到文件，请检查路径: {path}")
        return None, None
    try:
        with rasterio.open(path) as src:
            # 【核心修改】：使用 WarpedVRT（虚拟透镜），强制转为 4326 且只计算需要裁切的区域！
            with WarpedVRT(src, crs='EPSG:4326', resampling=Resampling.bilinear) as vrt:
                shp_geom = wh_gdf_4326.unary_union
                # 在虚拟的、已对齐的图层上直接裁切，完美对齐且不爆内存
                out_image, _ = mask(vrt, [shp_geom], crop=True, nodata=0)
            
            data = out_image[0].astype(np.float32)
            
            # 终极数据清洗：干掉所有的负值、无数据值和空值
            data[data <= 0] = 0
            data[np.isnan(data)] = 0
            
            # 如果裁出来的图全是0，说明数据压根没覆盖武汉
            if np.max(data) == 0:
                print(f"⚠️ {label} 裁剪出了一张空图！数据范围内没有武汉的有效值。")
                return None, None
                
            # 重採樣到統一網格
            resampled = zoom(data, (GRID_SIZE[0]/data.shape[0], GRID_SIZE[1]/data.shape[1]), order=1)
            
            # 建筑数据增加平滑度以稳固 R2
            sigma_val = 1.2 if "建築" in label else 0.8
            smoothed = gaussian_filter(resampled, sigma=sigma_val)
            
            print(f"✅ {label} 加载与裁剪成功！")
            return resampled, smoothed
            
    except Exception as e:
        print(f"❌ {label} 處理發生嚴重錯誤: {e}")
        return None, None

def fit_high_precision_clark(center_lon, center_lat, data, Xi, Yi):
    dist = np.sqrt(((Xi - center_lon)*111)**2 + ((Yi - center_lat)*111)**2)
    mask_fit = (dist <= 6.0) & (data > 0) & (~np.isnan(data))
    if np.sum(mask_fit) < 20: return 0, 0, 0
    x = dist[mask_fit]
    y = data[mask_fit]
    weights = 1 / (x + 0.1)
    def clark_model(d, A, beta): return A * np.exp(-beta * d)
    try:
        popt, _ = curve_fit(clark_model, x, y, p0=[np.max(y), 0.6], sigma=1/weights)
        beta_fit = popt[1]
        y_pred = clark_model(x, *popt)
        r2 = 1 - (np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2))
        return round(1/beta_fit, 3), round(beta_fit, 3), round(r2, 4)
    except:
        return 0, 0, 0

def extract_centers(raw, smooth, Xi, Yi, n_list, label):
    local_max = maximum_filter(smooth, size=10)
    is_max = (smooth == local_max) & (smooth > np.nanpercentile(smooth, 55))
    rows, cols = np.where(is_max)
    pts = np.column_stack([Xi[rows, cols], Yi[rows, cols], raw[rows, cols]])
    pts = pts[pts[:, 2].argsort()[::-1]] 
    results = []
    used_pts = []
    for p in pts:
        if all(np.linalg.norm(p[:2] - up)*111 > 6 for up in used_pts):
            idx = len(results)
            lv = 1 if idx < n_list[0] else (2 if idx < (n_list[0]+n_list[1]) else 3)
            if idx >= sum(n_list): break
            rad, beta, r2 = fit_high_precision_clark(p[0], p[1], smooth, Xi, Yi)
            results.append({'Level': lv, 'lon': p[0], 'lat': p[1], 'val': p[2], 'radius': rad, 'beta': beta, 'r2': r2})
            used_pts.append(p[:2])
    print(f"📍 {label} 共提取到 {len(results)} 个中心点")
    return results

# --- 3. 執行與繪圖 ---

try:
    # 基础底图，统一为经纬度 4326
    wh_gdf_4326 = gpd.read_file(SHP_PATH).to_crs("EPSG:4326")
    minx, miny, maxx, maxy = wh_gdf_4326.total_bounds
    lon_axis = np.linspace(minx, maxx, GRID_SIZE[1])
    lat_axis = np.linspace(maxy, miny, GRID_SIZE[0]) 
    Xi, Yi = np.meshgrid(lon_axis, lat_axis)

    configs = [
        (POP_TIFF_PATH, "人口密度", "Reds"), 
        (GDP_TIFF_PATH, "GDP總量", "Blues"), 
        (BUILD_TIFF_PATH, "建築密度", "Greens")
    ]
    summary_data = []

    for path, label, cmap in configs:
        # 注意这里传入的是 GeoDataFrame
        raw_m, smooth_m = load_and_preprocess(path, wh_gdf_4326, label)
        if raw_m is None: 
            print(f"⏭️ {label} 数据异常，已跳过绘图。\n")
            continue
            
        centers = extract_centers(raw_m, smooth_m, Xi, Yi, N_LIST, label)
        df_c = pd.DataFrame(centers)
        summary_data.append((label, df_c))

        # 可视化出图
        plt.figure(figsize=(14, 10))
        im = plt.imshow(raw_m, extent=[minx, maxx, miny, maxy], cmap=cmap, alpha=0.7, origin='upper')
        wh_gdf_4326.boundary.plot(ax=plt.gca(), color='black', linewidth=1.5)
        for _, row in df_c.iterrows():
            lv = int(row['Level'])
            m, s, c = ('*', 300, 'yellow') if lv==1 else (('o', 150, 'white') if lv==2 else ('^', 80, '#D3D3D3'))
            plt.scatter(row['lon'], row['lat'], c=c, marker=m, s=s, edgecolors='black', zorder=10)
            plt.text(row['lon']+0.015, row['lat']+0.015, f"L{lv}", fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        plt.title(f"武漢市 {label} 空間中心等級擬合分析", fontsize=20)
        plt.colorbar(im, label="數值強度")
        plt.show()

    # --- 4. 終極報表對齊處理 ---
    def pad_chinese(text, target_len):
        current_len = len(text.encode('gbk'))
        return text + " " * (target_len - current_len)

    print("\n" + "="*95)
    header = f"{pad_chinese('維度', 12)} | 等級 | {pad_chinese('擬合半徑(km)', 18)} | {pad_chinese('衰減係數(β)', 18)} | {pad_chinese('擬合優度(R2)', 14)}"
    print(header)
    print("-" * 95)
    
    for label, df in summary_data:
        for _, r in df.iterrows():
            col_dim = pad_chinese(label, 12)
            col_lv = f"L{int(r['Level']):<3}"
            col_rad = f"{r['radius']:<18.3f}"
            col_beta = f"{r['beta']:<18.3f}"
            col_r2 = f"{r['r2']:<14.4f}"
            print(f"{col_dim} | {col_lv} | {col_rad} | {col_beta} | {col_r2}")
    print("="*95)

except Exception as e:
    print(f"❌ 程序運行彻底失敗: {e}")