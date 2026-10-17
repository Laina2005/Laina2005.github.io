import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle
import networkx as nx

import sys
import os

def resource_path(relative_path):
    """ 获取资源的绝对路径，兼容 PyInstaller 打包 """
    try:
        # 如果是被打包成 .exe，PyInstaller 会把文件解压到 _MEIPASS 临时目录
        base_path = sys._MEIPASS
    except Exception:
        # 如果是直接运行 .py 文件，就用当前目录
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ----------------------------
# 中文字体支持
# ----------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ----------------------------
# 文件路径（请根据你的实际路径修改）
# ----------------------------
parks_file = resource_path("parks.csv")
distances_file = resource_path("distances.csv")

# ----------------------------
# 1. 加载数据
# ----------------------------
try:
    df_parks = pd.read_csv(parks_file, encoding='gbk', dtype={'编号': str})
except Exception as e:
    print(f"❌ 无法读取 parks.csv: {e}")
    exit(1)

required_cols = ['编号', '公园名称', '类型', '面积(km^2)', '纬度', '经度']
for col in required_cols:
    if col not in df_parks.columns:
        print(f"❌ parks.csv 缺少列：'{col}'")
        exit(1)

try:
    df_dist = pd.read_csv(distances_file, index_col=0, dtype=str)
    df_dist.index = df_dist.index.astype(str)
    df_dist.columns = df_dist.columns.astype(str)
    df_dist = df_dist.apply(pd.to_numeric, errors='coerce')
except Exception as e:
    print(f"❌ 无法读取 distances.csv: {e}")
    exit(1)

# 映射字典
id_to_name = dict(zip(df_parks['编号'], df_parks['公园名称']))
name_to_id = {name: id_ for id_, name in id_to_name.items()}
name_to_area = dict(zip(df_parks['公园名称'], df_parks['面积(km^2)']))

# ----------------------------
# 2. 处理南沙湿地公园位置（避免重叠）
# ----------------------------
other_parks = df_parks[df_parks['公园名称'] != '南沙湿地公园']
nan_sha_row = df_parks[df_parks['公园名称'] == '南沙湿地公园']

if nan_sha_row.empty:
    pos = {row['编号']: (row['经度'], row['纬度']) for _, row in df_parks.iterrows()}
else:
    min_lon = other_parks['经度'].min()
    max_lon = other_parks['经度'].max()
    min_lat = other_parks['纬度'].min()
    max_lat = other_parks['纬度'].max()
    nan_sha_x = max_lon + 0.005
    nan_sha_y = min_lat - 0.002
    pos = {}
    for _, row in df_parks.iterrows():
        pid = row['编号']
        if row['公园名称'] == '南沙湿地公园':
            pos[pid] = (nan_sha_x, nan_sha_y)
        else:
            pos[pid] = (row['经度'], row['纬度'])

# ----------------------------
# 3. 构建图（基于距离矩阵）
# ----------------------------
G = nx.Graph()
for pid in df_parks['编号']:
    G.add_node(pid)

INF = 1e9
for i in df_dist.index:
    for j in df_dist.columns:
        if i == j:
            continue
        dist = df_dist.loc[i, j]
        if pd.notna(dist) and dist < INF:
            G.add_edge(i, j, weight=float(dist))

print(f"✅ 图构建完成：{G.number_of_nodes()} 节点，{G.number_of_edges()} 边")

# ----------------------------
# 4. 绘图函数（仅高亮节点，不连边）
# ----------------------------
def draw_graph(ax, highlight_nodes=None):
    ax.clear()
    # 画所有边（始终为灰色，不因高亮节点而变红）
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], color='lightgray', linewidth=0.8, zorder=2)
    
    # 画节点
    for node in G.nodes():
        name = id_to_name[node]
        x, y = pos[node]
        color = 'red' if (highlight_nodes and node in highlight_nodes) else 'lightblue'
        circle = Circle((x, y), radius=0.0015, facecolor=color, edgecolor='black', linewidth=0.8, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y + 0.0018, name, ha='center', va='bottom', fontsize=8, zorder=4)
    
    # 设置坐标范围
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    margin = 0.003
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("广州市公园分析系统", fontsize=14, pad=20)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=1)

# ----------------------------
# 5. 功能函数
# ----------------------------
def query():
    choice = simpledialog.askstring(
        "查询方式",
        "请选择查询方式：\n1 - 按编号\n2 - 按类型\n3 - 按面积范围",
        parent=root
    )
    if not choice:
        return

    highlight_pids = []
    try:
        choice = choice.strip()
        if choice == '1':
            pid = simpledialog.askstring("按编号查询", "请输入公园编号：", parent=root)
            if pid and pid in id_to_name:
                highlight_pids = [pid]
            else:
                messagebox.showerror("错误", "编号不存在！", parent=root)
                return

        elif choice == '2':
            park_types = sorted(df_parks['类型'].dropna().unique())
            type_list = "\n".join([f"- {t}" for t in park_types])
            user_type = simpledialog.askstring(
                "按类型查询",
                f"可用类型：\n{type_list}\n\n请输入要查询的类型（精确匹配）：",
                parent=root
            )
            if user_type:
                matched = df_parks[df_parks['类型'] == user_type.strip()]
                if not matched.empty:
                    highlight_pids = matched['编号'].tolist()
                else:
                    messagebox.showinfo("提示", "未找到该类型的公园", parent=root)
                    return
            else:
                return

        elif choice == '3':
            area_input = simpledialog.askstring(
                "按面积范围查询",
                "请输入面积范围（单位：km²），格式：min,max\n例如：10,50",
                parent=root
            )
            if area_input:
                parts = [p.strip() for p in area_input.split(',')]
                if len(parts) == 2:
                    min_a = float(parts[0])
                    max_a = float(parts[1])
                    if min_a > max_a:
                        min_a, max_a = max_a, min_a
                    matched = df_parks[
                        (df_parks['面积(km^2)'] >= min_a) &
                        (df_parks['面积(km^2)'] <= max_a)
                    ]
                    if not matched.empty:
                        highlight_pids = matched['编号'].tolist()
                    else:
                        messagebox.showinfo("提示", "该面积范围内无公园", parent=root)
                        return
                else:
                    messagebox.showerror("错误", "格式错误！应为：min,max", parent=root)
                    return
            else:
                return
        else:
            messagebox.showerror("错误", "请输入 1、2 或 3", parent=root)
            return

        # 高亮并显示结果
        if highlight_pids:
            names = [id_to_name[pid] for pid in highlight_pids]
            info = "\n".join(names[:10])
            if len(names) > 10:
                info += f"\n... 等 {len(names)} 个公园"
            messagebox.showinfo("查询结果", f"找到 {len(names)} 个公园：\n{info}", parent=root)
            draw_graph(ax, highlight_nodes=highlight_pids)
            canvas.draw()

    except ValueError:
        messagebox.showerror("错误", "输入格式不正确（面积需为数字）！", parent=root)
    except Exception as e:
        messagebox.showerror("错误", f"查询失败：{str(e)}", parent=root)


def rank():
    """显示所有公园按面积从大到小的完整排名"""
    try:
        sorted_parks = df_parks.sort_values('面积(km^2)', ascending=False).reset_index(drop=True)
        lines = []
        for i, row in sorted_parks.iterrows():
            name = row['公园名称']
            area = row['面积(km^2)']
            lines.append(f"{i+1}. {name} — {area:.2f} km²")
        full_text = "\n".join(lines)
        messagebox.showinfo("全部公园面积排名（从大到小）", full_text, parent=root)
    except Exception as e:
        messagebox.showerror("错误", f"排序失败：{str(e)}", parent=root)


def shortest_path():
    start = simpledialog.askstring("起点", "请输入起点公园名称：", parent=root)
    end = simpledialog.askstring("终点", "请输入终点公园名称：", parent=root)
    if not start or not end:
        return
    if start not in name_to_id or end not in name_to_id:
        messagebox.showerror("错误", "公园名称不存在！", parent=root)
        return

    s_id, e_id = name_to_id[start], name_to_id[end]
    try:
        path = nx.dijkstra_path(G, s_id, e_id, weight='weight')
        total = nx.dijkstra_path_length(G, s_id, e_id, weight='weight')
        names = [id_to_name[n] for n in path]

        # 先绘制基础图（高亮路径节点）
        draw_graph(ax, highlight_nodes=path)

        # 再单独绘制红色路径线（覆盖在上面）
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ax.plot([x0, x1], [y0, y1], color='red', linewidth=3, zorder=5)

        canvas.draw()

        messagebox.showinfo(
            "最短路径",
            f"路径：{' → '.join(names)}\n总距离：{total:.2f} km",
            parent=root
        )
    except nx.NetworkXNoPath:
        messagebox.showerror("错误", "两公园之间不可达！", parent=root)
    except Exception as e:
        messagebox.showerror("错误", f"路径计算失败：{str(e)}", parent=root)


def reset():
    draw_graph(ax)
    canvas.draw()


# ----------------------------
# 6. 主窗口
# ----------------------------
root = tk.Tk()
root.title("广州市公园分析系统")
root.geometry("1200x800")

# 创建图形
fig, ax = plt.subplots(figsize=(10, 7))
canvas = FigureCanvasTkAgg(fig, root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# 初始绘图
draw_graph(ax)
canvas.draw()

# 按钮区域
btn_frame = tk.Frame(root)
btn_frame.pack(side=tk.BOTTOM, pady=10)

ttk.Button(btn_frame, text="公园查询", command=query, width=12).pack(side=tk.LEFT, padx=8)
ttk.Button(btn_frame, text="面积排序", command=rank, width=12).pack(side=tk.LEFT, padx=8)
ttk.Button(btn_frame, text="最短路径", command=shortest_path, width=12).pack(side=tk.LEFT, padx=8)
ttk.Button(btn_frame, text="重置视图", command=reset, width=12).pack(side=tk.LEFT, padx=8)

# 启动主循环
print("✅ 广州市公园分析系统启动成功！")
root.mainloop()