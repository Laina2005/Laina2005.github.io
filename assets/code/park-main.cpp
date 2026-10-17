#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
using namespace std;
#define INF 1000000000

double Park_Dist_Graph[20][20];
const int park_total_number = 20;

struct Park {
    int id;
    string name;
    string type;
    double area;
    string location;
};

vector<Park> parks = {
    {1, "白云山风景区", "自然风景区", 20.98, "（23.1847°N，113.2972°E）"},
    {2, "越秀公园", "综合公园", 6.82, "（23.1393°N，113.2714°E）"},
    {3, "广州动物园", "动物园", 2.31, "（23.1410°N，113.3084°E）"},
    {4, "珠江公园", "街心公园", 0.28, "（23.1235°N，113.3306°E）"},
    {5, "天河公园", "社区公园", 7.03, "（23.1267°N，113.3652°E）"},
    {6, "海珠湖公园", "湿地公园", 2.25, "（23.0695°N，113.3278°E）"},
    {7, "荔湾湖公园", "街心公园", 0.67, "（23.1248°N，113.2415°E）"},
    {8, "大夫山森林公园", "森林公园", 8.0, "（22.9512°N，113.3476°E）"},
    {9, "华南植物公园", "植物园", 3.33, "（23.1925°N，113.3701°E）"},
    {10, "广州塔公园", "主题公园", 0.45, "（23.1064°N，113.3245°E）"},
    {11, "流花湖公园", "综合公园", 0.54, "（23.1386°N，113.2538°E）"},
    {12, "东山湖公园", "街心公园", 0.33, "（23.1242°N，113.2943°E）"},
    {13, "晓港公园", "社区公园", 0.32, "（23.0968°N，113.2784°E）"},
    {14, "黄埔公园", "综合公园", 1.02, "（23.1028°N，113.4346°E）"},
    {15, "大学城中心湖公园", "湿地公园", 1.85, "（23.0593°N，113.3942°E）"},
    {16, "二沙岛体育公园", "体育公园", 0.28, "（23.1135°N，113.3089°E）"},
    {17, "广州科学城公园", "科技公园", 3.5, "（23.1657°N，113.4458°E）"},
    {18, "麓湖公园", "自然风景公园", 2.5, "（23.1501°N，113.2843°E）"},
    {19, "番禺儿童公园", "儿童公园", 0.42, "（22.9395°N，113.3648°E）"},
    {20, "南沙湿地公园", "湿地公园", 25.0, "（22.6012°N，113.6345°E）"}
};

Park* Search_By_Id(int id) {
    for (int i = 0; i < parks.size(); i++) {
        if (parks[i].id == id) {
            return &parks[i];
        }
    }
    return nullptr;
}

vector<Park> Search_By_Type(string type) {
    vector<Park> result;
    for (int i = 0; i < parks.size(); i++) {
        if (parks[i].type == type) {
            result.push_back(parks[i]);
        }
    }
    return result;
}

vector<Park> Search_By_Area_Range(double min_area, double max_area) {
    vector<Park> result;
    for (int i = 0; i < parks.size(); i++) {
        if (parks[i].area >= min_area && parks[i].area <= max_area) {
            result.push_back(parks[i]);
        }
    }
    return result;
}

// ANSI颜色定义
#define COLOR_RESET "\033[0m"
#define COLOR_RED "\033[31m"
#define COLOR_GREEN "\033[32m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_BLUE "\033[34m"
#define COLOR_MAGENTA "\033[35m"
#define COLOR_CYAN "\033[36m"
#define COLOR_WHITE "\033[37m"
#define COLOR_BOLD "\033[1m"
#define COLOR_UNDERLINE "\033[4m"

void Print_Park_Info(const Park* park) {
    if (park == nullptr) {
        cout << COLOR_RED << COLOR_BOLD << "\n┌─────────────────────────────────┐" << COLOR_RESET << endl;
        cout << COLOR_RED << COLOR_BOLD << "│          未找到该公园！          │" << COLOR_RESET << endl;
        cout << COLOR_RED << COLOR_BOLD << "└─────────────────────────────────┘" << COLOR_RESET << endl;
        return;
    }
    cout << COLOR_CYAN << "\n┌──────────────── 公园详情 ────────────────┐" << COLOR_RESET << endl;
    cout << COLOR_WHITE << "│  编号：" << COLOR_YELLOW << park->id << COLOR_WHITE << string(30 - to_string(park->id).size(), ' ') << "│" << COLOR_RESET << endl;
    cout << COLOR_WHITE << "│  名称：" << COLOR_GREEN << park->name << string(30 - park->name.size(), ' ') << "│" << COLOR_RESET << endl;
    cout << COLOR_WHITE << "│  类型：" << COLOR_MAGENTA << park->type << string(30 - park->type.size(), ' ') << "│" << COLOR_RESET << endl;
    cout << COLOR_WHITE << "│  面积：" << COLOR_BLUE << park->area << " km?" << string(24 - to_string(park->area).size() - 5, ' ') << "│" << COLOR_RESET << endl;
    cout << COLOR_WHITE << "│  位置：" << COLOR_CYAN << park->location << string(30 - park->location.size(), ' ') << "│" << COLOR_RESET << endl;
    cout << COLOR_CYAN << "└─────────────────────────────────────────┘" << COLOR_RESET << endl;
}

void Print_Multiple_Parks(const vector<Park>& list) {
    if (list.size() == 0) {
        cout << COLOR_RED << COLOR_BOLD << "\n┌─────────────────────────────────┐" << COLOR_RESET << endl;
        cout << COLOR_RED << COLOR_BOLD << "│        未找到符合条件的公园！        │" << COLOR_RESET << endl;
        cout << COLOR_RED << COLOR_BOLD << "└─────────────────────────────────┘" << COLOR_RESET << endl;
        return;
    }

    cout << COLOR_GREEN << COLOR_BOLD << "\n┌─────────────────────────────────┐" << COLOR_RESET << endl;
    cout << COLOR_GREEN << COLOR_BOLD << "│        共找到 " << list.size() << " 个符合条件的公园        │" << COLOR_RESET << endl;
    cout << COLOR_GREEN << COLOR_BOLD << "└─────────────────────────────────┘" << COLOR_RESET << endl;

    for (int i = 0; i < list.size(); i++) {
        cout << COLOR_YELLOW << "\n【第 " << i + 1 << " 个公园】" << COLOR_RESET << endl;
        Print_Park_Info(&list[i]);
        cout << COLOR_BLUE << "──────────────────────────────────────────────" << COLOR_RESET << endl;
    }
}

void BubbleSort(vector<Park>& parks) {
    int n = parks.size();
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < n - 1 - i; ++j) {
            if (parks[j].area > parks[j + 1].area) {
                Park temp = parks[j];
                parks[j] = parks[j + 1];
                parks[j + 1] = temp;
            }
        }
    }
}

int Partition(vector<Park>& parks, int low, int high) {
    double pivot = parks[high].area;
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (parks[j].area <= pivot) {
            i++;
            Park temp = parks[i];
            parks[i] = parks[j];
            parks[j] = temp;
        }
    }
    Park temp = parks[i + 1];
    parks[i + 1] = parks[high];
    parks[high] = temp;
    return i + 1;
}

void QuickSort(vector<Park>& parks, int low, int high) {
    if (low < high) {
        int pivotIdx = Partition(parks, low, high);
        QuickSort(parks, low, pivotIdx - 1);
        QuickSort(parks, pivotIdx + 1, high);
    }
}

bool Shortest_Path(int start_id, int end_id, vector<int>& path, double& total_dist) {
    path.clear();
    if (start_id == end_id) {
        path.push_back(start_id);
        total_dist = 0;
        return true;
    }

    int start = start_id - 1;
    int end = end_id - 1;
    double dist[park_total_number];
    bool visited[park_total_number];
    int prev_number[park_total_number];

    for (int i = 0; i < park_total_number; i++) {
        dist[i] = Park_Dist_Graph[start][i];
        visited[i] = false;
        prev_number[i] = (dist[i] < INF && i != start) ? start : -1;
    }
    dist[start] = 0;
    visited[start] = true;

    for (int i = 1; i < park_total_number; i++) {
        double min_dist = INF;
        int min_id = -1;

        for (int j = 0; j < park_total_number; j++) {
            if (!visited[j] && dist[j] < min_dist) {
                min_dist = dist[j];
                min_id = j;
            }
        }

        if (min_id == -1) break;
        visited[min_id] = true;

        for (int j = 0; j < park_total_number; j++) {
            if (!visited[j] && Park_Dist_Graph[min_id][j] < INF && dist[min_id] + Park_Dist_Graph[min_id][j] < dist[j]) {
                dist[j] = dist[min_id] + Park_Dist_Graph[min_id][j];
                prev_number[j] = min_id;
            }
        }
    }

    if (dist[end] >= INF) {
        return false;
    }

    int current = end;
    while (current != -1) {
        path.push_back(current + 1);
        current = prev_number[current];
    }

    reverse(path.begin(), path.end());
    total_dist = dist[end];
    return true;
}

void PrintBeforeSort() {
    cout << COLOR_CYAN << "\n┌─────────────────────────────────────────┐" << COLOR_RESET << endl;
    cout << COLOR_CYAN << "│           排序前 - 原始公园列表           │" << COLOR_RESET << endl;
    cout << COLOR_CYAN << "└─────────────────────────────────────────┘" << COLOR_RESET << endl;
    for (int i = 0; i < parks.size(); i++) {
        cout << COLOR_WHITE << "│  编号：" << COLOR_YELLOW << parks[i].id
            << COLOR_WHITE << "  |  名称：" << COLOR_GREEN << parks[i].name
            << COLOR_WHITE << "  |  面积：" << COLOR_BLUE << parks[i].area << " km?" << COLOR_RESET << endl;
    }
    cout << COLOR_CYAN << "─────────────────────────────────────────────" << COLOR_RESET << endl;
}

void PrintSortedParks(const vector<Park>& parks, const string& title) {
    cout << COLOR_MAGENTA << "\n┌─────────────────────────────────────────┐" << COLOR_RESET << endl;
    cout << COLOR_MAGENTA << "│               " << title << "               │" << COLOR_RESET << endl;
    cout << COLOR_MAGENTA << "└─────────────────────────────────────────┘" << COLOR_RESET << endl;
    for (int i = 0; i < parks.size(); i++) {
        cout << COLOR_WHITE << "│  编号：" << COLOR_YELLOW << parks[i].id
            << COLOR_WHITE << "  |  名称：" << COLOR_GREEN << parks[i].name
            << COLOR_WHITE << "  |  面积：" << COLOR_BLUE << parks[i].area << " km?" << COLOR_RESET << endl;
    }
    cout << COLOR_MAGENTA << "─────────────────────────────────────────────" << COLOR_RESET << endl;
}

void Print_Shortest_Path(int start_id, int end_id, const vector<int>& path, double total_dist) {
    cout << COLOR_GREEN << COLOR_BOLD << "\n┌─────────────────────────────────────────┐" << COLOR_RESET << endl;
    cout << COLOR_GREEN << COLOR_BOLD << "│           最短路径查询结果             │" << COLOR_RESET << endl;
    cout << COLOR_GREEN << COLOR_BOLD << "└─────────────────────────────────────────┘" << COLOR_RESET << endl;

    Park* start_park = Search_By_Id(start_id);
    Park* end_park = Search_By_Id(end_id);

    if (start_park != nullptr) {
        cout << COLOR_WHITE << "起点公园：" << COLOR_GREEN << start_park->name
            << COLOR_WHITE << " （编号：" << COLOR_YELLOW << start_park->id << COLOR_WHITE << "）" << COLOR_RESET << endl;
    }
    if (end_park != nullptr) {
        cout << COLOR_WHITE << "终点公园：" << COLOR_GREEN << end_park->name
            << COLOR_WHITE << " （编号：" << COLOR_YELLOW << end_park->id << COLOR_WHITE << "）" << COLOR_RESET << endl;
    }

    cout << COLOR_WHITE << "总距离：" << COLOR_BLUE << total_dist << " 千米" << COLOR_RESET << endl;
    cout << COLOR_WHITE << "路径顺序：" << COLOR_RESET;

    for (int i = 0; i < path.size(); i++) {
        Park* p = Search_By_Id(path[i]);
        if (p != nullptr) {
            cout << COLOR_MAGENTA << p->name << COLOR_RESET;
        }
        if (i < path.size() - 1) {
            cout << COLOR_YELLOW << " → " << COLOR_RESET;
        }
    }

    cout << COLOR_GREEN << "\n┌─────────────────────────────────────────┐" << COLOR_RESET << endl;
    cout << COLOR_GREEN << "│             查询完成！                 │" << COLOR_RESET << endl;
    cout << COLOR_GREEN << "└─────────────────────────────────────────┘" << COLOR_RESET << endl;
}

void Show_Sort_Submenu() {
    cout << COLOR_CYAN << COLOR_BOLD << "\n┌─────────────────────────────────┐" << COLOR_RESET << endl;
    cout << COLOR_CYAN << COLOR_BOLD << "│          排序方法选择           │" << COLOR_RESET << endl;
    cout << COLOR_CYAN << COLOR_BOLD << "├─────────────────────────────────┤" << COLOR_RESET << endl;
    cout << COLOR_WHITE << "│  1. 冒泡排序（稳定排序）        │" << COLOR_RESET << endl;
    cout << COLOR_WHITE << "│  2. 快速排序（高效排序）        │" << COLOR_RESET << endl;
    cout << COLOR_WHITE << "│  0. 返回主菜单                  │" << COLOR_RESET << endl;
    cout << COLOR_CYAN << COLOR_BOLD << "└─────────────────────────────────┘" << COLOR_RESET << endl;
    cout << COLOR_YELLOW << COLOR_BOLD << "请选择（0-2）：" << COLOR_RESET;
}

// 适配终端的全屏菜单（无截断、无乱码）
void Show_Menu() {
    // 清屏并定位到左上角
    cout << "\033[2J\033[H";

    // 像素风格全屏菜单（适配终端宽度，内容完整显示）
    cout << COLOR_BLUE << COLOR_BOLD << "─────────────────────────────────────────────────────────────────────────────────────────────────────" << COLOR_RESET << endl;
    cout << COLOR_BLUE << COLOR_BOLD << "│                                         公园信息查询系统 V1.0                                         │" << COLOR_RESET << endl;
    cout << COLOR_BLUE << COLOR_BOLD << "─────────────────────────────────────────────────────────────────────────────────────────────────────" << COLOR_RESET << endl;
    cout << COLOR_CYAN << COLOR_BOLD << "│  ██████╗  █████╗  ███╗   ██╗ ██████╗  █████╗  ███████╗ ███████╗                                       │" << COLOR_RESET << endl;
    cout << COLOR_CYAN << COLOR_BOLD << "│  ██╔══██╗██╔══██╗ ████╗  ██║ ██╔══██╗██╔══██╗ ██╔════╝ ██╔════╝                                │" << COLOR_RESET << endl;
    cout << COLOR_CYAN << COLOR_BOLD << "│  ██████╔╝███████║ ██╔██╗ ██║ ██████╔╝███████║ █████╗   ███████╗                                 │" << COLOR_RESET << endl;
    cout << COLOR_CYAN << COLOR_BOLD << "│  ██╔══██╗██╔══██║ ██║╚██╗██║ ██╔══██╗██╔══██║ ██╔══╝   ╚════██║                                │" << COLOR_RESET << endl;
    cout << COLOR_CYAN << COLOR_BOLD << "│  ██████╔╝██║  ██║ ██║ ╚████║ ██████╔╝██║  ██║ ███████╗ ███████║                                   │" << COLOR_RESET << endl;
    cout << COLOR_CYAN << COLOR_BOLD << "│  ╚═════╝ ╚═╝  ╚═╝ ╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═╝ ╚══════╝ ╚══════╝                                      │" << COLOR_RESET << endl;
    cout << COLOR_BLUE << COLOR_BOLD << "─────────────────────────────────────────────────────────────────────────────────────────────────────" << COLOR_RESET << endl;
    cout << COLOR_YELLOW << COLOR_BOLD << "│  [ 1 ] 按编号查询公园信息                [ 2 ] 按类型查询公园信息                                │" << COLOR_RESET << endl;
    cout << COLOR_YELLOW << COLOR_BOLD << "│  [ 3 ] 按面积范围查询公园                [ 4 ] 公园面积排序功能                                  │" << COLOR_RESET << endl;
    cout << COLOR_YELLOW << COLOR_BOLD << "│  [ 5 ] 计算公园之间最短路径              [ 6 ] 退出本查询系统                                    │" << COLOR_RESET << endl;
    cout << COLOR_BLUE << COLOR_BOLD << "─────────────────────────────────────────────────────────────────────────────────────────────────────" << COLOR_RESET << endl;
    cout << COLOR_GREEN << COLOR_BOLD << "│  提示：请输入 1-6 之间的数字选择对应功能                                                           │" << COLOR_RESET << endl;
    cout << COLOR_BLUE << COLOR_BOLD << "─────────────────────────────────────────────────────────────────────────────────────────────────────" << COLOR_RESET << endl;
}

int Enter_Number() {
    int number;
    cin >> number;
    while (number > 20 || number < 1) {
        cout << COLOR_RED << COLOR_BOLD << "\n┌─────────────────────────────────┐" << COLOR_RESET << endl;
        cout << COLOR_RED << COLOR_BOLD << "│  输入错误！公园编号范围为 1-20  │" << COLOR_RESET << endl;
        cout << COLOR_RED << COLOR_BOLD << "└─────────────────────────────────┘" << COLOR_RESET << endl;
        cout << COLOR_YELLOW << "请重新输入公园编号：" << COLOR_RESET;
        cin >> number;
    }
    return number;
}

int main() {
    for (int i = 0; i < 20; i++)
        for (int j = 0; j < 20; j++)
            Park_Dist_Graph[i][j] = (i == j) ? 0 : INF;

    Park_Dist_Graph[1][0] = 5.21;
    Park_Dist_Graph[8][0] = 6.87;
    Park_Dist_Graph[16][0] = 8.92;
    Park_Dist_Graph[17][0] = 4.03;
    Park_Dist_Graph[3][1] = 3.42;
    Park_Dist_Graph[6][1] = 2.96;
    Park_Dist_Graph[10][1] = 1.85;
    Park_Dist_Graph[11][1] = 2.1;
    Park_Dist_Graph[17][1] = 3.88;
    Park_Dist_Graph[3][2] = 2.78;
    Park_Dist_Graph[9][2] = 3.12;
    Park_Dist_Graph[11][2] = 2.35;
    Park_Dist_Graph[12][2] = 4.56;
    Park_Dist_Graph[4][3] = 3.87;
    Park_Dist_Graph[9][3] = 1.98;
    Park_Dist_Graph[15][3] = 1.25;
    Park_Dist_Graph[8][4] = 4.1;
    Park_Dist_Graph[14][4] = 6.33;
    Park_Dist_Graph[16][4] = 7.21;
    Park_Dist_Graph[7][5] = 7.89;
    Park_Dist_Graph[9][5] = 2.85;
    Park_Dist_Graph[12][5] = 3.02;
    Park_Dist_Graph[14][5] = 5.67;
    Park_Dist_Graph[18][5] = 8.44;
    Park_Dist_Graph[10][6] = 1.68;
    Park_Dist_Graph[13][6] = 9.21;
    Park_Dist_Graph[14][7] = 6.55;
    Park_Dist_Graph[18][7] = 4.33;
    Park_Dist_Graph[16][8] = 5.44;
    Park_Dist_Graph[17][8] = 5.12;
    Park_Dist_Graph[12][9] = 2.2;
    Park_Dist_Graph[15][9] = 1.1;
    Park_Dist_Graph[13][10] = 8.76;
    Park_Dist_Graph[12][11] = 2.65;
    Park_Dist_Graph[18][12] = 6.78;
    Park_Dist_Graph[16][13] = 3.21;
    Park_Dist_Graph[19][14] = 12.34;
    Park_Dist_Graph[19][16] = 15.67;
    Park_Dist_Graph[19][18] = 18.22;

    for (int i = 0; i < 20; i++) {
        for (int j = 0; j < i; j++) {
            Park_Dist_Graph[j][i] = Park_Dist_Graph[i][j];
        }
    }

    int choice;
    while (true) {
        Show_Menu();
        cout << COLOR_YELLOW << COLOR_BOLD << "请选择功能（1-6）: " << COLOR_RESET;
        cin >> choice;

        if (choice == 1) {
            cout << COLOR_CYAN << "\n请输入公园编号：" << COLOR_RESET;
            int id;
            cin >> id;
            Park* p = Search_By_Id(id);
            Print_Park_Info(p);
            cout << COLOR_WHITE << "\n按回车键继续..." << COLOR_RESET;
            cin.ignore();
            cin.get();
        }
        else if (choice == 2) {
            cout << COLOR_CYAN << "\n请输入公园类型：" << COLOR_RESET;
            string type;
            cin >> type;
            vector<Park> found = Search_By_Type(type);
            Print_Multiple_Parks(found);
            cout << COLOR_WHITE << "\n按回车键继续..." << COLOR_RESET;
            cin.ignore();
            cin.get();
        }
        else if (choice == 3) {
            double min_a, max_a;
            cout << COLOR_CYAN << "\n请输入面积范围（最小 最大）: " << COLOR_RESET;
            cin >> min_a >> max_a;

            if (cin.fail() || min_a < 0 || max_a < 0) {
                cout << COLOR_RED << "输入错误！请确保输入两个非负数字。\n" << COLOR_RESET;
                cin.clear();
                cin.ignore(1000, '\n');
                cout << COLOR_WHITE << "\n按回车键继续..." << COLOR_RESET;
                cin.get();
                continue;
            }

            if (min_a > max_a) {
                double temp = min_a;
                min_a = max_a;
                max_a = temp;
            }

            vector<Park> found = Search_By_Area_Range(min_a, max_a);
            Print_Multiple_Parks(found);
            cout << COLOR_WHITE << "\n按回车键继续..." << COLOR_RESET;
            cin.ignore();
            cin.get();
        }
        else if (choice == 4) {
            int sort_choice;
            while (true) {
                Show_Sort_Submenu();
                cin >> sort_choice;
                vector<Park> parkCopy = parks;

                if (sort_choice == 1) {
                    cout << COLOR_GREEN << "\n===== 选择1：冒泡排序 =====" << COLOR_RESET;
                    PrintBeforeSort();
                    BubbleSort(parkCopy);
                    PrintSortedParks(parkCopy, "冒泡排序后 - 按面积升序");
                    break;
                }
                else if (sort_choice == 2) {
                    cout << COLOR_GREEN << "\n===== 选择2：快速排序 =====" << COLOR_RESET;
                    PrintBeforeSort();
                    QuickSort(parkCopy, 0, parkCopy.size() - 1);
                    PrintSortedParks(parkCopy, "快速排序后 - 按面积升序");
                    break;
                }
                else if (sort_choice == 0) {
                    cout << COLOR_BLUE << "返回主菜单...\n" << COLOR_RESET;
                    break;
                }
                else {
                    cout << COLOR_RED << "输入错误！请选0-2之间的数字\n" << COLOR_RESET;
                    cin.clear();
                    cin.ignore(1000, '\n');
                }
            }
            cout << COLOR_WHITE << "\n按回车键继续..." << COLOR_RESET;
            cin.ignore();
            cin.get();
        }
        else if (choice == 5) {
            int start_id, end_id;
            cout << COLOR_CYAN << "\n请输入起始公园编号：" << COLOR_RESET;
            start_id = Enter_Number();
            cout << COLOR_CYAN << "请输入终止公园编号：" << COLOR_RESET;
            end_id = Enter_Number();
            vector<int> path;
            double total_dist;
            bool success = Shortest_Path(start_id, end_id, path, total_dist);
            if (success) {
                Print_Shortest_Path(start_id, end_id, path, total_dist);
            }
            else {
                cout << COLOR_RED << COLOR_BOLD << "\n┌─────────────────────────────────┐" << COLOR_RESET << endl;
                cout << COLOR_RED << COLOR_BOLD << "│    两个公园之间不存在可达路径！    │" << COLOR_RESET << endl;
                cout << COLOR_RED << COLOR_BOLD << "└─────────────────────────────────┘" << COLOR_RESET << endl;
            }
            cout << COLOR_WHITE << "\n按回车键继续..." << COLOR_RESET;
            cin.ignore();
            cin.get();
        }
        else if (choice == 6) {
            cout << COLOR_BLUE << COLOR_BOLD << "\n┌─────────────────────────────────┐" << COLOR_RESET << endl;
            cout << COLOR_BLUE << COLOR_BOLD << "│  感谢使用公园信息查询系统，再见！  │" << COLOR_RESET << endl;
            cout << COLOR_BLUE << COLOR_BOLD << "└─────────────────────────────────┘" << COLOR_RESET << endl;
            break;
        }
        else {
            cout << COLOR_RED << "无效选项，请输入 1~6。\n" << COLOR_RESET;
            cout << COLOR_WHITE << "\n按回车键继续..." << COLOR_RESET;
            cin.ignore();
            cin.get();
        }
    }
    return 0;
}