# Gerekli kütüphaneler
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import networkx as nx
from heapq import heappop, heappush

# Excel'den Veri Okuma
distances = pd.read_excel("/workspaces/Machine-Learninig-Project/in.xlsx", index_col=0)
aid_demand = pd.read_excel("/workspaces/Machine-Learninig-Project/yt.xlsx", index_col=0)

# Depoların Koordinatları
depots = {
    "Kuzey Depo": (0, 100),
    "Güney Depo": (0, -100),
    "Doğu Depo": (100, 0)
}

# X ve Y Koordinatlarını Mesafe Matrisi Kullanarak Hesaplama
needs_coords = pd.DataFrame(index=aid_demand.index)

# Referans depo olarak Kuzey Depo'yu kullanarak koordinat hesaplama
reference_depot = "Kuzey Depo"
needs_coords["X"] = distances.loc[reference_depot] * np.cos(np.linspace(0, 2 * np.pi, len(aid_demand)))
needs_coords["Y"] = distances.loc[reference_depot] * np.sin(np.linspace(0, 2 * np.pi, len(aid_demand)))
needs_coords["En Yakın Depo"] = aid_demand["En Yakın Depo"]

# Drone kapasitesi
DRONE_CAPACITY = 30

# Kümeleme (k=5)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
needs_coords["Cluster"] = kmeans.fit_predict(needs_coords[["X", "Y"]])

# A* Algoritması
def a_star_algorithm(graph, start, goal):
    open_list = [(0, start)]  # (cost, node)
    came_from = {}
    cost_so_far = {start: 0}
    
    while open_list:
        current_cost, current_node = heappop(open_list)
        if current_node == goal:
            break
        for neighbor in graph.neighbors(current_node):
            weight = graph.edges[current_node, neighbor]['weight']
            new_cost = cost_so_far[current_node] + weight
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                heappush(open_list, (new_cost, neighbor))
                came_from[neighbor] = current_node
    
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = came_from[node]
    path.append(start)
    path.reverse()
    return path, cost_so_far[goal]

# Drone Yükleme Stratejisi ve Rota Optimizasyonu
def optimize_drones(aid_demand, drone_capacity, clusters):
    drone_usage = {depo: 0 for depo in depots}  # Depolar için drone sayısı
    total_drones = 0
    drone_details = []

    for cluster in sorted(clusters["Cluster"].unique()):
        cluster_points = clusters[clusters["Cluster"] == cluster]
        current_depo = cluster_points["En Yakın Depo"].iloc[0]
        drone_counter = 1

        # İhtiyaç noktalarına yük dağıtma
        visited_points = set()
        current_load_a, current_load_b = 0, 0
        route = [current_depo]  # Depodan başlar
        current_distance = 0

        for point in cluster_points.index:
            if point in visited_points:
                continue
            
            tibbiyuk = aid_demand.loc[point, "Tıbbi Malzeme"]
            yiyecek_yuk = aid_demand.loc[point, "Yiyecek"]
            
            while tibbiyuk > 0 or yiyecek_yuk > 0:
                available_a = min(tibbiyuk, drone_capacity - current_load_a)
                available_b = min(yiyecek_yuk, drone_capacity - (current_load_a + available_a))
                
                current_load_a += available_a
                current_load_b += available_b
                tibbiyuk -= available_a
                yiyecek_yuk -= available_b
                
                if available_a > 0 or available_b > 0:
                    route.append(point)
                    visited_points.add(point)
                    current_distance += distances.loc[route[-2], route[-1]]
                    
                # Drone kapasitesi dolduysa geri dön
                if current_load_a + current_load_b == drone_capacity:
                    current_distance += distances.loc[route[-1], current_depo]
                    route.append(current_depo)
                    drone_details.append({
                        "Kume": cluster,
                        "Drone": f"{current_depo} - Drone {drone_counter}",
                        "Rota": route,
                        "Tasınan A (Tıbbi Malzeme)": current_load_a,
                        "Tasınan B (Yiyecek)": current_load_b,
                        "Toplam Mesafe": current_distance
                    })
                    drone_usage[current_depo] += 1
                    total_drones += 1
                    drone_counter += 1
                    
                    # Yeni drone için sıfırlama
                    current_load_a, current_load_b = 0, 0
                    route = [current_depo]
                    current_distance = 0
        
        # Kalan yük için drone dönüş
        if len(route) > 1:
            current_distance += distances.loc[route[-1], current_depo]
            route.append(current_depo)
            drone_details.append({
                "Kume": cluster,
                "Drone": f"{current_depo} - Drone {drone_counter}",
                "Rota": route,
                "Tasınan A (Tıbbi Malzeme)": current_load_a,
                "Tasınan B (Yiyecek)": current_load_b,
                "Toplam Mesafe": current_distance
            })
            drone_usage[current_depo] += 1
            total_drones += 1
    
    return drone_details, total_drones, drone_usage

# Optimizasyon çalıştırma
drone_details, total_drones, drone_usage = optimize_drones(aid_demand, DRONE_CAPACITY, needs_coords)

# Kümeleme ve Depo Grafiği
plt.figure(figsize=(10, 8))
for cluster in sorted(needs_coords["Cluster"].unique()):
    cluster_points = needs_coords[needs_coords["Cluster"] == cluster]
    plt.scatter(cluster_points["X"], cluster_points["Y"], label=f"Cluster {cluster}")
for depo, coord in depots.items():
    plt.scatter(coord[0], coord[1], color='red', marker='x', s=200, label=f"{depo}")
plt.title("Kümeleme Grafiği (k = 5)")
plt.xlabel("X Koordinatı")
plt.ylabel("Y Koordinatı")
plt.legend()
plt.grid(True)
plt.show()

# Sonuçları Gösterme
current_cluster = -1
for detail in drone_details:
    if detail["Kume"] != current_cluster:
        current_cluster = detail["Kume"]
        print(f"\nKüme {current_cluster}:")
    print(f"{detail['Drone']}: {detail['Rota']}")
    print(f"  Tasınan A (Tıbbi Malzeme): {detail['Tasınan A (Tıbbi Malzeme)']}, B (Yiyecek): {detail['Tasınan B (Yiyecek)']}")
    print(f"  Toplam Mesafe: {detail['Toplam Mesafe']:.2f} km")

print("\nToplam Kullanılan Drone Sayısı:", total_drones)
print("Her Depodan Kullanılan Drone Sayısı:")
for depo, count in drone_usage.items():
    print(f"{depo}: {count} drone")