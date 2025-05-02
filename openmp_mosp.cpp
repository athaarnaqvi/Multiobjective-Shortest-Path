#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <cfloat>
#include <omp.h>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <cmath>
#include <functional>

using namespace std::chrono;

struct Edge {
    long long from, to;
    double length, travel_time, cost;
};

using Graph = std::unordered_map<long long, std::vector<Edge>>;

struct Vertex {
    double length_dist = DBL_MAX;
    double time_dist = DBL_MAX;
    double cost_dist = DBL_MAX;
    long long parent = -1;
    bool affected = false;
};

void loadGraphFromCSV(const std::string& filename, Graph& graph, std::unordered_map<long long, Vertex>& vertices) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open file: " << filename << "\n";
        exit(1);
    }

    std::string line;
    int line_num = 0;
    int edges_loaded = 0;

    // Skip header
    if (!std::getline(file, line)) {
        std::cerr << "Error: Empty file or header missing\n";
        return;
    }
    line_num++;

    while (std::getline(file, line)) {
        line_num++;
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::vector<std::string> tokens;
        std::string token;

        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }

        if (tokens.size() != 5) {
            std::cerr << "Warning: Skipping malformed line " << line_num << ": " << line << "\n";
            continue;
        }

        try {
            long long from = std::stoll(tokens[0]);
            long long to = std::stoll(tokens[1]);
            double length = std::stod(tokens[2]);
            double time = std::stod(tokens[3]);
            double cost = std::stod(tokens[4]);

            if (vertices.find(from) == vertices.end()) {
                vertices[from] = Vertex();
            }
            if (vertices.find(to) == vertices.end()) {
                vertices[to] = Vertex();
            }

            graph[from].push_back({ from, to, length, time, cost });
            edges_loaded++;
        }
        catch (const std::exception& e) {
            std::cerr << "Error: Skipping invalid line " << line_num << ": " << line
                << " | Reason: " << e.what() << "\n";
            continue;
        }
    }

    std::cout << "Successfully loaded " << edges_loaded << " edges and "
        << vertices.size() << " vertices from " << filename << "\n";
}

void parallelSOSP_Update(Graph& graph, std::unordered_map<long long, Vertex>& vertices, long long source) {
    vertices[source].length_dist = 0;
    vertices[source].time_dist = 0;
    vertices[source].cost_dist = 0;

    // Parallelize the three Dijkstra runs since they're independent
#pragma omp parallel for
    for (int objective = 0; objective < 3; ++objective) {
        std::priority_queue<std::pair<double, long long>,
            std::vector<std::pair<double, long long>>,
            std::greater<>> pq;
        pq.push({ 0.0, source });

        while (!pq.empty()) {
            auto [dist_u, u] = pq.top();
            pq.pop();

            double current_dist;
            switch (objective) {
            case 0: current_dist = vertices[u].length_dist; break;
            case 1: current_dist = vertices[u].time_dist; break;
            case 2: current_dist = vertices[u].cost_dist; break;
            default: continue;
            }

            if (dist_u > current_dist) continue;

            // Parallelize edge processing
#pragma omp parallel for
            for (size_t i = 0; i < graph[u].size(); ++i) {
                const Edge& e = graph[u][i];
                long long v = e.to;
                double alt = current_dist;
                double edge_weight = 0;

                switch (objective) {
                case 0: edge_weight = e.length; break;
                case 1: edge_weight = e.travel_time; break;
                case 2: edge_weight = e.cost; break;
                }
                alt += edge_weight;

                double& target_dist = [&]() -> double& {
                    switch (objective) {
                    case 0: return vertices[v].length_dist;
                    case 1: return vertices[v].time_dist;
                    case 2: return vertices[v].cost_dist;
                    default: return vertices[v].length_dist;
                    }
                    }();

                // Critical section for distance updates
#pragma omp critical
                    {
                        if (alt < target_dist) {
                            target_dist = alt;
                            vertices[v].parent = u;
                            pq.push({ alt, v });
                        }
                    }
            }
        }
    }
}

void printTree(const std::unordered_map<long long, Vertex>& vertices, int objective, long long source) {
    const char* objective_names[] = { "Length", "Time", "Cost" };
    std::cout << "\nSOSP Tree for Objective: " << objective_names[objective] << "\n";

    for (const auto& [node, vtx] : vertices) {
        double dist;
        switch (objective) {
        case 0: dist = vtx.length_dist; break;
        case 1: dist = vtx.time_dist; break;
        case 2: dist = vtx.cost_dist; break;
        default: dist = DBL_MAX;
        }

        if (dist < DBL_MAX) {
            std::cout << "Node " << node << ": Distance = " << dist;
            if (node != source) {
                std::cout << ", Parent = " << vtx.parent;
            }
            std::cout << "\n";
        }
    }
}

std::unordered_map<long long, std::vector<Edge>> combineSOSPTrees(
    const Graph& original_graph,
    const std::unordered_map<long long, Vertex>& vertices,
    const std::vector<double>& objective_weights = { 1.0, 1.0, 1.0 }) {

    std::unordered_map<long long, std::vector<Edge>> combined_graph;
    const int k = 3;

    // First collect all keys to iterate over
    std::vector<long long> keys;
    for (const auto& pair : original_graph) {
        keys.push_back(pair.first);
    }

    // Parallelize over the keys
#pragma omp parallel
    {
        std::unordered_map<long long, std::vector<Edge>> local_combined;

#pragma omp for nowait
        for (size_t i = 0; i < keys.size(); ++i) {
            long long u = keys[i];
            auto it = original_graph.find(u);
            if (it == original_graph.end()) continue;

            const auto& edges = it->second;
            auto from_it = vertices.find(u);
            if (from_it == vertices.end()) continue;

            for (const Edge& e : edges) {
                auto to_it = vertices.find(e.to);
                if (to_it == vertices.end()) continue;

                const Vertex& from_vtx = from_it->second;
                const Vertex& to_vtx = to_it->second;

                double combined_weight = 0;
                if (to_vtx.parent == u) {
                    if (std::abs(to_vtx.length_dist - (from_vtx.length_dist + e.length)) < 1e-6)
                        combined_weight += objective_weights[0];
                    if (std::abs(to_vtx.time_dist - (from_vtx.time_dist + e.travel_time)) < 1e-6)
                        combined_weight += objective_weights[1];
                    if (std::abs(to_vtx.cost_dist - (from_vtx.cost_dist + e.cost)) < 1e-6)
                        combined_weight += objective_weights[2];

                    // Invert so higher priority objectives have lower weights
                    combined_weight = (k + 1) - combined_weight;
                }
                else {
                    combined_weight = k + 1; // Max weight for non-SOSP edges
                }

                local_combined[u].push_back({ e.from, e.to, combined_weight, combined_weight, combined_weight });
            }
        }

        // Merge local results
#pragma omp critical
        {
            for (auto& pair : local_combined) {
                combined_graph[pair.first].insert(combined_graph[pair.first].end(),
                    pair.second.begin(), pair.second.end());
            }
        }
    }

    return combined_graph;
}

void computeFinalMOSP(const Graph& combined_graph,
    const std::unordered_map<long long, Vertex>& vertices,
    long long source) {
    std::unordered_map<long long, Vertex> mosp_vertices;

    // Initialize vertices
    for (const auto& [node, _] : combined_graph) {
        mosp_vertices[node] = Vertex();
    }

    if (mosp_vertices.find(source) == mosp_vertices.end()) {
        std::cerr << "Error: Source node " << source << " not found in combined graph\n";
        return;
    }

    mosp_vertices[source].length_dist = 0;

    std::priority_queue<std::pair<double, long long>,
        std::vector<std::pair<double, long long>>,
        std::greater<>> pq;
    pq.push({ 0.0, source });

    while (!pq.empty()) {
        auto [dist_u, u] = pq.top(); pq.pop();
        if (dist_u > mosp_vertices[u].length_dist) continue;

        auto it = combined_graph.find(u);
        if (it == combined_graph.end()) continue;

        // Parallelize edge processing
#pragma omp parallel for
        for (size_t i = 0; i < it->second.size(); ++i) {
            const Edge& e = it->second[i];
            long long v = e.to;
            double alt = mosp_vertices[u].length_dist + e.length;

            // Critical section for distance updates
#pragma omp critical
            {
                if (alt < mosp_vertices[v].length_dist) {
                    mosp_vertices[v].length_dist = alt;
                    mosp_vertices[v].parent = u;
                    pq.push({ alt, v });
                }
            }
        }
    }

    std::cout << "\nFinal MOSP Tree with All Objective Values:\n";
    std::cout << "Node | Combined Weight | Parent | (Length, Time, Cost)\n";
    for (const auto& [node, vtx] : mosp_vertices) {
        if (vtx.length_dist < DBL_MAX) {
            std::cout << "Node " << node << ": " << vtx.length_dist;
            if (node != source) {
                std::cout << ", Parent = " << vtx.parent;
            }

            auto orig_it = vertices.find(node);
            if (orig_it != vertices.end()) {
                const Vertex& orig_vtx = orig_it->second;
                std::cout << ", Objectives = (" << orig_vtx.length_dist << ", "
                    << orig_vtx.time_dist << ", " << orig_vtx.cost_dist << ")";
            }
            std::cout << "\n";
        }
    }
}

void incrementalUpdate(Graph& graph, std::unordered_map<long long, Vertex>& vertices,
    long long source, const std::vector<Edge>& inserted_edges) {
    // Mark affected vertices
    std::unordered_set<long long> affected_vertices;
    for (const auto& edge : inserted_edges) {
        affected_vertices.insert(edge.from);
        affected_vertices.insert(edge.to);
        graph[edge.from].push_back(edge);

        if (vertices.find(edge.from) == vertices.end()) {
            vertices[edge.from] = Vertex();
        }
        if (vertices.find(edge.to) == vertices.end()) {
            vertices[edge.to] = Vertex();
        }
    }

    // Recompute only for affected vertices
    parallelSOSP_Update(graph, vertices, source);
}

int main() {
    omp_set_num_threads(omp_get_max_threads());
    auto start_time = high_resolution_clock::now();
    Graph graph;
    std::unordered_map<long long, Vertex> vertices;

    loadGraphFromCSV("multi_obj_graph.csv", graph, vertices);
    //loadGraphFromCSV("small_graph.csv", graph, vertices);

    if (vertices.empty()) {
        std::cerr << "Error: No vertices loaded. Exiting.\n";
        return 1;
    }

    long long source = 61283293;
    //long long source = 1;
    if (vertices.find(source) == vertices.end()) {
        std::cerr << "Error: Source node " << source << " not found. Available nodes:\n";
        for (const auto& [node, _] : vertices) {
            std::cerr << node << "\n";
        }
        return 1;
    }

    // Initial SOSP computation
    parallelSOSP_Update(graph, vertices, source);
    printTree(vertices, 0, source);
    printTree(vertices, 1, source);
    printTree(vertices, 2, source);

    // Edge insertions with incremental update
    std::vector<Edge> inserted_edges = {
        {61283293, 61283126, 50, 10, 0},
        {61323022, 61283287, 30, 5, 0},
        {61283322, 61283218, 80, 15, 0}
    };
    /*std::vector<Edge> inserted_edges = {
        {1, 2, 7, 10, 15},
        {1, 5, 4, 5, 7},
        {3, 5, 1, 15, 3}
    };*/

    for (const auto& edge : inserted_edges) {
        std::cout << "\nInserting Edge: " << edge.from << " -> " << edge.to << "\n";

        // Use incremental update instead of full recomputation
        incrementalUpdate(graph, vertices, source, { edge });

        printTree(vertices, 0, source);
        printTree(vertices, 1, source);
        printTree(vertices, 2, source);
    }

    // After all insertions: combine trees and compute final MOSP
    auto combined_graph = combineSOSPTrees(graph, vertices);
    computeFinalMOSP(combined_graph, vertices, source);

    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time);
    std::cout << "\nTotal execution time: " << duration.count() << " ms\n";

    return 0;
}