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

using namespace std;
using namespace std::chrono;

struct Edge {
    long long from, to;
    double obj1, obj2, obj3, obj4, obj5;
};

using Graph = unordered_map<long long, vector<Edge>>;

struct Vertex {
    double obj1_dist = DBL_MAX;
    double obj2_dist = DBL_MAX;
    double obj3_dist = DBL_MAX;
    double obj4_dist = DBL_MAX;
    double obj5_dist = DBL_MAX;
    long long parent = -1;
    bool affected = false;
};

void loadGraphFromCSV(const string& filename, Graph& graph, unordered_map<long long, Vertex>& vertices) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Failed to open file: " << filename << "\n";
        exit(1);
    }

    string line;
    int line_num = 0;
    int edges_loaded = 0;

    if (!getline(file, line)) {
        cerr << "Error: Empty file or header missing\n";
        return;
    }
    line_num++;

    while (getline(file, line)) {
        line_num++;
        if (line.empty()) continue;

        stringstream ss(line);
        vector<string> tokens;
        string token;

        while (getline(ss, token, ',')) {
            tokens.push_back(token);
        }

        if (tokens.size() != 7) {
            cerr << "Warning: Skipping malformed line " << line_num << ": " << line << "\n";
            continue;
        }

        try {
            long long from = stoll(tokens[0]);
            long long to = stoll(tokens[1]);
            double obj1 = stod(tokens[2]);
            double obj2 = stod(tokens[3]);
            double obj3 = stod(tokens[4]);
            double obj4 = stod(tokens[5]);
            double obj5 = stod(tokens[6]);

            if (vertices.find(from) == vertices.end()) {
                vertices[from] = Vertex();
            }
            if (vertices.find(to) == vertices.end()) {
                vertices[to] = Vertex();
            }

            graph[from].push_back({ from, to, obj1, obj2, obj3, obj4, obj5 });
            edges_loaded++;
        }
        catch (const exception& e) {
            cerr << "Error: Skipping invalid line " << line_num << ": " << line
                << " | Reason: " << e.what() << "\n";
            continue;
        }
    }

    cout << "Successfully loaded " << edges_loaded << " edges and "
        << vertices.size() << " vertices from " << filename << "\n";
}

void parallelSOSP_Update(Graph& graph, unordered_map<long long, Vertex>& vertices, long long source) {
    vertices[source].obj1_dist = 0;
    vertices[source].obj2_dist = 0;
    vertices[source].obj3_dist = 0;
    vertices[source].obj4_dist = 0;
    vertices[source].obj5_dist = 0;

    //parallelizing dijakstra for 5 objectives
#pragma omp parallel for
    for (int objective = 0; objective < 5; ++objective) {
        priority_queue<pair<double, long long>,
            vector<pair<double, long long>>,
            greater<>> pq;
        pq.push({ 0.0, source });

        while (!pq.empty()) {
            auto [dist_u, u] = pq.top();
            pq.pop();

            double current_dist;
            switch (objective) {
            case 0: current_dist = vertices[u].obj1_dist; break;
            case 1: current_dist = vertices[u].obj2_dist; break;
            case 2: current_dist = vertices[u].obj3_dist; break;
            case 3: current_dist = vertices[u].obj4_dist; break;
            case 4: current_dist = vertices[u].obj5_dist; break;
            default: continue;
            }

            if (dist_u > current_dist) continue;

            //parallelizing edge processing
#pragma omp parallel for
            for (size_t i = 0; i < graph[u].size(); ++i) {
                const Edge& e = graph[u][i];
                long long v = e.to;
                double alt = current_dist;
                double edge_weight = 0;

                switch (objective) {
                case 0: edge_weight = e.obj1; break;
                case 1: edge_weight = e.obj2; break;
                case 2: edge_weight = e.obj3; break;
                case 3: edge_weight = e.obj4; break;
                case 4: edge_weight = e.obj5; break;
                }
                alt += edge_weight;

                double& target_dist = [&]() -> double& {
                    switch (objective) {
                    case 0: return vertices[v].obj1_dist;
                    case 1: return vertices[v].obj2_dist;
                    case 2: return vertices[v].obj3_dist;
                    case 3: return vertices[v].obj4_dist;
                    case 4: return vertices[v].obj5_dist;
                    default: return vertices[v].obj1_dist;
                    }
                    }();

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

void printTree(const unordered_map<long long, Vertex>& vertices, int objective, long long source) {
    const char* objective_names[] = { "Objective1", "Objective2", "Objective3", "Objective4", "Objective5" };
    cout << "\nSOSP Tree for Objective: " << objective_names[objective] << "\n";

    for (const auto& [node, vtx] : vertices) {
        double dist;
        switch (objective) {
        case 0: dist = vtx.obj1_dist; break;
        case 1: dist = vtx.obj2_dist; break;
        case 2: dist = vtx.obj3_dist; break;
        case 3: dist = vtx.obj4_dist; break;
        case 4: dist = vtx.obj5_dist; break;
        default: dist = DBL_MAX;
        }

        if (dist < DBL_MAX) {
            cout << "Node " << node << ": Distance = " << dist;
            if (node != source) {
                cout << ", Parent = " << vtx.parent;
            }
            cout << "\n";
        }
    }
}

unordered_map<long long, vector<Edge>> combineSOSPTrees(
    const Graph& original_graph,
    const unordered_map<long long, Vertex>& vertices,
    const vector<double>& objective_weights = { 1.0, 1.0, 1.0, 1.0, 1.0 }) {

    unordered_map<long long, vector<Edge>> combined_graph;
    const int k = 5; 

    vector<long long> keys;
    for (const auto& pair : original_graph) {
        keys.push_back(pair.first);
    }

    //parallelizing over keys
#pragma omp parallel
    {
        unordered_map<long long, vector<Edge>> local_combined;

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
                    if (abs(to_vtx.obj1_dist - (from_vtx.obj1_dist + e.obj1)) < 1e-6)
                        combined_weight += objective_weights[0];
                    if (abs(to_vtx.obj2_dist - (from_vtx.obj2_dist + e.obj2)) < 1e-6)
                        combined_weight += objective_weights[1];
                    if (abs(to_vtx.obj3_dist - (from_vtx.obj3_dist + e.obj3)) < 1e-6)
                        combined_weight += objective_weights[2];
                    if (abs(to_vtx.obj4_dist - (from_vtx.obj4_dist + e.obj4)) < 1e-6)
                        combined_weight += objective_weights[3];
                    if (abs(to_vtx.obj5_dist - (from_vtx.obj5_dist + e.obj5)) < 1e-6)
                        combined_weight += objective_weights[4];

                    //higher priority objectives have lower weights
                    combined_weight = (k + 1) - combined_weight;
                }
                else {
                    combined_weight = k + 1; 
                }

                local_combined[u].push_back({ e.from, e.to, combined_weight, combined_weight,
                                            combined_weight, combined_weight, combined_weight });
            }
        }

        //merging local results
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
    const unordered_map<long long, Vertex>& vertices,
    long long source) {
    unordered_map<long long, Vertex> mosp_vertices;

    for (const auto& [node, _] : combined_graph) {
        mosp_vertices[node] = Vertex();
    }

    if (mosp_vertices.find(source) == mosp_vertices.end()) {
        cerr << "Error: Source node " << source << " not found in combined graph\n";
        return;
    }

    mosp_vertices[source].obj1_dist = 0;

    priority_queue<pair<double, long long>,
        vector<pair<double, long long>>,
        greater<>> pq;
    pq.push({ 0.0, source });

    while (!pq.empty()) {
        auto [dist_u, u] = pq.top(); pq.pop();
        if (dist_u > mosp_vertices[u].obj1_dist) continue;

        auto it = combined_graph.find(u);
        if (it == combined_graph.end()) continue;

#pragma omp parallel for
        for (size_t i = 0; i < it->second.size(); ++i) {
            const Edge& e = it->second[i];
            long long v = e.to;
            double alt = mosp_vertices[u].obj1_dist + e.obj1;

#pragma omp critical
            {
                if (alt < mosp_vertices[v].obj1_dist) {
                    mosp_vertices[v].obj1_dist = alt;
                    mosp_vertices[v].parent = u;
                    pq.push({ alt, v });
                }
            }
        }
    }

    cout << "\nFinal MOSP Tree with All Objective Values:\n";
    cout << "Node | Combined Weight | Parent | (Obj1, Obj2, Obj3, Obj4, Obj5)\n";
    for (const auto& [node, vtx] : mosp_vertices) {
        if (vtx.obj1_dist < DBL_MAX) {
            cout << "Node " << node << ": " << vtx.obj1_dist;
            if (node != source) {
                cout << ", Parent = " << vtx.parent;
            }

            auto orig_it = vertices.find(node);
            if (orig_it != vertices.end()) {
                const Vertex& orig_vtx = orig_it->second;
                cout << ", Objectives = (" << orig_vtx.obj1_dist << ", "
                    << orig_vtx.obj2_dist << ", " << orig_vtx.obj3_dist << ", "
                    << orig_vtx.obj4_dist << ", " << orig_vtx.obj5_dist << ")";
            }
            cout << "\n";
        }
    }
}

void incrementalUpdate(Graph& graph, unordered_map<long long, Vertex>& vertices,
    long long source, const vector<Edge>& inserted_edges) {
    //marking affected vertices
    unordered_set<long long> affected_vertices;
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

    //recomputing for affected vertices only
    parallelSOSP_Update(graph, vertices, source);
}

int main() {
    omp_set_num_threads(omp_get_max_threads());
    auto start_time = high_resolution_clock::now();
    Graph graph;
    unordered_map<long long, Vertex> vertices;

    //loadGraphFromCSV("multi_obj_graph_zurich.csv", graph, vertices);
    //loadGraphFromCSV("multi_obj_graph_usa.csv", graph, vertices);
    //loadGraphFromCSV("multi_obj_graph_switzerland.csv", graph, vertices);
    //loadGraphFromCSV("multi_obj_graph_netherland.csv", graph, vertices);
    loadGraphFromCSV("multi_obj_graph_Japan.csv", graph, vertices);

    if (vertices.empty()) {
        cerr << "Error: No vertices loaded. Exiting.\n";
        return 1;
    }

    //long long source = 453768;
    //long long source = 61182955;
    //long long source = 453768;
    //long long source = 6316199;
    long long source = 224811793;

    if (vertices.find(source) == vertices.end()) {
        cerr << "Error: Source node " << source << " not found. Available nodes:\n";
        for (const auto& [node, _] : vertices) {
            cerr << node << "\n";
        }
        return 1;
    }

    parallelSOSP_Update(graph, vertices, source);
    printTree(vertices, 0, source);
    printTree(vertices, 1, source);
    printTree(vertices, 2, source);
    printTree(vertices, 3, source);
    printTree(vertices, 4, source);

    /*vector<Edge> inserted_edges = {
        { 453768, 453814, 250.0, 19.0, 0, 0.12, 22.0 },
        {455416, 26807316,122.5, 17.5, 0, 0.03, 14.0},
        {26852721, 27005869, 230.0, 34.0, 0, 0.015, 38.0},
        {27006042, 27256986, 122.5, 12.8, 0, 0.39, 22.0},
        {27429337, 27488745, 139.0, 20.6, 1, 0.91, 23.0}
    };*/
    /*vector<Edge> inserted_edges = {
        {61182955, 61323205, 96.0, 17.2, 0.0077, 18.5, 2},
        {61283119, 61283127, 106.0, 19.0, 0.0085, 20.1, 1},
        {61283126, 61321150, 105.0, 18.8, 0.0083, 19.9, 1},
        {61283218, 61324635, 455.0, 65.5, 0.0365, 87.0, 1},
        {61283269, 61321315, 270.0, 38.8, 0.0216, 51.8, 3},
        {61283269, 6100432140, 500.0, 72.0, 0.0400, 96.0, 3},
        {61283269, 61323025, 120.0, 17.5, 0.0096, 23.0, 4},
        {61283273, 61283290, 162.0, 29.0, 0.0130, 31.0, 1},
        {61283273, 61283335, 86.0, 15.5, 0.0070, 16.6, 2},
        {61283273, 61283345, 75.0, 13.3, 0.0060, 14.3, 2},
        {61283287, 61321750, 73.0, 17.4, 0.0059, 14.0, 1}
    };*/
    /*vector<Edge> inserted_edges = {
        { 453768, 453814, 300.50, 35.20, 0.024, 58.70, 2 },
        { 453805, 52727137, 120.75, 15.50, 0.0095, 22.50, 2 },
        { 453805, 74574330, 280.00, 20.00, 0.0230, 55.00, 3 },
        { 453810, 47744453, 260.30, 31.00, 0.0200, 50.00, 1 },
        { 453814, 3112781363, 230.10, 27.00, 0.0180, 44.00, 2 },
        { 453816, 28579781, 190.65, 14.90, 0.0120, 28.00, 3 },
        { 453818, 1413703439, 30.00, 5.00, 0.0025, 6.00, 1 },
        { 453818, 301216590, 10.50, 1.00, 0.0005, 1.50, 1 },
        { 453828, 847222965, 75.00, 8.50, 0.0058, 13.50, 1 },
        { 453810, 453818, 200.00, 25.00, 0.0150, 33.00, 2 }
    };*/
    /*vector<Edge> inserted_edges = {
        { 6316199, 46389218, 230.00, 28.00, 0.0185, 44.00, 1 },
        { 25596477, 4489285115, 85.00, 10.20, 0.0068, 16.00, 1 },
        { 25658579, 1334338691, 40.00, 4.80, 0.0031, 7.50, 1 },
        { 26203121, 494157307, 90.00, 10.80, 0.0072, 17.00, 1 },
        { 26206556, 916874958, 100.00, 12.00, 0.0080, 19.00, 1 },
        { 26371846, 331526066, 140.00, 17.00, 0.0112, 27.00, 1 },
        { 26371856, 46334837, 60.00, 7.20, 0.0048, 11.50, 1 },
        { 26586128, 5828012924, 28.00, 3.40, 0.0023, 5.20, 1 },
        { 26596415, 46377905, 12.00, 1.40, 0.0010, 2.20, 1 },
        { 27219290, 469250875, 155.00, 18.60, 0.0123, 29.50, 1 }
    };*/
    vector<Edge> inserted_edges = {
        { 224811793, 12017420150, 200.00, 24.00, 0.0159, 38.00, 1 },
        { 224811793, 438102571, 110.00, 13.20, 0.0088, 21.00, 1 },
        { 224811818, 402756500, 142.00, 10.20, 0.0114, 27.00, 2 },
        { 224811845, 412144985, 140.00, 10.10, 0.0113, 27.00, 2 },
        { 224811905, 434344368, 170.00, 15.30, 0.0136, 32.50, 4 },
        { 224811905, 3834323577, 230.00, 21.00, 0.0183, 43.80, 1 },
        { 243776548, 3256136833, 410.00, 49.30, 0.0329, 78.80, 2 },
        { 243776548, 1332356080, 130.00, 15.60, 0.0104, 25.00, 1 },
        { 243776563, 1323463567, 52.00, 6.20, 0.0041, 10.00, 2 },
        { 243776563, 1323463523, 40.00, 4.80, 0.0032, 7.80, 1 }
    };

    for (const auto& edge : inserted_edges) {
        cout << "\nInserting Edge: " << edge.from << " -> " << edge.to << "\n";

        incrementalUpdate(graph, vertices, source, { edge });

        printTree(vertices, 0, source);
        printTree(vertices, 1, source);
        printTree(vertices, 2, source);
        printTree(vertices, 3, source);
        printTree(vertices, 4, source);
    }

    auto combined_graph = combineSOSPTrees(graph, vertices);
    computeFinalMOSP(combined_graph, vertices, source);

    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time);
    cout << "\nTotal execution time: " << duration.count() << " ms\n";

    return 0;
}
