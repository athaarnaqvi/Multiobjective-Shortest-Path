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
using namespace std;

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
            //parsing edge data
            long long from = stoll(tokens[0]);
            long long to = stoll(tokens[1]);
            double obj1 = stod(tokens[2]);
            double obj2 = stod(tokens[3]);
            double obj3 = stod(tokens[4]);
            double obj4 = stod(tokens[5]);
            double obj5 = stod(tokens[6]);

	    //adding vertices if they dont exist 
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

    for (int objective = 0; objective < 5; ++objective)  //processing objectives
    {
    	//using priority queue for Dijkstra's algorithm
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

                        if (alt < target_dist) {
                            target_dist = alt;
                            vertices[v].parent = u;
                            pq.push({ alt, v });
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

    //collecting all keys to iterate over
    vector<long long> keys;
    for (const auto& pair : original_graph) {
        keys.push_back(pair.first);
    }

        unordered_map<long long, vector<Edge>> local_combined;

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
                //for each objective, check if this edge is on the shortest path
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

                    //Inverting so higher priority objectives have lower weights
                    combined_weight = (k + 1) - combined_weight;
                }
                else {
                    combined_weight = k + 1; //max weight for non-SOSP edges
                }

                local_combined[u].push_back({ e.from, e.to, combined_weight, combined_weight,
                                            combined_weight, combined_weight, combined_weight });
            }
        }

            for (auto& pair : local_combined) {
                combined_graph[pair.first].insert(combined_graph[pair.first].end(),
                    pair.second.begin(), pair.second.end());
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

        for (size_t i = 0; i < it->second.size(); ++i) {
            const Edge& e = it->second[i];
            long long v = e.to;
            double alt = mosp_vertices[u].obj1_dist + e.obj1;

                if (alt < mosp_vertices[v].obj1_dist) {
                    mosp_vertices[v].obj1_dist = alt;
                    mosp_vertices[v].parent = u;
                    pq.push({ alt, v });
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

    // Recompute only for affected vertices
    parallelSOSP_Update(graph, vertices, source);
}

int main() {
    auto start_time = high_resolution_clock::now();
    Graph graph;
    unordered_map<long long, Vertex> vertices;

    loadGraphFromCSV("japan.csv", graph, vertices);

    if (vertices.empty()) {
        cerr << "Error: No vertices loaded. Exiting.\n";
        return 1;
    }

    //long long source = 61283293;
    //long long source = 1;
    long long source = 224811793;
    if (vertices.find(source) == vertices.end()) {
        cerr << "Error: Source node " << source << " not found. Available nodes:\n";
        for (const auto& [node, _] : vertices) {
            cerr << node << "\n";
        }
        return 1;
    }

    //initial SOSP computation
    parallelSOSP_Update(graph, vertices, source);
    printTree(vertices, 0, source);
    printTree(vertices, 1, source);
    printTree(vertices, 2, source);
    printTree(vertices, 3, source);
    printTree(vertices, 4, source);

    //edge insertions with incremental update
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
    /*std::vector<Edge> inserted_edges = {
        {1, 2, 7, 10, 15, 0, 0},
        {1, 5, 4, 5, 7, 0, 0},
        {3, 5, 1, 15, 3, 0, 0}
    };*/

    for (const auto& edge : inserted_edges) {
        cout << "\nInserting Edge: " << edge.from << " -> " << edge.to << "\n";

        //using incremental update instead of full recomputation
        incrementalUpdate(graph, vertices, source, { edge });

        printTree(vertices, 0, source);
        printTree(vertices, 1, source);
        printTree(vertices, 2, source);
        printTree(vertices, 3, source);
        printTree(vertices, 4, source);
    }

    //after all insertions, combining trees and computing final MOSP
    auto combined_graph = combineSOSPTrees(graph, vertices);
    computeFinalMOSP(combined_graph, vertices, source);

    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time);
    cout << "\nTotal execution time: " << duration.count() << " ms\n";

    return 0;
}
