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
#include <metis.h>
#include <mpi.h>

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

    // Skip header
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

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        cout << "Successfully loaded " << edges_loaded << " edges and "
            << vertices.size() << " vertices from " << filename << "\n";
    }
}

void parallelSOSP_Update(Graph& graph, unordered_map<long long, Vertex>& vertices,
    long long source, int rank, int num_procs) {
    // only the process that owns the source node initializes it
    if (vertices.find(source) != vertices.end()) {
        vertices[source].obj1_dist = 0;
        vertices[source].obj2_dist = 0;
        vertices[source].obj3_dist = 0;
        vertices[source].obj4_dist = 0;
        vertices[source].obj5_dist = 0;
    }

    // broadcast initial distances from source 
    double init_vals[5] = { 0, 0, 0, 0, 0 };
    MPI_Bcast(init_vals, 5, MPI_DOUBLE, rank, MPI_COMM_WORLD);

    // running parallely as they are independant
#pragma omp parallel for
    for (int objective = 0; objective < 5; ++objective) {
        priority_queue<pair<double, long long>,
            vector<pair<double, long long>>,
            greater<>> pq;

        // only push source if we have it
        if (vertices.find(source) != vertices.end()) {
            pq.push({ 0.0, source });
        }

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

            auto it = graph.find(u);
            if (it == graph.end()) continue;

            // parallelize edge processing
#pragma omp parallel for
            for (size_t i = 0; i < it->second.size(); ++i) {
                const Edge& e = it->second[i];
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

                //  distance updates 
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

void printTree(const unordered_map<long long, Vertex>& vertices, int objective,
    long long source, int rank) {
    if (rank != 0) return; //  print from rank 0

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
    const int k = 5; //number of objectives

    // all keys - iteration one by one -> collection
    vector<long long> keys;
    for (const auto& pair : original_graph) {
        keys.push_back(pair.first);
    }

    // keys - parallel
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

                    // invert so higher priority objectives have lower weights
                    combined_weight = (k + 1) - combined_weight;
                }
                else {
                    combined_weight = k + 1; // max weight for non-SOSP edges
                }

                local_combined[u].push_back({ e.from, e.to, combined_weight, combined_weight,
                                            combined_weight, combined_weight, combined_weight });
            }
        }

        //local results merging
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
    long long source, int rank) {
    if (rank != 0) return;

    unordered_map<long long, Vertex> mosp_vertices;

    // initialize vertices all the info we need
    for (const auto& [node, vtx] : vertices) {
        mosp_vertices[node] = vtx;
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

        // parallelize edge processing
#pragma omp parallel for
        for (size_t i = 0; i < it->second.size(); ++i) {
            const Edge& e = it->second[i];
            long long v = e.to;
            double alt = mosp_vertices[u].obj1_dist + e.obj1;

            // critical section for distance updates
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

            // use the original vertex data from the gathered results
            auto orig_it = vertices.find(node);
            if (orig_it != vertices.end()) {
                const Vertex& orig_vtx = orig_it->second;
                cout << ", Objectives = ("
                    << orig_vtx.obj1_dist << ", "
                    << orig_vtx.obj2_dist << ", "
                    << orig_vtx.obj3_dist << ", "
                    << orig_vtx.obj4_dist << ", "
                    << orig_vtx.obj5_dist << ")";
            }
            cout << "\n";
        }
    }
}

void incrementalUpdate(Graph& graph, unordered_map<long long, Vertex>& vertices,
    long long source, const vector<Edge>& inserted_edges, int rank) {
    // mark affected vertices
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

    // recompute only for affected vertices
    parallelSOSP_Update(graph, vertices, source, rank, 1);
}

void convertToMetisFormat(const Graph& graph,
    vector<idx_t>& xadj,
    vector<idx_t>& adjncy,
    unordered_map<long long, idx_t>& id_map,
    vector<long long>& reverse_map) {
    idx_t index = 0;
    for (const auto& [node, _] : graph) {
        id_map[node] = index++;
        reverse_map.push_back(node);
    }

    xadj.push_back(0);
    for (const auto& [node, edges] : graph) {
        for (const auto& edge : edges) {
            adjncy.push_back(id_map[edge.to]);
        }
        xadj.push_back(adjncy.size());
    }
}

void partitionGraphWithMETIS(const Graph& graph, int nparts,
    unordered_map<long long, int>& node_partitions) {
    // metic input
    vector<idx_t> xadj, adjncy;
    unordered_map<long long, idx_t> id_map;
    vector<long long> reverse_map;

    // to metis format
    convertToMetisFormat(graph, xadj, adjncy, id_map, reverse_map);

    // validate input
    idx_t nvtxs = reverse_map.size();
    if (nvtxs == 0 || xadj.back() != adjncy.size()) {
        cerr << "Invalid graph format for METIS\n";
        return;
    }

    // prepare metis parameters
    idx_t ncon = 1;
    idx_t objval;
    vector<idx_t> part(nvtxs);

    // set metis options
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_CONTIG] = 0;  // Don't require contiguous partitions
    options[METIS_OPTION_NUMBERING] = 0; // Use 0-based numbering

    //calling metis
    int result = METIS_PartGraphKway(&nvtxs, &ncon,
        xadj.data(), adjncy.data(),
        nullptr, nullptr, nullptr,
        &nparts, nullptr, nullptr,
        options, &objval,
        part.data());

    
    if (result == METIS_OK) {
        for (size_t i = 0; i < part.size(); ++i) {
            node_partitions[reverse_map[i]] = part[i];
        }
    }
    else {
        cerr << "METIS failed (Error: " << result << "), using simple partitioning\n";
        int counter = 0;
        for (const auto& [node, _] : graph) {
            node_partitions[node] = counter++ % nparts;
        }
    }
}

// serialize mpi data
void serializeEdges(const vector<Edge>& edges, vector<double>& buffer) {
    buffer.clear();
    for (const auto& edge : edges) {
        buffer.push_back(static_cast<double>(edge.from));
        buffer.push_back(static_cast<double>(edge.to));
        buffer.push_back(edge.obj1);
        buffer.push_back(edge.obj2);
        buffer.push_back(edge.obj3);
        buffer.push_back(edge.obj4);
        buffer.push_back(edge.obj5);
    }
}

// deserialize mpi data
vector<Edge> deserializeEdges(const vector<double>& buffer) {
    vector<Edge> edges;
    for (size_t i = 0; i < buffer.size(); i += 7) {
        edges.push_back({
            static_cast<long long>(buffer[i]),
            static_cast<long long>(buffer[i + 1]),
            buffer[i + 2],
            buffer[i + 3],
            buffer[i + 4],
            buffer[i + 5],
            buffer[i + 6]
            });
    }
    return edges;
}
// distribute partitioned graph
void distributePartitions(const Graph& full_graph,
    const unordered_map<long long, int>& node_partitions,
    Graph& local_graph,
    unordered_map<long long, Vertex>& local_vertices,
    int rank, int num_procs) {
    // first count how many nodes belong to each partition
    vector<int> counts(num_procs, 0);
    for (const auto& [node, part] : node_partitions) {
        counts[part]++;
    }

    // each process gets nodes and edges
    for (const auto& [node, edges] : full_graph) {
        if (node_partitions.at(node) == rank) {
            local_graph[node] = edges;
            local_vertices[node] = Vertex();

            //node structure for all destination nodes
            for (const auto& edge : edges) {
                if (local_vertices.find(edge.to) == local_vertices.end()) {
                    local_vertices[edge.to] = Vertex();
                }
            }
        }
    }
}

// gather result at rank 0
void gatherResults(unordered_map<long long, Vertex>& local_vertices,
    unordered_map<long long, Vertex>& global_vertices,
    int rank, int num_procs) {
    if (rank == 0) {
        
        global_vertices = local_vertices;

        // receive vertices from other processes
        for (int src = 1; src < num_procs; src++) {
            //receive the count of vertices
            int num_vertices;
            MPI_Recv(&num_vertices, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (num_vertices > 0) {
                // then receive the vertex data
                vector<long long> nodes(num_vertices);
                vector<double> obj1_dists(num_vertices);
                vector<double> obj2_dists(num_vertices);
                vector<double> obj3_dists(num_vertices);
                vector<double> obj4_dists(num_vertices);
                vector<double> obj5_dists(num_vertices);
                vector<long long> parents(num_vertices);

                MPI_Recv(nodes.data(), num_vertices, MPI_LONG_LONG, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(obj1_dists.data(), num_vertices, MPI_DOUBLE, src, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(obj2_dists.data(), num_vertices, MPI_DOUBLE, src, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(obj3_dists.data(), num_vertices, MPI_DOUBLE, src, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(obj4_dists.data(), num_vertices, MPI_DOUBLE, src, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(obj5_dists.data(), num_vertices, MPI_DOUBLE, src, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(parents.data(), num_vertices, MPI_LONG_LONG, src, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // store the received vertices - only update if the vertex doesn't exist or smaller distances
                for (int i = 0; i < num_vertices; i++) {
                    if (global_vertices.find(nodes[i]) == global_vertices.end() ||
                        obj1_dists[i] < global_vertices[nodes[i]].obj1_dist) {
                        Vertex& v = global_vertices[nodes[i]];
                        v.obj1_dist = obj1_dists[i];
                        v.obj2_dist = obj2_dists[i];
                        v.obj3_dist = obj3_dists[i];
                        v.obj4_dist = obj4_dists[i];
                        v.obj5_dist = obj5_dists[i];
                        v.parent = parents[i];
                    }
                }
            }
        }
    }
    else {
        // send vertex count to rank 0
        int num_vertices = local_vertices.size();
        MPI_Send(&num_vertices, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

        if (num_vertices > 0) {
            
            vector<long long> nodes;
            vector<double> obj1_dists, obj2_dists, obj3_dists, obj4_dists, obj5_dists;
            vector<long long> parents;

            for (const auto& [node, vertex] : local_vertices) {
                nodes.push_back(node);
                obj1_dists.push_back(vertex.obj1_dist);
                obj2_dists.push_back(vertex.obj2_dist);
                obj3_dists.push_back(vertex.obj3_dist);
                obj4_dists.push_back(vertex.obj4_dist);
                obj5_dists.push_back(vertex.obj5_dist);
                parents.push_back(vertex.parent);
            }

            
            MPI_Send(nodes.data(), num_vertices, MPI_LONG_LONG, 0, 1, MPI_COMM_WORLD);
            MPI_Send(obj1_dists.data(), num_vertices, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
            MPI_Send(obj2_dists.data(), num_vertices, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
            MPI_Send(obj3_dists.data(), num_vertices, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
            MPI_Send(obj4_dists.data(), num_vertices, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD);
            MPI_Send(obj5_dists.data(), num_vertices, MPI_DOUBLE, 0, 6, MPI_COMM_WORLD);
            MPI_Send(parents.data(), num_vertices, MPI_LONG_LONG, 0, 7, MPI_COMM_WORLD);
        }
    }
}

int main(int argc, char** argv) {
    
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    
    omp_set_num_threads(omp_get_max_threads());
    auto start_time = high_resolution_clock::now();

    // declare graph data structures
    Graph full_graph, local_graph;
    unordered_map<long long, Vertex> full_vertices, local_vertices;

    // only rank 0 loads the full graph
    if (rank == 0) {
        loadGraphFromCSV("multi_obj_graph_netherland.csv", full_graph, full_vertices);
        if (full_vertices.empty()) {
            cerr << "Error: No vertices loaded. Exiting.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    // broadcast graph size
    int num_vertices = 0;
    if (rank == 0) {
        num_vertices = full_vertices.size();
    }
    MPI_Bcast(&num_vertices, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // partition the graph
    unordered_map<long long, int> node_partitions;
    unordered_map<long long, idx_t> id_map;
    vector<long long> reverse_map;

    if (rank == 0) {
        // to metis format and partition
        vector<idx_t> xadj, adjncy;
        convertToMetisFormat(full_graph, xadj, adjncy, id_map, reverse_map);
        partitionGraphWithMETIS(full_graph, num_procs, node_partitions);
    }

    // Broadcast partition information and number of nodes
    int num_nodes = 0;
    if (rank == 0) {
        num_nodes = id_map.size();
    }
    MPI_Bcast(&num_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<long long> node_ids(num_nodes);
    vector<int> partitions(num_nodes);

    if (rank == 0) {
        for (const auto& [node, idx] : id_map) {
            node_ids[idx] = node;
            partitions[idx] = node_partitions[node];
        }
    }

    MPI_Bcast(node_ids.data(), num_nodes, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(partitions.data(), num_nodes, MPI_INT, 0, MPI_COMM_WORLD);

    // each process builds its own node_partitions map
    node_partitions.clear();
    for (int i = 0; i < num_nodes; i++) {
        node_partitions[node_ids[i]] = partitions[i];
    }

    // distribute the graph partitions
    if (rank == 0) {
       
        local_graph = full_graph;
        local_vertices = full_vertices;
    }
    else {
        // other ranks need to receive their portion
        for (const auto& [node, part] : node_partitions) {
            if (part == rank) {
                local_vertices[node] = Vertex();
            }
        }
    }

    // distribute the actual edges
    if (rank == 0) {
        for (int dest = 1; dest < num_procs; dest++) {
            // collect edges that belong to this destination
            vector<Edge> edges_to_send;
            for (const auto& [node, edges] : full_graph) {
                if (node_partitions[node] == dest) {
                    edges_to_send.insert(edges_to_send.end(), edges.begin(), edges.end());
                }
            }

            // serialize and send
            vector<double> edge_buffer;
            serializeEdges(edges_to_send, edge_buffer);
            int buffer_size = edge_buffer.size();
            MPI_Send(&buffer_size, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
            if (buffer_size > 0) {
                MPI_Send(edge_buffer.data(), buffer_size, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
            }
        }
    }
    else {
        // receive edges from rank 0
        int buffer_size;
        MPI_Recv(&buffer_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (buffer_size > 0) {
            vector<double> edge_buffer(buffer_size);
            MPI_Recv(edge_buffer.data(), buffer_size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            vector<Edge> edges = deserializeEdges(edge_buffer);

            // store the edges in local graph
            for (const auto& edge : edges) {
                local_graph[edge.from].push_back(edge);
                // destination vertices exist...
                if (local_vertices.find(edge.to) == local_vertices.end()) {
                    local_vertices[edge.to] = Vertex();
                }
            }
        }
    }

    
    long long source = 6316199;
   
    if (rank == 0 && full_vertices.find(source) == full_vertices.end()) {
        cerr << "Error: Source node " << source << " not found.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    
    MPI_Bcast(&source, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

  
    parallelSOSP_Update(local_graph, local_vertices, source, rank, num_procs);

   //gather results and compute mosp
    unordered_map<long long, Vertex> global_vertices;
    gatherResults(local_vertices, global_vertices, rank, num_procs);

    
    for (int i = 0; i < 5; ++i) {
        printTree(rank == 0 ? global_vertices : local_vertices, i, source, rank);
    }

    // final mosp
    if (rank == 0) {
        auto combined_graph = combineSOSPTrees(local_graph, global_vertices);
        computeFinalMOSP(combined_graph, global_vertices, source, rank);
    }
    // -------------------------------------------------------------------

    // insertions
    vector<Edge> inserted_edges;
    if (rank == 0) {
        inserted_edges = {
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
        };
    }

    
    int num_insertions = 0;
    if (rank == 0) {
        num_insertions = inserted_edges.size();
    }
    MPI_Bcast(&num_insertions, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //broadcast edge data
    vector<double> edge_buffer;
    if (rank == 0) {
        serializeEdges(inserted_edges, edge_buffer);
    }

    
    int buffer_size = 0;
    if (rank == 0) {
        buffer_size = edge_buffer.size();
    }
    MPI_Bcast(&buffer_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // resize buffer on non-root processes
    if (rank != 0) {
        edge_buffer.resize(buffer_size);
    }

  
    MPI_Bcast(edge_buffer.data(), buffer_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // deserialize edges on non-root processes
    if (rank != 0) {
        inserted_edges = deserializeEdges(edge_buffer);
    }

    // process edge insertions one by one
    for (const auto& edge : inserted_edges) {
        if (rank == 0) {
            cout << "\nInserting Edge: " << edge.from << " -> " << edge.to << "\n";
        }

        
        incrementalUpdate(local_graph, local_vertices, source, { edge }, rank);

        // updated trees
        for (int i = 0; i < 5; ++i) {
            printTree(rank == 0 ? global_vertices : local_vertices, i, source, rank);
        }
    }

    // gather results and if needed do mosp 
    if (!inserted_edges.empty()) {
        gatherResults(local_vertices, global_vertices, rank, num_procs);

        if (rank == 0) {
            auto combined_graph = combineSOSPTrees(local_graph, global_vertices);
            computeFinalMOSP(combined_graph, global_vertices, source, rank);
        }
    }

    
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time);

    if (rank == 0) {
        cout << "\nTotal execution time: " << duration.count() << " ms\n";
    }

    
    MPI_Finalize();
    return 0;
}
