import networkx as nx

def filter_edges(edges, threshold, starts, end):
    g = nx.DiGraph()

    # add nodes and edges
    for edge in edges:
        g.add_node(edge["source"])
        g.add_node(edge["target"])
        if edge["weight"] > threshold:
            g.add_edge(edge["source"], edge["target"], weight=edge["weight"])
    
    links = set()
    for start in starts:
        if start not in g.nodes:
            print(f"Warning: Start node {start} not in graph.")
            continue
        if end not in g.nodes:
            print(f"Warning: End node {end} not in graph.")
            continue
        
        all_paths = nx.all_simple_paths(g, source=start, target=end)
        unique_edges = set()
        for path in all_paths:
            for i in range(len(path) - 1):
                source = path[i]
                target = path[i+1]
                unique_edges.add((source, target))
        
        links.update(unique_edges)
    
    return [{"source": source, "target": target} for source, target in links]