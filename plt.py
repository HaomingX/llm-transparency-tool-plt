import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import argparse
import os
from collections import defaultdict, deque
from dfs import filter_edges
from tqdm import tqdm
from copy import deepcopy
from multiprocessing import Pool, cpu_count

# Configuration parameters for the graph visualization
config = {
    'spacing_factor': 2.0,  # Controls horizontal spacing between nodes
    'node_types': ['X', 'A', 'M', 'I'],  # Types of nodes to be plotted
    'default_edge_color': 'green',  # Default color for edges
    'special_edge_color': 'purple',  # Color for special edges (e.g., involving 'M' nodes)
    'active_node_color': 'green',  # Color for active nodes
    'inactive_node_color': 'gray',  # Color for inactive nodes
    'point_size': 3,  # Size of the nodes in the plot
    'darken_factor': 0.7,  # Factor to darken edge colors
    'lighten_factor': 0.85,  # Factor to lighten link colors
    'edge_linewidth': 0.3,  # Line width for node edges
    'plot_linewidth': 0.8,  # Line width for edges between nodes
    'plot_alpha': 0.7,  # Transparency level for edges
    'margins': {'x': 0.05, 'y': 0.1},  # Margins around the plot
    'min_layer_margin': -1.2,  # Minimum margin for the y-axis (layers)
    'max_layer_margin': 0.5,  # Maximum margin for the y-axis (layers)
    'min_word_margin_factor': -0.5,  # Minimum margin for the x-axis (words)
    'dpi': 700,  # Dots per inch for the output image
    "font_size": 6,  # Font size for labels
    'adjustment_A': -0.2,  # Vertical adjustment for 'A' nodes
    'adjustment_I': 0.2,  # Vertical adjustment for 'I' nodes
    'adjustment_M': 0.2,  # Horizontal adjustment for 'M' nodes
    'adjustment_X_layer0': -1  # Vertical adjustment for 'X' nodes in layer 0
}

def parse_node_id(node_id):
    """
    Parse the node ID string into its components: node type, layer, and word index.
    
    Parameters:
    - node_id: A string representing the node ID (e.g., "A0_1").
    
    Returns:
    - Tuple of (node_type, layer, word_index)
    """
    parts = node_id.split('_')
    node_type = parts[0][0]  # First character is the node type (A, I, M, X)
    layer = int(parts[0][1:])  # The rest is the layer
    word_index = int(parts[1])  # Second part is the word index
    return node_type, layer, word_index

def is_active_node(node_type, layer, word_index, active_node_ids=set()):
    """
    Check if a node is in the active nodes list.
    
    Parameters:
    - node_type: Type of the node (A, I, M, X).
    - layer: Layer of the node.
    - word_index: Index of the word associated with the node.
    
    Returns:
    - Boolean indicating whether the node is active.
    """
    node_id = f"{node_type}{layer}_{word_index}"
    return node_id in active_node_ids

def get_position_adjustment(node_type, word_index, layer):
    """
    Calculate the position adjustments for a node based on its type and layer.
    
    Parameters:
    - node_type: Type of the node (A, I, M, X).
    - word_index: Index of the word associated with the node.
    - layer: Layer of the node.
    
    Returns:
    - Tuple of (x, y) coordinates for the node.
    """
    x = word_index * config['spacing_factor']
    y = layer

    if node_type == 'A':
        y += config['adjustment_A']
    elif node_type == 'I':
        y += config['adjustment_I']
    elif node_type == 'M':
        x += config['adjustment_M'] * config['spacing_factor']
    elif node_type == 'X' and layer == 0:
        y += config['adjustment_X_layer0']

    return x, y

def adjust_color_depth(color, darken_factor=config['darken_factor'], lighten_factor=config['lighten_factor']):
    """
    Adjust the depth of a color for node edges and links.
    
    Parameters:
    - color: A string or RGB tuple representing the base color.
    - darken_factor: Factor to darken the edge color (closer to 0 means darker).
    - lighten_factor: Factor to lighten the link color (closer to 1 means lighter).
    
    Returns:
    - Tuple of RGB tuples: (darkened_edge_color, lightened_link_color)
    """
    if isinstance(color, str):
        color_rgb = mcolors.to_rgb(color)
    
    def adjust_component(c, factor, lighten=False):
        if lighten:
            return min(1, c + (1 - c) * (1 - factor))
        else:
            return min(1, c * factor)

    dark_edge_color = tuple(adjust_component(c, darken_factor) for c in color_rgb)
    light_link_color = tuple(adjust_component(c, lighten_factor, lighten=True) for c in color_rgb)
    
    return dark_edge_color, light_link_color

def plot_graph(question, gt, pred, active_node_ids, links, word_list, n_layers, config, output_filename):
    """
    Plot the graph visualization for the given data and configuration.
    
    Parameters:
    - data: The data containing the graph information.
    - config: The configuration parameters for the plot.
    """
    layers = list(range(n_layers))
    question_length = len(question)
    pred_length = min(len(pred), question_length * 2)
    gt_length = min(len(gt), question_length * 2)

    # formatted text
    pred_formatted = f"{pred[:question_length]}\n{pred[question_length:pred_length]}{'...' if len(pred) > pred_length else ''}"
    gt_formatted = f"{gt[:question_length]}\n{gt[question_length:gt_length]}{'...' if len(gt) > gt_length else ''}"

    # get formatted sentence
    sentence = f"Q: {question}\nA: {pred_formatted}\nRef: {gt_formatted}"

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # Draw edges between nodes
    for link in links:
        source_parts = parse_node_id(link['source'])
        target_parts = parse_node_id(link['target'])
        this_edge_color = config['special_edge_color'] if source_parts[0] == "M" or target_parts[0] == "M" else config['default_edge_color']

        _, this_edge_color_light = adjust_color_depth(this_edge_color, lighten_factor=config['lighten_factor'])

        source_pos = get_position_adjustment(source_parts[0], source_parts[2], source_parts[1])
        target_pos = get_position_adjustment(target_parts[0], target_parts[2], target_parts[1])

        ax.plot(
            [source_pos[0], target_pos[0]], [source_pos[1], target_pos[1]],
            color=this_edge_color_light, linewidth=config['plot_linewidth'], alpha=config['plot_alpha']
        )

    # Draw nodes on the graph
    for layer in layers:
        for i, word in enumerate(word_list):
            for node_type in config['node_types']:
                if node_type == 'X' and layer != 0:
                    continue  # X nodes only exist in the first layer
                color = config['active_node_color'] if is_active_node(node_type, layer, i, active_node_ids=active_node_ids) else config['inactive_node_color']

                edgecolor, _ = adjust_color_depth(color, darken_factor=config['darken_factor'])

                pos_x, pos_y = get_position_adjustment(node_type, i, layer)

                if node_type == 'A':
                    ax.scatter(pos_x, pos_y, s=config['point_size'], c=color, edgecolors=edgecolor, linewidth=config['edge_linewidth'])
                elif node_type == 'I':
                    ax.scatter(pos_x, pos_y, s=config['point_size'], c=color, edgecolors=edgecolor, linewidth=config['edge_linewidth'])
                elif node_type == 'M':
                    color = config['special_edge_color'] if color == config['active_node_color'] else color
                    edgecolor, _ = adjust_color_depth(color, darken_factor=config['darken_factor'])
                    ax.scatter(pos_x, pos_y, s=config['point_size'], c=color, marker='s', edgecolors=edgecolor, linewidth=config['edge_linewidth'])
                elif node_type == 'X' and layer == 0:
                    ax.scatter(pos_x, pos_y, s=config['point_size'], c=color, marker='^', edgecolors=edgecolor, linewidth=config['edge_linewidth'])

    # Set plot margins and labels
    plt.margins(x=config['margins']['x'], y=config['margins']['y'])
    ax.set_yticks(layers)
    ax.set_yticklabels(['L{}'.format(l) for l in layers], fontsize=config["font_size"])
    ax.set_xticks([i * config['spacing_factor'] for i in range(len(word_list))])
    ax.set_xticklabels(word_list, rotation=45, ha='right', fontsize=config["font_size"] * 2)

    # Set axis limits
    min_layer = min(layers) + config['min_layer_margin']
    max_layer = max(layers) + config['max_layer_margin']
    ax.set_ylim(min_layer, max_layer)
    min_word = config['min_word_margin_factor']
    max_word = len(word_list) * config['spacing_factor']
    ax.set_xlim(min_word, max_word)

    # Finalize and save the plot
    plt.tight_layout()
    # set title if needed
    ax.set_title(sentence, fontsize=config["font_size"])
    plt.savefig(output_filename, bbox_inches='tight', dpi=config['dpi'])
    return

def load_data(data_path, sentence_idx, thresholdu, save_fig_dir):
    """
    Load the data from the JSON file and extract the relevant information for the given sentence index.
    
    Parameters:
    - data_path: Path to the JSON file containing the data.
    - sentence_idx: Index of the sentence to visualize.
    - save_fig_dir: Path to save the figures
    
    Returns:
    - Tuple of (sentence, word_list, layers, active_node_ids, links)
    """
    with open(data_path, "r") as f:
        data = json.load(f)

    default_payload = {"idx": None, "q": None, "gt": None, "pred": None, "token_list": None, "n_layers": None, "nodes": None, "edges": None, "method": None, "ckpt": None, "threshold": None, "save_fig_dir": save_fig_dir}
    
    data_path = data_path.rstrip("-full.json")
    *_, method, ckpt = data_path.split("/")
    ckpt = ckpt.split("-")[1]

    # -1 for sweeping all sentences
    length = len(data)
    if sentence_idx == -1:
        idxs = list(range(length))
    else:
        idxs = [sentence_idx]
    
    if threshold == -1:
        # FIXME: all thresholds
        threshold = [ 0.06, 0.08, 0.1, ]
    else:
        threshold = [threshold]
    
    payloads = []
    for i in idxs:
        for t in threshold:
            payload = deepcopy(default_payload)
            payload["idx"] = i
            tdata = data[i]
            payload["q"] = tdata["q"]
            payload["gt"] = tdata["gt"]
            payload["pred"] = tdata["a"]
            payload["token_list"] = tdata["token_list"]
            payload["n_layers"] = tdata["n_layers"]
            payload["nodes"] = tdata["graph"][-1]["graph"]["nodes"]
            payload["edges"] = tdata["graph"][-1]["graph"]["links"]
            payload["method"] = method
            payload["ckpt"] = ckpt
            payload["threshold"] = t
            payloads.append(payload)
    return payloads


def process_payload(payload):
    """
    Process the payload data to extract the relevant information for plotting.
    
    Parameters:
    - payload: The payload data containing the graph information.
    
    Returns:
    - Tuple of (sentence, word_list, layers, active_node_ids, links)
    """
    # clip edges based on threshold
    # filter_edges(edges, threshold, starts, end)
    n_layers = payload["n_layers"]
    n_tokens = len(payload["token_list"])
    start_nodes = [f"X0_{i}" for i in range(n_tokens)]
    end_node = f"I{n_layers-1}_{n_tokens-1}"
    edges = filter_edges(payload["edges"], payload["threshold"], starts=start_nodes, end=end_node)
    nodes = set()
    for edge in edges:
        nodes.add(edge["source"])
        nodes.add(edge["target"])

    # plot graph
    output_filename = f'{payload["save_fig_dir"]}/{payload["method"]}/{payload["idx"]}/{payload["threshold"]}/ckpt-{payload["ckpt"]}.png'
    # if exists, skip
    if os.path.exists(output_filename):
        return
    output_dir = os.path.dirname(output_filename)
    os.makedirs(output_dir, exist_ok=True)
    plot_graph(payload["q"], payload["gt"], payload["pred"], nodes, edges, payload["token_list"], n_layers, config, output_filename)
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot the graph visualization for a given sentence.')

    parser.add_argument("--results_dir", type=str, default="../output", help="Path to the checkpoint results JSON file")
    parser.add_argument("--sentence_index", type=int, default=-1, help="Index of the sentence to visualize, -1 means all sentences")
    parser.add_argument("--threshold", type=float, default=-1, help="Threshold for filtering edges, -1 means all thresholds (defined in line 233)")
    parser.add_argument("--save_fig_dir", type=str, default="./plots", help="Path to save the figures")

    args = parser.parse_args()

    payloads = []
    # list folders in the results path
    results_dir = args.results_dir
    methods = os.listdir(results_dir)
    for method in methods:
        method_path = os.path.join(results_dir, method)
        ckpts = os.listdir(method_path)
        for ckpt in ckpts:
            ckpt_path = os.path.join(method_path, ckpt)
            ckpt_num = int(ckpt.split("-")[1])
            batch_payloads = load_data(ckpt_path, args.sentence_index, args.threshold, args.save_fig_dir)
            payloads.extend(batch_payloads)
    
    total = len(payloads)  

    print(f"cpu count: {int(cpu_count())}")
    with Pool(int(cpu_count())) as pool:
        for _ in tqdm(pool.imap_unordered(process_payload, payloads), total=total):
            pass  