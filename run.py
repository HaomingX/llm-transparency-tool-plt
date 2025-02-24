import pandas as pd
from llm_transparency_tool.models.tlens_model import TransformerLensTransparentLlm
from torch.amp import autocast
from llm_transparency_tool.server.utils import (
    B0,
    get_contribution_graph,
    load_model,
)
import os
import json
import torch
import networkx as nx
from typing import List, Optional, Tuple
import llm_transparency_tool
import llm_transparency_tool.components
import numpy as np
from tqdm import tqdm
from networkx.readwrite import json_graph
import sys
import argparse

def encode_graph_list(graph_list):
    return [json_graph.node_link_data(G) for G in graph_list]

def decode_graph_list(data_list):
    return [json_graph.node_link_graph(data) for data in data_list]

def cached_run_inference_and_populate_state(
    stateless_model,
    sentences,
):
    stateful_model = stateless_model.copy()
    stateful_model.run(sentences)
    return stateful_model
def cached_build_paths_to_predictions(
    graph: nx.Graph,
    n_layers: int,
    n_tokens: int,
    starting_tokens: List[int],
    threshold: float,
):
    return llm_transparency_tool.routes.graph.build_paths_to_predictions(
        graph, n_layers, n_tokens, starting_tokens, threshold
    )

# def function
def run_inference(
    stateful_model: TransformerLensTransparentLlm ,
    model_key: str ,
    sentence: str ,
    contribution_threshold: float ,
    renormalize_after_threshold: bool ,
    amp_enabled: bool ,
    dtype: torch.dtype ,
):

    # get inference state
    with autocast(enabled=amp_enabled, device_type="cuda", dtype=dtype):
        stateful_model = cached_run_inference_and_populate_state(stateful_model, [sentence])

    # get contribution graph for the sentence
    # node: seqLen * (1 + 3 * n_layers)
    # 1 denotes the input seq, 3 denotes the attention mlp res
    with autocast(enabled=amp_enabled, device_type="cuda", dtype=dtype):
        graph = get_contribution_graph(
            stateful_model,
            model_key,
            stateful_model.tokens()[B0].tolist(),
            (contribution_threshold if renormalize_after_threshold else 0.0),
    )
    # graph: full path
    # input -> output
    return stateful_model, graph

def draw_graph(
    graph: nx.Graph,
    stateful_model: TransformerLensTransparentLlm,
    contribution_threshold: float,
    ) :
    tokens = stateful_model.tokens()[B0]
    n_tokens = tokens.shape[0]
    model_info = stateful_model.model_info()

    graphs = cached_build_paths_to_predictions(
        graph,
        model_info.n_layers,
        n_tokens,
        range(n_tokens),
        contribution_threshold,
    )
    return graphs, stateful_model.tokens_to_strings(tokens)

def draw_graph_default(
    graph: nx.Graph,
    stateful_model: TransformerLensTransparentLlm,
    contribution_threshold: float,
    sentence: str
    ) :
    tokens = stateful_model.tokens()[B0]
    n_tokens = tokens.shape[0]
    model_info = stateful_model.model_info()

    graphs = cached_build_paths_to_predictions(
        graph,
        model_info.n_layers,
        n_tokens,
        range(n_tokens),
        contribution_threshold,
    )

    return llm_transparency_tool.components.contribution_graph(
    model_info,
    stateful_model.tokens_to_strings(tokens),
    graphs,
    key=f"graph_{hash(sentence)}",
)
 
def getLeafNode(rtree):
    # tree(output(root, one) -> input(leaf, many))
    tree = rtree.reverse()
    leaf_nodes = [node for node, degree in tree.out_degree() if degree == 0]
    leafPos = [node.split("_")[-1] for node in leaf_nodes]
    return leaf_nodes, leafPos

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, default="./llama2-7b-hf-ckpt/", help="path to the model checkpoint")
    parser.add_argument("--model_name", type=str, required=True, default="meta-llama/Llama-2-7b-hf", help="supported model names: meta-llama/Llama-2-7b-hf, meta-llama/Meta-Llama-3-8B, Qwen/Qwen1.5-0.5B, google/gemma-7b etc.")
    parser.add_argument("--dataset_path", type=str, required=True, default="./dataset/KnowUnDo/privacy/unlearn_train.json", help="path to the dataset")
    parser.add_argument("--save_path", type=str, required=True, default="./results.json", help="path to save the results")
    # model_name = "/models/Llama-3-8B/"
    # model_name = "meta-llama/Meta-Llama-3-8B"
    # model_name = "meta-llama/Llama-2-7b-hf"
    # model_name = "Qwen/Qwen1.5-0.5B"
    # model_name = "google/gemma-7b"
    # model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    args = parser.parse_args()

    # configs
    with open("./config/local.json") as f:
        config = json.load(f)
    dtype, amp_enabled = torch.bfloat16, True
    contribution_threshold = 0.01
    renormalize_after_threshold = True
    device = "gpu"
    # load data
    with open(args.dataset_path) as f:
        data = json.load(f)
    
    model_name = args.model_name
    model_path = args.model_path
    save_path = args.save_path

    stateless_model = load_model(
        model_name=model_name,
        _model_path=None,
        model_path=model_path,
        _device=device,
        _dtype=dtype,
    )
    stateless_model._model
    tokenizer = stateless_model.hf_tokenizer
    #tokenizer.pad_token = tokenizer.eos_token
    hf_model = stateless_model.hf_model
    hf_model.eval()
    
    model_key = model_name  # TODO: maybe something else?
    
    results_ = []
    input_sentences = [d["text"] for d in data]
    gt_sentences = [d["labels"] for d in data]
    pred_sentences = []
    inputs = tokenizer(input_sentences, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad(): 
        outputs = hf_model.generate(**inputs, max_length=50, num_return_sequences=1)
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    for i, response in enumerate(responses):
        response = response[len(input_sentences[i]):]
        pred_sentences.append(response)
        
    for idx, s in tqdm(enumerate(data)):
        r_ = {}
        r_["q"] = s["text"]
        r_["gt"] = s["labels"]
        warped_sentence = s["text"] # str
        assert s["text"] == input_sentences[idx]
        r_["a"] = pred_sentences[idx]
        with torch.inference_mode():
            stateful_model, graph = run_inference(
                stateful_model=stateless_model,
                model_key=model_key,
                sentence=warped_sentence,
                contribution_threshold=contribution_threshold,
                renormalize_after_threshold=renormalize_after_threshold,
                amp_enabled=amp_enabled,
                dtype=dtype,
            )
        graphs, strTokens = draw_graph(
            graph, 
            stateful_model, 
            contribution_threshold if renormalize_after_threshold else 0.0
            )
        json_graphs = encode_graph_list(graphs)
        res = []
        for g, t in zip(json_graphs, strTokens):
            res.append({"token": t, "graph": g})
        r_["graph"] = res
        r_["token_list"] = strTokens
        r_["n_layers"] = stateful_model.model_info().n_layers

        results_.append(r_)


    dir_path = os.path.dirname(save_path)
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # save results 
    with open(save_path, "w") as f:
        json.dump(results_, f)






