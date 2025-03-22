"""
Path-based Retrieval Module for LightRAG.

This module implements the path-based retrieval algorithm from PathRAG, focusing on
identifying key relational paths between entities in a knowledge graph to improve
retrieval quality.
"""

import asyncio
import re
import json
from collections import defaultdict, Counter
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional, Any, Union

from .base import BaseGraphStorage, BaseKVStorage, BaseVectorStorage, QueryParam
from .utils import logger, truncate_list_by_token_size, list_of_list_to_csv
from .prompt import GRAPH_FIELD_SEP


async def find_paths_and_edges_with_stats(graph, target_nodes):
    """
    Find paths and edges between target nodes in the graph.

    Args:
        graph: A NetworkX graph.
        target_nodes: A list of target node IDs.

    Returns:
        result: A dictionary mapping node pairs to paths and edges.
        path_stats: A dictionary with statistics about path lengths.
        one_hop_paths: A list of 1-hop paths.
        two_hop_paths: A list of 2-hop paths.
        three_hop_paths: A list of 3-hop paths.
    """
    result = defaultdict(lambda: {"paths": [], "edges": set()})
    path_stats = {"1-hop": 0, "2-hop": 0, "3-hop": 0}   
    one_hop_paths = []
    two_hop_paths = []
    three_hop_paths = []

    async def dfs(current, target, path, depth):
        if depth > 3:  # Limit path length to 3 hops
            return
        if current == target:  # Found a path
            result[(path[0], target)]["paths"].append(list(path))
            for u, v in zip(path[:-1], path[1:]):
                result[(path[0], target)]["edges"].add(tuple(sorted((u, v))))
            if depth == 1:
                path_stats["1-hop"] += 1
                one_hop_paths.append(list(path))
            elif depth == 2:
                path_stats["2-hop"] += 1
                two_hop_paths.append(list(path))
            elif depth == 3:
                path_stats["3-hop"] += 1
                three_hop_paths.append(list(path))
            return
        neighbors = graph.neighbors(current) 
        for neighbor in neighbors:
            if neighbor not in path:  # Avoid cycles
                await dfs(neighbor, target, path + [neighbor], depth + 1)

    for node1 in target_nodes:
        for node2 in target_nodes:
            if node1 != node2:
                await dfs(node1, node2, [node1], 0)

    # Convert edge sets to lists for serialization
    for key in result:
        result[key]["edges"] = list(result[key]["edges"])

    return dict(result), path_stats, one_hop_paths, two_hop_paths, three_hop_paths


def bfs_weighted_paths(G, path, source, target, threshold, alpha):
    """
    Find weighted paths between source and target nodes using breadth-first search.
    
    Args:
        G: A NetworkX graph
        path: A list of paths to explore
        source: The source node
        target: The target node
        threshold: The pruning threshold
        alpha: The decay rate for flow propagation
        
    Returns:
        A list of weighted paths
    """
    results = [] 
    edge_weights = defaultdict(float)  
    node = source
    follow_dict = {}

    # Build follow_dict mapping nodes to their neighbors in the paths
    for p in path:
        for i in range(len(p) - 1):  
            current = p[i]
            next_num = p[i + 1]

            if current in follow_dict:
                follow_dict[current].add(next_num)
            else:
                follow_dict[current] = {next_num}

    # Propagate weights through the graph
    for neighbor in follow_dict[node]:
        edge_weights[(node, neighbor)] += 1/len(follow_dict[node])

        if neighbor == target:
            results.append(([node, neighbor]))
            continue
        
        if edge_weights[(node, neighbor)] > threshold:
            # 2-hop paths
            for second_neighbor in follow_dict[neighbor]:
                weight = edge_weights[(node, neighbor)] * alpha / len(follow_dict[neighbor])
                edge_weights[(neighbor, second_neighbor)] += weight

                if second_neighbor == target:
                    results.append(([node, neighbor, second_neighbor]))
                    continue

                if edge_weights[(neighbor, second_neighbor)] > threshold:    
                    # 3-hop paths
                    for third_neighbor in follow_dict[second_neighbor]:
                        weight = edge_weights[(neighbor, second_neighbor)] * alpha / len(follow_dict[second_neighbor]) 
                        edge_weights[(second_neighbor, third_neighbor)] += weight

                        if third_neighbor == target:
                            results.append(([node, neighbor, second_neighbor, third_neighbor]))
                            continue
    
    # Calculate path weights
    path_weights = []
    for p in path:
        path_weight = 0
        for i in range(len(p) - 1):
            edge = (p[i], p[i + 1])
            path_weight += edge_weights.get(edge, 0)  
        path_weights.append(path_weight/(len(p)-1))

    # Combine paths with their weights
    combined = [(p, w) for p, w in zip(path, path_weights)]

    return combined


async def find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
) -> list[str]:
    """
    Find the most related edges between entities using path-based retrieval.
    
    Args:
        node_datas: List of node data dictionaries
        query_param: Query parameters
        knowledge_graph_inst: Knowledge graph storage
        
    Returns:
        List of path descriptions
    """
    # Build a NetworkX graph from the knowledge graph
    G = nx.Graph()
    edges = await knowledge_graph_inst.edges()
    nodes = await knowledge_graph_inst.nodes()

    for u, v in edges:
        G.add_edge(u, v) 
    G.add_nodes_from(nodes)
    
    # Get source nodes from node_datas
    source_nodes = [dp["entity_name"] for dp in node_datas]
    
    # Find paths and edges between source nodes
    result, path_stats, one_hop_paths, two_hop_paths, three_hop_paths = await find_paths_and_edges_with_stats(G, source_nodes)

    # Get threshold and alpha from query_param
    threshold = query_param.path_threshold
    alpha = query_param.path_decay_rate
    
    all_results = []
    
    # Apply weighted path search for each source-target pair
    for node1 in source_nodes: 
        for node2 in source_nodes: 
            if node1 != node2: 
                if (node1, node2) in result:
                    sub_G = nx.Graph()
                    paths = result[(node1,node2)]['paths']
                    edges = result[(node1,node2)]['edges']
                    sub_G.add_edges_from(edges)
                    results = bfs_weighted_paths(G, paths, node1, node2, threshold, alpha)
                    all_results += results
    
    # Sort results by weight
    all_results = sorted(all_results, key=lambda x: x[1], reverse=True)
    
    # Remove duplicates
    seen = set()
    result_edge = []
    for edge, weight in all_results:
        sorted_edge = tuple(sorted(edge))
        if sorted_edge not in seen:
            seen.add(sorted_edge)  
            result_edge.append((edge, weight))  

    # Sample from each path length category
    length_1 = int(len(one_hop_paths)/2)
    length_2 = int(len(two_hop_paths)/2) 
    length_3 = int(len(three_hop_paths)/2) 
    results = []
    
    if one_hop_paths:
        results = one_hop_paths[0:length_1]
    if two_hop_paths:
        results = results + two_hop_paths[0:length_2]
    if three_hop_paths:
        results = results + three_hop_paths[0:length_3]

    # Limit total edges to consider
    length = len(results)
    total_edges = 15
    if length < total_edges:
        total_edges = length
    
    sort_result = []
    if result_edge:
        if len(result_edge) > total_edges:
            sort_result = result_edge[0:total_edges]
        else: 
            sort_result = result_edge
    
    final_result = []
    for edge, weight in sort_result:
        final_result.append(edge)

    # Convert paths to textual descriptions
    relationship = []
    for path in final_result:
        if len(path) == 4:  # 3-hop path
            s_name, b1_name, b2_name, t_name = path[0], path[1], path[2], path[3]
            edge0 = await knowledge_graph_inst.get_edge(path[0], path[1]) or await knowledge_graph_inst.get_edge(path[1], path[0])
            edge1 = await knowledge_graph_inst.get_edge(path[1], path[2]) or await knowledge_graph_inst.get_edge(path[2], path[1])
            edge2 = await knowledge_graph_inst.get_edge(path[2], path[3]) or await knowledge_graph_inst.get_edge(path[3], path[2])
            
            if not all([edge0, edge1, edge2]):
                logger.warning(f"Missing edges in path {path}")
                continue
                
            e1 = f"through edge ({edge0['keywords']}) to connect to {s_name} and {b1_name}."
            e2 = f"through edge ({edge1['keywords']}) to connect to {b1_name} and {b2_name}."
            e3 = f"through edge ({edge2['keywords']}) to connect to {b2_name} and {t_name}."
            
            s = await knowledge_graph_inst.get_node(s_name)
            s = f"The entity {s_name} is a {s['entity_type']} with the description({s['description']})"
            
            b1 = await knowledge_graph_inst.get_node(b1_name)
            b1 = f"The entity {b1_name} is a {b1['entity_type']} with the description({b1['description']})"
            
            b2 = await knowledge_graph_inst.get_node(b2_name)
            b2 = f"The entity {b2_name} is a {b2['entity_type']} with the description({b2['description']})"
            
            t = await knowledge_graph_inst.get_node(t_name)
            t = f"The entity {t_name} is a {t['entity_type']} with the description({t['description']})"
            
            relationship.append([s + e1 + b1 + " and " + b1 + e2 + b2 + " and " + b2 + e3 + t])
            
        elif len(path) == 3:  # 2-hop path
            s_name, b_name, t_name = path[0], path[1], path[2]
            edge0 = await knowledge_graph_inst.get_edge(path[0], path[1]) or await knowledge_graph_inst.get_edge(path[1], path[0])
            edge1 = await knowledge_graph_inst.get_edge(path[1], path[2]) or await knowledge_graph_inst.get_edge(path[2], path[1])
            
            if not all([edge0, edge1]):
                logger.warning(f"Missing edges in path {path}")
                continue
                
            e1 = f"through edge({edge0['keywords']}) to connect to {s_name} and {b_name}."
            e2 = f"through edge({edge1['keywords']}) to connect to {b_name} and {t_name}."
            
            s = await knowledge_graph_inst.get_node(s_name)
            s = f"The entity {s_name} is a {s['entity_type']} with the description({s['description']})"
            
            b = await knowledge_graph_inst.get_node(b_name)
            b = f"The entity {b_name} is a {b['entity_type']} with the description({b['description']})"
            
            t = await knowledge_graph_inst.get_node(t_name)
            t = f"The entity {t_name} is a {t['entity_type']} with the description({t['description']})"
            
            relationship.append([s + e1 + b + " and " + b + e2 + t])
            
        elif len(path) == 2:  # 1-hop path
            s_name, t_name = path[0], path[1]
            edge0 = await knowledge_graph_inst.get_edge(path[0], path[1]) or await knowledge_graph_inst.get_edge(path[1], path[0])
            
            if not edge0:
                logger.warning(f"Missing edge in path {path}")
                continue
                
            e = f"through edge({edge0['keywords']}) to connect to {s_name} and {t_name}."
            
            s = await knowledge_graph_inst.get_node(s_name)
            s = f"The entity {s_name} is a {s['entity_type']} with the description({s['description']})"
            
            t = await knowledge_graph_inst.get_node(t_name)
            t = f"The entity {t_name} is a {t['entity_type']} with the description({t['description']})"
            
            relationship.append([s + e + t])

    # Truncate relationship texts to fit token limits
    relationship = truncate_list_by_token_size(
        relationship, 
        key=lambda x: x[0],
        max_token_size=query_param.max_token_for_local_context,
    )

    # Return relationships in reversed order (most important last)
    reversed_relationship = relationship[::-1]
    return reversed_relationship


async def path_query(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    param: QueryParam,
    global_config: dict,
    hashing_kv=None,
    system_prompt=None,
) -> str:
    """
    Execute a path-based query using the PathRAG approach.
    
    This function is similar to kg_query but uses path-based retrieval for better
    relationship finding between entities.
    
    Args:
        query: The user query
        knowledge_graph_inst: Knowledge graph storage
        entities_vdb: Entity vector database
        relationships_vdb: Relationship vector database
        text_chunks_db: Text chunks database
        param: Query parameters
        global_config: Global configuration
        hashing_kv: Cache for query results
        system_prompt: Custom system prompt
        
    Returns:
        The query response
    """
    from .operate import (
        _get_node_data,
        _find_most_related_entities_from_relationships,
        _find_related_text_unit_from_relationships,
        process_combine_contexts,
        handle_cache,
        save_to_cache,
        CacheData,
        compute_args_hash,
    )
    
    # Reuse the LLM model function from the global config
    use_model_func = global_config["llm_model_func"]
    
    # Check cache
    args_hash = compute_args_hash("path", query)
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, "path"
    )
    if cached_response is not None:
        return cached_response
        
    # Extract keywords for entity retrieval
    from .operate import query_with_keywords
    keywords = await query_with_keywords(query, global_config)
    
    if not keywords:
        logger.warning("No keywords extracted from query")
        return "I couldn't understand your query. Please try rephrasing it."
        
    # Get entity and relationship data
    ll_entities_context, ll_relations_context, ll_text_units_context = await _get_node_data(
        keywords,
        knowledge_graph_inst,
        entities_vdb,
        text_chunks_db,
        param,
    )
    
    # Get node data
    node_datas = await entities_vdb.query(keywords, top_k=param.top_k)
    
    # Process node data
    node_data_list = []
    for node in node_datas:
        graph_node = await knowledge_graph_inst.get_node(node.get("entity_name", ""))
        if graph_node:
            node_degree = await knowledge_graph_inst.node_degree(node.get("entity_name", ""))
            node_data_list.append({
                **graph_node,
                "entity_name": node.get("entity_name", ""),
                "rank": node_degree
            })
    
    # Get path-based relationships
    path_relations = await find_most_related_edges_from_entities(
        node_data_list,
        param,
        knowledge_graph_inst,
    )
    
    # Convert path relations to relations context
    path_relations_context = ""
    if path_relations:
        relations_section_list = [["id", "path"]]
        for i, rel in enumerate(path_relations):
            relations_section_list.append([i, rel[0]])
        path_relations_context = list_of_list_to_csv(relations_section_list)
    
    # If no path relations were found, fall back to regular relations
    if not path_relations_context:
        path_relations_context = ll_relations_context
    
    # Build the context for the prompt
    context = f"""
-----path-based-information-----
-----entity information-----
```csv
{ll_entities_context}
```
-----path information-----
```csv
{path_relations_context}
```
-----Sources-----
```csv
{ll_text_units_context}
```
"""
    
    if param.only_need_context:
        return context
        
    # Generate the response
    from .prompt import PROMPTS
    sys_prompt_temp = system_prompt or PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=param.response_type
    )
    
    if param.only_need_prompt:
        return sys_prompt
        
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=param.stream,
    )
    
    # Clean up the response if needed
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<s>", "")
            .replace("</s>", "")
            .strip()
        )
    
    # Cache the response
    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode="path",
        ),
    )
    
    return response