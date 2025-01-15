from collections import Counter 
import networkx as nx
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import seaborn as sns
from scipy.stats import ks_2samp
import time

import plotting
from constants_and_utils import *
from generate_personas import *

def load_list_of_graphs(prefix, start_seed, end_seed, directed=True, include_ts=False):
    """
    Load list of graphs from adjlist. By default, assume directed graphs.
    """
    list_of_G = []
    min_time, max_time = None, None
    for s in range(start_seed, end_seed):
        fn = os.path.join(PATH_TO_TEXT_FILES, f'{prefix}_{s}.adj')
        mod = time.ctime(os.path.getmtime(fn))  # last modified time
        if (min_time is None) or (mod < min_time):
            min_time = mod 
        elif (max_time is None) or (mod > max_time):
            max_time = mod 
        if directed:
            G = nx.read_adjlist(fn, create_using=nx.DiGraph)
        else:
            G = nx.read_adjlist(fn)
        list_of_G.append(G)
    if include_ts:
        return list_of_G, min_time, max_time
    return list_of_G

def get_edge_proportions(list_of_G):
    """
    What proportion of the time does each edge appear?
    """
    edge_counts = {}
    # initialize all possible edge counts to 0
    nodes = list_of_G[0].nodes()
    for ni in nodes:
        for nj in nodes:
#            if ni != nj:
            edge_counts[(ni, nj)] = 0
    assert len(edge_counts) == (len(nodes) * (len(nodes))) # CHANGE
    # add actual edges
    for G in list_of_G:
        for e in G.edges():
            edge_counts[e] = edge_counts[e] + 1
    # sort by highest to lowest count
    sorted_edges = sorted(edge_counts.keys(), key=lambda x: -edge_counts[x])
    sorted_props = [edge_counts[e]/len(list_of_G) for e in sorted_edges]
    return sorted_edges, sorted_props

def compute_edge_distance(G1, G2):
    """
    Out of all possible edges, how often do G1 and G2 disagree? 
    Disagree means edge is present in one and missing in the other.
    Return the proportion of edges where G1 and G2 disagree.
    """
    assert set(G1.nodes()) == set(G2.nodes())
    E1 = set(G1.edges())
    E2 = set(G2.edges())
    distance = len(E1 - E2)  # present in G1 but missing in G2
    distance += len(E2 - E1)  # present in G2 but missing in G1
    num_nodes = len(G1.nodes())
    num_edges = num_nodes * (num_nodes-1)  # total num possible edges
    return distance / num_edges

def get_edge_summary(list_of_G, save_name):
    """
    Summarize edge-related statistics: 
    1) num edges per graph,
    2) average edge distance between graph pairs,
    3) proportion of graphs that each edge appears.
    """
    num_edges = [len(G.edges()) for G in list_of_G]

    plotting.plot_edges(num_edges, save_name)

    all_real_d = []
    for i, G1 in enumerate(list_of_G):
        if i < (len(list_of_G)-1):
            for G2 in list_of_G[i+1:]:
                all_real_d.append(compute_edge_distance(G1, G2))
    print('Average edge distance between graphs: %.3f' % np.mean(all_real_d))

    plotting.plot_edge_dist(all_real_d, save_name)

    edges, props = get_edge_proportions(list_of_G)
    print('Most common edges:')
    for i in range(30):
        print('%d. %s -> %s (p=%.3f)' % (i, edges[i][0], edges[i][1], props[i]))

    plotting.plot_props(props, edges, save_name)

def compute_exp_cross_from_group_counts(group_counts):
    """
    Compute expected proportion of edges that are cross-relations, given
    number of nodes that are in each group.
    This matches the method below that uses the complete graph.
    """
    groups = list(group_counts.keys())
    cr_total = 0
    for i1, g1 in enumerate(groups[:-1]):
        for g2 in groups[i1+1:]:
            cr_total += group_counts[g1] * group_counts[g2]
    num_nodes = np.sum(list(group_counts.values()))
    total_num_edges = num_nodes * (num_nodes-1) / 2
    return cr_total / total_num_edges 

def compute_cross_proportions(G, personas, demo_keys, ratio=True):
    """
    Compute proportion of edges that are cross-relations, per demographic variable.
    If ratio is true, divide by expected proportions.
    """
    observed = _compute_cross_proportions(G, personas, demo_keys)
    if not ratio:
        return observed 
    complete = nx.complete_graph(G.nodes())
    expected = _compute_cross_proportions(complete, personas, demo_keys)
    return observed / expected

def _compute_cross_proportions(G, personas, demo_keys):
    """
    Helper function to compute the proportion of edges in the graph that are 
    cross-relations, per demographic variable.
    """ 
    # count cross-relationships in graph
    crs = np.zeros(len(demo_keys))
    for source, target in G.edges():
        demo1 = personas[source]
        demo2 = personas[target]
        for ind, d in enumerate(demo_keys):
            if d == 'age':  # take absolute difference for age
                diff = abs(int(demo1[d]) - int(demo2[d]))
            else:
                diff = int(demo1[d] != demo2[d])  # 1 if they are different, 0 otherwise
            crs[ind] += diff
    # get proportion of edges that are cross-relations or average difference in age
    props = crs / len(G.edges())  
    return props

def compute_same_proportions(G, personas, demo_keys, ratio=True):
    """
    Compute proportion of edges that are same-group relations, per demographic variable.
    If ratio is true, divide by expected proportions.
    """
    observed = _compute_same_proportions(G, personas, demo_keys)
    if not ratio:
        return observed 
    complete = nx.complete_graph(G.nodes())
    expected = _compute_same_proportions(complete, personas, demo_keys)
    return observed / expected

def _compute_same_proportions(G, personas, demo_keys):
    """
    Helper function to compute the proportion of edges in the graph that are 
    same relations, per demographic variable.
    """ 
    # count same-relationships in graph
    same_counts = np.zeros(len(demo_keys))
    for source, target in G.edges():
        demo1 = personas[source]
        demo2 = personas[target]
        for ind, d in enumerate(demo_keys):
            if d == 'age':  # check whether age is within 10
                same = int(abs(int(demo1[d]) - int(demo2[d])) <= 10)
            else:
                same = int(demo1[d] == demo2[d])
            same_counts[ind] += same
    # get proportion of edges that are same relation
    props = same_counts / len(G.edges())  
    return props
    
def global_efficiency(G):
    """
    Compute the global efficiency of the graph.
    """
    n = len(G)
    if n < 2:
        return 0

    # Ensure we handle the generator properly
    path_lengths = nx.shortest_path_length(G)
    efficiency = 0

    for source, target_dict in path_lengths:  # Unpack the tuples
        for distance in target_dict.values():
            if distance > 0:  # Avoid dividing by zero
                efficiency += 1 / distance

    return efficiency / (n * (n - 1))


def degree_gini_coefficient(G):
    """
    Compute the Gini coefficient of the degree distribution.
    """
    degrees = [d for _, d in G.degree()]
    sorted_degrees = sorted(degrees)
    n = len(degrees)
    cumulative_sum = sum((i + 1) * deg for i, deg in enumerate(sorted_degrees))
    return (2 * cumulative_sum / (n * sum(sorted_degrees))) - (n + 1) / n
    


def summarize_network_metrics(list_of_G, personas, demo_keys, save_name, demos=True):
    """
    Summarizes network metrics for a list of graphs, including homophily, graph-level,
    and node-level metrics, and saves them to CSV files.

    Parameters:
    - list_of_G: List of NetworkX graphs
    - personas: Dictionary mapping nodes to attributes
    - demo_keys: List of demographic keys to analyze for homophily
    - save_name: Directory name to save results
    - demos: Whether to compute homophily metrics
    """
    output_dir = os.path.join(PATH_TO_STATS_FILES, save_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ### ---------------------------------- Homophily Metrics ---------------------------------- ###
    if demos:
        homophily_metrics_df = pd.DataFrame({'graph_nr': [], 'demo': [], '_metric_value': [], 'save_name': []})
        for graph_nr, G in enumerate(list_of_G):
            try:
                # Compute same-group homophily
                same_homophily = list(compute_same_proportions(G, personas, demo_keys, ratio=True))
                same_df = pd.DataFrame({
                    'graph_nr': graph_nr,
                    'demo': demo_keys,
                    'metric_name': 'same_ratio',
                    '_metric_value': same_homophily,
                    'save_name': [save_name] * len(demo_keys)
                })

                # Compute cross-group homophily
                cross_homophily = list(compute_cross_proportions(G, personas, demo_keys, ratio=True))
                cross_df = pd.DataFrame({
                    'graph_nr': graph_nr,
                    'demo': demo_keys,
                    'metric_name': 'cross_ratio',
                    '_metric_value': cross_homophily,
                    'save_name': [save_name] * len(demo_keys)
                })

                homophily_metrics_df = pd.concat([homophily_metrics_df, same_df, cross_df], ignore_index=True)
            except Exception as e:
                print(f"Error computing homophily for graph {graph_nr}: {e}")

        fn = os.path.join(output_dir, "homophily.csv")
        if not homophily_metrics_df.empty:
            homophily_metrics_df.to_csv(fn, index=False)
            print(f"Saved homophily metrics to {fn}")
        else:
            print("No homophily metrics to save.")

    ### ---------------------------------- Graph-Level Metrics ---------------------------------- ###
    network_metrics_df = pd.DataFrame({'graph_nr': [], 'metric_name': [], '_metric_value': [], 'save_name': []})

    # Define graph-level metrics and functions
    network_metrics = ['density', 'avg_clustering_coef', 'prop_nodes_lcc', 'radius', 'diameter', 'avg_shortest_path', 'modularity']
    network_func = [
        nx.density,
        nx.average_clustering,
        prop_nodes_in_giant_component,
        nx.radius,
        nx.diameter,
        nx.average_shortest_path_length,
        nx.community.modularity
    ]

    # Add new metrics: Assortativity, Efficiency, Degree Gini, PageRank
    network_metrics += ['assortativity', 'efficiency', 'degree_gini', 'pagerank_avg', 'pagerank_normalized_entropy']
    network_func += [
        nx.degree_assortativity_coefficient,
        global_efficiency,
        degree_gini_coefficient,
        lambda G: compute_pagerank_avg(G),
        lambda G: compute_normalized_pagerank_entropy(G)
    ]

    for graph_nr, G in enumerate(list_of_G):
        for metric_name, f in zip(network_metrics, network_func):
            try:
                if metric_name in ['radius', 'diameter', 'avg_shortest_path']:
                    largest_cc = sorted(nx.connected_components(G.to_undirected()), key=len, reverse=True)[0]
                    _metric_value = f(G.subgraph(largest_cc).to_undirected()) / np.log(len(largest_cc))
                elif metric_name == 'modularity':
                    comms = nx.community.louvain_communities(G.to_undirected())
                    _metric_value = f(G.to_undirected(), comms)
                else:
                    _metric_value = f(G.to_undirected()) if metric_name != 'assortativity' else f(G)

                network_metrics_df = pd.concat([network_metrics_df, pd.DataFrame({
                    'graph_nr': [graph_nr],
                    'metric_name': [metric_name],
                    '_metric_value': [_metric_value],
                    'save_name': [save_name]
                })], ignore_index=True)
            except Exception as e:
                print(f"Error computing {metric_name} for graph {graph_nr}: {e}")

    fn = os.path.join(output_dir, "network_metrics.csv")
    if not network_metrics_df.empty:
        network_metrics_df.to_csv(fn, index=False)
        print(f"Saved network metrics to {fn}")
    else:
        print("No network metrics to save.")

    ### ---------------------------------- Node-Level Metrics ---------------------------------- ###
    node_metrics = ['degree_centrality', 'betweenness_centrality', 'closeness_centrality']
    node_func = [nx.degree_centrality, nx.betweenness_centrality, nx.closeness_centrality]

    node_metrics_df = pd.DataFrame({'node': [], 'graph_nr': [], 'metric_name': [], '_metric_value': [], 'save_name': []})

    for graph_nr, G in enumerate(list_of_G):
        for metric_name, f in zip(node_metrics, node_func):
            try:
                metric_dict = f(G.to_undirected())
                temp_df = pd.DataFrame(metric_dict.items(), columns=['node', '_metric_value'])
                temp_df['graph_nr'] = graph_nr
                temp_df['metric_name'] = metric_name
                temp_df['save_name'] = save_name
                node_metrics_df = pd.concat([node_metrics_df, temp_df], ignore_index=True)
            except Exception as e:
                print(f"Error computing {metric_name} for graph {graph_nr}: {e}")

    fn = os.path.join(output_dir, "node_metrics.csv")
    if not node_metrics_df.empty:
        node_metrics_df.to_csv(fn, index=False)
        print(f"Saved node metrics to {fn}")
    else:
        print("No node metrics to save.")

### ---------------------------------- PageRank Metric Helpers ---------------------------------- ###
def compute_pagerank_avg(G):
    """
    Compute the average PageRank for a graph.

    Parameters:
    - G: NetworkX graph

    Returns:
    - Average PageRank (float)
    """
    pagerank = nx.pagerank(G.to_undirected())
    return np.mean(list(pagerank.values()))

def compute_normalized_pagerank_entropy(G):
    """
    Compute the normalized entropy of the PageRank distribution for a graph.

    Parameters:
    - G: NetworkX graph

    Returns:
    - Normalized entropy of the PageRank distribution (float, between 0 and 1)
    """
    pagerank = nx.pagerank(G.to_undirected())
    pagerank_values = np.array(list(pagerank.values()))

    # Normalize PageRank values to create a probability distribution
    normalized = pagerank_values / pagerank_values.sum()

    # Compute entropy
    entropy = -np.sum(normalized * np.log2(normalized + 1e-9))  # Add small value to avoid log(0)

    # Normalize by the maximum possible entropy
    max_entropy = np.log2(len(G.nodes()))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0  # Avoid division by zero

    return normalized_entropy




def compute_pairwise_ratios(G, personas, demo, cutoff=1):
    """
    Compute matrix where m[a, b] represents the observed number of edges between
    nodes in group a and b divided by expected number of such edges.
    """
    vals = [personas[k][demo] for k in personas]
    groups = [g for g,c in Counter(vals).most_common() if c >= cutoff]
    obs_mat = _compute_pairwise_props(G, personas, demo, groups)
    exp_mat = _compute_pairwise_props(nx.complete_graph(G.nodes()), personas, demo, groups)
    ratio = obs_mat / exp_mat
    return groups, ratio


def _compute_pairwise_props(G, personas, demo, groups):
    assert type(groups) == list 
    mat = np.zeros((len(groups), len(groups)))
    for u,v in G.to_undirected().edges():
        if (personas[u][demo] in groups) and (personas[v][demo] in groups):
            g1 = groups.index(personas[u][demo])
            g2 = groups.index(personas[v][demo])
            mat[g1][g2] += 1
            mat[g2][g1] += 1
    return mat / np.sum(mat)
    

def compute_isolation_index(G, personas):
    """
    Compute political isolation index, following Halberstam and Knight (2016).
    """
    nodes = list(G.nodes())
    A = nx.adjacency_matrix(G, nodelist=nodes).todense()
    politics = np.array([personas[n]['political affiliation'] for n in nodes])
    assert A.shape == (len(politics), len(politics))

    # compute share conservative
    num_neighbors_c = A @ (politics == 'Republican').astype(int)
    num_neighbors_l = A @ (politics == 'Democrat').astype(int)
    share_conservative = num_neighbors_c / (num_neighbors_c + num_neighbors_l)
    
    # compute conservative exposure
    degree = np.sum(A, axis=0)
    conservative_exposure = (A @ share_conservative) / degree

    # compute isolation
    avg_exposure_c = np.mean(conservative_exposure[politics == 'Republican'])
    avg_exposure_l = np.mean(conservative_exposure[politics == 'Democrat'])
    isolation = avg_exposure_c-avg_exposure_l
    return isolation, avg_exposure_c, avg_exposure_l

def compute_polarization(G, personas):
    """
    Compute polarization, following Garimella and Weber (2017).
    """
    nodes = list(G.nodes())
    A = nx.adjacency_matrix(G, nodelist=nodes).todense()
    politics = np.array([personas[n]['political affiliation'] for n in nodes])
    assert A.shape == (len(politics), len(politics))

    alpha = np.ones(len(politics))
    beta = np.ones(len(politics))
    alpha += A @ (politics == 'Democrat').astype(int)
    beta += A @ (politics == 'Republican').astype(int)
    lean = alpha / (alpha + beta)
    pol = 2 * np.abs(0.5 - lean)
    return pol 
    

    
    
def save_polarization_metrics(list_of_G, personas, save_name):
    """
    Compute and save polarization metrics for a list of graphs.

    Parameters:
    - list_of_G: List of NetworkX graphs
    - personas: Dictionary mapping nodes to attributes
    - save_name: Directory name to save the polarization metrics
    """
    # Ensure the output directory exists within the stats folder
    output_dir = os.path.join(PATH_TO_STATS_FILES, save_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    polarization_metrics = []

    for graph_nr, G in enumerate(list_of_G):
        try:
            pol = compute_polarization(G, personas)
            avg_pol = np.mean(pol)  # Average polarization across users
            polarization_metrics.append({
                'graph_nr': graph_nr,
                'metric_name': 'polarization',
                '_metric_value': avg_pol,
                'save_name': save_name
            })
        except Exception as e:
            print(f"Error computing polarization for graph {graph_nr}: {e}")

    # Save to CSV in the correct directory
    fn = os.path.join(output_dir, "polarization_metrics.csv")
    polarization_df = pd.DataFrame(polarization_metrics)
    if not polarization_df.empty:
        polarization_df.to_csv(fn, index=False)
        print(f"Saved polarization metrics to {fn}")
    else:
        print("No polarization metrics to save.")

def plot_expected_vs_observed_age_gaps(list_of_G, personas):
    """
    """
    obs_gaps = []
    for G in list_of_G:
        for (u,v) in G.edges():
            gap = np.abs(personas[u]['age'] - personas[v]['age'])
            obs_gaps.append(gap)
    
    exp_gaps = []
    complete = nx.complete_graph(list_of_G[0].nodes())
    for (u,v) in complete.edges():
        gap = np.abs(personas[u]['age'] - personas[v]['age'])
        exp_gaps.append(gap)

    bins = np.arange(0, 101, 5)
    plt.figure(figsize=(6,4))
    plt.hist(exp_gaps, color='tab:blue', label='expected', density=True, bins=bins)
    plt.hist(obs_gaps, color='tab:orange', alpha=0.5, density=True, label='observed', bins=bins)
    plt.xlabel('Age gap btwn friends', fontsize=16)
    plt.grid(alpha=0.2)
    # ymin, ymax = plt.ylim()
    # plt.vlines([np.mean(exp_gaps)], ymin, ymax, color='tab:blue', label=f'exp mean={np.mean(exp_gaps):0.3f}')
    # plt.vlines([np.mean(obs_gaps)], ymin, ymax, color='tab:orange', label=f'obs mean={np.mean(obs_gaps):0.3f}')
    plt.legend()

def compare_network_metrics(network_df, metric_name, save_name):
    """
    Get quantitative comparison of real and generated networks.
    """
    real_vals = network_df[(network_df.save_name == 'real') & (network_df.metric_name == metric_name)]['_metric_value'].values
    gen_vals = network_df[(network_df.save_name == save_name) & (network_df.metric_name == metric_name)]['_metric_value'].values
    print(f'Found {len(real_vals)} real and {len(gen_vals)} generated values')
    mean_diff = np.abs(np.mean(real_vals)-np.mean(gen_vals))
    mean_diff_norm = mean_diff / np.std(real_vals)
    res = ks_2samp(real_vals, gen_vals)
    return mean_diff, mean_diff_norm, res.statistic, res.pvalue

def parse():
    # Create the parser
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    
    # Add arguments
    parser.add_argument('--persona_fn', type=str, default='us_50_gpt4o_w_interests.json', help='What is the name of the persona file you want to use?')
    parser.add_argument('--network_fn', type=str, help='What is the name of the network file you want to use?')
    parser.add_argument('--num_networks', type=int, help='How many networks are there?')
    parser.add_argument('--demos_to_include', nargs='+', default=['gender', 'race/ethnicity', 'age', 'religion', 'political affiliation'])

    # Parse the arguments
    args = parser.parse_args()

    # Print the arguments
    print("Persona file", args.persona_fn)
    print("Network file", args.network_fn)
    print("Number of networks", args.num_networks)
    
    return args


def count_communities(list_of_G, save_name):

    counts = []
    sizes = []
    mods = []
    for G in list_of_G:
        comms = nx.community.louvain_communities(G, seed=42)
        counts.append(len(comms))
        sizes = sizes + [len(c) for c in comms]

        modularity = nx.community.modularity(G, comms)
        mods.append(modularity)

    plotting.plot_communities(counts, sizes, mods, save_name)


if __name__ == '__main__':
    args = parse()
    
    # Load graphs
    list_of_G = load_list_of_graphs(args.network_fn, 0, args.num_networks)
    print(f"Loaded {len(list_of_G)} graphs.")
    
    # Load personas
    fn = os.path.join(PATH_TO_TEXT_FILES, args.persona_fn)
    with open(fn, 'r') as f:
        personas = json.load(f)
    print(f"Loaded personas from {fn}. Sample: {list(personas.items())[:5]}")

    # Compute and save polarization metrics
    save_polarization_metrics(list_of_G, personas, save_name=args.network_fn)

    # Compute and save other metrics
    summarize_network_metrics(list_of_G, personas, args.demos_to_include, save_name=args.network_fn)