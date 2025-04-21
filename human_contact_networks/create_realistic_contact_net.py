"""Create contact networks based on published datasets.
Developed in python 2.7.

--Bo Li
"""

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


def save_graph_as_csv(graph, node_data, data_name, dir, save_node_data=True):
    """Save the graph as CSV file."""
    nlist = list(graph.nodes())
    elist = list(graph.edges())
    df_v = node_data

    if save_node_data:
        ## Save node data:
        new_df_v = []
        for xv in range(1, graph.number_of_nodes()+1):  ## enumerate nodes starting from 1, for the convenience of programming in julia
            v = nlist[xv-1]
            v_type = df_v[ df_v.iloc[:, 0] == v ].iloc[0, 1]    ## node property
            new_df_v.append({"node":xv, "node_name":v, "node_type":v_type})
        new_df_v = pd.DataFrame(new_df_v, columns=["node", "node_name", "node_type"])
        new_df_v.to_csv(dir + data_name + "_nodes.csv", index=False)

    new_df_e = []
    for xe in range(1, graph.number_of_edges()+1):      ## enumerate edges starting from 1
        i, j = elist[xe-1]
        xi, xj = nlist.index(i)+1, nlist.index(j)+1     ## enumerate edges starting from 1
        duration = graph[i][j]["duration"]
        new_df_e.append({"edge":xe, "from_node":xi, "to_node":xj, "duration":duration})
    new_df_e = pd.DataFrame(new_df_e, columns=["edge", "from_node", "to_node", "duration"])
    new_df_e.to_csv(dir + data_name + "_edges.csv", index=False)


def create_net_US_high_school_2010_full_static():
    """US high school net, M. Salathe et al. PNAS 2010.
    788 individuals."""
    data_name = "US_high_school_2010"
    dir = "./" + data_name + "/"
    df_e = pd.read_csv(dir + "sd02.txt", sep="\s+", header=None)
    df_v = pd.read_csv(dir + "sd03.txt", sep="\s+", header=None)

    graph = nx.Graph()
    no_of_contacts, no_of_columns = df_e.shape
    for ind in range(no_of_contacts):
        i, j, duration = df_e.iloc[ind]
        if graph.has_edge(i, j):
            graph[i][j]["duration"] += duration
        else:
            graph.add_edge(i, j)
            graph[i][j]["duration"] = duration

    save_graph_as_csv(graph, df_v, data_name+"_full", "./full_static/")


def create_sociopatterns_net_full_static():
    """Sociapatterns data. Temperal network projected onto static network.
    Each contact between two individuals will register a link between them.
    The 'duration' property records the cumulative number of contacts."""

    ## Uncomment one of the following code block for generating the desired full-static network:

    # data_name = "sociopatterns_workplace_2013"
    # dir = "./" + data_name + "/"
    # df_e = pd.read_csv(dir + "tij_InVS.dat", sep="\s+", header=None)
    # df_v = pd.read_csv(dir + "metadata_InVS13.txt", sep="\s+", header=None)

    # data_name = "sociopatterns_workplace_2015"
    # dir = "./" + data_name + "/"
    # df_e = pd.read_csv(dir + "tij_InVS15.dat", sep="\s+", header=None)
    # df_v = pd.read_csv(dir + "metadata_InVS15.txt", sep="\s+", header=None)

    # data_name = "sociopatterns_primary_school_2014"
    # dir = "./" + data_name + "/"
    # df_e = pd.read_csv(dir + "primaryschool.csv", sep="\s+", header=None)
    # df_v = pd.read_csv(dir + "metadata_primaryschool.txt", sep="\s+", header=None)

    data_name = "sociopatterns_high_school_2013"
    dir = "./" + data_name + "/"
    df_e = pd.read_csv(dir + "High-School_data_2013.csv", sep="\s+", header=None)
    df_v = pd.read_csv(dir + "metadata_2013.txt", sep="\s+", header=None)

    graph = nx.Graph()
    no_of_contacts, no_of_columns = df_e.shape
    for ind in range(no_of_contacts):
        time, i, j = df_e.iloc[ind, :3]
        if graph.has_edge(i, j):
            graph[i][j]["duration"] += 1
        else:
            graph.add_edge(i, j)
            graph[i][j]["duration"] = 1

    save_graph_as_csv(graph, df_v, data_name+"_full", "./full_static/")


def create_human_contact_net_thresholded_static():
    """Temperal human contact network projected onto static network.
    An edge is present if and only if contact duration is larger than a threshold (set to be 3 here)."""

    ## Uncomment one of the following line for generating the desired thresholded network:

    # data_name = "US_high_school_2010"
    # data_name = "sociopatterns_workplace_2013"
    # data_name = "sociopatterns_workplace_2015"
    # data_name = "sociopatterns_primary_school_2014"
    data_name = "sociopatterns_high_school_2013"

    dir = "./full_static/"
    df_e = pd.read_csv(dir + data_name + "_full_edges.csv", sep=",")
    df_v = pd.read_csv(dir + data_name + "_full_nodes.csv", sep=",")
    threshold = 3

    graph = nx.Graph()
    no_of_edges, no_of_columns = df_e.shape
    print no_of_edges, no_of_columns
    for ind in range(no_of_edges):
        i, j, duration = df_e.iloc[ind, 1:4]
        # print i, j, duration
        if duration >= threshold:
            graph.add_edge(i, j)
            graph[i][j]["duration"] = duration

    print nx.is_connected(graph), graph.number_of_nodes(), graph.number_of_edges()

    save_graph_as_csv(graph, df_v, data_name, "./thresholded_static/")


def create_London_transit_net():
    """Create the London transit network,
    composed of a road net and metro net in the city center."""
    dir = "./london_transit_net/"
    df_e = pd.read_csv(dir + "aggregate_edge_list.csv", sep=",", header=None)
    df_c = pd.read_csv(dir + "road_coordinates.csv", sep=",", header=None)

    no_of_edges, no_of_e_columns = df_e.shape
    no_of_nodes, no_of_n_columns = df_c.shape
    graph = nx.Graph()
    for e in range(no_of_edges):
        i, j = df_e.iloc[e, :]
        graph.add_edge(i, j)
        graph[i][j]["duration"] = 1

    save_graph_as_csv(graph, df_c, "London_transit_net", "./thresholded_static/", save_node_data=False)

    ## Manually node data with coordinates:
    new_df_c = [ [i+1, i+1, df_c.iloc[i, 0], df_c.iloc[i, 1]] for i in range(no_of_nodes) ]
    new_df_c = pd.DataFrame(new_df_c, columns=["node", "node_name", "latitude", "longitude"])
    new_df_c.to_csv("./thresholded_static/London_transit_net_nodes.csv", index=False)


def network_characteristics():
    """Some network characteristics and visualization."""
    # data_name = "US_high_school_2010"
    data_name = "sociopatterns_workplace_2013"
    # data_name = "sociopatterns_workplace_2015"
    # data_name = "sociopatterns_primary_school_2014"
    # data_name = "sociopatterns_high_school_2013"
    # data_name = "London_transit_net"

    dir = "./thresholded_static/"
    df_e = pd.read_csv(dir + data_name + "_edges.csv", sep=",")
    df_v = pd.read_csv(dir + data_name + "_nodes.csv", sep=",")

    graph = nx.Graph()
    no_of_edges, no_of_columns = df_e.shape
    for ind in range(no_of_edges):
        xe, i, j, duration = df_e.iloc[ind, :4]
        graph.add_edge(i, j)
        graph[i][j]["duration"] = duration

    print data_name, graph.number_of_nodes(), graph.number_of_edges()

    nx.draw(graph, node_size=10)
    plt.show()


# create_net_US_high_school_2010_full_static()
# create_sociopatterns_net_full_static()
# create_human_contact_net_thresholded_static()
# create_London_transit_net()
network_characteristics()
