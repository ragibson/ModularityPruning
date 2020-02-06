import glob
import igraph as ig


def read_file(filename, format):
    with open(filename) as file:
        lines = file.readlines()
    if format is "tsv":
        edges = [tuple(int(x) for x in l.split()) for l in lines if len(l) and l[0] != '#']
        G = ig.Graph(edges, directed=False)
        return G.clusters().giant()
    elif format is "csv":
        edges = [tuple(int(x) for x in l.split(",")) for l in lines if len(l) and "node" not in l]
        G = ig.Graph(edges, directed=False)
        return G.clusters().giant()
    else:
        return None


def read_graphs(idx=None):
    graphs = []
    for file in glob.glob("social_networks/*.txt") + glob.glob("social_networks/*.csv"):
        if ".txt" in file:
            graphs.append(read_file(file, "tsv"))
        else:
            graphs.append(read_file(file, "csv"))

    graphs.sort(key=lambda x: x.vcount())
    if idx is None:
        return graphs
    else:
        return graphs[idx]
