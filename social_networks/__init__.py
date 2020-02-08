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
    social_networks_files = glob.glob("social_networks/*.txt") + glob.glob("social_networks/*.csv")

    # artist_edges.csv            Gemsec Facebook dataset
    # athletes_edges.csv          Gemsec Facebook dataset
    # company_edges.csv           Gemsec Facebook dataset
    # facebook_combined.txt       Social circles from Facebook
    # government_edges.csv        Gemsec Facebook dataset
    # HR_edges.csv                Gemsec Deezer dataset
    # HU_edges.csv                Gemsec Deezer dataset
    # new_sites_edges.csv         Gemsec Facebook dataset
    # politician_edges.csv        Gemsec Facebook dataset
    # public_figure_edges.csv     Gemsec Facebook dataset
    # RO_edges.csv                Gemsec Deezer dataset
    # Slashdot0811.txt            Slashdot social network from November 2008
    # Slashdot0902.txt            Slashdot social network from February 2009
    # soc-Epinions1.txt           Who-trusts-whom network of Epinions.com
    # tvshow_edges.csv            Gemsec Facebook dataset
    # Wiki-Vote.txt               Wikipedia who-votes-on-whom network
    expected_files = {'social_networks/Slashdot0902.txt', 'social_networks/facebook_combined.txt',
                      'social_networks/soc-Epinions1.txt', 'social_networks/Wiki-Vote.txt',
                      'social_networks/Slashdot0811.txt', 'social_networks/government_edges.csv',
                      'social_networks/public_figure_edges.csv', 'social_networks/artist_edges.csv',
                      'social_networks/politician_edges.csv', 'social_networks/new_sites_edges.csv',
                      'social_networks/HU_edges.csv', 'social_networks/company_edges.csv',
                      'social_networks/HR_edges.csv', 'social_networks/tvshow_edges.csv',
                      'social_networks/RO_edges.csv', 'social_networks/athletes_edges.csv'}

    for file in expected_files:
        if file not in social_networks_files:
            raise FileNotFoundError(f"Expected to find {file}, but this file does not exist")

    for file in social_networks_files:
        if ".txt" in file:
            graphs.append(read_file(file, "tsv"))
        else:
            graphs.append(read_file(file, "csv"))

    graphs.sort(key=lambda x: x.vcount())
    if idx is None:
        return graphs
    else:
        return graphs[idx]
