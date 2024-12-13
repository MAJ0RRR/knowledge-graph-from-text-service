import pandas as pd
import numpy as np
import os
from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader, PyPDFium2Loader
from langchain.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import random

katalog_wejsciowy = Path(f"./input_data")
katalog_wyjsciowy = Path(f"./data_output/")

loader = DirectoryLoader(katalog_wejsciowy, show_progress=True)
dokumenty = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150,
    length_function=len,
    is_separator_regex=False,
)

strony = splitter.split_documents(dokumenty)
print("Liczba fragmentów =", len(strony))
print(strony[0].page_content)


from helpers.df_helpers import documents2Dataframe
df = documents2Dataframe(strony)
print(df.shape)
print(df.head())


from helpers.df_helpers import df2Graph
from helpers.df_helpers import graph2Df

regenerate = True

if regenerate:
    concepts_list = df2Graph(df)
    dfg1 = graph2Df(concepts_list)
    if not os.path.exists(katalog_wyjsciowy):
        os.makedirs(katalog_wyjsciowy)

    dfg1.to_csv(katalog_wyjsciowy / "graph.csv", sep="|", index=False)
    df.to_csv(katalog_wyjsciowy / "chunks.csv", sep="|", index=False)
else:
    dfg1 = pd.read_csv(katalog_wyjsciowy / "graph.csv", sep="|")

dfg1.replace("", np.nan, inplace=True)
dfg1.dropna(subset=["node_1", "node_2", 'edge'], inplace=True)
dfg1['count'] = 4

print(dfg1.shape)
print(dfg1.head())


nodes = pd.concat([dfg1['node_1'], dfg1['node_2']], axis=0).unique()
nodes.shape
import networkx as nx
G = nx.Graph()

for node in nodes:
    G.add_node(
        str(node)
    )

for index, row in dfg1.iterrows():
    G.add_edge(
        str(row["node_1"]),
        str(row["node_2"]),
        title=row["edge"],
        weight=row['count']/4
    )
communities_generator = nx.community.girvan_newman(G)
top_level_communities = next(communities_generator)
next_level_communities = next(communities_generator)
communities = sorted(map(sorted, next_level_communities))
print("Number of Communities = ", len(communities))
print(communities)
import seaborn as sns
palette = "hls"

def colors2Community(communities) -> pd.DataFrame:
    p = sns.color_palette(palette, len(communities)).as_hex()
    random.shuffle(p)
    rows = []
    group = 0
    for community in communities:
        color = p.pop()
        group += 1
        for node in community:
            rows += [{"node": node, "color": color, "group": group}]
    df_colors = pd.DataFrame(rows)
    return df_colors



colors = colors2Community(communities)

for index, row in colors.iterrows():
    G.nodes[row['node']]['group'] = row['group']
    G.nodes[row['node']]['color'] = row['color']
    G.nodes[row['node']]['size'] = G.degree[row['node']]
from pyvis.network import Network

graph_output_directory = "./docs/index.html"

net = Network(
    notebook=False,
    cdn_resources="remote",
    height="900px",
    width="100%",
    select_menu=True,
    filter_menu=False,
)

net.from_nx(G)

net.force_atlas_2based(central_gravity=0.015, gravity=-31)
net.show_buttons(filter_=["physics"])

net.show(graph_output_directory, notebook=False)
