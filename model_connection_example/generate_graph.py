import argparse
import pandas as pd
import numpy as np
import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader, PyPDFium2Loader
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import random
import re
import seaborn as sns
import networkx as nx
from pyvis.network import Network
from helpers.df_helpers import documents2Dataframe, graph2Df, df2Graph


def contextual_proximity(df: pd.DataFrame) -> pd.DataFrame:
    # Melt the dataframe into a list of nodes
    df_long = pd.melt(
        df, id_vars=["chunk_id"], value_vars=["node_1", "node_2"], value_name="node"
    )
    df_long.drop(columns=["variable"], inplace=True)

    # Self join with chunk_id as the key will create a link between terms occurring in the same text chunk.
    df_wide = pd.merge(df_long, df_long, on="chunk_id", suffixes=("_1", "_2"))

    # Drop self-loops (node_1 == node_2)
    self_loops_drop = df_wide[df_wide["node_1"] == df_wide["node_2"]].index
    df2 = df_wide.drop(index=self_loops_drop).reset_index(drop=True)

    # Group and count edges
    df2 = (
        df2.groupby(["node_1", "node_2"])
        .agg({"chunk_id": [",".join, "count"]})
        .reset_index()
    )
    df2.columns = ["node_1", "node_2", "chunk_id", "count"]
    df2.replace("", np.nan, inplace=True)
    df2.dropna(subset=["node_1", "node_2"], inplace=True)

    # Drop edges with a count of 1
    df2 = df2[df2["count"] != 1]
    df2["edge"] = "kontekstowa bliskość"

    return df2


def main(session_id: str):
    input_directory = Path("./input_data")
    output_directory = Path("./data_output")

    # Check if input directory exists
    print(f"Input directory exists: {input_directory.exists()}")
    print(f"Input directory contents: {list(input_directory.glob('*'))}")

    # Load documents
    loader = DirectoryLoader(input_directory, show_progress=True)
    documents = loader.load()
    print(f"Number of documents loaded: {len(documents)}")

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(documents)
    print("Number of chunks =", len(chunks))

    if chunks:
        print(chunks[0].page_content)
    else:
        print("No documents found after splitting.")

    # Convert to DataFrame
    df = documents2Dataframe(chunks)
    print(df.shape)
    print(df.head())

    # Generate graphs for two models
    models = ["bielik", "llama"]
    regenerate = True

    for model in models:
        print(f"\nProcessing model: {model}")

        # Generate graph for the model
        if regenerate:
            if not df.empty:
                concepts_list = df2Graph(df, model=model)  # You can pass `model` as a parameter if needed
                df1 = graph2Df(concepts_list)

                # Create output directories for each model
                model_output_dir = output_directory / model
                if not model_output_dir.exists():
                    os.makedirs(model_output_dir)

                # Save results
                df1.to_csv(model_output_dir / f"graph_{session_id}.csv", sep="|", index=False)
                df.to_csv(model_output_dir / f"chunks_{session_id}.csv", sep="|", index=False)
            else:
                print("DataFrame is empty. No concepts to generate.")
                df1 = pd.DataFrame()
        else:
            model_output_dir = output_directory / model
            df1 = pd.read_csv(model_output_dir / "graph.csv", sep="|")

        # Check if the required columns are present
        required_columns = ["node_1", "node_2", 'edge']
        if all(column in df1.columns for column in required_columns):
            df1.replace("", np.nan, inplace=True)
            df1.dropna(subset=required_columns, inplace=True)
            df1['count'] = 4

            print(f"DataFrame for model {model}: {df1.shape}")
            print(df1.head())

            # Create contextual proximity dataframe
            df2 = contextual_proximity(df1)
            df2.tail()

            # Combine df1 and df2, removing duplicates
            dfg = pd.concat([df1, df2], axis=0)
            dfg = (
                dfg.groupby(["node_1", "node_2"])
                .agg({"chunk_id": ",".join, "edge": ','.join, 'count': 'sum'})
                .reset_index()
            )

            # Build graph
            nodes = pd.concat([dfg['node_1'], dfg['node_2']], axis=0).unique()
            G = nx.Graph()

            for node in nodes:
                G.add_node(str(node))

            for index, row in df1.iterrows():
                G.add_edge(
                    str(row["node_1"]),
                    str(row["node_2"]),
                    title=row["edge"],
                    weight=row['count'] / 4
                )

            # Community detection
            communities_generator = nx.community.girvan_newman(G)
            top_level_communities = next(communities_generator)
            next_level_communities = next(communities_generator)
            communities = sorted(map(sorted, next_level_communities))
            print(f"Number of Communities for model {model} = {len(communities)}")

            # Assign colors to communities
            def colors_to_community(communities) -> pd.DataFrame:
                colors = sns.color_palette("hls", len(communities)).as_hex()
                random.shuffle(colors)
                rows = []
                group = 0
                for community in communities:
                    color = colors.pop()
                    group += 1
                    for node in community:
                        rows.append({"node": node, "color": color, "group": group})
                return pd.DataFrame(rows)

            community_colors = colors_to_community(communities)

            for index, row in community_colors.iterrows():
                G.nodes[row['node']]['group'] = row['group']
                G.nodes[row['node']]['color'] = row['color']
                G.nodes[row['node']]['size'] = G.degree[row['node']]

            # Visualization
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

            graph_output_file = model_output_dir / f"index_{session_id}.html"
            net.show(str(graph_output_file), notebook=False)
        else:
            print(f"DataFrame is missing required columns: {required_columns}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process documents and generate graphs with session-specific outputs.")
    parser.add_argument("session_id", type=str, help="The session ID to be used for output files.")
    args = parser.parse_args()

    # Call the main function with the provided `session_id`
    main(args.session_id)