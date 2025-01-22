import argparse
import pandas as pd
import numpy as np
import os
from pathlib import Path
import random
import re

import seaborn as sns
import networkx as nx
from pyvis.network import Network

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredPDFLoader,
    PyPDFium2Loader,
    PyPDFDirectoryLoader,
    DirectoryLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

from helpers.df_helpers import documents2Dataframe, graph2Df, df2Graph

def contextual_proximity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create edges between terms that occur in the same text chunk,
    effectively measuring 'contextual proximity'.
    """
    # Melt the DataFrame into a list of nodes
    df_long = pd.melt(
        df, id_vars=["chunk_id"], value_vars=["node_1", "node_2"], value_name="node"
    )
    df_long.drop(columns=["variable"], inplace=True)

    # Self-join with chunk_id as the key
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

    # Drop edges with a count of 1 (optional, tweak as desired)
    df2 = df2[df2["count"] != 1]

    df2["edge"] = "kontekstowa bliskość"
    return df2


def main(session_id: str):
    """
    Main function which:
      1. Loads PDFs from input_data/{session_id}
      2. Splits them into chunks
      3. Builds conceptual graphs for each model
      4. Writes output (CSV, HTML) to data_output/{session_id}/... and docs/{session_id}/...
    """

    # Adjust these paths to match your container paths if needed (absolute vs. relative).
    input_directory  = Path(f"/app/model_connection_example/input_data/{session_id}")
    output_directory = Path(f"/app/model_connection_example/data_output/{session_id}")
    docs_directory   = Path(f"/app/model_connection_example/docs/{session_id}")
    # Create output/docs directories if they don't exist
    output_directory.mkdir(parents=True, exist_ok=True)
    docs_directory.mkdir(parents=True, exist_ok=True)

    # Debug / Info
    print("=== GENERATE_GRAPH.PY ===")
    print(f"Session ID: {session_id}")
    print(f"Looking for PDFs in: {input_directory}")
    print(f"Writing CSV/graph data to: {output_directory}")
    print(f"Writing final HTML to: {docs_directory}")

    # Check if input directory exists and has files
    if not input_directory.is_dir():
        print(f"[WARNING] Input directory {input_directory} does not exist or is not a directory.")
        return

    input_files = list(input_directory.glob("*.pdf"))
    print(f"Found {len(input_files)} PDF(s) in {input_directory}")

    if not input_files:
        print("[INFO] No PDFs found. Aborting graph generation.")
        return

    # ----------------------------------------
    # 1. Load documents
    # ----------------------------------------
    loader = DirectoryLoader(input_directory, show_progress=True)
    documents = loader.load()
    print(f"Number of documents loaded: {len(documents)}")

    # ----------------------------------------
    # 2. Split documents into chunks
    # ----------------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(documents)
    print(f"Number of chunks = {len(chunks)}")

    if chunks:
        print("[DEBUG] Example chunk:\n", chunks[0].page_content[:200], "...")
    else:
        print("[DEBUG] No chunks generated from PDFs")

    # ----------------------------------------
    # 3. Convert to DataFrame
    # ----------------------------------------
    df = documents2Dataframe(chunks)
    print("DataFrame shape (chunks):", df.shape)

    # ----------------------------------------
    # 4. Generate graphs for each model
    # ----------------------------------------
    models = ["bielik"] # llama
    regenerate = True  # If True, re-generate; otherwise read existing CSV

    for model in models:
        print(f"\n=== Processing model: {model} ===")

        # Each model gets its own subfolder in output_directory
        model_output_dir = output_directory / model
        model_output_dir.mkdir(parents=True, exist_ok=True)

        # If there's no data, skip
        if df.empty:
            print("[WARNING] DataFrame is empty. No concepts to generate for this model.")
            continue

        if regenerate:
            concepts_list = df2Graph(df, model=model)  
            df1 = graph2Df(concepts_list)

            # Save base graph as CSV for debugging or reference
            df1_file = model_output_dir / f"graph_{session_id}.csv"
            df1.to_csv(df1_file, sep="|", index=False)
            print(f"[INFO] Wrote base graph to {df1_file}")

            # Also save the chunked data as well
            df_chunks_file = model_output_dir / f"chunks_{session_id}.csv"
            df.to_csv(df_chunks_file, sep="|", index=False)
            print(f"[INFO] Wrote chunk data to {df_chunks_file}")

        else:
            # If not regenerating, read the CSV from disk
            df1_file = model_output_dir / f"graph_{session_id}.csv"
            df1 = pd.read_csv(df1_file, sep="|")

        # ----------------------------------------
        # Build final, merged graph data
        # ----------------------------------------
        required_columns = ["node_1", "node_2", "edge"]
        if not all(col in df1.columns for col in required_columns):
            print(f"[ERROR] DataFrame missing required columns: {required_columns}")
            continue

        df1.replace("", np.nan, inplace=True)
        df1.dropna(subset=required_columns, inplace=True)

        # Assign a basic 'count' to weigh edges
        df1["count"] = 4  
        print(f"[DEBUG] df1 shape for {model}: {df1.shape}")

        # Create context-proximity edges and merge
        df2 = contextual_proximity(df1)
        dfg = pd.concat([df1, df2], axis=0)

        # Group them to remove duplicates; sum the 'count'
        dfg = (
            dfg.groupby(["node_1", "node_2"])
            .agg({"chunk_id": ",".join, "edge": ",".join, "count": "sum"})
            .reset_index()
        )

        # ----------------------------------------
        # 5. Build a NetworkX graph
        # ----------------------------------------
        nodes = pd.concat([dfg["node_1"], dfg["node_2"]], axis=0).unique()
        G = nx.Graph()

        # Add nodes
        for node in nodes:
            G.add_node(str(node))

        # Add edges (just from df1 or from the merged dfg)
        for index, row in dfg.iterrows():
            G.add_edge(
                str(row["node_1"]),
                str(row["node_2"]),
                title=row["edge"],
                weight=row["count"] / 4.0,
            )

        # ----------------------------------------
        # 6. Community detection 
        # ----------------------------------------
        communities_generator = nx.community.girvan_newman(G)
        try:
            top_level_communities = next(communities_generator)
            next_level_communities = next(communities_generator)
        except StopIteration:
            # If the graph is too small or can't produce multiple levels
            next_level_communities = [list(G.nodes())]

        communities = sorted(map(sorted, next_level_communities))
        print(f"[INFO] Number of Communities for model '{model}': {len(communities)}")

        # Assign colors to communities
        def colors_to_community(comms):
            colors = sns.color_palette("hls", len(comms)).as_hex()
            random.shuffle(colors)
            rows = []
            group_idx = 0
            for community in comms:
                color = colors.pop()
                group_idx += 1
                for node in community:
                    rows.append({"node": node, "color": color, "group": group_idx})
            return pd.DataFrame(rows)

        community_colors = colors_to_community(communities)

        # Attach color/group to each node in the graph
        for _, row in community_colors.iterrows():
            node = row["node"]
            if node in G.nodes():
                G.nodes[node]["group"] = row["group"]
                G.nodes[node]["color"] = row["color"]
                G.nodes[node]["size"] = G.degree[node]

        # ----------------------------------------
        # 7. Visualize with PyVis
        # ----------------------------------------
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

        # Write final HTML to docs/{session_id} subfolder
        graph_output_file = docs_directory / f"index.html"
        net.show(str(graph_output_file), notebook=False)
        print(f"[INFO] Saved interactive graph to {graph_output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process documents and generate graphs with session-specific outputs."
    )
    parser.add_argument(
        "session_id",
        type=str,
        help="The session ID to be used for input/output subdirectories."
    )
    args = parser.parse_args()

    main(args.session_id)
