import uuid
import pandas as pd
import numpy as np
from .prompts import extractConcepts
from .prompts import bielikGraphPrompt, llamaGraphPrompt




def documents2Dataframe(documents) -> pd.DataFrame:
    rows = []
    for chunk in documents:
        row = {
            "text": chunk.page_content,
            **chunk.metadata,
            "chunk_id": uuid.uuid4().hex,
        }
        rows = rows + [row]

    df = pd.DataFrame(rows)
    return df


def df2ConceptsList(dataframe: pd.DataFrame) -> list:
    results = dataframe.apply(
        lambda row: extractConcepts(
            row.text, {"chunk_id": row.chunk_id, "type": "concept"}
        ),
        axis=1,
    )
    results = results.dropna()
    results = results.reset_index(drop=True)

    concept_list = np.concatenate(results).ravel().tolist()
    return concept_list


def concepts2Df(concepts_list) -> pd.DataFrame:
    concepts_dataframe = pd.DataFrame(concepts_list).replace(" ", np.nan)
    concepts_dataframe = concepts_dataframe.dropna(subset=["entity"])
    concepts_dataframe["entity"] = concepts_dataframe["entity"].apply(
        lambda x: x.lower()
    )

    return concepts_dataframe


# def df2Graph(dataframe: pd.DataFrame) -> list:
#     def process_row(row):
#         sentences = sent_tokenize(row.text)
#         sentence_results = [
#             graphPrompt(sentence, {"chunk_id": row.chunk_id})
#             for sentence in sentences
#         ]
#         merged_results = []
#         for result in sentence_results:
#             if result:
#                 merged_results.extend(result)
#
#         return merged_results
#
#     results = dataframe.apply(process_row, axis=1)
#     results = results.dropna()
#     results = results.reset_index(drop=True)
#
#     concept_list = np.concatenate(results).ravel().tolist()
#     return concept_list

def df2Graph(dataframe: pd.DataFrame, model: str) -> list:

    graph_prompt_functions = {
        "bielik": bielikGraphPrompt,
        "llama": llamaGraphPrompt,
    }

    # Wybór odpowiedniej funkcji na podstawie modelu
    graphPrompt = graph_prompt_functions.get(model)
    if not graphPrompt:
        raise ValueError(f"Unsupported model: {model}")

    # Przetwarzanie DataFrame przy użyciu wybranej funkcji
    results = dataframe.apply(
        lambda row: graphPrompt(row.text, {"chunk_id": row.chunk_id}), axis=1
    )
    results = results.dropna()
    results = results.reset_index(drop=True)

    concept_list = np.concatenate(results).ravel().tolist()
    return concept_list

def graph2Df(nodes_list) -> pd.DataFrame:
    graph_dataframe = pd.DataFrame(nodes_list).replace(" ", np.nan)
    graph_dataframe = graph_dataframe.dropna(subset=["node_1", "node_2"])
    graph_dataframe["node_1"] = graph_dataframe["node_1"].apply(lambda x: x.lower())
    graph_dataframe["node_2"] = graph_dataframe["node_2"].apply(lambda x: x.lower())

    return graph_dataframe
