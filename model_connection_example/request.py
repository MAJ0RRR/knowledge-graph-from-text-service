import pandas as pd
import numpy as np
import uuid
import json
import requests
from requests.exceptions import ChunkedEncodingError
from transformers import pipeline

# Funkcja do wysyłania zapytań do Bielika
BIELIK_URL = "https://153.19.239.239/api/llm/prompt/chat"
auth = ("grupa6", "zsFaaIn1pXdtOhix")  # Podstaw odpowiednie dane
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}

# System prompt for Bielik (in Polish)
system_prompt = (
    'Twoim zadaniem jest wyodrębnienie kluczowych bytów wspomnianych w tekście użytkownika.\n'
    'Byty mogą obejmować - wydarzenie, koncepcję, osobę, miejsce, obiekt, dokument, organizację, artefakt, inne, itp.\n'
    'Sformatuj swoją odpowiedź jako listę JSON o następującej strukturze, bez dodaktkowych słów:\n'
    '[{"byt": Ciąg znaków reprezentujący byt, "waznosc": Jak ważny jest byt w danym kontekście w skali od 1 do 5, gdzie 5 jest najwyższą wartością, "typ": Typ bytu}, { }]'
)


# Function to send queries to Bielik
def send_to_bielik(prompt, max_length=128, temperature=0.7, max_retries=3):
    data = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_length": max_length,
        "temperature": temperature
    }

    for attempt in range(max_retries):
        try:
            with requests.put(BIELIK_URL, json=data, headers=headers, auth=auth, verify=False, stream=True) as response:
                response.raise_for_status()
                full_response = ""
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        full_response += chunk.decode('utf-8')

            # Try to parse the full response as JSON
            try:
                return json.loads(full_response)
            except json.JSONDecodeError:
                # If it's not valid JSON, return the raw string
                return full_response

        except (ChunkedEncodingError, ConnectionError) as e:
            if attempt < max_retries - 1:
                print(f"Connection error occurred. Retrying... (Attempt {attempt + 1}/{max_retries})")
            else:
                print(f"Failed to get a complete response after {max_retries} attempts.")
                raise

    return None  # This line should never be reached due to the raise in the loop

# Function to extract concepts using Bielik
def extract_concepts_from_text(text):
    try:
        response = send_to_bielik(prompt=text)
        # Remove all newline characters from the input text
        cleaned_text = response
        return cleaned_text

    except Exception as e:
        print(f"Error: {str(e)}")
        print("Raw response:", response)
        return []


# Polish NER model (as specified)
ner = pipeline("token-classification", model="dkleczek/bert-base-polish-cased-v1", aggregation_strategy="simple")
print("Number of parameters ->", ner.model.num_parameters() / 1000000, "Mn")


def row2NamedEntities(row):
    ner_results = ner(row['text'])
    metadata = {'chunk_id': row['chunk_id']}
    entities = [{'name': result['word'], 'entity': result['entity_group'], **metadata} for result in ner_results]
    return entities


def dfText2DfNE(dataframe):
    results = dataframe.apply(row2NamedEntities, axis=1)
    entities_list = np.concatenate(results).ravel().tolist()
    entities_dataframe = pd.DataFrame(entities_list).replace(' ', np.nan)
    entities_dataframe = entities_dataframe.dropna(subset=['entity'])
    entities_dataframe = entities_dataframe.groupby(['name', 'entity', 'chunk_id']).size().reset_index(name='count')
    return entities_dataframe


# Example Polish text (you can replace this with your PDF content)
polish_text = """
W Warszawie powstała konstytucja RP.
"""

# Create a dataframe
rows = [{'text': polish_text, 'chunk_id': uuid.uuid4().hex}]
df = pd.DataFrame(rows)

# Extract named entities
dfne = dfText2DfNE(df)

# Aggregate results
df_ne = dfne.groupby(['name', 'entity']).agg({'count': 'sum', 'chunk_id': ','.join}).reset_index()
print("Named Entities (NER):")
print(df_ne.sort_values(by='count', ascending=False).head(10).to_string(index=False))

# Extract concepts using Bielik
concepts = extract_concepts_from_text(polish_text)
def extract_data(data):
    # Extract the JSON part from the response
    json_str = data["response"]
    json_end = json_str.find(']') + 1  # Find the end of the JSON array

    if json_end > 0:
        valid_json = json_str[:json_end]
        try:
            # Parse the valid JSON part
            parsed_list = json.loads(valid_json)

            # Create a new dictionary with the parsed list
            result_dict = {"entities": parsed_list}

            print("Resulting dictionary:")
            print(json.dumps(result_dict, indent=2, ensure_ascii=False))
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
    else:
        print("No valid JSON found in the response")
print("\nKey Concepts (Bielik):")
print(concepts)
if concepts:
    print(extract_data(concepts))
else:
    print("No concepts extracted or an error occurred.")
