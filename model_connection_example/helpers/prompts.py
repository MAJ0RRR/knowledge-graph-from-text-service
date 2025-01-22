import re
import sys
import time

import requests

sys.path.append("..")

import json

BIELIK_URL = "https://153.19.239.239/api/llm/prompt/chat"
auth = ("grupa6", "zsFaaIn1pXdtOhix")
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}

def extractConcepts(prompt: str, metadata={}, model="mistral-openorca:latest"):
    SYS_PROMPT = (
        "Your task is extract the key concepts (and non personal entities) mentioned in the given context. "
        "Extract only the most important and atomistic concepts, if  needed break the concepts down to the simpler concepts."
        "Categorize the concepts in one of the following categories: "
        "[event, concept, place, object, document, organisation, condition, misc]\n"
        "Format your output as a list of json with the following format:\n"
        "[\n"
        "   {\n"
        '       "entity": The Concept,\n'
        '       "importance": The concontextual importance of the concept on a scale of 1 to 5 (5 being the highest),\n'
        '       "category": The Type of Concept,\n'
        "   }, \n"
        "{ }, \n"
        "]\n"
    )
    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=prompt)
    try:
        result = json.loads(response)
        result = [dict(item, **metadata) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    return result


def graphPrompt2(input: str, metadata={}, model="mistral-openorca:latest"):
    if model == None:
        model = "mistral-openorca:latest"


    SYS_PROMPT = (
        "You are a network graph maker who extracts terms and their relations from a given context. "
        "You are provided with a context chunk (delimited by ```) Your task is to extract the ontology "
        "of terms mentioned in the given context. These terms should represent the key concepts as per the context. \n"
        "Thought 1: While traversing through each sentence, Think about the key terms mentioned in it.\n"
            "\tTerms may include object, entity, location, organization, person, \n"
            "\tcondition, acronym, documents, service, concept, etc.\n"
            "\tTerms should be as atomistic as possible\n\n"
        "Thought 2: Think about how these terms can have one on one relation with other terms.\n"
            "\tTerms that are mentioned in the same sentence or the same paragraph are typically related to each other.\n"
            "\tTerms can be related to many other terms\n\n"
        "Thought 3: Find out the relation between each such related pair of terms. \n\n"
        "Format your output as a list of json. Each element of the list contains a pair of terms"
        "and the relation between them, like the follwing: \n"
        "[\n"
        "   {\n"
        '       "node_1": "A concept from extracted ontology",\n'
        '       "node_2": "A related concept from extracted ontology",\n'
        '       "edge": "relationship between the two concepts, node_1 and node_2 in one or two sentences"\n'
        "   }, {...}\n"
        "]"
    )

    USER_PROMPT = f"context: ```{input}``` \n\n output: "
    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT)
    try:
        result = json.loads(response)
        result = [dict(item, **metadata) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    return result

def bielikGraphPrompt(input: str, metadata={},max_retries=3, retry_delay=5):
    SYS_PROMPT = (
        "Jesteś twórcą grafów sieciowych, który wyodrębnia terminy i ich relacje z podanego kontekstu. "
        "Otrzymujesz fragment kontekstu (oznaczony jako ```), Twoim zadaniem jest wyodrębnienie ontologii "
        "terminów wspomnianych w danym kontekście. Terminy te powinny reprezentować kluczowe pojęcia zgodnie z kontekstem.\n\n"
        "1. Analizując każde zdanie, zastanów się nad kluczowymi terminami w nim wspomnianymi.\n"
        "\t- Terminy mogą obejmować obiekt, byt, lokalizację, organizację, osobę, stan, akronim, dokumenty, usługę, pojęcie itd.\n"
        "\t- Terminy powinny być jak najbardziej atomistyczne.\n\n"
        "2. Zastanów się, jak te terminy mogą mieć relacje jeden na jeden z innymi terminami.\n"
        "\t- Terminy wspomniane w tym samym zdaniu lub w tym samym akapicie są zwykle ze sobą powiązane.\n"
        "\t- Terminy mogą być powiązane z wieloma innymi terminami.\n\n"
        "3. Znajdź relację między każdą taką powiązaną parą terminów.\n\n"
        "Sformatuj wynik jako listę JSON. Każdy element listy zawiera parę terminów oraz relację między nimi w formie:\n"
        "[\n"
        "   {\n"
        '       "node_1": "Termin z wyodrębnionej ontologii",\n'
        '       "node_2": "Powiązany termin z wyodrębnionej ontologii",\n'
        '       "edge": "Relacja między dwoma terminami, node_1 i node_2, w jednym lub dwóch zdaniach"\n'
        "   }\n"
        "]\n"
        "Upewnij się, że wynik jest poprawnym JSON-em. Jeśli wynik nie jest poprawnym JSON-em, zwróć komunikat o błędzie wskazujący problem."
    )

    USER_PROMPT = f"Kontekst: ```{input}``` \n\n Wynik: "

    data = {
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": USER_PROMPT}
        ],
        "max_length": 2000,
        "temperature": 0.7
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    attempt = 0
    result = None

    while attempt < max_retries and result is None:
        try:
            response = requests.put(BIELIK_URL, json=data, auth=auth, headers=headers, timeout=500, verify=False)
            response.raise_for_status()

            full_response = response.json()
            response_text = full_response.get('response', '')

            match = re.search(r'(\[.*?\s\])', response_text, re.DOTALL)
            if match:
                json_text = match.group(1)
                try:
                    ontologia = json.loads(json_text)
                    print("Znaleziono ontologię:", ontologia)
                    result = ontologia
                except json.JSONDecodeError as e:
                    print(f"Nie udało się sparsować JSON-a: {e}")
                    result = None
            else:
                print("Nie znaleziono sekcji JSON w odpowiedzi.")
                result = None

            if result is not None:
                result = [dict(item, **metadata) for item in result]

        except requests.RequestException as e:
            print(f"Błąd podczas połączenia z modelem Bielik: {e}")
            result = None
        except json.JSONDecodeError:
            print("Błąd dekodowania odpowiedzi na JSON.")
            result = None

        if result is None:
            attempt += 1
            print(f"Próba {attempt}/{max_retries} nieudana. Próba ponownie za {retry_delay} sek.")
            time.sleep(retry_delay)

    return result



def llamaGraphPrompt(input: str, metadata={}, max_retries=3, retry_delay=5,  model="custom-llama:latest"):

    USER_PROMPT = f"Kontekst: ```{input}``` \n\n Wynik: "
    data = {
        "model": model,
        "prompt": USER_PROMPT,
        "stream": False
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    api_url = "http://172.20.83.209:11434/api/generate"
    attempt = 0
    result = None

    while attempt < max_retries and result is None:
        try:
            # Wysyłanie zapytania do API
            response = requests.post(api_url, json=data, headers=headers, timeout=500)

            response.raise_for_status()

            full_response = response.json()
            response_text = full_response.get('response', '')

            match = re.search(r'(\[.*?\s\])', response_text, re.DOTALL)
            if match:
                json_text = match.group(1)
                try:
                    ontologia = json.loads(json_text)
                    print("Znaleziono ontologię:", ontologia)
                    result = ontologia
                except json.JSONDecodeError as e:
                    print(f"Nie udało się sparsować JSON-a: {e}")
                    result = None
            else:
                print("Nie znaleziono sekcji JSON w odpowiedzi.")
                result = None

            if result is not None:
                result = [dict(item, **metadata) for item in result]

        except requests.RequestException as e:
            print(f"Błąd podczas połączenia z modelem Bielik: {e}")
            result = None
        except json.JSONDecodeError:
            print("Błąd dekodowania odpowiedzi na JSON.")
            result = None

        if result is None:
            attempt += 1
            print(f"Próba {attempt}/{max_retries} nieudana. Próba ponownie za {retry_delay} sek.")
            time.sleep(retry_delay)

    return result