import re
import json
import time
import socket
import pandas as pd
import pubchempy as pcp
from configs.config import PROCESSED_DIR

# Drug names in GDSC2 that are bare numeric PubChem CIDs with no independently
# verifiable structure; excluded rather than guessed.
NUMERIC_CID_EXCLUSIONS = [
    "123829", "765771", "123138", "50869", "720427", "667880",
    "729189", "741909", "743380", "150412", "615590", "630600", "776928",
]


def get_smiles_from_name(drug_name: str, max_network_retries=3):
    for attempt in range(max_network_retries):
        try:
            results = pcp.get_compounds(drug_name, "name")
            if len(results) == 0:
                return None
            return results[0].smiles
        except (socket.gaierror, ConnectionError, TimeoutError) as e:
            print(f"  Network issue on '{drug_name}' (attempt {attempt + 1}/{max_network_retries}): {e}")
            time.sleep(10)
        except Exception as e:
            print(f"Error fetching '{drug_name}': {e}")
            return None
    print(f"  [NETWORK FAILURE, gave up] {drug_name}")
    return None


def try_recover_drug(drug_name: str):
    smiles = get_smiles_from_name(drug_name)
    if smiles is not None:
        return smiles

    cleaned_name = re.sub(r"\(.*?\)", "", drug_name).strip()
    cleaned_name = re.sub(r"\d+\s*u?M$", "", cleaned_name).strip()
    if cleaned_name != drug_name and cleaned_name != "":
        smiles = get_smiles_from_name(cleaned_name)
        if smiles is not None:
            return smiles

    if drug_name.strip().isdigit():
        try:
            results = pcp.get_compounds(int(drug_name.strip()), "cid")
            if len(results) > 0:
                return results[0].smiles
        except Exception:
            pass

    return None


def parse_synonyms(raw_value):
    if pd.isna(raw_value):
        return []
    parts = [p.strip() for p in str(raw_value).split(",")]
    return [p for p in parts if p != ""]


def build_synonym_lookup(screened_compounds):
    return (
        screened_compounds
        .drop_duplicates(subset="DRUG_ID")
        .set_index("DRUG_ID")["SYNONYMS"]
        .apply(parse_synonyms)
        .to_dict()
    )


def resolve_all_drug_smiles(final_dataset, screened_compounds):
    unique_drugs_df = final_dataset[["DRUG_ID", "DRUG_NAME"]].drop_duplicates(subset="DRUG_ID")
    drug_id_to_synonyms = build_synonym_lookup(screened_compounds)
    drug_name_to_id = dict(zip(unique_drugs_df["DRUG_NAME"], unique_drugs_df["DRUG_ID"]))

    drug_to_smiles = {}
    for _, row in unique_drugs_df.iterrows():
        drug_name = row["DRUG_NAME"]
        drug_to_smiles[drug_name] = get_smiles_from_name(drug_name)
        time.sleep(2)

    failed_drugs = [name for name, smiles in drug_to_smiles.items() if smiles is None]

    for drug_name in failed_drugs:
        drug_id = drug_name_to_id.get(drug_name)
        synonyms = drug_id_to_synonyms.get(drug_id, [])

        recovered_smiles = None
        for synonym in synonyms:
            recovered_smiles = get_smiles_from_name(synonym)
            time.sleep(2)
            if recovered_smiles is not None:
                break

        if recovered_smiles is None:
            recovered_smiles = try_recover_drug(drug_name)
            time.sleep(2)

        drug_to_smiles[drug_name] = recovered_smiles

    still_failed = [name for name, smiles in drug_to_smiles.items() if smiles is None]
    for drug_name in still_failed:
        time.sleep(5)
        smiles = get_smiles_from_name(drug_name)
        if smiles is not None:
            drug_to_smiles[drug_name] = smiles

    for name in NUMERIC_CID_EXCLUSIONS:
        drug_to_smiles[name] = None

    return drug_to_smiles


def save_smiles_cache(drug_to_smiles):
    with open(f"{PROCESSED_DIR}/drug_to_smiles.json", "w") as f:
        json.dump(drug_to_smiles, f, indent=2)


def load_smiles_cache():
    with open(f"{PROCESSED_DIR}/drug_to_smiles.json", "r") as f:
        return json.load(f)


def filter_dataset_to_resolved_drugs(final_dataset, drug_to_smiles):
    final_dataset = final_dataset.copy()
    final_dataset["DRUG_NAME"] = final_dataset["DRUG_NAME"].astype(str)
    valid_drugs = {name for name, smiles in drug_to_smiles.items() if smiles is not None}
    return final_dataset[final_dataset["DRUG_NAME"].isin(valid_drugs)]
