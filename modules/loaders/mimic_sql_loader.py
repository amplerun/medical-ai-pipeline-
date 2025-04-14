# FILE: modules/loaders/mimic_sql_loader.py

import pandas as pd
from sqlalchemy import create_engine

def load_lab_data(db_url: str):
    engine = create_engine(db_url)
    query = """
        SELECT l.subject_id, l.charttime, l.itemid, l.valuenum
        FROM mimiciii.labevents l
        WHERE l.valuenum IS NOT NULL
    """
    df = pd.read_sql(query, engine)
    return df

if __name__ == "__main__":
    # Example DB URL: 'postgresql://user:password@localhost:5432/mimic'
    db_url = "postgresql://postgres:password@localhost:5432/mimic"
    labs = load_lab_data(db_url)
    print(labs.head())
