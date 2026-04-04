import pandas as pd
from sqlalchemy import create_engine

def load_data(file_path, table_name):
    
    engine = create_engine('postgresql://postgres:postgres123@127.0.0.1:5432/marketing_campaign')
    df = pd.read_csv(file_path)
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    print(f"Data loaded successfully into {table_name} table.")

load_data('data/ifood_df.csv', 'marketing_campaign')
