import os
import pandas as pd
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
import torch
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
import tqdm
import requests

PARQUET_DIR = "./FLAN_CLIP_checkpoints"

def download_dataset():
    pd.set_option('display.max_colwidth', None)
    pd.set_option('expand_frame_repr', True)

    dir_list = os.listdir(PARQUET_DIR)
    df_list = []
    for file in dir_list:
        if file.endswith(".parquet"):
            path = os.path.join(PARQUET_DIR, file)
            df_list.append(pd.read_parquet(path))

    if not df_list:
        print("No Parquet files found in directory.")
        return

    df = pd.concat(df_list, axis=0, ignore_index=True)
    df.reset_index(drop=True, inplace=True)

    print("\n--- Data Summary ---")
    print("Columns:", list(df.columns))
    print("Total rows:", len(df))

    print("\n--- Sample Preview ---")
    for i in range(1500, min(1600, len(df))):
        row = df.iloc[i]
        print(f"\nRow {i}:")
        print(f"  URL:            {row['image_url']}")
        print(f"  CLIP label:     {row['clip_label']} (confidence: {row['clip_confidence']:.4f})")
        print(f"  flan label:    {row['flan_label']} (confidence: {row['flan_confidence']:.4f})")


if __name__ == "__main__":
    download_dataset()
