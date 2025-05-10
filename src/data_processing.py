import pandas as pd
from sklearn.model_selection import train_test_split
import re
import pytorch_lightning as pl
import zipfile
import os

pl.seed_everything(42)

def get_target_text(row):
    try:
        if isinstance(row["answers"], list) and 0 <= row["correct_answer_id"] < len(row["answers"]):
            return row["answers"][row["correct_answer_id"]]
    except:
        pass
    return None
def fix_answer_string(answer_str):
    if not isinstance(answer_str, str):
        return []
    cleaned = re.sub(r"'\s+'", "', '", answer_str)
    try:
        return eval(cleaned)
    except:
        return []
def reformat_df_cleaned1(df_cleaned):
    df_reformatted = df_cleaned.copy()
    df_reformatted['parsed_answer'] = df_reformatted['answers'].apply(
        lambda x: x if isinstance(x, str) else "not enough information"
    )
    df_reformatted['prompt'] = (
        "### Context:\n" + df_reformatted['context'] + "\n\n" +
        "### Question:\n" + df_reformatted['question'] + "\n\n" +
        "### Instruction:\nProvide the answer to the question based on the context.\n\n" +
        "### Answer:\n" + df_reformatted['parsed_answer']
    )
    df_reformatted = df_reformatted.drop(columns=['parsed_answer'])
    return df_reformatted

# QA Preprocessing
df = pd.read_csv("data//raw//train.csv")
df_cleaned = df[["context", "question", "answers", "correct_answer_id"]]

df_cleaned = df_cleaned[df_cleaned['question'].str.endswith('?', na=False)]
df_cleaned = df_cleaned.reset_index(drop=True)
df_cleaned["answers"] = df_cleaned["answers"].apply(fix_answer_string)
df_cleaned["answers"] = df_cleaned.apply(get_target_text, axis=1)

train_df, val_df = train_test_split(df_cleaned, test_size=0.2, random_state=42)
train_df_qa_answer = reformat_df_cleaned1(train_df)
val_df_qa_answer = reformat_df_cleaned1(val_df)

train_df_qa_answer.to_csv("data//processed//train_qa_answer.csv", index=False)
val_df_qa_answer.to_csv("data//processed//val_qa_answer.csv", index=False)

def reformat_df_cleaned(df_cleaned, task):
    """Reformat QA data for specified task (qa_generation or qa_answering)."""
    df_reformatted = df_cleaned.copy()
    df_reformatted['parsed_answer'] = df_reformatted['answers'].apply(
        lambda x: x.strip("[]").split("'")[1] if x else "not enough information"
    )
    
    if task == "qa_generation":
        df_reformatted['prompt'] = (
            "### Context:\n" + df_reformatted['context'] + "\n\n" +
            "### Instruction:\nGenerate a question and its answer based on the context.\n\n" +
            "### Output:\n*Question*: " + df_reformatted['question'] + "\n" +
            "*Answer*: " + df_reformatted['parsed_answer']
        )
    elif task == "qa_answering":
        df_reformatted['prompt'] = (
            "### Contents:\n" + df_reformatted['context'] + "\n\n" +
            "### Question:\n" + df_reformatted['question'] + "\n\n" +
            "### Instruction:\nProvide the answer to the question based on the context.\n\n" +
            "### Answer:\n" + df_reformatted['parsed_answer']
        )
    else:
        raise ValueError("Task must be 'qa_generation' or 'qa_answering'")
    
    df_reformatted = df_reformatted.drop(columns=['parsed_answer'])
    return df_reformatted

train_df, val_df = train_test_split(df_cleaned, test_size=0.2, random_state=42)
train_df_qa_gen = reformat_df_cleaned(train_df, task="qa_generation")
val_df_qa_gen = reformat_df_cleaned(val_df, task="qa_generation")

train_df_qa_gen.to_csv("data//processed//train_qa_gen.csv", index=False)
val_df_qa_gen.to_csv("data//processed//val_qa_gen.csv", index=False)

# Summarization Preprocessing
zip_path = "data//raw//news_summary.zip" 
extract_dir = "data//raw//"
if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

df_summary = pd.read_csv("data//raw//news_summary.csv", encoding="latin-1")
df_summary = df_summary[["text", "ctext"]]
df_summary.columns = ["summary", "text"]
df_summary = df_summary.dropna()

train_summary_df, test_summary_df = train_test_split(
    df_summary, test_size=0.1, random_state=42
)

train_summary_df.to_csv("data//processed//train_summary.csv", index=False)
test_summary_df.to_csv("data//processed//test_summary.csv", index=False)