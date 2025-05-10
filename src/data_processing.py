import pandas as pd
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import zipfile
import os

pl.seed_everything(42)

# QA Preprocessing
df = pd.read_csv("data//raw//train.csv")
df_cleaned = df[["context", "question", "answers", "correct_answer_id"]]
df_cleaned = df_cleaned[df_cleaned['question'].str.endswith('?', na=False)]
df_cleaned = df_cleaned.reset_index(drop=True)

df_task2 = df_cleaned.copy()
df_task2["input_text"] = (
    "answer question: " + df_cleaned["question"] + " context: " + df_cleaned["context"]
)
df_task2["target_text"] = df_cleaned.apply(
    lambda row: row["answers"][row["correct_answer_id"]], axis=1
)

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
train_df_qa_answer = reformat_df_cleaned(train_df, task="qa_answering")
val_df_qa_answer = reformat_df_cleaned(val_df, task="qa_answering")

train_df_qa_gen.to_csv("data//processed//train_qa_gen.csv", index=False)
val_df_qa_gen.to_csv("data//processed//val_qa_gen.csv", index=False)
train_df_qa_answer.to_csv("data//processed//train_qa_answer.csv", index=False)
val_df_qa_answer.to_csv("data//processed//val_qa_answer.csv", index=False)
df_task2.to_csv("data//processed//task2.csv", index=False)

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