import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress tokenizer parallelism warning

import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer

# 1. Load the cleaned SQuAD-style dataset
with open("/workspaces/bio-bert-medical-chatbot/data/chatdoctor_flat_squad_cleaned_ultrastrict.json") as f:
    data = json.load(f)

# 2. Convert to HuggingFace Dataset
dataset = Dataset.from_list(data)

# 3. Load tokenizer and model
model_name = "sivanimohan/my-bio-bert-qa-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# 4. Preprocessing function with label extraction
def preprocess_examples(examples):
    # Use batch processing for speed
    questions = examples["question"]
    contexts = examples["context"]
    answers = examples["answers"]

    # Tokenize with offset mapping
    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=384,
        padding="max_length",
        return_offsets_mapping=True,
    )

    start_positions = []
    end_positions = []

    for i in range(len(questions)):
        offsets = tokenized["offset_mapping"][i]
        answer = answers[i]
        answer_start = answer["answer_start"][0]
        answer_text = answer["text"][0]
        answer_end = answer_start + len(answer_text)

        # Find start and end token indices in the context
        sequence_ids = tokenized.sequence_ids(i)
        # context tokens have sequence_id == 1
        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        # Default positions (in case answer is not found)
        start_pos = context_start
        end_pos = context_start

        # Loop to find the tokens where the answer starts and ends
        for idx in range(context_start, context_end + 1):
            start_char, end_char = offsets[idx]
            if start_char <= answer_start < end_char:
                start_pos = idx
            if start_char < answer_end <= end_char:
                end_pos = idx
                break
        start_positions.append(start_pos)
        end_positions.append(end_pos)

    # Remove offset mapping as it's not needed anymore
    tokenized.pop("offset_mapping")
    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized

# 5. Tokenize the dataset, adding start/end positions
tokenized_dataset = dataset.map(preprocess_examples, batched=True, remove_columns=dataset.column_names)

# 6. Train/test split
split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

# 7. Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=3e-5,
    per_device_train_batch_size=4,   # Lower batch size if OOM
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=20,
    save_total_limit=1,
    save_strategy="epoch",
    fp16=True,  # Remove if your hardware does not support fp16
    dataloader_num_workers=2,
    report_to=[],
)

# 8. Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# 9. Train the model
trainer.train()

# 10. Save final model
trainer.save_model("./trained_biobert_qa_model")