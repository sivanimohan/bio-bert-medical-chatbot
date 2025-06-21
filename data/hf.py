import json

def flatten_chatdoctor_json(json_path, out_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    flat = []
    for entry in data:
        for para in entry["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                # SQuAD expects answers as a dict with 'text' and 'answer_start', both as lists
                answer = qa["answers"][0]
                flat.append({
                    "id": qa["id"],
                    "question": qa["question"],
                    "context": context,
                    "answers": {
                        "text": [answer["text"]],
                        "answer_start": [answer["answer_start"]]
                    }
                })

    with open(out_path, "w") as f:
        json.dump(flat, f, indent=2)

if __name__ == "__main__":
    flatten_chatdoctor_json("your_input.json", "chatdoctor_flat_squad.json")