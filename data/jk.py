import json

def generate_qa_examples(docs, out_path):
    squad_data = []
    for doc_id, doc in docs.items():
        context = doc['title'] + ' ' + doc['abstract']
        title = doc['title']
        mesh_to_text = {e['mesh']: e['text'] for e in doc['entities']}
        qas = []
        for chem_mesh, dis_mesh in doc['relations']:
            if chem_mesh in mesh_to_text and dis_mesh in mesh_to_text:
                chem_text = mesh_to_text[chem_mesh]
                dis_text = mesh_to_text[dis_mesh]
                # Q: diseases for chemical
                ans_start = context.find(dis_text)
                if ans_start != -1:
                    qas.append({
                        "id": f"{doc_id}_chem_{chem_mesh}",
                        "question": f"What diseases are associated with {chem_text}?",
                        "answers": [{"text": dis_text, "answer_start": ans_start}]
                    })
                # Q: chemicals for disease
                ans_start = context.find(chem_text)
                if ans_start != -1:
                    qas.append({
                        "id": f"{doc_id}_dis_{dis_mesh}",
                        "question": f"What chemicals are associated with {dis_text}?",
                        "answers": [{"text": chem_text, "answer_start": ans_start}]
                    })
        if qas:
            squad_data.append({
                "title": title,
                "paragraphs": [{
                    "context": context,
                    "qas": qas
                }]
            })
    with open("/workspaces/bio-bert-medical-chatbot/data/cdr_qa_train.json", "w") as f:
        json.dump({"data": squad_data}, f, indent=2)