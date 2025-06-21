import gradio as gr
from transformers import pipeline

MODEL_REPO = "sivanimohan/my-bio-bert-qa-model"  # Change to your model if different

qa_pipeline = pipeline("question-answering", model=MODEL_REPO, tokenizer=MODEL_REPO)

def answer_question(context, question):
    if not context.strip() or not question.strip():
        return "Please provide both context and a question."
    result = qa_pipeline(question=question, context=context)
    return result["answer"]

iface = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Textbox(label="Context", lines=8, placeholder="Paste biomedical text here..."),
        gr.Textbox(label="Question", lines=2, placeholder="Ask your question...")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="BioBERT Medical QA Chatbot",
    description="Paste biomedical content and ask a question. Powered by your fine-tuned BioBERT model."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)