import fitz  
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

def extract_text_from_pdfs(pdf_dir):
    text_data = []
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            with fitz.open(pdf_path) as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
            text_data.append(text)
    return text_data

def summarize_texts(texts):
    summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
    summaries = [summarizer(text, max_length=150, min_length=60, do_sample=False)[0]['summary_text'] for text in texts]
    return summaries

def generate_question(context):
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    input_text = "generate a question: " + context + " </s>"
    inputs = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question

pdf_dir = 'pdfs/'
texts = extract_text_from_pdfs(pdf_dir)
summaries = summarize_texts(texts)
questions = [generate_question(summary) for summary in summaries]

for i, question in enumerate(questions):
    print(f"Question {i+1}: {question}")
