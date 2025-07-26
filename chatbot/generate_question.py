import os
import fitz
import pandas as pd
from pathlib import Path
from langdetect import detect
from nltk.tokenize import sent_tokenize
import nltk
from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

nltk.download('punkt')

# --- CONFIGURATION ---
PDF_DIR = Path("data/raw_pdf/")
OUTPUT_DIR = Path("data/outputs/")
MODEL_NAME = "mistral" #"llama3-chatqa"
CHUNK_CONFIGS = [(256, 64), (512, 128), (768, 128), (1024, 256)]  # (chunk_size, overlap)


def extract_text_from_pdf(pdf_path):
    text_pages = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if len(text) >= 30:
                text_pages.append(text)
    return "\n\n".join(text_pages)


def chunk_text(text, chunk_size=512, overlap=128):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        tokens = sent.split()
        if current_len + len(tokens) > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = tokens[-overlap:]
            current_len = len(current_chunk)
        else:
            current_chunk += tokens
            current_len += len(tokens)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def generate_questions_ollama(chunks, model_name="llama3-chatqa"):
    llm = Ollama(model=model_name)
    prompt = PromptTemplate(
        input_variables=["chunk", "lang_prompt"],
        template="""
{lang_prompt}

{chunk}

Donne UNE SEULE question claire et pertinente :
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    questions, langs = [], []
    for i, chunk in enumerate(chunks):
        lang = detect(chunk)
        langs.append(lang)
        lang_prompt = (
            "Basé sur ce texte en français, génère une QUESTION pertinente :"
            if lang.startswith("fr") else
            "Based on this English text, generate a relevant QUESTION:"
        )

        try:
            out = chain.run(chunk=chunk[:1000], lang_prompt=lang_prompt)
            questions.append(out.strip())
            print(f"✅ Q{i+1} ({lang}) : {out.strip()[:60]}...")
        except Exception as e:
            questions.append("")
            print(f"❌ Erreur Q{i+1}: {e}")

    return questions, langs


def process_all_pdfs_with_configs(pdf_dir, output_dir, chunk_configs):
    output_dir.mkdir(parents=True, exist_ok=True)

    for chunk_size, overlap in chunk_configs:
        print(f"\n🧪 Configuration : chunk_size={chunk_size}, overlap={overlap}")
        all_chunks, all_questions, all_langs = [], [], []

        for pdf_path in sorted(pdf_dir.glob("*.pdf")):
            print(f"\n📄 Traitement de : {pdf_path.name}")
            raw_text = extract_text_from_pdf(pdf_path)
            chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
            questions, langs = generate_questions_ollama(chunks, model_name=MODEL_NAME)

            all_chunks.extend(chunks)
            all_questions.extend(questions)
            all_langs.extend(langs)

        df = pd.DataFrame({
            "chunk": all_chunks,
            "question": all_questions,
            "lang": all_langs
        })

        csv_name = f"questions_cs{chunk_size}_ov{overlap}.csv"
        csv_path = output_dir / csv_name
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"💾 Exporté vers : {csv_path.absolute()}")


if __name__ == "__main__":
    if not PDF_DIR.exists():
        print(f"❌ Répertoire PDF introuvable : {PDF_DIR}")
    else:
        process_all_pdfs_with_configs(PDF_DIR, OUTPUT_DIR, CHUNK_CONFIGS)
