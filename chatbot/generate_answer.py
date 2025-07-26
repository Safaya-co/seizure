import pandas as pd
from langdetect import detect
from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from tqdm import tqdm

def generate_double_answers_ollama(df, model_name="mistral"):
    """
    Génère deux types de réponses à chaque question :
    - Une réponse normale (complète)
    - Une réponse courte et précise
    """
    llm = Ollama(model=model_name)

    # Prompt pour réponse normale
    prompt_full = PromptTemplate(
        input_variables=["question", "chunk", "lang_prompt"],
        template="""
{lang_prompt}

Texte :
{chunk}

Question :
{question}

Fournis une réponse complète et pertinente :
"""
    )

    # Prompt pour réponse courte
    prompt_short = PromptTemplate(
        input_variables=["question", "chunk", "lang_prompt"],
        template="""
{lang_prompt}

Texte :
{chunk}

Question :
{question}

Donne une réponse courte et précise (moins de 30 mots) :
"""
    )

    chain_full = LLMChain(llm=llm, prompt=prompt_full)
    chain_short = LLMChain(llm=llm, prompt=prompt_short)

    full_answers, short_answers = [], []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="🧠 Génération des réponses"):
        question = str(row["question"]).strip()
        chunk = str(row["chunk"]).strip()
        lang = row.get("lang", "en")

        lang_prompt = (
            "Tu dois répondre à une question en français basée sur le texte suivant :"
            if lang.startswith("fr") else
            "You must answer a question in English based on the following text:"
        )

        try:
            full_ans = chain_full.run(question=question, chunk=chunk[:1500], lang_prompt=lang_prompt).strip()
        except Exception as e:
            print(f"❌ Erreur (full) Q{i}: {e}")
            full_ans = ""

        try:
            short_ans = chain_short.run(question=question, chunk=chunk[:1500], lang_prompt=lang_prompt).strip()
        except Exception as e:
            print(f"❌ Erreur (short) Q{i}: {e}")
            short_ans = ""

        full_answers.append(full_ans)
        short_answers.append(short_ans)

    df["generated_answer"] = full_answers
    df["short_answer"] = short_answers

    return df

if __name__ == "__main__":
    df = pd.read_csv("data/outputs/questions_cs1024_ov256.csv")

    # Générer les réponses avec Ollama local (modèle mistral)
    df_with_answers = generate_double_answers_ollama(df, model_name="mistral")

    # Exporter
    df_with_answers.to_csv("data/outputs/answers_cs1024_ov256_mistral.csv", index=False, encoding="utf-8")
    print("✅ Réponses générées et sauvegardées.")