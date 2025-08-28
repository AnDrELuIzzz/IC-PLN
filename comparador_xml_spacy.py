import spacy
import os
from sklearn.metrics import precision_recall_fscore_support
from tabulate import tabulate

def read_wikiner(path, max_sentences=None):
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            tokens, tags = [], []
            for tok in line.strip().split():
                try:
                    word, pos, ner = tok.split("|")
                except ValueError:
                    continue
                tokens.append(word)
                tags.append(ner)
            if tokens:
                sentences.append((tokens, tags))
            if max_sentences and i >= max_sentences:
                break
    return sentences

def get_gold_entities(tokens, tags):
    entities = []
    current_entity, current_label = [], None
    for token, tag in zip(tokens, tags):
        if tag == "O":
            if current_entity:
                entities.append((" ".join(current_entity), current_label))
                current_entity, current_label = [], None
        else:
            label = tag.split("-")[-1]
            if current_label == label:
                current_entity.append(token)
            else:
                if current_entity:
                    entities.append((" ".join(current_entity), current_label))
                current_entity, current_label = [token], label
    if current_entity:
        entities.append((" ".join(current_entity), current_label))
    return entities

def evaluate_model(nlp, sentences, output_path, max_sent=50):
    y_true, y_pred = [], []
    relatorio = []

    for i, (tokens, tags) in enumerate(sentences[:max_sent], 1):
        gold_ents = get_gold_entities(tokens, tags)
        text = " ".join(tokens)
        doc = nlp(text)
        pred_ents = [(ent.text, ent.label_) for ent in doc.ents]

        # Comparação para métricas
        for ent in gold_ents:
            y_true.append(ent[1])
            y_pred.append(ent[1] if ent in pred_ents else "O")
        for ent in pred_ents:
            if ent not in gold_ents:
                y_true.append("O")
                y_pred.append(ent[1])

        # --- Relatório por sentença ---
        relatorio.append(f"=== Sentença {i} ===")
        relatorio.append(f"Texto: {text}\n")

        relatorio.append("Gold Entities vs. Pred Entities:")
        gold_text = [f"{ent[0]} ({ent[1]})" for ent in gold_ents]
        pred_text = [f"{ent[0]} ({ent[1]})" for ent in pred_ents]
        relatorio.append(f"Gold: {gold_text}")
        relatorio.append(f"Pred: {pred_text}\n")

    # Cálculo das métricas globais
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", labels=["PER", "LOC", "ORG", "MISC"]
    )

    metricas = [
        ["Precisão (NER)", f"{precision:.2%}"],
        ["Recall (NER)", f"{recall:.2%}"],
        ["F1 (NER)", f"{f1:.2%}"],
    ]

    # Criar pasta caso não exista
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Escrever no arquivo
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== Métricas NER ===\n")
        f.write(tabulate(metricas, headers=["Métrica", "Valor"], tablefmt="grid"))
        f.write("\n\n")
        f.write("\n".join(relatorio))

    print(f"Relatório salvo em: {output_path}")

# ==== Uso ====
sentences = read_wikiner(
    "/home/anrdre/IC-NLP/5462500/aij-wikiner-pt-wp3",
    max_sentences=10000
)
nlp = spacy.load("pt_core_news_lg")  # ou seu modelo treinado
evaluate_model(nlp, sentences, "analise_spacy/resultado_NER.txt", max_sent=50)
