import stanza
import os
from sklearn.metrics import precision_recall_fscore_support
from tabulate import tabulate

def read_wikiner(path, max_sentences=None):
    """Lê o corpus WikiNER e retorna uma lista de sentenças."""
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
    """Extrai entidades 'gold' a partir de tokens e tags no formato IOB."""
    entities = []
    current_entity, current_label = [], None
    for token, tag in zip(tokens, tags):
        if tag == "O":
            if current_entity:
                entities.append((" ".join(current_entity), current_label))
                current_entity, current_label = [], None
        else:
            label = tag.split("-")[-1]
            if tag.startswith("B-") or current_label != label:
                if current_entity:
                    entities.append((" ".join(current_entity), current_label))
                current_entity, current_label = [token], label
            elif tag.startswith("I-") and current_label == label:
                current_entity.append(token)
    if current_entity:
        entities.append((" ".join(current_entity), current_label))
    return entities

def evaluate_model_stanza(nlp, sentences, output_path, max_sent=50):
    """
    Avalia o modelo Stanza para NER, compara com as entidades 'gold' e salva um relatório.
    """
    y_true, y_pred = [], []
    relatorio = []

    # Mapeamento de tags do Stanza para as do WikiNER (se necessário)
    # Stanza usa 'PER', 'LOC', 'ORG', 'MISC'
    tag_mapping = {
        "PER": "PER",
        "LOC": "LOC",
        "ORG": "ORG",
        "MISC": "MISC"
    }

    for i, (tokens, tags) in enumerate(sentences[:max_sent], 1):
        gold_ents = get_gold_entities(tokens, tags)
        text = " ".join(tokens)
        
        # Processa o texto com o pipeline do Stanza
        doc = nlp(text)
        
        # Extrai as entidades previstas pelo Stanza
        pred_ents = []
        for ent in doc.ents:
            if ent.type in tag_mapping:
                pred_ents.append((ent.text, tag_mapping[ent.type]))

        # Adiciona dados para o cálculo das métricas globais
        all_ents = set(gold_ents + pred_ents)
        for ent_text, ent_label in all_ents:
            is_in_gold = (ent_text, ent_label) in gold_ents
            is_in_pred = (ent_text, ent_label) in pred_ents

            if is_in_gold:
                y_true.append(ent_label)
                y_pred.append(ent_label if is_in_pred else "O")
            elif is_in_pred:
                y_true.append("O")
                y_pred.append(ent_label)

        # --- Relatório por sentença ---
        relatorio.append(f"=== Sentença {i} ===")
        relatorio.append(f"Texto: {text}\n")

        relatorio.append("Gold Entities vs. Pred Entities:")
        gold_text = [f"{ent[0]} ({ent[1]})" for ent in gold_ents]
        pred_text = [f"{ent[0]} ({ent[1]})" for ent in pred_ents]
        relatorio.append(f"Gold: {gold_text}")
        relatorio.append(f"Pred: {pred_text}\n")

    # Cálculo das métricas globais
    labels = ["PER", "LOC", "ORG", "MISC"]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", labels=labels, zero_division=0
    )

    metricas = [
        ["Precisão (NER)", f"{precision:.2%}"],
        ["Recall (NER)", f"{recall:.2%}"],
        ["F1-Score (NER)", f"{f1:.2%}"],
    ]

    # Cria a pasta de saída se não existir
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Escreve o relatório no arquivo
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== Métricas NER com Stanza ===\n")
        f.write(tabulate(metricas, headers=["Métrica", "Valor"], tablefmt="grid"))
        f.write("\n\n")
        f.write("\n".join(relatorio))

    print(f"Relatório da análise com Stanza salvo em: {output_path}")

# ==== Bloco Principal de Execução ====
if __name__ == "__main__":
    # Caminho para o arquivo WikiNER (ajuste se necessário)
    wikiner_path = "/home/andre/Dev-Ubuntu/IC/Corupus/5462500/aij-wikiner-pt-wp3"
    
    # Verifica se o arquivo de dados existe
    if not os.path.exists(wikiner_path):
        print(f"Erro: O arquivo de dados não foi encontrado em '{wikiner_path}'")
        print("Por favor, ajuste o caminho para o arquivo 'aij-wikiner-pt-wp3'.")
    else:
        # Carrega as sentenças do corpus
        sentences = read_wikiner(wikiner_path, max_sentences=10000)

        # Inicializa o pipeline do Stanza para português com os processadores de tokenização e NER
        print("Inicializando o pipeline do Stanza para português (pode levar um tempo)...")
        stanza.download('pt', processors='tokenize,ner')
        nlp_stanza = stanza.Pipeline('pt', processors='tokenize,ner')
        print("Pipeline do Stanza carregado com sucesso.")

        # Define o caminho para o arquivo de saída
        output_file_path = "analise_stanza/resultado_NER.txt"

        # Executa a avaliação e gera o relatório
        evaluate_model_stanza(nlp_stanza, sentences, output_file_path, max_sent=50)