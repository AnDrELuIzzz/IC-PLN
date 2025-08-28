from sklearn.metrics import precision_recall_fscore_support

def calculate_ner_metrics(gold_entities_list, pred_entities_list):
    y_true, y_pred = [], []

    for gold_ents, pred_ents in zip(gold_entities_list, pred_entities_list):
        # Comparação para métricas
        for ent in gold_ents:
            y_true.append(ent[1])
            y_pred.append(ent[1] if ent in pred_ents else "O")
        for ent in pred_ents:
            if ent not in gold_ents:
                y_true.append("O")
                y_pred.append(ent[1])

    # Cálculo das métricas globais
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", labels=["PER", "LOC", "ORG", "MISC"]
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


