def calculate_conllu_metrics(results):
    """
    Calcula métricas de comparação entre os dados gold e os preditos para CONLL-U.
    Detalhes técnicos:
    - Acurácia de POS é calculada comparando elemento a elemento dos tokens.
    - UAS e LAS são derivados da verificação de acertos na estrutura de dependências.
    - Precision, Recall e F1 em tokenização são derivados dos spans sobrepostos.
    """
    total_tokens = 0
    pos_correct = 0
    uas_correct = 0
    las_correct = 0
    lemma_correct = 0
    tp_token, fp_token, fn_token = 0, 0, 0
    for sent in results:
        n = len(sent["gold_tokens"])
        total_tokens += n
        pos_correct += sum(1 for g, p in zip(sent["gold_pos"], sent["pred_pos"]) if g == p)
        lemma_correct += sum(1 for g, p in zip(sent["gold_lemmas"], sent["pred_lemmas"]) if g == p)
        for g_head, p_head, g_deprel, p_deprel in zip(
            sent["gold_heads"], sent["pred_heads"], sent["gold_deprels"], sent["pred_deprels"]
        ):
            if g_head == p_head:
                uas_correct += 1
                if g_deprel == p_deprel:
                    las_correct += 1
        gold_spans = set(sent["gold_spans"])
        pred_spans = set(sent["pred_spans"])
        tp = len(gold_spans & pred_spans)
        fp = len(pred_spans - gold_spans)
        fn = len(gold_spans - pred_spans)
        tp_token += tp
        fp_token += fp
        fn_token += fn
    precision = tp_token / (tp_token + fp_token) if (tp_token + fp_token) > 0 else 0
    recall = tp_token / (tp_token + fn_token) if (tp_token + fn_token) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {
        "pos_accuracy": pos_correct / total_tokens,
        "lemma_accuracy": lemma_correct / total_tokens,
        "uas": uas_correct / total_tokens,
        "las": las_correct / total_tokens,
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
    }

def analyze_conllu_errors(results):
    """
    Gera um relatório detalhado dos erros cometidos pelo modelo em POS tagging, lematização e análise de dependências.
    Inclui informações sobre a sentença, o token específico e sua posição.
    """
    error_analysis = {
        "pos_errors": [],
        "lemma_errors": [],
        "dependency_errors": [],
    }

    for sent in results:
        sent_id = sent.get("sent_id", "N/A")
        text = sent["text"]
        for idx, (g_pos, p_pos, g_lemma, p_lemma, g_head, p_head, g_deprel, p_deprel) in enumerate(zip(
            sent["gold_pos"], sent["pred_pos"],
            sent["gold_lemmas"], sent["pred_lemmas"],
            sent["gold_heads"], sent["pred_heads"],
            sent["gold_deprels"], sent["pred_deprels"]
        )):
            token = sent["gold_tokens"][idx]
            if g_pos != p_pos:
                error_analysis["pos_errors"].append({
                    "sent_id": sent_id,
                    "text": text,
                    "token": token,
                    "position": idx + 1,
                    "gold": g_pos,
                    "predicted": p_pos,
                })
            if g_lemma != p_lemma:
                error_analysis["lemma_errors"].append({
                    "sent_id": sent_id,
                    "text": text,
                    "token": token,
                    "position": idx + 1,
                    "gold": g_lemma,
                    "predicted": p_lemma,
                })
            if g_head != p_head or g_deprel != p_deprel:
                error_analysis["dependency_errors"].append({
                    "sent_id": sent_id,
                    "text": text,
                    "token": token,
                    "position": idx + 1,
                    "gold": (g_head, g_deprel),
                    "predicted": (p_head, p_deprel),
                })
    return error_analysis


