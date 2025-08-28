from tabulate import tabulate

def save_conllu_results_to_file(results, metrics, output_file, max_sentences=None):
    metric_rows = [
        ["Acurácia de POS", f"{metrics['pos_accuracy']:.2%}"],
        ["Acurácia de Lemmas", f"{metrics['lemma_accuracy']:.2%}"],
        ["UAS", f"{metrics['uas']:.2%}"],
        ["LAS", f"{metrics['las']:.2%}"],
        ["Precisão em Tokenização", f"{metrics['token_precision']:.2%}"],
        ["Recall em Tokenização", f"{metrics['token_recall']:.2%}"],
        ["F1-Score em Tokenização", f"{metrics['token_f1']:.2%}"]
    ]
    metrics_table = tabulate(metric_rows, headers=["Métrica", "Valor"], tablefmt="grid")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== Métricas Gerais ===\n")
        f.write(metrics_table + "\n\n")
        if max_sentences is None:
            max_sentences = len(results)
        for i, sent in enumerate(results[:max_sentences]):
            f.write(f"\n=== Sentença {i+1} ({sent['sent_id']}) ===\n")
            f.write("Texto: " + sent["text"] + "\n\n")
            f.write("Tokens Gold vs. spaCy:\n")
            tokens_data = [
                ["Gold Tokens", "spaCy Tokens"],
                [" | ".join(sent["gold_tokens"]), " | ".join(sent["pred_tokens"])],
            ]
            f.write(tabulate(tokens_data, tablefmt="plain") + "\n\n")
            comp_headers = [
                "Token",
                "Gold POS",
                "spaCy POS",
                "Gold HEAD",
                "spaCy HEAD",
                "Gold DEPREL",
                "spaCy DEPREL",
                "Gold Lemma",
                "spaCy Lemma",
            ]
            comp_rows = []
            for j in range(len(sent["gold_tokens"])):
                comp_rows.append(
                    [
                        sent["gold_tokens"][j],
                        sent["gold_pos"][j],
                        sent["pred_pos"][j],
                        sent["gold_heads"][j],
                        sent["pred_heads"][j],
                        sent["gold_deprels"][j],
                        sent["pred_deprels"][j],
                        sent["gold_lemmas"][j],
                        sent["pred_lemmas"][j],
                    ]
                )
            table = tabulate(comp_rows, headers=comp_headers, tablefmt="grid")
            f.write(table + "\n")

def save_conllu_error_analysis(error_analysis, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== Análise de Erros ===\n\n")

        # Relatório de Erros de POS Tagging
        f.write(f"Erros de POS Tagging ({len(error_analysis['pos_errors'])})\n")
        for error in error_analysis["pos_errors"]:
            f.write(f"Sentença ID: {error['sent_id']}\n")
            f.write(f"Texto: {error['text']}\n")
            f.write(f"Token: {error['token']} (Posição: {error['position']})\n")
            f.write(f"Gold: {error['gold']} | Predito: {error['predicted']}\n\n")

        # Relatório de Erros de Lematização
        f.write(f"\nErros de Lematização ({len(error_analysis['lemma_errors'])})\n")
        for error in error_analysis["lemma_errors"]:
            f.write(f"Sentença ID: {error['sent_id']}\n")
            f.write(f"Texto: {error['text']}\n")
            f.write(f"Token: {error['token']} (Posição: {error['position']})\n")
            f.write(f"Gold: {error['gold']} | Predito: {error['predicted']}\n\n")

        # Relatório de Erros de Dependências
        f.write(f"\nErros de Dependências ({len(error_analysis['dependency_errors'])})\n")
        for error in error_analysis["dependency_errors"]:
            f.write(f"Sentença ID: {error['sent_id']}\n")
            f.write(f"Texto: {error['text']}\n")
            f.write(f"Token: {error['token']} (Posição: {error['position']})\n")
            f.write(f"Gold: {error['gold']} | Predito: {error['predicted']}\n\n")

def save_ner_results_to_file(metrics, report_data, output_path):
    metricas = [
        ["Precisão (NER)", f"{metrics['precision']:.2%}"],
        ["Recall (NER)", f"{metrics['recall']:.2%}"],
        ["F1 (NER)", f"{metrics['f1']:.2%}"]
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== Métricas NER ===\n")
        f.write(tabulate(metricas, headers=["Métrica", "Valor"], tablefmt="grid"))
        f.write("\n\n")
        f.write("\n".join(report_data))


