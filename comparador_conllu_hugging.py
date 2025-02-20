import os
from transformers import pipeline
from tabulate import tabulate
from graphviz import Digraph

# Função para parsear arquivo CONLL-U (igual ao original)
def parse_conllu(file_path):
    sentences = []
    current_sent = {}
    in_sentence = False
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('# text ='):
                current_sent['text'] = line.split('=', 1)[1].strip()
            elif line.startswith('# sent_id ='):
                current_sent['sent_id'] = line.split('=', 1)[1].strip()
            elif line and line[0].isdigit() and '-' not in line.split('\t')[0]:
                if not in_sentence:
                    in_sentence = True
                    current_sent['tokens'] = []
                    current_sent['gold_pos'] = []
                    current_sent['gold_heads'] = []
                    current_sent['gold_deprels'] = []
                    current_sent['gold_lemmas'] = []
                parts = line.split('\t')
                current_sent['tokens'].append(parts[1])
                current_sent['gold_pos'].append(parts[3])
                current_sent['gold_heads'].append(int(parts[6]))
                current_sent['gold_deprels'].append(parts[7])
                current_sent['gold_lemmas'].append(parts[2])
            elif line == '':
                if in_sentence and current_sent.get('tokens'):
                    text = current_sent['text']
                    current_pos = 0
                    spans = []
                    for token in current_sent['tokens']:
                        start = text.find(token, current_pos)
                        end = start + len(token)
                        spans.append((start, end))
                        current_pos = end
                    current_sent['gold_spans'] = spans
                    sentences.append(current_sent)
                    current_sent = {}
                    in_sentence = False
    return sentences

# Função para avaliar as sentenças utilizando Hugging Face
def evaluate_huggingface(sentences, model_id):
    # Cria pipeline para token classification (ex: POS tagging)
    pos_pipeline = pipeline("token-classification", model=model_id, aggregation_strategy="simple")
    results = []
    for sent in sentences:
        text = sent['text']
        # Obtem predições a partir do texto completo
        predictions = pos_pipeline(text)
        # Alinha os resultados com os tokens gold usando os spans já calculados
        gold_spans = sent['gold_spans']
        pred_pos = []
        for span in gold_spans:
            label_found = None
            for pred in predictions:
                if abs(pred['start'] - span[0]) < 3:
                    label_found = pred.get("entity_group", pred.get("entity"))


                    break
            if label_found is None:
                label_found = "O"
            pred_pos.append(label_found)
        # Para as demais informações (tokens, spans, lemas e dependências) usamos aproximações:
        pred_tokens = sent['tokens']
        pred_spans = gold_spans  # Para demonstração, considera os spans gold
        # Dependências: para efeito de demonstração, o primeiro token é a raiz e os demais apontam para ele
        pred_heads = [0] + [0]*(len(pred_tokens)-1)
        pred_deprels = ["root"] + ["dep"]*(len(pred_tokens)-1)
        # Lematização: aqui, usa o token em minúsculo como lema predito
        pred_lemmas = [token.lower() for token in pred_tokens]
        results.append({
            'sent_id': sent.get('sent_id', ''),
            'text': text,
            'gold_tokens': sent['tokens'],
            'gold_spans': gold_spans,
            'pred_tokens': pred_tokens,
            'pred_spans': pred_spans,
            'gold_pos': sent['gold_pos'],
            'gold_heads': sent['gold_heads'],
            'gold_deprels': sent['gold_deprels'],
            'gold_lemmas': sent['gold_lemmas'],
            'pred_pos': pred_pos,
            'pred_heads': pred_heads,
            'pred_deprels': pred_deprels,
            'pred_lemmas': pred_lemmas
        })
    return results

# Função que calcula métricas de avaliação
def calculate_metrics(results):
    total_tokens = 0
    pos_correct = 0
    uas_correct = 0
    las_correct = 0
    lemma_correct = 0
    tp_token, fp_token, fn_token = 0, 0, 0
    for sent in results:
        n = len(sent['gold_tokens'])
        total_tokens += n
        pos_correct += sum(1 for g, p in zip(sent['gold_pos'], sent['pred_pos']) if g == p)
        lemma_correct += sum(1 for g, p in zip(sent['gold_lemmas'], sent['pred_lemmas']) if g == p)
        for g_head, p_head, g_deprel, p_deprel in zip(
            sent['gold_heads'], sent['pred_heads'], sent['gold_deprels'], sent['pred_deprels']
        ):
            if g_head == p_head:
                uas_correct += 1
                if g_deprel == p_deprel:
                    las_correct += 1
        gold_spans = set(sent['gold_spans'])
        pred_spans = set(sent['pred_spans'])
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
        'pos_accuracy': pos_correct / total_tokens,
        'lemma_accuracy': lemma_correct / total_tokens,
        'uas': uas_correct / total_tokens,
        'las': las_correct / total_tokens,
        'token_precision': precision,
        'token_recall': recall,
        'token_f1': f1
    }

# Função para salvar resultados e métricas em arquivo
def save_results_to_file(results, metrics, output_file="resultados_completos.txt", max_sentences=None):
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
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== Métricas Gerais ===\n")
        f.write(metrics_table + "\n\n")
        if max_sentences is None:
            max_sentences = len(results)
        for i, sent in enumerate(results[:max_sentences]):
            f.write(f"\n=== Sentença {i+1} ({sent['sent_id']}) ===\n")
            f.write("Texto: " + sent['text'] + "\n\n")
            tokens_data = [
                ["Gold Tokens", "HuggingFace Tokens"],
                [" | ".join(sent['gold_tokens']), " | ".join(sent['pred_tokens'])]
            ]
            f.write(tabulate(tokens_data, tablefmt="plain") + "\n\n")
            comp_headers = ["Token", "Gold POS", "HuggingFace POS", "Gold HEAD", "HuggingFace HEAD",
                            "Gold DEPREL", "HuggingFace DEPREL", "Gold Lemma", "HuggingFace Lemma"]
            comp_rows = []
            for j in range(len(sent['gold_tokens'])):
                comp_rows.append([
                    sent['gold_tokens'][j],
                    sent['gold_pos'][j],
                    sent['pred_pos'][j],
                    sent['gold_heads'][j],
                    sent['pred_heads'][j],
                    sent['gold_deprels'][j],
                    sent['pred_deprels'][j],
                    sent['gold_lemmas'][j],
                    sent['pred_lemmas'][j]
                ])
            table = tabulate(comp_rows, headers=comp_headers, tablefmt="grid")
            f.write(table + "\n")

# Função de análise de erros
def analyze_errors(results, output_file="analise_erros.txt"):
    error_analysis = {
        'pos_errors': [],
        'lemma_errors': [],
        'dependency_errors': []
    }
    for sent in results:
        sent_id = sent.get('sent_id', 'N/A')
        text = sent['text']
        for idx, (g_pos, p_pos, g_lemma, p_lemma, g_head, p_head, g_deprel, p_deprel) in enumerate(zip(
            sent['gold_pos'], sent['pred_pos'],
            sent['gold_lemmas'], sent['pred_lemmas'],
            sent['gold_heads'], sent['pred_heads'],
            sent['gold_deprels'], sent['pred_deprels']
        )):
            token = sent['gold_tokens'][idx]
            if g_pos != p_pos:
                error_analysis['pos_errors'].append({
                    'sent_id': sent_id,
                    'text': text,
                    'token': token,
                    'position': idx + 1,
                    'gold': g_pos,
                    'predicted': p_pos
                })
            if g_lemma != p_lemma:
                error_analysis['lemma_errors'].append({
                    'sent_id': sent_id,
                    'text': text,
                    'token': token,
                    'position': idx + 1,
                    'gold': g_lemma,
                    'predicted': p_lemma
                })
            if g_head != p_head or g_deprel != p_deprel:
                error_analysis['dependency_errors'].append({
                    'sent_id': sent_id,
                    'text': text,
                    'token': token,
                    'position': idx + 1,
                    'gold': (g_head, g_deprel),
                    'predicted': (p_head, p_deprel)
                })
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== Análise de Erros ===\n\n")
        f.write(f"Erros de POS Tagging ({len(error_analysis['pos_errors'])}):\n")
        for error in error_analysis['pos_errors']:
            f.write(f"Sentença ID: {error['sent_id']}\n")
            f.write(f"Texto: {error['text']}\n")
            f.write(f"Token: {error['token']} (Posição: {error['position']})\n")
            f.write(f"Gold: {error['gold']} | Predito: {error['predicted']}\n\n")
        f.write("\nErros de Lematização ({len(error_analysis['lemma_errors'])}):\n")
        for error in error_analysis['lemma_errors']:
            f.write(f"Sentença ID: {error['sent_id']}\n")
            f.write(f"Texto: {error['text']}\n")
            f.write(f"Token: {error['token']} (Posição: {error['position']})\n")
            f.write(f"Gold: {error['gold']} | Predito: {error['predicted']}\n\n")
        f.write("\nErros de Dependências ({len(error_analysis['dependency_errors'])}):\n")
        for error in error_analysis['dependency_errors']:
            f.write(f"Sentença ID: {error['sent_id']}\n")
            f.write(f"Texto: {error['text']}\n")
            f.write(f"Token: {error['token']} (Posição: {error['position']})\n")
            f.write(f"Gold: {error['gold']} | Predito: {error['predicted']}\n\n")
    print(f"Relatório de erros gerado: '{output_file}'")

# Função para visualizar dependências usando graphviz
def visualize_dependencies(sentences, output_dir="visualizations"):
    os.makedirs(output_dir, exist_ok=True)
    for i, sent in enumerate(sentences):
        dot = Digraph(comment=f"Árvore de dependência - Sentença {i+1}")
        tokens = sent['tokens']
        # Para demonstração, o primeiro token é a raiz; os demais apontam para ele
        for idx, token in enumerate(tokens):
            dot.node(str(idx), token)
        for idx in range(1, len(tokens)):
            dot.edge("0", str(idx), label="dep")
        output_path = os.path.join(output_dir, f"sentence_{i+1}.gv")
        dot.render(output_path, format='svg', cleanup=True)
    print(f"Visualizações de dependências salvas em '{output_dir}'")

# Execução principal
if __name__ == "__main__":
    # Cria diretório para arquivos txt se não existir
    txt_dir = "analise_hf"
    os.makedirs(txt_dir, exist_ok=True)

    # 1. Parseia arquivo CONLL-U
    conllu_file = "/home/andre/Dev-Ubuntu/IC/experimentos/scripts/data/UD_Portuguese-Bosque/pt_bosque-ud-test.conllu"
    sentences = parse_conllu(conllu_file)
    # 2. Processa cada sentença utilizando Hugging Face com o modelo especificado
    results = evaluate_huggingface(sentences, model_id="ricardoz/BERTugues-base-portuguese-cased")
    # 3. Calcula métricas comparativas
    metrics = calculate_metrics(results)
    # 4. Salva resultados detalhados e métricas
    save_results_to_file(results, metrics, output_file=f"{txt_dir}/resultados_completos.txt", max_sentences=len(results))
    # 5. Gera análise de erros
    analyze_errors(results, output_file=f"{txt_dir}/analise_erros.txt")
    # 6. Gera visualizações de dependências (usando graphviz)
    visualize_dependencies(sentences, output_dir="visualizations")
    # 7. Mensagem de confirmação
    print("Processamento concluído! Resultados salvos em 'analise_hf/resultados_completos.txt'")