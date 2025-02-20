import os
import stanza
from tabulate import tabulate

# Função para parsear um arquivo CONLL-U e extrair anotações linguísticas
def parse_conllu(file_path):
    sentences = []
    current_sent = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('# text ='):
                current_sent['text'] = line.split('=', 1)[1].strip()
            elif line.startswith('# sent_id ='):
                current_sent['sent_id'] = line.split('=', 1)[1].strip()
            elif line and line[0].isdigit() and '-' not in line.split('\t')[0]:
                parts = line.split('\t')
                # Supondo que:
                # parts[1] = token, parts[3] = POS, parts[6] = HEAD, parts[7] = DEPREL e parts[2] = Lemma
                token = parts[1]
                pos = parts[3]
                head = int(parts[6])
                deprel = parts[7]
                lemma = parts[2]
                if 'tokens' not in current_sent:
                    current_sent['tokens'] = []
                    current_sent['gold_pos'] = []
                    current_sent['gold_heads'] = []
                    current_sent['gold_deprels'] = []
                    current_sent['gold_lemmas'] = []
                    # Para spans gold, usa-se um placeholder; idealmente, calcular a posição do token no texto
                    current_sent['gold_spans'] = []
                current_sent['tokens'].append(token)
                current_sent['gold_pos'].append(pos)
                current_sent['gold_heads'].append(head)
                current_sent['gold_deprels'].append(deprel)
                current_sent['gold_lemmas'].append(lemma)
                # Aqui podemos aproximar os gold_spans procurando o token no texto
                start = current_sent['text'].find(token) if 'text' in current_sent else -1
                current_sent['gold_spans'].append((start, start + len(token)) if start != -1 else (0, 0))
            elif line == '':
                if current_sent:
                    sentences.append(current_sent)
                    current_sent = {}
    return sentences

# Função para avaliar e comparar o processamento com o Stanza
def evaluate_stanza(sentences, pipeline):
    results = []
    for sent in sentences:
        text = sent['text']
        doc = pipeline(text)
        # Considera-se que cada texto corresponde a uma única sentença
        stanza_sent = doc.sentences[0]
        pred_tokens = [word.text for word in stanza_sent.words]
        # Estima os spans dos tokens no texto usando o método find sequential
        pred_spans = []
        start_offset = 0
        for token in pred_tokens:
            start = text.find(token, start_offset)
            end = start + len(token)
            pred_spans.append((start, end))
            start_offset = end
        pred_pos = [word.upos for word in stanza_sent.words]
        pred_heads = [word.head for word in stanza_sent.words]  # Stanza já usa indexação 1-indexada para HEAD
        pred_deprels = [word.deprel for word in stanza_sent.words]
        pred_lemmas = [word.lemma for word in stanza_sent.words]
        results.append({
            'sent_id': sent.get('sent_id', ''),
            'text': text,
            'gold_tokens': sent['tokens'],
            'gold_spans': sent['gold_spans'],
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

# Função que salva os resultados e métricas em um arquivo de saída
def save_results_to_file(results, metrics, output_file="resultados_completos.txt", max_sentences=None):
    metric_rows = [
        ["Acurácia de POS", f"{metrics['pos_accuracy']:.2%}"],
        ["Acurácia de Lemas", f"{metrics['lemma_accuracy']:.2%}"],
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
            f.write("Tokens Gold vs. Stanza:\n")
            tokens_data = [
                ["Gold Tokens", "Stanza Tokens"],
                [" | ".join(sent['gold_tokens']), " | ".join(sent['pred_tokens'])]
            ]
            f.write(tabulate(tokens_data, tablefmt="plain") + "\n\n")
            comp_headers = ["Token", "Gold POS", "Stanza POS", "Gold HEAD", "Stanza HEAD",
                            "Gold DEPREL", "Stanza DEPREL", "Gold Lemma", "Stanza Lemma"]
            comp_rows = []
            for j in range(len(sent['gold_tokens'])):
                comp_rows.append([
                    sent['gold_tokens'][j],
                    sent['gold_pos'][j],
                    sent['pred_pos'][j] if j < len(sent['pred_pos']) else "",
                    sent['gold_heads'][j],
                    sent['pred_heads'][j] if j < len(sent['pred_heads']) else "",
                    sent['gold_deprels'][j],
                    sent['pred_deprels'][j] if j < len(sent['pred_deprels']) else "",
                    sent['gold_lemmas'][j],
                    sent['pred_lemmas'][j] if j < len(sent['pred_lemmas']) else ""
                ])
            table = tabulate(comp_rows, headers=comp_headers, tablefmt="grid")
            f.write(table + "\n")

# Função para gerar relatório de erros
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
                    'position': idx,
                    'gold': g_pos,
                    'predicted': p_pos
                })
            if g_lemma != p_lemma:
                error_analysis['lemma_errors'].append({
                    'sent_id': sent_id,
                    'text': text,
                    'token': token,
                    'position': idx,
                    'gold': g_lemma,
                    'predicted': p_lemma
                })
            if g_head != p_head or g_deprel != p_deprel:
                error_analysis['dependency_errors'].append({
                    'sent_id': sent_id,
                    'text': text,
                    'token': token,
                    'position': idx,
                    'gold': f"{g_head} ({g_deprel})",
                    'predicted': f"{p_head} ({p_deprel})"
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

if __name__ == "__main__":
    # Criar diretório para arquivos txt se não existir
    txt_dir = "analise_stanza"
    os.makedirs(txt_dir, exist_ok=True)

    # Inicializa o pipeline do Stanza para o português
    stanza.download('pt')  # Baixa os modelos se necessário
    nlp_stanza = stanza.Pipeline('pt')

    # 1. Parsear o arquivo CONLL-U para extrair as sentenças e atributos
    sentences = parse_conllu("/home/andre/Dev-Ubuntu/IC/experimentos/scripts/data/UD_Portuguese-Bosque/pt_bosque-ud-test.conllu")
    # 2. Processar cada sentença com o modelo Stanza para obter dados preditos
    results = evaluate_stanza(sentences, nlp_stanza)
    # 3. Calcular as métricas comparativas entre os dados gold e os preditos
    metrics = calculate_metrics(results)
    # 4. Salvar os resultados detalhados e as métricas em um arquivo de saída
    save_results_to_file(results, metrics, output_file=f"{txt_dir}/resultados_completos.txt", max_sentences=len(results))
    # 5. Gerar análise de erros salvando em analise_stanza
    analyze_errors(results, output_file=f"{txt_dir}/analise_erros.txt")
    print("Processamento concluído! Resultados salvos em 'analise_stanza/resultados_completos.txt'")