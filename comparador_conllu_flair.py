import os
from flair.data import Sentence
from flair.models import SequenceTagger
from tabulate import tabulate

# Carregar apenas o modelo de POS do Flair
tagger_pos = SequenceTagger.load("pos-multi")  # Modelo de POS

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
                token, pos, head, deprel, lemma = parts[1], parts[3], int(parts[6]), parts[7], parts[2]
                if 'tokens' not in current_sent:
                    current_sent.update({'tokens': [], 'gold_pos': [], 'gold_heads': [],
                                         'gold_deprels': [], 'gold_lemmas': [], 'gold_spans': []})
                current_sent['tokens'].append(token)
                current_sent['gold_pos'].append(pos)
                current_sent['gold_heads'].append(head)
                current_sent['gold_deprels'].append(deprel)
                current_sent['gold_lemmas'].append(lemma)
                start = current_sent['text'].find(token) if 'text' in current_sent else -1
                current_sent['gold_spans'].append((start, start + len(token)) if start != -1 else (0, 0))
            elif line == '':
                if current_sent:
                    sentences.append(current_sent)
                    current_sent = {}
    return sentences

# Função para avaliar com o Flair
def evaluate_flair(sentences):
    results = []
    for sent in sentences:
        text = sent['text']
        sentence = Sentence(text)

        # Processa POS tagging
        tagger_pos.predict(sentence)
        pred_tokens = [token.text for token in sentence]
        
        # Extrair POS tags com tratamento de diferentes versões do Flair
        pred_pos = []
        for token in sentence:
            try:
                # Tenta o método mais recente
                if hasattr(token, 'get_tag'):
                    pred_pos.append(token.get_tag('pos').value)
                elif token.labels:
                    pred_pos.append(token.labels[0].value)
                else:
                    pred_pos.append("UNK")
            except (AttributeError, IndexError):
                pred_pos.append("UNK")

        # Lematização não disponível - usar tokens como lemmas
        pred_lemmas = pred_tokens  # Simples fallback

        # Estima os spans
        pred_spans = []
        start_offset = 0
        for token in pred_tokens:
            start = text.find(token, start_offset)
            end = start + len(token)
            pred_spans.append((start, end))
            start_offset = end

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
            'pred_lemmas': pred_lemmas
        })
    return results

# Função que calcula métricas de avaliação
def calculate_metrics(results):
    total_tokens = 0
    pos_correct = 0
    lemma_correct = 0
    tp_token, fp_token, fn_token = 0, 0, 0

    for sent in results:
        n = len(sent['gold_tokens'])
        total_tokens += n
        pos_correct += sum(1 for g, p in zip(sent['gold_pos'], sent['pred_pos']) if g == p)
        lemma_correct += sum(1 for g, p in zip(sent['gold_lemmas'], sent['pred_lemmas']) if g == p)

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
        'token_precision': precision,
        'token_recall': recall,
        'token_f1': f1
    }

# Função que salva os resultados e métricas
def save_results_to_file(results, metrics, output_file="resultados_completos.txt"):
    metric_rows = [
        ["Acurácia de POS", f"{metrics['pos_accuracy']:.2%}"],
        ["Acurácia de Lemas", f"{metrics['lemma_accuracy']:.2%}"],
        ["Precisão em Tokenização", f"{metrics['token_precision']:.2%}"],
        ["Recall em Tokenização", f"{metrics['token_recall']:.2%}"],
        ["F1-Score em Tokenização", f"{metrics['token_f1']:.2%}"]
    ]
    metrics_table = tabulate(metric_rows, headers=["Métrica", "Valor"], tablefmt="grid")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== Métricas Gerais ===\n")
        f.write(metrics_table + "\n\n")

        for i, sent in enumerate(results):
            f.write(f"\n=== Sentença {i+1} ({sent['sent_id']}) ===\n")
            f.write("Texto: " + sent['text'] + "\n\n")
            tokens_data = [
                ["Gold Tokens", "Flair Tokens"],
                [" | ".join(sent['gold_tokens']), " | ".join(sent['pred_tokens'])]
            ]
            f.write(tabulate(tokens_data, tablefmt="plain") + "\n\n")

            comp_headers = ["Token", "Gold POS", "Flair POS", "Gold Lemma", "Flair Lemma"]
            comp_rows = []
            for j in range(len(sent['gold_tokens'])):
                comp_rows.append([
                    sent['gold_tokens'][j],
                    sent['gold_pos'][j],
                    sent['pred_pos'][j] if j < len(sent['pred_pos']) else "",
                    sent['gold_lemmas'][j],
                    sent['pred_lemmas'][j] if j < len(sent['pred_lemmas']) else ""
                ])
            table = tabulate(comp_rows, headers=comp_headers, tablefmt="grid")
            f.write(table + "\n")

if __name__ == "__main__":
    # Criar diretório para salvar arquivos
    txt_dir = "analise_flair"
    os.makedirs(txt_dir, exist_ok=True)

    # 1. Parsear o arquivo CONLL-U
    sentences = parse_conllu("/home/andre/Dev-Ubuntu/IC/experimentos/scripts/Corupus/UD_Portuguese-Bosque-master/pt_bosque-ud-test.conllu")
    
    # 2. Avaliar com Flair
    results = evaluate_flair(sentences)

    # 3. Calcular métricas
    metrics = calculate_metrics(results)

    # 4. Salvar resultados
    save_results_to_file(results, metrics, output_file=f"{txt_dir}/resultados_completos.txt")

    print("Processamento concluído! Resultados salvos em 'analise_flair/resultados_completos.txt'")
