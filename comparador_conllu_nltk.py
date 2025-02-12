import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tabulate import tabulate  # Importa tabulate para formatação de tabelas

#O NLTK não possui um modelo pré-treinado específico para POS tagging em português.

# Função para parsear um arquivo CONLL-U e extrair anotações linguísticas
def parse_conllu(file_path):
    """
    Faz parsing de um arquivo no formato CONLL-U extraindo informações essenciais.
    Detalhes técnicos:
    - Identifica metadados como texto original e id da sentença.
    - A partir do token '1\t' inicia a extração dos tokens e atributos (POS, HEAD, DEPREL).
    - Calcula spans dos tokens de acordo com sua posição no texto original, utilizando 
      a função find() para manter consistência com a formatação do texto.
    """
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
                    current_sent['gold_lemmas'] = []  # Nova chave para armazenar lemas
                parts = line.split('\t')
                current_sent['tokens'].append(parts[1])
                current_sent['gold_pos'].append(parts[3])
                current_sent['gold_heads'].append(int(parts[6]))
                current_sent['gold_deprels'].append(parts[7])
                current_sent['gold_lemmas'].append(parts[2])  # Captura o lema
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


# Função para avaliar e comparar o processamento do NLTK com os dados gold
def evaluate_nltk(sentences):
    """
    Espelha a abordagem do evaluate_spacy, mas usando o NLTK para tokens, POS, lemas e placeholders de HEAD/DEPREL.
    """
    lemmatizer = WordNetLemmatizer()
    results = []
    for sent in sentences:
        text = sent['text']
        # Para manter a coerência com a abordagem do spaCy, usa-se os tokens gold
        gold_tokens = sent['tokens']
        # POS tagging: NLTK não tem suporte nativo para PT, então utiliza-se modo genérico
        nltk_pos = nltk.pos_tag(gold_tokens)
        # HEADs e DEPRELs como placeholders (pois NLTK não oferece parsing de dependência PT)
        pred_heads = [0] * len(gold_tokens)
        pred_deprels = ["dep"] * len(gold_tokens)
        # Lemmatização simples via WordNetLemmatizer (mais adequado ao inglês)
        pred_lemmas = [lemmatizer.lemmatize(tok.lower()) for tok in gold_tokens]
        # Spans preditos com base no texto
        pred_spans = []
        current_pos = 0
        for token in gold_tokens:
            start = text.find(token, current_pos)
            end = start + len(token)
            pred_spans.append((start, end))
            current_pos = end
        results.append({
            'sent_id': sent.get('sent_id', ''),
            'text': text,
            'gold_tokens': gold_tokens,
            'gold_spans': sent['gold_spans'],
            'pred_tokens': gold_tokens,
            'pred_spans': pred_spans,
            'gold_pos': sent['gold_pos'],
            'gold_heads': sent['gold_heads'],
            'gold_deprels': sent['gold_deprels'],
            'gold_lemmas': sent['gold_lemmas'],
            'pred_pos': [pos for (_, pos) in nltk_pos],
            'pred_heads': pred_heads,
            'pred_deprels': pred_deprels,
            'pred_lemmas': pred_lemmas
        })
    return results


# Função que calcula métricas de avaliação comparando dados de tokens, POS e dependências
def calculate_metrics(results):
    """
    Calcula métricas de comparação entre os dados gold e os preditos.
    Detalhes técnicos:
    - Acurácia de POS é calculada comparando elemento a elemento dos tokens.
    - UAS e LAS são derivados da verificação de acertos na estrutura de dependências.
    - Precision, Recall e F1 em tokenização são derivados dos spans sobrepostos.
    """
    total_tokens = 0
    pos_correct = 0
    uas_correct = 0
    las_correct = 0
    lemma_correct = 0  # Contador para lemas corretos
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
def save_results_to_file(results, metrics, output_file="resultados_completos_nltk.txt", max_sentences=None):
    # Mesmo estilo da versão com spaCy
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
            f.write("Tokens Gold vs. NLTK:\n")
            tokens_data = [
                ["Gold Tokens", "NLTK Tokens"],
                [" | ".join(sent['gold_tokens']), " | ".join(sent['pred_tokens'])]
            ]
            f.write(tabulate(tokens_data, tablefmt="plain") + "\n\n")
            comp_headers = ["Token", "Gold POS", "NLTK POS", "Gold HEAD", "NLTK HEAD",
                            "Gold DEPREL", "NLTK DEPREL", "Gold Lemma", "NLTK Lemma"]
            comp_rows = []
            max_len = min(
                len(sent['gold_tokens']),
                len(sent['pred_tokens']),
                len(sent['gold_pos']),
                len(sent['pred_pos']),
                len(sent['gold_heads']),
                len(sent['pred_heads']),
                len(sent['gold_deprels']),
                len(sent['pred_deprels']),
                len(sent['gold_lemmas']),
                len(sent['pred_lemmas'])
            )
            for j in range(max_len):
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


# Execução principal
if __name__ == "__main__":
    # 1. Parsear o arquivo CONLL-U para extrair as sentenças e respectivos atributos
    sentences = parse_conllu("/home/andre/Dev-Ubuntu/IC/experimentos/scripts/data/UD_Portuguese-Bosque/pt_bosque-ud-test.conllu")
    # 2. Processar cada sentença com o NLTK para obter dados preditos
    results = evaluate_nltk(sentences)
    # 3. Calcular as métricas comparativas entre os dados gold e os preditos
    metrics = calculate_metrics(results)
    # 4. Salvar os resultados detalhados e as métricas em um arquivo de saída
    save_results_to_file(results, metrics, max_sentences=len(results))
    # 5. Mensagem de confirmação da execução
    print("Processamento concluído! Resultados salvos em 'resultados_completos_nltk.txt'")