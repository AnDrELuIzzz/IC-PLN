import spacy  # Importa o spaCy para processamento de linguagem natural
from spacy import displacy  # Para visualização de dependências
from spacy.tokens import Doc  # Usado para criar um objeto Doc com tokens gold
from tabulate import tabulate  # Importa tabulate para formatação de tabelas
import os  # Para manipulação de diretórios

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

# Função para avaliar e comparar o processamento do spaCy com os dados gold
def evaluate_spacy(sentences, nlp_model):
    """
    Processa sentenças utilizando spaCy para realizar tokenização, POS tagging e extração de dependências.
    Detalhes técnicos:
    - Carrega o modelo spaCy especificado.
    - Processa o texto completo para gerar tokens "pred".
    - Alinha os tokens gold criando um objeto Doc e processa com o pipeline spaCy para obter atributos.
    - Extrai spans de tokens e mapeia as dependências levando em conta a indexação do spaCy.
    """
    nlp = spacy.load(nlp_model)
    results = []
    for sent in sentences:
        text = sent['text']
        doc_raw = nlp(text)
        pred_spans = [(token.idx, token.idx + len(token)) for token in doc_raw]
        doc_gold = Doc(nlp.vocab, words=sent['tokens'])
        doc_gold = nlp(doc_gold)
        results.append({
            'sent_id': sent.get('sent_id', ''),
            'text': text,
            'gold_tokens': sent['tokens'],
            'gold_spans': sent['gold_spans'],
            'pred_tokens': [token.text for token in doc_raw],
            'pred_spans': pred_spans,
            'gold_pos': sent['gold_pos'],
            'gold_heads': sent['gold_heads'],
            'gold_deprels': sent['gold_deprels'],
            'gold_lemmas': sent['gold_lemmas'],  # Adiciona lemas gold
            'pred_pos': [token.pos_ for token in doc_gold],
            'pred_heads': [token.head.i + 1 if token.head != token else 0 for token in doc_gold],
            'pred_deprels': [token.dep_ for token in doc_gold],
            'pred_lemmas': [token.lemma_ for token in doc_gold]  # Adiciona lemas preditos
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
        lemma_correct += sum(1 for g, p in zip(sent['gold_lemmas'], sent['pred_lemmas']) if g == p)  # Compara lemas
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
        'lemma_accuracy': lemma_correct / total_tokens,  # Acurácia dos lemas
        'uas': uas_correct / total_tokens,
        'las': las_correct / total_tokens,
        'token_precision': precision,
        'token_recall': recall,
        'token_f1': f1
    }

# Função que salva os resultados e métricas em um arquivo de saída
def save_results_to_file(results, metrics, output_file="resultados_completos.txt", max_sentences=None):
    """
    Registra as métricas gerais e detalhes de cada sentença para análise posterior,
    utilizando formatação aprimorada com a biblioteca tabulate, que melhora a visualização
    dos dados para um artigo científico.
    """
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
            f.write("Tokens Gold vs. spaCy:\n")
            tokens_data = [
                ["Gold Tokens", "spaCy Tokens"],
                [" | ".join(sent['gold_tokens']), " | ".join(sent['pred_tokens'])]
            ]
            f.write(tabulate(tokens_data, tablefmt="plain") + "\n\n")
            comp_headers = ["Token", "Gold POS", "spaCy POS", "Gold HEAD", "spaCy HEAD",
                            "Gold DEPREL", "spaCy DEPREL", "Gold Lemma", "spaCy Lemma"]
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


# Nova função: Análise de Erros
def analyze_errors(results, output_file="analise_erros.txt"):
    """
    Gera um relatório detalhado dos erros cometidos pelo modelo em POS tagging, lematização e análise de dependências.
    Inclui informações sobre a sentença, o token específico e sua posição.
    """
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

        # Relatório de Erros de POS Tagging
        f.write(f"Erros de POS Tagging ({len(error_analysis['pos_errors'])}):\n")
        for error in error_analysis['pos_errors']:
            f.write(f"Sentença ID: {error['sent_id']}\n")
            f.write(f"Texto: {error['text']}\n")
            f.write(f"Token: {error['token']} (Posição: {error['position']})\n")
            f.write(f"Gold: {error['gold']} | Predito: {error['predicted']}\n\n")

        # Relatório de Erros de Lematização
        f.write("\nErros de Lematização ({len(error_analysis['lemma_errors'])}):\n")
        for error in error_analysis['lemma_errors']:
            f.write(f"Sentença ID: {error['sent_id']}\n")
            f.write(f"Texto: {error['text']}\n")
            f.write(f"Token: {error['token']} (Posição: {error['position']})\n")
            f.write(f"Gold: {error['gold']} | Predito: {error['predicted']}\n\n")

        # Relatório de Erros de Dependências
        f.write("\nErros de Dependências ({len(error_analysis['dependency_errors'])}):\n")
        for error in error_analysis['dependency_errors']:
            f.write(f"Sentença ID: {error['sent_id']}\n")
            f.write(f"Texto: {error['text']}\n")
            f.write(f"Token: {error['token']} (Posição: {error['position']})\n")
            f.write(f"Gold: {error['gold']} | Predito: {error['predicted']}\n\n")

    print(f"Relatório de erros gerado: '{output_file}'")
    """
    Gera um relatório detalhado dos erros cometidos pelo modelo em POS tagging, lematização e análise de dependências.
    """
    error_analysis = {
        'pos_errors': [],
        'lemma_errors': [],
        'dependency_errors': []
    }
    for sent in results:
        for g_pos, p_pos, g_lemma, p_lemma, g_head, p_head, g_deprel, p_deprel in zip(
            sent['gold_pos'], sent['pred_pos'],
            sent['gold_lemmas'], sent['pred_lemmas'],
            sent['gold_heads'], sent['pred_heads'],
            sent['gold_deprels'], sent['pred_deprels']
        ):
            if g_pos != p_pos:
                error_analysis['pos_errors'].append((g_pos, p_pos))
            if g_lemma != p_lemma:
                error_analysis['lemma_errors'].append((g_lemma, p_lemma))
            if g_head != p_head or g_deprel != p_deprel:
                error_analysis['dependency_errors'].append(((g_head, g_deprel), (p_head, p_deprel)))

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== Análise de Erros ===\n\n")
        f.write(f"Erros de POS Tagging ({len(error_analysis['pos_errors'])}):\n")
        for error in error_analysis['pos_errors']:
            f.write(f"Gold: {error[0]} | Predito: {error[1]}\n")
        
        f.write("\nErros de Lematização ({len(error_analysis['lemma_errors'])}):\n")
        for error in error_analysis['lemma_errors']:
            f.write(f"Gold: {error[0]} | Predito: {error[1]}\n")
        
        f.write("\nErros de Dependências ({len(error_analysis['dependency_errors'])}):\n")
        for error in error_analysis['dependency_errors']:
            f.write(f"Gold: {error[0]} | Predito: {error[1]}\n")

    print(f"Relatório de erros gerado: '{output_file}'")

# Nova função: Visualização de Dependências
def visualize_dependencies(sentences, nlp_model, output_dir="visualizations"):
    """
    Gera visualizações gráficas das árvores de dependência para cada sentença.
    """
    nlp = spacy.load(nlp_model)
    os.makedirs(output_dir, exist_ok=True)  # Cria o diretório se ele não existir
    for i, sent in enumerate(sentences):
        text = sent['text']
        doc = nlp(text)
        svg = displacy.render(doc, style="dep", jupyter=False)
        with open(f"{output_dir}/sentence_{i+1}.svg", "w", encoding="utf-8") as f:
            f.write(svg)
    print(f"Visualizações de dependências salvas em '{output_dir}'")

# Execução principal
if __name__ == "__main__":
    # Criar diretório para arquivos txt se não existir
    txt_dir = "analise_spacy"
    os.makedirs(txt_dir, exist_ok=True)

    # 1. Parsear o arquivo CONLL-U para extrair as sentenças e respectivos atributos
    sentences = parse_conllu("/home/andre/Dev-Ubuntu/IC/experimentos/scripts/data/UD_Portuguese-Bosque/pt_bosque-ud-test.conllu")
    # 2. Processar cada sentença com o modelo spaCy para obter dados preditos
    results = evaluate_spacy(sentences, "pt_core_news_lg")
    # 3. Calcular as métricas comparativas entre os dados gold e os preditos
    metrics = calculate_metrics(results)
    # 4. Salvar os resultados detalhados e as métricas em um arquivo de saída dentro de analise_spacy
    save_results_to_file(results, metrics, output_file=f"{txt_dir}/resultados_completos.txt", max_sentences=len(results))
    # 5. Gerar análise de erros salvando em analise_spacy
    analyze_errors(results, output_file=f"{txt_dir}/analise_erros.txt")
    # 6. Gerar visualizações de dependências
    visualize_dependencies(sentences, "pt_core_news_lg", output_dir="visualizations")
    # 7. Mensagem de confirmação da execução
    print("Processamento concluído! Resultados salvos em 'analise_spacy/resultados_completos.txt'")