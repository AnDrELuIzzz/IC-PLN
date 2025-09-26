import ufal.udpipe as udpipe  # Biblioteca UDPipe para processamento
from tabulate import tabulate  # Para formatação de tabelas
import os  # Para manipulação de diretórios

# Função para parsear um arquivo CONLL-U e extrair anotações linguísticas
def parse_conllu(file_path):
    """
    Faz parsing de um arquivo no formato CONLL-U extraindo informações essenciais.
    Detalhes técnicos:
    - Identifica metadados como texto original e id da sentença.
    - A partir do token '1\t' inicia a extração dos tokens e atributos (POS, HEAD, DEPREL).
    - Calcula spans dos tokens de acordo com sua posição no texto original.
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
                    current_sent['gold_lemmas'] = []
                    current_sent['gold_upos'] = []
                    current_sent['gold_xpos'] = []
                    current_sent['gold_feats'] = []
                
                parts = line.split('\t')
                current_sent['tokens'].append(parts[1])
                current_sent['gold_lemmas'].append(parts[2])
                current_sent['gold_upos'].append(parts[3])
                current_sent['gold_xpos'].append(parts[4])
                current_sent['gold_feats'].append(parts[5])
                current_sent['gold_heads'].append(int(parts[6]))
                current_sent['gold_deprels'].append(parts[7])
                
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

# Função para carregar o modelo UDPipe
def load_udpipe_model(model_path):
    """
    Carrega o modelo UDPipe para português brasileiro.
    O modelo pode ser baixado de: https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3131
    """
    model = udpipe.Model.load(model_path)
    if not model:
        raise Exception(f"Erro ao carregar o modelo UDPipe: {model_path}")
    
    pipeline = udpipe.Pipeline(model, "tokenize", udpipe.Pipeline.DEFAULT, udpipe.Pipeline.DEFAULT, "conllu")
    return model, pipeline

# Função para processar texto com UDPipe
def process_with_udpipe(text, pipeline):
    """
    Processa o texto usando UDPipe e retorna as anotações em formato CONLL-U.
    """
    processed = pipeline.process(text)
    return processed

# Função para parsear a saída do UDPipe
def parse_udpipe_output(conllu_output):
    """
    Converte a saída CONLL-U do UDPipe em estruturas de dados Python.
    """
    lines = conllu_output.strip().split('\n')
    tokens = []
    lemmas = []
    upos_tags = []
    xpos_tags = []
    feats = []
    heads = []
    deprels = []
    spans = []
    
    for line in lines:
        if line and not line.startswith('#') and line[0].isdigit():
            parts = line.split('\t')
            if '-' not in parts[0]:  # Ignora tokens multi-word
                tokens.append(parts[1])
                lemmas.append(parts[2])
                upos_tags.append(parts[3])
                xpos_tags.append(parts[4])
                feats.append(parts[5])
                heads.append(int(parts[6]))
                deprels.append(parts[7])
                # Para spans, assumimos posições sequenciais (UDPipe não fornece spans exatos)
                start = len(' '.join(tokens[:-1])) + (1 if len(tokens) > 1 else 0)
                end = start + len(parts[1])
                spans.append((start, end))
    
    return {
        'tokens': tokens,
        'lemmas': lemmas,
        'upos': upos_tags,
        'xpos': xpos_tags,
        'feats': feats,
        'heads': heads,
        'deprels': deprels,
        'spans': spans
    }

# Função para avaliar e comparar o processamento do UDPipe com os dados gold
def evaluate_udpipe(sentences, model_path):
    """
    Processa sentenças utilizando UDPipe para realizar tokenização, POS tagging e extração de dependências.
    """
    model, pipeline = load_udpipe_model(model_path)
    results = []
    
    for sent in sentences:
        text = sent['text']
        
        # Processar com UDPipe
        udpipe_output = process_with_udpipe(text, pipeline)
        pred_data = parse_udpipe_output(udpipe_output)
        
        results.append({
            'sent_id': sent.get('sent_id', ''),
            'text': text,
            'gold_tokens': sent['tokens'],
            'gold_spans': sent['gold_spans'],
            'pred_tokens': pred_data['tokens'],
            'pred_spans': pred_data['spans'],
            'gold_upos': sent['gold_upos'],
            'gold_xpos': sent['gold_xpos'],
            'gold_feats': sent['gold_feats'],
            'gold_heads': sent['gold_heads'],
            'gold_deprels': sent['gold_deprels'],
            'gold_lemmas': sent['gold_lemmas'],
            'pred_upos': pred_data['upos'],
            'pred_xpos': pred_data['xpos'],
            'pred_feats': pred_data['feats'],
            'pred_heads': pred_data['heads'],
            'pred_deprels': pred_data['deprels'],
            'pred_lemmas': pred_data['lemmas']
        })
    
    return results

# Função que calcula métricas de avaliação
def calculate_metrics(results):
    """
    Calcula métricas de comparação entre os dados gold e os preditos pelo UDPipe.
    Inclui UPOS, XPOS, lematização, UAS e LAS.
    """
    total_tokens = 0
    upos_correct = 0
    xpos_correct = 0
    lemma_correct = 0
    uas_correct = 0
    las_correct = 0
    tp_token, fp_token, fn_token = 0, 0, 0
    
    for sent in results:
        n = len(sent['gold_tokens'])
        total_tokens += n
        
        # Calcular acurácia para cada tarefa
        upos_correct += sum(1 for g, p in zip(sent['gold_upos'], sent['pred_upos']) if g == p)
        xpos_correct += sum(1 for g, p in zip(sent['gold_xpos'], sent['pred_xpos']) if g == p)
        lemma_correct += sum(1 for g, p in zip(sent['gold_lemmas'], sent['pred_lemmas']) if g == p)
        
        # Calcular UAS e LAS
        for g_head, p_head, g_deprel, p_deprel in zip(
            sent['gold_heads'], sent['pred_heads'], sent['gold_deprels'], sent['pred_deprels']
        ):
            if g_head == p_head:
                uas_correct += 1
                if g_deprel == p_deprel:
                    las_correct += 1
        
        # Calcular métricas de tokenização
        gold_spans = set(sent['gold_spans'])
        pred_spans = set(sent['pred_spans'])
        tp = len(gold_spans & pred_spans)
        fp = len(pred_spans - gold_spans)
        fn = len(gold_spans - pred_spans)
        tp_token += tp
        fp_token += fp
        fn_token += fn
    
    # Calcular precision, recall e F1 para tokenização
    precision = tp_token / (tp_token + fp_token) if (tp_token + fp_token) > 0 else 0
    recall = tp_token / (tp_token + fn_token) if (tp_token + fn_token) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'upos_accuracy': upos_correct / total_tokens,
        'xpos_accuracy': xpos_correct / total_tokens,
        'lemma_accuracy': lemma_correct / total_tokens,
        'uas': uas_correct / total_tokens,
        'las': las_correct / total_tokens,
        'token_precision': precision,
        'token_recall': recall,
        'token_f1': f1
    }

# Função que salva os resultados e métricas em um arquivo de saída
def save_results_to_file(results, metrics, output_file="resultados_udpipe.txt", max_sentences=None):
    """
    Registra as métricas gerais e detalhes de cada sentença para análise posterior.
    """
    metric_rows = [
        ["Acurácia de UPOS", f"{metrics['upos_accuracy']:.2%}"],
        ["Acurácia de XPOS", f"{metrics['xpos_accuracy']:.2%}"],
        ["Acurácia de Lemmas", f"{metrics['lemma_accuracy']:.2%}"],
        ["UAS", f"{metrics['uas']:.2%}"],
        ["LAS", f"{metrics['las']:.2%}"],
        ["Precisão em Tokenização", f"{metrics['token_precision']:.2%}"],
        ["Recall em Tokenização", f"{metrics['token_recall']:.2%}"],
        ["F1-Score em Tokenização", f"{metrics['token_f1']:.2%}"]
    ]
    
    metrics_table = tabulate(metric_rows, headers=["Métrica", "Valor"], tablefmt="grid")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== Métricas Gerais - UDPipe ===\n")
        f.write(metrics_table + "\n\n")
        
        if max_sentences is None:
            max_sentences = len(results)
            
        for i, sent in enumerate(results[:max_sentences]):
            f.write(f"\n=== Sentença {i+1} ({sent['sent_id']}) ===\n")
            f.write("Texto: " + sent['text'] + "\n\n")
            
            f.write("Tokens Gold vs. UDPipe:\n")
            tokens_data = [
                ["Gold Tokens", "UDPipe Tokens"],
                [" | ".join(sent['gold_tokens']), " | ".join(sent['pred_tokens'])]
            ]
            f.write(tabulate(tokens_data, tablefmt="plain") + "\n\n")
            
            # Tabela detalhada de comparação
            comp_headers = ["Token", "Gold UPOS", "UDPipe UPOS", "Gold XPOS", "UDPipe XPOS",
                           "Gold HEAD", "UDPipe HEAD", "Gold DEPREL", "UDPipe DEPREL", 
                           "Gold Lemma", "UDPipe Lemma"]
            comp_rows = []
            
            for j in range(len(sent['gold_tokens'])):
                comp_rows.append([
                    sent['gold_tokens'][j],
                    sent['gold_upos'][j],
                    sent['pred_upos'][j],
                    sent['gold_xpos'][j],
                    sent['pred_xpos'][j],
                    sent['gold_heads'][j],
                    sent['pred_heads'][j],
                    sent['gold_deprels'][j],
                    sent['pred_deprels'][j],
                    sent['gold_lemmas'][j],
                    sent['pred_lemmas'][j]
                ])
            
            table = tabulate(comp_rows, headers=comp_headers, tablefmt="grid")
            f.write(table + "\n")

# Função para análise de erros
def analyze_errors(results, output_file="analise_erros_udpipe.txt"):
    """
    Gera um relatório detalhado dos erros cometidos pelo UDPipe.
    """
    error_analysis = {
        'upos_errors': [],
        'xpos_errors': [],
        'lemma_errors': [],
        'dependency_errors': []
    }

    for sent in results:
        sent_id = sent.get('sent_id', 'N/A')
        text = sent['text']
        
        for idx, (token, g_upos, p_upos, g_xpos, p_xpos, g_lemma, p_lemma, 
                 g_head, p_head, g_deprel, p_deprel) in enumerate(zip(
            sent['gold_tokens'], sent['gold_upos'], sent['pred_upos'],
            sent['gold_xpos'], sent['pred_xpos'],
            sent['gold_lemmas'], sent['pred_lemmas'],
            sent['gold_heads'], sent['pred_heads'],
            sent['gold_deprels'], sent['pred_deprels']
        )):
            
            if g_upos != p_upos:
                error_analysis['upos_errors'].append({
                    'sent_id': sent_id, 'text': text, 'token': token,
                    'position': idx + 1, 'gold': g_upos, 'predicted': p_upos
                })
            
            if g_xpos != p_xpos:
                error_analysis['xpos_errors'].append({
                    'sent_id': sent_id, 'text': text, 'token': token,
                    'position': idx + 1, 'gold': g_xpos, 'predicted': p_xpos
                })
            
            if g_lemma != p_lemma:
                error_analysis['lemma_errors'].append({
                    'sent_id': sent_id, 'text': text, 'token': token,
                    'position': idx + 1, 'gold': g_lemma, 'predicted': p_lemma
                })
            
            if g_head != p_head or g_deprel != p_deprel:
                error_analysis['dependency_errors'].append({
                    'sent_id': sent_id, 'text': text, 'token': token,
                    'position': idx + 1, 'gold': (g_head, g_deprel), 
                    'predicted': (p_head, p_deprel)
                })

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== Análise de Erros - UDPipe ===\n\n")

        for error_type, errors in error_analysis.items():
            error_name = error_type.replace('_', ' ').title()
            f.write(f"{error_name} ({len(errors)} erros):\n")
            for error in errors[:50]:  # Limita a 50 erros por tipo para não ficar muito longo
                f.write(f"Sentença ID: {error['sent_id']}\n")
                f.write(f"Texto: {error['text']}\n")
                f.write(f"Token: {error['token']} (Posição: {error['position']})\n")
                f.write(f"Gold: {error['gold']} | Predito: {error['predicted']}\n\n")
            f.write("\n" + "="*50 + "\n\n")

    print(f"Relatório de erros gerado: '{output_file}'")

# Execução principal
if __name__ == "__main__":
    # Criar diretório para arquivos txt se não existir
    txt_dir = "analise_udpipe"
    os.makedirs(txt_dir, exist_ok=True)

    # Configurações
    conllu_file = "/home/andre/Dev-Ubuntu/IC/experimentos/scripts/Corupus/UD_Portuguese-Bosque-master/pt_bosque-ud-test.conllu"  # Caminho para o arquivo CONLL-U
    udpipe_model = "portuguese-bosque-ud-2.5-191206.udpipe"  # Caminho para o modelo UDPipe
    
    print("Iniciando análise com UDPipe...")
    
    # 1. Parsear o arquivo CONLL-U
    print("1. Parseando arquivo CONLL-U...")
    sentences = parse_conllu(conllu_file)
    print(f"   Carregadas {len(sentences)} sentenças")
    
    # 2. Processar com UDPipe
    print("2. Processando com UDPipe...")
    results = evaluate_udpipe(sentences, udpipe_model)
    
    # 3. Calcular métricas
    print("3. Calculando métricas...")
    metrics = calculate_metrics(results)
    
    # 4. Salvar resultados
    print("4. Salvando resultados...")
    save_results_to_file(results, metrics, 
                        output_file=f"{txt_dir}/resultados_completos_udpipe.txt", 
                        max_sentences=len(results))
    
    # 5. Gerar análise de erros
    print("5. Gerando análise de erros...")
    analyze_errors(results, output_file=f"{txt_dir}/analise_erros_udpipe.txt")
    
    # 6. Exibir métricas finais
    print("\n=== MÉTRICAS FINAIS ===")
    print(f"UPOS Accuracy: {metrics['upos_accuracy']:.2%}")
    print(f"XPOS Accuracy: {metrics['xpos_accuracy']:.2%}")
    print(f"Lemma Accuracy: {metrics['lemma_accuracy']:.2%}")
    print(f"UAS: {metrics['uas']:.2%}")
    print(f"LAS: {metrics['las']:.2%}")
    print(f"Token F1: {metrics['token_f1']:.2%}")
    
    print(f"\nProcessamento concluído! Resultados salvos em '{txt_dir}/'")