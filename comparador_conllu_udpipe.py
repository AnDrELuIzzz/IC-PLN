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
                    current_sent['gold_pos'] = []  # Mudança: usa POS em vez de separar UPOS/XPOS
                    current_sent['gold_heads'] = []
                    current_sent['gold_deprels'] = []
                    current_sent['gold_lemmas'] = []
                
                parts = line.split('\t')
                current_sent['tokens'].append(parts[1])
                current_sent['gold_lemmas'].append(parts[2])
                current_sent['gold_pos'].append(parts[3])  # UPOS como POS padrão
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
    pos_tags = []  # Mudança: apenas POS (UPOS)
    heads = []
    deprels = []
    spans = []
    
    for line in lines:
        if line and not line.startswith('#') and line[0].isdigit():
            parts = line.split('\t')
            if '-' not in parts[0]:  # Ignora tokens multi-word
                tokens.append(parts[1])
                lemmas.append(parts[2])
                pos_tags.append(parts[3])  # UPOS como POS padrão
                heads.append(int(parts[6]))
                deprels.append(parts[7])
                # Para spans, assumimos posições sequenciais (UDPipe não fornece spans exatos)
                start = len(' '.join(tokens[:-1])) + (1 if len(tokens) > 1 else 0)
                end = start + len(parts[1])
                spans.append((start, end))
    
    return {
        'tokens': tokens,
        'lemmas': lemmas,
        'pos': pos_tags,  # Mudança: chave simplificada
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
            'gold_pos': sent['gold_pos'],  # Mudança: POS unificado
            'gold_heads': sent['gold_heads'],
            'gold_deprels': sent['gold_deprels'],
            'gold_lemmas': sent['gold_lemmas'],
            'pred_pos': pred_data['pos'],  # Mudança: POS unificado
            'pred_heads': pred_data['heads'],
            'pred_deprels': pred_data['deprels'],
            'pred_lemmas': pred_data['lemmas']
        })
    
    return results

# Função que calcula métricas de avaliação - PADRONIZADA COM SPACY
def calculate_metrics(results):
    """
    Calcula métricas de comparação entre os dados gold e os preditos pelo UDPipe.
    PADRONIZADO: Usa as mesmas métricas do spaCy (POS, Lemma, UAS, LAS, Token F1).
    """
    total_tokens = 0
    pos_correct = 0  # Mudança: nome padronizado
    lemma_correct = 0
    uas_correct = 0
    las_correct = 0
    tp_token, fp_token, fn_token = 0, 0, 0
    
    for sent in results:
        n = len(sent['gold_tokens'])
        total_tokens += n
        
        # Calcular acurácia para cada tarefa
        pos_correct += sum(1 for g, p in zip(sent['gold_pos'], sent['pred_pos']) if g == p)
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
    
    # RETORNO PADRONIZADO - MESMAS CHAVES DO SPACY
    return {
        'pos_accuracy': pos_correct / total_tokens,  # Mudança: nome padronizado
        'lemma_accuracy': lemma_correct / total_tokens,
        'uas': uas_correct / total_tokens,
        'las': las_correct / total_tokens,
        'token_precision': precision,
        'token_recall': recall,
        'token_f1': f1
    }

# Função que salva os resultados e métricas em um arquivo de saída - PADRONIZADA
def save_results_to_file(results, metrics, output_file="resultados_udpipe.txt", max_sentences=None):
    """
    Registra as métricas gerais e detalhes de cada sentença para análise posterior.
    PADRONIZADO: Usa a mesma formatação e ordem de métricas do spaCy.
    """
    # MÉTRICAS PADRONIZADAS - MESMA ORDEM DO SPACY
    metric_rows = [
        ["Acurácia de POS", f"{metrics['pos_accuracy']:.2%}"],  # Mudança: nome padronizado
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
            # Verificar se os comprimentos coincidem
            if len(sent['gold_tokens']) != len(sent['pred_tokens']):
                f.write(f"\n=== Sentença {i+1} ({sent['sent_id']}) ===\n")
                f.write("Texto: " + sent['text'] + "\n")
                f.write("Aviso: Comprimentos de tokens gold e previstos não coincidem. Pulando detalhes.\n\n")
                continue
            
            f.write(f"\n=== Sentença {i+1} ({sent['sent_id']}) ===\n")
            f.write("Texto: " + sent['text'] + "\n\n")
            
            f.write("Tokens Gold vs. UDPipe:\n")
            tokens_data = [
                ["Gold Tokens", "UDPipe Tokens"],
                [" | ".join(sent['gold_tokens']), " | ".join(sent['pred_tokens'])]
            ]
            f.write(tabulate(tokens_data, tablefmt="plain") + "\n\n")
            
            # Tabela detalhada de comparação - PADRONIZADA COM SPACY
            comp_headers = ["Token", "Gold POS", "UDPipe POS", "Gold HEAD", "UDPipe HEAD",
                           "Gold DEPREL", "UDPipe DEPREL", "Gold Lemma", "UDPipe Lemma"]
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

# Função para análise de erros - PADRONIZADA
def analyze_errors(results, output_file="analise_erros_udpipe.txt"):
    """
    Gera um relatório detalhado dos erros cometidos pelo UDPipe.
    PADRONIZADO: Usa a mesma estrutura de análise de erros do spaCy.
    """
    error_analysis = {
        'pos_errors': [],  # Mudança: nome padronizado
        'lemma_errors': [],
        'dependency_errors': []
    }

    for sent in results:
        sent_id = sent.get('sent_id', 'N/A')
        text = sent['text']
        
        for idx, (token, g_pos, p_pos, g_lemma, p_lemma, 
                 g_head, p_head, g_deprel, p_deprel) in enumerate(zip(
            sent['gold_tokens'], sent['gold_pos'], sent['pred_pos'],
            sent['gold_lemmas'], sent['pred_lemmas'],
            sent['gold_heads'], sent['pred_heads'],
            sent['gold_deprels'], sent['pred_deprels']
        )):
            
            if g_pos != p_pos:
                error_analysis['pos_errors'].append({
                    'sent_id': sent_id, 'text': text, 'token': token,
                    'position': idx + 1, 'gold': g_pos, 'predicted': p_pos
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

        # FORMATO PADRONIZADO - MESMO DO SPACY
        error_types = [
            ('pos_errors', 'Erros de POS Tagging'),
            ('lemma_errors', 'Erros de Lematização'),
            ('dependency_errors', 'Erros de Dependências')
        ]

        for error_type, error_name in error_types:
            errors = error_analysis[error_type]
            f.write(f"{error_name} ({len(errors)} erros):\n")
            for error in errors[:50]:  # Limita a 50 erros por tipo
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
    
    # 6. Exibir métricas finais - PADRONIZADO COM SPACY
    print("\n=== MÉTRICAS FINAIS ===")
    print(f"POS Accuracy: {metrics['pos_accuracy']:.2%}")  # Mudança: nome padronizado
    print(f"Lemma Accuracy: {metrics['lemma_accuracy']:.2%}")
    print(f"UAS: {metrics['uas']:.2%}")
    print(f"LAS: {metrics['las']:.2%}")
    print(f"Token F1: {metrics['token_f1']:.2%}")
    
    print(f"\nProcessamento concluído! Resultados salvos em '{txt_dir}/'")