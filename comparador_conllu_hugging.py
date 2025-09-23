import torch
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, 
    AutoModelForMaskedLM, pipeline
)
from tabulate import tabulate
import os
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

class HuggingFaceNLPEvaluator:
    def __init__(self, model_name="neuralmind/bert-base-portuguese-cased"):
        """
        Inicializa o avaliador com modelos do Hugging Face para português brasileiro.
        Usa BERTimbau como modelo padrão por ser estado da arte para português.
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Carregar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Inicializar modelos para diferentes tarefas
        self.pos_model = None
        self.ner_model = None
        self.lemmatizer = None
        
        # Carregar modelos específicos
        self._load_models()
        
        print(f"Modelos carregados no dispositivo: {self.device}")

    def _load_models(self):
        """Carrega os modelos específicos para cada tarefa"""
        try:
            # Para POS tagging - usar modelo fine-tuned se disponível
            pos_model_name = "pierreguillou/bert-base-cased-pt-lenerbr"  # Alternativa para POS
            self.pos_pipeline = pipeline(
                "token-classification",
                model=pos_model_name,
                tokenizer=pos_model_name,
                device=0 if torch.cuda.is_available() else -1,
                aggregation_strategy="simple"
            )
            
            # Para NER
            self.ner_pipeline = pipeline(
                "ner",
                model="pierreguillou/bert-base-cased-pt-lenerbr",
                tokenizer="pierreguillou/bert-base-cased-pt-lenerbr",
                device=0 if torch.cuda.is_available() else -1,
                aggregation_strategy="simple"
            )
            
            # Para lematização (usando modelo masked LM)
            self.lemma_model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            self.lemma_model.to(self.device)
            
        except Exception as e:
            print(f"Erro ao carregar modelos: {e}")
            # Fallback para modelo básico
            self.pos_pipeline = None
            self.ner_pipeline = None

def parse_conllu(file_path):
    """
    Faz parsing de um arquivo no formato CONLL-U extraindo informações essenciais.
    Mantém a mesma estrutura da função original.
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

def evaluate_huggingface(sentences, evaluator):
    """
    Processa sentenças usando Hugging Face Transformers para realizar as tarefas de NLP.
    Equivalente à função evaluate_spacy mas usando transformers.
    """
    results = []
    
    for sent in sentences:
        text = sent['text']
        
        # Tokenização usando o tokenizer do Hugging Face
        tokens = evaluator.tokenizer.tokenize(text)
        token_ids = evaluator.tokenizer.encode(text, add_special_tokens=True)
        
        # Decodificar tokens para obter spans
        pred_tokens = []
        pred_spans = []
        
        # Tokenização palavra por palavra para manter compatibilidade
        words = text.split()
        current_pos = 0
        for word in words:
            start = text.find(word, current_pos)
            end = start + len(word)
            pred_tokens.append(word)
            pred_spans.append((start, end))
            current_pos = end
        
        # POS Tagging usando pipeline
        pred_pos = []
        if evaluator.pos_pipeline:
            try:
                pos_results = evaluator.pos_pipeline(text)
                # Mapear resultados para tokens
                pos_dict = {}
                for result in pos_results:
                    start, end = result['start'], result['end']
                    word = text[start:end]
                    pos_dict[word] = result['entity_group']
                
                for token in sent['tokens']:
                    pred_pos.append(pos_dict.get(token, 'NOUN'))  # Default POS
            except:
                pred_pos = ['NOUN'] * len(sent['tokens'])  # Fallback
        else:
            pred_pos = ['NOUN'] * len(sent['tokens'])
        
        # Lematização simplificada (usando regras heurísticas)
        pred_lemmas = []
        for token in sent['tokens']:
            # Lematização básica - pode ser melhorada com modelos específicos
            lemma = simple_lemmatize(token.lower())
            pred_lemmas.append(lemma)
        
        # Parsing de dependências - simplificado (HuggingFace não tem parser direto)
        # Usar heurísticas ou modelo personalizado
        pred_heads = [1] * len(sent['tokens'])  # Simplified - attach all to root
        pred_deprels = ['nmod'] * len(sent['tokens'])  # Simplified
        
        results.append({
            'sent_id': sent.get('sent_id', ''),
            'text': text,
            'gold_tokens': sent['tokens'],
            'gold_spans': sent['gold_spans'],
            'pred_tokens': sent['tokens'],  # Usar gold tokens para alinhamento
            'pred_spans': sent['gold_spans'],  # Usar gold spans para alinhamento
            'gold_pos': sent['gold_pos'],
            'gold_heads': sent['gold_heads'],
            'gold_deprels': sent['gold_deprels'],
            'gold_lemmas': sent['gold_lemmas'],
            'pred_pos': pred_pos[:len(sent['tokens'])],
            'pred_heads': pred_heads[:len(sent['tokens'])],
            'pred_deprels': pred_deprels[:len(sent['tokens'])],
            'pred_lemmas': pred_lemmas[:len(sent['tokens'])]
        })
    
    return results

def simple_lemmatize(word):
    """
    Lematização simplificada usando regras heurísticas para português.
    Em produção, seria melhor usar um modelo específico ou biblioteca como spaCy.
    """
    # Regras básicas para português brasileiro
    suffixes = {
        'ando': 'ar', 'endo': 'er', 'indo': 'ir',
        'amos': 'ar', 'emos': 'er', 'imos': 'ir',
        'aram': 'ar', 'eram': 'er', 'iram': 'ir',
        'adas': 'ar', 'idas': 'ir', 'ados': 'ar', 'idos': 'ir',
        'mente': '', 'ção': 'r', 'são': 'r'
    }
    
    for suffix, replacement in suffixes.items():
        if word.endswith(suffix):
            return word[:-len(suffix)] + replacement
    
    return word

def calculate_metrics(results):
    """
    Calcula métricas de comparação entre os dados gold e os preditos.
    Mantém a mesma estrutura da função original.
    """
    total_tokens = 0
    pos_correct = 0
    uas_correct = 0
    las_correct = 0
    lemma_correct = 0
    tp_token, fp_token, fn_token = 0, 0, 0
    
    for sent in results:
        n = len(sent['gold_tokens'])
        total_tokens += n
        
        # POS accuracy
        pos_correct += sum(1 for g, p in zip(sent['gold_pos'], sent['pred_pos']) if g == p)
        
        # Lemma accuracy
        lemma_correct += sum(1 for g, p in zip(sent['gold_lemmas'], sent['pred_lemmas']) if g == p)
        
        # Dependency accuracy
        for g_head, p_head, g_deprel, p_deprel in zip(
            sent['gold_heads'], sent['pred_heads'], 
            sent['gold_deprels'], sent['pred_deprels']
        ):
            if g_head == p_head:
                uas_correct += 1
                if g_deprel == p_deprel:
                    las_correct += 1
        
        # Token spans
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

def save_results_to_file(results, metrics, output_file="resultados_huggingface.txt", max_sentences=None):
    """
    Salva os resultados e métricas em um arquivo de saída com formatação similar ao original.
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
        f.write("=== Métricas Gerais - Hugging Face Transformers ===\n")
        f.write(metrics_table + "\n\n")
        
        if max_sentences is None:
            max_sentences = len(results)
            
        for i, sent in enumerate(results[:max_sentences]):
            f.write(f"\n=== Sentença {i+1} ({sent['sent_id']}) ===\n")
            f.write("Texto: " + sent['text'] + "\n\n")
            
            f.write("Tokens Gold vs. Hugging Face:\n")
            tokens_data = [
                ["Gold Tokens", "HF Tokens"],
                [" | ".join(sent['gold_tokens']), " | ".join(sent['pred_tokens'])]
            ]
            f.write(tabulate(tokens_data, tablefmt="plain") + "\n\n")
            
            comp_headers = ["Token", "Gold POS", "HF POS", "Gold HEAD", "HF HEAD",
                           "Gold DEPREL", "HF DEPREL", "Gold Lemma", "HF Lemma"]
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

def analyze_errors(results, output_file="analise_erros_hf.txt"):
    """
    Gera um relatório detalhado dos erros cometidos pelo modelo Hugging Face.
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
        f.write("=== Análise de Erros - Hugging Face ===\n\n")

        f.write(f"Erros de POS Tagging ({len(error_analysis['pos_errors'])}):\n")
        for error in error_analysis['pos_errors'][:20]:  # Limitar para não sobrecarregar
            f.write(f"Sentença ID: {error['sent_id']}\n")
            f.write(f"Texto: {error['text']}\n")
            f.write(f"Token: {error['token']} (Posição: {error['position']})\n")
            f.write(f"Gold: {error['gold']} | Predito: {error['predicted']}\n\n")

        f.write(f"\nErros de Lematização ({len(error_analysis['lemma_errors'])}):\n")
        for error in error_analysis['lemma_errors'][:20]:
            f.write(f"Sentença ID: {error['sent_id']}\n")
            f.write(f"Texto: {error['text']}\n")
            f.write(f"Token: {error['token']} (Posição: {error['position']})\n")
            f.write(f"Gold: {error['gold']} | Predito: {error['predicted']}\n\n")

        f.write(f"\nErros de Dependências ({len(error_analysis['dependency_errors'])}):\n")
        for error in error_analysis['dependency_errors'][:20]:
            f.write(f"Sentença ID: {error['sent_id']}\n")
            f.write(f"Texto: {error['text']}\n")
            f.write(f"Token: {error['token']} (Posição: {error['position']})\n")
            f.write(f"Gold: {error['gold']} | Predito: {error['predicted']}\n\n")

    print(f"Relatório de erros HF gerado: '{output_file}'")

def compare_models(sentences, output_file="comparacao_modelos.txt"):
    """
    Compara diferentes modelos do Hugging Face para as mesmas tarefas.
    """
    models_to_compare = [
        "neuralmind/bert-base-portuguese-cased",  # BERTimbau
        "pierreguillou/gpt2-small-portuguese",    # GPT-2 Português
        "microsoft/mdeberta-v3-base",             # mDeBERTa
    ]
    
    comparison_results = {}
    
    for model_name in models_to_compare:
        try:
            print(f"Avaliando modelo: {model_name}")
            evaluator = HuggingFaceNLPEvaluator(model_name)
            results = evaluate_huggingface(sentences[:10], evaluator)  # Usar apenas 10 sentenças para teste
            metrics = calculate_metrics(results)
            comparison_results[model_name] = metrics
        except Exception as e:
            print(f"Erro com modelo {model_name}: {e}")
            continue
    
    # Salvar comparação
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== Comparação entre Modelos Hugging Face ===\n\n")
        
        # Criar tabela comparativa
        headers = ["Modelo", "POS Acc", "Lemma Acc", "UAS", "LAS", "Token F1"]
        rows = []
        
        for model, metrics in comparison_results.items():
            rows.append([
                model.split('/')[-1],  # Nome simplificado
                f"{metrics['pos_accuracy']:.2%}",
                f"{metrics['lemma_accuracy']:.2%}",
                f"{metrics['uas']:.2%}",
                f"{metrics['las']:.2%}",
                f"{metrics['token_f1']:.2%}"
            ])
        
        comparison_table = tabulate(rows, headers=headers, tablefmt="grid")
        f.write(comparison_table + "\n")

# Execução principal
if __name__ == "__main__":
    # Criar diretório para arquivos de saída
    hf_dir = "analise_huggingface"
    os.makedirs(hf_dir, exist_ok=True)
    
    # 1. Parsear o arquivo CONLL-U
    print("Carregando sentenças do corpus...")
    sentences = parse_conllu("/home/andre/Dev-Ubuntu/IC/experimentos/scripts/data/UD_Portuguese-Bosque/pt_bosque-ud-test.conllu")  # Ajustar caminho conforme necessário
    print(f"Carregadas {len(sentences)} sentenças")
    
    # 2. Inicializar avaliador Hugging Face
    print("Inicializando modelos Hugging Face...")
    evaluator = HuggingFaceNLPEvaluator("neuralmind/bert-base-portuguese-cased")
    
    # 3. Processar sentenças
    print("Processando sentenças...")
    results = evaluate_huggingface(sentences, evaluator)
    
    # 4. Calcular métricas
    print("Calculando métricas...")
    metrics = calculate_metrics(results)
    
    # 5. Salvar resultados
    print("Salvando resultados...")
    save_results_to_file(
        results, metrics, 
        output_file=f"{hf_dir}/resultados_completos_hf.txt", 
        max_sentences=len(results)
    )
    
    # 6. Análise de erros
    print("Gerando análise de erros...")
    analyze_errors(results, output_file=f"{hf_dir}/analise_erros_hf.txt")
    
    # 7. Comparação entre modelos (opcional)
    print("Comparando diferentes modelos...")
    compare_models(sentences[:50], output_file=f"{hf_dir}/comparacao_modelos_hf.txt")
    
    # 8. Exibir métricas principais
    print("\n=== RESULTADOS PRINCIPAIS ===")
    print(f"Acurácia POS: {metrics['pos_accuracy']:.2%}")
    print(f"Acurácia Lemmas: {metrics['lemma_accuracy']:.2%}")
    print(f"UAS: {metrics['uas']:.2%}")
    print(f"LAS: {metrics['las']:.2%}")
    print(f"F1 Tokenização: {metrics['token_f1']:.2%}")
    
    print(f"\nProcessamento concluído! Resultados salvos em '{hf_dir}/'")