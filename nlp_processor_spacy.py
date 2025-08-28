import spacy
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Importações para análise
from spacy import displacy
from spacy.tokens import Doc
from tabulate import tabulate
from sklearn.metrics import precision_recall_fscore_support

# Carrega variáveis de ambiente
load_dotenv()

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configurações do processamento obtidas do ambiente"""
    conllu_file: str = os.getenv('CONLLU_FILE', 'pt_bosque-ud-test.conllu')
    wikiner_file: str = os.getenv('WIKINER_FILE', 'aij-wikiner-pt-wp3')
    spacy_model: str = os.getenv('SPACY_MODEL', 'pt_core_news_lg')
    output_dir: str = os.getenv('OUTPUT_DIR', 'analise_spacy2')
    max_workers: int = int(os.getenv('MAX_WORKERS', '4'))
    max_sentences_conllu: Optional[int] = int(os.getenv('MAX_SENTENCES_CONLLU')) if os.getenv('MAX_SENTENCES_CONLLU') else None
    max_sentences_ner: Optional[int] = int(os.getenv('MAX_SENTENCES_NER', '1000'))
    enable_visualizations: bool = os.getenv('ENABLE_VISUALIZATIONS', 'true').lower() == 'true'
    chunk_size: int = int(os.getenv('CHUNK_SIZE', '100'))

class CONLLUProcessor:
    """Processador para arquivos CONLL-U"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.nlp = None
        
    def _load_model(self):
        """Carrega o modelo spaCy uma vez"""
        if self.nlp is None:
            self.nlp = spacy.load(self.config.spacy_model)
        return self.nlp
    
    def parse_conllu(self, file_path: str) -> List[Dict]:
        """Parse otimizado do arquivo CONLL-U"""
        logger.info(f"Parseando arquivo CONLL-U: {file_path}")
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
                        current_sent.update({
                            'tokens': [], 'gold_pos': [], 'gold_heads': [],
                            'gold_deprels': [], 'gold_lemmas': []
                        })
                    
                    parts = line.split('\t')
                    current_sent['tokens'].append(parts[1])
                    current_sent['gold_pos'].append(parts[3])
                    current_sent['gold_heads'].append(int(parts[6]))
                    current_sent['gold_deprels'].append(parts[7])
                    current_sent['gold_lemmas'].append(parts[2])
                    
                elif line == '':
                    if in_sentence and current_sent.get('tokens'):
                        self._calculate_spans(current_sent)
                        sentences.append(current_sent)
                        current_sent = {}
                        in_sentence = False
                        
                        if (self.config.max_sentences_conllu and 
                            len(sentences) >= self.config.max_sentences_conllu):
                            break
                            
        logger.info(f"Parseadas {len(sentences)} sentenças")
        return sentences
    
    def _calculate_spans(self, sent: Dict):
        """Calcula spans dos tokens"""
        text = sent['text']
        current_pos = 0
        spans = []
        
        for token in sent['tokens']:
            start = text.find(token, current_pos)
            end = start + len(token)
            spans.append((start, end))
            current_pos = end
            
        sent['gold_spans'] = spans

def process_sentence_batch(sentences_batch: List[Dict], model_name: str) -> List[Dict]:
    """Processa um lote de sentenças (função para paralelização)"""
    nlp = spacy.load(model_name)
    results = []
    
    for sent in sentences_batch:
        try:
            text = sent['text']
            doc_raw = nlp(text)
            pred_spans = [(token.idx, token.idx + len(token)) for token in doc_raw]
            
            # Cria Doc com tokens gold
            doc_gold = Doc(nlp.vocab, words=sent['tokens'])
            doc_gold = nlp(doc_gold)
            
            result = {
                'sent_id': sent.get('sent_id', ''),
                'text': text,
                'gold_tokens': sent['tokens'],
                'gold_spans': sent['gold_spans'],
                'pred_tokens': [token.text for token in doc_raw],
                'pred_spans': pred_spans,
                'gold_pos': sent['gold_pos'],
                'gold_heads': sent['gold_heads'],
                'gold_deprels': sent['gold_deprels'],
                'gold_lemmas': sent['gold_lemmas'],
                'pred_pos': [token.pos_ for token in doc_gold],
                'pred_heads': [token.head.i + 1 if token.head != token else 0 for token in doc_gold],
                'pred_deprels': [token.dep_ for token in doc_gold],
                'pred_lemmas': [token.lemma_ for token in doc_gold]
            }
            results.append(result)
        except Exception as e:
            logger.error(f"Erro processando sentença {sent.get('sent_id', 'N/A')}: {e}")
            
    return results

class NERProcessor:
    """Processador para NER (WikiNER)"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        
    def read_wikiner(self, path: str) -> List[Tuple[List[str], List[str]]]:
        """Lê arquivo WikiNER de forma otimizada"""
        logger.info(f"Lendo arquivo WikiNER: {path}")
        sentences = []
        
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                tokens, tags = [], []
                
                for tok in line.strip().split():
                    try:
                        word, pos, ner = tok.split("|")
                        tokens.append(word)
                        tags.append(ner)
                    except ValueError:
                        continue
                        
                if tokens:
                    sentences.append((tokens, tags))
                    
                if (self.config.max_sentences_ner and 
                    i >= self.config.max_sentences_ner):
                    break
                    
        logger.info(f"Lidas {len(sentences)} sentenças para NER")
        return sentences
    
    def get_gold_entities(self, tokens: List[str], tags: List[str]) -> List[Tuple[str, str]]:
        """Extrai entidades gold do formato BIO"""
        entities = []
        current_entity, current_label = [], None
        
        for token, tag in zip(tokens, tags):
            if tag == "O":
                if current_entity:
                    entities.append((" ".join(current_entity), current_label))
                    current_entity, current_label = [], None
            else:
                label = tag.split("-")[-1]
                if current_label == label:
                    current_entity.append(token)
                else:
                    if current_entity:
                        entities.append((" ".join(current_entity), current_label))
                    current_entity, current_label = [token], label
                    
        if current_entity:
            entities.append((" ".join(current_entity), current_label))
            
        return entities

def process_ner_batch(sentences_batch: List[Tuple], model_name: str) -> Tuple[List[str], List[str], List[str]]:
    """Processa lote de sentenças para NER"""
    nlp = spacy.load(model_name)
    y_true, y_pred, reports = [], [], []
    
    processor = NERProcessor(ProcessingConfig())
    
    for i, (tokens, tags) in enumerate(sentences_batch):
        try:
            gold_ents = processor.get_gold_entities(tokens, tags)
            text = " ".join(tokens)
            doc = nlp(text)
            pred_ents = [(ent.text, ent.label_) for ent in doc.ents]
            
            # Métricas
            batch_true, batch_pred = [], []
            for ent in gold_ents:
                batch_true.append(ent[1])
                batch_pred.append(ent[1] if ent in pred_ents else "O")
            for ent in pred_ents:
                if ent not in gold_ents:
                    batch_true.append("O")
                    batch_pred.append(ent[1])
                    
            y_true.extend(batch_true)
            y_pred.extend(batch_pred)
            
            # Relatório
            gold_text = [f"{ent[0]} ({ent[1]})" for ent in gold_ents]
            pred_text = [f"{ent[0]} ({ent[1]})" for ent in pred_ents]
            
            report = f"=== Sentença {i+1} ===\n"
            report += f"Texto: {text}\n\n"
            report += f"Gold: {gold_text}\n"
            report += f"Pred: {pred_text}\n\n"
            reports.append(report)
            
        except Exception as e:
            logger.error(f"Erro processando sentença NER {i}: {e}")
            
    return y_true, y_pred, reports

class MetricsCalculator:
    """Calculador de métricas"""
    
    @staticmethod
    def calculate_conllu_metrics(results: List[Dict]) -> Dict[str, float]:
        """Calcula métricas para dados CONLL-U"""
        total_tokens = 0
        pos_correct = lemma_correct = uas_correct = las_correct = 0
        tp_token = fp_token = fn_token = 0
        
        for sent in results:
            n = len(sent['gold_tokens'])
            total_tokens += n
            
            # Acurácia POS e Lemma
            pos_correct += sum(1 for g, p in zip(sent['gold_pos'], sent['pred_pos']) if g == p)
            lemma_correct += sum(1 for g, p in zip(sent['gold_lemmas'], sent['pred_lemmas']) if g == p)
            
            # UAS e LAS
            for g_head, p_head, g_deprel, p_deprel in zip(
                sent['gold_heads'], sent['pred_heads'], 
                sent['gold_deprels'], sent['pred_deprels']
            ):
                if g_head == p_head:
                    uas_correct += 1
                    if g_deprel == p_deprel:
                        las_correct += 1
            
            # Tokenização
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

class ResultsManager:
    """Gerenciador de resultados"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def save_conllu_results(self, results: List[Dict], metrics: Dict[str, float]):
        """Salva resultados CONLL-U"""
        output_file = self.output_dir / "resultados_conllu.txt"
        
        # Tabela de métricas
        metric_rows = [
            ["Acurácia de POS", f"{metrics['pos_accuracy']:.2%}"],
            ["Acurácia de Lemmas", f"{metrics['lemma_accuracy']:.2%}"],
            ["UAS", f"{metrics['uas']:.2%}"],
            ["LAS", f"{metrics['las']:.2%}"],
            ["Precisão em Tokenização", f"{metrics['token_precision']:.2%}"],
            ["Recall em Tokenização", f"{metrics['token_recall']:.2%}"],
            ["F1-Score em Tokenização", f"{metrics['token_f1']:.2%}"]
        ]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=== Métricas CONLL-U ===\n")
            f.write(tabulate(metric_rows, headers=["Métrica", "Valor"], tablefmt="grid"))
            f.write("\n\n")
            
            # Detalhes das sentenças
            for i, sent in enumerate(results):
                f.write(f"\n=== Sentença {i+1} ({sent['sent_id']}) ===\n")
                f.write(f"Texto: {sent['text']}\n\n")
                
                # Tabela comparativa
                comp_headers = ["Token", "Gold POS", "spaCy POS", "Gold HEAD", 
                              "spaCy HEAD", "Gold DEPREL", "spaCy DEPREL", 
                              "Gold Lemma", "spaCy Lemma"]
                comp_rows = []
                
                for j in range(len(sent['gold_tokens'])):
                    comp_rows.append([
                        sent['gold_tokens'][j], sent['gold_pos'][j], sent['pred_pos'][j],
                        sent['gold_heads'][j], sent['pred_heads'][j],
                        sent['gold_deprels'][j], sent['pred_deprels'][j],
                        sent['gold_lemmas'][j], sent['pred_lemmas'][j]
                    ])
                
                f.write(tabulate(comp_rows, headers=comp_headers, tablefmt="grid"))
                f.write("\n")
        
        logger.info(f"Resultados CONLL-U salvos em: {output_file}")
    
    def save_ner_results(self, y_true: List[str], y_pred: List[str], reports: List[str]):
        """Salva resultados NER"""
        output_file = self.output_dir / "resultado_NER.txt"
        
        # Calcula métricas
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="micro", labels=["PER", "LOC", "ORG", "MISC"],
            zero_division=0
        )
        
        metricas = [
            ["Precisão (NER)", f"{precision:.2%}"],
            ["Recall (NER)", f"{recall:.2%}"],
            ["F1 (NER)", f"{f1:.2%}"],
        ]
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=== Métricas NER ===\n")
            f.write(tabulate(metricas, headers=["Métrica", "Valor"], tablefmt="grid"))
            f.write("\n\n")
            f.write("\n".join(reports))
        
        logger.info(f"Resultados NER salvos em: {output_file}")

def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Divide uma lista em chunks"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def main():
    """Função principal"""
    config = ProcessingConfig()
    logger.info("Iniciando processamento NLP unificado")
    logger.info(f"Configuração: {config}")
    
    results_manager = ResultsManager(config)
    
    # Processamento CONLL-U com paralelismo
    if os.path.exists(config.conllu_file):
        logger.info("=== Processando CONLL-U ===")
        conllu_processor = CONLLUProcessor(config)
        sentences = conllu_processor.parse_conllu(config.conllu_file)
        
        if sentences:
            # Divide em chunks para paralelização
            sentence_chunks = chunk_list(sentences, config.chunk_size)
            
            with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
                process_func = partial(process_sentence_batch, model_name=config.spacy_model)
                all_results = []
                
                for chunk_results in executor.map(process_func, sentence_chunks):
                    all_results.extend(chunk_results)
            
            # Calcula métricas e salva resultados
            metrics = MetricsCalculator.calculate_conllu_metrics(all_results)
            results_manager.save_conllu_results(all_results, metrics)
            
            logger.info("Processamento CONLL-U concluído")
    else:
        logger.warning(f"Arquivo CONLL-U não encontrado: {config.conllu_file}")
    
    # Processamento NER com paralelismo
    if os.path.exists(config.wikiner_file):
        logger.info("=== Processando NER ===")
        ner_processor = NERProcessor(config)
        sentences = ner_processor.read_wikiner(config.wikiner_file)
        
        if sentences:
            # Divide em chunks para paralelização
            sentence_chunks = chunk_list(sentences, config.chunk_size)
            
            with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
                process_func = partial(process_ner_batch, model_name=config.spacy_model)
                all_y_true, all_y_pred, all_reports = [], [], []
                
                for y_true, y_pred, reports in executor.map(process_func, sentence_chunks):
                    all_y_true.extend(y_true)
                    all_y_pred.extend(y_pred)
                    all_reports.extend(reports)
            
            results_manager.save_ner_results(all_y_true, all_y_pred, all_reports)
            logger.info("Processamento NER concluído")
    else:
        logger.warning(f"Arquivo WikiNER não encontrado: {config.wikiner_file}")
    
    logger.info("Processamento completo finalizado!")

if __name__ == "__main__":
    main()