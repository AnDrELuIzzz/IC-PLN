# Este arquivo implementa um avaliador para diversas tarefas de NLP utilizando múltiplas bibliotecas.
# As funcionalidades principais incluem tokenização com métricas e placeholders para parsing, NER e tradução.
# Também há uma função de normalização que realiza diversas transformações de pré-processamento.

import spacy
import stanza
from flair.data import Sentence as FlairSentence
from transformers import AutoTokenizer, pipeline
from nltk.tokenize import word_tokenize
import nltk
nltk.download('rslp')
from collections import defaultdict
from tabulate import tabulate
import numpy as np
import warnings

# Configurar warnings
warnings.filterwarnings('ignore')

class NLPProcessor:
    def __init__(self, library_name):
        # Inicializa o processador e carrega o modelo específico da biblioteca indicada
        self.lib = library_name
        self.models = {}
        try:
            if self.lib == 'spacy':
                # Carrega o modelo SpaCy para português
                self.models['spacy'] = spacy.load("pt_core_news_lg")
            elif self.lib == 'stanza':
                # Inicializa o pipeline do Stanza para tokenização em português
                self.models['stanza'] = stanza.Pipeline('pt', processors='tokenize', download_method=None)
            elif self.lib == 'flair':
                # Flair não possui tokenizador específico para PT, por isso usa-se o padrão
                pass
            elif self.lib == 'transformers':
                # Carrega o tokenizador do modelo Hugging Face para português
                self.models['transformers_tokenizer'] = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
            elif self.lib == 'nltk':
                # Garante que o pacote 'punkt' esteja disponível para tokenização com o NLTK
                nltk.download('punkt', quiet=True)
                nltk.download('punkt_tab', quiet=True)
        except Exception as e:
            print(f"Erro ao carregar {library_name}: {str(e)}")

    def tokenize(self, text):
        # Realiza a tokenização do texto utilizando a biblioteca selecionada
        try:
            if self.lib == 'spacy':
                doc = self.models['spacy'](text)
                return [token.text for token in doc]
            elif self.lib == 'stanza':
                doc = self.models['stanza'](text)
                return [token.text for sent in doc.sentences for token in sent.tokens]
            elif self.lib == 'flair':
                sentence = FlairSentence(text)
                return [token.text for token in sentence]
            elif self.lib == 'transformers':
                return self.models['transformers_tokenizer'].tokenize(text)
            elif self.lib == 'nltk':
                return word_tokenize(text, language='portuguese')
        except Exception as e:
            print(f"Erro na tokenização com {self.lib}: {str(e)}")
            return []

class Evaluator:
    @staticmethod
    def tokenization_metrics(predictions, gold_standard):
        # Avalia as métricas de tokenização comparando a saída do processador com o gold standard
        aligned = Evaluator.align_tokens(predictions, gold_standard)
        tp = sum(1 for p, g in aligned if p == g)
        fp = sum(1 for p, g in aligned if p != g and p is not None)
        fn = sum(1 for p, g in aligned if p != g and g is not None)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Cálculo de métricas adicionais: número de tokens, média de tokens por sentença e cobertura de contrações
        token_count = len(predictions)
        avg_tokens = token_count / len(gold_standard) if gold_standard else 0
        contraction_coverage = Evaluator.contraction_coverage(predictions)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'error_rate': (fp + fn) / len(gold_standard) if gold_standard else 0,
            'avg_tokens': avg_tokens,
            'contraction_coverage': contraction_coverage
        }

    @staticmethod
    def align_tokens(predicted, gold):
        # Implementa o algoritmo de Needleman-Wunsch para alinhar sequências de tokens
        n = len(predicted)
        m = len(gold)
        dp = np.zeros((n+1, m+1))
        for i in range(n+1):
            dp[i][0] = i
        for j in range(m+1):
            dp[0][j] = j
            
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = 0 if predicted[i-1] == gold[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,       # Deleção
                    dp[i][j-1] + 1,       # Inserção
                    dp[i-1][j-1] + cost   # Substituição
                )
        
        # Backtracking para reconstruir o alinhamento entre os tokens
        i, j = n, m
        aligned = []
        while i > 0 or j > 0:
            if i > 0 and j > 0 and predicted[i-1] == gold[j-1]:
                aligned.insert(0, (predicted[i-1], gold[j-1]))
                i -= 1
                j -= 1
            else:
                if dp[i][j] == dp[i-1][j] + 1:
                    aligned.insert(0, (predicted[i-1], None))
                    i -= 1
                elif dp[i][j] == dp[i][j-1] + 1:
                    aligned.insert(0, (None, gold[j-1]))
                    j -= 1
                else:
                    aligned.insert(0, (predicted[i-1], gold[j-1]))
                    i -= 1
                    j -= 1
        
        return aligned

    @staticmethod
    def contraction_coverage(tokens):
        # Calcula a cobertura de contrações específicas do PT-BR na lista de tokens
        contracoes_ptbr = {'pra', 'pro', 'pra', 'no', 'na', 'dos', 'das', 'dum', 'duns', 'num', 'numa'}
        found = sum(1 for token in tokens if token.lower() in contracoes_ptbr)
        return found / len(contracoes_ptbr) if contracoes_ptbr else 0

def evaluate_all(texts, gold_standards):
    # Processa os textos usando todas as bibliotecas para tokenização e agrega as métricas
    libraries = ['spacy', 'stanza', 'nltk', 'flair', 'transformers']
    results = defaultdict(dict)
    
    for lib in libraries:
        processor = NLPProcessor(lib)
        token_results = []
        for text, gold in zip(texts, gold_standards):
            try:
                tokens = processor.tokenize(text)
                # Avalia as métricas de tokenização comparando com o gold standard
                metrics = Evaluator.tokenization_metrics(tokens, gold)
                token_results.append(metrics)
            except Exception as e:
                print(f"Erro processando {lib}: {str(e)}")
                continue
        
        if token_results:
            # Agrega os valores médios e o desvio padrão de cada métrica
            results[lib] = aggregate_metrics(token_results)
    
    return results

def aggregate_metrics(metrics_list):
    # Consolida as métricas individuais calculando média e desvio padrão para cada métrica
    aggregated = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]
        aggregated[key] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    return aggregated

# NOVAS FUNÇÕES PARA AVALIAÇÃO DE TAREFAS ADICIONAIS

def normalize_text(text):
    # Realiza diversas etapas de normalização no texto para reduzir inconsistências e ruídos
    import re
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import RSLPStemmer

    # Tentativa de usar num2words para converter números em palavras
    try:
        from num2words import num2words
        def num_to_words(num_str):
            try:
                return num2words(int(num_str), lang='pt')
            except:
                return num_str
    except ImportError:
        def num_to_words(num_str):
            return num_str

    # Conversão para minúsculas para uniformidade
    text = text.lower()

    # Expansão de contrações: substitui formas contraídas por suas versões completas
    contractions = {
        "tá": "está",
        "cê": "você",
        "vc": "você",
        "pq": "porque",
        "q": "que",
        "tb": "também"
    }
    for contr, full in contractions.items():
        text = re.sub(r'\b' + re.escape(contr) + r'\b', full, text)

    # Remoção de pontuação e caracteres especiais para limpar o texto
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenização simples realizada pelo split (divisão por espaços)
    tokens = text.split()

    # Remoção de stopwords utilizando a lista do NLTK para português
    stop_words = set(stopwords.words('portuguese'))
    tokens = [t for t in tokens if t not in stop_words]

    # Lematização usando SpaCy, convertendo cada token em sua forma canônica
    try:
        nlp = spacy.load("pt_core_news_lg")
        doc = nlp(" ".join(tokens))
        tokens = [token.lemma_ for token in doc]
    except Exception as e:
        pass

    # Stemming utilizando o RSLPStemmer para reduzir as palavras à sua raiz
    stemmer = RSLPStemmer()
    tokens = [stemmer.stem(t) for t in tokens]

    # Conversão de sequências numéricas para palavras (ex.: "5" → "cinco")
    tokens = [re.sub(r'\d+', lambda x: num_to_words(x.group()), t) for t in tokens]

    return " ".join(tokens)

def evaluate_parsing(texts, gold_parses):
    libraries = ['spacy', 'stanza', 'nltk', 'flair', 'transformers']
    results = {}
    for lib in libraries:
        if lib == 'spacy':
            try:
                nlp = spacy.load("pt_core_news_lg")
                parses = []
                for text in texts:
                    doc = nlp(text)
                    # Representação simplificada: (token, relação, cabeça)
                    parse = [(token.text, token.dep_, token.head.text) for token in doc]
                    parses.append(parse)
                results[lib] = parses
            except Exception as e:
                results[lib] = f"Erro: {e}"
        elif lib == 'stanza':
            try:
                nlp = stanza.Pipeline('pt', processors='tokenize,pos,lemma,depparse')
                parses = []
                for text in texts:
                    doc = nlp(text)
                    parse = []
                    for sentence in doc.sentences:
                        for word in sentence.words:
                            head = sentence.words[int(word.head)-1].text if int(word.head) > 0 else "ROOT"
                            parse.append((word.text, word.deprel, head))
                    parses.append(parse)
                results[lib] = parses
            except Exception as e:
                results[lib] = f"Erro: {e}"
        elif lib == 'nltk':
            try:
                from nltk import pos_tag, RegexpParser, word_tokenize
                # Como pos_tag não suporta bem o português, atribuímos uma tag dummy 'NN' a todos os tokens
                grammar = "NP: {<NN>*}"
                cp = RegexpParser(grammar)
                parses = []
                for text in texts:
                    tokens = word_tokenize(text, language='portuguese')
                    tagged = [(token, 'NN') for token in tokens]
                    tree = cp.parse(tagged)
                    parses.append(tree.pformat(margin=60))
                results[lib] = parses
            except Exception as e:
                results[lib] = f"Erro: {e}"
        elif lib == 'flair':
            try:
                from flair.data import Sentence as FlairSentence
                from flair.models import SequenceTagger
                # Adiciona weights_only=False para evitar erro
                tagger = SequenceTagger.load("ner-multi", weights_only=False)
                parses = []
                for text in texts:
                    sentence = FlairSentence(text)
                    tagger.predict(sentence)
                    spans = [(span.text, span.tag) for span in sentence.get_spans("ner")]
                    parses.append(spans)
                results[lib] = parses
            except Exception as e:
                results[lib] = f"Erro: {e}"
        elif lib == 'transformers':
            # Sem implementação prática de parsing com Transformers
            results[lib] = "Parsing não implementado para esta biblioteca."
    return results

def evaluate_ner(texts, gold_ner):
    libraries = ['spacy', 'stanza', 'nltk', 'flair', 'transformers']
    results = {}
    for lib in libraries:
        if lib == 'spacy':
            try:
                nlp = spacy.load("pt_core_news_lg")
                entities = []
                for text in texts:
                    doc = nlp(text)
                    ents = [(ent.text, ent.label_) for ent in doc.ents]
                    entities.append(ents)
                results[lib] = entities
            except Exception as e:
                results[lib] = f"Erro: {e}"
        elif lib == 'stanza':
            try:
                # Remove download_method e garanta o download se necessário
                stanza.download('pt')
                nlp = stanza.Pipeline('pt', processors='tokenize,ner')
                entities = []
                for text in texts:
                    doc = nlp(text)
                    ents = [(ent.text, ent.type) for ent in doc.entities]
                    entities.append(ents)
                results[lib] = entities
            except Exception as e:
                results[lib] = f"Erro: {e}"
        elif lib == 'nltk':
            try:
                import nltk
                tokens = [nltk.word_tokenize(text, language='portuguese') for text in texts]
                results[lib] = [["NER não implementado"] for _ in texts]
            except Exception as e:
                results[lib] = f"Erro: {e}"
        elif lib == 'flair':
            try:
                from flair.data import Sentence as FlairSentence
                from flair.models import SequenceTagger
                # Adiciona weights_only=False para evitar erro
                tagger = SequenceTagger.load("ner-multi", weights_only=False)
                entities = []
                for text in texts:
                    sentence = FlairSentence(text)
                    tagger.predict(sentence)
                    ents = [(ent.text, ent.get_label("ner").value) for ent in sentence.get_spans("ner")]
                    entities.append(ents)
                results[lib] = entities
            except Exception as e:
                results[lib] = f"Erro: {e}"
        elif lib == 'transformers':
            try:
                # Como alternativa, utiliza-se o pipeline de NER dos Transformers
                ner_pipeline = pipeline("ner", grouped_entities=True)
                entities = []
                for text in texts:
                    ents = ner_pipeline(text)
                    # Ajusta a saída para (texto, label)
                    ents = [(ent['word'], ent['entity_group']) for ent in ents]
                    entities.append(ents)
                results[lib] = entities
            except Exception as e:
                results[lib] = f"Erro: {e}"
    return results

def evaluate_translation(texts, gold_translations):
    libraries = ['spacy', 'stanza', 'nltk', 'flair', 'transformers']
    results = {}
    for lib in libraries:
        try:
            if lib == 'transformers':
                translator = pipeline("translation", model="Helsinki-NLP/opus-mt-pt-en")
                translations = [translator(text)[0]['translation_text'] for text in texts]
            else:
                try:
                    from googletrans import Translator
                    translator = Translator()
                    translations = [translator.translate(text, src='pt', dest='en').text for text in texts]
                except ImportError:
                    translations = ["googletrans não instalado" for _ in texts]
            results[lib] = translations
        except Exception as e:
            results[lib] = f"Erro: {e}"
    return results

def evaluate_all_tasks(texts, gold_tokenizations, gold_parses, gold_ner, gold_translations):
    # Aplica a normalização no texto e executa as avaliações para todas as tarefas
    texts_normalized = [normalize_text(t) for t in texts]
    tokenization_results = evaluate_all(texts_normalized, gold_tokenizations)
    parsing_results = evaluate_parsing(texts_normalized, gold_parses)
    ner_results = evaluate_ner(texts_normalized, gold_ner)
    translation_results = evaluate_translation(texts_normalized, gold_translations)
    
    return {
        'Tokenização': tokenization_results,
        'Parsing': parsing_results,
        'NER': ner_results,
        'Tradução': translation_results
    }

# Exemplo de uso com dados em português
if __name__ == "__main__":
    sample_texts = [
        "Isso é um exemplo de texto com contrações: na, pro, dum.",
        "A linguagem portuguesa tem expressões complexas e diferentes regiões."
    ]
    
    gold_tokenizations = [
        ["isso", "é", "um", "exemplo", "de", "texto", "com", "contrações", ":", "na", ",", "pro", ",", "dum", "."],
        ["a", "linguagem", "portuguesa", "tem", "expressões", "complexas", "e", "diferentes", "regiões", "."]
    ]
    # Placeholders para gold standards das outras tarefas
    gold_parses = [None, None]
    gold_ner = [None, None]
    gold_translations = [None, None]
    
    results = evaluate_all_tasks(sample_texts, gold_tokenizations, gold_parses, gold_ner, gold_translations)
    
    # Exibindo os resultados da Tokenização em tabela
    from tabulate import tabulate
    headers_tok = ["Biblioteca", "F1", "Precisão", "Recall", "Erro", "Tokens/Sent", "Cobertura"]
    table_data_tok = []
    
    for lib, metrics in results['Tokenização'].items():
        row = [
            lib,
            f"{metrics['f1']['mean']:.3f} ± {metrics['f1']['std']:.3f}",
            f"{metrics['precision']['mean']:.3f} ± {metrics['precision']['std']:.3f}",
            f"{metrics['recall']['mean']:.3f} ± {metrics['recall']['std']:.3f}",
            f"{metrics['error_rate']['mean']:.3f} ± {metrics['error_rate']['std']:.3f}",
            f"{metrics['avg_tokens']['mean']:.1f} ± {metrics['avg_tokens']['std']:.1f}",
            f"{metrics['contraction_coverage']['mean']:.1%}"
        ]
        table_data_tok.append(row)
    
    print("Resultados da Tokenização:")
    print(tabulate(table_data_tok, headers=headers_tok, tablefmt="grid", stralign="center", numalign="center"))
    
    # Exibindo resultados para Parsing, NER e Tradução
    print("\nResultados da Análise Sintática (Parsing):")
    for lib, res in results['Parsing'].items():
        print(f"{lib}: {res}")
    
    print("\nResultados de Reconhecimento de Entidades Nomeadas (NER):")
    for lib, res in results['NER'].items():
        print(f"{lib}: {res}")
    
    print("\nResultados de Tradução Automática:")
    for lib, res in results['Tradução'].items():
        print(f"{lib}: {res}")