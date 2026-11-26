"""
[ENSEMBLE] Estratégias de predição sobre a MESMA KB e o MESMO test set do controle.

    single    -> controle: 1 LLM, top-k do retrieval (RAGPipeline.produce_answer, sem mudança).
    knn       -> baseline sem LLM: voto majoritário nos rótulos ("Next activity") dos top-k recuperados.
    ensemble  -> n LLMs + voto majoritário. Dois modos de contexto:
                   shared:    todos recebem os MESMOS top-k documentos (isola o ganho do ensemble).
                   partition: busca k * n documentos e distribui em rodízio (LLM i recebe os ranks
                              i, i+n, i+2n, ...): cada LLM vê documentos diferentes, de qualidade parecida.
                   bagging:   busca um pool top-K UMA vez e sorteia, para cada LLM, k documentos COM
                              reposição (bootstrap). Mesmo pool do partition (K = k * n por padrão): os dois
                              modos diferem só em "fatia fixa" vs "sorteio". É bagging do CONTEXTO recuperado,
                              não da KB (Breiman, 1996, reamostra o conjunto de treino; aqui não há treino).

Todas devolvem um Prediction. A avaliação continua no VerificationOracle: knn e ensemble entregam a
classe já decidida (already_parsed=True); single entrega o texto cru do LLM, como antes.

O módulo não importa pipeline.py (torch/transformers): os membros do ensemble só precisam ter
produce_answer_with_context(question, context, info_run) -> (prompt, answer). Isso deixa os testes leves.
"""
import hashlib
import random
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import vector_store as vs
from errors import QuotaExhausted
from oracle import WRONG_FORMAT, is_correct, parse_model_answer


def _usage_of(pipelines) -> Dict:
    # [GROQ] Tokens gastos por modelo (uma entrada por instância; ids repetidos compartilham a instância)
    report, seen = {}, set()
    for name, pipe in pipelines:
        if id(pipe) in seen or not hasattr(pipe, 'usage'):
            continue
        seen.add(id(pipe))
        report[name] = dict(pipe.usage)
    return report


@dataclass
class Prediction:
    prompt: str                 # o que é gravado como "Prompt" no relatório
    answer: str                 # texto cru do LLM (single) ou classe decidida (knn/ensemble)
    already_parsed: bool        # True -> o oracle não parseia de novo
    details: Optional[Dict] = field(default=None)


def majority_vote(labels: Sequence[str], weights: Optional[Sequence[float]] = None) -> Tuple[Optional[str], Dict]:
    """
    Voto majoritário (ou ponderado, se weights for dado). Ignora None e WRONG_FORMAT.
    Desempate: vence o rótulo que aparece PRIMEIRO em `labels` (= melhor rank no kNN; ordem dos
    membros no ensemble). Determinístico, sem sorteio.
    Devolve (vencedor ou None, {rótulo: total de votos}).
    """
    totals = OrderedDict()
    for i, label in enumerate(labels):
        if label is None or label == WRONG_FORMAT:
            continue
        totals[label] = totals.get(label, 0.0) + (1.0 if weights is None else float(weights[i]))
    if not totals:
        return None, {}
    winner = max(totals, key=totals.get)   # max() devolve o primeiro empatado na ordem de inserção
    return winner, dict(totals)


class SinglePredictor:
    """Controle: exatamente o fluxo antigo (um LLM, top-k)."""
    name = 'single'

    def __init__(self, pipeline_instance, vect_db, num_docs: int):
        self.pipeline = pipeline_instance
        self.vect_db = vect_db
        self.num_docs = num_docs

    def predict(self, prefix: str, info_run: Dict) -> Prediction:
        prompt, answer = self.pipeline.produce_answer(prefix, self.vect_db, self.num_docs, info_run)
        return Prediction(prompt, answer, already_parsed=False)

    def usage_report(self) -> Dict:
        return _usage_of([(getattr(self.pipeline, 'model_id', 'llm'), self.pipeline)])


class KNNPredictor:
    """
    Baseline sem LLM: recupera os k vizinhos e vota nos rótulos deles.
    weighting='count': 1 voto por documento. weighting='score': voto = similaridade de cosseno.
    É o complemento do --rag False: aquele mede o LLM sem retrieval, este mede o retrieval sem LLM.
    """
    name = 'knn'

    def __init__(self, vect_db, k: int = 10, weighting: str = 'count'):
        if weighting not in ('count', 'score'):
            raise ValueError("weighting deve ser 'count' ou 'score'")
        self.vect_db = vect_db
        self.k = k
        self.weighting = weighting

    def predict(self, prefix: str, info_run: Dict) -> Prediction:
        docs = vs.retrieve_documents(self.vect_db, prefix, self.k)
        labels = [d['label'] for d in docs]
        weights = [d['score'] for d in docs] if self.weighting == 'score' else None
        winner, votes = majority_vote(labels, weights)
        details = {'strategy': 'knn', 'k': self.k, 'weighting': self.weighting,
                   'retrieved_labels': labels, 'votes': votes}
        return Prediction(prompt=f'[kNN k={self.k} weighting={self.weighting}] {prefix}',
                          answer=winner if winner is not None else WRONG_FORMAT,
                          already_parsed=True, details=details)


def split_round_robin(docs: List[Dict], n: int) -> List[List[Dict]]:
    """Fatia i recebe os documentos de posição i, i+n, i+2n, ... (ranks parecidos entre as fatias)."""
    return [docs[i::n] for i in range(n)]


def bag_rng(seed: int, prefix: str) -> random.Random:
    """
    [BAGGING] Gerador próprio por prefixo, derivado de sha256(seed | prefixo).
    Por que não o random global: o sorteio de um prefixo não depende da ordem de execução nem de quantos
    prefixos vieram antes. Retomar do checkpoint, rodar outra estratégia antes ou reordenar o test set
    produz exatamente as mesmas bags. (hash() do Python é salgado por processo: não serve.)
    Efeito colateral: o mesmo prefixo repetido no test set recebe as mesmas bags (sorteio idêntico).
    """
    digest = hashlib.sha256(f'{seed}|{prefix}'.encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def bootstrap_bags(docs: List[Dict], n_bags: int, bag_size: int, rng: random.Random,
                   replace: bool = True) -> List[List[Dict]]:
    """
    [BAGGING] n_bags amostras de bag_size documentos do pool `docs`.
      replace=True  -> bootstrap (com reposição): bagging. Um documento pode aparecer 2x na mesma bag;
                       ele entra repetido no prompt, que é o equivalente a peso 2 no bootstrap clássico.
      replace=False -> sem reposição (pasting / subamostragem). Útil como ablação.
    Dentro de cada bag os documentos ficam na ordem do rank (melhor primeiro), como no controle: o que
    muda entre as bags é QUAIS documentos entram, não a ordem de apresentação.
    """
    if not docs:
        return [[] for _ in range(n_bags)]
    bags = []
    for _ in range(n_bags):
        if replace:
            idx = [rng.randrange(len(docs)) for _ in range(bag_size)]
        else:
            idx = rng.sample(range(len(docs)), min(bag_size, len(docs)))
        bags.append([docs[i] for i in sorted(idx)])
    return bags


class LLMEnsemble:
    """
    n LLMs + voto majoritário sobre as respostas parseadas (mesmo parsing do oracle).
    members: lista de (nome, pipeline). Repetir o mesmo nome é permitido (mesmo modelo com
             contextos diferentes, no modo partition); quem monta a lista deve reutilizar a instância.
    docs_per_llm: documentos no contexto de CADA LLM (no partition, a busca total é docs_per_llm * n).
    [BAGGING] pool_size: tamanho do pool top-K do qual as bags são sorteadas (default docs_per_llm * n,
              o mesmo do partition). bag_replacement: True = bootstrap; False = sem reposição.
              bag_seed: semente do sorteio (ver bag_rng).
    Uma resposta fora do formato ou um erro de API de um membro vira voto nulo, sem derrubar a run.
    """
    name = 'ensemble'
    MODES = ('shared', 'partition', 'bagging')

    def __init__(self, members: List[Tuple[str, object]], vect_db, docs_per_llm: int = 3, mode: str = 'partition',
                 pool_size: Optional[int] = None, bag_replacement: bool = True, bag_seed: int = 10):
        if not members:
            raise ValueError('O ensemble precisa de pelo menos 1 membro.')
        if mode not in self.MODES:
            raise ValueError(f"mode deve ser um de {self.MODES}")
        self.members = members
        self.vect_db = vect_db
        self.docs_per_llm = docs_per_llm
        self.mode = mode
        # [BAGGING]
        self.pool_size = pool_size if pool_size is not None else docs_per_llm * len(members)
        if mode == 'bagging' and self.pool_size < 1:
            raise ValueError('pool_size deve ser >= 1')
        if mode == 'bagging' and not bag_replacement and self.pool_size < docs_per_llm:
            raise ValueError('Sem reposição, pool_size precisa ser >= docs_per_llm')
        self.bag_replacement = bag_replacement
        self.bag_seed = bag_seed

    def retrieved_per_prefix(self) -> int:
        """Documentos buscados no índice por prefixo (vai para o run_info)."""
        return {'shared': self.docs_per_llm, 'partition': self.docs_per_llm * len(self.members),
                'bagging': self.pool_size}[self.mode]

    def contexts_for(self, prefix: str) -> Tuple[List[Dict], List[List[Dict]]]:
        n = len(self.members)
        if self.mode == 'shared':
            docs = vs.retrieve_documents(self.vect_db, prefix, self.docs_per_llm)
            return docs, [docs] * n
        if self.mode == 'bagging':   # [BAGGING]
            docs = vs.retrieve_documents(self.vect_db, prefix, self.pool_size)
            return docs, bootstrap_bags(docs, n, self.docs_per_llm, bag_rng(self.bag_seed, prefix),
                                        replace=self.bag_replacement)
        docs = vs.retrieve_documents(self.vect_db, prefix, self.docs_per_llm * n)
        return docs, split_round_robin(docs, n)

    def predict(self, prefix: str, info_run: Dict) -> Prediction:
        classes = list(info_run['Activities'])
        all_docs, contexts = self.contexts_for(prefix)
        member_results, prompts, parsed_labels = [], [], []
        for (name, member), ctx_docs in zip(self.members, contexts):
            try:
                prompt, raw = member.produce_answer_with_context(prefix, vs.format_context(ctx_docs), info_run)
                parsed = parse_model_answer(raw, classes)
                error = None
            except QuotaExhausted:
                # [GROQ] Cota acabou: NÃO vira voto nulo (isso contaminaria todos os prefixos seguintes
                #        em silêncio). Sobe para o eval parar e retomar depois pelo checkpoint.
                raise
            except Exception as e:   # timeout, erro pontual etc.: voto nulo deste membro
                prompt, parsed, error = '', WRONG_FORMAT, f'{type(e).__name__}: {e}'
            parsed_labels.append(parsed)
            prompts.append(f'===== [{name}] =====\n{prompt}')
            member_results.append({'member': name, 'answer': parsed,
                                   'context_ranks': [d['rank'] for d in ctx_docs],
                                   'context_labels': [d['label'] for d in ctx_docs],
                                   **({'error': error} if error else {})})
        winner, votes = majority_vote(parsed_labels)
        details = {'strategy': 'ensemble', 'mode': self.mode, 'docs_per_llm': self.docs_per_llm,
                   'members': member_results, 'votes': votes,
                   'retrieved_labels': [d['label'] for d in all_docs]}
        if self.mode == 'bagging':   # [BAGGING] o suficiente para reconstruir as bags a partir do relatório
            details.update({'pool_size': self.pool_size, 'bag_replacement': self.bag_replacement,
                            'bag_seed': self.bag_seed})
        return Prediction(prompt='\n'.join(prompts),
                          answer=winner if winner is not None else WRONG_FORMAT,
                          already_parsed=True, details=details)

    def usage_report(self) -> Dict:
        return _usage_of(self.members)


class StrategyTracker:
    """
    Métricas que o oracle não calcula, acumuladas prefixo a prefixo:
      - label_in_context: o rótulo verdadeiro está entre os rótulos recuperados (recall@k do retrieval;
                          é o teto do que o contexto permite acertar).
      - por membro do ensemble: acurácia individual no MESMO test set.
      - any_member_correct: algum membro acertou (teto do voto).
      - unanimous: todos os membros válidos concordaram.
      - [BAGGING] por membro: label_in_context (o rótulo verdadeiro estava no contexto DAQUELE membro) e
        documentos únicos no contexto (bootstrap de 3 em pool de 9: esperado 9·(1−(8/9)³) ≈ 2,68 distintos).
        Servem para partition e bagging: mostram quanto do recall do pool cada membro perde.
    A acurácia do voto em si continua vindo do oracle.
    """
    def __init__(self):
        self.n = 0
        self.label_in_context = 0
        self.member_correct = OrderedDict()
        self.member_wrong_format = OrderedDict()
        self.any_member_correct = 0
        self.unanimous = 0
        self.has_members = False
        self.member_label_in_context = OrderedDict()   # [BAGGING]
        self.member_unique_docs = OrderedDict()        # [BAGGING]

    def update(self, details: Optional[Dict], expected: str):
        if not details:
            return
        self.n += 1
        retrieved = details.get('retrieved_labels') or []
        if any(lbl is not None and is_correct(lbl, expected) for lbl in retrieved):
            self.label_in_context += 1
        members = details.get('members')
        if not members:
            return
        self.has_members = True
        hits = []
        for i, m in enumerate(members):
            key = f"{i}:{m['member']}"   # índice no nome: o mesmo modelo pode aparecer mais de uma vez
            ok = m['answer'] != WRONG_FORMAT and is_correct(m['answer'], expected)
            self.member_correct[key] = self.member_correct.get(key, 0) + int(ok)
            self.member_wrong_format[key] = self.member_wrong_format.get(key, 0) + int(m['answer'] == WRONG_FORMAT)
            # [BAGGING]
            ctx_labels = m.get('context_labels') or []
            in_ctx = any(lbl is not None and is_correct(lbl, expected) for lbl in ctx_labels)
            self.member_label_in_context[key] = self.member_label_in_context.get(key, 0) + int(in_ctx)
            self.member_unique_docs[key] = self.member_unique_docs.get(key, 0) + len(set(m.get('context_ranks') or []))
            hits.append(ok)
        self.any_member_correct += int(any(hits))
        valid = [m['answer'] for m in members if m['answer'] != WRONG_FORMAT]
        self.unanimous += int(len(valid) == len(members) and len(set(valid)) == 1)

    def summary(self) -> Dict:
        if self.n == 0:
            return {}
        out = {'samples': self.n, 'label_in_context_rate': round(self.label_in_context / self.n, 4)}
        if self.has_members:
            out['member_accuracy'] = {k: round(v / self.n, 4) for k, v in self.member_correct.items()}
            out['member_wrong_format_rate'] = {k: round(v / self.n, 4) for k, v in self.member_wrong_format.items()}
            out['any_member_correct_rate'] = round(self.any_member_correct / self.n, 4)
            out['unanimous_rate'] = round(self.unanimous / self.n, 4)
            out['member_label_in_context_rate'] = {k: round(v / self.n, 4)
                                                   for k, v in self.member_label_in_context.items()}
            out['member_mean_unique_docs'] = {k: round(v / self.n, 2) for k, v in self.member_unique_docs.items()}
        return out