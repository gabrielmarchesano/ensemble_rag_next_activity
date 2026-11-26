"""
Testes do ensemble.py, com foco no modo bagging (bootstrap do contexto recuperado).
Qdrant em memória + embedding falso (conftest.py). Nenhum LLM real: os membros são classes falsas.
"""
import hashlib
import importlib
import json
import os
from collections import Counter

import pytest

import ensemble as ens
import oracle as orc
import vector_store as vs

INFO_RUN = {'Activities': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
            'Event Attributes': "{'orgr': 'org:group'}"}
QUESTION = "[X,Y] - Values: {'orgr': 'A'}"


def _docs(n):
    """Pool falso de n documentos já ranqueados (rank 1 = mais similar)."""
    return [{'rank': i + 1, 'content': f'[X,Y] - Values: {{}} | Next activity: {chr(65 + i)}',
             'label': chr(65 + i), 'score': 1.0 - i / 100} for i in range(n)]


@pytest.fixture
def kb(qdrant_mem, fake_embed):
    """KB com 10 documentos de mesma chave (empate de similaridade) e rótulos A..J."""
    client, store, collection = qdrant_mem
    vs.store_traces([('X,Y', f'Values: {{}} | Next activity: {chr(65 + i)}') for i in range(10)],
                    client, 'x', fake_embed, collection)
    return store


class _ContextVoter:
    """Membro falso: responde o rótulo mais frequente do contexto que recebeu (empate: o primeiro)."""
    def __init__(self):
        self.contexts = []

    def produce_answer_with_context(self, question, context, info_run):
        self.contexts.append(context)
        labels = [line.rsplit('| Next activity:', 1)[1].strip()
                  for line in context.strip().split('\n') if '| Next activity:' in line]
        label = Counter(labels).most_common(1)[0][0] if labels else 'None'
        return f'prompt:{context}', f'\\boxed{{{label}}}'


# ==============================================================================================
# Sorteio
# ==============================================================================================
class TestBagRng:
    def test_mesmo_prefixo_mesma_sequencia(self):
        a, b = ens.bag_rng(10, QUESTION), ens.bag_rng(10, QUESTION)
        assert [a.random() for _ in range(5)] == [b.random() for _ in range(5)]

    def test_seed_ou_prefixo_diferente_muda_a_sequencia(self):
        base = ens.bag_rng(10, QUESTION).random()
        assert ens.bag_rng(11, QUESTION).random() != base
        assert ens.bag_rng(10, QUESTION + 'x').random() != base

    def test_nao_depende_do_random_global(self):
        import random
        random.seed(1)
        first = ens.bag_rng(10, QUESTION).random()
        random.seed(999)
        assert ens.bag_rng(10, QUESTION).random() == first


class TestBootstrapBags:
    def test_tamanho_e_numero_de_bags(self):
        bags = ens.bootstrap_bags(_docs(9), n_bags=4, bag_size=3, rng=ens.bag_rng(10, 'p'))
        assert len(bags) == 4 and all(len(b) == 3 for b in bags)

    def test_bags_ordenadas_por_rank(self):
        for bag in ens.bootstrap_bags(_docs(9), 20, 3, ens.bag_rng(10, 'p')):
            ranks = [d['rank'] for d in bag]
            assert ranks == sorted(ranks)

    def test_com_reposicao_repete_documento(self):
        # pool de 2 e bag de 3: pelo princípio da casa dos pombos toda bag tem repetição
        for bag in ens.bootstrap_bags(_docs(2), 10, 3, ens.bag_rng(10, 'p'), replace=True):
            assert len({d['rank'] for d in bag}) < 3

    def test_sem_reposicao_nao_repete(self):
        for bag in ens.bootstrap_bags(_docs(9), 50, 3, ens.bag_rng(10, 'p'), replace=False):
            assert len({d['rank'] for d in bag}) == 3

    def test_pool_vazio(self):
        assert ens.bootstrap_bags([], 3, 3, ens.bag_rng(10, 'p')) == [[], [], []]

    def test_distribuicao_do_bootstrap(self):
        # Média de documentos distintos numa bag de 3 sorteada de 9: 9 * (1 - (8/9)^3) ≈ 2,68
        rng = ens.bag_rng(10, 'p')
        bags = ens.bootstrap_bags(_docs(9), 4000, 3, rng)
        mean_unique = sum(len({d['rank'] for d in b}) for b in bags) / len(bags)
        assert abs(mean_unique - 9 * (1 - (8 / 9) ** 3)) < 0.03
        # cada rank do pool é sorteado com a mesma frequência (≈ 1/9)
        freq = Counter(d['rank'] for b in bags for d in b)
        assert max(freq.values()) / min(freq.values()) < 1.15


# ==============================================================================================
# LLMEnsemble em modo bagging
# ==============================================================================================
class TestEnsembleBagging:
    def _ens(self, kb, n=3, **kw):
        members = [(f'm{i}', _ContextVoter()) for i in range(n)]
        return ens.LLMEnsemble(members, kb, docs_per_llm=3, mode='bagging', **kw), members

    def test_pool_default_igual_ao_partition(self, kb):
        e, _ = self._ens(kb)
        assert e.pool_size == 9 and e.retrieved_per_prefix() == 9
        part = ens.LLMEnsemble([('a', None), ('b', None), ('c', None)], kb, docs_per_llm=3, mode='partition')
        assert part.retrieved_per_prefix() == e.retrieved_per_prefix()

    def test_cada_membro_recebe_k_documentos_do_pool(self, kb):
        e, _ = self._ens(kb, pool_size=5)
        pool, contexts = e.contexts_for(QUESTION)
        assert len(pool) == 5
        pool_ranks = {d['rank'] for d in pool}
        for ctx in contexts:
            assert len(ctx) == 3 and {d['rank'] for d in ctx} <= pool_ranks

    def test_mesmo_prefixo_mesmas_bags_entre_instancias(self, kb):
        # garante que a retomada pelo checkpoint e runs separadas sorteiam as mesmas bags
        a, _ = self._ens(kb)
        b, _ = self._ens(kb)
        ranks = lambda e: [[d['rank'] for d in c] for c in e.contexts_for(QUESTION)[1]]
        assert ranks(a) == ranks(b)

    def test_seed_muda_as_bags(self, kb):
        ranks = lambda seed: [[d['rank'] for d in c]
                              for c in self._ens(kb, bag_seed=seed)[0].contexts_for(QUESTION)[1]]
        assert any(ranks(10) != ranks(s) for s in range(11, 16))

    def test_predict_registra_bags_e_vota(self, kb):
        e, members = self._ens(kb)
        pred = e.predict(QUESTION, INFO_RUN)
        assert pred.already_parsed
        d = pred.details
        assert d['mode'] == 'bagging' and d['pool_size'] == 9 and d['bag_replacement'] is True
        assert len(d['retrieved_labels']) == 9
        for m in d['members']:
            assert len(m['context_ranks']) == 3
            assert m['answer'] in INFO_RUN['Activities']
        # o vencedor é o rótulo mais votado entre os membros
        winner, _ = ens.majority_vote([m['answer'] for m in d['members']])
        assert pred.answer == winner
        # o membro recebeu o contexto no formato do controle ("\n1. doc\n2. doc ...")
        assert members[0][1].contexts[0].startswith('\n1. [X,Y]')

    def test_mesmo_modelo_repetido_bagging_homogeneo(self, kb):
        one = _ContextVoter()
        e = ens.LLMEnsemble([('m', one)] * 5, kb, docs_per_llm=3, mode='bagging')
        pred = e.predict(QUESTION, INFO_RUN)
        assert len(one.contexts) == 5 and len(pred.details['members']) == 5
        assert e.retrieved_per_prefix() == 15

    def test_sem_reposicao(self, kb):
        e, _ = self._ens(kb, bag_replacement=False)
        for ctx in e.contexts_for(QUESTION)[1]:
            assert len({d['rank'] for d in ctx}) == 3

    def test_validacoes(self, kb):
        with pytest.raises(ValueError):
            ens.LLMEnsemble([('a', None)], kb, mode='boosting')
        with pytest.raises(ValueError):
            ens.LLMEnsemble([('a', None)], kb, docs_per_llm=3, mode='bagging', pool_size=2, bag_replacement=False)

    def test_modos_antigos_inalterados(self, kb):
        shared = ens.LLMEnsemble([('a', None), ('b', None)], kb, docs_per_llm=3, mode='shared')
        docs, ctxs = shared.contexts_for(QUESTION)
        assert len(docs) == 3 and ctxs[0] == ctxs[1] == docs
        part = ens.LLMEnsemble([('a', None), ('b', None)], kb, docs_per_llm=3, mode='partition')
        docs, ctxs = part.contexts_for(QUESTION)
        assert [d['rank'] for d in ctxs[0]] == [1, 3, 5] and [d['rank'] for d in ctxs[1]] == [2, 4, 6]


# ==============================================================================================
# Métricas por membro
# ==============================================================================================
def test_tracker_recall_e_documentos_unicos_por_membro():
    t = ens.StrategyTracker()
    details = {'retrieved_labels': ['A', 'B', 'C'],
               'members': [{'member': 'x', 'answer': 'A', 'context_ranks': [1, 1, 2], 'context_labels': ['A', 'A', 'B']},
                           {'member': 'y', 'answer': 'C', 'context_ranks': [3, 3, 3], 'context_labels': ['C', 'C', 'C']}]}
    t.update(details, 'A')
    s = t.summary()
    assert s['member_label_in_context_rate'] == {'0:x': 1.0, '1:y': 0.0}
    assert s['member_mean_unique_docs'] == {'0:x': 2.0, '1:y': 1.0}
    assert s['member_accuracy'] == {'0:x': 1.0, '1:y': 0.0}
    assert s['label_in_context_rate'] == 1.0


# ==============================================================================================
# Integração com o eval.py (checkpoint e fluxo completo)
# ==============================================================================================
@pytest.fixture
def eval_mod(monkeypatch):
    monkeypatch.setenv('QDRANT_GRPC_PORT', '6334')
    import eval as ev
    importlib.reload(ev)
    monkeypatch.setattr(orc.VerificationOracle, 'write_results_to_file', lambda self: self.compute_stats())
    return ev


BASE_ARGS = dict(log='sepsis.xes', evaluation_modality='evaluation-attributes', strategy='ensemble',
                 llm_id='gpt-4.1', ensemble_llms='a,b,c', ensemble_mode='bagging', num_documents_in_context=3,
                 knn_k=10, knn_weighting='count', embed_model_id='x', rag=True, max_new_tokens=1024,
                 reasoning_effort='low', prefix_base=1, prefix_gap=3, test_size=300, test_proportion=0.3,
                 bag_pool_size=None, bag_replacement=True, bag_seed=10)


def _ckpt(eval_mod, csv_path, **over):
    ns = type('A', (), {**BASE_ARGS, **over})()
    return eval_mod.checkpoint_path_for(ns, csv_path)


def test_checkpoint_bagging(eval_mod, tmp_path):
    csv_path = tmp_path / 't.csv'
    csv_path.write_text('prefix,prediction\na,b\n')
    p = lambda **o: _ckpt(eval_mod, str(csv_path), **o)
    assert p() != p(bag_seed=11)
    assert p() != p(bag_replacement=False)
    assert p() == p(bag_pool_size=9)              # None = k * n = 9: mesma configuração, mesmo checkpoint
    assert p() != p(bag_pool_size=12)
    assert p() != p(ensemble_mode='partition')


def test_checkpoint_de_shared_partition_nao_muda(eval_mod, tmp_path):
    """Runs shared/partition já em andamento (Groq) continuam retomando: a chave é a mesma de antes."""
    csv_path = tmp_path / 't.csv'
    csv_path.write_text('prefix,prediction\na,b\n')
    for mode in ('shared', 'partition'):
        args = {k: v for k, v in BASE_ARGS.items() if not k.startswith('bag_')}
        args['ensemble_mode'] = mode
        cfg = {k: args.get(k) for k in eval_mod.CHECKPOINT_KEYS}
        cfg['llm_id'] = None
        cfg['test_set_md5'] = hashlib.md5(csv_path.read_bytes()).hexdigest()
        old_key = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
        assert os.path.basename(_ckpt(eval_mod, str(csv_path), ensemble_mode=mode)) == f'sepsis_ensemble_{old_key}.jsonl'


def test_evaluate_pipeline_bagging_ponta_a_ponta(eval_mod, kb, tmp_path):
    questions = [[f"[X,Y] - Values: {{'i': {i}}}", 'A'] for i in range(6)]
    members = [(f'm{i}', _ContextVoter()) for i in range(3)]
    e = ens.LLMEnsemble(members, kb, docs_per_llm=3, mode='bagging')
    ckpt = str(tmp_path / 'run.jsonl')
    info = dict(INFO_RUN)
    oracle = eval_mod.evaluate_pipeline(e, questions, info, checkpoint_path=ckpt)
    assert len(oracle.results) == 6
    summary = info['Strategy Summary']
    assert set(summary['member_label_in_context_rate']) == {'0:m0', '1:m1', '2:m2'}
    assert all(1.0 <= v <= 3.0 for v in summary['member_mean_unique_docs'].values())
    # retomar do checkpoint dá exatamente as mesmas respostas, sem chamar os membros
    fresh = [(f'm{i}', _ContextVoter()) for i in range(3)]
    oracle2 = eval_mod.evaluate_pipeline(ens.LLMEnsemble(fresh, kb, docs_per_llm=3, mode='bagging'),
                                         questions, dict(INFO_RUN), checkpoint_path=ckpt)
    assert all(not m.contexts for _, m in fresh)
    assert [r['model_answer'] for r in oracle2.results] == [r['model_answer'] for r in oracle.results]


# ==============================================================================================
# Oracle: interface usada pelo ensemble
# ==============================================================================================
class TestOracleInterface:
    def test_parse(self):
        assert orc.parse_model_answer('blah \\boxed{x} ... \\boxed{C}', INFO_RUN['Activities']) == 'C'
        assert orc.parse_model_answer('sem resposta', INFO_RUN['Activities']) == orc.WRONG_FORMAT

    def test_already_parsed_nao_parseia_de_novo(self):
        o = orc.VerificationOracle(INFO_RUN)
        assert o.verify_answer('p', '[X] - Values: {}', 'C', expected_answer='C', already_parsed=True)
        assert 'details' not in o.results[-1]   # só entra no resultado quando details != None
