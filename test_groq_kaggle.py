"""
Testes das mudanças para rodar com Groq (plano gratuito) ou no Kaggle:
  - RAGPipeline: família groq, parâmetros de init, 429 -> QuotaExhausted, contagem de tokens, dtype.
  - vector_store: Qdrant local em disco (QDRANT_URL=path:...).
  - eval: checkpoint por prefixo e retomada depois de cota estourada.
Sem rede: init_chat_model é trocado por um LLM falso.
"""
import json
import os

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import ensemble as ens
import oracle as orc
import pipeline as p
import vector_store as vs
from errors import QuotaExhausted

INFO_RUN = {'Activities': ['CRP', 'LacticAcid', 'Leucocytes', 'Release A'],
            'Event Attributes': "{'orgr': 'org:group'}"}
QUESTION = "[ER Registration,ER Triage] - Values: {'orgr': 'A'}"


class RateLimitError(Exception):
    """Mesmo nome da exceção dos SDKs groq/openai (é pelo nome que o pipeline reconhece)."""
    status_code = 429


@pytest.fixture
def fake_groq(monkeypatch):
    rec = {'kwargs': None, 'calls': 0, 'fail_after': None}

    def fake_init(model, **kwargs):
        rec['kwargs'] = dict(model=model, **kwargs)

        def _llm(prompt_value):
            rec['calls'] += 1
            if rec['fail_after'] is not None and rec['calls'] > rec['fail_after']:
                raise RateLimitError('429 Too Many Requests: tokens per day (TPD)')
            return AIMessage(content='raciocínio... \\boxed{CRP}',
                             usage_metadata={'input_tokens': 900, 'output_tokens': 150, 'total_tokens': 1050})
        return RunnableLambda(_llm)

    monkeypatch.setattr(p, 'init_chat_model', fake_init)
    return rec


# ----------------------------------------------------------------------------------------------
# RAGPipeline
# ----------------------------------------------------------------------------------------------
class TestGroqPipeline:
    @pytest.mark.parametrize('model_id', ['openai/gpt-oss-120b', 'openai/gpt-oss-20b', 'qwen/qwen3.8-27b',
                                          'llama-3.3-70b-versatile'])
    def test_familia_groq(self, model_id, fake_groq):
        pipe = p.RAGPipeline(model_id, 512, True)
        assert (pipe.model_family, pipe.model_type) == ('groq', 'api')
        assert fake_groq['kwargs']['model_provider'] == 'groq'
        assert pipe.TEMPLATE_MAPPING['groq'] == 'template-generic'

    def test_parametros_de_init(self, fake_groq):
        p.RAGPipeline('openai/gpt-oss-20b', 512, True, requests_per_minute=6, max_retries=10,
                      reasoning_effort='low')
        kw = fake_groq['kwargs']
        assert kw['max_retries'] == 10
        assert kw['reasoning_effort'] == 'low'
        assert kw['max_tokens'] == 512 and kw['temperature'] == 0.1
        assert kw['rate_limiter'].requests_per_second == pytest.approx(0.1)

    def test_reasoning_effort_so_em_modelo_que_aceita(self, fake_groq):
        p.RAGPipeline('qwen/qwen3.8-27b', 512, True, reasoning_effort='low')
        assert fake_groq['kwargs']['reasoning_effort'] == 'low'
        p.RAGPipeline('llama-3.3-70b-versatile', 512, True, reasoning_effort='low')
        assert 'reasoning_effort' not in fake_groq['kwargs']

    def test_sem_throttle_nem_effort_nao_manda_os_campos(self, fake_groq):
        p.RAGPipeline('openai/gpt-oss-20b', 512, True)
        assert 'rate_limiter' not in fake_groq['kwargs']
        assert 'reasoning_effort' not in fake_groq['kwargs']

    def test_openai_continua_com_a_chamada_antiga(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(p, 'init_chat_model',
                            lambda model, model_provider, temperature, max_tokens: seen.update(
                                provider=model_provider) or RunnableLambda(lambda x: AIMessage(content='x')))
        p.RAGPipeline('gpt-4.1', 512, True, requests_per_minute=6, reasoning_effort='low')
        assert seen['provider'] == 'openai'

    def test_conta_tokens(self, fake_groq):
        pipe = p.RAGPipeline('openai/gpt-oss-20b', 512, True)
        for _ in range(3):
            pipe.produce_answer_with_context(QUESTION, '\n1. [a] - Values: {} | Next activity: CRP', INFO_RUN)
        assert pipe.usage == {'calls': 3, 'input_tokens': 2700, 'output_tokens': 450}

    def test_429_vira_quota_exhausted(self, fake_groq):
        fake_groq['fail_after'] = 0
        pipe = p.RAGPipeline('openai/gpt-oss-20b', 512, True)
        with pytest.raises(QuotaExhausted, match='gpt-oss-20b'):
            pipe.produce_answer_with_context(QUESTION, '', INFO_RUN)

    def test_outro_erro_continua_sendo_o_erro_original(self, monkeypatch):
        def boom(x):
            raise TimeoutError('lento')
        monkeypatch.setattr(p, 'init_chat_model', lambda *a, **k: RunnableLambda(boom))
        pipe = p.RAGPipeline('openai/gpt-oss-20b', 512, True)
        with pytest.raises(TimeoutError):
            pipe.produce_answer_with_context(QUESTION, '', INFO_RUN)


class TestKaggleDtype:
    def test_t4_sem_bf16_usa_float16(self, monkeypatch):
        monkeypatch.setattr(p.torch.cuda, 'is_available', lambda: True)
        monkeypatch.setattr(p.torch.cuda, 'is_bf16_supported', lambda: False)
        assert p._compute_dtype() is p.float16

    def test_gpu_com_bf16_usa_bfloat16(self, monkeypatch):
        monkeypatch.setattr(p.torch.cuda, 'is_available', lambda: True)
        monkeypatch.setattr(p.torch.cuda, 'is_bf16_supported', lambda: True)
        assert p._compute_dtype() is p.bfloat16


# ----------------------------------------------------------------------------------------------
# Ensemble não engole cota estourada
# ----------------------------------------------------------------------------------------------
class _QuotaMember:
    def produce_answer_with_context(self, question, context, info_run):
        raise QuotaExhausted('TPD')


def test_ensemble_propaga_quota_exhausted(qdrant_mem, fake_embed):
    client, store, collection = qdrant_mem
    vs.store_traces([('ER Registration,ER Triage', 'Values: {} | Next activity: CRP')], client, 'x',
                    fake_embed, collection)
    with pytest.raises(QuotaExhausted):
        ens.LLMEnsemble([('a', _QuotaMember())], store, docs_per_llm=1).predict(QUESTION, INFO_RUN)


# ----------------------------------------------------------------------------------------------
# Qdrant local em disco
# ----------------------------------------------------------------------------------------------
def test_qdrant_em_disco_persiste_entre_clientes(tmp_path, fake_embed):
    url = f'path:{tmp_path / "qdrant"}'
    client, store = vs.initialize_vector_store(url, 6334, 'kb', fake_embed, fake_embed.dim, rebuild_db=True)
    vs.store_traces([('A,B', 'Values: {} | Next activity: C'), ('A', 'Values: {} | Next activity: B')],
                    client, 'x', fake_embed, 'kb')
    client.close()
    client2, store2 = vs.initialize_vector_store(url, 6334, 'kb', fake_embed, fake_embed.dim, rebuild_db=False)
    assert client2.count('kb').count == 2
    assert vs.retrieve_documents(store2, '[A,B] - Values: {}', 1)[0]['label'] == 'C'
    client2.close()


# ----------------------------------------------------------------------------------------------
# Checkpoint e retomada (eval.evaluate_pipeline)
# ----------------------------------------------------------------------------------------------
@pytest.fixture
def eval_mod(monkeypatch):
    monkeypatch.setenv('QDRANT_GRPC_PORT', '6334')
    import importlib
    import eval as ev
    importlib.reload(ev)
    monkeypatch.setattr(orc.VerificationOracle, 'write_results_to_file', lambda self: self.compute_stats())
    return ev


class _CountingMember:
    """Responde o rótulo pedido; estoura a cota depois de `quota` chamadas."""
    def __init__(self, quota=None):
        self.calls, self.quota = 0, quota

    def produce_answer_with_context(self, question, context, info_run):
        self.calls += 1
        if self.quota is not None and self.calls > self.quota:
            raise QuotaExhausted('TPD')
        label = 'CRP' if 'Triage]' in question else 'Release A'
        return 'prompt', f'\\boxed{{{label}}}'


QUESTIONS = [[f"[ER Registration,ER Triage] - Values: {{'i': {i}}}", 'CRP'] if i % 2 == 0 else
             [f"[IV Liquid,Admission NC] - Values: {{'i': {i}}}", 'Release A'] for i in range(10)]


@pytest.fixture
def kb(qdrant_mem, fake_embed):
    client, store, collection = qdrant_mem
    vs.store_traces([('ER Registration,ER Triage', 'Values: {} | Next activity: CRP'),
                     ('IV Liquid,Admission NC', 'Values: {} | Next activity: Release A')],
                    client, 'x', fake_embed, collection)
    return store


def test_retoma_de_onde_parou_e_da_o_mesmo_resultado(eval_mod, kb, tmp_path):
    ckpt = str(tmp_path / 'run.jsonl')
    first = _CountingMember(quota=4)
    assert eval_mod.evaluate_pipeline(ens.LLMEnsemble([('m', first)], kb, docs_per_llm=1),
                                      QUESTIONS, dict(INFO_RUN), checkpoint_path=ckpt) is None
    assert len(open(ckpt).readlines()) == 4

    second = _CountingMember()
    info = dict(INFO_RUN)
    oracle = eval_mod.evaluate_pipeline(ens.LLMEnsemble([('m', second)], kb, docs_per_llm=1),
                                        QUESTIONS, info, checkpoint_path=ckpt)
    assert second.calls == 6                          # só os prefixos que faltavam
    assert oracle.accuracy == 1.0 and len(oracle.results) == 10
    assert info['Resumed From Checkpoint'] == '4 prefixos'
    assert info['Strategy Summary']['samples'] == 10  # o tracker também recebe os retomados

    fresh = eval_mod.evaluate_pipeline(ens.LLMEnsemble([('m', _CountingMember())], kb, docs_per_llm=1),
                                       QUESTIONS, dict(INFO_RUN), checkpoint_path=str(tmp_path / 'b.jsonl'))
    assert fresh.predicted_next_activities == oracle.predicted_next_activities


def test_linha_cortada_no_checkpoint_e_ignorada(eval_mod, kb, tmp_path):
    ckpt = tmp_path / 'run.jsonl'
    ckpt.write_text('{"index": 0, "prefix": "corta')   # queda no meio da escrita
    member = _CountingMember()
    oracle = eval_mod.evaluate_pipeline(ens.LLMEnsemble([('m', member)], kb, docs_per_llm=1),
                                        QUESTIONS, dict(INFO_RUN), checkpoint_path=str(ckpt))
    assert member.calls == 10 and oracle.accuracy == 1.0


def test_checkpoint_nao_reaproveita_prefixo_diferente(eval_mod, kb, tmp_path):
    ckpt = tmp_path / 'run.jsonl'
    ckpt.write_text(json.dumps({'index': 0, 'prefix': 'OUTRO', 'prompt': '', 'answer': 'Leucocytes',
                                'already_parsed': True, 'details': None}) + '\n')
    member = _CountingMember()
    oracle = eval_mod.evaluate_pipeline(ens.LLMEnsemble([('m', member)], kb, docs_per_llm=1),
                                        QUESTIONS, dict(INFO_RUN), checkpoint_path=str(ckpt))
    assert member.calls == 10 and oracle.accuracy == 1.0


def test_chave_do_checkpoint(eval_mod, tmp_path):
    test_csv = tmp_path / 't.csv'
    test_csv.write_text('prefix,prediction\na,b\n')
    base = dict(log='sepsis.xes', evaluation_modality='evaluation-attributes', strategy='ensemble',
                llm_id='gpt-4.1', ensemble_llms='openai/gpt-oss-20b,qwen/qwen3.8-27b', ensemble_mode='shared',
                num_documents_in_context=3, knn_k=10, knn_weighting='count', embed_model_id='x', rag=True,
                max_new_tokens=1024, reasoning_effort='low', prefix_base=1, prefix_gap=3, test_size=300,
                test_proportion=0.3, requests_per_minute=6)

    def path(**over):
        ns = type('A', (), {**base, **over})()
        return eval_mod.checkpoint_path_for(ns, str(test_csv))

    assert path() == path(requests_per_minute=2)          # throttle não muda o resultado
    assert path() == path(llm_id='outro')                 # llm_id não é usado no ensemble
    assert path() != path(reasoning_effort='medium')      # muda o resultado
    assert path() != path(ensemble_mode='partition')
    assert os.path.basename(path()).startswith('sepsis_ensemble_')
    before = path()
    test_csv.write_text('prefix,prediction\na,c\n')        # test set diferente -> checkpoint novo
    assert path() != before
