"""
Testes de pipeline.py: modelo correto, prompt, contexto recuperado e parsing da resposta.
Nenhum teste chama API paga nem carrega modelo — o LLM é um RunnableLambda que grava o prompt recebido.
"""
import os
import subprocess
import sys
import textwrap
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import pipeline as p
import vector_store as vs

INFO_RUN = {
    'Activities': {'ER Registration', 'ER Triage', 'CRP', 'Leucocytes', 'Release A'},
    'Event Attributes': "{'orgr': 'org:group', 'Ag': 'Age'}",
}
QUESTION = "[ER Registration,ER Triage] - Values: {'orgr': 'C', 'Ag': '70'}"
FAKE_CONTEXT = "\n1. [ER Registration,ER Triage] - Values: {'orgr': 'C'} | Next activity: CRP"


@pytest.fixture
def fake_api(monkeypatch):
    """Troca init_chat_model por um LLM falso. Retorna dict com os kwargs da chamada e os prompts enviados."""
    rec = {'init': None, 'sent': []}

    def fake_init_chat_model(model, model_provider, temperature, max_tokens):
        rec['init'] = dict(model=model, model_provider=model_provider,
                           temperature=temperature, max_tokens=max_tokens)

        def _llm(prompt_value):
            rec['sent'].append(prompt_value.to_string())
            return AIMessage(content='Reasoning... \\boxed{CRP}')
        return RunnableLambda(_llm)

    monkeypatch.setattr(p, 'init_chat_model', fake_init_chat_model)
    return rec


@pytest.fixture
def spy_retrieve(monkeypatch):
    calls = []

    def _spy(vectdb, question, num_chunks, *a, **k):
        calls.append((vectdb, question, num_chunks))
        return FAKE_CONTEXT
    monkeypatch.setattr(p, 'retrieve_context', _spy)
    return calls


def bare(model_id, family=None, use_rag=True):
    """RAGPipeline sem __init__ (não carrega modelo)."""
    obj = p.RAGPipeline.__new__(p.RAGPipeline)
    obj.model_id = model_id
    obj.use_rag = use_rag
    if family:
        obj.model_family = family
    return obj


# ==============================================================================================
# Modelo correto
# ==============================================================================================
class TestModelResolution:
    @pytest.mark.parametrize('model_id, family, mtype', [
        ('gpt-4.1', 'openai', 'api'),
        ('gpt-4o-mini', 'openai', 'api'),
        ('gemini-2.5-flash', 'google_genai', 'api'),
        ('deepseek-chat', 'deepseek', 'api'),
        ('meta-llama/Llama-3.1-8B-Instruct', 'metaai', 'local'),
        ('mistralai/Mistral-7B-Instruct-v0.3', 'mistral', 'local'),
        ('Qwen/Qwen2.5-7B-Instruct', 'qwen', 'local'),
        ('microsoft/phi-4', 'microsoft', 'local'),
        ('deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'deepseek', 'local'),
        ('google/gemma-2-9b-it', 'google_genai', 'local'),
        ('angeloc1/autotrain-phi-ft-nap', 'angeloc1', 'local'),
    ])
    def test_familia_e_tipo(self, model_id, family, mtype):
        assert bare(model_id)._get_model_family_type() == (family, mtype)

    @pytest.mark.xfail(strict=True, reason=(
        'Modelo fora de MODELS cai em ("openai", "api") com um print. O ValueError do __init__ nunca '
        'dispara. Ex.: qualquer modelo Claude (lista "anthropic" vazia) iria para o provider openai.'))
    def test_modelo_desconhecido_falha_alto(self):
        with pytest.raises(ValueError):
            bare('claude-sonnet-4-5')._get_model_family_type()

    @pytest.mark.parametrize('model_id, provider', [('gpt-4.1', 'openai'), ('gemini-2.5-flash', 'google_genai'),
                                                    ('deepseek-chat', 'deepseek')])
    def test_api_inicializa_com_id_provider_e_hiperparametros(self, fake_api, model_id, provider):
        p.RAGPipeline(model_id, max_new_tokens=512, use_rag=True)
        assert fake_api['init'] == dict(model=model_id, model_provider=provider,
                                        temperature=0.1, max_tokens=512)

    def test_local_carrega_o_checkpoint_pedido(self, monkeypatch):
        mid = 'meta-llama/Llama-3.1-8B-Instruct'
        for name in ('AutoConfig', 'AutoModelForCausalLM', 'AutoTokenizer', 'BitsAndBytesConfig'):
            monkeypatch.setattr(p, name, MagicMock(name=name))
        p.AutoTokenizer.from_pretrained.return_value.eos_token_id = 1
        p.AutoTokenizer.from_pretrained.return_value.convert_tokens_to_ids.return_value = 2
        hf_pipeline = MagicMock(name='hf_pipeline')
        monkeypatch.setattr(p, 'pipeline', hf_pipeline)
        monkeypatch.setattr(p, 'HuggingFacePipeline',
                            lambda pipeline: RunnableLambda(lambda pv: pv.to_string() + ' \\boxed{CRP}'))

        pipe = p.RAGPipeline(mid, max_new_tokens=256, use_rag=True)

        assert pipe.model_type == 'local'
        assert p.AutoModelForCausalLM.from_pretrained.call_args.args[0] == mid
        assert p.AutoTokenizer.from_pretrained.call_args.args[0] == mid
        kw = hf_pipeline.call_args.kwargs
        assert kw['max_new_tokens'] == 256 and kw['temperature'] == 0.1
        assert kw['eos_token_id'] == [1, 2]                     # eos + <|eot_id|>

    @pytest.mark.xfail(strict=True, reason='Typo: terminador do Mistral é "[/INST]]" (colchete extra) -> vira token desconhecido.')
    def test_terminadores_bem_formados(self):
        for fam, tok in p.RAGPipeline.TERMINATOR_TOKENS.items():
            assert not tok.endswith(']]'), fam


# ==============================================================================================
# Contexto recuperado e enviado ao modelo
# ==============================================================================================
class TestProduceAnswer:
    def test_rag_chama_retrieve_com_os_argumentos_certos(self, fake_api, spy_retrieve):
        pipe = p.RAGPipeline('gpt-4.1', 256, use_rag=True)
        db = object()
        pipe.produce_answer(QUESTION, db, 3, INFO_RUN)
        assert spy_retrieve == [(db, QUESTION, 3)]

    def test_contexto_e_pergunta_chegam_no_prompt_enviado(self, fake_api, spy_retrieve):
        pipe = p.RAGPipeline('gpt-4.1', 256, use_rag=True)
        _, answer = pipe.produce_answer(QUESTION, None, 3, INFO_RUN)
        sent = fake_api['sent'][0]
        assert FAKE_CONTEXT.strip() in sent
        assert f'Prefix to predict: {QUESTION}' in sent
        assert answer == 'Reasoning... \\boxed{CRP}'

    def test_sem_rag_nao_busca_nem_envia_contexto(self, fake_api, spy_retrieve):
        pipe = p.RAGPipeline('gpt-4.1', 256, use_rag=False)
        pipe.produce_answer(QUESTION, None, 3, INFO_RUN)
        assert spy_retrieve == []
        assert 'past traces:' not in fake_api['sent'][0].lower().split('example for the output format')[0]
        assert 'Next activity: CRP' not in fake_api['sent'][0]

    def test_ponta_a_ponta_com_qdrant_em_memoria(self, fake_api, qdrant_mem, fake_embed):
        client, store, col = qdrant_mem
        vs.store_traces([('ER Registration,ER Triage', "Values: {'orgr': 'C'} | Next activity: CRP"),
                         ('CRP,Leucocytes', "Values: {} | Next activity: Release A")],
                        client, 'x', fake_embed, col)
        pipe = p.RAGPipeline('gpt-4.1', 256, use_rag=True)
        pipe.produce_answer(QUESTION, store, 1, INFO_RUN)
        sent = fake_api['sent'][0]
        assert "1. [ER Registration,ER Triage] - Values: {'orgr': 'C'} | Next activity: CRP" in sent
        assert 'Release A' not in sent.split('Here is the past traces:')[1]   # k=1: só o mais similar

    @pytest.mark.xfail(strict=True, reason=(
        'Para modelos de API o prompt devolvido (e gravado nos arquivos de output/validation) é uma '
        'reconstrução manual ("Here is the most similar past traces") diferente do prompt realmente '
        'enviado ("Here is the past traces"). O log não reflete o experimento.'))
    def test_prompt_logado_e_o_prompt_enviado(self, fake_api, spy_retrieve):
        pipe = p.RAGPipeline('gpt-4.1', 256, use_rag=True)
        logged, _ = pipe.produce_answer(QUESTION, None, 3, INFO_RUN)
        assert logged.strip() == fake_api['sent'][0].strip()


# ==============================================================================================
# System message e templates
# ==============================================================================================
FAMILIES = list(p.RAGPipeline.TEMPLATE_MAPPING)


def _cases(known_bugs):
    """Todas as combinações (família, rag); as de known_bugs viram xfail strict (quebram quando o bug for corrigido)."""
    return [pytest.param(f, r, marks=pytest.mark.xfail(strict=True, reason=known_bugs[(f, r)]))
            if (f, r) in known_bugs else (f, r)
            for f in FAMILIES for r in (True, False)]


class TestPrompts:
    def test_system_message_sem_placeholders(self):
        msg = bare('gpt-4.1', 'openai')._get_system_message(INFO_RUN)
        assert 'ACTIVITIES' not in msg and 'REPLACE' not in msg
        assert all(a in msg for a in INFO_RUN['Activities'])
        assert "'orgr': 'org:group'" in msg

    @pytest.mark.parametrize('family, use_rag', _cases(
        {('angeloc1', False): 'prompts_no_rag.json não tem "template-phi-ft": cai no fallback genérico'}))
    def test_template_existe_e_tem_as_variaveis_certas(self, family, use_rag, capsys):
        tpl = bare('x', family, use_rag)._generate_prompt_template()
        assert 'Warning' not in capsys.readouterr().out
        expected = {'system_message', 'question'} | ({'context'} if use_rag else set())
        assert set(tpl.input_variables) == expected

    @pytest.mark.parametrize('family, use_rag', _cases({
        (fam, rag): r'template-phi(-ft) tem "\\n" no JSON -> o modelo recebe "\n" literal em vez de quebra de linha'
        for fam, rag in [('microsoft', True), ('microsoft', False), ('angeloc1', True)]}))
    def test_template_sem_barra_n_literal(self, family, use_rag):
        tpl = bare('x', family, use_rag)._generate_prompt_template()
        assert '\\n' not in tpl.template


# ==============================================================================================
# Parsing da resposta de modelos locais (return_full_text=True)
# ==============================================================================================
@pytest.mark.parametrize('family', ['metaai', 'mistral', 'qwen', 'microsoft', 'angeloc1', 'deepseek',
                                    pytest.param('google_genai', marks=pytest.mark.xfail(strict=True, reason=(
                                        'gemma local usa o delimitador default "Answer:", que aparece antes no '
                                        'exemplo do system message -> a "resposta" parseada inclui o prompt')))])
def test_parse_separa_prompt_e_resposta(family):
    pipe = bare('x', family, True)
    prompt = pipe._generate_prompt_template().format(
        system_message=pipe._get_system_message(INFO_RUN), context=FAKE_CONTEXT, question=QUESTION)
    parsed_prompt, answer = pipe._parse_llm_answer(prompt + ' The answer is \\boxed{CRP}')
    assert answer.strip() == 'The answer is \\boxed{CRP}'
    assert parsed_prompt.strip() == prompt.strip()


# ==============================================================================================
# Reprodutibilidade
# ==============================================================================================
@pytest.mark.xfail(strict=True, reason=(
    'seed_everything seta PYTHONHASHSEED em tempo de execução, o que NÃO afeta o processo atual. '
    'A ordem de str(set_de_atividades) no system message (e de oracle.total_classes) muda a cada '
    'execução -> prompts diferentes entre runs. Fix: sorted(activities_set).'))
def test_lista_de_atividades_no_prompt_e_deterministica():
    src = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
    code = textwrap.dedent(f"""
        import os, sys
        os.environ['PYTHONHASHSEED'] = '10'          # o que seed_everything faz
        sys.path.insert(0, {src!r})
        import log_preprocessing as lp
        ev = lambda a: '<event><string key="concept:name" value="%s"/></event>' % a
        acts = ['ER Registration','ER Triage','ER Sepsis Triage','IV Liquid','IV Antibiotics','CRP',
                'Leucocytes','LacticAcid','Admission NC','Admission IC','Release A','Release B','Return ER']
        _, _, activities = lp.process_prefixes([''.join(ev(a) for a in acts)])
        print(str(activities))
    """)
    outs = {subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                           env={**os.environ, 'PYTHONHASHSEED': str(s)}).stdout for s in range(1, 6)}
    assert len(outs) == 1
