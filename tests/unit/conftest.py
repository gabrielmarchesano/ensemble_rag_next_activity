"""
Fixtures compartilhadas da suíte de testes da IC (RAG para Next Activity Prediction).

Rodar a partir da raiz do repositório:
    pytest tests/unit -v                 # rápido, sem GPU, sem API, sem servidor Qdrant
    pytest tests/unit -v -m "not slow"   # exclui o teste com embedding real (MiniLM)

Decisões de design:
- Qdrant roda em memória (QdrantClient(":memory:")): não precisa de servidor.
- Embedding é FALSO e determinístico (hash md5 de unigramas/bigramas de atividades): os testes
  validam a LÓGICA de indexação/busca, não a qualidade semântica do modelo de embedding.
- LLM é falso (RunnableLambda): nenhum teste chama API paga nem carrega modelo na GPU.
- Se torch/transformers/sentence_transformers/langchain não estiverem instalados (ex.: notebook
  sem GPU), entram stubs vazios só para o `import pipeline` funcionar. No Kaggle as libs reais
  são usadas.
"""
import hashlib
import importlib
import math
import os
import random
import re
import sys
import types
from unittest.mock import MagicMock

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC = os.path.join(ROOT, 'src')
LOGS = os.path.join(ROOT, 'logs')
if SRC not in sys.path:
    sys.path.insert(0, SRC)


# ----------------------------------------------------------------------------------------------
# Stubs para dependências pesadas (só entram se a lib real não existir)
# ----------------------------------------------------------------------------------------------
def _make_stub(name):
    """Módulo vazio cujo qualquer atributo é um MagicMock (dunders levantam AttributeError, como num módulo real)."""
    mod = types.ModuleType(name)

    def __getattr__(attr):
        if attr.startswith('__'):
            raise AttributeError(attr)
        value = MagicMock(name=f'{name}.{attr}')
        setattr(mod, attr, value)
        return value
    mod.__getattr__ = __getattr__
    mod.__path__ = []          # permite "import pacote.submodulo"
    mod.__is_test_stub__ = True
    return mod


def _stub_if_missing(module_name):
    try:
        importlib.import_module(module_name)
    except Exception:
        parent = module_name.split('.')[0]
        if parent != module_name and parent not in sys.modules:
            sys.modules[parent] = _make_stub(parent)
        stub = _make_stub(module_name)
        if module_name == 'torch':
            # scikit-learn (array_api_compat) faz isinstance(x, torch.Tensor) se 'torch' estiver em sys.modules
            stub.Tensor = type('Tensor', (), {})
        sys.modules[module_name] = stub
        if parent != module_name:
            setattr(sys.modules[parent], module_name.split('.', 1)[1], stub)


def is_stub(module_name):
    return getattr(sys.modules.get(module_name), '__is_test_stub__', False)


for _mod in ('torch', 'transformers', 'sentence_transformers',
             'langchain_huggingface', 'langchain.chat_models'):
    _stub_if_missing(_mod)


# ----------------------------------------------------------------------------------------------
# Builder de XES sintético
# ----------------------------------------------------------------------------------------------
def make_event(activity, lifecycle='complete', **attrs):
    """Gera um <event> no mesmo formato do XES do sepsis. attrs viram <string key=... value=.../>."""
    lines = [f'<string key="concept:name" value="{activity}"/>']
    if lifecycle is not None:
        lines.append(f'<string key="lifecycle:transition" value="{lifecycle}"/>')
    for k, v in attrs.items():
        lines.append(f'<string key="{k.replace("__", ":")}" value="{v}"/>')
    return '<event>\n' + '\n'.join(lines) + '\n</event>'


def make_trace(case_id, events):
    """events: lista de strings (make_event) ou de nomes de atividade."""
    evs = [e if e.startswith('<event>') else make_event(e) for e in events]
    return f'<trace>\n<string key="concept:name" value="{case_id}"/>\n' + '\n'.join(evs) + '\n</trace>'


def make_log(traces):
    return '<?xml version="1.0" encoding="UTF-8" ?>\n<log>\n' + '\n'.join(traces) + '\n</log>'


def inner(trace_xml):
    """Conteúdo entre <trace> e </trace> (formato que extract_traces devolve)."""
    return re.search(r'<trace>(.*?)</trace>', trace_xml, re.DOTALL).group(1)


@pytest.fixture
def xes():
    """Expõe o builder para os testes: xes.event(...), xes.trace(...), xes.log(...), xes.inner(...)."""
    class _B:
        event = staticmethod(make_event)
        trace = staticmethod(make_trace)
        log = staticmethod(make_log)
        inner = staticmethod(inner)
    return _B


@pytest.fixture
def toy_log():
    """
    Log pequeno com casos de borda:
      c1: 6 eventos normais (com atributos)
      c2: 1 evento            -> deve ser descartado por extract_traces
      c3: 2 eventos           -> mantido por extract_traces, mas build_prefixes exige > 2
      c4: mesmo prefixo de c1, próxima atividade diferente (continuação alternativa)
      c5: self-loop (CRP, CRP)
    """
    traces = [
        make_trace('c1', [make_event('ER Registration', org__group='A', Age='70'),
                          make_event('ER Triage', org__group='C'),
                          make_event('CRP', org__group='B'),
                          make_event('Leucocytes', org__group='B'),
                          make_event('IV Antibiotics', org__group='A'),
                          make_event('Release A', org__group='E')]),
        make_trace('c2', ['ER Registration']),
        make_trace('c3', ['ER Registration', 'ER Triage']),
        make_trace('c4', [make_event('ER Registration', org__group='A', Age='45'),
                          make_event('ER Triage', org__group='C'),
                          make_event('LacticAcid', org__group='B')]),
        make_trace('c5', ['ER Registration', 'ER Triage', 'CRP', 'CRP', 'Leucocytes']),
    ]
    return make_log(traces)


# ----------------------------------------------------------------------------------------------
# Embedding falso determinístico
# ----------------------------------------------------------------------------------------------
from langchain_core.embeddings import Embeddings  # noqa: E402


class FakeActivityEmbeddings(Embeddings):
    """
    Bag of unigramas + bigramas de atividades, hasheado com md5 em `dim` posições e normalizado.
    Mesma sequência -> mesmo vetor (cosseno 1). Sequências que compartilham atividades -> cosseno alto.
    Determinístico entre processos (md5, não hash() do Python).
    """
    def __init__(self, dim=64):
        self.dim = dim
        self.calls = []   # registra o que foi embedado (para testar o formato da query/documento)

    def _vec(self, text):
        acts = [a.strip() for a in text.strip().strip('[]').split(',') if a.strip()]
        feats = acts + [f'{a}>>{b}' for a, b in zip(acts, acts[1:])]
        v = [0.0] * self.dim
        for f in feats:
            v[int(hashlib.md5(f.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        self.calls.extend(('doc', t) for t in texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        self.calls.append(('query', text))
        return self._vec(text)


@pytest.fixture
def fake_embed():
    return FakeActivityEmbeddings(dim=64)


@pytest.fixture
def qdrant_mem(fake_embed):
    """(client, store, collection_name) com Qdrant em memória, criado pelo próprio initialize_vector_store."""
    import vector_store as vs
    collection = 'test-nap-rag'
    client, store = vs.initialize_vector_store(':memory:', 6334, collection, fake_embed,
                                               fake_embed.dim, rebuild_db=True)
    return client, store, collection


@pytest.fixture(autouse=True)
def _seed():
    random.seed(10)


@pytest.fixture
def sepsis_path():
    path = os.path.join(LOGS, 'sepsis.xes')
    if not os.path.exists(path):
        pytest.skip('logs/sepsis.xes não encontrado (descompacte logs.zip na raiz do repo)')
    return path
