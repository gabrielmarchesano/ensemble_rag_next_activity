"""
Testes de vector_store.py: armazenamento no Qdrant e busca por similaridade.
Qdrant em memória + embedding falso determinístico (ver conftest.py).
"""
import csv
import pytest

import log_preprocessing as lp
import vector_store as vs

ACTS = ['ER Registration', 'ER Triage', 'ER Sepsis Triage', 'IV Liquid', 'IV Antibiotics',
        'CRP', 'Leucocytes', 'LacticAcid', 'Admission NC', 'Release A']


def _docs():
    return [
        ('ER Registration,ER Triage', "Values: {'orgr': 'C'} | Next activity: ER Sepsis Triage"),
        ('ER Registration,ER Triage', "Values: {'orgr': 'C'} | Next activity: CRP"),
        ('ER Registration,ER Triage,ER Sepsis Triage,IV Liquid', "Values: {'orgr': 'A'} | Next activity: IV Antibiotics"),
        ('CRP,Leucocytes,LacticAcid', "Values: {'orgr': 'B'} | Next activity: Admission NC"),
        ('Admission NC,CRP,Leucocytes', "Values: {'orgr': 'B'} | Next activity: Release A"),
    ]


def _all_points(client, collection):
    points, _ = client.scroll(collection, limit=100_000, with_payload=True, with_vectors=True)
    return points


# ==============================================================================================
# [Raf #4] Tudo armazenado no Qdrant
# ==============================================================================================
class TestStoreTraces:
    def test_todos_os_documentos_sao_armazenados(self, qdrant_mem, fake_embed):
        client, _, col = qdrant_mem
        n = vs.store_traces(_docs(), client, 'sepsis.xes', fake_embed, col)
        assert n == len(_docs())
        assert client.count(col, exact=True).count == len(_docs())

    def test_payload_tem_o_documento_completo(self, qdrant_mem, fake_embed):
        client, _, col = qdrant_mem
        vs.store_traces(_docs(), client, 'sepsis.xes', fake_embed, col)
        stored = {p.payload['page_content'] for p in _all_points(client, col)}
        assert stored == {f'[{k}] - {d}' for k, d in _docs()}

    def test_embedding_e_so_da_sequencia_de_atividades(self, qdrant_mem, fake_embed):
        # O vetor não pode carregar Values nem o rótulo — senão a query (sem rótulo) nunca casa bem.
        client, _, col = qdrant_mem
        fake_embed.calls.clear()   # o QdrantVectorStore embeda um 'dummy_text' no init para checar a dimensão
        vs.store_traces(_docs(), client, 'sepsis.xes', fake_embed, col)
        embedded = [t for kind, t in fake_embed.calls if kind == 'doc']
        assert embedded == [k for k, _ in _docs()]
        assert not any('Next activity' in t or 'Values' in t for t in embedded)

    def test_dimensao_do_vetor_bate_com_a_colecao(self, qdrant_mem, fake_embed):
        client, _, col = qdrant_mem
        vs.store_traces(_docs(), client, 'sepsis.xes', fake_embed, col)
        size = client.get_collection(col).config.params.vectors.size
        assert size == fake_embed.dim
        assert all(len(p.vector) == size for p in _all_points(client, col))

    def test_documentos_identicos_sao_deduplicados(self, qdrant_mem, fake_embed):
        client, _, col = qdrant_mem
        n = vs.store_traces(_docs() + _docs(), client, 'sepsis.xes', fake_embed, col)
        assert n == client.count(col, exact=True).count == len(_docs())

    def test_lote_acima_de_512_pontos(self, qdrant_mem, fake_embed):
        client, _, col = qdrant_mem
        docs = [(f'ER Registration,CRP{i}', f"Values: {{}} | Next activity: X{i}") for i in range(1300)]
        assert vs.store_traces(docs, client, 'sepsis.xes', fake_embed, col) == 1300
        assert client.count(col, exact=True).count == 1300

    def test_aceita_formato_antigo_dict(self, qdrant_mem, fake_embed):
        client, _, col = qdrant_mem
        assert vs.store_traces(dict(_docs()), client, 'sepsis.xes', fake_embed, col) == 4

    def test_ids_reiniciam_em_0_segunda_chamada_sobrescreve(self, qdrant_mem, fake_embed):
        # Comportamento ATUAL documentado: ids começam em 0 a cada chamada. Ok com rebuild=True,
        # mas guardar 2 logs (ou 2 lotes) na MESMA coleção sobrescreve o primeiro silenciosamente.
        client, _, col = qdrant_mem
        vs.store_traces(_docs(), client, 'a.xes', fake_embed, col)
        vs.store_traces(_docs()[:2], client, 'b.xes', fake_embed, col)
        names = {p.payload['name'] for p in _all_points(client, col)}
        assert client.count(col, exact=True).count == 5          # 5 + 2 inseridos, só 5 sobrevivem
        assert names == {'b.xes Trace 0', 'b.xes Trace 1', 'a.xes Trace 2', 'a.xes Trace 3', 'a.xes Trace 4'}


class TestInitializeVectorStore:
    @pytest.fixture
    def shared_client(self, monkeypatch):
        """Mesmo client em memória entre chamadas (simula o servidor Qdrant persistindo entre execuções)."""
        from qdrant_client import QdrantClient
        shared = QdrantClient(':memory:')
        monkeypatch.setattr(vs, 'QdrantClient', lambda *a, **k: shared)
        return shared

    def test_rebuild_false_preserva_e_true_apaga(self, shared_client, fake_embed):
        vs.initialize_vector_store('url', 6334, 'c', fake_embed, fake_embed.dim, True)
        vs.store_traces(_docs(), shared_client, 'x', fake_embed, 'c')
        vs.initialize_vector_store('url', 6334, 'c', fake_embed, fake_embed.dim, False)
        assert shared_client.count('c', exact=True).count == len(_docs())
        vs.initialize_vector_store('url', 6334, 'c', fake_embed, fake_embed.dim, True)
        assert shared_client.count('c', exact=True).count == 0

    def test_troca_de_modelo_de_embedding_recria_colecao(self, shared_client, fake_embed):
        # Coleção antiga com dimensão 32; novo embedding com 64 e rebuild=False -> deve recriar com 64.
        from conftest import FakeActivityEmbeddings
        old = FakeActivityEmbeddings(dim=32)
        vs.initialize_vector_store('url', 6334, 'c', old, 32, True)
        vs.initialize_vector_store('url', 6334, 'c', fake_embed, fake_embed.dim, False)
        assert shared_client.get_collection('c').config.params.vectors.size == fake_embed.dim

    def test_colecao_criada_com_cosseno_e_dimensao_correta(self, qdrant_mem, fake_embed):
        client, _, col = qdrant_mem
        params = client.get_collection(col).config.params.vectors
        assert params.size == fake_embed.dim
        assert params.distance == vs.Distance.COSINE


# ==============================================================================================
# [Raf #5] Busca por similaridade
# ==============================================================================================
@pytest.fixture
def filled_store(qdrant_mem, fake_embed):
    client, store, col = qdrant_mem
    vs.store_traces(_docs(), client, 'sepsis.xes', fake_embed, col)
    return store


class TestRetrieveContext:
    def test_query_e_parseada_para_o_mesmo_formato_do_documento(self, filled_store, fake_embed):
        q = "[ER Registration,ER Triage] - Values: {'orgr': 'C', 'Ag': '70'}"
        vs.retrieve_context(filled_store, q, 3)
        assert fake_embed.calls[-1] == ('query', 'ER Registration,ER Triage')

    def test_atividade_com_hifen_nao_quebra_a_query(self, filled_store, fake_embed):
        # [FIX 6] split('-') antigo cortaria em "Pre"
        q = "[ER Registration,Pre-Op Check] - Values: {}"
        vs.retrieve_context(filled_store, q, 1)
        assert fake_embed.calls[-1] == ('query', 'ER Registration,Pre-Op Check')

    def test_match_exato_vem_no_topo(self, filled_store):
        ctx = vs.retrieve_context(filled_store, "[CRP,Leucocytes,LacticAcid] - Values: {}", 3)
        first = ctx.strip().split('\n')[0]
        assert first.startswith('1. [CRP,Leucocytes,LacticAcid] - ')
        assert 'Next activity: Admission NC' in first

    def test_prefixo_com_continuacoes_alternativas_traz_ambas(self, filled_store):
        ctx = vs.retrieve_context(filled_store, "[ER Registration,ER Triage] - Values: {}", 2)
        assert 'Next activity: ER Sepsis Triage' in ctx and 'Next activity: CRP' in ctx

    def test_ranking_relativo(self, filled_store):
        # Mais atividades em comum -> posição melhor; nenhuma em comum -> última.
        ctx = vs.retrieve_context(filled_store, "[ER Registration,ER Triage,ER Sepsis Triage] - Values: {}", 5)
        lines = [l for l in ctx.strip().split('\n')]
        pos = {l.split('. ', 1)[1].split(' - ')[0]: i for i, l in enumerate(lines)}
        assert pos['[ER Registration,ER Triage,ER Sepsis Triage,IV Liquid]'] < pos['[CRP,Leucocytes,LacticAcid]']

    @pytest.mark.parametrize('k', [1, 3, 5])
    def test_retorna_k_documentos_numerados(self, filled_store, k):
        ctx = vs.retrieve_context(filled_store, "[ER Registration] - Values: {}", k)
        lines = ctx.strip().split('\n')
        assert len(lines) == k
        assert [l.split('.')[0] for l in lines] == [str(i) for i in range(1, k + 1)]

    def test_k_maior_que_a_base_retorna_tudo(self, filled_store):
        ctx = vs.retrieve_context(filled_store, "[ER Registration] - Values: {}", 50)
        assert len(ctx.strip().split('\n')) == len(_docs())


# ==============================================================================================
# Contrato entre módulos: pergunta do CSV  <->  chave embedada na KB
# ==============================================================================================
def test_pergunta_do_csv_gera_query_igual_a_chave_da_kb(xes, tmp_path, qdrant_mem, fake_embed):
    """Se o formato do CSV e o do documento divergirem, a busca degrada em silêncio. Este teste trava isso."""
    traces = [xes.inner(xes.trace(f'c{i}', [xes.event(a, org__group=g) for a, g in
                                            zip(ACTS[i % 3:i % 3 + 5], 'ABCDE')])) for i in range(6)]
    client, store, col = qdrant_mem
    kb_docs, _, _ = lp.process_prefixes(lp.build_prefixes(traces, 1, 1))
    vs.store_traces(kb_docs, client, 'x', fake_embed, col)
    kb_keys = {k for k, _ in kb_docs}

    path = tmp_path / 't.csv'
    lp.generate_csv_from_test_set(traces, str(path), size=50, base=1, gap=1)
    for question, label in list(csv.reader(open(path)))[1:]:
        vs.retrieve_context(store, question, 1)
        parsed = fake_embed.calls[-1][1]
        assert parsed in kb_keys
        # e o top-1 é um documento com exatamente essa sequência
        top1 = vs.retrieve_context(store, question, 1).strip()
        assert top1.startswith(f'1. [{parsed}] - ')


# ==============================================================================================
# Integração com embedding REAL (lento; roda no Kaggle/Colab)
# ==============================================================================================
@pytest.mark.slow
def test_similaridade_com_minilm_real(qdrant_mem):
    from conftest import is_stub
    if is_stub('langchain_huggingface') or is_stub('sentence_transformers'):
        pytest.skip('langchain_huggingface / sentence_transformers não instalados')
    from langchain_huggingface import HuggingFaceEmbeddings
    embed = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2')
    dim = len(embed.embed_query('test'))
    client, store = vs.initialize_vector_store(':memory:', 6334, 'real', embed, dim, True)
    vs.store_traces(_docs(), client, 'x', embed, 'real')
    ctx = vs.retrieve_context(store, "[CRP,Leucocytes,LacticAcid] - Values: {}", 1)
    assert '[CRP,Leucocytes,LacticAcid]' in ctx
