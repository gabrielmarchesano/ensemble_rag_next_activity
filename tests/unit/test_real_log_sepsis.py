"""
Testes de integração com o log REAL (logs/sepsis.xes). Pulados se o log não estiver descompactado.
Rodam em segundos, sem GPU: o embedding é o falso do conftest.
"""
import csv
import random
import re

import pytest

import log_preprocessing as lp
import vector_store as vs

SEPSIS_ACTS = {'ER Registration', 'ER Triage', 'ER Sepsis Triage', 'IV Liquid', 'IV Antibiotics',
               'LacticAcid', 'CRP', 'Leucocytes', 'Admission NC', 'Admission IC',
               'Release A', 'Release B', 'Release C', 'Release D', 'Release E', 'Return ER'}


@pytest.fixture(scope='module')
def sepsis():
    import os
    from conftest import LOGS
    path = os.path.join(LOGS, 'sepsis.xes')
    if not os.path.exists(path):
        pytest.skip('logs/sepsis.xes não encontrado (descompacte logs.zip na raiz do repo)')
    content = lp.read_event_log('sepsis.xes')
    traces = lp.extract_traces(content)
    random.seed(10)
    kb, test = lp.split_traces(traces, 0.3)
    kb_docs, keys, acts = lp.process_prefixes(lp.build_prefixes(kb, 1, 3), 'sepsis.xes')
    test_docs, _, test_acts = lp.process_prefixes(lp.build_prefixes(test, 1, 3), 'sepsis.xes')
    return dict(content=content, traces=traces, kb=kb, test=test, kb_docs=kb_docs,
                test_docs=test_docs, acts=acts, test_acts=test_acts, keys=keys)


def test_extracao_bate_com_contagem_independente(sepsis):
    raw = re.findall(r'<trace>(.*?)</trace>', sepsis['content'], re.DOTALL)
    expected = sum(1 for t in raw if len(re.findall(r'<event>', t)) >= 2)
    assert len(sepsis['traces']) == expected == 1050


def test_espaco_de_classes_do_sepsis(sepsis):
    assert sepsis['acts'] | sepsis['test_acts'] == SEPSIS_ACTS


def test_split_70_30_por_trace(sepsis):
    assert len(sepsis['test']) == int(1050 * 0.3)
    assert set(sepsis['kb']).isdisjoint(sepsis['test'])


def test_nenhuma_pergunta_de_teste_existe_literalmente_na_kb(sepsis):
    """
    O Values carrega o timestamp (titi) do último evento antes do rótulo, que é praticamente único por
    caso. Pergunta de teste idêntica a um documento da KB => mesmo caso nos dois lados => vazamento.
    (Sequência de atividades igual é esperado: variantes se repetem entre casos diferentes.)
    """
    q = lambda docs: {f'[{k}] - {d.split(" | Next activity:")[0]}' for k, d in docs}
    assert q(sepsis['kb_docs']).isdisjoint(q(sepsis['test_docs']))


def test_csv_gerado_tem_formato_de_pergunta(sepsis, tmp_path):
    path = tmp_path / 'test_set_sepsis.csv'
    lp.generate_csv_from_test_set(sepsis['test'], str(path), size=300, log_name='sepsis.xes', base=1, gap=3)
    rows = list(csv.reader(open(path, newline='')))[1:]
    assert len(rows) == 300
    for question, label in rows:
        assert re.match(r"^\[[^\[\]]+\] - Values: \{.*\}$", question), question[:120]
        assert label in SEPSIS_ACTS


def test_todos_os_documentos_da_kb_vao_para_o_qdrant(sepsis, qdrant_mem, fake_embed):
    client, _, col = qdrant_mem
    n = vs.store_traces(sepsis['kb_docs'], client, 'sepsis.xes', fake_embed, col)
    unique = {f'[{k}] - {d}' for k, d in sepsis['kb_docs']}
    assert n == len(unique) == client.count(col, exact=True).count
