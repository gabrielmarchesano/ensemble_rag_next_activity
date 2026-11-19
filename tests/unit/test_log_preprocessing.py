"""
Testes de log_preprocessing.py: extração de traces, prefixos, rótulo, atributos, split e CSV de teste.
"""
import csv
import random
import re

import pytest

import log_preprocessing as lp

EVENT_RE = re.compile(r'<event>.*?</event>', re.DOTALL)
ACT_RE = re.compile(r'<string key="concept:name" value="(.*?)"\s*/>')


def activities_of(prefix_xml):
    """Sequência de atividades (concept:name) de um prefixo/trace em XML."""
    return [ACT_RE.search(e).group(1) for e in EVENT_RE.findall(prefix_xml)]


def split_doc(doc):
    """'Values: {...} | Next activity: X' -> ('{...}', 'X')"""
    values, label = doc.split(' | Next activity: ')
    return values.removeprefix('Values: '), label


# ==============================================================================================
# Extração das traces
# ==============================================================================================
class TestExtractTraces:
    def test_mantem_so_traces_com_2_ou_mais_eventos(self, toy_log):
        traces = lp.extract_traces(toy_log)
        # c1 (6 ev), c3 (2 ev), c4 (3 ev), c5 (5 ev) ficam; c2 (1 evento) sai
        assert [len(EVENT_RE.findall(t)) for t in traces] == [6, 2, 3, 5]

    def test_preserva_conteudo_e_ordem_das_traces(self, toy_log):
        traces = lp.extract_traces(toy_log)
        case_ids = [re.search(r'<string key="concept:name" value="(c\d)"/>\n<event>', t).group(1) for t in traces]
        assert case_ids == ['c1', 'c3', 'c4', 'c5']
        assert activities_of(traces[0]) == ['ER Registration', 'ER Triage', 'CRP', 'Leucocytes',
                                            'IV Antibiotics', 'Release A']

    def test_trace_extraida_nao_contem_tags_de_trace(self, toy_log):
        for t in lp.extract_traces(toy_log):
            assert '<trace>' not in t and '</trace>' not in t

    def test_log_vazio_retorna_lista_vazia(self, xes):
        assert lp.extract_traces(xes.log([])) == []


# ==============================================================================================
# Prefixos (base / gap)
# ==============================================================================================
class TestBuildPrefixes:
    @pytest.mark.parametrize('n_events, base, gap, expected_lengths', [
        (6, 1, 3, [2, 5]),
        (10, 1, 3, [2, 5, 8]),
        (8, 1, 3, [2, 5, 8]),          # aqui o último prefixo é a trace inteira
        (6, 1, 1, [2, 3, 4, 5, 6]),
        (6, 2, 2, [3, 5]),
    ])
    def test_tamanhos_dos_prefixos(self, xes, n_events, base, gap, expected_lengths):
        trace = xes.inner(xes.trace('c', [f'A{i}' for i in range(n_events)]))
        prefixes = lp.build_prefixes([trace], base, gap)
        assert [len(EVENT_RE.findall(p)) for p in prefixes] == expected_lengths

    def test_cada_prefixo_e_prefixo_real_da_trace(self, toy_log):
        traces = lp.extract_traces(toy_log)
        for p in lp.build_prefixes(traces, 1, 1):
            acts = activities_of(p)
            assert any(activities_of(t)[:len(acts)] == acts for t in traces)

    def test_trace_com_2_eventos_nao_gera_prefixo(self, xes):
        # Comportamento ATUAL documentado: extract_traces aceita >= 2 eventos, mas build_prefixes exige > 2.
        # Uma trace [a1, a2] poderia gerar o par ([a1] -> a2). Confirmar se é intencional.
        trace = xes.inner(xes.trace('c', ['ER Registration', 'ER Triage']))
        assert lp.build_prefixes([trace], 1, 1) == []


# ==============================================================================================
# Rótulo = última atividade do prefixo; input = tudo antes dela
# ==============================================================================================
class TestLabelConstruction:
    def test_rotulo_e_ultima_atividade_e_input_exclui_o_rotulo(self, toy_log):
        prefixes = lp.build_prefixes(lp.extract_traces(toy_log), 1, 1)
        docs, _, _ = lp.process_prefixes(prefixes)
        assert docs, 'nenhum documento gerado'
        produced = {(inp, split_doc(doc)[1]) for inp, doc in docs}
        expected = {(','.join(activities_of(p)[:-1]), activities_of(p)[-1]) for p in prefixes}
        assert produced == expected

    def test_input_nunca_contem_a_atividade_rotulo_na_ultima_posicao_por_construcao(self, xes):
        # [a, b, c] -> input "a,b" e rótulo "c". Garante que o rótulo NÃO está anexado ao input.
        prefix = ''.join(xes.event(a) for a in ['ER Registration', 'ER Triage', 'CRP'])
        docs, _, _ = lp.process_prefixes([prefix])
        assert docs[0][0] == 'ER Registration,ER Triage'
        assert split_doc(docs[0][1])[1] == 'CRP'

    def test_self_loop_e_legitimo(self, xes):
        # Atenção: "próxima atividade == última atividade do input" NÃO é sinal de vazamento.
        # No sepsis há loops reais (CRP -> CRP, Leucocytes -> Leucocytes): ~12% do test set atual.
        prefix = ''.join(xes.event(a) for a in ['ER Registration', 'CRP', 'CRP'])
        docs, _, _ = lp.process_prefixes([prefix])
        assert docs == [('ER Registration,CRP', 'Values: {} | Next activity: CRP')]

    def test_prefixo_com_uma_unica_atividade_e_descartado(self, xes):
        docs, _, _ = lp.process_prefixes([xes.event('ER Registration')])
        assert docs == []


# ==============================================================================================
# Atributos (Values) — vazamento do evento-rótulo
# ==============================================================================================
class TestAttributes:
    def _prefix(self, xes):
        return ''.join([
            xes.event('ER Registration', org__group='A', Age='70'),
            xes.event('ER Triage', org__group='C'),
            xes.event('Release A', org__group='E'),       # evento-rótulo
        ])

    def test_atributos_do_evento_rotulo_nao_vazam_por_padrao(self, xes):
        docs, _, _ = lp.process_prefixes([self._prefix(xes)])
        values, label = split_doc(docs[0][1])
        assert label == 'Release A'
        assert "'orgr': 'C'" in values           # último valor ANTES do rótulo
        assert "'E'" not in values                # org:group do evento-rótulo não aparece

    def test_modo_antigo_inclui_atributos_do_rotulo(self, xes):
        docs, _, _ = lp.process_prefixes([self._prefix(xes)], include_label_attributes=True)
        assert "'orgr': 'E'" in split_doc(docs[0][1])[0]

    def test_ultimo_valor_de_cada_atributo_prevalece(self, xes):
        docs, _, _ = lp.process_prefixes([self._prefix(xes)])
        values = split_doc(docs[0][1])[0]
        assert values == "{'orgr': 'C', 'Ag': '70'}"

    def test_mapa_de_abreviacoes_de_atributos(self, xes):
        _, keys, activities = lp.process_prefixes([self._prefix(xes)])
        assert keys == {'orgr': 'org:group', 'Ag': 'Age'}
        assert activities == {'ER Registration', 'ER Triage', 'Release A'}

    @pytest.mark.xfail(strict=True, reason=(
        'BUG LATENTE (não afeta o sepsis, que só tem eventos "complete"): '
        'attr_events = events[:kept_idx[-1]] inclui o evento "start" da própria atividade-rótulo, '
        'cujo org:resource vaza para o Values. Afeta logs com start/complete (ex.: BPIC 2012).'))
    def test_evento_start_da_atividade_rotulo_nao_vaza(self, xes):
        prefix = ''.join([
            xes.event('A', 'complete', org__resource='R1'),
            xes.event('B', 'start', org__resource='R9'),     # início da atividade-rótulo
            xes.event('B', 'complete', org__resource='R9'),  # rótulo
        ])
        docs, _, _ = lp.process_prefixes([prefix])
        assert docs[0][0] == 'A'
        assert "'R9'" not in docs[0][1]


# ==============================================================================================
# Lifecycle e deduplicação
# ==============================================================================================
class TestLifecycleAndDedup:
    def test_eventos_nao_complete_sao_ignorados(self, xes):
        prefix = ''.join([xes.event('A', 'start'), xes.event('A', 'complete'),
                          xes.event('B', 'start'), xes.event('B', 'complete')])
        docs, _, activities = lp.process_prefixes([prefix])
        assert docs[0][0] == 'A' and split_doc(docs[0][1])[1] == 'B'
        assert activities == {'A', 'B'}

    def test_evento_sem_lifecycle_e_aceito(self, xes):
        prefix = ''.join([xes.event('A', None), xes.event('B', None)])
        docs, _, _ = lp.process_prefixes([prefix])
        assert docs[0][0] == 'A'

    def test_mesmo_prefixo_com_continuacoes_diferentes_gera_2_documentos(self, xes):
        p1 = ''.join(xes.event(a) for a in ['ER Registration', 'ER Triage', 'CRP'])
        p2 = ''.join(xes.event(a) for a in ['ER Registration', 'ER Triage', 'LacticAcid'])
        docs, _, _ = lp.process_prefixes([p1, p2])
        assert sorted(split_doc(d)[1] for _, d in docs) == ['CRP', 'LacticAcid']

    def test_transicao_repetida_gera_1_documento(self, xes):
        p = ''.join(xes.event(a) for a in ['ER Registration', 'ER Triage', 'CRP'])
        docs, _, _ = lp.process_prefixes([p, p])
        assert len(docs) == 1


# ==============================================================================================
# [Raf #3] Split por trace sem vazamento
# ==============================================================================================
def _toy_traces(xes, n=40):
    """Traces com um atributo único por caso (case:id) -> qualquer documento carrega a 'impressão digital' do caso."""
    random.seed(0)
    pool = ['ER Registration', 'ER Triage', 'CRP', 'Leucocytes', 'LacticAcid', 'IV Liquid', 'Release A']
    traces = []
    for c in range(n):
        acts = [pool[0], pool[1]] + random.choices(pool[2:], k=random.randint(2, 8))
        traces.append(xes.inner(xes.trace(f'case{c}', [xes.event(a, case__id=f'case{c}') for a in acts])))
    return traces


class TestSplitByTrace:
    def test_split_e_particao_disjunta_e_completa(self, xes):
        traces = _toy_traces(xes)
        kb, test = lp.split_traces(traces, 0.3)
        assert len(test) == int(len(traces) * 0.3)
        assert set(kb).isdisjoint(test)
        assert sorted(kb + test) == sorted(traces)

    def test_split_reprodutivel_com_mesma_seed(self, xes):
        traces = _toy_traces(xes)
        random.seed(10); a = lp.split_traces(traces, 0.3)
        random.seed(10); b = lp.split_traces(traces, 0.3)
        assert a == b

    @staticmethod
    def _questions(docs):
        """Mesmo formato da pergunta do CSV: '[a,b] - Values: {...}'"""
        return {f'[{inp}] - {doc.split(" | Next activity:")[0]}' for inp, doc in docs}

    def test_nenhum_prefixo_de_teste_vem_de_trace_da_kb(self, xes):
        traces = _toy_traces(xes)
        kb, test = lp.split_traces(traces, 0.3)
        kb_docs, _, _ = lp.process_prefixes(lp.build_prefixes(kb, 1, 1))
        test_docs, _, _ = lp.process_prefixes(lp.build_prefixes(test, 1, 1))
        kb_cases = {m for _, d in kb_docs for m in re.findall(r"'caid': '(case\d+)'", d)}
        test_cases = {m for _, d in test_docs for m in re.findall(r"'caid': '(case\d+)'", d)}
        assert kb_cases and test_cases
        assert kb_cases.isdisjoint(test_cases)
        assert self._questions(kb_docs).isdisjoint(self._questions(test_docs))

    def test_controle_negativo_detector_pega_vazamento(self, xes):
        # Prova que o teste acima tem dente: se a KB for montada com TODAS as traces (bug antigo),
        # as perguntas de teste aparecem literalmente na KB.
        traces = _toy_traces(xes)
        _, test = lp.split_traces(traces, 0.3)
        kb_docs, _, _ = lp.process_prefixes(lp.build_prefixes(traces, 1, 1))
        test_docs, _, _ = lp.process_prefixes(lp.build_prefixes(test, 1, 1))
        assert not self._questions(kb_docs).isdisjoint(self._questions(test_docs))


# ==============================================================================================
# CSV de teste
# ==============================================================================================
QUESTION_RE = re.compile(r"^\[[^\[\]]+\] - Values: \{.*\}$")


class TestGenerateCsv:
    def test_formato_do_csv(self, xes, tmp_path):
        traces = _toy_traces(xes)
        path = tmp_path / 'test.csv'
        keys, acts = lp.generate_csv_from_test_set(traces, str(path), size=10, base=1, gap=1)
        rows = list(csv.reader(open(path, newline='')))
        assert rows[0] == ['prefix', 'prediction']
        assert len(rows) == 11
        for question, label in rows[1:]:
            assert QUESTION_RE.match(question), question
            assert 'Next activity' not in question
            assert 'Values: Values:' not in question
            assert label in acts
        assert keys['caid'] == 'case:id'

    def test_size_maior_que_o_disponivel_e_truncado(self, xes, tmp_path):
        trace = xes.inner(xes.trace('c', ['A', 'B', 'C', 'D']))
        path = tmp_path / 'test.csv'
        lp.generate_csv_from_test_set([trace], str(path), size=300, base=1, gap=1)
        assert len(list(csv.reader(open(path)))) - 1 == 3   # [A]->B, [A,B]->C, [A,B,C]->D

    def test_load_csv_questions_le_o_que_generate_escreve(self, xes, tmp_path, monkeypatch):
        import utility as u
        traces = _toy_traces(xes)
        test_sets = tmp_path / 'tests' / 'test_sets'
        test_sets.mkdir(parents=True)
        lp.generate_csv_from_test_set(traces, str(test_sets / 'x.csv'), size=5, base=1, gap=1)
        # load_csv_questions resolve o caminho relativo a src/: aponta para tmp
        fake_src = tmp_path / 'src'
        fake_src.mkdir()
        monkeypatch.setattr(u, '__file__', str(fake_src / 'utility.py'))
        questions = u.load_csv_questions('x.csv')
        assert len(questions) == 5 and all(len(q) == 2 for q in questions)
