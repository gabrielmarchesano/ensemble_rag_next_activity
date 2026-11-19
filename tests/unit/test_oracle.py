"""
Testes de oracle.py: extração da resposta do modelo, comparação com o rótulo, métricas e earlyness.
"""
import pytest

from oracle import VerificationOracle

SEPSIS_ACTS = {'ER Registration', 'ER Triage', 'ER Sepsis Triage', 'IV Liquid', 'IV Antibiotics',
               'LacticAcid', 'CRP', 'Leucocytes', 'Admission NC', 'Admission IC',
               'Release A', 'Release B', 'Release C', 'Release D', 'Release E', 'Return ER'}
PREFIX = "[ER Registration,ER Triage,CRP] - Values: {'orgr': 'B', 'x': 'a,b,c'}"


@pytest.fixture
def oracle():
    return VerificationOracle({'Activities': SEPSIS_ACTS})


class TestAnswerExtraction:
    @pytest.mark.parametrize('model_answer, expected, ok', [
        ('Reasoning...\n\\boxed{CRP}', 'CRP', True),
        # [FIX 10] o modelo repete o formato pedido antes da resposta final -> vale o ÚLTIMO \boxed
        ("Format: '\\boxed{predicted_activity_name}'. Final: \\boxed{Leucocytes}", 'Leucocytes', True),
        ('\\boxed{\\text{Release A}}', 'Release A', True),
        ('\\boxed{Admision NC}', 'Admission NC', True),        # typo corrigido pelo difflib (cutoff 0.6)
        ('\\boxed{Release B}', 'Release A', False),            # [FIX 11] igualdade exata, não substring
        ('Sem formato nenhum', 'CRP', False),
        ('\\boxed{None}', 'CRP', False),
        ('\\boxed{Pizza}', 'CRP', False),                      # fora do espaço de classes
    ])
    def test_verificacao(self, oracle, model_answer, expected, ok):
        assert oracle.verify_answer('prompt', PREFIX, model_answer, expected_answer=expected) is ok

    def test_resposta_invalida_vira_wrong_format_nas_metricas(self, oracle):
        oracle.verify_answer('p', PREFIX, 'blá', expected_answer='CRP')
        assert oracle.predicted_next_activities == ['Wrong format']
        assert oracle.true_next_activities == ['CRP']

    def test_rotulo_da_linha_tem_prioridade_sobre_o_dict(self, oracle):
        # [FIX 7] mesmo prefixo com rótulos diferentes em linhas diferentes do test set
        oracle.add_prefix_with_expected_answer_pair(PREFIX, 'Leucocytes')
        assert oracle.verify_answer('p', PREFIX, '\\boxed{CRP}', expected_answer='CRP') is True


class TestMetrics:
    def test_acuracia_e_f1(self, oracle):
        for ans, exp in [('CRP', 'CRP'), ('Leucocytes', 'Leucocytes'), ('CRP', 'CRP'), ('CRP', 'LacticAcid')]:
            oracle.verify_answer('p', PREFIX, f'\\boxed{{{ans}}}', expected_answer=exp)
        oracle.compute_stats()
        assert oracle.accuracy == pytest.approx(0.75)
        # F1 macro sobre {CRP, Leucocytes, LacticAcid}: CRP=0.8, Leucocytes=1.0, LacticAcid=0.0
        assert oracle.f1score_macro == pytest.approx((0.8 + 1.0 + 0.0) / 3)

    def test_classes_consideradas_incluem_wrong_format(self, oracle):
        oracle.verify_answer('p', PREFIX, 'lixo', expected_answer='CRP')
        oracle.verify_answer('p', PREFIX, '\\boxed{CRP}', expected_answer='CRP')
        oracle.compute_stats()
        assert set(oracle.considered_classes) == {'CRP', 'Wrong format'}
        assert oracle.accuracy == pytest.approx(0.5)


class TestEarlyness:
    def test_tamanho_do_prefixo_ignora_virgulas_dentro_de_values(self, oracle):
        assert oracle._calculate_prefix_length(PREFIX) == 3

    @pytest.mark.parametrize('length, bucket', [
        (1, 'Very Early (1-5)'), (5, 'Very Early (1-5)'), (6, 'Bucket 2 (6-10)'),
        (20, 'Bucket 3 (11-20)'), (30, 'Bucket 4 (21-30)'), (31, 'Very Late (31+)'),
    ])
    def test_buckets(self, oracle, length, bucket):
        assert oracle._get_earlyness_bucket(length) == bucket

    def test_ordem_dos_buckets_cobre_todos(self, oracle):
        order = oracle._get_bucket_order()
        assert all(oracle._get_earlyness_bucket(n) in order for n in range(1, 100))

    def test_metricas_por_bucket(self, oracle):
        short = "[ER Registration,ER Triage] - Values: {}"
        long_ = '[' + ','.join(['CRP'] * 40) + '] - Values: {}'
        oracle.verify_answer('p', short, '\\boxed{CRP}', expected_answer='CRP')
        oracle.verify_answer('p', long_, '\\boxed{CRP}', expected_answer='Leucocytes')
        oracle.compute_stats()
        assert oracle.earlyness_metrics['Very Early (1-5)']['accuracy'] == 1.0
        assert oracle.earlyness_metrics['Very Late (31+)']['accuracy'] == 0.0
