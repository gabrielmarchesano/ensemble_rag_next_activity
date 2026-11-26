from argparse import ArgumentParser
from dotenv import load_dotenv
import hashlib
import json
import os
import sys
import torch
import warnings
from typing import Dict, List

import ensemble as ens
import log_preprocessing as lp
import pipeline as p
import utility as u
import vector_store as vs
from errors import QuotaExhausted
from oracle import VerificationOracle

DEVICE = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
load_dotenv()
HF_AUTH = os.getenv('HF_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# [KAGGLE] Sem QDRANT_URL no .env, usa o Qdrant local em disco (sem servidor). Antes: os.getenv('QDRANT_URL')
URL = os.getenv('QDRANT_URL') or 'path:' + os.path.join(ROOT, 'qdrant_data')
# [KAGGLE] Antes: int(os.getenv('QDRANT_GRPC_PORT')) -> TypeError quando a variável não existia
GRPC_PORT = int(os.getenv('QDRANT_GRPC_PORT') or 6334)
COLLECTION_NAME = 'nap-rag'
SEED = 10
warnings.filterwarnings('ignore')


# [GROQ] checkpoint_path: cada prefixo avaliado vira uma linha JSON gravada na hora. Se a run cair (cota
#        diária da API, fim da sessão do Kaggle, queda de rede), rodar o MESMO comando de novo retoma de
#        onde parou: os prefixos já feitos são relidos do arquivo, sem nova chamada ao LLM.
def _load_checkpoint(path):
    done = {}
    if path and os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:   # linha cortada por uma queda no meio da escrita
                    continue
                done[rec['index']] = rec
    return done


# [ENSEMBLE] Recebe um "predictor" (ensemble.SinglePredictor / KNNPredictor / LLMEnsemble) em vez da
#            pipeline. Com SinglePredictor o fluxo é idêntico ao anterior.
#            Antes: def evaluate_pipeline(pipeline_instance, vect_db, num_chunks, list_questions, info_run,
#                                         earlyness_boundaries=None)
def evaluate_pipeline(predictor, list_questions: List, info_run: Dict, earlyness_boundaries=None,
                      checkpoint_path=None):
    oracle = VerificationOracle(info_run, earlyness_boundaries)
    tracker = ens.StrategyTracker()
    done = _load_checkpoint(checkpoint_path)
    if done:
        print(f'[checkpoint] Retomando {checkpoint_path}: {len(done)} de {len(list_questions)} prefixos já feitos.')
        info_run['Resumed From Checkpoint'] = f'{len(done)} prefixos'
    ckpt = open(checkpoint_path, 'a') if checkpoint_path else None
    count = 0

    try:
        for idx, el in enumerate(list_questions):
            prefix = el[0]
            expected_prediction = el[1]
            oracle.add_prefix_with_expected_answer_pair(prefix, expected_prediction)
            rec = done.get(idx)
            if rec is not None and rec['prefix'] == prefix:
                pred = ens.Prediction(rec['prompt'], rec['answer'], rec['already_parsed'], rec['details'])
            else:
                # Antes: prompt, answer = pipeline_instance.produce_answer(prefix, vect_db, num_chunks, info_run)
                try:
                    pred = predictor.predict(prefix, info_run)
                except QuotaExhausted as e:
                    print(f'\n[checkpoint] Run interrompida no prefixo {idx + 1} de {len(list_questions)}: {e}\n'
                          f'[checkpoint] {count} prefixos salvos em {checkpoint_path}. '
                          f'Rode o MESMO comando depois que o limite resetar.')
                    return None
                if ckpt:
                    ckpt.write(json.dumps({'index': idx, 'prefix': prefix, 'prompt': pred.prompt,
                                           'answer': pred.answer, 'already_parsed': pred.already_parsed,
                                           'details': pred.details}) + '\n')
                    ckpt.flush()
            # [FIX 7] Passa o rótulo da própria linha (o dict por prefixo trocava rótulos de prefixos repetidos)
            #         Antes: oracle.verify_answer(prompt, prefix, answer)
            oracle.verify_answer(pred.prompt, prefix, pred.answer, expected_answer=expected_prediction,
                                 already_parsed=pred.already_parsed, details=pred.details)
            tracker.update(pred.details, expected_prediction)
            count += 1
            print(f'Processing prediction for prefix {count} of {len(list_questions)}...')
    finally:
        if ckpt:
            ckpt.close()

    # [ENSEMBLE] Métricas extras (recall@k, acurácia por membro, teto do voto) entram no cabeçalho do relatório
    strategy_summary = tracker.summary()
    if strategy_summary:
        info_run['Strategy Summary'] = strategy_summary
    if hasattr(predictor, 'usage_report'):   # [GROQ] tokens gastos nesta sessão (não inclui os retomados)
        info_run['Token Usage (this session)'] = predictor.usage_report()
    print('Validation process completed. Check the output file.')
    oracle.write_results_to_file()
    print('\n' + '='*60)
    print('EARLYNESS ANALYSIS SUMMARY')
    print('='*60)
    
    summary = oracle.get_earlyness_summary()
    
    print(f"Overall Performance:")
    overall = summary['overall_metrics']
    print(f"  Total Samples: {overall['total_samples']}")
    print(f"  Accuracy: {overall['accuracy']:.4f}")
    print(f"  Precision (macro): {overall['precision_macro']:.4f}")
    print(f"  Recall (macro): {overall['recall_macro']:.4f}")
    print(f"  F1-score (macro): {overall['f1score_macro']:.4f}")
    
    print(f"\nPrefix Length Statistics:")
    stats = summary['prefix_length_stats']
    print(f"  Min Length: {stats['min']}")
    print(f"  Max Length: {stats['max']}")
    print(f"  Average Length: {stats['average']:.2f}")
    
    print(f"\nPerformance by Earlyness Buckets:")
    bucket_order = oracle._get_bucket_order()
    
    for bucket in bucket_order:
        if bucket in summary['earlyness_metrics']:
            metrics = summary['earlyness_metrics'][bucket]
            print(f"\n  {bucket}:")
            print(f"    Samples: {metrics['count']}")
            print(f"    Accuracy: {metrics['accuracy']:.4f}")
            print(f"    Precision: {metrics['precision_macro']:.4f}")
            print(f"    Recall: {metrics['recall_macro']:.4f}")
            print(f"    F1-score: {metrics['f1score_macro']:.4f}")

    if strategy_summary:
        print(f"\nStrategy Summary ({predictor.name}):")
        for key, value in strategy_summary.items():
            print(f"  {key}: {value}")
    if info_run.get('Token Usage (this session)'):
        print(f"\nToken Usage (this session): {info_run['Token Usage (this session)']}")

    print('='*60)
    return oracle


def parse_arguments():
    """Parse command line arguments for evaluation."""
    parser = ArgumentParser(description="Evaluate RAG Pipeline for Next Activity Prediction.")
    parser.add_argument('--embed_model_id', type=str, default='sentence-transformers/all-MiniLM-L12-v2',
                        help='Embedding model identifier')
    parser.add_argument('--vector_dimension', type=int, default=384,
                        help='Vector space dimension')
    parser.add_argument('--llm_id', type=str, default='gpt-4.1',
                        help='LLM model identifier')
    parser.add_argument('--model_max_length', type=int, help='Maximum input length (context window)',
                        default=128000)
    parser.add_argument('--num_documents_in_context', type=int, help='Number of documents in the context',
                        default=3)
    parser.add_argument('--log', type=str, help='The event log to use for the next activity prediction',
                        default='sepsis.xes')
    parser.add_argument('--prefix_base', type=int, help='Base number of events in a prefix trace',
                        default=1)
    # [FIX 8] O default declarado era 1, mas o build_prefixes era chamado sem argumentos e rodava com 3.
    #         Agora o argumento é usado de verdade; default 3 = o que o código efetivamente rodava.
    #         Antes: parser.add_argument('--prefix_gap', ..., default=1)
    parser.add_argument('--prefix_gap', type=int, help='Gap number of events in a prefix trace',
                        default=3)
    # [FIX 4] Argumentos novos (antes 0.3 e 300 eram fixos no código: generate_test_set(traces, 0.3) e size=300).

    parser.add_argument('--test_proportion', type=float, default=0.3,
                        help='Proportion of TRACES held out for testing (paper §5.2: 70/30)')
    parser.add_argument('--test_size', type=int, default=300,
                        help='Number of test prefixes sampled from the held-out traces')

    parser.add_argument('--max_new_tokens', type=int, help='Maximum number of tokens to generate',
                        default=2048)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--rebuild_db_and_tests', type=u.str2bool,
                        help='Rebuild the vector index and the test set', default=True)
    parser.add_argument('--evaluation_modality', type=str, default='evaluation-concept_names',
                        help='Evaluation modality to use (e.g., evaluation-concept_names, evaluation-attributes)')
    parser.add_argument('--rag', type=u.str2bool,
                        help='Support for Retrieval-Augmented Generation', default=True)
    parser.add_argument('--earlyness_buckets', type=str,
                        default='5,10,20,30',
                        help='Comma-separated boundaries for earlyness buckets (e.g., "5,10,20,30" creates buckets: 1-5, 6-10, 11-20, 21-30, 31+)')
    # [ENSEMBLE] Estratégia de predição. single = controle (comportamento antigo).
    parser.add_argument('--strategy', type=str, default='single', choices=['single', 'knn', 'ensemble'],
                        help='single: 1 LLM (controle); knn: voto nos rótulos recuperados, sem LLM; '
                             'ensemble: n LLMs + voto majoritário')
    parser.add_argument('--knn_k', type=int, default=10,
                        help='[knn] Número de vizinhos recuperados')
    parser.add_argument('--knn_weighting', type=str, default='count', choices=['count', 'score'],
                        help='[knn] count: 1 voto por vizinho; score: voto = similaridade de cosseno')
    parser.add_argument('--ensemble_llms', type=str, default=None,
                        help='[ensemble] LLMs separados por vírgula, ex.: "gpt-4.1,gpt-4o-mini,gemini-2.5-flash". '
                             'Repetir um id usa o mesmo modelo com contextos diferentes (modo partition)')
    parser.add_argument('--ensemble_mode', type=str, default='partition', choices=['shared', 'partition', 'bagging'],
                        help='[ensemble] shared: todos veem os mesmos top-k; partition: busca k*n e distribui '
                             'em rodízio; bagging: busca um pool top-K e sorteia k docs com reposição para '
                             'cada LLM (k = --num_documents_in_context por LLM)')
    # [BAGGING]
    parser.add_argument('--bag_pool_size', type=int, default=None,
                        help='[bagging] Tamanho K do pool recuperado. Default: k * n (o mesmo do partition)')
    parser.add_argument('--bag_replacement', type=u.str2bool, default=True,
                        help='[bagging] True = bootstrap (com reposição); False = sem reposição (pasting)')
    parser.add_argument('--bag_seed', type=int, default=SEED,
                        help='[bagging] Semente do sorteio das bags (independente da seed do split)')
    # [GROQ] Controle de cota da API e retomada
    parser.add_argument('--requests_per_minute', type=float, default=None,
                        help='[api] Throttle por modelo. Groq grátis: ~6 com k=3 (8K tokens/min ÷ ~1,3K por chamada)')
    parser.add_argument('--max_retries', type=int, default=8,
                        help='[api] Retentativas do SDK em 429/timeout (respeita o retry-after)')
    parser.add_argument('--reasoning_effort', type=str, default=None,
                        help="[groq] gpt-oss: 'low' | 'medium' | 'high'. Afeta o resultado e o gasto de tokens")
    parser.add_argument('--resume', type=u.str2bool, default=True,
                        help='Retoma do checkpoint se a mesma configuração já rodou em parte (False = do zero)')
    args = parser.parse_args()
    if args.strategy == 'ensemble' and not args.ensemble_llms:
        parser.error('--strategy ensemble exige --ensemble_llms')
    if args.strategy == 'ensemble' and not args.rag:
        parser.error('--strategy ensemble exige --rag True')
    if args.ensemble_mode != 'bagging' and (args.bag_pool_size is not None or not args.bag_replacement):
        parser.error('--bag_pool_size/--bag_replacement só valem com --ensemble_mode bagging')
    return args


# [GROQ] Só entram os parâmetros que mudam o resultado (throttle e retentativas não mudam).
CHECKPOINT_KEYS = ('log', 'evaluation_modality', 'strategy', 'llm_id', 'ensemble_llms', 'ensemble_mode',
                   'num_documents_in_context', 'knn_k', 'knn_weighting', 'embed_model_id', 'rag',
                   'max_new_tokens', 'reasoning_effort', 'prefix_base', 'prefix_gap', 'test_size',
                   'test_proportion')


def checkpoint_path_for(args, test_set_path):
    cfg = {k: getattr(args, k, None) for k in CHECKPOINT_KEYS}
    if cfg['strategy'] != 'single':
        cfg['llm_id'] = None   # o --llm_id só é usado pela estratégia single
    # [BAGGING] Parâmetros do sorteio entram na chave SÓ no modo bagging: assim os checkpoints de
    #           shared/partition já em andamento continuam com a mesma chave e retomam normalmente.
    if cfg['strategy'] == 'ensemble' and cfg['ensemble_mode'] == 'bagging':
        n = len([m for m in args.ensemble_llms.split(',') if m.strip()])
        cfg['bag_pool_size'] = args.bag_pool_size if args.bag_pool_size is not None else args.num_documents_in_context * n
        cfg['bag_replacement'] = args.bag_replacement
        cfg['bag_seed'] = args.bag_seed
    with open(test_set_path, 'rb') as f:
        cfg['test_set_md5'] = hashlib.md5(f.read()).hexdigest()
    key = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
    folder = os.path.join(ROOT, 'tests', 'checkpoints')
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{args.log.split('.xes')[0]}_{args.strategy}_{key}.jsonl")


# [ENSEMBLE] Monta o predictor da estratégia escolhida. Só carrega LLM quando a estratégia usa LLM
#            (o kNN roda sem GPU e sem chave de API).
def build_predictor(args, q_store, num_docs, max_new_tokens):
    # [GROQ] parâmetros de API repassados a todo RAGPipeline (locais e outros provedores ignoram)
    api_kwargs = {'requests_per_minute': args.requests_per_minute, 'max_retries': args.max_retries,
                  'reasoning_effort': args.reasoning_effort}
    if args.strategy == 'knn':
        return ens.KNNPredictor(q_store, k=args.knn_k, weighting=args.knn_weighting), {
            'LLM ID': None, 'KNN k': args.knn_k, 'KNN Weighting': args.knn_weighting}
    if args.strategy == 'ensemble':
        ids = [m.strip() for m in args.ensemble_llms.split(',') if m.strip()]
        cache = {}   # um id repetido reaproveita a mesma instância (não carrega o modelo local duas vezes)
        for model_id in ids:
            if model_id not in cache:
                cache[model_id] = p.RAGPipeline(model_id, max_new_tokens, True, **api_kwargs)
        predictor = ens.LLMEnsemble([(m, cache[m]) for m in ids], q_store, docs_per_llm=num_docs,
                                    mode=args.ensemble_mode, pool_size=args.bag_pool_size,
                                    bag_replacement=args.bag_replacement, bag_seed=args.bag_seed)
        info = {'LLM ID': ids, 'Ensemble Mode': args.ensemble_mode,
                'Ensemble Aggregation': 'majority vote (tie: member order)',
                'Documents Retrieved per Prefix': predictor.retrieved_per_prefix()}
        if args.ensemble_mode == 'bagging':   # [BAGGING]
            info.update({'Bagging Pool Size': predictor.pool_size,
                         'Bagging Sampling': 'with replacement (bootstrap)' if args.bag_replacement
                                             else 'without replacement (pasting)',
                         'Bagging Seed': args.bag_seed})
        return predictor, info
    pipeline_instance = p.RAGPipeline(args.llm_id, max_new_tokens, args.rag, **api_kwargs)
    return ens.SinglePredictor(pipeline_instance, q_store, num_docs), {'LLM ID': args.llm_id}


def main():
    args = parse_arguments()
    embed_model_id = args.embed_model_id
    embed_model, actual_dimension = p.initialize_embedding_model(embed_model_id, DEVICE, args.batch_size)
    space_dimension = actual_dimension
    print(f"Using embedding dimension: {space_dimension}")
    
    rag = args.rag
    q_client, q_store = vs.initialize_vector_store(URL, GRPC_PORT, COLLECTION_NAME, 
                                                   embed_model, space_dimension, 
                                                   args.rebuild_db_and_tests)
    num_docs = args.num_documents_in_context
    test_set_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'test_sets',
                                 f"test_set_{args.log.split('.xes')[0]}_{args.evaluation_modality}.csv")
    test_set_size = 0
    traces_to_store_size = 0

    # [FIX 4] Split por trace ANTES de indexar: a KB só recebe traces de treino.
    # [FIX 9] O log é sempre lido (é barato), para 'Activities'/'Event Attributes' existirem mesmo com
    #         --rebuild_db_and_tests False (antes o oracle quebrava com KeyError nesse caso).
    # Antes:
    #     event_attributes = []
    #     activities_set = set()
    #     total_traces_size = 0
    #     test_set_size = 0
    #     traces_to_store_size = 0
    #     if args.rebuild_db_and_tests:
    #         content = lp.read_event_log(args.log)
    #         traces = lp.extract_traces(content)
    #         prefixes = lp.build_prefixes(traces)                       # 100% das traces, gap fixo em 3
    #         prefixes, event_attributes, activities_set = lp.process_prefixes(prefixes, args.log)
    #         traces_to_store_size = len(prefixes)
    #         vs.store_traces(prefixes, q_client, args.log, embed_model, COLLECTION_NAME)   # indexa tudo
    #         test_set = lp.generate_test_set(traces, 0.3)               # teste sorteado das MESMAS traces
    #         test_set_size = len(test_set)
    #         lp.generate_csv_from_test_set(test_set, test_set_path, log_name=args.log)
    content = lp.read_event_log(args.log)
    traces = lp.extract_traces(content)
    total_traces_size = len(traces)
    kb_traces, test_traces = lp.split_traces(traces, args.test_proportion)
    kb_prefixes = lp.build_prefixes(kb_traces, args.prefix_base, args.prefix_gap)
    kb_docs, event_attributes, activities_set = lp.process_prefixes(kb_prefixes, args.log)
    if args.rebuild_db_and_tests:
        traces_to_store_size = vs.store_traces(kb_docs, q_client, args.log, embed_model, COLLECTION_NAME)
        test_attrs, test_acts = lp.generate_csv_from_test_set(test_traces, test_set_path, size=args.test_size,
                                                              log_name=args.log, base=args.prefix_base,
                                                              gap=args.prefix_gap)
    else:
        # [ENSEMBLE] Sem rebuild, o espaço de classes ficava só com as atividades da KB, e a lista de
        #            atividades entra no prompt e no parsing: o controle (rebuild True) e as runs seguintes
        #            (rebuild False) viam prompts diferentes. O split é determinístico (seed), então dá para
        #            recalcular as atividades/atributos do teste sem regravar o CSV.
        _, test_attrs, test_acts = lp.process_prefixes(
            lp.build_prefixes(test_traces, args.prefix_base, args.prefix_gap), args.log)
    event_attributes = {**test_attrs, **event_attributes}
    activities_set = activities_set | test_acts   # espaço de classes = todas as atividades do log
    test_set_size = len(test_traces)

    max_new_tokens = args.max_new_tokens
    os.environ['HF_TOKEN'] = HF_AUTH if HF_AUTH else ""
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY if OPENAI_API_KEY else ""
    # [ENSEMBLE] Antes: model_id = args.llm_id
    #                   pipeline_instance = p.RAGPipeline(model_id, max_new_tokens, rag)
    predictor, strategy_info = build_predictor(args, q_store, num_docs, max_new_tokens)
    run_data = {
        'Strategy': args.strategy,                                    # [ENSEMBLE]
        'Batch Size': args.batch_size,
        'Embedding Model ID': embed_model_id,
        'Vector Space Dimension': space_dimension,
        'Evaluation Modality': args.evaluation_modality,
        'Event Log': args.log,
        'Total Traces in Log': total_traces_size,
        'Test Set Size': test_set_size,
        'KB Traces': len(kb_traces),                                  # [FIX 4] campo novo no relatório
        'Traces Stored Size': traces_to_store_size,
        # [ENSEMBLE] Removido: 'Include Label Attributes' (a flag não existe mais -> AttributeError)
        **strategy_info,                                              # [ENSEMBLE] LLM ID(s) e parâmetros da estratégia
        'Context Window LLM': args.model_max_length,
        'Max Generated Tokens LLM': max_new_tokens,
        'Number of Documents in the Context': num_docs,
        'Prefix Base': args.prefix_base,
        'Prefix Gap': args.prefix_gap,
        'Rebuilt Vector Index and Test Set': args.rebuild_db_and_tests,
        'RAG': rag,
        'Reasoning Effort': args.reasoning_effort,                    # [GROQ] muda o resultado: registrar
    }
    earlyness_boundaries = [int(x.strip()) for x in args.earlyness_buckets.split(',')]
    run_data['Earlyness Boundaries'] = earlyness_boundaries
    
    if event_attributes:
        run_data['Event Attributes'] = str(event_attributes)
        # [ENSEMBLE] sorted(): a ordem de str(set) muda a cada processo (PYTHONHASHSEED é setado tarde
        #            demais em seed_everything), e a lista entra no prompt. Sem isso, controle e ensemble
        #            rodariam com prompts diferentes. Antes: run_data['Activities'] = activities_set
        run_data['Activities'] = sorted(activities_set)

    test_list = u.load_csv_questions(test_set_path)
    print(f"Starting evaluation with {len(test_list)} test cases...")
    print(f"Using earlyness boundaries: {earlyness_boundaries}")
    # [GROQ] Checkpoint por configuração + conteúdo do test set: o mesmo comando retoma; qualquer
    #        mudança de configuração (ou de test set) começa um arquivo novo.
    ckpt_path = checkpoint_path_for(args, test_set_path)
    if not args.resume and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    run_data['Checkpoint'] = ckpt_path
    oracle = evaluate_pipeline(predictor, test_list, run_data, earlyness_boundaries, checkpoint_path=ckpt_path)
    if oracle is None:
        sys.exit(3)   # interrompida por cota: código != 0 para scripts/notebooks perceberem


if __name__ == "__main__":
    u.seed_everything(SEED)
    main()