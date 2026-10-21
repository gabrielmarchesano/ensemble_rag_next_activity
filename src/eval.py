from argparse import ArgumentParser
from dotenv import load_dotenv
import os
import torch
import warnings
from typing import Dict, List

import log_preprocessing as lp
import pipeline as p
import utility as u
import vector_store as vs
from oracle import VerificationOracle

DEVICE = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
load_dotenv()
HF_AUTH = os.getenv('HF_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
URL = os.getenv('QDRANT_URL')
GRPC_PORT = int(os.getenv('QDRANT_GRPC_PORT'))
COLLECTION_NAME = 'nap-rag'
SEED = 10
warnings.filterwarnings('ignore')


def evaluate_pipeline(pipeline_instance: p.RAGPipeline, vect_db, num_chunks: int, 
                     list_questions: List, info_run: Dict, earlyness_boundaries=None):
    oracle = VerificationOracle(info_run, earlyness_boundaries)
    count = 0
    
    for el in list_questions:
        prefix = el[0]
        expected_prediction = el[1]
        oracle.add_prefix_with_expected_answer_pair(prefix, expected_prediction)
        prompt, answer = pipeline_instance.produce_answer(prefix, vect_db, num_chunks, info_run)
        # [FIX 7] Passa o rótulo da própria linha (o dict por prefixo trocava rótulos de prefixos repetidos)
        #         Antes: oracle.verify_answer(prompt, prefix, answer)
        oracle.verify_answer(prompt, prefix, answer, expected_answer=expected_prediction)
        count += 1
        print(f'Processing prediction for prefix {count} of {len(list_questions)}...')

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
    # [FIX 2] --include_label_attributes True volta ao comportamento antigo dos atributos.
    parser.add_argument('--test_proportion', type=float, default=0.3,
                        help='Proportion of TRACES held out for testing (paper §5.2: 70/30)')
    parser.add_argument('--test_size', type=int, default=300,
                        help='Number of test prefixes sampled from the held-out traces')
    parser.add_argument('--include_label_attributes', type=u.str2bool, default=False,
                        help='Old behaviour: include the attributes of the label event in Values (leaks the label)')
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
    args = parser.parse_args()
    return args


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
    kb_docs, event_attributes, activities_set = lp.process_prefixes(kb_prefixes, args.log,
                                                                    args.include_label_attributes)
    if args.rebuild_db_and_tests:
        traces_to_store_size = vs.store_traces(kb_docs, q_client, args.log, embed_model, COLLECTION_NAME)
        test_attrs, test_acts = lp.generate_csv_from_test_set(test_traces, test_set_path, size=args.test_size,
                                                              log_name=args.log, base=args.prefix_base,
                                                              gap=args.prefix_gap,
                                                              include_label_attributes=args.include_label_attributes)
        event_attributes = {**test_attrs, **event_attributes}
        activities_set = activities_set | test_acts   # espaço de classes = todas as atividades do log
    test_set_size = len(test_traces)

    model_id = args.llm_id
    max_new_tokens = args.max_new_tokens
    os.environ['HF_TOKEN'] = HF_AUTH if HF_AUTH else ""
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY if OPENAI_API_KEY else ""
    pipeline_instance = p.RAGPipeline(model_id, max_new_tokens, rag)
    run_data = {
        'Batch Size': args.batch_size,
        'Embedding Model ID': embed_model_id,
        'Vector Space Dimension': space_dimension,
        'Evaluation Modality': args.evaluation_modality,
        'Event Log': args.log,
        'Total Traces in Log': total_traces_size,
        'Test Set Size': test_set_size,
        'KB Traces': len(kb_traces),                                  # [FIX 4] campo novo no relatório
        'Traces Stored Size': traces_to_store_size,
        'Include Label Attributes': args.include_label_attributes,    # [FIX 2] campo novo no relatório
        'LLM ID': model_id,
        'Context Window LLM': args.model_max_length,
        'Max Generated Tokens LLM': max_new_tokens,
        'Number of Documents in the Context': num_docs,
        'Prefix Base': args.prefix_base,
        'Prefix Gap': args.prefix_gap,
        'Rebuilt Vector Index and Test Set': args.rebuild_db_and_tests,
        'RAG': rag
    }
    earlyness_boundaries = [int(x.strip()) for x in args.earlyness_buckets.split(',')]
    run_data['Earlyness Boundaries'] = earlyness_boundaries
    
    if event_attributes:
        run_data['Event Attributes'] = str(event_attributes)
        run_data['Activities'] = activities_set

    test_list = u.load_csv_questions(test_set_path)
    print(f"Starting evaluation with {len(test_list)} test cases...")
    print(f"Using earlyness boundaries: {earlyness_boundaries}")
    evaluate_pipeline(pipeline_instance, q_store, num_docs, test_list, run_data, earlyness_boundaries)


if __name__ == "__main__":
    u.seed_everything(SEED)
    main()
