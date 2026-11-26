#!/bin/bash
# [ENSEMBLE] Escada controle -> kNN -> ensemble no MESMO test set (sepsis, split por trace, seed 10).
# Rodar da raiz do repositório. A 1ª linha reconstrói a KB e o test set; as demais reaproveitam
# (--rebuild_db_and_tests False), então todas as estratégias são comparadas nos mesmos prefixos.
#
# [GROQ] Plano gratuito, POR MODELO: 30 req/min, 8K tokens/min, 200K tokens/dia. Com k=3 cada chamada
#        gasta ~1,3K tokens (prompt ~0,9K + resposta), então ~150 prefixos/dia por modelo. Uma run de 300
#        prefixos leva ~2 dias: quando a cota acabar a run sai com código 3; rode o MESMO comando no dia
#        seguinte e ela continua do checkpoint (tests/checkpoints/).
#        Ensemble: use 3 modelos DIFERENTES (cada um tem a própria cota). O mesmo modelo 3x divide uma
#        cota só e leva o triplo do tempo.
#
# Pré-requisitos: pip install -r requirements.txt; GROQ_API_KEY no ambiente ou no .env;
#                 sem servidor Qdrant: deixe QDRANT_URL vazio (usa ./qdrant_data) ou QDRANT_URL=path:/pasta

LOG=sepsis.xes
MOD=evaluation-attributes
K=3                                   # documentos no contexto de CADA LLM
GROQ="--requests_per_minute 6 --reasoning_effort low --max_new_tokens 1024"
COMMON="--log $LOG --evaluation_modality $MOD --num_documents_in_context $K"

# 0) Controle: 1 LLM, top-3 (reconstrói KB + test set)
python3 src/eval.py $COMMON --rebuild_db_and_tests True --strategy single --llm_id openai/gpt-oss-20b $GROQ

# 1) kNN baseline (sem LLM, sem cota): voto nos rótulos recuperados
python3 src/eval.py $COMMON --rebuild_db_and_tests False --strategy knn --knn_k 3
python3 src/eval.py $COMMON --rebuild_db_and_tests False --strategy knn --knn_k 10
python3 src/eval.py $COMMON --rebuild_db_and_tests False --strategy knn --knn_k 10 --knn_weighting score

# 2) Ensemble, contexto compartilhado: 3 LLMs veem os MESMOS top-3 (isola o ganho do voto)
python3 src/eval.py $COMMON --rebuild_db_and_tests False --strategy ensemble --ensemble_mode shared \
    --ensemble_llms openai/gpt-oss-120b,openai/gpt-oss-20b,qwen/qwen3.8-27b $GROQ

# 3) Ensemble, contexto particionado: busca 9 docs e distribui 3 para cada LLM (ranks 1,4,7 / 2,5,8 / 3,6,9)
python3 src/eval.py $COMMON --rebuild_db_and_tests False --strategy ensemble --ensemble_mode partition \
    --ensemble_llms openai/gpt-oss-120b,openai/gpt-oss-20b,qwen/qwen3.8-27b $GROQ

# 4) [BAGGING] Bootstrap do contexto: busca o MESMO pool de 9 do partition e sorteia, para cada LLM,
#    3 docs COM reposição (sorteio fixo por prefixo, --bag_seed). Partition x bagging = mesmo pool,
#    mesmos modelos; só muda "fatia fixa" x "sorteio".
python3 src/eval.py $COMMON --rebuild_db_and_tests False --strategy ensemble --ensemble_mode bagging \
    --ensemble_llms openai/gpt-oss-120b,openai/gpt-oss-20b,qwen/qwen3.8-27b $GROQ

# 4b) Teto do pool sem LLM (sem cota): label_in_context_rate do kNN com k = tamanho do pool
python3 src/eval.py $COMMON --rebuild_db_and_tests False --strategy knn --knn_k 9

# 4c) Ablações (uma de cada vez, depois da 4):
#   pool maior:        --bag_pool_size 20
#   sem reposição:     --bag_replacement False          (pasting)
#   outra semente:     --bag_seed 11                    (variância do sorteio; rodar 2-3 seeds)

# 5) [BAGGING] Homogêneo (bagging "clássico": mesmo modelo, diversidade só das bags) -> Kaggle.
#    Na Groq o mesmo id 5x divide UMA cota (5x mais lento). Ex.:
#   python3 src/eval.py $COMMON --rebuild_db_and_tests False --strategy ensemble --ensemble_mode bagging \
#       --ensemble_llms Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-7B-Instruct

# Kaggle (GPU T4 x2, modelos locais em 4-bit): sem os parâmetros $GROQ, ex.:
#   python3 src/eval.py $COMMON --rebuild_db_and_tests False --strategy ensemble --ensemble_mode shared \
#       --ensemble_llms Qwen/Qwen2.5-7B-Instruct,microsoft/phi-4,meta-llama/Llama-3.1-8B-Instruct