from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore


def initialize_vector_store(url, grpc_port, collection_name, embed_model, dimension, rebuild_db):
    client = QdrantClient(url, grpc_port=grpc_port, prefer_grpc=True)
    if rebuild_db:
        try:
            client.delete_collection(collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except Exception as e:
            print(f"Collection {collection_name} doesn't exist or couldn't be deleted: {e}")

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )
        print(f"Created new collection: {collection_name} with dimension: {dimension}")

    try:
        store = QdrantVectorStore(client, collection_name=collection_name, embedding=embed_model)
    except Exception as e:
        if "dimensions" in str(e).lower() and "force_recreate" in str(e):
            print(f"Dimension mismatch detected. Recreating collection with correct dimension: {dimension}")
            try:
                client.delete_collection(collection_name)
                print(f"Deleted existing collection: {collection_name}")
            except Exception as delete_e:
                print(f"Warning: Could not delete collection: {delete_e}")
            
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
            print(f"Created new collection: {collection_name} with dimension: {dimension}")
            store = QdrantVectorStore(client, collection_name=collection_name, embedding=embed_model)
        else:
            raise e

    return client, store


def store_prefixes(prefixes, qdrant_client, log_name, embed_model, collection_name):
    points = []
    identifier = 0
    for p in prefixes:
        metadata = {'page_content': p, 'name': f'{log_name} Chunk {identifier}'}
        point = models.PointStruct(
            id=identifier,
            vector=embed_model.embed_documents([p])[0],
            payload=metadata
        )
        print(f'Processing point {identifier + 1} of {len(prefixes)}...')
        points.append(point)
        identifier += 1

    print('Storing points into the vector store...')
    qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )

    return identifier


def store_xes_traces(traces, qdrant_client, log_name, embed_model, collection_name):
    points = []
    identifier = 0
    for t in traces:
        t = ''.join(t)
        metadata = {'page_content': t, 'name': f'{log_name} Trace {identifier}'}
        point = models.PointStruct(
            id=identifier,
            vector=embed_model.embed_documents([t])[0],
            payload=metadata
        )
        print(f'Processing point for trace {identifier + 1} of {len(traces)}...')
        points.append(point)
        identifier += 1

    print('Storing points into the vector store...')
    qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )

    return identifier


def store_traces(traces, qdrant_client, log_name, embed_model, collection_name):
    # [FIX 3] `traces` agora é a lista de (prefixo, documento) do process_prefixes.
    #         Antes o dedup era pelo prefixo: com o rótulo fora da chave, isso guardaria só UMA
    #         continuação por prefixo e apagaria as alternativas que o LLM precisa ver.
    # [PERF]  Embeddings em lote e upsert em blocos (antes: 1 chamada de embedding por documento).
    # Antes:
    #     points = []
    #     identifier = 0
    #     stored = []
    #     for t, prediction in traces.items():
    #         content = f'[{t}] - {prediction}'
    #         metadata = {'page_content': content, 'name': f'{log_name} Trace {identifier}'}
    #         if t not in stored:
    #             point = models.PointStruct(
    #                 id=identifier,
    #                 vector=embed_model.embed_documents([t])[0],
    #                 payload=metadata
    #             )
    #             print(f'Processing point for trace {identifier + 1} of {len(traces)}...')
    #             points.append(point)
    #             stored.append(t)
    #             identifier += 1
    #     print('Storing points into the vector store...')
    #     qdrant_client.upsert(collection_name=collection_name, points=points)
    #     return identifier
    items = list(traces.items()) if isinstance(traces, dict) else list(traces)   # aceita dict (formato antigo) ou lista
    seen, keys_to_embed, contents = set(), [], []
    for t, prediction in items:
        content = f'[{t}] - {prediction}'
        if content in seen:
            continue
        seen.add(content)
        keys_to_embed.append(t)   # o embedding continua sendo só da sequência de atividades
        contents.append(content)

    print(f'Embedding {len(contents)} documents...')
    vectors = embed_model.embed_documents(keys_to_embed)
    points = [
        models.PointStruct(id=i, vector=v, payload={'page_content': c, 'name': f'{log_name} Trace {i}'})
        for i, (v, c) in enumerate(zip(vectors, contents))
    ]

    print('Storing points into the vector store...')
    for start in range(0, len(points), 512):
        qdrant_client.upsert(collection_name=collection_name, points=points[start:start + 512])

    return len(points)


def retrieve_context(vector_index, query, num_chunks, key=None, search_filter=None):
    # [FIX 6] Isola a sequência de atividades pelo separador real (' - Values:') e tira os colchetes,
    #         para o embedding da query ter o mesmo formato do embedding dos documentos.
    #         split('-') cortava no primeiro hífen, inclusive dentro de nomes de atividade (BPIC).
    #         Antes: query = query.split('-')[0].strip()
    query = query.split(' - Values:')[0].strip().strip('[]').strip()
    retrieved = vector_index.similarity_search(query, num_chunks)
    if key is not None and search_filter is not None:
        filter_ = models.Filter(
            must=[
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=search_filter)
                )
            ]
        )
        meta_retrieved = vector_index.similarity_search(query, filter=filter_, k=num_chunks)
        if len(meta_retrieved) > 0:
            retrieved = meta_retrieved
    retrieved_text = ''
    for i in range(len(retrieved)):
        content = retrieved[i].page_content
        retrieved_text += f'\n{i+1}. {content}'

    return retrieved_text


def delete_qdrant_collection(q_client, q_collection_name):
    q_client.delete_collection(q_collection_name)
