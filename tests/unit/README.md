# Testes unitários e de integração

```bash
pip install pytest qdrant-client langchain-core langchain-qdrant   # já estão no requirements.txt, menos o pytest
unzip logs.zip                                                      # opcional: habilita test_real_log_sepsis.py
pytest                     # suíte rápida (~5 s, sem GPU, sem API, sem servidor Qdrant)
pytest -m slow             # teste com embedding real (MiniLM) — rodar no Kaggle/Colab
pytest --runxfail          # mostra os bugs conhecidos falhando de verdade
```

| Arquivo | Cobre |
|---|---|
| `test_log_preprocessing.py` | extração de traces, prefixos (base/gap), rótulo, vazamento de atributos, dedup, split por trace, CSV |
| `test_vector_store.py` | armazenamento no Qdrant, rebuild, busca por similaridade, contrato CSV ↔ KB |
| `test_pipeline.py` | modelo correto (API e local), contexto no prompt, templates, parsing, reprodutibilidade |
| `test_oracle.py` | extração do `\boxed{}`, comparação exata, métricas, earlyness |
| `test_real_log_sepsis.py` | os mesmos contratos sobre o `sepsis.xes` real |

Os testes marcados `xfail(strict=True)` documentam bugs conhecidos. Quando um bug for corrigido, o teste
passa a dar **XPASS → falha**: é o lembrete para remover o `xfail`.
