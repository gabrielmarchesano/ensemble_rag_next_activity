class QuotaExhausted(RuntimeError):
    """
    [GROQ] A API recusou por limite (HTTP 429) mesmo depois das retentativas: tipicamente o teto
    diário de tokens (TPD) do plano gratuito. Não é erro do modelo: a run deve PARAR (o checkpoint
    guarda o que já foi feito) e ser retomada depois, em vez de contar o prefixo como errado.
    """
