import csv
import os
import random
import re


def read_event_log(filename):
    file_path = os.path.join(os.path.dirname(__file__), '..', 'logs', filename)
    with open(file_path, 'r') as file:
        return file.read()


def extract_traces(log_content):
    pattern_traces = re.findall(r"<trace>(.*?)</trace>", log_content, re.DOTALL) #Extrai as traces do log de eventos. Tudo dentro das flags <trace> e </trace> é considerado uma trace. O parâmetro re.DOTALL permite que o ponto (.) corresponda a qualquer caractere, incluindo quebras de linha.
    traces_result = []
    c = 1
    for single_trace in pattern_traces:
        events = re.findall(r"<event>(.*?)</event>", single_trace, re.DOTALL) #Extrai os eventos de cada trace. Tudo dentro das flags <event> e </event> é considerado um evento. O parâmetro re.DOTALL permite que o ponto (.) corresponda a qualquer caractere, incluindo quebras de linha.   
        if len(events) >= 2:
            traces_result.append(single_trace) #Inclui a trace na lista de resultados se ela tiver pelo menos 2 eventos.
            print(f"Processed trace: {c}")
            c += 1
    print(f"Total traces: {c}")
    return traces_result


def build_prefixes(traces, base=1, gap=3):
    prefixes = []
    for trace in traces:
        events = re.findall(r"<event>.*?</event>", trace, re.DOTALL) #Extrai os eventos de cada trace. Tudo dentro das flags <event> e </event> é considerado um evento. O parâmetro re.DOTALL permite que o ponto (.) corresponda a qualquer caractere, incluindo quebras de linha.
        if len(events) <= 2:
            continue
        else:
            for i in range(base, len(events), gap):
                i += 1
                if i <= len(events):
                    prefix = events[:i]  #Cria a lista de prefixos a partir dos eventos da trace. 
                    #O prefixo é formado pelos eventos desde o início da trace até o índice i (não inclusivo).
                    prefixes.append(''.join(prefix))
    return prefixes


# [FIX 2] Novo parâmetro include_label_attributes.
#         Antes: def process_prefixes(traces, log_name=None):
def process_prefixes(traces, log_name=None, include_label_attributes=False):
    # Resumo das correções nesta função:
    # [FIX 1] Cada prefixo <a1,...,an> vira o documento "[a1,...,a(n-1)] - Values: {...} | Next activity: an".
    #         Antes a chave era <a1,...,an> inteira e o rótulo era an: o input já continha a resposta.
    # [FIX 2] Por padrão os atributos vêm só dos eventos ANTERIORES ao evento-rótulo. O org:group do
    #         evento-rótulo, sozinho, acerta 61% do test set do sepsis (ex.: grupo E -> Release A).
    #         include_label_attributes=True reproduz o comportamento antigo (leitura literal do §4.1 do paper).
    # [FIX 3] Retorna uma LISTA de (prefixo, documento), com dedup por transição (prefixo, próxima atividade).
    #         Um dict indexado pelo prefixo descartaria continuações alternativas do mesmo prefixo.
    concept_name_pattern = re.compile(r'<string key="concept:name" value="(.*?)"\s*/>', re.DOTALL)
    lifecycle_transition_pattern = re.compile(r'<string key="lifecycle:transition" value="(.*?)"\s*/>', re.DOTALL)
    attribute_pattern = re.compile(r'key="(.*?)" value="(.*?)"\s*/>')
    # <string key="concept:name" value="..." /> no XML, capturando o valor atribuído a concept:name (o nome da atividade).
    # <string key="lifecycle:transition" value="..." /> no XML, capturando o valor atribuído a lifecycle:transition (o estado do ciclo de vida da atividade).
    # <string key="..." value="..." /> no XML, capturando todos os pares de chave-valor de atributos adicionais presentes no evento.

    # [FIX 3] seen_prefixes guarda transições (prefixo, próxima atividade) num set; results vira lista de pares.
    #         Antes: seen_prefixes = []   (lista de sequências; busca O(n) a cada prefixo)
    #         Antes: results = {}         (dict indexado pela sequência: 1 rótulo por sequência)
    seen_prefixes = set()           # Inicializa as estruturas de dados para armazenar as sequências de eventos já processados,
    results = []                    # resultados, chaves e atividades únicas.
    keys = {}
    activities = set()
    

    is_bpic_special = log_name and any(bpic_log in log_name.lower() for bpic_log in ['bpic_2013'])
    
    for i in range(0, len(traces)):
        trace = traces[i]
        events = re.findall(r"<event>.*?</event>", trace, re.DOTALL)
        cl_list = []
        # [FIX 2] kept_idx guarda a posição (em events) de cada atividade aceita em cl_list, para saber
        #         onde está o evento-rótulo mesmo quando eventos não-"complete" são descartados.
        #         Antes: for event in events:
        kept_idx = []
        for ev_idx, event in enumerate(events):
            n_before = len(cl_list)
            concept_name_match = concept_name_pattern.search(event)
            lifecycle_transition_match = lifecycle_transition_pattern.search(event)
            # Atribui a concept_name_match e lifecycle_transition_match os resultados das buscas usando os padrões definidos anteriormente.
            if concept_name_match:
                if is_bpic_special:  # For BPIC logs, include more attributes for uniqueness
                    event_name = concept_name_match.group(1)
                    resource_pattern = re.compile(r'<string key="org:resource" value="(.*?)"\s*/>')
                    role_pattern = re.compile(r'<string key="org:role" value="(.*?)"\s*/>')
        
                    resource_match = resource_pattern.search(event)
                    role_match = role_pattern.search(event)
                    #Inclui os atributos org:resource e org:role para logs BPIC, garantindo que a combinação de nome da atividade, transição do ciclo de vida, recurso e função seja única.
                    event_identifier = event_name
                    if lifecycle_transition_match:
                        event_identifier += f"+{lifecycle_transition_match.group(1)}"
                    if resource_match:
                        event_identifier += f"+{resource_match.group(1)}"
                    if role_match:
                        event_identifier += f"+{role_match.group(1)}"
                    # Constrói um identificador único composto juntando os valores com o separador + (ex.: NomeAtividade+complete+Niklas+A2_3).
                    cl_list.append(event_identifier) #Adiciona o identificador único à lista de atividades da trace.
                    activities.add(event_identifier) #Adiciona o identificador único ao conjunto de atividades únicas globais
                else:
                    if lifecycle_transition_match:
                        if lifecycle_transition_match.group(1).lower() == "complete":
                            event_name = concept_name_match.group(1)
                            cl_list.append(event_name)
                            activities.add(event_name)
                            # Se existir transição de ciclo de vida, só aceita o evento se a transição for especificamente igual a "complete".
                    elif not lifecycle_transition_match:
                        event_name = concept_name_match.group(1)
                        cl_list.append(event_name)
                        activities.add(event_name)
                        # Se o evento não tiver a tag de transição de ciclo de vida, aceita-o diretamente e adiciona o nome da atividade a cl_list e a activities.
            if len(cl_list) > n_before:   # [FIX 2] registra a posição do evento que acabou de ser aceito
                kept_idx.append(ev_idx)
        # [FIX 1] Precisa de pelo menos 1 atividade de input + 1 de rótulo.
        #         Antes: if not cl_list:
        if len(cl_list) < 2:
            continue
        # [FIX 1] O rótulo é a última atividade e o input é tudo ANTES dela (§4.1 do paper).
        #         Antes: cl_list_string = ','.join(cl_list)     (input incluía o rótulo)
        #                last_evt = cl_list[-1]
        last_evt = cl_list[-1]                     # rótulo = próxima atividade
        cl_list_string = ','.join(cl_list[:-1])    # input = prefixo SEM o rótulo
        # [FIX 3] Dedup pela transição (input, rótulo), não só pelo input.
        #         Antes: if cl_list_string in seen_prefixes:
        #                    continue
        #                else:
        #                    seen_prefixes.append(cl_list_string)
        transition = (cl_list_string, last_evt)
        if transition in seen_prefixes:
            continue
        seen_prefixes.add(transition)
        #Se esta transição (prefixo -> próxima atividade) já tiver sido processada, ignora-a.
        attr_vals = {}
        # [FIX 2] Atributos só dos eventos ANTES do evento-rótulo (padrão) ou de todos (modo antigo).
        #         Antes: for evt in events:     (incluía o evento-rótulo; o último valor sobrescreve os outros)
        attr_events = events if include_label_attributes else events[:kept_idx[-1]]
        for evt in attr_events:
            for attribute_match in attribute_pattern.findall(evt):       
                key, value = attribute_match    #Percorre todos os eventos do trace e encontra todos os pares key / value presentes.
                if key != "lifecycle:transition" and key != "concept:name":
                    key_initial = ''.join([part[:2] for part in key.split(':')])
                    attr_vals[key_initial] = value
                    keys[key_initial] = key
                    #Ignora as chaves padrão lifecycle:transition e concept:name.
                    #Cria uma sigla curta para a chave pegando nos 2 primeiros caracteres de cada parte separada por : (ex.: "org:resource" vira "orre").
                    #Guarda o valor no dicionário attr_vals e guarda o mapeamento da sigla para o nome completo em keys.
        
        # [FIX 3] Guarda o par (input, documento) numa lista.
        #         Antes: if cl_list_string not in results:
        #                    results[cl_list_string] = f'Values: {str(attr_vals)} | Next activity: {last_evt}'
        #                print(f"Processed prefix {i}/{len(traces)}")
        results.append((cl_list_string, f'Values: {str(attr_vals)} | Next activity: {last_evt}'))
        #Adiciona a sequência ao dicionário results formatada com os seus atributos e a última atividade executada, e imprime o progresso do processamento.
    # Antes: print(f"Total unique prefixes: {len(results)}")
    print(f"Total unique prefixes (transitions): {len(results)}")
    return results, keys, activities


# Test set proportion must be a decimal from 0 to 1
def generate_test_set(traces, test_set_proportion):
    test_set_size = int(len(traces) * test_set_proportion)
    test_set = random.sample(traces, test_set_size)
    return test_set
#Seleciona randomicamente uma proporção de traces para formar o conjunto de teste, com base na proporção fornecida (test_set_proportion).


# [FIX 4] Função nova (não existia). Usada no eval.py no lugar de generate_test_set.
def split_traces(traces, test_set_proportion):
    # [FIX 4] Split por TRACE antes de indexar: as traces de teste não entram na base de busca.
    #         Antes a KB era montada com 100% das traces e o teste era sorteado dessas mesmas traces
    #         (generate_test_set acima, mantida só por compatibilidade).
    #         Split por prefixo também vazaria: um prefixo mais longo da mesma trace contém a resposta.
    test_idx = set(random.sample(range(len(traces)), int(len(traces) * test_set_proportion)))
    kb_traces = [t for i, t in enumerate(traces) if i not in test_idx]
    test_traces = [t for i, t in enumerate(traces) if i in test_idx]
    return kb_traces, test_traces


# [FIX 8] Recebe base/gap (antes build_prefixes rodava sempre com o default gap=3).
# [FIX 2] Recebe include_label_attributes e repassa ao process_prefixes.
#         Antes: def generate_csv_from_test_set(test_set, test_path, size=300, log_name=None):
def generate_csv_from_test_set(test_set, test_path, size=300, log_name=None, base=1, gap=3,
                               include_label_attributes=False):
    # [FIX 8] Antes: test_set = build_prefixes(test_set)
    test_set = build_prefixes(test_set, base, gap)
    # [FIX 2] Antes: test_set, attr_keys, act_list = process_prefixes(test_set, log_name)
    test_set, attr_keys, act_list = process_prefixes(test_set, log_name, include_label_attributes)
    if size > len(test_set):
        size = len(test_set)
    # [FIX 3] process_prefixes agora devolve lista de pares, não dict.
    #         Antes: test_set = dict(random.sample(list(test_set.items()), size))
    test_set = random.sample(test_set, size)
    with open(test_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['prefix', 'prediction'])
        # [FIX 3] Antes: for prefix, prediction in test_set.items():
        for prefix, prediction in test_set:
            attributes = prediction.split('|')[0].strip()          # já começa com "Values: "
            next_activity = prediction.split('| Next activity: ')[1].strip()
            # [FIX 5] Mesmo formato dos documentos da KB e do exemplo do prompt: "[a,b,c] - Values: {...}".
            #         Antes saía "a,b,c - Values: Values: {...}".
            #         Antes: csvwriter.writerow([f'{prefix} - Values: {attributes}', next_activity])
            csvwriter.writerow([f'[{prefix}] - {attributes}', next_activity])
    # [FIX 9] Devolve atributos e atividades do test set para o eval.py (antes não retornava nada).
    return attr_keys, act_list
