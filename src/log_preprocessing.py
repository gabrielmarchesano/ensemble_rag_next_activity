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


def process_prefixes(traces, log_name=None):
    concept_name_pattern = re.compile(r'<string key="concept:name" value="(.*?)"\s*/>', re.DOTALL)
    lifecycle_transition_pattern = re.compile(r'<string key="lifecycle:transition" value="(.*?)"\s*/>', re.DOTALL)
    attribute_pattern = re.compile(r'key="(.*?)" value="(.*?)"\s*/>')
    # <string key="concept:name" value="..." /> no XML, capturando o valor atribuído a concept:name (o nome da atividade).
    # <string key="lifecycle:transition" value="..." /> no XML, capturando o valor atribuído a lifecycle:transition (o estado do ciclo de vida da atividade).
    # <string key="..." value="..." /> no XML, capturando todos os pares de chave-valor de atributos adicionais presentes no evento.

    seen_prefixes = []              # Inicializa as estruturas de dados para armazenar as sequências de eventos já processados,
    results = {}                    # resultados, chaves e atividades únicas.
    keys = {}
    activities = set()
    

    is_bpic_special = log_name and any(bpic_log in log_name.lower() for bpic_log in ['bpic_2013'])
    
    for i in range(0, len(traces)):
        trace = traces[i]
        events = re.findall(r"<event>.*?</event>", trace, re.DOTALL)
        cl_list = []
        for event in events:
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
        if not cl_list:
            continue
        #Se nenhum evento válido foi extraído deste trace, salta para o próximo trace.
        cl_list_string = ','.join(cl_list) #Converte a lista de eventos numa única string separada por vírgulas (ex.: "AtividadeA,AtividadeB,AtividadeC").
        if cl_list_string in seen_prefixes:
            continue
        else:
            seen_prefixes.append(cl_list_string)
        #Se esta sequência de eventos já tiver sido processada anteriormente, ignora-a. 
        #Caso contrário, adiciona-a à lista seen_prefixes.
        last_evt = cl_list[-1]
        attr_vals = {}
        #Identifica a última atividade da sequência (last_evt) e cria o dicionário attr_vals para guardar os atributos adicionais.
        for evt in events:
            for attribute_match in attribute_pattern.findall(evt):       
                key, value = attribute_match    #Percorre todos os eventos do trace e encontra todos os pares key / value presentes.
                if key != "lifecycle:transition" and key != "concept:name":
                    key_initial = ''.join([part[:2] for part in key.split(':')])
                    attr_vals[key_initial] = value
                    keys[key_initial] = key
                    #Ignora as chaves padrão lifecycle:transition e concept:name.
                    #Cria uma sigla curta para a chave pegando nos 2 primeiros caracteres de cada parte separada por : (ex.: "org:resource" vira "orre").
                    #Guarda o valor no dicionário attr_vals e guarda o mapeamento da sigla para o nome completo em keys.
        
        if cl_list_string not in results:
            results[cl_list_string] = f'Values: {str(attr_vals)} | Next activity: {last_evt}'
        print(f"Processed prefix {i}/{len(traces)}")
        #Adiciona a sequência ao dicionário results formatada com os seus atributos e a última atividade executada, e imprime o progresso do processamento.
    print(f"Total unique prefixes: {len(results)}")
    return results, keys, activities


# Test set proportion must be a decimal from 0 to 1
def generate_test_set(traces, test_set_proportion):
    test_set_size = int(len(traces) * test_set_proportion)
    test_set = random.sample(traces, test_set_size)
    return test_set
#Seleciona randomicamente uma proporção de traces para formar o conjunto de teste, com base na proporção fornecida (test_set_proportion).

def generate_csv_from_test_set(test_set, test_path, size=300, log_name=None):
    test_set = build_prefixes(test_set)
    test_set, attr_keys, act_list = process_prefixes(test_set, log_name)
    if size > len(test_set):
        size = len(test_set)
    test_set = dict(random.sample(list(test_set.items()), size))
    with open(test_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['prefix', 'prediction'])
        for prefix, prediction in test_set.items():
            attributes = prediction.split('|')[0].strip()
            next_activity = prediction.split('| Next activity: ')[1].strip()
            csvwriter.writerow([f'{prefix} - Values: {attributes}', next_activity])
