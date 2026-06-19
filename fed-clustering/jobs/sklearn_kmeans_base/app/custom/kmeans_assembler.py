# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Optional
import logging
import requests
import numpy as np
from sklearn.cluster import KMeans

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.aggregators.assembler import Assembler
from nvflare.app_common.app_constant import AppConstants
from dfa_lib_python.dataflow import Dataflow
from dfa_lib_python.transformation import Transformation
from dfa_lib_python.attribute import Attribute
from dfa_lib_python.attribute_type import AttributeType
from dfa_lib_python.set import Set
from dfa_lib_python.set_type import SetType
from dfa_lib_python.task import Task
from dfa_lib_python.dependency import Dependency
from dfa_lib_python.dataset import DataSet
from dfa_lib_python.element import Element
from dfa_lib_python.task_status import TaskStatus
from dfa_lib_python.extractor_extension import ExtractorExtension

from time import perf_counter
import pickle
import datetime

from pyDataverse.api import NativeApi
from pyDataverse.models import Dataverse, Dataset, Datafile
from pyDataverse.utils import read_file

import json
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Cria uma sessão HTTP persistente para reutilizar conexões TCP
session = requests.Session()

# Define estratégia de retry (repetição de tentativas) para lidar com falhas temporárias
retry_strategy = Retry(
    total=2,
    backoff_factor=0.5,
    status_forcelist=[500, 502, 503, 504], 
)

# Monta o adaptador HTTP com retry strategy em URLs HTTPS
session.mount("https://", HTTPAdapter(max_retries=retry_strategy))

# Monta o adaptador HTTP com retry strategy em URLs HTTP
session.mount("http://", HTTPAdapter(max_retries=retry_strategy))


log_file = "kmeans_assembler.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
dataflow_tag = "nvidiaflare-df"

# variáveis para utilização do FADE
BASE_URL = 'http://dataverse.ic.uff.br/api'
API_TOKEN = {API_TOKEN}
PARENT_DATAVERSE_ALIAS = "flareprov"

total_time_job = 0


# Headers HTTP padrão para todas as requisições à API Dataverse
# - X-Dataverse-key: token de autenticação
# - Content-Type: especifica que será JSON
# - User-Agent: identifica a aplicação fazendo as requisições
HEADERS = {
    "X-Dataverse-key": API_TOKEN,
    "Content-Type": "application/json",
    "User-Agent": "KMeansAssembler/1.0"
}


class KMeansAssembler(Assembler):
    def __init__(self, hash_trial: str):
        super().__init__(data_kind=DataKind.WEIGHTS)
        # Aggregator needs to keep record of historical
        # center and count information for mini-batch kmeans
        self.center = None  
        self.count = None  
        self.n_cluster = 0  
        self.current_round = 0 
        self.hash_trial = hash_trial 
        
        # Atributos para cache e gerenciamento de Dataverse
        self.trial_dv_alias = None  # Alias do Sub-Dataverse criado para este trial
        self.dataset_pids = {}  # Mapeamento {round_number: persistent_id} dos datasets criados
        
        logger.info(f"SUCESSO: KMeansAssembler inicializado com trial: {hash_trial}")
        
        # Configura o Sub-Dataverse para este trial (cria se não existir)
        self._setup_trial_dataverse()

    def _setup_trial_dataverse(self):
        
        try:

            self.trial_dv_alias = f"dv-{self.hash_trial[:12]}".replace("_", "-").lower()

            logger.info(f"Criando Sub-Dataverse: {self.trial_dv_alias}")
            logger.info(f"  Parent Dataverse: {PARENT_DATAVERSE_ALIAS}")

            # Faz uma requisição GET para verificar acesso ao Dataverse pai
            check_url = f"{BASE_URL}/dataverses/{PARENT_DATAVERSE_ALIAS}"
            resp_check = session.get(check_url, headers=HEADERS, timeout=30, verify=False)
            logger.debug(f"  Verificação pai - status {resp_check.status_code}")

            # Se retorna 404, o Dataverse pai não existe (erro crítico)
            if resp_check.status_code == 404:
                logger.error(
                    f"FALHA: Dataverse pai '{PARENT_DATAVERSE_ALIAS}' NÃO ENCONTRADO (404)"
                    f"Crie o Dataverse pai manualmente antes de rodar"
                )
                self.trial_dv_alias = None  
                return  
            # Se retorna outro status diferente de 200, erro de conectividade ou permissão
            elif resp_check.status_code != 200:
                logger.error(
                    f"FALHA: Erro inesperado ao verificar Dataverse pai: {resp_check.status_code} "
                    f"→ {resp_check.text[:300]}"
                )
                self.trial_dv_alias = None 
                return  

            # Verifica se um Sub-Dataverse com este alias já existe (reutiliza se existir)
            sub_check_url = f"{BASE_URL}/dataverses/{self.trial_dv_alias}"
            resp_sub_check = session.get(sub_check_url, headers=HEADERS, timeout=30, verify=False)
            
            # Se retorna 200, o Sub-Dataverse já existe — não precisa criar
            if resp_sub_check.status_code == 200:
                logger.info(f"SUCESSO: Sub-Dataverse já existe: {self.trial_dv_alias}")
                return  

            # Prepara URL para criação de novo Dataverse
            create_url = f"{BASE_URL}/dataverses/{PARENT_DATAVERSE_ALIAS}"
            
            # Payload JSON com metadados do novo Sub-Dataverse
            payload = {
                "name": f"Trial {self.hash_trial[:12]}",  # Nome legível para humanos
                "alias": self.trial_dv_alias,  # Identificador único (URL-safe)
                "description": f"Federated clustering experiment — trial {self.hash_trial}",  # Descrição
                "affiliation": "TCC Lylian Pacheco",  # Instituição responsável
                "dataverseContacts": [  # Contatos para suporte/perguntas
                    {"contactEmail": "lylianpacheco@id.uff.br"}
                ],
                "dataverseType": "RESEARCH_PROJECT"  # Classificação do tipo de Dataverse
            }

            logger.debug(f"  POST {create_url}  payload={payload}")
            
            # Faz POST para criar o Sub-Dataverse no Dataverse pai
            resp = session.post(create_url, headers=HEADERS, json=payload, timeout=60, verify=False)
            logger.debug(f"  Criação: status {resp.status_code}  body={resp.text[:500]}")

            if resp.status_code in [200, 201]:
                logger.info(f"SUCESSO: Sub-Dataverse criado: {self.trial_dv_alias}")
            elif resp.status_code == 400:
                # 400 = alias já existe (pode ser condição de corrida em execução paralela)
                # Prossegue mesmo assim pois o Dataverse já está pronto para uso
                logger.info(f"SUCESSO: Sub-Dataverse já registrado (400): {self.trial_dv_alias}")
            else:
                # Outro erro = falha crítica
                logger.error(
                    f"FALHA: Falha ao criar Sub-Dataverse: {resp.status_code} → {resp.text[:500]}"
                )
                self.trial_dv_alias = None 
                return 

            # Para aceitar datasets, o Sub-Dataverse precisa estar em estado "publicado"
            publish_url = f"{BASE_URL}/dataverses/{self.trial_dv_alias}/actions/:publish"
            resp_pub = session.post(publish_url, headers=HEADERS, timeout=30, verify=False)
            
            if resp_pub.status_code in [200, 201]:
                # Sucesso — Sub-Dataverse está pronto para aceitar datasets
                logger.info(f"SUCESSO: Sub-Dataverse publicado: {self.trial_dv_alias}")
            else:
                # Aviso: não conseguiu publicar, mas datasets ainda podem ser criados em draft
                # Prossegue mesmo assim pois não é erro fatal
                logger.warning(
                    f"⚠ Não foi possível publicar o Sub-Dataverse: "
                    f"{resp_pub.status_code} → {resp_pub.text[:300]}"
                )

        except requests.exceptions.Timeout:
            logger.error("FALHA: Timeout ao configurar Sub-Dataverse")
            self.trial_dv_alias = None  
        except Exception as e:
            logger.error(f"FALHA: Erro crítico ao configurar Sub-Dataverse: {e}", exc_info=True)
            self.trial_dv_alias = None 

    def get_model_params(self, dxo: DXO):
        
        logger.debug(f"[Round {self.current_round}] get_model_params iniciado")

        t6 = Task(
            6 + 4 * (self.current_round), 
            dataflow_tag,  
            "GetModelParams",  
            dependency=Task(
                5 + 4 * (self.current_round),  
                dataflow_tag,
                "ClientTraining"
            ),
        )
        
        t6.begin()
        data = dxo.data

        to_dfanalyzer = [self.hash_trial, data["center"], data["count"]]
        t6_input = DataSet("iGetModelParams", [Element(to_dfanalyzer)])
        t6.add_dataset(t6_input)
        t6_output = DataSet("oGetModelParams", [Element([])])
        t6.add_dataset(t6_output)        
        t6.end()
        logger.debug(f"[Round {self.current_round}] get_model_params finalizado")

        return {"center": data["center"], "count": data["count"]}

    def _ensure_dataset_for_round(self, round_number: int, fl_ctx: FLContext = None) -> Optional[str]:
        
        if round_number in self.dataset_pids:
            logger.debug(f"Dataset para round {round_number} já em cache: {self.dataset_pids[round_number]}")
            return self.dataset_pids[round_number]  

        # Se Sub-Dataverse não foi configurado com sucesso, não é possível criar dataset
        if not self.trial_dv_alias:
            logger.error("FALHA: Sub-Dataverse não disponível — não é possível criar Dataset")
            return None  

        try:
            dataset_name = f"Round {round_number} — Trial {self.hash_trial[:12]}"
            logger.info(f"Criando Dataset: '{dataset_name}' em {self.trial_dv_alias}")

            # URL da API para criar datasets dentro do Sub-Dataverse
            # POST /api/dataverses/{alias}/datasets
            url = f"{BASE_URL}/dataverses/{self.trial_dv_alias}/datasets"

            # Payload JSON com todas as informações do dataset
            # Segue a estrutura esperada pela API Dataverse
            payload = {
                "datasetVersion": {
                    # Licença CC0 (domínio público) para máxima compartilhabilidade
                    "license": {
                        "name": "CC0 1.0",
                        "uri": "http://creativecommons.org/publicdomain/zero/1.0"
                    },
                    # Blocos de metadados Dublin Core (padrão internacional)
                    "metadataBlocks": {
                        # Bloco "Citation" com informações de citação
                        "citation": {
                            "displayName": "Citation Metadata",
                            "fields": [
                                # Campo: Título do dataset
                                {
                                    "typeName": "title",
                                    "multiple": False, 
                                    "typeClass": "primitive", 
                                    "value": dataset_name
                                },
                                # Campo: Autores/Pesquisadores
                                {
                                    "typeName": "author",
                                    "multiple": True,  # Múltiplos autores permitidos
                                    "typeClass": "compound",  
                                    "value": [
                                        {
                                            "authorName": {
                                                "typeName": "authorName",
                                                "multiple": False,
                                                "typeClass": "primitive",
                                                "value": "Federated Clustering — NVFlare"
                                            },
                                            "authorAffiliation": {
                                                "typeName": "authorAffiliation",
                                                "multiple": False,
                                                "typeClass": "primitive",
                                                "value": "TCC Lylian Pacheco"
                                            }
                                        }
                                    ]
                                },
                                # Campo: Contatos para perguntas sobre o dataset
                                {
                                    "typeName": "datasetContact",
                                    "multiple": True,  # Múltiplos contatos permitidos
                                    "typeClass": "compound",
                                    "value": [
                                        {
                                            "datasetContactName": {
                                                "typeName": "datasetContactName",
                                                "multiple": False,
                                                "typeClass": "primitive",
                                                "value": "Lylian Pacheco"
                                            },
                                            "datasetContactEmail": {
                                                "typeName": "datasetContactEmail",
                                                "multiple": False,
                                                "typeClass": "primitive",
                                                "value": "lylianpacheco@id.uff.br"
                                            }
                                        }
                                    ]
                                },
                                # Campo: Descrição do dataset (propósito, conteúdo)
                                {
                                    "typeName": "dsDescription",
                                    "multiple": True,  # Múltiplas descrições permitidas
                                    "typeClass": "compound",
                                    "value": [
                                        {
                                            "dsDescriptionValue": {
                                                "typeName": "dsDescriptionValue",
                                                "multiple": False,
                                                "typeClass": "primitive",
                                                "value": (
                                                    f"Federated KMeans clustering results — "
                                                    f"round {round_number}, trial {self.hash_trial}"
                                                )
                                            }
                                        }
                                    ]
                                },
                                # Campo: Assuntos/Tópicos (vocabulário controlado)
                                # Nota: deve usar valores do vocabulário oficial Dataverse
                                # Lista completa: https://guides.dataverse.org/en/latest/user/appendix.html
                                {
                                    "typeName": "subject",
                                    "multiple": True,  # Múltiplos assuntos permitidos
                                    "typeClass": "controlledVocabulary",  # Valores de vocabulário pré-definido
                                    "value": ["Computer and Information Science"]  # Valor da lista oficial
                                }
                            ]
                        }
                    }
                }
            }

            logger.debug(f"  POST {url}")
            resp = session.post(url, headers=HEADERS, json=payload, timeout=60, verify=False)
            logger.debug(f"  Criação dataset → status {resp.status_code}  body={resp.text[:600]}")

            
            if resp.status_code in [200, 201]:
                resp_data = resp.json()
                
                # Obtém o persistent_id (DOI/Handle) do dataset criado
                # Este ID é usado para referência permanente do dataset
                pid = resp_data.get("data", {}).get("persistentId")
                
                # Valida se o ID foi retornado
                if not pid:
                    logger.error(f"FALHA: Dataset criado mas sem persistentId na resposta: {resp.text[:300]}")
                    return None  
                
                # Armazena no cache para evitar recriação futura
                self.dataset_pids[round_number] = pid
                logger.info(f"SUCESSO: Dataset criado para round {round_number}: {pid}")
                
                if fl_ctx:
                    self.log_info(fl_ctx, f"SUCESSO: Dataset Round {round_number}: {pid}")
                
                return pid  
            else:
                logger.error(f"FALHA: Erro ao criar Dataset (status {resp.status_code}): {resp.text[:500]}")
                return None 

        except requests.exceptions.Timeout:
            logger.error(f"FALHA: Timeout ao criar Dataset para round {round_number}")
            return None  
        except Exception as e:
            logger.error(f"FALHA: Erro ao criar Dataset para round {round_number}: {e}", exc_info=True)
            return None 
            
    def _upload_to_dataverse(self, file_path: str, round_number: int, client_id: Optional[str] = None, fl_ctx: FLContext = None) -> bool:
        """
        Estrutura de organização no Dataverse:
        - round_{round_number}/
          ├── client_X/ (para dados de clientes)
          └── server/ (para dados agregados do servidor)
        """
        try:
            # Verifica se o arquivo existe antes de tentar fazer upload
            if not os.path.exists(file_path):
                logger.error(f"FALHA: Arquivo não encontrado: {file_path}")
                return False 
            
            # Garante que existe um Dataset Dataverse para este round
            pid = self._ensure_dataset_for_round(round_number, fl_ctx)
            if not pid:
                logger.error(f"FALHA: Não conseguiu obter PID para o Dataset do round {round_number}")
                return False 
            
            # Organiza arquivos em diretórios virtuais por round e cliente
            if client_id is not None:
                # Arquivo de cliente: round_X/client_Y/
                dir_label = f"round_{round_number}/client_{client_id}"
                upload_info = f"cliente {client_id}"
            else:
                # Arquivo de servidor/agregado: round_X/server/
                dir_label = f"round_{round_number}/server"
                upload_info = "servidor (agregado)"
            
            logger.debug(f"Fazendo upload para {upload_info}: {os.path.basename(file_path)}")
            
            # URL da API para adicionar arquivo a um dataset existente
            # Formato: /datasets/:persistentId/add?persistentId={pid}
            upload_url = f"{BASE_URL}/datasets/:persistentId/add?persistentId={pid}"
            
            with open(file_path, 'rb') as f:
                # Prepara componentes para requisição multipart/form-data
                # (necessário para envio de arquivos em HTTP)
                files = {
                    'file': (
                        os.path.basename(file_path),  # Nome do arquivo
                        f,  # Conteúdo (file object)
                        'application/octet-stream'  # MIME type (dados binários genéricos)
                    )
                }
                
                # Dados adicionais (metadados do upload)
                data = {
                    'directoryLabel': dir_label,  # Caminho virtual no Dataverse
                    'forceReplace': 'true'  # Sobrescreve se arquivo com mesmo nome existir
                }
                
                resp = session.post(
                    upload_url,
                    headers={"X-Dataverse-key": API_TOKEN}, 
                    files=files,  
                    data=data,  
                    timeout=60,
                    verify=False 
                )
            
            if resp.status_code in [200, 201]:
                logger.info(f"SUCESSO: Upload OK: {os.path.basename(file_path)} → {dir_label}")
                if fl_ctx:
                    self.log_info(fl_ctx, f"SUCESSO: Upload: {os.path.basename(file_path)}")
                return True  
            else:
                logger.error(f"FALHA: Upload falhou (status {resp.status_code})")
                logger.debug(f"Resposta: {resp.text[:500]}")
                return False
                
        except requests.exceptions.Timeout:
            # Timeout: upload levou muito tempo (arquivo muito grande ou rede lenta)
            logger.error(f"FALHA: Timeout ao fazer upload de {file_path}")
            return False 
        except Exception as e:
            logger.error(f"FALHA: Erro ao fazer upload: {e}", exc_info=True)
            return False
        
    def assemble(self, data: Dict[str, dict], fl_ctx: FLContext) -> DXO:
        global total_time_job

        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        self.current_round = current_round
        
        logger.info(f"\n{'='*70}")
        logger.info(f"[ROUND {current_round}] Iniciando assemble")
        logger.info(f"{'='*70}")
        
        n_feature = 0
        t8 = Task(
            7 + 4 * (current_round), 
            dataflow_tag,  
            "Assemble",
            dependency=Task(6 + 4 * (current_round), dataflow_tag, "GetModelParams"),
        )
        t8.begin()
        start = perf_counter()
        kmeans_time = 0
        kmeans_time_job = 0
        timestamp_beginning = datetime.datetime.now() 

        if current_round == 0:
            logger.info("[ROUND 0] Inicializando clustering - primeira rodada")
            # First round, collect the information regarding n_feature and n_cluster
            # Initialize the aggregated center and count to all zero rodada, cada cliente envia seus centros locais
            client_0 = list(self.collection.keys())[0]  
            self.n_cluster = self.collection[client_0]["center"].shape[0]  # Número de clusters/linhas
            n_feature = self.collection[client_0]["center"].shape[1]  # Número de features/colunas
            logger.info(f"SUCESSO: Configuração inicial: clusters={self.n_cluster}, features={n_feature}")
            logger.debug(f"Clientes na coleção: {list(self.collection.keys())}")
            
            self.center = np.zeros([self.n_cluster, n_feature])
            self.count = np.zeros([self.n_cluster])
            # perform one round of KMeans over the submitted centers
            # to be used as the original center points
            # no count for this round           
            center_collect = []  
            for _, record in self.collection.items():
                center_collect.append(record["center"]) 
            centers = np.concatenate(center_collect)            
            kmeans_center_initial = KMeans(n_clusters=self.n_cluster)
            kmeans_center_initial.fit(centers)
            self.center = kmeans_center_initial.cluster_centers_
            logger.info(f"SUCESSO: Centros iniciais calculados via KMeans")
        else:
            # Mini-batch k-Means step to assemble the received centers
            logger.info(f"[ROUND {current_round}] Executando agregação mini-batch KMeans")
            start_kmeans = perf_counter()
            for client_name, record in self.collection.items():
                try:
                    # Criar nome único para arquivo do cliente
                    # Formato: kmeans_client_{nome}_r{round}.npz
                    client_filename = f"kmeans_client_{client_name}_r{current_round}.npz"
                    
                    center_arr = np.asarray(record["center"], dtype=np.float32)
                    count_arr = np.asarray(record["count"], dtype=np.float32)
                    
                    # Salvar arquivo localmente em formato NPZ
                    np.savez_compressed(client_filename, center=center_arr, count=count_arr)
                    logger.debug(f"SUCESSO: Arquivo salvo: {client_filename}")
                    
                    # Fazer upload para Dataverse
                    self._upload_to_dataverse(
                        client_filename,
                        round_number=current_round,
                        client_id=str(client_name),
                        fl_ctx=fl_ctx
                    )
                except Exception as e:
                    logger.error(f"Falha ao salvar/upload cliente {client_name}: {e}", exc_info=True)
                    try:
                        self.log_warning(fl_ctx, f"Falha cliente {client_name}: {e}")
                    except Exception:
                        pass  
            
           
            for center_idx in range(self.n_cluster):
                centers_global_rescale = (
                    self.center[center_idx] * self.count[center_idx]
                )
                # Aggregate center, add new center to previous estimate, weighted by counts
                for _, record in self.collection.items():
                    centers_global_rescale += (
                        record["center"][center_idx] * record["count"][center_idx]
                    )
                    self.count[center_idx] += record["count"][center_idx]
                # Rescale to compute mean of all points (old and new combined)
                alpha = 1 / self.count[center_idx]
                centers_global_rescale *= alpha  
                # Update the global center
                self.center[center_idx] = centers_global_rescale
            kmeans_time = perf_counter() - start_kmeans

            logger.info(f"SUCESSO: Agregação concluída em {kmeans_time:.4f}s")
            kmeans_time_job = kmeans_time_job + kmeans_time
            
            try:
                # Formato: kmeans_r{round}.npz (sem ID de cliente = agregado)
                server_filename = f"kmeans_r{current_round}.npz"
                
                center_arr = np.asarray(self.center, dtype=np.float32)
                count_arr = np.asarray(self.count, dtype=np.float32)
                
                np.savez_compressed(server_filename, center=center_arr, count=count_arr)
                logger.debug(f"SUCESSO: Arquivo agregado salvo: {server_filename}")
                
                self._upload_to_dataverse(
                    server_filename,
                    round_number=current_round,
                    client_id=None, 
                    fl_ctx=fl_ctx
                )
            except Exception as e:
                logger.error(f"Erro ao fazer upload de dados agregados: {e}", exc_info=True)
                try:
                    self.log_error(fl_ctx, f"Erro upload agregado: {e}")
                except Exception:
                    pass  

        # Define what you want to save
        model_state = {
            'center': self.center,  # Array de centros agregados
            'count': self.count,  # Array de contadores por cluster
            'collection': self.collection,  # Dados de todos os clientes neste round
            'hash_trial': self.hash_trial,  # Identificador do trial
            'n_cluster': self.n_cluster,  # Número de clusters K
        }

        # Save the model to disk
        with open('kmeans_model.pkl', 'wb') as f:
            pickle.dump(model_state, f)
        logger.debug(f"SUCESSO: Modelo salvo em kmeans_model.pkl")

        assembling_time = perf_counter() - start
        logger.info(f"[ROUND {current_round}]******tempo do assemble_time: {assembling_time:.4f}s************")
        logger.info(f"[ROUND {current_round}]******tempo do total_time_job: {total_time_job:.4f}s************")

        total_time_job += assembling_time
        timestamp = datetime.datetime.now()
        logger.info(f"[ROUND {current_round}]******tempo total de assemble: {total_time_job:.4f}s************")

        to_dfanalyzer = [self.hash_trial, current_round, n_feature, self.n_cluster, timestamp_beginning]
        t8_input = DataSet("iAssemble", [Element(to_dfanalyzer)])
        t8.add_dataset(t8_input)
        t8_output = DataSet(
            "oAssemble",
            [Element([self.hash_trial, self.current_round, self.center, self.count, assembling_time, kmeans_time, timestamp])],
        )
        t8.add_dataset(t8_output)
        t8.end()
        params = {"center": self.center} 
        dxo = DXO(data_kind=self.expected_data_kind, data=params)

        self.current_round = current_round + 1
        
        logger.info(f"[ROUND {current_round}] Assemble finalizado com sucesso")
        logger.info(f"{'='*70}\n")

        logger.info(f"tempo total de assemble: {assembling_time:.4f}s")
        logger.info(f"{'='*70}\n")
        
        return dxo