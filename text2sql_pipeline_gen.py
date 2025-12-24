# From https://github.com/OpenDCAI/DataFlow
# Based on: 
# https://github.com/OpenDCAI/DataFlow/blob/main/dataflow/statics/pipelines/api_pipelines/text2sql_pipeline_gen.py
# Apache License 2.0 - https://github.com/OpenDCAI/DataFlow?tab=Apache-2.0-1-ov-file#readme
# 
#  
# @article{liang2025dataflow,
#  title={DataFlow: An LLM-Driven Framework for Unified Data Preparation and Workflow Automation in the Era of Data-Centric AI},
#  author={Liang, Hao and Ma, Xiaochen and Liu, Zhou and Wong, Zhen Hao and Zhao, Zhengyang and Meng, Zimo and He, Runming and Shen, Chengyu and Cai, Qifeng and Han, Zhaoyang and others},
#  journal={arXiv preprint arXiv:2512.16676},
#  year={2025}
# }

import os
from dataflow import get_logger
import zipfile
from huggingface_hub import snapshot_download

from dataflow.operators.text2sql import (
    SQLGenerator,
    SQLByColumnGenerator,
    Text2SQLQuestionGenerator,
    Text2SQLPromptGenerator,
    Text2SQLCoTGenerator
)
from dataflow.operators.text2sql import (
    SQLExecutionFilter
)
from dataflow.operators.text2sql import (
    SQLComponentClassifier,
    SQLExecutionClassifier
)
from dataflow.prompts.text2sql import (
    Text2SQLCotGeneratorPrompt,
    SelectSQLGeneratorPrompt,
    Text2SQLQuestionGeneratorPrompt,
    Text2SQLPromptGeneratorPrompt
)
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request
from dataflow.utils.text2sql.database_manager import DatabaseManager

#NEW for OracleDB
from utils.database_connectors.oracledb_connector import OracleConnector
import json

# Monkey patch to add Oracle support to DatabaseManager
DatabaseManager.CONNECTORS['oracle'] = OracleConnector
DB_CFG = {}
LLM_CFG = {}
GEN_CONF = {}

class Text2SQLGeneration_APIPipeline():
    def __init__(self, db_root_path=""):
        self.logger = get_logger()
        self.db_root_path = db_root_path
        
        self.logger.info(f"Using manually specified database path: {self.db_root_path}")

        self.storage = FileStorage(
            first_entry_file_name="",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )

        self.llm_serving = APILLMServing_request(
            api_url=LLM_CFG['chat_url'],
            model_name=LLM_CFG['chat_model'],
            max_workers=100
        )

        # It is recommended to use better LLMs for the generation of Chain-of-Thought (CoT) reasoning process.
        cot_generation_api_llm_serving = APILLMServing_request(
            api_url=LLM_CFG['chat_url'],
            model_name=LLM_CFG['chat_model'], # You can change to a more powerful model for CoT generation
            max_workers=100
        )

        embedding_serving = APILLMServing_request(
            api_url=LLM_CFG['embedding_url'],
            model_name=LLM_CFG['embedding_model'],
            max_workers=100
        )        
      
        database_manager = DatabaseManager(
            db_type='oracle',
            config=DB_CFG
        )
        
        
        self.sql_generator_step1 = SQLGenerator(
            llm_serving=self.llm_serving,
            database_manager=database_manager,
            generate_num=GEN_CONF['nq'],
            prompt_template=SelectSQLGeneratorPrompt()
        )

        self.sql_execution_filter_step2 = SQLExecutionFilter(
            database_manager=database_manager
        )

        self.text2sql_question_generator_step3 = Text2SQLQuestionGenerator(
            llm_serving=self.llm_serving,
            embedding_serving=embedding_serving,
            database_manager=database_manager,
            question_candidates_num=5,
            prompt_template=Text2SQLQuestionGeneratorPrompt()
        )

        self.text2sql_prompt_generator_step4 = Text2SQLPromptGenerator(
            database_manager=database_manager,
            prompt_template=Text2SQLPromptGeneratorPrompt()
        )

        self.sql_cot_generator_step5 = Text2SQLCoTGenerator(
            llm_serving=cot_generation_api_llm_serving,
            database_manager=database_manager,
            prompt_template=Text2SQLCotGeneratorPrompt()
        )

        self.sql_component_classifier_step6 = SQLComponentClassifier(
            difficulty_thresholds=[2, 4, 6],
            difficulty_labels=['easy', 'medium', 'hard', 'extra']
        )

        self.sql_execution_classifier_step7 = SQLExecutionClassifier(
            llm_serving=self.llm_serving,
            database_manager=database_manager,
            num_generations=10,
            difficulty_thresholds=[2, 5, 9],
           difficulty_labels=['extra', 'hard', 'medium', 'easy']
        )
        
    def forward(self):

        sql_key = "SQL"
        db_id_key = "db_id"
        question_key = "question"
        evidence_key = "evidence"

        self.sql_generator_step1.run(
            storage=self.storage.step(),
            output_sql_key=sql_key,
            output_db_id_key=db_id_key
        )

        self.sql_execution_filter_step2.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            input_db_id_key=db_id_key
        )

        self.text2sql_question_generator_step3.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            input_db_id_key=db_id_key,
            output_question_key=question_key,
            output_evidence_key=evidence_key
        )

        # ONLY Question & SQL generation 
        if (False) :         
            self.text2sql_prompt_generator_step4.run(
                storage=self.storage.step(),
                input_question_key=question_key,
                input_db_id_key=db_id_key,
                input_evidence_key=evidence_key,
                output_prompt_key="prompt"
            )

            self.sql_cot_generator_step5.run(
                storage=self.storage.step(),
                input_sql_key=sql_key,
                input_question_key=question_key,
                input_db_id_key=db_id_key,
                input_evidence_key=evidence_key,
                output_cot_key="cot_reasoning"
            )

            self.sql_component_classifier_step6.run(
                storage=self.storage.step(),
                input_sql_key=sql_key,
                output_difficulty_key="sql_component_difficulty"
            )

            self.sql_execution_classifier_step7.run(
                storage=self.storage.step(),
                input_sql_key=sql_key,
                input_db_id_key=db_id_key,
                input_prompt_key="prompt",
                output_difficulty_key="sql_execution_difficulty"
            )
        

if __name__ == "__main__":
   
    db_root_path = "oracledb"
    DB_CFG = {
            'user': 'SH',
            'password': 'Welcome_12345', 
            'host': 'localhost',
            'port': 1521,
            'service_name': 'FREEPDB1'
        }

    LLM_CFG =  {
            'chat_model': 'gpt-4o-mini',
            'chat_url' : 'https://api.openai.com/v1/chat/completions',
            'embedding_model' : 'text-embedding-3-small',
            'embedding_url': 'https://api.openai.com/v1/embeddings'
        }

    GEN_CONF = {
            'nq':10
        }
    
    model = Text2SQLGeneration_APIPipeline(db_root_path=db_root_path)
    model.forward()

    # Input and output file paths
    input_file = 'cache/dataflow_cache_step_step3.jsonl'
    output_file = 'cache/dataflow_cache_step_step3_updated.jsonl'  # Or overwrite by setting to input_file

    # Add to the file extra gold_field list: by default all. 
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            line = line.strip()
            if line:  
                data = json.loads(line)
                data['gold_fields'] = [] 
                outfile.write(json.dumps(data) + '\n')

