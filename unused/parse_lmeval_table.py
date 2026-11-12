"""把 lm-eval 的结果表汇总成一行一个模型的 CSV（表头顺序固定，可直接贴进表格）。

用法：把各模型的输出粘进下面的 model_outputs（键是模型名），然后不带参数运行
  python unused/parse_lmeval_table.py
值可以只是结果表，也可以是整段 eval 日志：其中 eval.py 打印的结果表（每个阶段一张）会被挑出来，行首的时间戳、
rank 等前缀会去掉，其余行忽略。同一个模型的几份日志（例如 32K、128K、256K 各一份）接着贴进同一个字符串，合成一行。
"""
import pandas as pd
import numpy as np
import io
import re

# 把你的各个模型的原始输出放在这里。这里以 Model_A 为例（填入你提供的原数据）
model_outputs = {
    "base-prolong": """
|                         Tasks                         |Version|Filter|n-shot|        Metric         |   | Value |   |Stderr|
|-------------------------------------------------------|-------|------|-----:|-----------------------|---|------:|---|------|
|arc_challenge                                          |      1|none  |     0|acc                    |↑  | 0.4130|±  |0.0144|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4497|±  |0.0145|
|arc_easy                                               |      1|none  |     0|acc                    |↑  | 0.7374|±  |0.0090|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6852|±  |0.0095|
|boolq                                                  |      2|none  |     0|acc                    |↑  | 0.7914|±  |0.0071|
|ceval-valid                                            |      2|none  |      |acc                    |↑  | 0.6493|±  |0.0126|
|                                                       |       |none  |      |acc_norm               |↑  | 0.6493|±  |0.0126|
| - ceval-valid_accountant                              |      2|none  |     0|acc                    |↑  | 0.5510|±  |0.0718|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5510|±  |0.0718|
| - ceval-valid_advanced_mathematics                    |      2|none  |     0|acc                    |↑  | 0.3684|±  |0.1137|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3684|±  |0.1137|
| - ceval-valid_art_studies                             |      2|none  |     0|acc                    |↑  | 0.5758|±  |0.0874|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5758|±  |0.0874|
| - ceval-valid_basic_medicine                          |      2|none  |     0|acc                    |↑  | 0.6842|±  |0.1096|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6842|±  |0.1096|
| - ceval-valid_business_administration                 |      2|none  |     0|acc                    |↑  | 0.6364|±  |0.0850|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6364|±  |0.0850|
| - ceval-valid_chinese_language_and_literature         |      2|none  |     0|acc                    |↑  | 0.5652|±  |0.1057|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5652|±  |0.1057|
| - ceval-valid_civil_servant                           |      2|none  |     0|acc                    |↑  | 0.5319|±  |0.0736|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5319|±  |0.0736|
| - ceval-valid_clinical_medicine                       |      2|none  |     0|acc                    |↑  | 0.5455|±  |0.1087|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5455|±  |0.1087|
| - ceval-valid_college_chemistry                       |      2|none  |     0|acc                    |↑  | 0.5833|±  |0.1028|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5833|±  |0.1028|
| - ceval-valid_college_economics                       |      2|none  |     0|acc                    |↑  | 0.5818|±  |0.0671|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5818|±  |0.0671|
| - ceval-valid_college_physics                         |      2|none  |     0|acc                    |↑  | 0.5789|±  |0.1164|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5789|±  |0.1164|
| - ceval-valid_college_programming                     |      2|none  |     0|acc                    |↑  | 0.7027|±  |0.0762|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7027|±  |0.0762|
| - ceval-valid_computer_architecture                   |      2|none  |     0|acc                    |↑  | 0.7619|±  |0.0952|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7619|±  |0.0952|
| - ceval-valid_computer_network                        |      2|none  |     0|acc                    |↑  | 0.5263|±  |0.1177|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5263|±  |0.1177|
| - ceval-valid_discrete_mathematics                    |      2|none  |     0|acc                    |↑  | 0.3750|±  |0.1250|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3750|±  |0.1250|
| - ceval-valid_education_science                       |      2|none  |     0|acc                    |↑  | 0.7931|±  |0.0766|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7931|±  |0.0766|
| - ceval-valid_electrical_engineer                     |      2|none  |     0|acc                    |↑  | 0.4324|±  |0.0826|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4324|±  |0.0826|
| - ceval-valid_environmental_impact_assessment_engineer|      2|none  |     0|acc                    |↑  | 0.7097|±  |0.0829|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7097|±  |0.0829|
| - ceval-valid_fire_engineer                           |      2|none  |     0|acc                    |↑  | 0.4839|±  |0.0912|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4839|±  |0.0912|
| - ceval-valid_high_school_biology                     |      2|none  |     0|acc                    |↑  | 0.7368|±  |0.1038|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7368|±  |0.1038|
| - ceval-valid_high_school_chemistry                   |      2|none  |     0|acc                    |↑  | 0.4737|±  |0.1177|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4737|±  |0.1177|
| - ceval-valid_high_school_chinese                     |      2|none  |     0|acc                    |↑  | 0.5263|±  |0.1177|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5263|±  |0.1177|
| - ceval-valid_high_school_geography                   |      2|none  |     0|acc                    |↑  | 0.6842|±  |0.1096|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6842|±  |0.1096|
| - ceval-valid_high_school_history                     |      2|none  |     0|acc                    |↑  | 0.7500|±  |0.0993|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7500|±  |0.0993|
| - ceval-valid_high_school_mathematics                 |      2|none  |     0|acc                    |↑  | 0.4444|±  |0.1205|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4444|±  |0.1205|
| - ceval-valid_high_school_physics                     |      2|none  |     0|acc                    |↑  | 0.6842|±  |0.1096|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6842|±  |0.1096|
| - ceval-valid_high_school_politics                    |      2|none  |     0|acc                    |↑  | 0.8947|±  |0.0723|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8947|±  |0.0723|
| - ceval-valid_ideological_and_moral_cultivation       |      2|none  |     0|acc                    |↑  | 0.9474|±  |0.0526|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.9474|±  |0.0526|
| - ceval-valid_law                                     |      2|none  |     0|acc                    |↑  | 0.5417|±  |0.1039|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5417|±  |0.1039|
| - ceval-valid_legal_professional                      |      2|none  |     0|acc                    |↑  | 0.4348|±  |0.1057|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4348|±  |0.1057|
| - ceval-valid_logic                                   |      2|none  |     0|acc                    |↑  | 0.5909|±  |0.1073|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5909|±  |0.1073|
| - ceval-valid_mao_zedong_thought                      |      2|none  |     0|acc                    |↑  | 0.8333|±  |0.0777|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8333|±  |0.0777|
| - ceval-valid_marxism                                 |      2|none  |     0|acc                    |↑  | 0.9474|±  |0.0526|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.9474|±  |0.0526|
| - ceval-valid_metrology_engineer                      |      2|none  |     0|acc                    |↑  | 0.7917|±  |0.0847|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7917|±  |0.0847|
| - ceval-valid_middle_school_biology                   |      2|none  |     0|acc                    |↑  | 0.9524|±  |0.0476|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.9524|±  |0.0476|
| - ceval-valid_middle_school_chemistry                 |      2|none  |     0|acc                    |↑  | 0.9000|±  |0.0688|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.9000|±  |0.0688|
| - ceval-valid_middle_school_geography                 |      2|none  |     0|acc                    |↑  | 0.6667|±  |0.1421|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6667|±  |0.1421|
| - ceval-valid_middle_school_history                   |      2|none  |     0|acc                    |↑  | 0.9545|±  |0.0455|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.9545|±  |0.0455|
| - ceval-valid_middle_school_mathematics               |      2|none  |     0|acc                    |↑  | 0.5263|±  |0.1177|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5263|±  |0.1177|
| - ceval-valid_middle_school_physics                   |      2|none  |     0|acc                    |↑  | 0.7368|±  |0.1038|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7368|±  |0.1038|
| - ceval-valid_middle_school_politics                  |      2|none  |     0|acc                    |↑  | 0.8571|±  |0.0782|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8571|±  |0.0782|
| - ceval-valid_modern_chinese_history                  |      2|none  |     0|acc                    |↑  | 0.8261|±  |0.0808|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8261|±  |0.0808|
| - ceval-valid_operating_system                        |      2|none  |     0|acc                    |↑  | 0.6316|±  |0.1137|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6316|±  |0.1137|
| - ceval-valid_physician                               |      2|none  |     0|acc                    |↑  | 0.6531|±  |0.0687|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6531|±  |0.0687|
| - ceval-valid_plant_protection                        |      2|none  |     0|acc                    |↑  | 0.8182|±  |0.0842|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8182|±  |0.0842|
| - ceval-valid_probability_and_statistics              |      2|none  |     0|acc                    |↑  | 0.3889|±  |0.1182|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3889|±  |0.1182|
| - ceval-valid_professional_tour_guide                 |      2|none  |     0|acc                    |↑  | 0.5517|±  |0.0940|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5517|±  |0.0940|
| - ceval-valid_sports_science                          |      2|none  |     0|acc                    |↑  | 0.6842|±  |0.1096|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6842|±  |0.1096|
| - ceval-valid_tax_accountant                          |      2|none  |     0|acc                    |↑  | 0.5714|±  |0.0714|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5714|±  |0.0714|
| - ceval-valid_teacher_qualification                   |      2|none  |     0|acc                    |↑  | 0.7727|±  |0.0639|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7727|±  |0.0639|
| - ceval-valid_urban_and_rural_planner                 |      2|none  |     0|acc                    |↑  | 0.6739|±  |0.0699|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6739|±  |0.0699|
| - ceval-valid_veterinary_medicine                     |      2|none  |     0|acc                    |↑  | 0.7391|±  |0.0936|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7391|±  |0.0936|
|hellaswag                                              |      1|none  |     0|acc                    |↑  | 0.4931|±  |0.0050|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6639|±  |0.0047|
|ifeval                                                 |      4|none  |     0|inst_level_loose_acc   |↑  | 0.3561|±  |   N/A|
|                                                       |       |none  |     0|inst_level_strict_acc  |↑  | 0.3333|±  |   N/A|
|                                                       |       |none  |     0|prompt_level_loose_acc |↑  | 0.2384|±  |0.0183|
|                                                       |       |none  |     0|prompt_level_strict_acc|↑  | 0.2144|±  |0.0177|
|long_ppl_books_32k                                     |    N/A|      |      |                       |   |       |   |      |
| - long_ppl_books_32k_16k                              |      1|none  |     0|bits_per_byte          |↓  | 0.8701|±  |   N/A|
|                                                       |       |none  |     0|ppl                    |↓  |11.0588|±  |   N/A|
| - long_ppl_books_32k_32k                              |      1|none  |     0|bits_per_byte          |↓  | 0.8810|±  |   N/A|
|                                                       |       |none  |     0|ppl                    |↓  |11.3850|±  |   N/A|
| - long_ppl_books_32k_8k                               |      1|none  |     0|bits_per_byte          |↓  | 0.8544|±  |   N/A|
|                                                       |       |none  |     0|ppl                    |↓  |10.5433|±  |   N/A|
|longbench_2wikimqa                                     |      5|none  |     0|qa_f1_score            |↑  | 0.2046|±  |0.0257|
|                                                       |       |none  |     0|score                  |↑  | 0.2046|±  |0.0257|
|longbench_2wikimqa_e                                   |      5|none  |     0|qa_f1_score            |↑  | 0.2554|±  |0.0225|
|                                                       |       |none  |     0|score                  |↑  | 0.2554|±  |0.0225|
|longbench_dureader                                     |      5|none  |     0|rouge_zh_score         |↑  | 0.3198|±  |0.0180|
|                                                       |       |none  |     0|score                  |↑  | 0.3198|±  |0.0180|
|longbench_gov_report                                   |      5|none  |     0|rouge_score            |↑  | 0.3059|±  |0.0061|
|                                                       |       |none  |     0|score                  |↑  | 0.3059|±  |0.0061|
|longbench_gov_report_e                                 |      5|none  |     0|rouge_score            |↑  | 0.3044|±  |0.0044|
|                                                       |       |none  |     0|score                  |↑  | 0.3044|±  |0.0044|
|longbench_hotpotqa                                     |      5|none  |     0|qa_f1_score            |↑  | 0.3626|±  |0.0294|
|                                                       |       |none  |     0|score                  |↑  | 0.3626|±  |0.0294|
|longbench_hotpotqa_e                                   |      5|none  |     0|qa_f1_score            |↑  | 0.3785|±  |0.0243|
|                                                       |       |none  |     0|score                  |↑  | 0.3785|±  |0.0243|
|longbench_lcc                                          |      5|none  |     0|code_sim_score         |↑  | 0.0525|±  |0.0041|
|                                                       |       |none  |     0|score                  |↑  | 0.0525|±  |0.0041|
|longbench_lcc_e                                        |      5|none  |     0|code_sim_score         |↑  | 0.0662|±  |0.0081|
|                                                       |       |none  |     0|score                  |↑  | 0.0662|±  |0.0081|
|longbench_lsht                                         |      5|none  |     0|classification_score   |↑  | 0.3750|±  |0.0330|
|                                                       |       |none  |     0|score                  |↑  | 0.3750|±  |0.0330|
|longbench_multi_news                                   |      5|none  |     0|rouge_score            |↑  | 0.2541|±  |0.0052|
|                                                       |       |none  |     0|score                  |↑  | 0.2541|±  |0.0052|
|longbench_multi_news_e                                 |      5|none  |     0|rouge_score            |↑  | 0.2293|±  |0.0036|
|                                                       |       |none  |     0|score                  |↑  | 0.2293|±  |0.0036|
|longbench_multifieldqa_en                              |      5|none  |     0|qa_f1_score            |↑  | 0.4899|±  |0.0269|
|                                                       |       |none  |     0|score                  |↑  | 0.4899|±  |0.0269|
|longbench_multifieldqa_en_e                            |      5|none  |     0|qa_f1_score            |↑  | 0.4899|±  |0.0269|
|                                                       |       |none  |     0|score                  |↑  | 0.4899|±  |0.0269|
|longbench_multifieldqa_zh                              |      5|none  |     0|qa_f1_zh_score         |↑  | 0.5047|±  |0.0226|
|                                                       |       |none  |     0|score                  |↑  | 0.5047|±  |0.0226|
|longbench_musique                                      |      5|none  |     0|qa_f1_score            |↑  | 0.1620|±  |0.0211|
|                                                       |       |none  |     0|score                  |↑  | 0.1620|±  |0.0211|
|longbench_narrativeqa                                  |      5|none  |     0|qa_f1_score            |↑  | 0.1914|±  |0.0163|
|                                                       |       |none  |     0|score                  |↑  | 0.1914|±  |0.0163|
|longbench_passage_count                                |      5|none  |     0|count_score            |↑  | 0.0250|±  |0.0111|
|                                                       |       |none  |     0|score                  |↑  | 0.0250|±  |0.0111|
|longbench_passage_count_e                              |      5|none  |     0|count_score            |↑  | 0.0333|±  |0.0104|
|                                                       |       |none  |     0|score                  |↑  | 0.0333|±  |0.0104|
|longbench_passage_retrieval_en                         |      5|none  |     0|retrieval_score        |↑  | 0.5150|±  |0.0354|
|                                                       |       |none  |     0|score                  |↑  | 0.5150|±  |0.0354|
|longbench_passage_retrieval_en_e                       |      5|none  |     0|retrieval_score        |↑  | 0.7267|±  |0.0258|
|                                                       |       |none  |     0|score                  |↑  | 0.7267|±  |0.0258|
|longbench_qasper                                       |      5|none  |     0|qa_f1_score            |↑  | 0.3736|±  |0.0244|
|                                                       |       |none  |     0|score                  |↑  | 0.3736|±  |0.0244|
|longbench_qasper_e                                     |      5|none  |     0|qa_f1_score            |↑  | 0.3538|±  |0.0232|
|                                                       |       |none  |     0|score                  |↑  | 0.3538|±  |0.0232|
|longbench_qmsum                                        |      5|none  |     0|rouge_score            |↑  | 0.2378|±  |0.0051|
|                                                       |       |none  |     0|score                  |↑  | 0.2378|±  |0.0051|
|longbench_repobench-p                                  |      5|none  |     0|code_sim_score         |↑  | 0.0690|±  |0.0053|
|                                                       |       |none  |     0|score                  |↑  | 0.0690|±  |0.0053|
|longbench_repobench-p_e                                |      5|none  |     0|code_sim_score         |↑  | 0.0615|±  |0.0057|
|                                                       |       |none  |     0|score                  |↑  | 0.0615|±  |0.0057|
|longbench_samsum                                       |      5|none  |     0|rouge_score            |↑  | 0.3031|±  |0.0078|
|                                                       |       |none  |     0|score                  |↑  | 0.3031|±  |0.0078|
|longbench_samsum_e                                     |      5|none  |     0|rouge_score            |↑  | 0.3058|±  |0.0064|
|                                                       |       |none  |     0|score                  |↑  | 0.3058|±  |0.0064|
|longbench_trec                                         |      5|none  |     0|classification_score   |↑  | 0.4527|±  |0.0264|
|                                                       |       |none  |     0|score                  |↑  | 0.4527|±  |0.0264|
|longbench_trec_e                                       |      5|none  |     0|classification_score   |↑  | 0.4100|±  |0.0204|
|                                                       |       |none  |     0|score                  |↑  | 0.4100|±  |0.0204|
|longbench_triviaqa                                     |      5|none  |     0|qa_f1_score            |↑  | 0.2143|±  |0.0093|
|                                                       |       |none  |     0|score                  |↑  | 0.2143|±  |0.0093|
|longbench_triviaqa_e                                   |      5|none  |     0|qa_f1_score            |↑  | 0.2178|±  |0.0086|
|                                                       |       |none  |     0|score                  |↑  | 0.2178|±  |0.0086|
|longbench_vcsum                                        |      5|none  |     0|rouge_zh_score         |↑  | 0.1574|±  |0.0029|
|                                                       |       |none  |     0|score                  |↑  | 0.1574|±  |0.0029|
|mmlu                                                   |      2|none  |      |acc                    |↑  | 0.6039|±  |0.0039|
| - humanities                                          |      2|none  |      |acc                    |↑  | 0.5120|±  |0.0068|
|  - formal_logic                                       |      1|none  |     0|acc                    |↑  | 0.5079|±  |0.0447|
|  - high_school_european_history                       |      1|none  |     0|acc                    |↑  | 0.7333|±  |0.0345|
|  - high_school_us_history                             |      1|none  |     0|acc                    |↑  | 0.7108|±  |0.0318|
|  - high_school_world_history                          |      1|none  |     0|acc                    |↑  | 0.7764|±  |0.0271|
|  - international_law                                  |      1|none  |     0|acc                    |↑  | 0.7190|±  |0.0410|
|  - jurisprudence                                      |      1|none  |     0|acc                    |↑  | 0.7222|±  |0.0433|
|  - logical_fallacies                                  |      1|none  |     0|acc                    |↑  | 0.7607|±  |0.0335|
|  - moral_disputes                                     |      1|none  |     0|acc                    |↑  | 0.6590|±  |0.0255|
|  - moral_scenarios                                    |      1|none  |     0|acc                    |↑  | 0.2402|±  |0.0143|
|  - philosophy                                         |      1|none  |     0|acc                    |↑  | 0.6431|±  |0.0272|
|  - prehistory                                         |      1|none  |     0|acc                    |↑  | 0.6512|±  |0.0265|
|  - professional_law                                   |      1|none  |     0|acc                    |↑  | 0.4042|±  |0.0125|
|  - world_religions                                    |      1|none  |     0|acc                    |↑  | 0.7719|±  |0.0322|
| - other                                               |      2|none  |      |acc                    |↑  | 0.6572|±  |0.0083|
|  - business_ethics                                    |      1|none  |     0|acc                    |↑  | 0.6500|±  |0.0479|
|  - clinical_knowledge                                 |      1|none  |     0|acc                    |↑  | 0.6981|±  |0.0283|
|  - college_medicine                                   |      1|none  |     0|acc                    |↑  | 0.5954|±  |0.0374|
|  - global_facts                                       |      1|none  |     0|acc                    |↑  | 0.3300|±  |0.0473|
|  - human_aging                                        |      1|none  |     0|acc                    |↑  | 0.6592|±  |0.0318|
|  - management                                         |      1|none  |     0|acc                    |↑  | 0.7476|±  |0.0430|
|  - marketing                                          |      1|none  |     0|acc                    |↑  | 0.8333|±  |0.0244|
|  - medical_genetics                                   |      1|none  |     0|acc                    |↑  | 0.7000|±  |0.0461|
|  - miscellaneous                                      |      1|none  |     0|acc                    |↑  | 0.7267|±  |0.0159|
|  - nutrition                                          |      1|none  |     0|acc                    |↑  | 0.6993|±  |0.0263|
|  - professional_accounting                            |      1|none  |     0|acc                    |↑  | 0.4752|±  |0.0298|
|  - professional_medicine                              |      1|none  |     0|acc                    |↑  | 0.6176|±  |0.0295|
|  - virology                                           |      1|none  |     0|acc                    |↑  | 0.4940|±  |0.0389|
| - social sciences                                     |      2|none  |      |acc                    |↑  | 0.7124|±  |0.0080|
|  - econometrics                                       |      1|none  |     0|acc                    |↑  | 0.5439|±  |0.0469|
|  - high_school_geography                              |      1|none  |     0|acc                    |↑  | 0.7929|±  |0.0289|
|  - high_school_government_and_politics                |      1|none  |     0|acc                    |↑  | 0.7824|±  |0.0298|
|  - high_school_macroeconomics                         |      1|none  |     0|acc                    |↑  | 0.6487|±  |0.0242|
|  - high_school_microeconomics                         |      1|none  |     0|acc                    |↑  | 0.7353|±  |0.0287|
|  - high_school_psychology                             |      1|none  |     0|acc                    |↑  | 0.8422|±  |0.0156|
|  - human_sexuality                                    |      1|none  |     0|acc                    |↑  | 0.7099|±  |0.0398|
|  - professional_psychology                            |      1|none  |     0|acc                    |↑  | 0.6176|±  |0.0197|
|  - public_relations                                   |      1|none  |     0|acc                    |↑  | 0.6091|±  |0.0467|
|  - security_studies                                   |      1|none  |     0|acc                    |↑  | 0.6612|±  |0.0303|
|  - sociology                                          |      1|none  |     0|acc                    |↑  | 0.7662|±  |0.0299|
|  - us_foreign_policy                                  |      1|none  |     0|acc                    |↑  | 0.8100|±  |0.0394|
| - stem                                                |      2|none  |      |acc                    |↑  | 0.5826|±  |0.0085|
|  - abstract_algebra                                   |      1|none  |     0|acc                    |↑  | 0.3700|±  |0.0485|
|  - anatomy                                            |      1|none  |     0|acc                    |↑  | 0.6370|±  |0.0415|
|  - astronomy                                          |      1|none  |     0|acc                    |↑  | 0.7039|±  |0.0372|
|  - college_biology                                    |      1|none  |     0|acc                    |↑  | 0.7500|±  |0.0362|
|  - college_chemistry                                  |      1|none  |     0|acc                    |↑  | 0.5000|±  |0.0503|
|  - college_computer_science                           |      1|none  |     0|acc                    |↑  | 0.5300|±  |0.0502|
|  - college_mathematics                                |      1|none  |     0|acc                    |↑  | 0.4300|±  |0.0498|
|  - college_physics                                    |      1|none  |     0|acc                    |↑  | 0.5000|±  |0.0498|
|  - computer_security                                  |      1|none  |     0|acc                    |↑  | 0.8000|±  |0.0402|
|  - conceptual_physics                                 |      1|none  |     0|acc                    |↑  | 0.6468|±  |0.0312|
|  - electrical_engineering                             |      1|none  |     0|acc                    |↑  | 0.6069|±  |0.0407|
|  - elementary_mathematics                             |      1|none  |     0|acc                    |↑  | 0.5476|±  |0.0256|
|  - high_school_biology                                |      1|none  |     0|acc                    |↑  | 0.7839|±  |0.0234|
|  - high_school_chemistry                              |      1|none  |     0|acc                    |↑  | 0.6010|±  |0.0345|
|  - high_school_computer_science                       |      1|none  |     0|acc                    |↑  | 0.7200|±  |0.0451|
|  - high_school_mathematics                            |      1|none  |     0|acc                    |↑  | 0.4074|±  |0.0300|
|  - high_school_physics                                |      1|none  |     0|acc                    |↑  | 0.4172|±  |0.0403|
|  - high_school_statistics                             |      1|none  |     0|acc                    |↑  | 0.5370|±  |0.0340|
|  - machine_learning                                   |      1|none  |     0|acc                    |↑  | 0.4375|±  |0.0471|
|niah_turns_32k                                         |      1|none  |     0|acc                    |↑  | 0.9700|±  |0.0108|
|                                                       |       |none  |     0|turn1                  |↑  | 0.9700|±  |0.0171|
|                                                       |       |none  |     0|turn2                  |↑  | 0.9700|±  |0.0171|
|                                                       |       |none  |     0|turn3                  |↑  | 0.9800|±  |0.0141|
|                                                       |       |none  |     0|turn4                  |↑  | 0.9600|±  |0.0197|
|openbookqa                                             |      1|none  |     0|acc                    |↑  | 0.3000|±  |0.0205|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3880|±  |0.0218|
|piqa                                                   |      1|none  |     0|acc                    |↑  | 0.7568|±  |0.0100|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7595|±  |0.0100|
|social_iqa                                             |      0|none  |     0|acc                    |↑  | 0.4882|±  |0.0113|
|truthfulqa_gen                                         |      3|none  |     0|bleu_acc               |↑  | 0.3231|±  |0.0164|
|                                                       |       |none  |     0|bleu_diff              |↑  |-0.1780|±  |0.0426|
|                                                       |       |none  |     0|bleu_max               |↑  | 1.3020|±  |0.0743|
|                                                       |       |none  |     0|rouge1_acc             |↑  | 0.3390|±  |0.0166|
|                                                       |       |none  |     0|rouge1_diff            |↑  |-0.2994|±  |0.0826|
|                                                       |       |none  |     0|rouge1_max             |↑  | 5.2326|±  |0.1612|
|                                                       |       |none  |     0|rouge2_acc             |↑  | 0.2803|±  |0.0157|
|                                                       |       |none  |     0|rouge2_diff            |↑  |-0.4267|±  |0.0907|
|                                                       |       |none  |     0|rouge2_max             |↑  | 3.1285|±  |0.1364|
|                                                       |       |none  |     0|rougeL_acc             |↑  | 0.3280|±  |0.0164|
|                                                       |       |none  |     0|rougeL_diff            |↑  |-0.3549|±  |0.0859|
|                                                       |       |none  |     0|rougeL_max             |↑  | 4.8723|±  |0.1518|
|truthfulqa_mc1                                         |      2|none  |     0|acc                    |↑  | 0.3182|±  |0.0163|
|truthfulqa_mc2                                         |      3|none  |     0|acc                    |↑  | 0.4880|±  |0.0148|
|winogrande                                             |      1|none  |     0|acc                    |↑  | 0.6448|±  |0.0135|
|niah_multikey_1               |      1|none  |     0|  1024|   |1.0000|±  |     0|
|                              |       |none  |     0| 16384|↑  |0.9740|±  |   N/A|
|                              |       |none  |     0|  2048|   |1.0000|±  |     0|
|                              |       |none  |     0| 32768|↑  |0.9660|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |0.9960|±  |   N/A|
|niah_multikey_2               |      1|none  |     0|  1024|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 16384|↑  |0.9500|±  |   N/A|
|                              |       |none  |     0|  2048|   |0.9980|±  |0.0020|
|                              |       |none  |     0| 32768|↑  |0.8500|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |0.9920|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |0.9820|±  |   N/A|
|niah_multikey_3               |      1|none  |     0|  1024|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 16384|↑  |0.5440|±  |   N/A|
|                              |       |none  |     0|  2048|   |0.9880|±  |0.0049|
|                              |       |none  |     0| 32768|↑  |0.1340|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |0.9820|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |0.8580|±  |   N/A|
|niah_multiquery               |      1|none  |     0|  1024|   |0.9995|±  |0.0005|
|                              |       |none  |     0| 16384|↑  |0.9645|±  |   N/A|
|                              |       |none  |     0|  2048|   |0.9995|±  |0.0005|
|                              |       |none  |     0| 32768|↑  |0.9225|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |0.9805|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |0.9670|±  |   N/A|
|niah_multivalue               |      1|none  |     0|  1024|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 16384|↑  |0.9645|±  |   N/A|
|                              |       |none  |     0|  2048|   |0.9910|±  |0.0025|
|                              |       |none  |     0| 32768|↑  |0.9045|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |0.9650|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |0.9670|±  |   N/A|
|niah_single_1                 |      1|none  |     0|  1024|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 16384|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  2048|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 32768|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |1.0000|±  |   N/A|
|niah_single_1_paraphrase      |      1|none  |     0|  1024|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 16384|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  2048|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 32768|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |1.0000|±  |   N/A|
|niah_single_1_preamble        |      1|none  |     0|  1024|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 16384|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  2048|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 32768|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |1.0000|±  |   N/A|
|niah_single_1_qend            |      1|none  |     0|  1024|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 16384|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  2048|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 32768|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |1.0000|±  |   N/A|
|niah_single_2                 |      1|none  |     0|  1024|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 16384|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  2048|   |0.9980|±  |0.0020|
|                              |       |none  |     0| 32768|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |0.9980|±  |   N/A|
|niah_single_2_paraphrase      |      1|none  |     0|  1024|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 16384|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  2048|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 32768|↑  |0.9980|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |1.0000|±  |   N/A|
|niah_single_2_preamble        |      1|none  |     0|  1024|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 16384|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  2048|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 32768|↑  |0.9980|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |1.0000|±  |   N/A|
|niah_single_2_qend            |      1|none  |     0|  1024|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 16384|↑  |0.9980|±  |   N/A|
|                              |       |none  |     0|  2048|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 32768|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |0.9980|±  |   N/A|
|niah_single_2_words           |      1|none  |     0|  1024|   |0.9880|±  |0.0049|
|                              |       |none  |     0| 16384|↑  |0.9180|±  |   N/A|
|                              |       |none  |     0|  2048|   |0.9940|±  |0.0035|
|                              |       |none  |     0| 32768|↑  |0.8400|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |0.9920|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |0.9720|±  |   N/A|
|niah_single_2_words_paraphrase|      1|none  |     0|  1024|   |0.9820|±  |0.0060|
|                              |       |none  |     0| 16384|↑  |0.9800|±  |   N/A|
|                              |       |none  |     0|  2048|   |0.9920|±  |0.0040|
|                              |       |none  |     0| 32768|↑  |0.9140|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |0.9900|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |0.9860|±  |   N/A|
|niah_single_2_words_qend      |      1|none  |     0|  1024|   |0.9700|±  |0.0076|
|                              |       |none  |     0| 16384|↑  |0.9640|±  |   N/A|
|                              |       |none  |     0|  2048|   |0.9900|±  |0.0045|
|                              |       |none  |     0| 32768|↑  |0.9080|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |0.9780|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |0.9840|±  |   N/A|
|niah_single_3                 |      1|none  |     0|  1024|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 16384|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  2048|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 32768|↑  |0.9980|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |0.9980|±  |   N/A|
|niah_single_3_paraphrase      |      1|none  |     0|  1024|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 16384|↑  |0.9980|±  |   N/A|
|                              |       |none  |     0|  2048|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 32768|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |0.9980|±  |   N/A|
|niah_single_3_preamble        |      1|none  |     0|  1024|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 16384|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  2048|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 32768|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |1.0000|±  |   N/A|
|niah_single_3_qend            |      1|none  |     0|  1024|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 16384|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  2048|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 32768|↑  |0.9980|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |0.9980|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |1.0000|±  |   N/A|
|babilong_32k                            |      0|none  |      |acc          |↑  | 0.2420|±  |0.0190|
| - babilong_qa1_32k                     |      0|none  |     2|acc          |↑  | 0.2500|±  |0.0435|
| - babilong_qa2_32k                     |      0|none  |     2|acc          |↑  | 0.1400|±  |0.0349|
| - babilong_qa3_32k                     |      0|none  |     2|acc          |↑  | 0.2300|±  |0.0423|
| - babilong_qa4_32k                     |      0|none  |     2|acc          |↑  | 0.3400|±  |0.0476|
| - babilong_qa5_32k                     |      0|none  |     2|acc          |↑  | 0.2500|±  |0.0435|
|infinitebench_32k                       |      1|none  |      |score        |↑  | 0.3095|±  |0.0108|
| - infinitebench_code_debug_32k         |      1|none  |     0|score        |↑  | 0.3200|±  |0.0469|
| - infinitebench_kv_retrieval_32k       |      1|none  |     0|score        |↑  | 0.0300|±  |0.0171|
| - infinitebench_longbook_choice_eng_32k|      1|none  |     0|score        |↑  | 0.3900|±  |0.0490|
| - infinitebench_longbook_qa_chn_32k    |      1|none  |     0|score        |↑  | 0.1970|±  |0.0251|
| - infinitebench_longbook_qa_eng_32k    |      1|none  |     0|score        |↑  | 0.1644|±  |0.0269|
| - infinitebench_longbook_sum_eng_32k   |      1|none  |     0|score        |↑  | 0.1340|±  |0.0069|
| - infinitebench_longdialogue_qa_eng_32k|      1|none  |     0|score        |↑  | 0.0500|±  |0.0219|
| - infinitebench_math_find_32k          |      1|none  |     0|score        |↑  | 0.2100|±  |0.0409|
| - infinitebench_number_string_32k      |      1|none  |     0|score        |↑  | 0.8000|±  |0.0402|
| - infinitebench_passkey_32k            |      1|none  |     0|score        |↑  | 0.8000|±  |0.0402|
|long_ppl_32k                            |      1|none  |     0|bits_per_byte|↓  | 0.8853|±  |   N/A|
|                                        |       |none  |     0|ppl          |↓  |13.0134|±  |   N/A|
|ruler_cwe      |      1|none  |     0| 32768|↑  |0.0860|±  |   N/A|
|ruler_fwe      |      1|none  |     0| 32768|↑  |0.8267|±  |   N/A|
|ruler_qa_hotpot|      1|none  |     0| 32768|↑  |0.4200|±  |   N/A|
|ruler_qa_squad |      1|none  |     0| 32768|↑  |0.4367|±  |   N/A|
|ruler_vt       |      1|none  |     0| 32768|↑  |0.6740|±  |   N/A|
|babilong_64k                            |      0|none  |      |acc          |↑  | 0.1020|±  |0.0133|
| - babilong_qa1_64k                     |      0|none  |     2|acc          |↑  | 0.0900|±  |0.0288|
| - babilong_qa2_64k                     |      0|none  |     2|acc          |↑  | 0.0400|±  |0.0197|
| - babilong_qa3_64k                     |      0|none  |     2|acc          |↑  | 0.0800|±  |0.0273|
| - babilong_qa4_64k                     |      0|none  |     2|acc          |↑  | 0.2300|±  |0.0423|
| - babilong_qa5_64k                     |      0|none  |     2|acc          |↑  | 0.0700|±  |0.0256|
|infinitebench_64k                       |      1|none  |      |score        |↑  | 0.3605|±  |0.0093|
| - infinitebench_code_debug_64k         |      1|none  |     0|score        |↑  | 0.3300|±  |0.0473|
| - infinitebench_kv_retrieval_64k       |      1|none  |     0|score        |↑  | 0.0300|±  |0.0171|
| - infinitebench_longbook_choice_eng_64k|      1|none  |     0|score        |↑  | 0.4900|±  |0.0502|
| - infinitebench_longbook_qa_chn_64k    |      1|none  |     0|score        |↑  | 0.2155|±  |0.0248|
| - infinitebench_longbook_qa_eng_64k    |      1|none  |     0|score        |↑  | 0.1318|±  |0.0238|
| - infinitebench_longbook_sum_eng_64k   |      1|none  |     0|score        |↑  | 0.1577|±  |0.0078|
| - infinitebench_longdialogue_qa_eng_64k|      1|none  |     0|score        |↑  | 0.0700|±  |0.0256|
| - infinitebench_math_find_64k          |      1|none  |     0|score        |↑  | 0.1900|±  |0.0394|
| - infinitebench_number_string_64k      |      1|none  |     0|score        |↑  | 0.9900|±  |0.0100|
| - infinitebench_passkey_64k            |      1|none  |     0|score        |↑  | 1.0000|±  |     0|
|long_ppl_64k                            |      1|none  |     0|bits_per_byte|↓  | 0.8868|±  |   N/A|
|                                        |       |none  |     0|ppl          |↓  |12.8004|±  |   N/A|
|long_ppl_books_64k                      |    N/A|      |      |             |   |       |   |      |
| - long_ppl_books_64k_16k               |      1|none  |     0|bits_per_byte|↓  | 0.8807|±  |   N/A|
|                                        |       |none  |     0|ppl          |↓  |11.3895|±  |   N/A|
| - long_ppl_books_64k_32k               |      1|none  |     0|bits_per_byte|↓  | 0.8923|±  |   N/A|
|                                        |       |none  |     0|ppl          |↓  |11.7453|±  |   N/A|
| - long_ppl_books_64k_64k               |      1|none  |     0|bits_per_byte|↓  | 0.8673|±  |   N/A|
|                                        |       |none  |     0|ppl          |↓  |11.0359|±  |   N/A|
| - long_ppl_books_64k_8k                |      1|none  |     0|bits_per_byte|↓  | 0.8622|±  |   N/A|
|                                        |       |none  |     0|ppl          |↓  |10.7706|±  |   N/A|
|niah_turns_64k                          |      1|none  |     0|acc          |↑  | 0.7750|±  |0.0262|
|                                        |       |none  |     0|turn1        |↑  | 0.8100|±  |0.0394|
|                                        |       |none  |     0|turn2        |↑  | 0.7300|±  |0.0446|
|                                        |       |none  |     0|turn3        |↑  | 0.7700|±  |0.0423|
|                                        |       |none  |     0|turn4        |↑  | 0.7900|±  |0.0409|
|niah_multikey_1|      1|none  |     0| 65536|↑  |0.6800|±  |   N/A|
|niah_multikey_2|      1|none  |     0| 65536|↑  |0.5100|±  |   N/A|
|niah_multikey_3|      1|none  |     0| 65536|↑  |0.0100|±  |   N/A|
|niah_multiquery|      1|none  |     0| 65536|↑  |0.8150|±  |   N/A|
|niah_multivalue|      1|none  |     0| 65536|↑  |0.7675|±  |   N/A|
|niah_single_1  |      1|none  |     0| 65536|↑  |1.0000|±  |   N/A|
|niah_single_2  |      1|none  |     0| 65536|↑  |0.8800|±  |   N/A|
|niah_single_3  |      1|none  |     0| 65536|↑  |1.0000|±  |   N/A|
|ruler_cwe      |      1|none  |     0| 65536|↑  |0.0480|±  |   N/A|
|ruler_fwe      |      1|none  |     0| 65536|↑  |0.7167|±  |   N/A|
|ruler_qa_hotpot|      1|none  |     0| 65536|↑  |0.3600|±  |   N/A|
|ruler_qa_squad |      1|none  |     0| 65536|↑  |0.3575|±  |   N/A|
|ruler_vt       |      1|none  |     0| 65536|↑  |0.6680|±  |   N/A|
|babilong_128k                            |      0|none  |      |acc          |↑  | 0.0220|±  |0.0066|
| - babilong_qa1_128k                     |      0|none  |     2|acc          |↑  | 0.0000|±  |     0|
| - babilong_qa2_128k                     |      0|none  |     2|acc          |↑  | 0.0200|±  |0.0141|
| - babilong_qa3_128k                     |      0|none  |     2|acc          |↑  | 0.0100|±  |0.0100|
| - babilong_qa4_128k                     |      0|none  |     2|acc          |↑  | 0.0400|±  |0.0197|
| - babilong_qa5_128k                     |      0|none  |     2|acc          |↑  | 0.0400|±  |0.0197|
|infinitebench_128k                       |      1|none  |      |score        |↑  | 0.3242|±  |0.0102|
| - infinitebench_code_debug_128k         |      1|none  |     0|score        |↑  | 0.3300|±  |0.0473|
| - infinitebench_kv_retrieval_128k       |      1|none  |     0|score        |↑  | 0.0000|±  |     0|
| - infinitebench_longbook_choice_eng_128k|      1|none  |     0|score        |↑  | 0.3800|±  |0.0488|
| - infinitebench_longbook_qa_chn_128k    |      1|none  |     0|score        |↑  | 0.1210|±  |0.0160|
| - infinitebench_longbook_qa_eng_128k    |      1|none  |     0|score        |↑  | 0.1098|±  |0.0220|
| - infinitebench_longbook_sum_eng_128k   |      1|none  |     0|score        |↑  | 0.1512|±  |0.0066|
| - infinitebench_longdialogue_qa_eng_128k|      1|none  |     0|score        |↑  | 0.0500|±  |0.0219|
| - infinitebench_math_find_128k          |      1|none  |     0|score        |↑  | 0.4000|±  |0.0492|
| - infinitebench_number_string_128k      |      1|none  |     0|score        |↑  | 0.7000|±  |0.0461|
| - infinitebench_passkey_128k            |      1|none  |     0|score        |↑  | 1.0000|±  |     0|
|long_ppl_128k                            |      1|none  |     0|bits_per_byte|↓  | 0.9815|±  |   N/A|
|                                         |       |none  |     0|ppl          |↓  |15.6106|±  |   N/A|
|long_ppl_books_128k                      |    N/A|      |      |             |   |       |   |      |
| - long_ppl_books_128k_128k              |      1|none  |     0|bits_per_byte|↓  | 0.9546|±  |   N/A|
|                                         |       |none  |     0|ppl          |↓  |13.9162|±  |   N/A|
| - long_ppl_books_128k_16k               |      1|none  |     0|bits_per_byte|↓  | 1.0966|±  |   N/A|
|                                         |       |none  |     0|ppl          |↓  |20.6746|±  |   N/A|
| - long_ppl_books_128k_32k               |      1|none  |     0|bits_per_byte|↓  | 1.0728|±  |   N/A|
|                                         |       |none  |     0|ppl          |↓  |19.3295|±  |   N/A|
| - long_ppl_books_128k_64k               |      1|none  |     0|bits_per_byte|↓  | 0.9945|±  |   N/A|
|                                         |       |none  |     0|ppl          |↓  |15.6907|±  |   N/A|
| - long_ppl_books_128k_8k                |      1|none  |     0|bits_per_byte|↓  | 0.9026|±  |   N/A|
|                                         |       |none  |     0|ppl          |↓  |12.0401|±  |   N/A|
|niah_turns_128k                          |      1|none  |     0|acc          |↑  | 0.6225|±  |0.0272|
|                                         |       |none  |     0|turn1        |↑  | 0.5700|±  |0.0498|
|                                         |       |none  |     0|turn2        |↑  | 0.7000|±  |0.0461|
|                                         |       |none  |     0|turn3        |↑  | 0.6500|±  |0.0479|
|                                         |       |none  |     0|turn4        |↑  | 0.5700|±  |0.0498|
|niah_multikey_1|      1|none  |     0|131072|↑  |0.6700|±  |   N/A|
|niah_multikey_2|      1|none  |     0|131072|↑  |0.0700|±  |   N/A|
|niah_multikey_3|      1|none  |     0|131072|↑  |0.0000|±  |   N/A|
|niah_multiquery|      1|none  |     0|131072|↑  |0.4350|±  |   N/A|
|niah_multivalue|      1|none  |     0|131072|↑  |0.4500|±  |   N/A|
|niah_single_1  |      1|none  |     0|131072|↑  |0.9700|±  |   N/A|
|niah_single_2  |      1|none  |     0|131072|↑  |0.7200|±  |   N/A|
|niah_single_3  |      1|none  |     0|131072|↑  |0.9400|±  |   N/A|
|ruler_cwe      |      1|none  |     0|131072|↑  |0.0460|±  |   N/A|
|ruler_fwe      |      1|none  |     0|131072|↑  |0.3767|±  |   N/A|
|ruler_qa_hotpot|      1|none  |     0|131072|↑  |0.2600|±  |   N/A|
|ruler_qa_squad |      1|none  |     0|131072|↑  |0.1983|±  |   N/A|
|ruler_vt       |      1|none  |     0|131072|↑  |0.0040|±  |   N/A|
|babilong_256k                            |      0|none  |      |acc          |↑  | 0.0040|±  |0.0028|
| - babilong_qa1_256k                     |      0|none  |     2|acc          |↑  | 0.0000|±  |     0|
| - babilong_qa2_256k                     |      0|none  |     2|acc          |↑  | 0.0000|±  |     0|
| - babilong_qa3_256k                     |      0|none  |     2|acc          |↑  | 0.0000|±  |     0|
| - babilong_qa4_256k                     |      0|none  |     2|acc          |↑  | 0.0000|±  |     0|
| - babilong_qa5_256k                     |      0|none  |     2|acc          |↑  | 0.0200|±  |0.0141|
|infinitebench_256k                       |      1|none  |      |score        |↑  | 0.2035|±  |0.0087|
| - infinitebench_code_debug_256k         |      1|none  |     0|score        |↑  | 0.0000|±  |     0|
| - infinitebench_kv_retrieval_256k       |      1|none  |     0|score        |↑  | 0.0000|±  |     0|
| - infinitebench_longbook_choice_eng_256k|      1|none  |     0|score        |↑  | 0.2400|±  |0.0429|
| - infinitebench_longbook_qa_chn_256k    |      1|none  |     0|score        |↑  | 0.0329|±  |0.0056|
| - infinitebench_longbook_qa_eng_256k    |      1|none  |     0|score        |↑  | 0.0140|±  |0.0048|
| - infinitebench_longbook_sum_eng_256k   |      1|none  |     0|score        |↑  | 0.0782|±  |0.0043|
| - infinitebench_longdialogue_qa_eng_256k|      1|none  |     0|score        |↑  | 0.0100|±  |0.0100|
| - infinitebench_math_find_256k          |      1|none  |     0|score        |↑  | 0.5000|±  |0.0503|
| - infinitebench_number_string_256k      |      1|none  |     0|score        |↑  | 0.2800|±  |0.0451|
| - infinitebench_passkey_256k            |      1|none  |     0|score        |↑  | 0.8800|±  |0.0327|
|long_ppl_256k                            |      1|none  |     0|bits_per_byte|↓  | 1.1171|±  |   N/A|
|                                         |       |none  |     0|ppl          |↓  |26.0855|±  |   N/A|
|long_ppl_books_256k                      |    N/A|      |      |             |   |       |   |      |
| - long_ppl_books_256k_128k              |      1|none  |     0|bits_per_byte|↓  | 1.3032|±  |   N/A|
|                                         |       |none  |     0|ppl          |↓  |36.3972|±  |   N/A|
| - long_ppl_books_256k_16k               |      1|none  |     0|bits_per_byte|↓  | 1.2828|±  |   N/A|
|                                         |       |none  |     0|ppl          |↓  |34.5800|±  |   N/A|
| - long_ppl_books_256k_256k              |      1|none  |     0|bits_per_byte|↓  | 1.1893|±  |   N/A|
|                                         |       |none  |     0|ppl          |↓  |25.5616|±  |   N/A|
| - long_ppl_books_256k_32k               |      1|none  |     0|bits_per_byte|↓  | 1.3029|±  |   N/A|
|                                         |       |none  |     0|ppl          |↓  |36.4858|±  |   N/A|
| - long_ppl_books_256k_64k               |      1|none  |     0|bits_per_byte|↓  | 1.3152|±  |   N/A|
|                                         |       |none  |     0|ppl          |↓  |38.1300|±  |   N/A|
| - long_ppl_books_256k_8k                |      1|none  |     0|bits_per_byte|↓  | 1.0785|±  |   N/A|
|                                         |       |none  |     0|ppl          |↓  |19.5530|±  |   N/A|
|niah_turns_256k                          |      1|none  |     0|acc          |↑  | 0.0150|±  |0.0060|
|                                         |       |none  |     0|turn1        |↑  | 0.0500|±  |0.0219|
|                                         |       |none  |     0|turn2        |↑  | 0.0100|±  |0.0100|
|                                         |       |none  |     0|turn3        |↑  | 0.0000|±  |0.0000|
|                                         |       |none  |     0|turn4        |↑  | 0.0000|±  |0.0000|
|niah_multikey_1|      1|none  |     0|262144|   |0.0400|±  |0.0197|
|niah_multikey_2|      1|none  |     0|262144|   |0.0000|±  |0.0000|
|niah_multikey_3|      1|none  |     0|262144|   |0.0000|±  |0.0000|
|niah_multiquery|      1|none  |     0|262144|   |0.0000|±  |0.0000|
|niah_multivalue|      1|none  |     0|262144|   |0.0050|±  |0.0035|
|niah_single_1  |      1|none  |     0|262144|   |0.1200|±  |0.0327|
|niah_single_2  |      1|none  |     0|262144|   |0.0200|±  |0.0141|
|niah_single_3  |      1|none  |     0|262144|   |0.0000|±  |0.0000|
|ruler_cwe      |      1|none  |     0|262144|   |0.0040|±  |0.0020|
|ruler_fwe      |      1|none  |     0|262144|   |0.0167|±  |0.0073|
|ruler_qa_hotpot|      1|none  |     0|262144|   |0.0700|±  |0.0256|
|ruler_vt       |      1|none  |     0|262144|   |0.0000|±  |0.0000|

    """
}


# ==============================================================================
# 1. 配置区：表头顺序、映射与指标字典
# ==============================================================================

ORDERED_COLUMNS = [
    "Common Sense Reasoning (ACC)", "recall-intensive tasks", "LongBench", "LongBench_e", 
    "BoolQ", "PIQA", "SIQA", "HellaSwag", "WinoGrande", "ARC-e", "ARC-c", "OpenBookQA", "AVG_CSR", 
    "ceval-valid", "ifeval", "mmlu", "truthfulqa_gen", "truthfulqa_mc1", "truthfulqa_mc2", "AVG_General",
    "SWDE", "FDA", "TriviaQA", "NQ", "DROP", "SQUAD", "AVG_Recall",
    "gov_report", "hotpotqa", "lcc", "Qasper", "MFQA_zh", "MFQA_en", "musique", "narrativeqa", 
    "passage_count", "passage_retrieval_en", "longbench_repobench-p", "samsum", "triviaqa", 
    "trec", "lsht", "multi_news", "qmsum", "dureader", "2wikimqa", "vcsum", "AVG_LB", "AVG_LB_wo",
    "2wikimqa_e", "gov_report_e", "hotpotqa_e", "lcc_e", "multi_news_e", "MFQA_en_e", 
    "passage_count_e", "passage_retrieval_en_e", "qasper_e", "repobench-p_e", "samsum_e", 
    "trec_e", "triviaqa_e", "AVG_LBe", "AVG_LBe_wo", "AVG_all",
    "niah_single_1", "niah_single_2", "niah_single_3", "AVG_NIAH",
    # RULER 多针（lm-eval 自带）与 cue 适用边界的提示扰动（eval_tasks/niah_cue），均取 32K。
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3", "niah_multivalue", "niah_multiquery", "AVG_NIAH_multi",
    "niah_single_1_qend", "niah_single_1_preamble", "niah_single_1_paraphrase",
    "niah_single_2_qend", "niah_single_2_preamble", "niah_single_2_paraphrase",
    "niah_single_3_qend", "niah_single_3_preamble", "niah_single_3_paraphrase", "AVG_NIAH_cue",
    # niahLogKV 泛化：值为随机英文词、不含数字的针（eval_tasks/niah_cue/*_words），取 32K。
    "niah_single_2_words", "niah_single_2_words_qend", "niah_single_2_words_paraphrase", "AVG_NIAH_words",
]
LONG_LENGTHS = {"32768": "32k", "65536": "64k", "131072": "128k", "262144": "256k"}
LONG_NIAH = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2",
             "niah_multikey_3", "niah_multivalue", "niah_multiquery"]
LONG_RULER = LONG_NIAH + ["ruler_vt", "ruler_cwe", "ruler_fwe", "ruler_qa_squad", "ruler_qa_hotpot"]
BLOCK_LENGTHS = ("32k", "64k", "128k", "256k")
BABILONG_LENGTHS = PPL_LENGTHS = LONGBENCH2_LENGTHS = TURN_LENGTHS = BLOCK_LENGTHS
TURN_TASKS = [f"niah_turns_{n}" for n in TURN_LENGTHS]
# 书的困惑度曲线的截断点：每个配置长度只到它自己的长度（long_ppl_books_32k 为 8k、16k、32k）。
BOOK_POINTS = {n: [f"{2 ** k}k" for k in range(3, 9) if 2 ** k <= int(n[:-1])] for n in BLOCK_LENGTHS}
# InfiniteBench 的十个任务（eval_tasks/infinitebench，32K-256K），AVG_InfiniteBench 为十个的平均（论文的 Avg）。
INFINITEBENCH = ["passkey", "number_string", "kv_retrieval", "longbook_sum_eng", "longbook_qa_eng", "longbook_choice_eng",
                 "longbook_qa_chn", "longdialogue_qa_eng", "code_debug", "math_find"]
INFINITEBENCH_LENGTHS = BLOCK_LENGTHS
# LongBench v2 的 20 个子任务（lm-eval 的 longbench2_*，各长度的组 longbench2_<长度> 都是这 20 个，日志里子任务名不带长度，
# 按它所在的组记到该长度），LongBench2_<长度> 是组的总分（按题数加权，即全部题目的正确率，lm-eval 报告的那个）。
LONGBENCH2_SUBTASKS = ["academic_multi", "academic_single", "agent_history", "code", "detective", "dialogue_history",
                       "event_order", "fin_multi", "fin_single", "govt_multi", "govt_single", "graph", "legal_multi",
                       "legal_single", "lit_single", "many_shot", "news_multi", "table", "translate", "user_guide"]
LONGBENCH2_GROUP = re.compile(r"longbench2_(32k|64k|128k|256k)")
# 每个长度实际跑的 RULER 任务：256K 没有 ruler_qa_squad（SQuAD 的文档池填不满 256K）。
RULER_AT = {n: [t for t in LONG_RULER if n != "256k" or t != "ruler_qa_squad"] for n in BLOCK_LENGTHS}


# 长文本列先按长度分块（32K 主线之后是 32K、64K、128K、256K 各一块，论文里 32K-256K 可以单独成表），块内再按任务分组、
# 每组的平均紧跟在它平均的任务之后。只列评测会产出结果的列，各块的顺序相同：
#   RULER 单针 niah_single_1-3、AVG_NIAH；多针 niah_multikey_1-3、niah_multivalue、niah_multiquery、AVG_NIAH_multi；
#   其余 ruler_vt、ruler_cwe、ruler_fwe、ruler_qa_squad（256K 没有）、ruler_qa_hotpot、AVG_RULER（该长度全部 RULER 任务的
#   平均：13 个，256K 为 12 个）；BABILong qa1-qa5、AVG_babilong；
#   LongBench v2 的 20 个子任务 longbench2_<子任务>_<长度>、LongBench2_<长度>（组的总分，按题数加权）；
#   同一份缓存上逐轮问 niah_turns_<长度>_turn1-4、niah_turns_<长度>（四轮平均）；
#   书的困惑度曲线 long_ppl_books_<长度>_<截断长度>（截断长度 8k 起到该长度，同一批书，越低越好），之后是同长度的
#   long_ppl_<长度>（PG19 测试集文档末尾的困惑度，越低越好）；
#   InfiniteBench 十个任务 infinitebench_<任务>_<长度>（官方评分）、AVG_InfiniteBench_<长度>。
# 32K 块的捞针列与 32K 主线的 niah_single_1 等是同一份结果，重复列出。都不计入 AVG_all。新列加在所属任务组里、它的平均
# 之前，并更新 tests/parse_lmeval_table_columns.txt（测试检查各块顺序一致、各长度的列连在一起、平均紧跟其任务）。
def _length_block(n):
    """One length's columns: grouped by task, each average right after the tasks it averages."""
    single, multi = LONG_NIAH[:3], LONG_NIAH[3:]
    rest = [t for t in RULER_AT[n] if t not in LONG_NIAH]
    columns = [f"{t}_{n}" for t in single] + [f"AVG_NIAH_{n}"]  # RULER: the single needles
    columns += [f"{t}_{n}" for t in multi] + [f"AVG_NIAH_multi_{n}"]  # the multi-needle tasks
    columns += [f"{t}_{n}" for t in rest] + [f"AVG_RULER_{n}"]  # the rest; AVG_RULER: all of RULER at n
    columns += [f"babilong_qa{i}_{n}" for i in range(1, 6)] + [f"AVG_babilong_{n}"]
    columns += [f"longbench2_{t}_{n}" for t in LONGBENCH2_SUBTASKS] + [f"LongBench2_{n}"]  # the group's total last
    columns += [f"niah_turns_{n}_turn{i}" for i in range(1, 5)] + [f"niah_turns_{n}"]  # niah_turns: the four turns' mean
    columns += [f"long_ppl_books_{n}_{q}" for q in BOOK_POINTS[n]] + [f"long_ppl_{n}"]
    if n in INFINITEBENCH_LENGTHS:
        columns += [f"infinitebench_{t}_{n}" for t in INFINITEBENCH] + [f"AVG_InfiniteBench_{n}"]
    return columns


for _length in BLOCK_LENGTHS:
    ORDERED_COLUMNS += _length_block(_length)
NIAH_MULTI = ["niah_multikey_1", "niah_multikey_2", "niah_multikey_3", "niah_multivalue", "niah_multiquery"]
NIAH_CUE = [f"niah_single_{n}_{v}" for n in (1, 2, 3) for v in ("qend", "preamble", "paraphrase")]
NIAH_WORDS = ["niah_single_2_words", "niah_single_2_words_qend", "niah_single_2_words_paraphrase"]

RAW_TO_TARGET_MAP = {
    "boolq": "BoolQ", "piqa": "PIQA", "social_iqa": "SIQA", "hellaswag": "HellaSwag",
    "winogrande": "WinoGrande", "arc_easy": "ARC-e", "arc_challenge": "ARC-c", "openbookqa": "OpenBookQA",
    "longbench_gov_report": "gov_report", "longbench_hotpotqa": "hotpotqa", "longbench_lcc": "lcc",
    "longbench_qasper": "Qasper", "longbench_multifieldqa_zh": "MFQA_zh", "longbench_multifieldqa_en": "MFQA_en",
    "longbench_musique": "musique", "longbench_narrativeqa": "narrativeqa", "longbench_passage_count": "passage_count",
    "longbench_passage_retrieval_en": "passage_retrieval_en", "longbench_samsum": "samsum", 
    "longbench_triviaqa": "triviaqa", "longbench_trec": "trec", "longbench_lsht": "lsht", 
    "longbench_multi_news": "multi_news", "longbench_qmsum": "qmsum", "longbench_dureader": "dureader",
    "longbench_2wikimqa": "2wikimqa", "longbench_vcsum": "vcsum",
    "longbench_2wikimqa_e": "2wikimqa_e", "longbench_gov_report_e": "gov_report_e", "longbench_hotpotqa_e": "hotpotqa_e",
    "longbench_lcc_e": "lcc_e", "longbench_multi_news_e": "multi_news_e", "longbench_multifieldqa_en_e": "MFQA_en_e",
    "longbench_passage_count_e": "passage_count_e", "longbench_passage_retrieval_en_e": "passage_retrieval_en_e",
    "longbench_qasper_e": "qasper_e", "longbench_repobench-p_e": "repobench-p_e", "longbench_samsum_e": "samsum_e",
    "longbench_trec_e": "trec_e", "longbench_triviaqa_e": "triviaqa_e",
    **{f"longbench2_{n}": f"LongBench2_{n}" for n in LONGBENCH2_LENGTHS},
}

TARGET_METRIC_MAP = {
    "BoolQ": "acc", "PIQA": "acc", "SIQA": "acc", "HellaSwag": "acc_norm",
    "WinoGrande": "acc", "ARC-e": "acc", "ARC-c": "acc_norm", "OpenBookQA": "acc_norm",
    "ceval-valid": "acc", "mmlu": "acc", "ifeval": "prompt_level_strict_acc",
    "truthfulqa_gen": "rouge1_max", "truthfulqa_mc1": "acc", "truthfulqa_mc2": "acc",
    "SWDE": "f1", "FDA": "f1", "TriviaQA": "exact_match", "NQ": "exact_match", "DROP": "f1", "SQUAD": "exact_match",
    "niah_single_1": "32768", "niah_single_2": "32768", "niah_single_3": "32768",
    **{task: "32768" for task in NIAH_MULTI + NIAH_CUE + NIAH_WORDS},
    **{f"babilong_qa{i}_{n}": "acc" for i in range(1, 6) for n in BABILONG_LENGTHS},
    **{f"long_ppl_{n}": "ppl" for n in PPL_LENGTHS},
    **{f"LongBench2_{n}": "acc" for n in LONGBENCH2_LENGTHS},
    **{t: "acc" for t in TURN_TASKS},
    **{f"long_ppl_books_{n}_{q}": "ppl" for n in BLOCK_LENGTHS for q in BOOK_POINTS[n]},
}
for col in ORDERED_COLUMNS:
    if col not in TARGET_METRIC_MAP and not col.startswith(("AVG", "Common", "recall", "LongBench")):
        TARGET_METRIC_MAP[col] = "score"

# ==============================================================================
# 2. 定义各模块的平均值计算范围
# ==============================================================================
AVG_GROUPS = {
    "AVG_CSR": ["BoolQ", "PIQA", "SIQA", "HellaSwag", "WinoGrande", "ARC-e", "ARC-c", "OpenBookQA"],
    "AVG_General": ["ceval-valid", "ifeval", "mmlu", "truthfulqa_gen", "truthfulqa_mc1", "truthfulqa_mc2"],
    "AVG_Recall": ["SWDE", "FDA", "TriviaQA", "NQ", "DROP", "SQUAD"],
    "AVG_LB": ["gov_report", "hotpotqa", "lcc", "Qasper", "MFQA_zh", "MFQA_en", "musique", "narrativeqa", "passage_count", "passage_retrieval_en", "longbench_repobench-p", "samsum", "triviaqa", "trec", "lsht", "multi_news", "qmsum", "dureader", "2wikimqa", "vcsum"],
    # 注：如果你需要计算 w/o（比如去掉代码或篇章检索），可以在下面的列表里删掉对应的名称
    "AVG_LB_wo": ["gov_report", "hotpotqa", "Qasper", "MFQA_zh", "MFQA_en", "musique", "narrativeqa", "samsum", "triviaqa", "trec", "lsht", "multi_news", "qmsum", "dureader", "2wikimqa", "vcsum"], 
    "AVG_LBe": ["2wikimqa_e", "gov_report_e", "hotpotqa_e", "lcc_e", "multi_news_e", "MFQA_en_e", "passage_count_e", "passage_retrieval_en_e", "qasper_e", "repobench-p_e", "samsum_e", "trec_e", "triviaqa_e"],
    "AVG_NIAH": ["niah_single_1", "niah_single_2", "niah_single_3"],
    "AVG_NIAH_multi": NIAH_MULTI,
    "AVG_NIAH_cue": NIAH_CUE,
    "AVG_NIAH_words": NIAH_WORDS,
    **{f"AVG_NIAH_{n}": [f"niah_single_{k}_{n}" for k in (1, 2, 3)] for n in BLOCK_LENGTHS},
    **{f"AVG_NIAH_multi_{n}": [f"{t}_{n}" for t in NIAH_MULTI] for n in BLOCK_LENGTHS},
    **{f"AVG_RULER_{n}": [f"{t}_{n}" for t in RULER_AT[n]] for n in BLOCK_LENGTHS},
    **{f"AVG_babilong_{n}": [f"babilong_qa{i}_{n}" for i in range(1, 6)] for n in BABILONG_LENGTHS},
    **{f"AVG_InfiniteBench_{n}": [f"infinitebench_{t}_{n}" for t in INFINITEBENCH] for n in INFINITEBENCH_LENGTHS},
}
# 每个平均及其任务都是表头里的列（否则 build_table 会在表尾多出一列、或悄悄少平均一项）。
assert all(c in ORDERED_COLUMNS for avg, tasks in AVG_GROUPS.items() for c in (avg, *tasks)), \
    [c for avg, tasks in AVG_GROUPS.items() for c in (avg, *tasks) if c not in ORDERED_COLUMNS]
# 新增的诊断组不计入 AVG_all，旧结果与新结果的总平均保持可比。
DIAGNOSTIC_GROUPS = {"AVG_NIAH_multi", "AVG_NIAH_cue", "AVG_NIAH_words",
                     *(f"AVG_NIAH_{n}" for n in BLOCK_LENGTHS), *(f"AVG_NIAH_multi_{n}" for n in BLOCK_LENGTHS),
                     *(f"AVG_RULER_{n}" for n in BLOCK_LENGTHS), *(f"AVG_babilong_{n}" for n in BABILONG_LENGTHS),
                     *(f"AVG_InfiniteBench_{n}" for n in INFINITEBENCH_LENGTHS)}

# ==============================================================================
# 3. 解析与计算核心代码
# ==============================================================================
TABLE_HEADER = "|Tasks|Version|Filter|n-shot|Metric| |Value| |Stderr|"
TABLE_ROW = re.compile(r"\|(?:[^|]*\|){9}\s*$")  # lm-eval 结果表的一行：九列，十个竖线


def table_lines(text):
    """``text`` 里所有 lm-eval 结果表的数据行：粘贴的表，或整份 eval 日志（每个阶段一张表，行首可能有前缀）。
    去掉表头（Tasks / Groups）与分隔行，保留 lm-eval 的九列。"""
    rows = []
    for line in text.splitlines():
        match = TABLE_ROW.search(line.rstrip())
        if not match:
            continue
        row = match.group(0).strip()
        first = row.split("|")[1].strip()
        if first in ("Tasks", "Groups") or set(row) <= set("|-: "):
            continue
        rows.append(row)
    return rows


def parse_single_model(model_name, markdown_text):
    rows = table_lines(markdown_text)
    if not rows: return {"Model": model_name}

    # 全按字符串读：只有 RULER 行的表，Metric 列（65536 等）也要和 LONG_LENGTHS 的键比较。
    df = pd.read_csv(io.StringIO('\n'.join([TABLE_HEADER] + rows)), sep='|', dtype=str, keep_default_na=False)
    df = df.iloc[:, 1:-1]
    df.columns = df.columns.str.strip()
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    df['Tasks'] = df['Tasks'].replace('', np.nan).ffill()
    row_data = {"Model": model_name}
    longbench2_length = None  # the length of the longbench2_<length> group whose subtasks follow

    for _, row in df.iterrows():
        listed = str(row['Tasks'])
        raw_task = re.sub(r"^-\s*", "", listed)  # a group's subtasks are listed as " - name"
        metric = row['Metric']
        value = row['Value']
        if not listed.startswith("-"):
            group = LONGBENCH2_GROUP.fullmatch(raw_task)
            longbench2_length = group.group(1) if group else None
        elif longbench2_length and raw_task.startswith("longbench2_") and metric == "acc":
            # LongBench v2's subtasks carry no length: they belong to the group above them.
            if raw_task[len("longbench2_"):] in LONGBENCH2_SUBTASKS:
                try:
                    row_data[f"{raw_task}_{longbench2_length}"] = float(value)
                except ValueError:
                    pass
            continue

        target_task = RAW_TO_TARGET_MAP.get(raw_task, raw_task)
        expected_metric = TARGET_METRIC_MAP.get(target_task)
        if target_task in TURN_TASKS and re.fullmatch(r"turn\d", metric):  # niah_turns: one column per turn
            try:
                row_data[f"{target_task}_{metric}"] = float(value)
            except ValueError:
                pass
            continue
        if target_task in LONG_RULER and metric in LONG_LENGTHS:  # RULER at 32K-256K: <task>_<length>
            try:
                row_data[f"{target_task}_{LONG_LENGTHS[metric]}"] = float(value)
            except ValueError:
                pass
            if metric != "32768":
                continue  # 32K 的捞针结果另外还填进 32K 主线的 niah_single_1 等列（下面的通用匹配）

        if expected_metric and metric == expected_metric:
            try:
                row_data[target_task] = float(value)
            except ValueError:
                pass # 忽略非数字
    return row_data


def build_table(outputs):
    """``{模型名: 结果表或日志文本}`` -> 一行一个模型、列为 ORDERED_COLUMNS 的表，含各组平均与 AVG_all。"""
    # 1. 组合数据为 DataFrame
    final_df = pd.DataFrame([parse_single_model(m, text) for m, text in outputs.items()])
    final_df = final_df.reindex(columns=['Model'] + ORDERED_COLUMNS) # 对齐格式

    # 2. ================= 自动计算各组平均值 =================
    for avg_col, sub_tasks in AVG_GROUPS.items():
        # 选出有实际成绩的列，利用 pd.to_numeric 把可能混入的字符转为 NaN 并求均值
        valid_cols = [t for t in sub_tasks if t in final_df.columns]
        final_df[avg_col] = final_df[valid_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1).round(4)

    # 计算全局汇总平均 AVG_all (综合所有分组子任务的基础列)
    all_base_tasks = list(set([task for group, group_tasks in AVG_GROUPS.items() if group not in DIAGNOSTIC_GROUPS
                               for task in group_tasks]))
    valid_base_tasks = [t for t in all_base_tasks if t in final_df.columns]
    final_df['AVG_all'] = final_df[valid_base_tasks].apply(pd.to_numeric, errors='coerce').mean(axis=1).round(4)
    return final_df


if __name__ == "__main__":
    final_df = build_table(model_outputs)
    # 3. 清理空白，输出剪贴板友好格式
    final_df = final_df.fillna("")
    print(final_df.to_csv(sep=',', index=False))
