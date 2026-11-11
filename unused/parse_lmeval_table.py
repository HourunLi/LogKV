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
|-------------------------------------------------------|------:|------|-----:|-----------------------|---|------:|---|------|
|arc_challenge                                          |      1|none  |     0|acc                    |↑  | 0.3933|±  |0.0143|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4292|±  |0.0145|
|arc_easy                                               |      1|none  |     0|acc                    |↑  | 0.7471|±  |0.0089|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7003|±  |0.0094|
|boolq                                                  |      2|none  |     0|acc                    |↑  | 0.7352|±  |0.0077|
|ceval-valid                                            |      2|none  |      |acc                    |↑  | 0.6114|±  |0.0128|
|                                                       |       |none  |      |acc_norm               |↑  | 0.6114|±  |0.0128|
| - ceval-valid_accountant                              |      2|none  |     0|acc                    |↑  | 0.5102|±  |0.0722|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5102|±  |0.0722|
| - ceval-valid_advanced_mathematics                    |      2|none  |     0|acc                    |↑  | 0.5263|±  |0.1177|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5263|±  |0.1177|
| - ceval-valid_art_studies                             |      2|none  |     0|acc                    |↑  | 0.5455|±  |0.0880|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5455|±  |0.0880|
| - ceval-valid_basic_medicine                          |      2|none  |     0|acc                    |↑  | 0.6842|±  |0.1096|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6842|±  |0.1096|
| - ceval-valid_business_administration                 |      2|none  |     0|acc                    |↑  | 0.6970|±  |0.0812|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6970|±  |0.0812|
| - ceval-valid_chinese_language_and_literature         |      2|none  |     0|acc                    |↑  | 0.6087|±  |0.1041|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6087|±  |0.1041|
| - ceval-valid_civil_servant                           |      2|none  |     0|acc                    |↑  | 0.4255|±  |0.0729|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4255|±  |0.0729|
| - ceval-valid_clinical_medicine                       |      2|none  |     0|acc                    |↑  | 0.5909|±  |0.1073|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5909|±  |0.1073|
| - ceval-valid_college_chemistry                       |      2|none  |     0|acc                    |↑  | 0.4583|±  |0.1039|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4583|±  |0.1039|
| - ceval-valid_college_economics                       |      2|none  |     0|acc                    |↑  | 0.6182|±  |0.0661|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6182|±  |0.0661|
| - ceval-valid_college_physics                         |      2|none  |     0|acc                    |↑  | 0.4211|±  |0.1164|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4211|±  |0.1164|
| - ceval-valid_college_programming                     |      2|none  |     0|acc                    |↑  | 0.6757|±  |0.0780|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6757|±  |0.0780|
| - ceval-valid_computer_architecture                   |      2|none  |     0|acc                    |↑  | 0.7619|±  |0.0952|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7619|±  |0.0952|
| - ceval-valid_computer_network                        |      2|none  |     0|acc                    |↑  | 0.4737|±  |0.1177|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4737|±  |0.1177|
| - ceval-valid_discrete_mathematics                    |      2|none  |     0|acc                    |↑  | 0.2500|±  |0.1118|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.2500|±  |0.1118|
| - ceval-valid_education_science                       |      2|none  |     0|acc                    |↑  | 0.7586|±  |0.0809|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7586|±  |0.0809|
| - ceval-valid_electrical_engineer                     |      2|none  |     0|acc                    |↑  | 0.4054|±  |0.0818|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4054|±  |0.0818|
| - ceval-valid_environmental_impact_assessment_engineer|      2|none  |     0|acc                    |↑  | 0.6129|±  |0.0889|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6129|±  |0.0889|
| - ceval-valid_fire_engineer                           |      2|none  |     0|acc                    |↑  | 0.4194|±  |0.0901|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4194|±  |0.0901|
| - ceval-valid_high_school_biology                     |      2|none  |     0|acc                    |↑  | 0.7368|±  |0.1038|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7368|±  |0.1038|
| - ceval-valid_high_school_chemistry                   |      2|none  |     0|acc                    |↑  | 0.3684|±  |0.1137|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3684|±  |0.1137|
| - ceval-valid_high_school_chinese                     |      2|none  |     0|acc                    |↑  | 0.3684|±  |0.1137|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3684|±  |0.1137|
| - ceval-valid_high_school_geography                   |      2|none  |     0|acc                    |↑  | 0.6316|±  |0.1137|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6316|±  |0.1137|
| - ceval-valid_high_school_history                     |      2|none  |     0|acc                    |↑  | 0.8000|±  |0.0918|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8000|±  |0.0918|
| - ceval-valid_high_school_mathematics                 |      2|none  |     0|acc                    |↑  | 0.3889|±  |0.1182|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3889|±  |0.1182|
| - ceval-valid_high_school_physics                     |      2|none  |     0|acc                    |↑  | 0.6316|±  |0.1137|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6316|±  |0.1137|
| - ceval-valid_high_school_politics                    |      2|none  |     0|acc                    |↑  | 0.8421|±  |0.0859|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8421|±  |0.0859|
| - ceval-valid_ideological_and_moral_cultivation       |      2|none  |     0|acc                    |↑  | 0.9474|±  |0.0526|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.9474|±  |0.0526|
| - ceval-valid_law                                     |      2|none  |     0|acc                    |↑  | 0.4167|±  |0.1028|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4167|±  |0.1028|
| - ceval-valid_legal_professional                      |      2|none  |     0|acc                    |↑  | 0.3043|±  |0.0981|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3043|±  |0.0981|
| - ceval-valid_logic                                   |      2|none  |     0|acc                    |↑  | 0.5455|±  |0.1087|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5455|±  |0.1087|
| - ceval-valid_mao_zedong_thought                      |      2|none  |     0|acc                    |↑  | 0.7917|±  |0.0847|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7917|±  |0.0847|
| - ceval-valid_marxism                                 |      2|none  |     0|acc                    |↑  | 0.8421|±  |0.0859|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8421|±  |0.0859|
| - ceval-valid_metrology_engineer                      |      2|none  |     0|acc                    |↑  | 0.7083|±  |0.0948|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7083|±  |0.0948|
| - ceval-valid_middle_school_biology                   |      2|none  |     0|acc                    |↑  | 0.9524|±  |0.0476|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.9524|±  |0.0476|
| - ceval-valid_middle_school_chemistry                 |      2|none  |     0|acc                    |↑  | 0.8500|±  |0.0819|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8500|±  |0.0819|
| - ceval-valid_middle_school_geography                 |      2|none  |     0|acc                    |↑  | 0.5000|±  |0.1508|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5000|±  |0.1508|
| - ceval-valid_middle_school_history                   |      2|none  |     0|acc                    |↑  | 0.9091|±  |0.0627|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.9091|±  |0.0627|
| - ceval-valid_middle_school_mathematics               |      2|none  |     0|acc                    |↑  | 0.4737|±  |0.1177|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4737|±  |0.1177|
| - ceval-valid_middle_school_physics                   |      2|none  |     0|acc                    |↑  | 0.8421|±  |0.0859|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8421|±  |0.0859|
| - ceval-valid_middle_school_politics                  |      2|none  |     0|acc                    |↑  | 0.9048|±  |0.0656|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.9048|±  |0.0656|
| - ceval-valid_modern_chinese_history                  |      2|none  |     0|acc                    |↑  | 0.6522|±  |0.1015|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6522|±  |0.1015|
| - ceval-valid_operating_system                        |      2|none  |     0|acc                    |↑  | 0.6842|±  |0.1096|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6842|±  |0.1096|
| - ceval-valid_physician                               |      2|none  |     0|acc                    |↑  | 0.6327|±  |0.0696|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6327|±  |0.0696|
| - ceval-valid_plant_protection                        |      2|none  |     0|acc                    |↑  | 0.7727|±  |0.0914|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7727|±  |0.0914|
| - ceval-valid_probability_and_statistics              |      2|none  |     0|acc                    |↑  | 0.3333|±  |0.1143|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3333|±  |0.1143|
| - ceval-valid_professional_tour_guide                 |      2|none  |     0|acc                    |↑  | 0.5862|±  |0.0931|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5862|±  |0.0931|
| - ceval-valid_sports_science                          |      2|none  |     0|acc                    |↑  | 0.6316|±  |0.1137|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6316|±  |0.1137|
| - ceval-valid_tax_accountant                          |      2|none  |     0|acc                    |↑  | 0.5102|±  |0.0722|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5102|±  |0.0722|
| - ceval-valid_teacher_qualification                   |      2|none  |     0|acc                    |↑  | 0.7727|±  |0.0639|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7727|±  |0.0639|
| - ceval-valid_urban_and_rural_planner                 |      2|none  |     0|acc                    |↑  | 0.6304|±  |0.0720|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6304|±  |0.0720|
| - ceval-valid_veterinary_medicine                     |      2|none  |     0|acc                    |↑  | 0.6522|±  |0.1015|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6522|±  |0.1015|
|hellaswag                                              |      1|none  |     0|acc                    |↑  | 0.4740|±  |0.0050|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6386|±  |0.0048|
|ifeval                                                 |      4|none  |     0|inst_level_loose_acc   |↑  | 0.2746|±  |   N/A|
|                                                       |       |none  |     0|inst_level_strict_acc  |↑  | 0.2578|±  |   N/A|
|                                                       |       |none  |     0|prompt_level_loose_acc |↑  | 0.1608|±  |0.0158|
|                                                       |       |none  |     0|prompt_level_strict_acc|↑  | 0.1479|±  |0.0153|
|longbench_2wikimqa                                     |      5|none  |     0|qa_f1_score            |↑  | 0.1066|±  |0.0100|
|                                                       |       |none  |     0|score                  |↑  | 0.1066|±  |0.0100|
|longbench_2wikimqa_e                                   |      5|none  |     0|qa_f1_score            |↑  | 0.0989|±  |0.0088|
|                                                       |       |none  |     0|score                  |↑  | 0.0989|±  |0.0088|
|longbench_dureader                                     |      5|none  |     0|rouge_zh_score         |↑  | 0.1918|±  |0.0085|
|                                                       |       |none  |     0|score                  |↑  | 0.1918|±  |0.0085|
|longbench_gov_report                                   |      5|none  |     0|rouge_score            |↑  | 0.1367|±  |0.0041|
|                                                       |       |none  |     0|score                  |↑  | 0.1367|±  |0.0041|
|longbench_gov_report_e                                 |      5|none  |     0|rouge_score            |↑  | 0.1520|±  |0.0038|
|                                                       |       |none  |     0|score                  |↑  | 0.1520|±  |0.0038|
|longbench_hotpotqa                                     |      5|none  |     0|qa_f1_score            |↑  | 0.0759|±  |0.0100|
|                                                       |       |none  |     0|score                  |↑  | 0.0759|±  |0.0100|
|longbench_hotpotqa_e                                   |      5|none  |     0|qa_f1_score            |↑  | 0.0874|±  |0.0082|
|                                                       |       |none  |     0|score                  |↑  | 0.0874|±  |0.0082|
|longbench_lcc                                          |      5|none  |     0|code_sim_score         |↑  | 0.0887|±  |0.0066|
|                                                       |       |none  |     0|score                  |↑  | 0.0887|±  |0.0066|
|longbench_lcc_e                                        |      5|none  |     0|code_sim_score         |↑  | 0.1689|±  |0.0119|
|                                                       |       |none  |     0|score                  |↑  | 0.1689|±  |0.0119|
|longbench_lsht                                         |      5|none  |     0|classification_score   |↑  | 0.2200|±  |0.0287|
|                                                       |       |none  |     0|score                  |↑  | 0.2200|±  |0.0287|
|longbench_multi_news                                   |      5|none  |     0|rouge_score            |↑  | 0.1584|±  |0.0067|
|                                                       |       |none  |     0|score                  |↑  | 0.1584|±  |0.0067|
|longbench_multi_news_e                                 |      5|none  |     0|rouge_score            |↑  | 0.1229|±  |0.0050|
|                                                       |       |none  |     0|score                  |↑  | 0.1229|±  |0.0050|
|longbench_multifieldqa_en                              |      5|none  |     0|qa_f1_score            |↑  | 0.2518|±  |0.0213|
|                                                       |       |none  |     0|score                  |↑  | 0.2518|±  |0.0213|
|longbench_multifieldqa_en_e                            |      5|none  |     0|qa_f1_score            |↑  | 0.2518|±  |0.0213|
|                                                       |       |none  |     0|score                  |↑  | 0.2518|±  |0.0213|
|longbench_multifieldqa_zh                              |      5|none  |     0|qa_f1_zh_score         |↑  | 0.2288|±  |0.0165|
|                                                       |       |none  |     0|score                  |↑  | 0.2288|±  |0.0165|
|longbench_musique                                      |      5|none  |     0|qa_f1_score            |↑  | 0.0369|±  |0.0049|
|                                                       |       |none  |     0|score                  |↑  | 0.0369|±  |0.0049|
|longbench_narrativeqa                                  |      5|none  |     0|qa_f1_score            |↑  | 0.0462|±  |0.0059|
|                                                       |       |none  |     0|score                  |↑  | 0.0462|±  |0.0059|
|longbench_passage_count                                |      5|none  |     0|count_score            |↑  | 0.0292|±  |0.0036|
|                                                       |       |none  |     0|score                  |↑  | 0.0292|±  |0.0036|
|longbench_passage_count_e                              |      5|none  |     0|count_score            |↑  | 0.0635|±  |0.0045|
|                                                       |       |none  |     0|score                  |↑  | 0.0635|±  |0.0045|
|longbench_passage_retrieval_en                         |      5|none  |     0|retrieval_score        |↑  | 0.0375|±  |0.0127|
|                                                       |       |none  |     0|score                  |↑  | 0.0375|±  |0.0127|
|longbench_passage_retrieval_en_e                       |      5|none  |     0|retrieval_score        |↑  | 0.0779|±  |0.0147|
|                                                       |       |none  |     0|score                  |↑  | 0.0779|±  |0.0147|
|longbench_qasper                                       |      5|none  |     0|qa_f1_score            |↑  | 0.1583|±  |0.0121|
|                                                       |       |none  |     0|score                  |↑  | 0.1583|±  |0.0121|
|longbench_qasper_e                                     |      5|none  |     0|qa_f1_score            |↑  | 0.1494|±  |0.0120|
|                                                       |       |none  |     0|score                  |↑  | 0.1494|±  |0.0120|
|longbench_qmsum                                        |      5|none  |     0|rouge_score            |↑  | 0.2067|±  |0.0050|
|                                                       |       |none  |     0|score                  |↑  | 0.2067|±  |0.0050|
|longbench_repobench-p                                  |      5|none  |     0|code_sim_score         |↑  | 0.2176|±  |0.0096|
|                                                       |       |none  |     0|score                  |↑  | 0.2176|±  |0.0096|
|longbench_repobench-p_e                                |      5|none  |     0|code_sim_score         |↑  | 0.2114|±  |0.0113|
|                                                       |       |none  |     0|score                  |↑  | 0.2114|±  |0.0113|
|longbench_samsum                                       |      5|none  |     0|rouge_score            |↑  | 0.2868|±  |0.0095|
|                                                       |       |none  |     0|score                  |↑  | 0.2868|±  |0.0095|
|longbench_samsum_e                                     |      5|none  |     0|rouge_score            |↑  | 0.2768|±  |0.0076|
|                                                       |       |none  |     0|score                  |↑  | 0.2768|±  |0.0076|
|longbench_trec                                         |      5|none  |     0|classification_score   |↑  | 0.3192|±  |0.0266|
|                                                       |       |none  |     0|score                  |↑  | 0.3192|±  |0.0266|
|longbench_trec_e                                       |      5|none  |     0|classification_score   |↑  | 0.2961|±  |0.0212|
|                                                       |       |none  |     0|score                  |↑  | 0.2961|±  |0.0212|
|longbench_triviaqa                                     |      5|none  |     0|qa_f1_score            |↑  | 0.1907|±  |0.0169|
|                                                       |       |none  |     0|score                  |↑  | 0.1907|±  |0.0169|
|longbench_triviaqa_e                                   |      5|none  |     0|qa_f1_score            |↑  | 0.2275|±  |0.0153|
|                                                       |       |none  |     0|score                  |↑  | 0.2275|±  |0.0153|
|longbench_vcsum                                        |      5|none  |     0|rouge_zh_score         |↑  | 0.0730|±  |0.0042|
|                                                       |       |none  |     0|score                  |↑  | 0.0730|±  |0.0042|
|mmlu                                                   |      2|none  |      |acc                    |↑  | 0.5643|±  |0.0040|
| - humanities                                          |      2|none  |      |acc                    |↑  | 0.4593|±  |0.0069|
|  - formal_logic                                       |      1|none  |     0|acc                    |↑  | 0.4444|±  |0.0444|
|  - high_school_european_history                       |      1|none  |     0|acc                    |↑  | 0.5030|±  |0.0390|
|  - high_school_us_history                             |      1|none  |     0|acc                    |↑  | 0.4902|±  |0.0351|
|  - high_school_world_history                          |      1|none  |     0|acc                    |↑  | 0.6582|±  |0.0309|
|  - international_law                                  |      1|none  |     0|acc                    |↑  | 0.7025|±  |0.0417|
|  - jurisprudence                                      |      1|none  |     0|acc                    |↑  | 0.6759|±  |0.0452|
|  - logical_fallacies                                  |      1|none  |     0|acc                    |↑  | 0.6871|±  |0.0364|
|  - moral_disputes                                     |      1|none  |     0|acc                    |↑  | 0.6040|±  |0.0263|
|  - moral_scenarios                                    |      1|none  |     0|acc                    |↑  | 0.2637|±  |0.0147|
|  - philosophy                                         |      1|none  |     0|acc                    |↑  | 0.6141|±  |0.0276|
|  - prehistory                                         |      1|none  |     0|acc                    |↑  | 0.6049|±  |0.0272|
|  - professional_law                                   |      1|none  |     0|acc                    |↑  | 0.3507|±  |0.0122|
|  - world_religions                                    |      1|none  |     0|acc                    |↑  | 0.7368|±  |0.0338|
| - other                                               |      2|none  |      |acc                    |↑  | 0.6244|±  |0.0085|
|  - business_ethics                                    |      1|none  |     0|acc                    |↑  | 0.6400|±  |0.0482|
|  - clinical_knowledge                                 |      1|none  |     0|acc                    |↑  | 0.6792|±  |0.0287|
|  - college_medicine                                   |      1|none  |     0|acc                    |↑  | 0.6012|±  |0.0373|
|  - global_facts                                       |      1|none  |     0|acc                    |↑  | 0.3800|±  |0.0488|
|  - human_aging                                        |      1|none  |     0|acc                    |↑  | 0.6143|±  |0.0327|
|  - management                                         |      1|none  |     0|acc                    |↑  | 0.7476|±  |0.0430|
|  - marketing                                          |      1|none  |     0|acc                    |↑  | 0.8120|±  |0.0256|
|  - medical_genetics                                   |      1|none  |     0|acc                    |↑  | 0.6600|±  |0.0476|
|  - miscellaneous                                      |      1|none  |     0|acc                    |↑  | 0.6948|±  |0.0165|
|  - nutrition                                          |      1|none  |     0|acc                    |↑  | 0.6699|±  |0.0269|
|  - professional_accounting                            |      1|none  |     0|acc                    |↑  | 0.4326|±  |0.0296|
|  - professional_medicine                              |      1|none  |     0|acc                    |↑  | 0.4963|±  |0.0304|
|  - virology                                           |      1|none  |     0|acc                    |↑  | 0.4699|±  |0.0389|
| - social sciences                                     |      2|none  |      |acc                    |↑  | 0.6789|±  |0.0082|
|  - econometrics                                       |      1|none  |     0|acc                    |↑  | 0.4211|±  |0.0464|
|  - high_school_geography                              |      1|none  |     0|acc                    |↑  | 0.7727|±  |0.0299|
|  - high_school_government_and_politics                |      1|none  |     0|acc                    |↑  | 0.7358|±  |0.0318|
|  - high_school_macroeconomics                         |      1|none  |     0|acc                    |↑  | 0.6308|±  |0.0245|
|  - high_school_microeconomics                         |      1|none  |     0|acc                    |↑  | 0.6891|±  |0.0301|
|  - high_school_psychology                             |      1|none  |     0|acc                    |↑  | 0.8165|±  |0.0166|
|  - human_sexuality                                    |      1|none  |     0|acc                    |↑  | 0.6794|±  |0.0409|
|  - professional_psychology                            |      1|none  |     0|acc                    |↑  | 0.5931|±  |0.0199|
|  - public_relations                                   |      1|none  |     0|acc                    |↑  | 0.6545|±  |0.0455|
|  - security_studies                                   |      1|none  |     0|acc                    |↑  | 0.5102|±  |0.0320|
|  - sociology                                          |      1|none  |     0|acc                    |↑  | 0.7861|±  |0.0290|
|  - us_foreign_policy                                  |      1|none  |     0|acc                    |↑  | 0.8400|±  |0.0368|
| - stem                                                |      2|none  |      |acc                    |↑  | 0.5500|±  |0.0086|
|  - abstract_algebra                                   |      1|none  |     0|acc                    |↑  | 0.3500|±  |0.0479|
|  - anatomy                                            |      1|none  |     0|acc                    |↑  | 0.5926|±  |0.0424|
|  - astronomy                                          |      1|none  |     0|acc                    |↑  | 0.6579|±  |0.0386|
|  - college_biology                                    |      1|none  |     0|acc                    |↑  | 0.7014|±  |0.0383|
|  - college_chemistry                                  |      1|none  |     0|acc                    |↑  | 0.4800|±  |0.0502|
|  - college_computer_science                           |      1|none  |     0|acc                    |↑  | 0.5200|±  |0.0502|
|  - college_mathematics                                |      1|none  |     0|acc                    |↑  | 0.3500|±  |0.0479|
|  - college_physics                                    |      1|none  |     0|acc                    |↑  | 0.3725|±  |0.0481|
|  - computer_security                                  |      1|none  |     0|acc                    |↑  | 0.7700|±  |0.0423|
|  - conceptual_physics                                 |      1|none  |     0|acc                    |↑  | 0.6468|±  |0.0312|
|  - electrical_engineering                             |      1|none  |     0|acc                    |↑  | 0.6069|±  |0.0407|
|  - elementary_mathematics                             |      1|none  |     0|acc                    |↑  | 0.5132|±  |0.0257|
|  - high_school_biology                                |      1|none  |     0|acc                    |↑  | 0.7516|±  |0.0246|
|  - high_school_chemistry                              |      1|none  |     0|acc                    |↑  | 0.6010|±  |0.0345|
|  - high_school_computer_science                       |      1|none  |     0|acc                    |↑  | 0.6600|±  |0.0476|
|  - high_school_mathematics                            |      1|none  |     0|acc                    |↑  | 0.4037|±  |0.0299|
|  - high_school_physics                                |      1|none  |     0|acc                    |↑  | 0.4040|±  |0.0401|
|  - high_school_statistics                             |      1|none  |     0|acc                    |↑  | 0.4444|±  |0.0339|
|  - machine_learning                                   |      1|none  |     0|acc                    |↑  | 0.4196|±  |0.0468|
|openbookqa                                             |      1|none  |     0|acc                    |↑  | 0.2860|±  |0.0202|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3760|±  |0.0217|
|piqa                                                   |      1|none  |     0|acc                    |↑  | 0.7476|±  |0.0101|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7541|±  |0.0100|
|social_iqa                                             |      0|none  |     0|acc                    |↑  | 0.4616|±  |0.0113|
|truthfulqa_gen                                         |      3|none  |     0|bleu_acc               |↑  | 0.0747|±  |0.0092|
|                                                       |       |none  |     0|bleu_diff              |↑  |-0.6084|±  |0.1945|
|                                                       |       |none  |     0|bleu_max               |↑  | 2.4214|±  |0.3256|
|                                                       |       |none  |     0|rouge1_acc             |↑  | 0.0832|±  |0.0097|
|                                                       |       |none  |     0|rouge1_diff            |↑  |-0.9509|±  |0.2930|
|                                                       |       |none  |     0|rouge1_max             |↑  | 5.5251|±  |0.5521|
|                                                       |       |none  |     0|rouge2_acc             |↑  | 0.0575|±  |0.0082|
|                                                       |       |none  |     0|rouge2_diff            |↑  |-1.3663|±  |0.3420|
|                                                       |       |none  |     0|rouge2_max             |↑  | 3.4641|±  |0.4492|
|                                                       |       |none  |     0|rougeL_acc             |↑  | 0.0771|±  |0.0093|
|                                                       |       |none  |     0|rougeL_diff            |↑  |-1.1196|±  |0.2889|
|                                                       |       |none  |     0|rougeL_max             |↑  | 5.0788|±  |0.5197|
|truthfulqa_mc1                                         |      2|none  |     0|acc                    |↑  | 0.3060|±  |0.0161|
|truthfulqa_mc2                                         |      3|none  |     0|acc                    |↑  | 0.4686|±  |0.0146|
|winogrande                                             |      1|none  |     0|acc                    |↑  | 0.6401|±  |0.0135|
|niah_multikey_1               |      1|none  |     0|  1024|   |1.0000|±  |     0|
|                              |       |none  |     0| 16384|↑  |0.5900|±  |   N/A|
|                              |       |none  |     0|  2048|   |1.0000|±  |     0|
|                              |       |none  |     0| 32768|↑  |0.4040|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |0.8660|±  |   N/A|
|niah_multikey_2               |      1|none  |     0|  1024|   |0.9980|±  |0.0020|
|                              |       |none  |     0| 16384|↑  |0.2140|±  |   N/A|
|                              |       |none  |     0|  2048|   |0.9980|±  |0.0020|
|                              |       |none  |     0| 32768|↑  |0.1120|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |0.9860|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |0.5440|±  |   N/A|
|niah_multikey_3               |      1|none  |     0|  1024|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 16384|↑  |0.0560|±  |   N/A|
|                              |       |none  |     0|  2048|   |0.9780|±  |0.0066|
|                              |       |none  |     0| 32768|↑  |0.0080|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |0.9580|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |0.2300|±  |   N/A|
|niah_multiquery               |      1|none  |     0|  1024|   |0.9995|±  |0.0005|
|                              |       |none  |     0| 16384|↑  |0.4965|±  |   N/A|
|                              |       |none  |     0|  2048|   |0.9980|±  |0.0012|
|                              |       |none  |     0| 32768|↑  |0.3515|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |0.9800|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |0.7670|±  |   N/A|
|niah_multivalue               |      1|none  |     0|  1024|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 16384|↑  |0.4960|±  |   N/A|
|                              |       |none  |     0|  2048|   |0.9910|±  |0.0024|
|                              |       |none  |     0| 32768|↑  |0.3015|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |0.9840|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |0.7535|±  |   N/A|
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
|                              |       |none  |     0| 16384|↑  |0.9980|±  |   N/A|
|                              |       |none  |     0|  2048|   |0.9980|±  |0.0020|
|                              |       |none  |     0| 32768|↑  |0.9940|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |0.9980|±  |   N/A|
|niah_single_2_paraphrase      |      1|none  |     0|  1024|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 16384|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  2048|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 32768|↑  |0.9700|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |1.0000|±  |   N/A|
|niah_single_2_preamble        |      1|none  |     0|  1024|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 16384|↑  |0.9960|±  |   N/A|
|                              |       |none  |     0|  2048|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 32768|↑  |0.9580|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |0.9980|±  |   N/A|
|niah_single_2_qend            |      1|none  |     0|  1024|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 16384|↑  |0.9940|±  |   N/A|
|                              |       |none  |     0|  2048|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 32768|↑  |0.9900|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |0.9980|±  |   N/A|
|niah_single_2_words           |      1|none  |     0|  1024|   |0.9860|±  |0.0053|
|                              |       |none  |     0| 16384|↑  |0.9700|±  |   N/A|
|                              |       |none  |     0|  2048|   |0.9940|±  |0.0035|
|                              |       |none  |     0| 32768|↑  |0.9720|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |0.9680|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |0.9560|±  |   N/A|
|niah_single_2_words_paraphrase|      1|none  |     0|  1024|   |0.9780|±  |0.0066|
|                              |       |none  |     0| 16384|↑  |0.9880|±  |   N/A|
|                              |       |none  |     0|  2048|   |0.9860|±  |0.0053|
|                              |       |none  |     0| 32768|↑  |0.9180|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |0.9680|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |0.9760|±  |   N/A|
|niah_single_2_words_qend      |      1|none  |     0|  1024|   |0.9380|±  |0.0108|
|                              |       |none  |     0| 16384|↑  |0.9060|±  |   N/A|
|                              |       |none  |     0|  2048|   |0.9740|±  |0.0071|
|                              |       |none  |     0| 32768|↑  |0.9960|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |0.9620|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |0.9680|±  |   N/A|
|niah_single_3                 |      1|none  |     0|  1024|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 16384|↑  |0.5760|±  |   N/A|
|                              |       |none  |     0|  2048|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 32768|↑  |0.2800|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |0.9920|±  |   N/A|
|niah_single_3_paraphrase      |      1|none  |     0|  1024|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 16384|↑  |0.9680|±  |   N/A|
|                              |       |none  |     0|  2048|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 32768|↑  |0.6980|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |1.0000|±  |   N/A|
|niah_single_3_preamble        |      1|none  |     0|  1024|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 16384|↑  |0.4920|±  |   N/A|
|                              |       |none  |     0|  2048|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 32768|↑  |0.3620|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |1.0000|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |0.9880|±  |   N/A|
|niah_single_3_qend            |      1|none  |     0|  1024|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 16384|↑  |0.9480|±  |   N/A|
|                              |       |none  |     0|  2048|   |1.0000|±  |0.0000|
|                              |       |none  |     0| 32768|↑  |0.6560|±  |   N/A|
|                              |       |none  |     0|  4096|↑  |0.9980|±  |   N/A|
|                              |       |none  |     0|  8192|↑  |0.9980|±  |   N/A|
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
# InfiniteBench 的十个任务（eval_tasks/infinitebench，只测 64K 起），AVG_InfiniteBench 为十个的平均（论文的 Avg）。
INFINITEBENCH = ["passkey", "number_string", "kv_retrieval", "longbook_sum_eng", "longbook_qa_eng", "longbook_choice_eng",
                 "longbook_qa_chn", "longdialogue_qa_eng", "code_debug", "math_find"]
INFINITEBENCH_LENGTHS = ("64k", "128k", "256k")
# 每个长度实际跑的 RULER 任务：256K 没有 ruler_qa_squad（SQuAD 的文档池填不满 256K）。
RULER_AT = {n: [t for t in LONG_RULER if n != "256k" or t != "ruler_qa_squad"] for n in BLOCK_LENGTHS}


# 长文本列先按长度分块（32K 主线之后是 32K、64K、128K、256K 各一块，论文里 32K-256K 可以单独成表），块内再按任务分组、
# 每组的平均紧跟在它平均的任务之后。只列评测会产出结果的列，各块的顺序相同：
#   RULER 单针 niah_single_1-3、AVG_NIAH；多针 niah_multikey_1-3、niah_multivalue、niah_multiquery、AVG_NIAH_multi；
#   其余 ruler_vt、ruler_cwe、ruler_fwe、ruler_qa_squad（256K 没有）、ruler_qa_hotpot、AVG_RULER（该长度全部 RULER 任务的
#   平均：13 个，256K 为 12 个）；BABILong qa1-qa5、AVG_babilong；LongBench v2（longbench2_<长度>）；
#   同一份缓存上逐轮问 niah_turns_<长度>_turn1-4、niah_turns_<长度>（四轮平均）；
#   书的困惑度曲线 long_ppl_books_<长度>_<截断长度>（截断长度 8k 起到该长度，同一批书，越低越好），之后是同长度的
#   long_ppl_<长度>（PG19 测试集文档末尾的困惑度，越低越好）；
#   InfiniteBench 十个任务 infinitebench_<任务>_<长度>（官方评分）、AVG_InfiniteBench_<长度>（只有 64K 起的块有）。
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
    columns += [f"LongBench2_{n}"]
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

    for _, row in df.iterrows():
        raw_task = re.sub(r"^-\s*", "", str(row['Tasks']))  # a group's subtasks are listed as " - name"
        metric = row['Metric']
        value = row['Value']

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
