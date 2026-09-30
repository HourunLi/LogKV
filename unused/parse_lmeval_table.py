import pandas as pd
import numpy as np
import io

# 把你的各个模型的原始输出放在这里。这里以 Model_A 为例（填入你提供的原数据）
model_outputs = {
    "base-prolong": """
|                         Tasks                         |Version|Filter|n-shot|        Metric         |   | Value |   |Stderr|
|-------------------------------------------------------|------:|------|-----:|-----------------------|---|------:|---|------|
|arc_challenge                                          |      1|none  |     0|acc                    |↑  | 0.4198|±  |0.0144|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4531|±  |0.0145|
|arc_easy                                               |      1|none  |     0|acc                    |↑  | 0.7605|±  |0.0088|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7327|±  |0.0091|
|boolq                                                  |      2|none  |     0|acc                    |↑  | 0.7853|±  |0.0072|
|ceval-valid                                            |      2|none  |      |acc                    |↑  | 0.6441|±  |0.0126|
|                                                       |       |none  |      |acc_norm               |↑  | 0.6441|±  |0.0126|
| - ceval-valid_accountant                              |      2|none  |     0|acc                    |↑  | 0.5306|±  |0.0720|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5306|±  |0.0720|
| - ceval-valid_advanced_mathematics                    |      2|none  |     0|acc                    |↑  | 0.3684|±  |0.1137|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3684|±  |0.1137|
| - ceval-valid_art_studies                             |      2|none  |     0|acc                    |↑  | 0.5758|±  |0.0874|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5758|±  |0.0874|
| - ceval-valid_basic_medicine                          |      2|none  |     0|acc                    |↑  | 0.6842|±  |0.1096|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6842|±  |0.1096|
| - ceval-valid_business_administration                 |      2|none  |     0|acc                    |↑  | 0.6364|±  |0.0850|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6364|±  |0.0850|
| - ceval-valid_chinese_language_and_literature         |      2|none  |     0|acc                    |↑  | 0.6087|±  |0.1041|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6087|±  |0.1041|
| - ceval-valid_civil_servant                           |      2|none  |     0|acc                    |↑  | 0.5106|±  |0.0737|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5106|±  |0.0737|
| - ceval-valid_clinical_medicine                       |      2|none  |     0|acc                    |↑  | 0.5909|±  |0.1073|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5909|±  |0.1073|
| - ceval-valid_college_chemistry                       |      2|none  |     0|acc                    |↑  | 0.5417|±  |0.1039|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5417|±  |0.1039|
| - ceval-valid_college_economics                       |      2|none  |     0|acc                    |↑  | 0.5273|±  |0.0679|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5273|±  |0.0679|
| - ceval-valid_college_physics                         |      2|none  |     0|acc                    |↑  | 0.5263|±  |0.1177|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5263|±  |0.1177|
| - ceval-valid_college_programming                     |      2|none  |     0|acc                    |↑  | 0.7027|±  |0.0762|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7027|±  |0.0762|
| - ceval-valid_computer_architecture                   |      2|none  |     0|acc                    |↑  | 0.8095|±  |0.0878|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8095|±  |0.0878|
| - ceval-valid_computer_network                        |      2|none  |     0|acc                    |↑  | 0.5263|±  |0.1177|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5263|±  |0.1177|
| - ceval-valid_discrete_mathematics                    |      2|none  |     0|acc                    |↑  | 0.1875|±  |0.1008|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.1875|±  |0.1008|
| - ceval-valid_education_science                       |      2|none  |     0|acc                    |↑  | 0.7586|±  |0.0809|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7586|±  |0.0809|
| - ceval-valid_electrical_engineer                     |      2|none  |     0|acc                    |↑  | 0.4054|±  |0.0818|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4054|±  |0.0818|
| - ceval-valid_environmental_impact_assessment_engineer|      2|none  |     0|acc                    |↑  | 0.6774|±  |0.0853|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6774|±  |0.0853|
| - ceval-valid_fire_engineer                           |      2|none  |     0|acc                    |↑  | 0.5484|±  |0.0909|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5484|±  |0.0909|
| - ceval-valid_high_school_biology                     |      2|none  |     0|acc                    |↑  | 0.7368|±  |0.1038|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7368|±  |0.1038|
| - ceval-valid_high_school_chemistry                   |      2|none  |     0|acc                    |↑  | 0.4737|±  |0.1177|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4737|±  |0.1177|
| - ceval-valid_high_school_chinese                     |      2|none  |     0|acc                    |↑  | 0.5789|±  |0.1164|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5789|±  |0.1164|
| - ceval-valid_high_school_geography                   |      2|none  |     0|acc                    |↑  | 0.7368|±  |0.1038|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7368|±  |0.1038|
| - ceval-valid_high_school_history                     |      2|none  |     0|acc                    |↑  | 0.7500|±  |0.0993|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7500|±  |0.0993|
| - ceval-valid_high_school_mathematics                 |      2|none  |     0|acc                    |↑  | 0.4444|±  |0.1205|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4444|±  |0.1205|
| - ceval-valid_high_school_physics                     |      2|none  |     0|acc                    |↑  | 0.7895|±  |0.0961|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7895|±  |0.0961|
| - ceval-valid_high_school_politics                    |      2|none  |     0|acc                    |↑  | 0.8947|±  |0.0723|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8947|±  |0.0723|
| - ceval-valid_ideological_and_moral_cultivation       |      2|none  |     0|acc                    |↑  | 0.9474|±  |0.0526|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.9474|±  |0.0526|
| - ceval-valid_law                                     |      2|none  |     0|acc                    |↑  | 0.4583|±  |0.1039|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4583|±  |0.1039|
| - ceval-valid_legal_professional                      |      2|none  |     0|acc                    |↑  | 0.4348|±  |0.1057|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4348|±  |0.1057|
| - ceval-valid_logic                                   |      2|none  |     0|acc                    |↑  | 0.5909|±  |0.1073|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5909|±  |0.1073|
| - ceval-valid_mao_zedong_thought                      |      2|none  |     0|acc                    |↑  | 0.8333|±  |0.0777|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8333|±  |0.0777|
| - ceval-valid_marxism                                 |      2|none  |     0|acc                    |↑  | 0.8947|±  |0.0723|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8947|±  |0.0723|
| - ceval-valid_metrology_engineer                      |      2|none  |     0|acc                    |↑  | 0.7083|±  |0.0948|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7083|±  |0.0948|
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
| - ceval-valid_middle_school_physics                   |      2|none  |     0|acc                    |↑  | 0.8421|±  |0.0859|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8421|±  |0.0859|
| - ceval-valid_middle_school_politics                  |      2|none  |     0|acc                    |↑  | 0.8095|±  |0.0878|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8095|±  |0.0878|
| - ceval-valid_modern_chinese_history                  |      2|none  |     0|acc                    |↑  | 0.7826|±  |0.0879|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7826|±  |0.0879|
| - ceval-valid_operating_system                        |      2|none  |     0|acc                    |↑  | 0.6842|±  |0.1096|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6842|±  |0.1096|
| - ceval-valid_physician                               |      2|none  |     0|acc                    |↑  | 0.6327|±  |0.0696|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6327|±  |0.0696|
| - ceval-valid_plant_protection                        |      2|none  |     0|acc                    |↑  | 0.8182|±  |0.0842|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8182|±  |0.0842|
| - ceval-valid_probability_and_statistics              |      2|none  |     0|acc                    |↑  | 0.3889|±  |0.1182|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3889|±  |0.1182|
| - ceval-valid_professional_tour_guide                 |      2|none  |     0|acc                    |↑  | 0.5862|±  |0.0931|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5862|±  |0.0931|
| - ceval-valid_sports_science                          |      2|none  |     0|acc                    |↑  | 0.6842|±  |0.1096|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6842|±  |0.1096|
| - ceval-valid_tax_accountant                          |      2|none  |     0|acc                    |↑  | 0.5714|±  |0.0714|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5714|±  |0.0714|
| - ceval-valid_teacher_qualification                   |      2|none  |     0|acc                    |↑  | 0.8409|±  |0.0558|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8409|±  |0.0558|
| - ceval-valid_urban_and_rural_planner                 |      2|none  |     0|acc                    |↑  | 0.6304|±  |0.0720|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6304|±  |0.0720|
| - ceval-valid_veterinary_medicine                     |      2|none  |     0|acc                    |↑  | 0.7391|±  |0.0936|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7391|±  |0.0936|
|hellaswag                                              |      1|none  |     0|acc                    |↑  | 0.4882|±  |0.0050|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6588|±  |0.0047|
|ifeval                                                 |      4|none  |     0|inst_level_loose_acc   |↑  | 0.3165|±  |   N/A|
|                                                       |       |none  |     0|inst_level_strict_acc  |↑  | 0.2914|±  |   N/A|
|                                                       |       |none  |     0|prompt_level_loose_acc |↑  | 0.2163|±  |0.0177|
|                                                       |       |none  |     0|prompt_level_strict_acc|↑  | 0.1922|±  |0.0170|
|longbench_2wikimqa                                     |      5|none  |     0|qa_f1_score            |↑  | 0.0000|±  |     0|
|                                                       |       |none  |     0|score                  |↑  | 0.0000|±  |     0|
|longbench_2wikimqa_e                                   |      5|none  |     0|qa_f1_score            |↑  | 0.0000|±  |     0|
|                                                       |       |none  |     0|score                  |↑  | 0.0000|±  |     0|
|longbench_dureader                                     |      5|none  |     0|rouge_zh_score         |↑  | 0.0068|±  |0.0007|
|                                                       |       |none  |     0|score                  |↑  | 0.0068|±  |0.0007|
|longbench_gov_report                                   |      5|none  |     0|rouge_score            |↑  | 0.0009|±  |0.0002|
|                                                       |       |none  |     0|score                  |↑  | 0.0009|±  |0.0002|
|longbench_gov_report_e                                 |      5|none  |     0|rouge_score            |↑  | 0.0012|±  |0.0002|
|                                                       |       |none  |     0|score                  |↑  | 0.0012|±  |0.0002|
|longbench_hotpotqa                                     |      5|none  |     0|qa_f1_score            |↑  | 0.0000|±  |     0|
|                                                       |       |none  |     0|score                  |↑  | 0.0000|±  |     0|
|longbench_hotpotqa_e                                   |      5|none  |     0|qa_f1_score            |↑  | 0.0055|±  |0.0032|
|                                                       |       |none  |     0|score                  |↑  | 0.0055|±  |0.0032|
|longbench_lcc                                          |      5|none  |     0|code_sim_score         |↑  | 0.1089|±  |0.0032|
|                                                       |       |none  |     0|score                  |↑  | 0.1089|±  |0.0032|
|longbench_lcc_e                                        |      5|none  |     0|code_sim_score         |↑  | 0.0882|±  |0.0033|
|                                                       |       |none  |     0|score                  |↑  | 0.0882|±  |0.0033|
|longbench_lsht                                         |      5|none  |     0|classification_score   |↑  | 0.0050|±  |0.0050|
|                                                       |       |none  |     0|score                  |↑  | 0.0050|±  |0.0050|
|longbench_multi_news                                   |      5|none  |     0|rouge_score            |↑  | 0.0324|±  |0.0061|
|                                                       |       |none  |     0|score                  |↑  | 0.0324|±  |0.0061|
|longbench_multi_news_e                                 |      5|none  |     0|rouge_score            |↑  | 0.0180|±  |0.0040|
|                                                       |       |none  |     0|score                  |↑  | 0.0180|±  |0.0040|
|longbench_multifieldqa_en                              |      5|none  |     0|qa_f1_score            |↑  | 0.0006|±  |0.0006|
|                                                       |       |none  |     0|score                  |↑  | 0.0006|±  |0.0006|
|longbench_multifieldqa_en_e                            |      5|none  |     0|qa_f1_score            |↑  | 0.0006|±  |0.0006|
|                                                       |       |none  |     0|score                  |↑  | 0.0006|±  |0.0006|
|longbench_multifieldqa_zh                              |      5|none  |     0|qa_f1_zh_score         |↑  | 0.0072|±  |0.0052|
|                                                       |       |none  |     0|score                  |↑  | 0.0072|±  |0.0052|
|longbench_musique                                      |      5|none  |     0|qa_f1_score            |↑  | 0.0000|±  |     0|
|                                                       |       |none  |     0|score                  |↑  | 0.0000|±  |     0|
|longbench_narrativeqa                                  |      5|none  |     0|qa_f1_score            |↑  | 0.0000|±  |     0|
|                                                       |       |none  |     0|score                  |↑  | 0.0000|±  |     0|
|longbench_passage_count                                |      5|none  |     0|count_score            |↑  | 0.0000|±  |     0|
|                                                       |       |none  |     0|score                  |↑  | 0.0000|±  |     0|
|longbench_passage_count_e                              |      5|none  |     0|count_score            |↑  | 0.0000|±  |     0|
|                                                       |       |none  |     0|score                  |↑  | 0.0000|±  |     0|
|longbench_passage_retrieval_en                         |      5|none  |     0|retrieval_score        |↑  | 0.0000|±  |     0|
|                                                       |       |none  |     0|score                  |↑  | 0.0000|±  |     0|
|longbench_passage_retrieval_en_e                       |      5|none  |     0|retrieval_score        |↑  | 0.0100|±  |0.0058|
|                                                       |       |none  |     0|score                  |↑  | 0.0100|±  |0.0058|
|longbench_qasper                                       |      5|none  |     0|qa_f1_score            |↑  | 0.0020|±  |0.0013|
|                                                       |       |none  |     0|score                  |↑  | 0.0020|±  |0.0013|
|longbench_qasper_e                                     |      5|none  |     0|qa_f1_score            |↑  | 0.0005|±  |0.0003|
|                                                       |       |none  |     0|score                  |↑  | 0.0005|±  |0.0003|
|longbench_qmsum                                        |      5|none  |     0|rouge_score            |↑  | 0.0002|±  |0.0001|
|                                                       |       |none  |     0|score                  |↑  | 0.0002|±  |0.0001|
|longbench_repobench-p                                  |      5|none  |     0|code_sim_score         |↑  | 0.1022|±  |0.0025|
|                                                       |       |none  |     0|score                  |↑  | 0.1022|±  |0.0025|
|longbench_repobench-p_e                                |      5|none  |     0|code_sim_score         |↑  | 0.0986|±  |0.0030|
|                                                       |       |none  |     0|score                  |↑  | 0.0986|±  |0.0030|
|longbench_samsum                                       |      5|none  |     0|rouge_score            |↑  | 0.0004|±  |0.0003|
|                                                       |       |none  |     0|score                  |↑  | 0.0004|±  |0.0003|
|longbench_samsum_e                                     |      5|none  |     0|rouge_score            |↑  | 0.0002|±  |0.0002|
|                                                       |       |none  |     0|score                  |↑  | 0.0002|±  |0.0002|
|longbench_trec                                         |      5|none  |     0|classification_score   |↑  | 0.0000|±  |     0|
|                                                       |       |none  |     0|score                  |↑  | 0.0000|±  |     0|
|longbench_trec_e                                       |      5|none  |     0|classification_score   |↑  | 0.0000|±  |     0|
|                                                       |       |none  |     0|score                  |↑  | 0.0000|±  |     0|
|longbench_triviaqa                                     |      5|none  |     0|qa_f1_score            |↑  | 0.0000|±  |     0|
|                                                       |       |none  |     0|score                  |↑  | 0.0000|±  |     0|
|longbench_triviaqa_e                                   |      5|none  |     0|qa_f1_score            |↑  | 0.0000|±  |     0|
|                                                       |       |none  |     0|score                  |↑  | 0.0000|±  |     0|
|longbench_vcsum                                        |      5|none  |     0|rouge_zh_score         |↑  | 0.0092|±  |0.0008|
|                                                       |       |none  |     0|score                  |↑  | 0.0092|±  |0.0008|
|mmlu                                                   |      2|none  |      |acc                    |↑  | 0.6072|±  |0.0039|
| - humanities                                          |      2|none  |      |acc                    |↑  | 0.5243|±  |0.0068|
|  - formal_logic                                       |      1|none  |     0|acc                    |↑  | 0.5159|±  |0.0447|
|  - high_school_european_history                       |      1|none  |     0|acc                    |↑  | 0.7273|±  |0.0348|
|  - high_school_us_history                             |      1|none  |     0|acc                    |↑  | 0.7402|±  |0.0308|
|  - high_school_world_history                          |      1|none  |     0|acc                    |↑  | 0.7679|±  |0.0275|
|  - international_law                                  |      1|none  |     0|acc                    |↑  | 0.7686|±  |0.0385|
|  - jurisprudence                                      |      1|none  |     0|acc                    |↑  | 0.7315|±  |0.0428|
|  - logical_fallacies                                  |      1|none  |     0|acc                    |↑  | 0.7301|±  |0.0349|
|  - moral_disputes                                     |      1|none  |     0|acc                    |↑  | 0.6879|±  |0.0249|
|  - moral_scenarios                                    |      1|none  |     0|acc                    |↑  | 0.2648|±  |0.0148|
|  - philosophy                                         |      1|none  |     0|acc                    |↑  | 0.6559|±  |0.0270|
|  - prehistory                                         |      1|none  |     0|acc                    |↑  | 0.6636|±  |0.0263|
|  - professional_law                                   |      1|none  |     0|acc                    |↑  | 0.4113|±  |0.0126|
|  - world_religions                                    |      1|none  |     0|acc                    |↑  | 0.7778|±  |0.0319|
| - other                                               |      2|none  |      |acc                    |↑  | 0.6540|±  |0.0083|
|  - business_ethics                                    |      1|none  |     0|acc                    |↑  | 0.6300|±  |0.0485|
|  - clinical_knowledge                                 |      1|none  |     0|acc                    |↑  | 0.6868|±  |0.0285|
|  - college_medicine                                   |      1|none  |     0|acc                    |↑  | 0.6127|±  |0.0371|
|  - global_facts                                       |      1|none  |     0|acc                    |↑  | 0.3500|±  |0.0479|
|  - human_aging                                        |      1|none  |     0|acc                    |↑  | 0.6188|±  |0.0326|
|  - management                                         |      1|none  |     0|acc                    |↑  | 0.7767|±  |0.0412|
|  - marketing                                          |      1|none  |     0|acc                    |↑  | 0.8504|±  |0.0234|
|  - medical_genetics                                   |      1|none  |     0|acc                    |↑  | 0.7000|±  |0.0461|
|  - miscellaneous                                      |      1|none  |     0|acc                    |↑  | 0.7190|±  |0.0161|
|  - nutrition                                          |      1|none  |     0|acc                    |↑  | 0.6928|±  |0.0264|
|  - professional_accounting                            |      1|none  |     0|acc                    |↑  | 0.4752|±  |0.0298|
|  - professional_medicine                              |      1|none  |     0|acc                    |↑  | 0.6287|±  |0.0293|
|  - virology                                           |      1|none  |     0|acc                    |↑  | 0.4759|±  |0.0389|
| - social sciences                                     |      2|none  |      |acc                    |↑  | 0.7163|±  |0.0080|
|  - econometrics                                       |      1|none  |     0|acc                    |↑  | 0.5263|±  |0.0470|
|  - high_school_geography                              |      1|none  |     0|acc                    |↑  | 0.7879|±  |0.0291|
|  - high_school_government_and_politics                |      1|none  |     0|acc                    |↑  | 0.8031|±  |0.0287|
|  - high_school_macroeconomics                         |      1|none  |     0|acc                    |↑  | 0.6333|±  |0.0244|
|  - high_school_microeconomics                         |      1|none  |     0|acc                    |↑  | 0.7437|±  |0.0284|
|  - high_school_psychology                             |      1|none  |     0|acc                    |↑  | 0.8404|±  |0.0157|
|  - human_sexuality                                    |      1|none  |     0|acc                    |↑  | 0.7176|±  |0.0395|
|  - professional_psychology                            |      1|none  |     0|acc                    |↑  | 0.6225|±  |0.0196|
|  - public_relations                                   |      1|none  |     0|acc                    |↑  | 0.5909|±  |0.0471|
|  - security_studies                                   |      1|none  |     0|acc                    |↑  | 0.6939|±  |0.0295|
|  - sociology                                          |      1|none  |     0|acc                    |↑  | 0.7960|±  |0.0285|
|  - us_foreign_policy                                  |      1|none  |     0|acc                    |↑  | 0.8100|±  |0.0394|
| - stem                                                |      2|none  |      |acc                    |↑  | 0.5785|±  |0.0085|
|  - abstract_algebra                                   |      1|none  |     0|acc                    |↑  | 0.3600|±  |0.0482|
|  - anatomy                                            |      1|none  |     0|acc                    |↑  | 0.6074|±  |0.0422|
|  - astronomy                                          |      1|none  |     0|acc                    |↑  | 0.7303|±  |0.0361|
|  - college_biology                                    |      1|none  |     0|acc                    |↑  | 0.7708|±  |0.0351|
|  - college_chemistry                                  |      1|none  |     0|acc                    |↑  | 0.4700|±  |0.0502|
|  - college_computer_science                           |      1|none  |     0|acc                    |↑  | 0.5800|±  |0.0496|
|  - college_mathematics                                |      1|none  |     0|acc                    |↑  | 0.3900|±  |0.0490|
|  - college_physics                                    |      1|none  |     0|acc                    |↑  | 0.4510|±  |0.0495|
|  - computer_security                                  |      1|none  |     0|acc                    |↑  | 0.7800|±  |0.0416|
|  - conceptual_physics                                 |      1|none  |     0|acc                    |↑  | 0.6553|±  |0.0311|
|  - electrical_engineering                             |      1|none  |     0|acc                    |↑  | 0.6138|±  |0.0406|
|  - elementary_mathematics                             |      1|none  |     0|acc                    |↑  | 0.5397|±  |0.0257|
|  - high_school_biology                                |      1|none  |     0|acc                    |↑  | 0.7935|±  |0.0230|
|  - high_school_chemistry                              |      1|none  |     0|acc                    |↑  | 0.6158|±  |0.0342|
|  - high_school_computer_science                       |      1|none  |     0|acc                    |↑  | 0.6600|±  |0.0476|
|  - high_school_mathematics                            |      1|none  |     0|acc                    |↑  | 0.3519|±  |0.0291|
|  - high_school_physics                                |      1|none  |     0|acc                    |↑  | 0.4636|±  |0.0407|
|  - high_school_statistics                             |      1|none  |     0|acc                    |↑  | 0.5324|±  |0.0340|
|  - machine_learning                                   |      1|none  |     0|acc                    |↑  | 0.4643|±  |0.0473|
|niah_single_1                                          |      1|none  |     0|1024                   |   | 1.0000|±  |     0|
|                                                       |       |none  |     0|16384                  |↑  | 0.0000|±  |   N/A|
|                                                       |       |none  |     0|2048                   |   | 0.0000|±  |     0|
|                                                       |       |none  |     0|32768                  |↑  | 0.0000|±  |   N/A|
|                                                       |       |none  |     0|4096                   |↑  | 0.0000|±  |   N/A|
|                                                       |       |none  |     0|8192                   |↑  | 0.0000|±  |   N/A|
|niah_single_2                                          |      1|none  |     0|1024                   |   | 1.0000|±  |     0|
|                                                       |       |none  |     0|16384                  |↑  | 0.0000|±  |   N/A|
|                                                       |       |none  |     0|2048                   |   | 0.0000|±  |     0|
|                                                       |       |none  |     0|32768                  |↑  | 0.0000|±  |   N/A|
|                                                       |       |none  |     0|4096                   |↑  | 0.0000|±  |   N/A|
|                                                       |       |none  |     0|8192                   |↑  | 0.0000|±  |   N/A|
|niah_single_3                                          |      1|none  |     0|1024                   |   | 0.9980|±  |0.0020|
|                                                       |       |none  |     0|16384                  |↑  | 0.0000|±  |   N/A|
|                                                       |       |none  |     0|2048                   |   | 0.0000|±  |     0|
|                                                       |       |none  |     0|32768                  |↑  | 0.0000|±  |   N/A|
|                                                       |       |none  |     0|4096                   |↑  | 0.0000|±  |   N/A|
|                                                       |       |none  |     0|8192                   |↑  | 0.0000|±  |   N/A|
|openbookqa                                             |      1|none  |     0|acc                    |↑  | 0.3040|±  |0.0206|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3920|±  |0.0219|
|piqa                                                   |      1|none  |     0|acc                    |↑  | 0.7573|±  |0.0100|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7633|±  |0.0099|
|social_iqa                                             |      0|none  |     0|acc                    |↑  | 0.4800|±  |0.0113|
|truthfulqa_gen                                         |      3|none  |     0|bleu_acc               |↑  | 0.3721|±  |0.0169|
|                                                       |       |none  |     0|bleu_diff              |↑  |-0.1206|±  |0.0441|
|                                                       |       |none  |     0|bleu_max               |↑  | 1.2901|±  |0.0566|
|                                                       |       |none  |     0|rouge1_acc             |↑  | 0.3868|±  |0.0170|
|                                                       |       |none  |     0|rouge1_diff            |↑  |-0.1262|±  |0.0844|
|                                                       |       |none  |     0|rouge1_max             |↑  | 5.1008|±  |0.1531|
|                                                       |       |none  |     0|rouge2_acc             |↑  | 0.3403|±  |0.0166|
|                                                       |       |none  |     0|rouge2_diff            |↑  |-0.2060|±  |0.0885|
|                                                       |       |none  |     0|rouge2_max             |↑  | 3.1218|±  |0.1238|
|                                                       |       |none  |     0|rougeL_acc             |↑  | 0.3758|±  |0.0170|
|                                                       |       |none  |     0|rougeL_diff            |↑  |-0.1483|±  |0.0832|
|                                                       |       |none  |     0|rougeL_max             |↑  | 4.8016|±  |0.1450|
|truthfulqa_mc1                                         |      2|none  |     0|acc                    |↑  | 0.3121|±  |0.0162|
|truthfulqa_mc2                                         |      3|none  |     0|acc                    |↑  | 0.4732|±  |0.0147|
|winogrande                                             |      1|none  |     0|acc                    |↑  | 0.6464|±  |0.0134|
|niah_single_1|      1|none  |     0|  1024|   |1.000|±  |     0|
|             |       |none  |     0| 16384|↑  |0.000|±  |   N/A|
|             |       |none  |     0|  2048|   |0.000|±  |     0|
|             |       |none  |     0| 32768|↑  |0.000|±  |   N/A|
|             |       |none  |     0|  4096|↑  |0.000|±  |   N/A|
|             |       |none  |     0|  8192|↑  |0.000|±  |   N/A|
|niah_single_2|      1|none  |     0|  1024|   |1.000|±  |0.0000|
|             |       |none  |     0| 16384|↑  |0.000|±  |   N/A|
|             |       |none  |     0|  2048|   |0.000|±  |0.0000|
|             |       |none  |     0| 32768|↑  |0.000|±  |   N/A|
|             |       |none  |     0|  4096|↑  |0.000|±  |   N/A|
|             |       |none  |     0|  8192|↑  |0.000|±  |   N/A|
|niah_single_3|      1|none  |     0|  1024|   |0.998|±  |0.0020|
|             |       |none  |     0| 16384|↑  |0.000|±  |   N/A|
|             |       |none  |     0|  2048|   |0.000|±  |0.0000|
|             |       |none  |     0| 32768|↑  |0.000|±  |   N/A|
|             |       |none  |     0|  4096|↑  |0.000|±  |   N/A|
|             |       |none  |     0|  8192|↑  |0.000|±  |   N/A|

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
    "niah_single_1", "niah_single_2", "niah_single_3", "AVG_NIAH"
]

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
    "longbench_trec_e": "trec_e", "longbench_triviaqa_e": "triviaqa_e"
}

TARGET_METRIC_MAP = {
    "BoolQ": "acc", "PIQA": "acc", "SIQA": "acc", "HellaSwag": "acc_norm",
    "WinoGrande": "acc", "ARC-e": "acc", "ARC-c": "acc_norm", "OpenBookQA": "acc_norm",
    "ceval-valid": "acc", "mmlu": "acc", "ifeval": "prompt_level_strict_acc",
    "truthfulqa_gen": "rouge1_max", "truthfulqa_mc1": "acc", "truthfulqa_mc2": "acc",
    "SWDE": "f1", "FDA": "f1", "TriviaQA": "exact_match", "NQ": "exact_match", "DROP": "f1", "SQUAD": "exact_match",
    "niah_single_1": "32768", "niah_single_2": "32768", "niah_single_3": "32768" 
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
    "AVG_NIAH": ["niah_single_1", "niah_single_2", "niah_single_3"]
}

# ==============================================================================
# 3. 解析与计算核心代码
# ==============================================================================
def parse_single_model(model_name, markdown_text):
    lines = [line for line in markdown_text.strip().split('\n') if not line.strip().startswith('|---')]
    if not lines: return {"Model": model_name}
    
    df = pd.read_csv(io.StringIO('\n'.join(lines)), sep='|')
    df = df.iloc[:, 1:-1]
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if df[col].dtype == 'object': df[col] = df[col].str.strip()
    
    df['Tasks'] = df['Tasks'].replace('', np.nan).ffill()
    row_data = {"Model": model_name}
    
    for _, row in df.iterrows():
        raw_task = row['Tasks']
        metric = row['Metric']
        value = row['Value']
        
        target_task = RAW_TO_TARGET_MAP.get(raw_task, raw_task)
        expected_metric = TARGET_METRIC_MAP.get(target_task)
        
        if expected_metric and metric == expected_metric:
            try:
                row_data[target_task] = float(value)
            except ValueError:
                pass # 忽略非数字
    return row_data

# 1. 组合数据为 DataFrame
final_df = pd.DataFrame([parse_single_model(m, text) for m, text in model_outputs.items()])
final_df = final_df.reindex(columns=['Model'] + ORDERED_COLUMNS) # 对齐格式

# 2. ================= 自动计算各组平均值 =================
for avg_col, sub_tasks in AVG_GROUPS.items():
    # 选出有实际成绩的列，利用 pd.to_numeric 把可能混入的字符转为 NaN 并求均值
    valid_cols = [t for t in sub_tasks if t in final_df.columns]
    final_df[avg_col] = final_df[valid_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1).round(4)

# 计算全局汇总平均 AVG_all (综合所有分组子任务的基础列)
all_base_tasks = list(set([task for group_tasks in AVG_GROUPS.values() for task in group_tasks]))
valid_base_tasks = [t for t in all_base_tasks if t in final_df.columns]
final_df['AVG_all'] = final_df[valid_base_tasks].apply(pd.to_numeric, errors='coerce').mean(axis=1).round(4)

# 3. 清理空白，输出剪贴板友好格式
final_df = final_df.fillna("")
print(final_df.to_csv(sep=',', index=False))