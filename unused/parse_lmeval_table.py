import pandas as pd
import numpy as np
import io

# 把你的各个模型的原始输出放在这里。这里以 Model_A 为例（填入你提供的原数据）
model_outputs = {
    "base-prolong": """
|                         Tasks                         |Version|Filter|n-shot|        Metric         |   | Value |   |Stderr|
|-------------------------------------------------------|------:|------|-----:|-----------------------|---|------:|---|------|
|arc_challenge                                          |      1|none  |     0|acc                    |↑  | 0.4411|±  |0.0145|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4710|±  |0.0146|
|arc_easy                                               |      1|none  |     0|acc                    |↑  | 0.7740|±  |0.0086|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7597|±  |0.0088|
|boolq                                                  |      2|none  |     0|acc                    |↑  | 0.7532|±  |0.0075|
|ceval-valid                                            |      2|none  |      |acc                    |↑  | 0.6404|±  |0.0126|
|                                                       |       |none  |      |acc_norm               |↑  | 0.6404|±  |0.0126|
| - ceval-valid_accountant                              |      2|none  |     0|acc                    |↑  | 0.5510|±  |0.0718|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5510|±  |0.0718|
| - ceval-valid_advanced_mathematics                    |      2|none  |     0|acc                    |↑  | 0.2105|±  |0.0961|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.2105|±  |0.0961|
| - ceval-valid_art_studies                             |      2|none  |     0|acc                    |↑  | 0.5455|±  |0.0880|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5455|±  |0.0880|
| - ceval-valid_basic_medicine                          |      2|none  |     0|acc                    |↑  | 0.6316|±  |0.1137|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6316|±  |0.1137|
| - ceval-valid_business_administration                 |      2|none  |     0|acc                    |↑  | 0.6364|±  |0.0850|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6364|±  |0.0850|
| - ceval-valid_chinese_language_and_literature         |      2|none  |     0|acc                    |↑  | 0.4783|±  |0.1065|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4783|±  |0.1065|
| - ceval-valid_civil_servant                           |      2|none  |     0|acc                    |↑  | 0.5106|±  |0.0737|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5106|±  |0.0737|
| - ceval-valid_clinical_medicine                       |      2|none  |     0|acc                    |↑  | 0.6364|±  |0.1050|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6364|±  |0.1050|
| - ceval-valid_college_chemistry                       |      2|none  |     0|acc                    |↑  | 0.5833|±  |0.1028|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5833|±  |0.1028|
| - ceval-valid_college_economics                       |      2|none  |     0|acc                    |↑  | 0.5636|±  |0.0675|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5636|±  |0.0675|
| - ceval-valid_college_physics                         |      2|none  |     0|acc                    |↑  | 0.4211|±  |0.1164|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4211|±  |0.1164|
| - ceval-valid_college_programming                     |      2|none  |     0|acc                    |↑  | 0.6757|±  |0.0780|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6757|±  |0.0780|
| - ceval-valid_computer_architecture                   |      2|none  |     0|acc                    |↑  | 0.7619|±  |0.0952|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7619|±  |0.0952|
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
| - ceval-valid_fire_engineer                           |      2|none  |     0|acc                    |↑  | 0.5161|±  |0.0912|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5161|±  |0.0912|
| - ceval-valid_high_school_biology                     |      2|none  |     0|acc                    |↑  | 0.6842|±  |0.1096|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6842|±  |0.1096|
| - ceval-valid_high_school_chemistry                   |      2|none  |     0|acc                    |↑  | 0.4737|±  |0.1177|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4737|±  |0.1177|
| - ceval-valid_high_school_chinese                     |      2|none  |     0|acc                    |↑  | 0.6316|±  |0.1137|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6316|±  |0.1137|
| - ceval-valid_high_school_geography                   |      2|none  |     0|acc                    |↑  | 0.6316|±  |0.1137|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6316|±  |0.1137|
| - ceval-valid_high_school_history                     |      2|none  |     0|acc                    |↑  | 0.7500|±  |0.0993|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7500|±  |0.0993|
| - ceval-valid_high_school_mathematics                 |      2|none  |     0|acc                    |↑  | 0.5000|±  |0.1213|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5000|±  |0.1213|
| - ceval-valid_high_school_physics                     |      2|none  |     0|acc                    |↑  | 0.7895|±  |0.0961|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7895|±  |0.0961|
| - ceval-valid_high_school_politics                    |      2|none  |     0|acc                    |↑  | 0.8947|±  |0.0723|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8947|±  |0.0723|
| - ceval-valid_ideological_and_moral_cultivation       |      2|none  |     0|acc                    |↑  | 0.9474|±  |0.0526|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.9474|±  |0.0526|
| - ceval-valid_law                                     |      2|none  |     0|acc                    |↑  | 0.5000|±  |0.1043|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5000|±  |0.1043|
| - ceval-valid_legal_professional                      |      2|none  |     0|acc                    |↑  | 0.5217|±  |0.1065|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5217|±  |0.1065|
| - ceval-valid_logic                                   |      2|none  |     0|acc                    |↑  | 0.6818|±  |0.1016|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6818|±  |0.1016|
| - ceval-valid_mao_zedong_thought                      |      2|none  |     0|acc                    |↑  | 0.8750|±  |0.0690|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8750|±  |0.0690|
| - ceval-valid_marxism                                 |      2|none  |     0|acc                    |↑  | 0.7895|±  |0.0961|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7895|±  |0.0961|
| - ceval-valid_metrology_engineer                      |      2|none  |     0|acc                    |↑  | 0.7083|±  |0.0948|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7083|±  |0.0948|
| - ceval-valid_middle_school_biology                   |      2|none  |     0|acc                    |↑  | 0.9524|±  |0.0476|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.9524|±  |0.0476|
| - ceval-valid_middle_school_chemistry                 |      2|none  |     0|acc                    |↑  | 0.9000|±  |0.0688|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.9000|±  |0.0688|
| - ceval-valid_middle_school_geography                 |      2|none  |     0|acc                    |↑  | 0.6667|±  |0.1421|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6667|±  |0.1421|
| - ceval-valid_middle_school_history                   |      2|none  |     0|acc                    |↑  | 1.0000|±  |     0|
|                                                       |       |none  |     0|acc_norm               |↑  | 1.0000|±  |     0|
| - ceval-valid_middle_school_mathematics               |      2|none  |     0|acc                    |↑  | 0.5263|±  |0.1177|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5263|±  |0.1177|
| - ceval-valid_middle_school_physics                   |      2|none  |     0|acc                    |↑  | 0.7368|±  |0.1038|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7368|±  |0.1038|
| - ceval-valid_middle_school_politics                  |      2|none  |     0|acc                    |↑  | 0.8571|±  |0.0782|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8571|±  |0.0782|
| - ceval-valid_modern_chinese_history                  |      2|none  |     0|acc                    |↑  | 0.7391|±  |0.0936|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7391|±  |0.0936|
| - ceval-valid_operating_system                        |      2|none  |     0|acc                    |↑  | 0.6316|±  |0.1137|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6316|±  |0.1137|
| - ceval-valid_physician                               |      2|none  |     0|acc                    |↑  | 0.6939|±  |0.0665|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6939|±  |0.0665|
| - ceval-valid_plant_protection                        |      2|none  |     0|acc                    |↑  | 0.8636|±  |0.0749|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8636|±  |0.0749|
| - ceval-valid_probability_and_statistics              |      2|none  |     0|acc                    |↑  | 0.3889|±  |0.1182|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3889|±  |0.1182|
| - ceval-valid_professional_tour_guide                 |      2|none  |     0|acc                    |↑  | 0.5862|±  |0.0931|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5862|±  |0.0931|
| - ceval-valid_sports_science                          |      2|none  |     0|acc                    |↑  | 0.6842|±  |0.1096|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6842|±  |0.1096|
| - ceval-valid_tax_accountant                          |      2|none  |     0|acc                    |↑  | 0.5714|±  |0.0714|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5714|±  |0.0714|
| - ceval-valid_teacher_qualification                   |      2|none  |     0|acc                    |↑  | 0.7727|±  |0.0639|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7727|±  |0.0639|
| - ceval-valid_urban_and_rural_planner                 |      2|none  |     0|acc                    |↑  | 0.6739|±  |0.0699|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6739|±  |0.0699|
| - ceval-valid_veterinary_medicine                     |      2|none  |     0|acc                    |↑  | 0.6957|±  |0.0981|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6957|±  |0.0981|
|hellaswag                                              |      1|none  |     0|acc                    |↑  | 0.4820|±  |0.0050|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6521|±  |0.0048|
|ifeval                                                 |      4|none  |     0|inst_level_loose_acc   |↑  | 0.2854|±  |   N/A|
|                                                       |       |none  |     0|inst_level_strict_acc  |↑  | 0.2650|±  |   N/A|
|                                                       |       |none  |     0|prompt_level_loose_acc |↑  | 0.1590|±  |0.0157|
|                                                       |       |none  |     0|prompt_level_strict_acc|↑  | 0.1405|±  |0.0150|
|longbench_2wikimqa                                     |      5|none  |     0|qa_f1_score            |↑  | 0.1690|±  |0.0214|
|                                                       |       |none  |     0|score                  |↑  | 0.1690|±  |0.0214|
|longbench_2wikimqa_e                                   |      5|none  |     0|qa_f1_score            |↑  | 0.1637|±  |0.0157|
|                                                       |       |none  |     0|score                  |↑  | 0.1637|±  |0.0157|
|longbench_dureader                                     |      5|none  |     0|rouge_zh_score         |↑  | 0.2438|±  |0.0158|
|                                                       |       |none  |     0|score                  |↑  | 0.2438|±  |0.0158|
|longbench_gov_report                                   |      5|none  |     0|rouge_score            |↑  | 0.3037|±  |0.0064|
|                                                       |       |none  |     0|score                  |↑  | 0.3037|±  |0.0064|
|longbench_gov_report_e                                 |      5|none  |     0|rouge_score            |↑  | 0.3125|±  |0.0050|
|                                                       |       |none  |     0|score                  |↑  | 0.3125|±  |0.0050|
|longbench_hotpotqa                                     |      5|none  |     0|qa_f1_score            |↑  | 0.1325|±  |0.0158|
|                                                       |       |none  |     0|score                  |↑  | 0.1325|±  |0.0158|
|longbench_hotpotqa_e                                   |      5|none  |     0|qa_f1_score            |↑  | 0.1886|±  |0.0162|
|                                                       |       |none  |     0|score                  |↑  | 0.1886|±  |0.0162|
|longbench_lcc                                          |      5|none  |     0|code_sim_score         |↑  | 0.0593|±  |0.0047|
|                                                       |       |none  |     0|score                  |↑  | 0.0593|±  |0.0047|
|longbench_lcc_e                                        |      5|none  |     0|code_sim_score         |↑  | 0.1072|±  |0.0104|
|                                                       |       |none  |     0|score                  |↑  | 0.1072|±  |0.0104|
|longbench_lsht                                         |      5|none  |     0|classification_score   |↑  | 0.3417|±  |0.0322|
|                                                       |       |none  |     0|score                  |↑  | 0.3417|±  |0.0322|
|longbench_multi_news                                   |      5|none  |     0|rouge_score            |↑  | 0.2416|±  |0.0070|
|                                                       |       |none  |     0|score                  |↑  | 0.2416|±  |0.0070|
|longbench_multi_news_e                                 |      5|none  |     0|rouge_score            |↑  | 0.1882|±  |0.0062|
|                                                       |       |none  |     0|score                  |↑  | 0.1882|±  |0.0062|
|longbench_multifieldqa_en                              |      5|none  |     0|qa_f1_score            |↑  | 0.3802|±  |0.0250|
|                                                       |       |none  |     0|score                  |↑  | 0.3802|±  |0.0250|
|longbench_multifieldqa_en_e                            |      5|none  |     0|qa_f1_score            |↑  | 0.3802|±  |0.0250|
|                                                       |       |none  |     0|score                  |↑  | 0.3802|±  |0.0250|
|longbench_multifieldqa_zh                              |      5|none  |     0|qa_f1_zh_score         |↑  | 0.2911|±  |0.0179|
|                                                       |       |none  |     0|score                  |↑  | 0.2911|±  |0.0179|
|longbench_musique                                      |      5|none  |     0|qa_f1_score            |↑  | 0.0619|±  |0.0087|
|                                                       |       |none  |     0|score                  |↑  | 0.0619|±  |0.0087|
|longbench_narrativeqa                                  |      5|none  |     0|qa_f1_score            |↑  | 0.0709|±  |0.0095|
|                                                       |       |none  |     0|score                  |↑  | 0.0709|±  |0.0095|
|longbench_passage_count                                |      5|none  |     0|count_score            |↑  | 0.0237|±  |0.0101|
|                                                       |       |none  |     0|score                  |↑  | 0.0237|±  |0.0101|
|longbench_passage_count_e                              |      5|none  |     0|count_score            |↑  | 0.0471|±  |0.0122|
|                                                       |       |none  |     0|score                  |↑  | 0.0471|±  |0.0122|
|longbench_passage_retrieval_en                         |      5|none  |     0|retrieval_score        |↑  | 0.1825|±  |0.0273|
|                                                       |       |none  |     0|score                  |↑  | 0.1825|±  |0.0273|
|longbench_passage_retrieval_en_e                       |      5|none  |     0|retrieval_score        |↑  | 0.2950|±  |0.0263|
|                                                       |       |none  |     0|score                  |↑  | 0.2950|±  |0.0263|
|longbench_qasper                                       |      5|none  |     0|qa_f1_score            |↑  | 0.2676|±  |0.0204|
|                                                       |       |none  |     0|score                  |↑  | 0.2676|±  |0.0204|
|longbench_qasper_e                                     |      5|none  |     0|qa_f1_score            |↑  | 0.2498|±  |0.0183|
|                                                       |       |none  |     0|score                  |↑  | 0.2498|±  |0.0183|
|longbench_qmsum                                        |      5|none  |     0|rouge_score            |↑  | 0.2080|±  |0.0052|
|                                                       |       |none  |     0|score                  |↑  | 0.2080|±  |0.0052|
|longbench_repobench-p                                  |      5|none  |     0|code_sim_score         |↑  | 0.0959|±  |0.0071|
|                                                       |       |none  |     0|score                  |↑  | 0.0959|±  |0.0071|
|longbench_repobench-p_e                                |      5|none  |     0|code_sim_score         |↑  | 0.0995|±  |0.0091|
|                                                       |       |none  |     0|score                  |↑  | 0.0995|±  |0.0091|
|longbench_samsum                                       |      5|none  |     0|rouge_score            |↑  | 0.3584|±  |0.0101|
|                                                       |       |none  |     0|score                  |↑  | 0.3584|±  |0.0101|
|longbench_samsum_e                                     |      5|none  |     0|rouge_score            |↑  | 0.3463|±  |0.0076|
|                                                       |       |none  |     0|score                  |↑  | 0.3463|±  |0.0076|
|longbench_trec                                         |      5|none  |     0|classification_score   |↑  | 0.4050|±  |0.0220|
|                                                       |       |none  |     0|score                  |↑  | 0.4050|±  |0.0220|
|longbench_trec_e                                       |      5|none  |     0|classification_score   |↑  | 0.3750|±  |0.0172|
|                                                       |       |none  |     0|score                  |↑  | 0.3750|±  |0.0172|
|longbench_triviaqa                                     |      5|none  |     0|qa_f1_score            |↑  | 0.2134|±  |0.0077|
|                                                       |       |none  |     0|score                  |↑  | 0.2134|±  |0.0077|
|longbench_triviaqa_e                                   |      5|none  |     0|qa_f1_score            |↑  | 0.2057|±  |0.0059|
|                                                       |       |none  |     0|score                  |↑  | 0.2057|±  |0.0059|
|longbench_vcsum                                        |      5|none  |     0|rouge_zh_score         |↑  | 0.1002|±  |0.0033|
|                                                       |       |none  |     0|score                  |↑  | 0.1002|±  |0.0033|
|mmlu                                                   |      2|none  |      |acc                    |↑  | 0.5998|±  |0.0039|
| - humanities                                          |      2|none  |      |acc                    |↑  | 0.5107|±  |0.0068|
|  - formal_logic                                       |      1|none  |     0|acc                    |↑  | 0.4762|±  |0.0447|
|  - high_school_european_history                       |      1|none  |     0|acc                    |↑  | 0.6909|±  |0.0361|
|  - high_school_us_history                             |      1|none  |     0|acc                    |↑  | 0.7108|±  |0.0318|
|  - high_school_world_history                          |      1|none  |     0|acc                    |↑  | 0.7848|±  |0.0268|
|  - international_law                                  |      1|none  |     0|acc                    |↑  | 0.8017|±  |0.0364|
|  - jurisprudence                                      |      1|none  |     0|acc                    |↑  | 0.7222|±  |0.0433|
|  - logical_fallacies                                  |      1|none  |     0|acc                    |↑  | 0.7546|±  |0.0338|
|  - moral_disputes                                     |      1|none  |     0|acc                    |↑  | 0.6590|±  |0.0255|
|  - moral_scenarios                                    |      1|none  |     0|acc                    |↑  | 0.2469|±  |0.0144|
|  - philosophy                                         |      1|none  |     0|acc                    |↑  | 0.6399|±  |0.0273|
|  - prehistory                                         |      1|none  |     0|acc                    |↑  | 0.6327|±  |0.0268|
|  - professional_law                                   |      1|none  |     0|acc                    |↑  | 0.4055|±  |0.0125|
|  - world_religions                                    |      1|none  |     0|acc                    |↑  | 0.7310|±  |0.0340|
| - other                                               |      2|none  |      |acc                    |↑  | 0.6437|±  |0.0083|
|  - business_ethics                                    |      1|none  |     0|acc                    |↑  | 0.6300|±  |0.0485|
|  - clinical_knowledge                                 |      1|none  |     0|acc                    |↑  | 0.6453|±  |0.0294|
|  - college_medicine                                   |      1|none  |     0|acc                    |↑  | 0.5896|±  |0.0375|
|  - global_facts                                       |      1|none  |     0|acc                    |↑  | 0.2900|±  |0.0456|
|  - human_aging                                        |      1|none  |     0|acc                    |↑  | 0.6233|±  |0.0325|
|  - management                                         |      1|none  |     0|acc                    |↑  | 0.7670|±  |0.0419|
|  - marketing                                          |      1|none  |     0|acc                    |↑  | 0.8462|±  |0.0236|
|  - medical_genetics                                   |      1|none  |     0|acc                    |↑  | 0.7000|±  |0.0461|
|  - miscellaneous                                      |      1|none  |     0|acc                    |↑  | 0.7356|±  |0.0158|
|  - nutrition                                          |      1|none  |     0|acc                    |↑  | 0.6797|±  |0.0267|
|  - professional_accounting                            |      1|none  |     0|acc                    |↑  | 0.4433|±  |0.0296|
|  - professional_medicine                              |      1|none  |     0|acc                    |↑  | 0.5846|±  |0.0299|
|  - virology                                           |      1|none  |     0|acc                    |↑  | 0.4880|±  |0.0389|
| - social sciences                                     |      2|none  |      |acc                    |↑  | 0.7127|±  |0.0080|
|  - econometrics                                       |      1|none  |     0|acc                    |↑  | 0.4825|±  |0.0470|
|  - high_school_geography                              |      1|none  |     0|acc                    |↑  | 0.7879|±  |0.0291|
|  - high_school_government_and_politics                |      1|none  |     0|acc                    |↑  | 0.8083|±  |0.0284|
|  - high_school_macroeconomics                         |      1|none  |     0|acc                    |↑  | 0.6538|±  |0.0241|
|  - high_school_microeconomics                         |      1|none  |     0|acc                    |↑  | 0.7395|±  |0.0285|
|  - high_school_psychology                             |      1|none  |     0|acc                    |↑  | 0.8257|±  |0.0163|
|  - human_sexuality                                    |      1|none  |     0|acc                    |↑  | 0.7176|±  |0.0395|
|  - professional_psychology                            |      1|none  |     0|acc                    |↑  | 0.6062|±  |0.0198|
|  - public_relations                                   |      1|none  |     0|acc                    |↑  | 0.5909|±  |0.0471|
|  - security_studies                                   |      1|none  |     0|acc                    |↑  | 0.7265|±  |0.0285|
|  - sociology                                          |      1|none  |     0|acc                    |↑  | 0.7811|±  |0.0292|
|  - us_foreign_policy                                  |      1|none  |     0|acc                    |↑  | 0.8000|±  |0.0402|
| - stem                                                |      2|none  |      |acc                    |↑  | 0.5794|±  |0.0085|
|  - abstract_algebra                                   |      1|none  |     0|acc                    |↑  | 0.3700|±  |0.0485|
|  - anatomy                                            |      1|none  |     0|acc                    |↑  | 0.5926|±  |0.0424|
|  - astronomy                                          |      1|none  |     0|acc                    |↑  | 0.7105|±  |0.0369|
|  - college_biology                                    |      1|none  |     0|acc                    |↑  | 0.7569|±  |0.0359|
|  - college_chemistry                                  |      1|none  |     0|acc                    |↑  | 0.4200|±  |0.0496|
|  - college_computer_science                           |      1|none  |     0|acc                    |↑  | 0.6000|±  |0.0492|
|  - college_mathematics                                |      1|none  |     0|acc                    |↑  | 0.3900|±  |0.0490|
|  - college_physics                                    |      1|none  |     0|acc                    |↑  | 0.4412|±  |0.0494|
|  - computer_security                                  |      1|none  |     0|acc                    |↑  | 0.8000|±  |0.0402|
|  - conceptual_physics                                 |      1|none  |     0|acc                    |↑  | 0.6213|±  |0.0317|
|  - electrical_engineering                             |      1|none  |     0|acc                    |↑  | 0.6069|±  |0.0407|
|  - elementary_mathematics                             |      1|none  |     0|acc                    |↑  | 0.5556|±  |0.0256|
|  - high_school_biology                                |      1|none  |     0|acc                    |↑  | 0.7839|±  |0.0234|
|  - high_school_chemistry                              |      1|none  |     0|acc                    |↑  | 0.5813|±  |0.0347|
|  - high_school_computer_science                       |      1|none  |     0|acc                    |↑  | 0.6900|±  |0.0465|
|  - high_school_mathematics                            |      1|none  |     0|acc                    |↑  | 0.4296|±  |0.0302|
|  - high_school_physics                                |      1|none  |     0|acc                    |↑  | 0.4371|±  |0.0405|
|  - high_school_statistics                             |      1|none  |     0|acc                    |↑  | 0.5602|±  |0.0339|
|  - machine_learning                                   |      1|none  |     0|acc                    |↑  | 0.4464|±  |0.0472|
|openbookqa                                             |      1|none  |     0|acc                    |↑  | 0.2980|±  |0.0205|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3920|±  |0.0219|
|piqa                                                   |      1|none  |     0|acc                    |↑  | 0.7541|±  |0.0100|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7601|±  |0.0100|
|social_iqa                                             |      0|none  |     0|acc                    |↑  | 0.4616|±  |0.0113|
|truthfulqa_gen                                         |      3|none  |     0|bleu_acc               |↑  | 0.4027|±  |0.0172|
|                                                       |       |none  |     0|bleu_diff              |↑  |-0.1318|±  |0.0498|
|                                                       |       |none  |     0|bleu_max               |↑  | 1.4033|±  |0.0856|
|                                                       |       |none  |     0|rouge1_acc             |↑  | 0.4406|±  |0.0174|
|                                                       |       |none  |     0|rouge1_diff            |↑  |-0.2033|±  |0.0886|
|                                                       |       |none  |     0|rouge1_max             |↑  | 5.3768|±  |0.1446|
|                                                       |       |none  |     0|rouge2_acc             |↑  | 0.3660|±  |0.0169|
|                                                       |       |none  |     0|rouge2_diff            |↑  |-0.2816|±  |0.0959|
|                                                       |       |none  |     0|rouge2_max             |↑  | 3.2554|±  |0.1305|
|                                                       |       |none  |     0|rougeL_acc             |↑  | 0.4272|±  |0.0173|
|                                                       |       |none  |     0|rougeL_diff            |↑  |-0.2099|±  |0.0884|
|                                                       |       |none  |     0|rougeL_max             |↑  | 5.1236|±  |0.1410|
|truthfulqa_mc1                                         |      2|none  |     0|acc                    |↑  | 0.2950|±  |0.0160|
|truthfulqa_mc2                                         |      3|none  |     0|acc                    |↑  | 0.4511|±  |0.0144|
|winogrande                                             |      1|none  |     0|acc                    |↑  | 0.6401|±  |0.0135|
|niah_single_1|      1|none  |     0|  1024|   |1.000|±  |     0|
|             |       |none  |     0| 16384|↑  |0.998|±  |   N/A|
|             |       |none  |     0|  2048|   |1.000|±  |     0|
|             |       |none  |     0| 32768|↑  |0.996|±  |   N/A|
|             |       |none  |     0|  4096|↑  |1.000|±  |   N/A|
|             |       |none  |     0|  8192|↑  |0.998|±  |   N/A|
|niah_single_2|      1|none  |     0|  1024|   |1.000|±  |0.0000|
|             |       |none  |     0| 16384|↑  |0.998|±  |   N/A|
|             |       |none  |     0|  2048|   |1.000|±  |0.0000|
|             |       |none  |     0| 32768|↑  |0.888|±  |   N/A|
|             |       |none  |     0|  4096|↑  |1.000|±  |   N/A|
|             |       |none  |     0|  8192|↑  |1.000|±  |   N/A|
|niah_single_3|      1|none  |     0|  1024|   |1.000|±  |0.0000|
|             |       |none  |     0| 16384|↑  |0.992|±  |   N/A|
|             |       |none  |     0|  2048|   |0.998|±  |0.0020|
|             |       |none  |     0| 32768|↑  |0.922|±  |   N/A|
|             |       |none  |     0|  4096|↑  |0.996|±  |   N/A|
|             |       |none  |     0|  8192|↑  |0.980|±  |   N/A|
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