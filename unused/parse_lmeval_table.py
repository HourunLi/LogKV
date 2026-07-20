import pandas as pd
import numpy as np
import io

# 把你的各个模型的原始输出放在这里。这里以 Model_A 为例（填入你提供的原数据）
model_outputs = {
    "base-prolong": """
|                         Tasks                         |Version|Filter|n-shot|        Metric         |   | Value |   |Stderr|
|-------------------------------------------------------|------:|------|-----:|-----------------------|---|------:|---|------|
|arc_challenge                                          |      1|none  |     0|acc                    |↑  | 0.4411|±  |0.0145|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4770|±  |0.0146|
|arc_easy                                               |      1|none  |     0|acc                    |↑  | 0.7698|±  |0.0086|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7609|±  |0.0088|
|boolq                                                  |      2|none  |     0|acc                    |↑  | 0.7618|±  |0.0075|
|ceval-valid                                            |      2|none  |      |acc                    |↑  | 0.6367|±  |0.0126|
|                                                       |       |none  |      |acc_norm               |↑  | 0.6367|±  |0.0126|
| - ceval-valid_accountant                              |      2|none  |     0|acc                    |↑  | 0.5306|±  |0.0720|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5306|±  |0.0720|
| - ceval-valid_advanced_mathematics                    |      2|none  |     0|acc                    |↑  | 0.3158|±  |0.1096|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3158|±  |0.1096|
| - ceval-valid_art_studies                             |      2|none  |     0|acc                    |↑  | 0.5455|±  |0.0880|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5455|±  |0.0880|
| - ceval-valid_basic_medicine                          |      2|none  |     0|acc                    |↑  | 0.6316|±  |0.1137|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6316|±  |0.1137|
| - ceval-valid_business_administration                 |      2|none  |     0|acc                    |↑  | 0.6364|±  |0.0850|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6364|±  |0.0850|
| - ceval-valid_chinese_language_and_literature         |      2|none  |     0|acc                    |↑  | 0.4348|±  |0.1057|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4348|±  |0.1057|
| - ceval-valid_civil_servant                           |      2|none  |     0|acc                    |↑  | 0.5745|±  |0.0729|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5745|±  |0.0729|
| - ceval-valid_clinical_medicine                       |      2|none  |     0|acc                    |↑  | 0.6364|±  |0.1050|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6364|±  |0.1050|
| - ceval-valid_college_chemistry                       |      2|none  |     0|acc                    |↑  | 0.5833|±  |0.1028|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5833|±  |0.1028|
| - ceval-valid_college_economics                       |      2|none  |     0|acc                    |↑  | 0.5818|±  |0.0671|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5818|±  |0.0671|
| - ceval-valid_college_physics                         |      2|none  |     0|acc                    |↑  | 0.4737|±  |0.1177|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4737|±  |0.1177|
| - ceval-valid_college_programming                     |      2|none  |     0|acc                    |↑  | 0.6486|±  |0.0796|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6486|±  |0.0796|
| - ceval-valid_computer_architecture                   |      2|none  |     0|acc                    |↑  | 0.7619|±  |0.0952|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7619|±  |0.0952|
| - ceval-valid_computer_network                        |      2|none  |     0|acc                    |↑  | 0.5263|±  |0.1177|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5263|±  |0.1177|
| - ceval-valid_discrete_mathematics                    |      2|none  |     0|acc                    |↑  | 0.1250|±  |0.0854|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.1250|±  |0.0854|
| - ceval-valid_education_science                       |      2|none  |     0|acc                    |↑  | 0.7586|±  |0.0809|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7586|±  |0.0809|
| - ceval-valid_electrical_engineer                     |      2|none  |     0|acc                    |↑  | 0.3514|±  |0.0796|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3514|±  |0.0796|
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
| - ceval-valid_legal_professional                      |      2|none  |     0|acc                    |↑  | 0.4783|±  |0.1065|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4783|±  |0.1065|
| - ceval-valid_logic                                   |      2|none  |     0|acc                    |↑  | 0.6818|±  |0.1016|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6818|±  |0.1016|
| - ceval-valid_mao_zedong_thought                      |      2|none  |     0|acc                    |↑  | 0.8750|±  |0.0690|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8750|±  |0.0690|
| - ceval-valid_marxism                                 |      2|none  |     0|acc                    |↑  | 0.8421|±  |0.0859|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8421|±  |0.0859|
| - ceval-valid_metrology_engineer                      |      2|none  |     0|acc                    |↑  | 0.7083|±  |0.0948|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7083|±  |0.0948|
| - ceval-valid_middle_school_biology                   |      2|none  |     0|acc                    |↑  | 0.9524|±  |0.0476|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.9524|±  |0.0476|
| - ceval-valid_middle_school_chemistry                 |      2|none  |     0|acc                    |↑  | 0.9500|±  |0.0500|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.9500|±  |0.0500|
| - ceval-valid_middle_school_geography                 |      2|none  |     0|acc                    |↑  | 0.5833|±  |0.1486|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5833|±  |0.1486|
| - ceval-valid_middle_school_history                   |      2|none  |     0|acc                    |↑  | 1.0000|±  |     0|
|                                                       |       |none  |     0|acc_norm               |↑  | 1.0000|±  |     0|
| - ceval-valid_middle_school_mathematics               |      2|none  |     0|acc                    |↑  | 0.5263|±  |0.1177|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5263|±  |0.1177|
| - ceval-valid_middle_school_physics                   |      2|none  |     0|acc                    |↑  | 0.7895|±  |0.0961|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7895|±  |0.0961|
| - ceval-valid_middle_school_politics                  |      2|none  |     0|acc                    |↑  | 0.8571|±  |0.0782|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8571|±  |0.0782|
| - ceval-valid_modern_chinese_history                  |      2|none  |     0|acc                    |↑  | 0.6957|±  |0.0981|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6957|±  |0.0981|
| - ceval-valid_operating_system                        |      2|none  |     0|acc                    |↑  | 0.6316|±  |0.1137|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6316|±  |0.1137|
| - ceval-valid_physician                               |      2|none  |     0|acc                    |↑  | 0.6531|±  |0.0687|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6531|±  |0.0687|
| - ceval-valid_plant_protection                        |      2|none  |     0|acc                    |↑  | 0.8636|±  |0.0749|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8636|±  |0.0749|
| - ceval-valid_probability_and_statistics              |      2|none  |     0|acc                    |↑  | 0.3333|±  |0.1143|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3333|±  |0.1143|
| - ceval-valid_professional_tour_guide                 |      2|none  |     0|acc                    |↑  | 0.5172|±  |0.0944|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5172|±  |0.0944|
| - ceval-valid_sports_science                          |      2|none  |     0|acc                    |↑  | 0.6316|±  |0.1137|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6316|±  |0.1137|
| - ceval-valid_tax_accountant                          |      2|none  |     0|acc                    |↑  | 0.5714|±  |0.0714|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5714|±  |0.0714|
| - ceval-valid_teacher_qualification                   |      2|none  |     0|acc                    |↑  | 0.7727|±  |0.0639|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7727|±  |0.0639|
| - ceval-valid_urban_and_rural_planner                 |      2|none  |     0|acc                    |↑  | 0.6739|±  |0.0699|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6739|±  |0.0699|
| - ceval-valid_veterinary_medicine                     |      2|none  |     0|acc                    |↑  | 0.6957|±  |0.0981|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6957|±  |0.0981|
|hellaswag                                              |      1|none  |     0|acc                    |↑  | 0.4853|±  |0.0050|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6527|±  |0.0048|
|ifeval                                                 |      4|none  |     0|inst_level_loose_acc   |↑  | 0.2710|±  |   N/A|
|                                                       |       |none  |     0|inst_level_strict_acc  |↑  | 0.2566|±  |   N/A|
|                                                       |       |none  |     0|prompt_level_loose_acc |↑  | 0.1553|±  |0.0156|
|                                                       |       |none  |     0|prompt_level_strict_acc|↑  | 0.1386|±  |0.0149|
|longbench_2wikimqa                                     |      5|none  |     0|qa_f1_score            |↑  | 0.1150|±  |0.0122|
|                                                       |       |none  |     0|score                  |↑  | 0.1150|±  |0.0122|
|longbench_2wikimqa_e                                   |      5|none  |     0|qa_f1_score            |↑  | 0.1066|±  |0.0089|
|                                                       |       |none  |     0|score                  |↑  | 0.1066|±  |0.0089|
|longbench_dureader                                     |      5|none  |     0|rouge_zh_score         |↑  | 0.1652|±  |0.0088|
|                                                       |       |none  |     0|score                  |↑  | 0.1652|±  |0.0088|
|longbench_gov_report                                   |      5|none  |     0|rouge_score            |↑  | 0.2051|±  |0.0054|
|                                                       |       |none  |     0|score                  |↑  | 0.2051|±  |0.0054|
|longbench_gov_report_e                                 |      5|none  |     0|rouge_score            |↑  | 0.2176|±  |0.0047|
|                                                       |       |none  |     0|score                  |↑  | 0.2176|±  |0.0047|
|longbench_hotpotqa                                     |      5|none  |     0|qa_f1_score            |↑  | 0.0906|±  |0.0106|
|                                                       |       |none  |     0|score                  |↑  | 0.0906|±  |0.0106|
|longbench_hotpotqa_e                                   |      5|none  |     0|qa_f1_score            |↑  | 0.1011|±  |0.0093|
|                                                       |       |none  |     0|score                  |↑  | 0.1011|±  |0.0093|
|longbench_lcc                                          |      5|none  |     0|code_sim_score         |↑  | 0.0572|±  |0.0054|
|                                                       |       |none  |     0|score                  |↑  | 0.0572|±  |0.0054|
|longbench_lcc_e                                        |      5|none  |     0|code_sim_score         |↑  | 0.1310|±  |0.0126|
|                                                       |       |none  |     0|score                  |↑  | 0.1310|±  |0.0126|
|longbench_lsht                                         |      5|none  |     0|classification_score   |↑  | 0.1942|±  |0.0252|
|                                                       |       |none  |     0|score                  |↑  | 0.1942|±  |0.0252|
|longbench_multi_news                                   |      5|none  |     0|rouge_score            |↑  | 0.1909|±  |0.0093|
|                                                       |       |none  |     0|score                  |↑  | 0.1909|±  |0.0093|
|longbench_multi_news_e                                 |      5|none  |     0|rouge_score            |↑  | 0.1705|±  |0.0067|
|                                                       |       |none  |     0|score                  |↑  | 0.1705|±  |0.0067|
|longbench_multifieldqa_en                              |      5|none  |     0|qa_f1_score            |↑  | 0.3033|±  |0.0241|
|                                                       |       |none  |     0|score                  |↑  | 0.3033|±  |0.0241|
|longbench_multifieldqa_en_e                            |      5|none  |     0|qa_f1_score            |↑  | 0.3033|±  |0.0241|
|                                                       |       |none  |     0|score                  |↑  | 0.3033|±  |0.0241|
|longbench_multifieldqa_zh                              |      5|none  |     0|qa_f1_zh_score         |↑  | 0.2349|±  |0.0158|
|                                                       |       |none  |     0|score                  |↑  | 0.2349|±  |0.0158|
|longbench_musique                                      |      5|none  |     0|qa_f1_score            |↑  | 0.0403|±  |0.0050|
|                                                       |       |none  |     0|score                  |↑  | 0.0403|±  |0.0050|
|longbench_narrativeqa                                  |      5|none  |     0|qa_f1_score            |↑  | 0.0250|±  |0.0025|
|                                                       |       |none  |     0|score                  |↑  | 0.0250|±  |0.0025|
|longbench_passage_count                                |      5|none  |     0|count_score            |↑  | 0.0470|±  |0.0147|
|                                                       |       |none  |     0|score                  |↑  | 0.0470|±  |0.0147|
|longbench_passage_count_e                              |      5|none  |     0|count_score            |↑  | 0.0667|±  |0.0141|
|                                                       |       |none  |     0|score                  |↑  | 0.0667|±  |0.0141|
|longbench_passage_retrieval_en                         |      5|none  |     0|retrieval_score        |↑  | 0.0708|±  |0.0173|
|                                                       |       |none  |     0|score                  |↑  | 0.0708|±  |0.0173|
|longbench_passage_retrieval_en_e                       |      5|none  |     0|retrieval_score        |↑  | 0.1006|±  |0.0169|
|                                                       |       |none  |     0|score                  |↑  | 0.1006|±  |0.0169|
|longbench_qasper                                       |      5|none  |     0|qa_f1_score            |↑  | 0.1709|±  |0.0141|
|                                                       |       |none  |     0|score                  |↑  | 0.1709|±  |0.0141|
|longbench_qasper_e                                     |      5|none  |     0|qa_f1_score            |↑  | 0.1447|±  |0.0122|
|                                                       |       |none  |     0|score                  |↑  | 0.1447|±  |0.0122|
|longbench_qmsum                                        |      5|none  |     0|rouge_score            |↑  | 0.2162|±  |0.0049|
|                                                       |       |none  |     0|score                  |↑  | 0.2162|±  |0.0049|
|longbench_repobench-p                                  |      5|none  |     0|code_sim_score         |↑  | 0.1434|±  |0.0110|
|                                                       |       |none  |     0|score                  |↑  | 0.1434|±  |0.0110|
|longbench_repobench-p_e                                |      5|none  |     0|code_sim_score         |↑  | 0.1444|±  |0.0131|
|                                                       |       |none  |     0|score                  |↑  | 0.1444|±  |0.0131|
|longbench_samsum                                       |      5|none  |     0|rouge_score            |↑  | 0.3191|±  |0.0092|
|                                                       |       |none  |     0|score                  |↑  | 0.3191|±  |0.0092|
|longbench_samsum_e                                     |      5|none  |     0|rouge_score            |↑  | 0.3094|±  |0.0072|
|                                                       |       |none  |     0|score                  |↑  | 0.3094|±  |0.0072|
|longbench_trec                                         |      5|none  |     0|classification_score   |↑  | 0.4092|±  |0.0233|
|                                                       |       |none  |     0|score                  |↑  | 0.4092|±  |0.0233|
|longbench_trec_e                                       |      5|none  |     0|classification_score   |↑  | 0.3483|±  |0.0189|
|                                                       |       |none  |     0|score                  |↑  | 0.3483|±  |0.0189|
|longbench_triviaqa                                     |      5|none  |     0|qa_f1_score            |↑  | 0.1978|±  |0.0078|
|                                                       |       |none  |     0|score                  |↑  | 0.1978|±  |0.0078|
|longbench_triviaqa_e                                   |      5|none  |     0|qa_f1_score            |↑  | 0.2002|±  |0.0073|
|                                                       |       |none  |     0|score                  |↑  | 0.2002|±  |0.0073|
|longbench_vcsum                                        |      5|none  |     0|rouge_zh_score         |↑  | 0.0562|±  |0.0045|
|                                                       |       |none  |     0|score                  |↑  | 0.0562|±  |0.0045|
|mmlu                                                   |      2|none  |      |acc                    |↑  | 0.5994|±  |0.0039|
| - humanities                                          |      2|none  |      |acc                    |↑  | 0.5097|±  |0.0068|
|  - formal_logic                                       |      1|none  |     0|acc                    |↑  | 0.4683|±  |0.0446|
|  - high_school_european_history                       |      1|none  |     0|acc                    |↑  | 0.6909|±  |0.0361|
|  - high_school_us_history                             |      1|none  |     0|acc                    |↑  | 0.7255|±  |0.0313|
|  - high_school_world_history                          |      1|none  |     0|acc                    |↑  | 0.7679|±  |0.0275|
|  - international_law                                  |      1|none  |     0|acc                    |↑  | 0.7603|±  |0.0390|
|  - jurisprudence                                      |      1|none  |     0|acc                    |↑  | 0.7315|±  |0.0428|
|  - logical_fallacies                                  |      1|none  |     0|acc                    |↑  | 0.7607|±  |0.0335|
|  - moral_disputes                                     |      1|none  |     0|acc                    |↑  | 0.6532|±  |0.0256|
|  - moral_scenarios                                    |      1|none  |     0|acc                    |↑  | 0.2436|±  |0.0144|
|  - philosophy                                         |      1|none  |     0|acc                    |↑  | 0.6399|±  |0.0273|
|  - prehistory                                         |      1|none  |     0|acc                    |↑  | 0.6420|±  |0.0267|
|  - professional_law                                   |      1|none  |     0|acc                    |↑  | 0.4055|±  |0.0125|
|  - world_religions                                    |      1|none  |     0|acc                    |↑  | 0.7427|±  |0.0335|
| - other                                               |      2|none  |      |acc                    |↑  | 0.6479|±  |0.0083|
|  - business_ethics                                    |      1|none  |     0|acc                    |↑  | 0.6400|±  |0.0482|
|  - clinical_knowledge                                 |      1|none  |     0|acc                    |↑  | 0.6642|±  |0.0291|
|  - college_medicine                                   |      1|none  |     0|acc                    |↑  | 0.6069|±  |0.0372|
|  - global_facts                                       |      1|none  |     0|acc                    |↑  | 0.3800|±  |0.0488|
|  - human_aging                                        |      1|none  |     0|acc                    |↑  | 0.6323|±  |0.0324|
|  - management                                         |      1|none  |     0|acc                    |↑  | 0.7670|±  |0.0419|
|  - marketing                                          |      1|none  |     0|acc                    |↑  | 0.8376|±  |0.0242|
|  - medical_genetics                                   |      1|none  |     0|acc                    |↑  | 0.7100|±  |0.0456|
|  - miscellaneous                                      |      1|none  |     0|acc                    |↑  | 0.7216|±  |0.0160|
|  - nutrition                                          |      1|none  |     0|acc                    |↑  | 0.6797|±  |0.0267|
|  - professional_accounting                            |      1|none  |     0|acc                    |↑  | 0.4397|±  |0.0296|
|  - professional_medicine                              |      1|none  |     0|acc                    |↑  | 0.6029|±  |0.0297|
|  - virology                                           |      1|none  |     0|acc                    |↑  | 0.4940|±  |0.0389|
| - social sciences                                     |      2|none  |      |acc                    |↑  | 0.7088|±  |0.0080|
|  - econometrics                                       |      1|none  |     0|acc                    |↑  | 0.4737|±  |0.0470|
|  - high_school_geography                              |      1|none  |     0|acc                    |↑  | 0.7929|±  |0.0289|
|  - high_school_government_and_politics                |      1|none  |     0|acc                    |↑  | 0.7927|±  |0.0293|
|  - high_school_macroeconomics                         |      1|none  |     0|acc                    |↑  | 0.6308|±  |0.0245|
|  - high_school_microeconomics                         |      1|none  |     0|acc                    |↑  | 0.7101|±  |0.0295|
|  - high_school_psychology                             |      1|none  |     0|acc                    |↑  | 0.8239|±  |0.0163|
|  - human_sexuality                                    |      1|none  |     0|acc                    |↑  | 0.7099|±  |0.0398|
|  - professional_psychology                            |      1|none  |     0|acc                    |↑  | 0.6225|±  |0.0196|
|  - public_relations                                   |      1|none  |     0|acc                    |↑  | 0.5727|±  |0.0474|
|  - security_studies                                   |      1|none  |     0|acc                    |↑  | 0.7102|±  |0.0290|
|  - sociology                                          |      1|none  |     0|acc                    |↑  | 0.7910|±  |0.0287|
|  - us_foreign_policy                                  |      1|none  |     0|acc                    |↑  | 0.8300|±  |0.0378|
| - stem                                                |      2|none  |      |acc                    |↑  | 0.5788|±  |0.0085|
|  - abstract_algebra                                   |      1|none  |     0|acc                    |↑  | 0.3800|±  |0.0488|
|  - anatomy                                            |      1|none  |     0|acc                    |↑  | 0.5778|±  |0.0427|
|  - astronomy                                          |      1|none  |     0|acc                    |↑  | 0.6908|±  |0.0376|
|  - college_biology                                    |      1|none  |     0|acc                    |↑  | 0.7500|±  |0.0362|
|  - college_chemistry                                  |      1|none  |     0|acc                    |↑  | 0.4400|±  |0.0499|
|  - college_computer_science                           |      1|none  |     0|acc                    |↑  | 0.5600|±  |0.0499|
|  - college_mathematics                                |      1|none  |     0|acc                    |↑  | 0.4000|±  |0.0492|
|  - college_physics                                    |      1|none  |     0|acc                    |↑  | 0.4314|±  |0.0493|
|  - computer_security                                  |      1|none  |     0|acc                    |↑  | 0.7800|±  |0.0416|
|  - conceptual_physics                                 |      1|none  |     0|acc                    |↑  | 0.6213|±  |0.0317|
|  - electrical_engineering                             |      1|none  |     0|acc                    |↑  | 0.6207|±  |0.0404|
|  - elementary_mathematics                             |      1|none  |     0|acc                    |↑  | 0.5741|±  |0.0255|
|  - high_school_biology                                |      1|none  |     0|acc                    |↑  | 0.7774|±  |0.0237|
|  - high_school_chemistry                              |      1|none  |     0|acc                    |↑  | 0.5813|±  |0.0347|
|  - high_school_computer_science                       |      1|none  |     0|acc                    |↑  | 0.7200|±  |0.0451|
|  - high_school_mathematics                            |      1|none  |     0|acc                    |↑  | 0.3963|±  |0.0298|
|  - high_school_physics                                |      1|none  |     0|acc                    |↑  | 0.4503|±  |0.0406|
|  - high_school_statistics                             |      1|none  |     0|acc                    |↑  | 0.5694|±  |0.0338|
|  - machine_learning                                   |      1|none  |     0|acc                    |↑  | 0.4643|±  |0.0473|
|openbookqa                                             |      1|none  |     0|acc                    |↑  | 0.3040|±  |0.0206|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3900|±  |0.0218|
|piqa                                                   |      1|none  |     0|acc                    |↑  | 0.7573|±  |0.0100|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7644|±  |0.0099|
|social_iqa                                             |      0|none  |     0|acc                    |↑  | 0.4611|±  |0.0113|
|truthfulqa_gen                                         |      3|none  |     0|bleu_acc               |↑  | 0.4076|±  |0.0172|
|                                                       |       |none  |     0|bleu_diff              |↑  |-0.0927|±  |0.0543|
|                                                       |       |none  |     0|bleu_max               |↑  | 1.5308|±  |0.1144|
|                                                       |       |none  |     0|rouge1_acc             |↑  | 0.4468|±  |0.0174|
|                                                       |       |none  |     0|rouge1_diff            |↑  |-0.1681|±  |0.0893|
|                                                       |       |none  |     0|rouge1_max             |↑  | 5.5804|±  |0.1829|
|                                                       |       |none  |     0|rouge2_acc             |↑  | 0.3856|±  |0.0170|
|                                                       |       |none  |     0|rouge2_diff            |↑  |-0.2461|±  |0.0970|
|                                                       |       |none  |     0|rouge2_max             |↑  | 3.4486|±  |0.1623|
|                                                       |       |none  |     0|rougeL_acc             |↑  | 0.4321|±  |0.0173|
|                                                       |       |none  |     0|rougeL_diff            |↑  |-0.1468|±  |0.0882|
|                                                       |       |none  |     0|rougeL_max             |↑  | 5.3392|±  |0.1815|
|truthfulqa_mc1                                         |      2|none  |     0|acc                    |↑  | 0.2938|±  |0.0159|
|truthfulqa_mc2                                         |      3|none  |     0|acc                    |↑  | 0.4563|±  |0.0145|
|winogrande                                             |      1|none  |     0|acc                    |↑  | 0.6472|±  |0.0134|
|niah_single_1|      1|none  |     0|  1024|   |1.000|±  |     0|
|             |       |none  |     0| 16384|↑  |0.054|±  |   N/A|
|             |       |none  |     0|  2048|   |0.716|±  |0.0202|
|             |       |none  |     0| 32768|↑  |0.024|±  |   N/A|
|             |       |none  |     0|  4096|↑  |0.402|±  |   N/A|
|             |       |none  |     0|  8192|↑  |0.194|±  |   N/A|
|niah_single_2|      1|none  |     0|  1024|   |1.000|±  |0.0000|
|             |       |none  |     0| 16384|↑  |0.110|±  |   N/A|
|             |       |none  |     0|  2048|   |0.852|±  |0.0159|
|             |       |none  |     0| 32768|↑  |0.046|±  |   N/A|
|             |       |none  |     0|  4096|↑  |0.358|±  |   N/A|
|             |       |none  |     0|  8192|↑  |0.190|±  |   N/A|
|niah_single_3|      1|none  |     0|  1024|   |1.000|±  |0.0000|   
|             |       |none  |     0| 16384|↑  |0.082|±  |   N/A|
|             |       |none  |     0|  2048|   |0.774|±  |0.0187|
|             |       |none  |     0| 32768|↑  |0.026|±  |   N/A|
|             |       |none  |     0|  4096|↑  |0.314|±  |   N/A|
|             |       |none  |     0|  8192|↑  |0.150|±  |   N/A|
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