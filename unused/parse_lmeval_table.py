import pandas as pd
import numpy as np
import io

# 把你的各个模型的原始输出放在这里。这里以 Model_A 为例（填入你提供的原数据）
model_outputs = {
    "base-prolong": """
|                         Tasks                         |Version|Filter|n-shot|        Metric         |   | Value |   |Stderr|
|-------------------------------------------------------|------:|------|-----:|-----------------------|---|------:|---|------|
|arc_challenge                                          |      1|none  |     0|acc                    |↑  | 0.4113|±  |0.0144|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4462|±  |0.0145|
|arc_easy                                               |      1|none  |     0|acc                    |↑  | 0.7374|±  |0.0090|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6940|±  |0.0095|
|boolq                                                  |      2|none  |     0|acc                    |↑  | 0.7881|±  |0.0071|
|ceval-valid                                            |      2|none  |      |acc                    |↑  | 0.6441|±  |0.0126|
|                                                       |       |none  |      |acc_norm               |↑  | 0.6441|±  |0.0126|
| - ceval-valid_accountant                              |      2|none  |     0|acc                    |↑  | 0.5714|±  |0.0714|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5714|±  |0.0714|
| - ceval-valid_advanced_mathematics                    |      2|none  |     0|acc                    |↑  | 0.3684|±  |0.1137|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3684|±  |0.1137|
| - ceval-valid_art_studies                             |      2|none  |     0|acc                    |↑  | 0.5455|±  |0.0880|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5455|±  |0.0880|
| - ceval-valid_basic_medicine                          |      2|none  |     0|acc                    |↑  | 0.6842|±  |0.1096|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6842|±  |0.1096|
| - ceval-valid_business_administration                 |      2|none  |     0|acc                    |↑  | 0.6061|±  |0.0864|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6061|±  |0.0864|
| - ceval-valid_chinese_language_and_literature         |      2|none  |     0|acc                    |↑  | 0.5652|±  |0.1057|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5652|±  |0.1057|
| - ceval-valid_civil_servant                           |      2|none  |     0|acc                    |↑  | 0.5319|±  |0.0736|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5319|±  |0.0736|
| - ceval-valid_clinical_medicine                       |      2|none  |     0|acc                    |↑  | 0.5909|±  |0.1073|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5909|±  |0.1073|
| - ceval-valid_college_chemistry                       |      2|none  |     0|acc                    |↑  | 0.5833|±  |0.1028|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5833|±  |0.1028|
| - ceval-valid_college_economics                       |      2|none  |     0|acc                    |↑  | 0.5455|±  |0.0678|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5455|±  |0.0678|
| - ceval-valid_college_physics                         |      2|none  |     0|acc                    |↑  | 0.5789|±  |0.1164|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5789|±  |0.1164|
| - ceval-valid_college_programming                     |      2|none  |     0|acc                    |↑  | 0.7297|±  |0.0740|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7297|±  |0.0740|
| - ceval-valid_computer_architecture                   |      2|none  |     0|acc                    |↑  | 0.8095|±  |0.0878|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8095|±  |0.0878|
| - ceval-valid_computer_network                        |      2|none  |     0|acc                    |↑  | 0.5263|±  |0.1177|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5263|±  |0.1177|
| - ceval-valid_discrete_mathematics                    |      2|none  |     0|acc                    |↑  | 0.2500|±  |0.1118|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.2500|±  |0.1118|
| - ceval-valid_education_science                       |      2|none  |     0|acc                    |↑  | 0.7931|±  |0.0766|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7931|±  |0.0766|
| - ceval-valid_electrical_engineer                     |      2|none  |     0|acc                    |↑  | 0.3514|±  |0.0796|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3514|±  |0.0796|
| - ceval-valid_environmental_impact_assessment_engineer|      2|none  |     0|acc                    |↑  | 0.6774|±  |0.0853|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6774|±  |0.0853|
| - ceval-valid_fire_engineer                           |      2|none  |     0|acc                    |↑  | 0.5484|±  |0.0909|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5484|±  |0.0909|
| - ceval-valid_high_school_biology                     |      2|none  |     0|acc                    |↑  | 0.7368|±  |0.1038|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7368|±  |0.1038|
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
| - ceval-valid_high_school_physics                     |      2|none  |     0|acc                    |↑  | 0.6842|±  |0.1096|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6842|±  |0.1096|
| - ceval-valid_high_school_politics                    |      2|none  |     0|acc                    |↑  | 0.8947|±  |0.0723|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8947|±  |0.0723|
| - ceval-valid_ideological_and_moral_cultivation       |      2|none  |     0|acc                    |↑  | 0.9474|±  |0.0526|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.9474|±  |0.0526|
| - ceval-valid_law                                     |      2|none  |     0|acc                    |↑  | 0.4583|±  |0.1039|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4583|±  |0.1039|
| - ceval-valid_legal_professional                      |      2|none  |     0|acc                    |↑  | 0.4348|±  |0.1057|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4348|±  |0.1057|
| - ceval-valid_logic                                   |      2|none  |     0|acc                    |↑  | 0.6364|±  |0.1050|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6364|±  |0.1050|
| - ceval-valid_mao_zedong_thought                      |      2|none  |     0|acc                    |↑  | 0.8333|±  |0.0777|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8333|±  |0.0777|
| - ceval-valid_marxism                                 |      2|none  |     0|acc                    |↑  | 0.8421|±  |0.0859|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8421|±  |0.0859|
| - ceval-valid_metrology_engineer                      |      2|none  |     0|acc                    |↑  | 0.7500|±  |0.0903|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7500|±  |0.0903|
| - ceval-valid_middle_school_biology                   |      2|none  |     0|acc                    |↑  | 0.9524|±  |0.0476|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.9524|±  |0.0476|
| - ceval-valid_middle_school_chemistry                 |      2|none  |     0|acc                    |↑  | 0.9000|±  |0.0688|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.9000|±  |0.0688|
| - ceval-valid_middle_school_geography                 |      2|none  |     0|acc                    |↑  | 0.6667|±  |0.1421|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6667|±  |0.1421|
| - ceval-valid_middle_school_history                   |      2|none  |     0|acc                    |↑  | 0.9545|±  |0.0455|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.9545|±  |0.0455|
| - ceval-valid_middle_school_mathematics               |      2|none  |     0|acc                    |↑  | 0.4737|±  |0.1177|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4737|±  |0.1177|
| - ceval-valid_middle_school_physics                   |      2|none  |     0|acc                    |↑  | 0.7895|±  |0.0961|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7895|±  |0.0961|
| - ceval-valid_middle_school_politics                  |      2|none  |     0|acc                    |↑  | 0.8571|±  |0.0782|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8571|±  |0.0782|
| - ceval-valid_modern_chinese_history                  |      2|none  |     0|acc                    |↑  | 0.7826|±  |0.0879|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7826|±  |0.0879|
| - ceval-valid_operating_system                        |      2|none  |     0|acc                    |↑  | 0.6842|±  |0.1096|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6842|±  |0.1096|
| - ceval-valid_physician                               |      2|none  |     0|acc                    |↑  | 0.6327|±  |0.0696|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6327|±  |0.0696|
| - ceval-valid_plant_protection                        |      2|none  |     0|acc                    |↑  | 0.8182|±  |0.0842|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8182|±  |0.0842|
| - ceval-valid_probability_and_statistics              |      2|none  |     0|acc                    |↑  | 0.4444|±  |0.1205|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4444|±  |0.1205|
| - ceval-valid_professional_tour_guide                 |      2|none  |     0|acc                    |↑  | 0.5862|±  |0.0931|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5862|±  |0.0931|
| - ceval-valid_sports_science                          |      2|none  |     0|acc                    |↑  | 0.6842|±  |0.1096|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6842|±  |0.1096|
| - ceval-valid_tax_accountant                          |      2|none  |     0|acc                    |↑  | 0.5510|±  |0.0718|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5510|±  |0.0718|
| - ceval-valid_teacher_qualification                   |      2|none  |     0|acc                    |↑  | 0.7273|±  |0.0679|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7273|±  |0.0679|
| - ceval-valid_urban_and_rural_planner                 |      2|none  |     0|acc                    |↑  | 0.6739|±  |0.0699|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6739|±  |0.0699|
| - ceval-valid_veterinary_medicine                     |      2|none  |     0|acc                    |↑  | 0.7391|±  |0.0936|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7391|±  |0.0936|
|hellaswag                                              |      1|none  |     0|acc                    |↑  | 0.4896|±  |0.0050|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6601|±  |0.0047|
|ifeval                                                 |      4|none  |     0|inst_level_loose_acc   |↑  | 0.3118|±  |   N/A|
|                                                       |       |none  |     0|inst_level_strict_acc  |↑  | 0.2902|±  |   N/A|
|                                                       |       |none  |     0|prompt_level_loose_acc |↑  | 0.1959|±  |0.0171|
|                                                       |       |none  |     0|prompt_level_strict_acc|↑  | 0.1774|±  |0.0164|
|longbench_2wikimqa                                     |      5|none  |     0|qa_f1_score            |↑  | 0.2087|±  |0.0267|
|                                                       |       |none  |     0|score                  |↑  | 0.2087|±  |0.0267|
|longbench_2wikimqa_e                                   |      5|none  |     0|qa_f1_score            |↑  | 0.2393|±  |0.0220|
|                                                       |       |none  |     0|score                  |↑  | 0.2393|±  |0.0220|
|longbench_dureader                                     |      5|none  |     0|rouge_zh_score         |↑  | 0.1870|±  |0.0110|
|                                                       |       |none  |     0|score                  |↑  | 0.1870|±  |0.0110|
|longbench_gov_report                                   |      5|none  |     0|rouge_score            |↑  | 0.2319|±  |0.0053|
|                                                       |       |none  |     0|score                  |↑  | 0.2319|±  |0.0053|
|longbench_gov_report_e                                 |      5|none  |     0|rouge_score            |↑  | 0.2444|±  |0.0043|
|                                                       |       |none  |     0|score                  |↑  | 0.2444|±  |0.0043|
|longbench_hotpotqa                                     |      5|none  |     0|qa_f1_score            |↑  | 0.2116|±  |0.0229|
|                                                       |       |none  |     0|score                  |↑  | 0.2116|±  |0.0229|
|longbench_hotpotqa_e                                   |      5|none  |     0|qa_f1_score            |↑  | 0.2652|±  |0.0217|
|                                                       |       |none  |     0|score                  |↑  | 0.2652|±  |0.0217|
|longbench_lcc                                          |      5|none  |     0|code_sim_score         |↑  | 0.0434|±  |0.0038|
|                                                       |       |none  |     0|score                  |↑  | 0.0434|±  |0.0038|
|longbench_lcc_e                                        |      5|none  |     0|code_sim_score         |↑  | 0.0849|±  |0.0097|
|                                                       |       |none  |     0|score                  |↑  | 0.0849|±  |0.0097|
|longbench_lsht                                         |      5|none  |     0|classification_score   |↑  | 0.3125|±  |0.0314|
|                                                       |       |none  |     0|score                  |↑  | 0.3125|±  |0.0314|
|longbench_multi_news                                   |      5|none  |     0|rouge_score            |↑  | 0.2495|±  |0.0058|
|                                                       |       |none  |     0|score                  |↑  | 0.2495|±  |0.0058|
|longbench_multi_news_e                                 |      5|none  |     0|rouge_score            |↑  | 0.1781|±  |0.0062|
|                                                       |       |none  |     0|score                  |↑  | 0.1781|±  |0.0062|
|longbench_multifieldqa_en                              |      5|none  |     0|qa_f1_score            |↑  | 0.4030|±  |0.0255|
|                                                       |       |none  |     0|score                  |↑  | 0.4030|±  |0.0255|
|longbench_multifieldqa_en_e                            |      5|none  |     0|qa_f1_score            |↑  | 0.4030|±  |0.0255|
|                                                       |       |none  |     0|score                  |↑  | 0.4030|±  |0.0255|
|longbench_multifieldqa_zh                              |      5|none  |     0|qa_f1_zh_score         |↑  | 0.5030|±  |0.0236|
|                                                       |       |none  |     0|score                  |↑  | 0.5030|±  |0.0236|
|longbench_musique                                      |      5|none  |     0|qa_f1_score            |↑  | 0.0880|±  |0.0121|
|                                                       |       |none  |     0|score                  |↑  | 0.0880|±  |0.0121|
|longbench_narrativeqa                                  |      5|none  |     0|qa_f1_score            |↑  | 0.1330|±  |0.0142|
|                                                       |       |none  |     0|score                  |↑  | 0.1330|±  |0.0142|
|longbench_passage_count                                |      5|none  |     0|count_score            |↑  | 0.0105|±  |0.0071|
|                                                       |       |none  |     0|score                  |↑  | 0.0105|±  |0.0071|
|longbench_passage_count_e                              |      5|none  |     0|count_score            |↑  | 0.0143|±  |0.0067|
|                                                       |       |none  |     0|score                  |↑  | 0.0143|±  |0.0067|
|longbench_passage_retrieval_en                         |      5|none  |     0|retrieval_score        |↑  | 0.1450|±  |0.0250|
|                                                       |       |none  |     0|score                  |↑  | 0.1450|±  |0.0250|
|longbench_passage_retrieval_en_e                       |      5|none  |     0|retrieval_score        |↑  | 0.4700|±  |0.0289|
|                                                       |       |none  |     0|score                  |↑  | 0.4700|±  |0.0289|
|longbench_qasper                                       |      5|none  |     0|qa_f1_score            |↑  | 0.3215|±  |0.0260|
|                                                       |       |none  |     0|score                  |↑  | 0.3215|±  |0.0260|
|longbench_qasper_e                                     |      5|none  |     0|qa_f1_score            |↑  | 0.3042|±  |0.0240|
|                                                       |       |none  |     0|score                  |↑  | 0.3042|±  |0.0240|
|longbench_qmsum                                        |      5|none  |     0|rouge_score            |↑  | 0.2163|±  |0.0062|
|                                                       |       |none  |     0|score                  |↑  | 0.2163|±  |0.0062|
|longbench_repobench-p                                  |      5|none  |     0|code_sim_score         |↑  | 0.1007|±  |0.0074|
|                                                       |       |none  |     0|score                  |↑  | 0.1007|±  |0.0074|
|longbench_repobench-p_e                                |      5|none  |     0|code_sim_score         |↑  | 0.0993|±  |0.0077|
|                                                       |       |none  |     0|score                  |↑  | 0.0993|±  |0.0077|
|longbench_samsum                                       |      5|none  |     0|rouge_score            |↑  | 0.3594|±  |0.0099|
|                                                       |       |none  |     0|score                  |↑  | 0.3594|±  |0.0099|
|longbench_samsum_e                                     |      5|none  |     0|rouge_score            |↑  | 0.3531|±  |0.0083|
|                                                       |       |none  |     0|score                  |↑  | 0.3531|±  |0.0083|
|longbench_trec                                         |      5|none  |     0|classification_score   |↑  | 0.4200|±  |0.0258|
|                                                       |       |none  |     0|score                  |↑  | 0.4200|±  |0.0258|
|longbench_trec_e                                       |      5|none  |     0|classification_score   |↑  | 0.4022|±  |0.0208|
|                                                       |       |none  |     0|score                  |↑  | 0.4022|±  |0.0208|
|longbench_triviaqa                                     |      5|none  |     0|qa_f1_score            |↑  | 0.2816|±  |0.0193|
|                                                       |       |none  |     0|score                  |↑  | 0.2816|±  |0.0193|
|longbench_triviaqa_e                                   |      5|none  |     0|qa_f1_score            |↑  | 0.2709|±  |0.0150|
|                                                       |       |none  |     0|score                  |↑  | 0.2709|±  |0.0150|
|longbench_vcsum                                        |      5|none  |     0|rouge_zh_score         |↑  | 0.1120|±  |0.0054|
|                                                       |       |none  |     0|score                  |↑  | 0.1120|±  |0.0054|
|mmlu                                                   |      2|none  |      |acc                    |↑  | 0.6022|±  |0.0039|
| - humanities                                          |      2|none  |      |acc                    |↑  | 0.5133|±  |0.0068|
|  - formal_logic                                       |      1|none  |     0|acc                    |↑  | 0.5000|±  |0.0447|
|  - high_school_european_history                       |      1|none  |     0|acc                    |↑  | 0.7333|±  |0.0345|
|  - high_school_us_history                             |      1|none  |     0|acc                    |↑  | 0.7157|±  |0.0317|
|  - high_school_world_history                          |      1|none  |     0|acc                    |↑  | 0.7679|±  |0.0275|
|  - international_law                                  |      1|none  |     0|acc                    |↑  | 0.7355|±  |0.0403|
|  - jurisprudence                                      |      1|none  |     0|acc                    |↑  | 0.7222|±  |0.0433|
|  - logical_fallacies                                  |      1|none  |     0|acc                    |↑  | 0.7301|±  |0.0349|
|  - moral_disputes                                     |      1|none  |     0|acc                    |↑  | 0.6590|±  |0.0255|
|  - moral_scenarios                                    |      1|none  |     0|acc                    |↑  | 0.2380|±  |0.0142|
|  - philosophy                                         |      1|none  |     0|acc                    |↑  | 0.6624|±  |0.0269|
|  - prehistory                                         |      1|none  |     0|acc                    |↑  | 0.6605|±  |0.0263|
|  - professional_law                                   |      1|none  |     0|acc                    |↑  | 0.4081|±  |0.0126|
|  - world_religions                                    |      1|none  |     0|acc                    |↑  | 0.7602|±  |0.0327|
| - other                                               |      2|none  |      |acc                    |↑  | 0.6530|±  |0.0083|
|  - business_ethics                                    |      1|none  |     0|acc                    |↑  | 0.6500|±  |0.0479|
|  - clinical_knowledge                                 |      1|none  |     0|acc                    |↑  | 0.7019|±  |0.0282|
|  - college_medicine                                   |      1|none  |     0|acc                    |↑  | 0.6127|±  |0.0371|
|  - global_facts                                       |      1|none  |     0|acc                    |↑  | 0.3400|±  |0.0476|
|  - human_aging                                        |      1|none  |     0|acc                    |↑  | 0.6502|±  |0.0320|
|  - management                                         |      1|none  |     0|acc                    |↑  | 0.7767|±  |0.0412|
|  - marketing                                          |      1|none  |     0|acc                    |↑  | 0.8333|±  |0.0244|
|  - medical_genetics                                   |      1|none  |     0|acc                    |↑  | 0.6900|±  |0.0465|
|  - miscellaneous                                      |      1|none  |     0|acc                    |↑  | 0.7203|±  |0.0161|
|  - nutrition                                          |      1|none  |     0|acc                    |↑  | 0.6732|±  |0.0269|
|  - professional_accounting                            |      1|none  |     0|acc                    |↑  | 0.4645|±  |0.0298|
|  - professional_medicine                              |      1|none  |     0|acc                    |↑  | 0.6066|±  |0.0297|
|  - virology                                           |      1|none  |     0|acc                    |↑  | 0.5000|±  |0.0389|
| - social sciences                                     |      2|none  |      |acc                    |↑  | 0.7121|±  |0.0080|
|  - econometrics                                       |      1|none  |     0|acc                    |↑  | 0.5175|±  |0.0470|
|  - high_school_geography                              |      1|none  |     0|acc                    |↑  | 0.7828|±  |0.0294|
|  - high_school_government_and_politics                |      1|none  |     0|acc                    |↑  | 0.8031|±  |0.0287|
|  - high_school_macroeconomics                         |      1|none  |     0|acc                    |↑  | 0.6615|±  |0.0240|
|  - high_school_microeconomics                         |      1|none  |     0|acc                    |↑  | 0.7353|±  |0.0287|
|  - high_school_psychology                             |      1|none  |     0|acc                    |↑  | 0.8404|±  |0.0157|
|  - human_sexuality                                    |      1|none  |     0|acc                    |↑  | 0.7099|±  |0.0398|
|  - professional_psychology                            |      1|none  |     0|acc                    |↑  | 0.6127|±  |0.0197|
|  - public_relations                                   |      1|none  |     0|acc                    |↑  | 0.6091|±  |0.0467|
|  - security_studies                                   |      1|none  |     0|acc                    |↑  | 0.6735|±  |0.0300|
|  - sociology                                          |      1|none  |     0|acc                    |↑  | 0.7562|±  |0.0304|
|  - us_foreign_policy                                  |      1|none  |     0|acc                    |↑  | 0.7900|±  |0.0409|
| - stem                                                |      2|none  |      |acc                    |↑  | 0.5775|±  |0.0085|
|  - abstract_algebra                                   |      1|none  |     0|acc                    |↑  | 0.3400|±  |0.0476|
|  - anatomy                                            |      1|none  |     0|acc                    |↑  | 0.6074|±  |0.0422|
|  - astronomy                                          |      1|none  |     0|acc                    |↑  | 0.6974|±  |0.0374|
|  - college_biology                                    |      1|none  |     0|acc                    |↑  | 0.7222|±  |0.0375|
|  - college_chemistry                                  |      1|none  |     0|acc                    |↑  | 0.4600|±  |0.0501|
|  - college_computer_science                           |      1|none  |     0|acc                    |↑  | 0.5600|±  |0.0499|
|  - college_mathematics                                |      1|none  |     0|acc                    |↑  | 0.3900|±  |0.0490|
|  - college_physics                                    |      1|none  |     0|acc                    |↑  | 0.4608|±  |0.0496|
|  - computer_security                                  |      1|none  |     0|acc                    |↑  | 0.7900|±  |0.0409|
|  - conceptual_physics                                 |      1|none  |     0|acc                    |↑  | 0.6383|±  |0.0314|
|  - electrical_engineering                             |      1|none  |     0|acc                    |↑  | 0.6207|±  |0.0404|
|  - elementary_mathematics                             |      1|none  |     0|acc                    |↑  | 0.5556|±  |0.0256|
|  - high_school_biology                                |      1|none  |     0|acc                    |↑  | 0.7935|±  |0.0230|
|  - high_school_chemistry                              |      1|none  |     0|acc                    |↑  | 0.5813|±  |0.0347|
|  - high_school_computer_science                       |      1|none  |     0|acc                    |↑  | 0.7300|±  |0.0446|
|  - high_school_mathematics                            |      1|none  |     0|acc                    |↑  | 0.3963|±  |0.0298|
|  - high_school_physics                                |      1|none  |     0|acc                    |↑  | 0.4702|±  |0.0408|
|  - high_school_statistics                             |      1|none  |     0|acc                    |↑  | 0.5278|±  |0.0340|
|  - machine_learning                                   |      1|none  |     0|acc                    |↑  | 0.4375|±  |0.0471|
|openbookqa                                             |      1|none  |     0|acc                    |↑  | 0.2980|±  |0.0205|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3880|±  |0.0218|
|piqa                                                   |      1|none  |     0|acc                    |↑  | 0.7557|±  |0.0100|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7541|±  |0.0100|
|social_iqa                                             |      0|none  |     0|acc                    |↑  | 0.4826|±  |0.0113|
|truthfulqa_gen                                         |      3|none  |     0|bleu_acc               |↑  | 0.2044|±  |0.0141|
|                                                       |       |none  |     0|bleu_diff              |↑  |-0.1802|±  |0.0446|
|                                                       |       |none  |     0|bleu_max               |↑  | 0.9479|±  |0.0791|
|                                                       |       |none  |     0|rouge1_acc             |↑  | 0.2326|±  |0.0148|
|                                                       |       |none  |     0|rouge1_diff            |↑  |-0.2486|±  |0.0784|
|                                                       |       |none  |     0|rouge1_max             |↑  | 3.8484|±  |0.2040|
|                                                       |       |none  |     0|rouge2_acc             |↑  | 0.1836|±  |0.0136|
|                                                       |       |none  |     0|rouge2_diff            |↑  |-0.3934|±  |0.0886|
|                                                       |       |none  |     0|rouge2_max             |↑  | 2.2440|±  |0.1486|
|                                                       |       |none  |     0|rougeL_acc             |↑  | 0.2142|±  |0.0144|
|                                                       |       |none  |     0|rougeL_diff            |↑  |-0.3148|±  |0.0840|
|                                                       |       |none  |     0|rougeL_max             |↑  | 3.5436|±  |0.1912|
|truthfulqa_mc1                                         |      2|none  |     0|acc                    |↑  | 0.3244|±  |0.0164|
|truthfulqa_mc2                                         |      3|none  |     0|acc                    |↑  | 0.4821|±  |0.0147|
|winogrande                                             |      1|none  |     0|acc                    |↑  | 0.6448|±  |0.0135|
|niah_single_1|      1|none  |     0|  1024|   |1.000|±  |     0|
|             |       |none  |     0| 16384|↑  |1.000|±  |   N/A|
|             |       |none  |     0|  2048|   |1.000|±  |     0|
|             |       |none  |     0| 32768|↑  |1.000|±  |   N/A|
|             |       |none  |     0|  4096|↑  |1.000|±  |   N/A|
|             |       |none  |     0|  8192|↑  |1.000|±  |   N/A|
|niah_single_2|      1|none  |     0|  1024|   |1.000|±  |0.0000|
|             |       |none  |     0| 16384|↑  |0.422|±  |   N/A|
|             |       |none  |     0|  2048|   |0.998|±  |0.0020|
|             |       |none  |     0| 32768|↑  |0.230|±  |   N/A|
|             |       |none  |     0|  4096|↑  |1.000|±  |   N/A|
|             |       |none  |     0|  8192|↑  |0.870|±  |   N/A|
|niah_single_3|      1|none  |     0|  1024|   |1.000|±  |0.0000|
|             |       |none  |     0| 16384|↑  |0.798|±  |   N/A|
|             |       |none  |     0|  2048|   |1.000|±  |0.0000|
|             |       |none  |     0| 32768|↑  |0.276|±  |   N/A|
|             |       |none  |     0|  4096|↑  |1.000|±  |   N/A|
|             |       |none  |     0|  8192|↑  |0.976|±  |   N/A|
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