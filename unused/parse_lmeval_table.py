import pandas as pd
import numpy as np
import io

# 把你的各个模型的原始输出放在这里。这里以 Model_A 为例（填入你提供的原数据）
model_outputs = {
    "base-prolong": """
|                         Tasks                         |Version|Filter|n-shot|        Metric         |   | Value |   |Stderr|
|-------------------------------------------------------|------:|------|-----:|-----------------------|---|------:|---|------|
|arc_challenge                                          |      1|none  |     0|acc                    |↑  | 0.4113|±  |0.0144|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4454|±  |0.0145|
|arc_easy                                               |      1|none  |     0|acc                    |↑  | 0.7323|±  |0.0091|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6835|±  |0.0095|
|boolq                                                  |      2|none  |     0|acc                    |↑  | 0.7930|±  |0.0071|
|ceval-valid                                            |      2|none  |      |acc                    |↑  | 0.6464|±  |0.0126|
|                                                       |       |none  |      |acc_norm               |↑  | 0.6464|±  |0.0126|
| - ceval-valid_accountant                              |      2|none  |     0|acc                    |↑  | 0.5510|±  |0.0718|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5510|±  |0.0718|
| - ceval-valid_advanced_mathematics                    |      2|none  |     0|acc                    |↑  | 0.3158|±  |0.1096|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3158|±  |0.1096|
| - ceval-valid_art_studies                             |      2|none  |     0|acc                    |↑  | 0.6364|±  |0.0850|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6364|±  |0.0850|
| - ceval-valid_basic_medicine                          |      2|none  |     0|acc                    |↑  | 0.6842|±  |0.1096|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6842|±  |0.1096|
| - ceval-valid_business_administration                 |      2|none  |     0|acc                    |↑  | 0.6061|±  |0.0864|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6061|±  |0.0864|
| - ceval-valid_chinese_language_and_literature         |      2|none  |     0|acc                    |↑  | 0.6087|±  |0.1041|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6087|±  |0.1041|
| - ceval-valid_civil_servant                           |      2|none  |     0|acc                    |↑  | 0.5106|±  |0.0737|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5106|±  |0.0737|
| - ceval-valid_clinical_medicine                       |      2|none  |     0|acc                    |↑  | 0.5455|±  |0.1087|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5455|±  |0.1087|
| - ceval-valid_college_chemistry                       |      2|none  |     0|acc                    |↑  | 0.6250|±  |0.1009|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6250|±  |0.1009|
| - ceval-valid_college_economics                       |      2|none  |     0|acc                    |↑  | 0.5455|±  |0.0678|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5455|±  |0.0678|
| - ceval-valid_college_physics                         |      2|none  |     0|acc                    |↑  | 0.5263|±  |0.1177|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5263|±  |0.1177|
| - ceval-valid_college_programming                     |      2|none  |     0|acc                    |↑  | 0.7027|±  |0.0762|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7027|±  |0.0762|
| - ceval-valid_computer_architecture                   |      2|none  |     0|acc                    |↑  | 0.8095|±  |0.0878|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8095|±  |0.0878|
| - ceval-valid_computer_network                        |      2|none  |     0|acc                    |↑  | 0.5789|±  |0.1164|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5789|±  |0.1164|
| - ceval-valid_discrete_mathematics                    |      2|none  |     0|acc                    |↑  | 0.3750|±  |0.1250|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3750|±  |0.1250|
| - ceval-valid_education_science                       |      2|none  |     0|acc                    |↑  | 0.7931|±  |0.0766|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7931|±  |0.0766|
| - ceval-valid_electrical_engineer                     |      2|none  |     0|acc                    |↑  | 0.3514|±  |0.0796|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3514|±  |0.0796|
| - ceval-valid_environmental_impact_assessment_engineer|      2|none  |     0|acc                    |↑  | 0.6774|±  |0.0853|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6774|±  |0.0853|
| - ceval-valid_fire_engineer                           |      2|none  |     0|acc                    |↑  | 0.4839|±  |0.0912|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4839|±  |0.0912|
| - ceval-valid_high_school_biology                     |      2|none  |     0|acc                    |↑  | 0.7368|±  |0.1038|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7368|±  |0.1038|
| - ceval-valid_high_school_chemistry                   |      2|none  |     0|acc                    |↑  | 0.4737|±  |0.1177|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4737|±  |0.1177|
| - ceval-valid_high_school_chinese                     |      2|none  |     0|acc                    |↑  | 0.6316|±  |0.1137|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6316|±  |0.1137|
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
| - ceval-valid_law                                     |      2|none  |     0|acc                    |↑  | 0.5000|±  |0.1043|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5000|±  |0.1043|
| - ceval-valid_legal_professional                      |      2|none  |     0|acc                    |↑  | 0.4348|±  |0.1057|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4348|±  |0.1057|
| - ceval-valid_logic                                   |      2|none  |     0|acc                    |↑  | 0.6364|±  |0.1050|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6364|±  |0.1050|
| - ceval-valid_mao_zedong_thought                      |      2|none  |     0|acc                    |↑  | 0.8333|±  |0.0777|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8333|±  |0.0777|
| - ceval-valid_marxism                                 |      2|none  |     0|acc                    |↑  | 0.8421|±  |0.0859|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8421|±  |0.0859|
| - ceval-valid_metrology_engineer                      |      2|none  |     0|acc                    |↑  | 0.7917|±  |0.0847|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7917|±  |0.0847|
| - ceval-valid_middle_school_biology                   |      2|none  |     0|acc                    |↑  | 0.9524|±  |0.0476|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.9524|±  |0.0476|
| - ceval-valid_middle_school_chemistry                 |      2|none  |     0|acc                    |↑  | 0.9000|±  |0.0688|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.9000|±  |0.0688|
| - ceval-valid_middle_school_geography                 |      2|none  |     0|acc                    |↑  | 0.6667|±  |0.1421|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6667|±  |0.1421|
| - ceval-valid_middle_school_history                   |      2|none  |     0|acc                    |↑  | 1.0000|±  |     0|
|                                                       |       |none  |     0|acc_norm               |↑  | 1.0000|±  |     0|
| - ceval-valid_middle_school_mathematics               |      2|none  |     0|acc                    |↑  | 0.4737|±  |0.1177|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4737|±  |0.1177|
| - ceval-valid_middle_school_physics                   |      2|none  |     0|acc                    |↑  | 0.7368|±  |0.1038|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7368|±  |0.1038|
| - ceval-valid_middle_school_politics                  |      2|none  |     0|acc                    |↑  | 0.8095|±  |0.0878|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8095|±  |0.0878|
| - ceval-valid_modern_chinese_history                  |      2|none  |     0|acc                    |↑  | 0.8261|±  |0.0808|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8261|±  |0.0808|
| - ceval-valid_operating_system                        |      2|none  |     0|acc                    |↑  | 0.6316|±  |0.1137|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6316|±  |0.1137|
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
| - ceval-valid_teacher_qualification                   |      2|none  |     0|acc                    |↑  | 0.7955|±  |0.0615|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7955|±  |0.0615|
| - ceval-valid_urban_and_rural_planner                 |      2|none  |     0|acc                    |↑  | 0.6739|±  |0.0699|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6739|±  |0.0699|
| - ceval-valid_veterinary_medicine                     |      2|none  |     0|acc                    |↑  | 0.7391|±  |0.0936|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7391|±  |0.0936|
|hellaswag                                              |      1|none  |     0|acc                    |↑  | 0.4910|±  |0.0050|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6613|±  |0.0047|
|ifeval                                                 |      4|none  |     0|inst_level_loose_acc   |↑  | 0.3309|±  |   N/A|
|                                                       |       |none  |     0|inst_level_strict_acc  |↑  | 0.3118|±  |   N/A|
|                                                       |       |none  |     0|prompt_level_loose_acc |↑  | 0.2218|±  |0.0179|
|                                                       |       |none  |     0|prompt_level_strict_acc|↑  | 0.1978|±  |0.0171|
|longbench_2wikimqa                                     |      5|none  |     0|qa_f1_score            |↑  | 0.1507|±  |0.0212|
|                                                       |       |none  |     0|score                  |↑  | 0.1507|±  |0.0212|
|longbench_2wikimqa_e                                   |      5|none  |     0|qa_f1_score            |↑  | 0.1678|±  |0.0178|
|                                                       |       |none  |     0|score                  |↑  | 0.1678|±  |0.0178|
|longbench_dureader                                     |      5|none  |     0|rouge_zh_score         |↑  | 0.1363|±  |0.0079|
|                                                       |       |none  |     0|score                  |↑  | 0.1363|±  |0.0079|
|longbench_gov_report                                   |      5|none  |     0|rouge_score            |↑  | 0.1663|±  |0.0054|
|                                                       |       |none  |     0|score                  |↑  | 0.1663|±  |0.0054|
|longbench_gov_report_e                                 |      5|none  |     0|rouge_score            |↑  | 0.1856|±  |0.0043|
|                                                       |       |none  |     0|score                  |↑  | 0.1856|±  |0.0043|
|longbench_hotpotqa                                     |      5|none  |     0|qa_f1_score            |↑  | 0.1185|±  |0.0157|
|                                                       |       |none  |     0|score                  |↑  | 0.1185|±  |0.0157|
|longbench_hotpotqa_e                                   |      5|none  |     0|qa_f1_score            |↑  | 0.1833|±  |0.0174|
|                                                       |       |none  |     0|score                  |↑  | 0.1833|±  |0.0174|
|longbench_lcc                                          |      5|none  |     0|code_sim_score         |↑  | 0.0599|±  |0.0049|
|                                                       |       |none  |     0|score                  |↑  | 0.0599|±  |0.0049|
|longbench_lcc_e                                        |      5|none  |     0|code_sim_score         |↑  | 0.1629|±  |0.0130|
|                                                       |       |none  |     0|score                  |↑  | 0.1629|±  |0.0130|
|longbench_lsht                                         |      5|none  |     0|classification_score   |↑  | 0.2221|±  |0.0282|
|                                                       |       |none  |     0|score                  |↑  | 0.2221|±  |0.0282|
|longbench_multi_news                                   |      5|none  |     0|rouge_score            |↑  | 0.2306|±  |0.0067|
|                                                       |       |none  |     0|score                  |↑  | 0.2306|±  |0.0067|
|longbench_multi_news_e                                 |      5|none  |     0|rouge_score            |↑  | 0.1410|±  |0.0072|
|                                                       |       |none  |     0|score                  |↑  | 0.1410|±  |0.0072|
|longbench_multifieldqa_en                              |      5|none  |     0|qa_f1_score            |↑  | 0.2956|±  |0.0236|
|                                                       |       |none  |     0|score                  |↑  | 0.2956|±  |0.0236|
|longbench_multifieldqa_en_e                            |      5|none  |     0|qa_f1_score            |↑  | 0.2956|±  |0.0236|
|                                                       |       |none  |     0|score                  |↑  | 0.2956|±  |0.0236|
|longbench_multifieldqa_zh                              |      5|none  |     0|qa_f1_zh_score         |↑  | 0.3497|±  |0.0232|
|                                                       |       |none  |     0|score                  |↑  | 0.3497|±  |0.0232|
|longbench_musique                                      |      5|none  |     0|qa_f1_score            |↑  | 0.0565|±  |0.0085|
|                                                       |       |none  |     0|score                  |↑  | 0.0565|±  |0.0085|
|longbench_narrativeqa                                  |      5|none  |     0|qa_f1_score            |↑  | 0.0747|±  |0.0102|
|                                                       |       |none  |     0|score                  |↑  | 0.0747|±  |0.0102|
|longbench_passage_count                                |      5|none  |     0|count_score            |↑  | 0.0188|±  |0.0034|
|                                                       |       |none  |     0|score                  |↑  | 0.0188|±  |0.0034|
|longbench_passage_count_e                              |      5|none  |     0|count_score            |↑  | 0.0348|±  |0.0052|
|                                                       |       |none  |     0|score                  |↑  | 0.0348|±  |0.0052|
|longbench_passage_retrieval_en                         |      5|none  |     0|retrieval_score        |↑  | 0.0785|±  |0.0181|
|                                                       |       |none  |     0|score                  |↑  | 0.0785|±  |0.0181|
|longbench_passage_retrieval_en_e                       |      5|none  |     0|retrieval_score        |↑  | 0.2440|±  |0.0245|
|                                                       |       |none  |     0|score                  |↑  | 0.2440|±  |0.0245|
|longbench_qasper                                       |      5|none  |     0|qa_f1_score            |↑  | 0.2579|±  |0.0200|
|                                                       |       |none  |     0|score                  |↑  | 0.2579|±  |0.0200|
|longbench_qasper_e                                     |      5|none  |     0|qa_f1_score            |↑  | 0.2199|±  |0.0184|
|                                                       |       |none  |     0|score                  |↑  | 0.2199|±  |0.0184|
|longbench_qmsum                                        |      5|none  |     0|rouge_score            |↑  | 0.1764|±  |0.0070|
|                                                       |       |none  |     0|score                  |↑  | 0.1764|±  |0.0070|
|longbench_repobench-p                                  |      5|none  |     0|code_sim_score         |↑  | 0.1470|±  |0.0095|
|                                                       |       |none  |     0|score                  |↑  | 0.1470|±  |0.0095|
|longbench_repobench-p_e                                |      5|none  |     0|code_sim_score         |↑  | 0.1516|±  |0.0111|
|                                                       |       |none  |     0|score                  |↑  | 0.1516|±  |0.0111|
|longbench_samsum                                       |      5|none  |     0|rouge_score            |↑  | 0.3713|±  |0.0108|
|                                                       |       |none  |     0|score                  |↑  | 0.3713|±  |0.0108|
|longbench_samsum_e                                     |      5|none  |     0|rouge_score            |↑  | 0.3622|±  |0.0083|
|                                                       |       |none  |     0|score                  |↑  | 0.3622|±  |0.0083|
|longbench_trec                                         |      5|none  |     0|classification_score   |↑  | 0.3837|±  |0.0273|
|                                                       |       |none  |     0|score                  |↑  | 0.3837|±  |0.0273|
|longbench_trec_e                                       |      5|none  |     0|classification_score   |↑  | 0.3997|±  |0.0228|
|                                                       |       |none  |     0|score                  |↑  | 0.3997|±  |0.0228|
|longbench_triviaqa                                     |      5|none  |     0|qa_f1_score            |↑  | 0.2725|±  |0.0192|
|                                                       |       |none  |     0|score                  |↑  | 0.2725|±  |0.0192|
|longbench_triviaqa_e                                   |      5|none  |     0|qa_f1_score            |↑  | 0.2675|±  |0.0164|
|                                                       |       |none  |     0|score                  |↑  | 0.2675|±  |0.0164|
|longbench_vcsum                                        |      5|none  |     0|rouge_zh_score         |↑  | 0.0625|±  |0.0059|
|                                                       |       |none  |     0|score                  |↑  | 0.0625|±  |0.0059|
|mmlu                                                   |      2|none  |      |acc                    |↑  | 0.6008|±  |0.0039|
| - humanities                                          |      2|none  |      |acc                    |↑  | 0.5061|±  |0.0068|
|  - formal_logic                                       |      1|none  |     0|acc                    |↑  | 0.4841|±  |0.0447|
|  - high_school_european_history                       |      1|none  |     0|acc                    |↑  | 0.7515|±  |0.0337|
|  - high_school_us_history                             |      1|none  |     0|acc                    |↑  | 0.7010|±  |0.0321|
|  - high_school_world_history                          |      1|none  |     0|acc                    |↑  | 0.7806|±  |0.0269|
|  - international_law                                  |      1|none  |     0|acc                    |↑  | 0.6860|±  |0.0424|
|  - jurisprudence                                      |      1|none  |     0|acc                    |↑  | 0.7130|±  |0.0437|
|  - logical_fallacies                                  |      1|none  |     0|acc                    |↑  | 0.7423|±  |0.0344|
|  - moral_disputes                                     |      1|none  |     0|acc                    |↑  | 0.6358|±  |0.0259|
|  - moral_scenarios                                    |      1|none  |     0|acc                    |↑  | 0.2413|±  |0.0143|
|  - philosophy                                         |      1|none  |     0|acc                    |↑  | 0.6559|±  |0.0270|
|  - prehistory                                         |      1|none  |     0|acc                    |↑  | 0.6481|±  |0.0266|
|  - professional_law                                   |      1|none  |     0|acc                    |↑  | 0.3963|±  |0.0125|
|  - world_religions                                    |      1|none  |     0|acc                    |↑  | 0.7544|±  |0.0330|
| - other                                               |      2|none  |      |acc                    |↑  | 0.6501|±  |0.0083|
|  - business_ethics                                    |      1|none  |     0|acc                    |↑  | 0.6300|±  |0.0485|
|  - clinical_knowledge                                 |      1|none  |     0|acc                    |↑  | 0.6755|±  |0.0288|
|  - college_medicine                                   |      1|none  |     0|acc                    |↑  | 0.6069|±  |0.0372|
|  - global_facts                                       |      1|none  |     0|acc                    |↑  | 0.3400|±  |0.0476|
|  - human_aging                                        |      1|none  |     0|acc                    |↑  | 0.6413|±  |0.0322|
|  - management                                         |      1|none  |     0|acc                    |↑  | 0.7670|±  |0.0419|
|  - marketing                                          |      1|none  |     0|acc                    |↑  | 0.8333|±  |0.0244|
|  - medical_genetics                                   |      1|none  |     0|acc                    |↑  | 0.7200|±  |0.0451|
|  - miscellaneous                                      |      1|none  |     0|acc                    |↑  | 0.7165|±  |0.0161|
|  - nutrition                                          |      1|none  |     0|acc                    |↑  | 0.6830|±  |0.0266|
|  - professional_accounting                            |      1|none  |     0|acc                    |↑  | 0.4574|±  |0.0297|
|  - professional_medicine                              |      1|none  |     0|acc                    |↑  | 0.6140|±  |0.0296|
|  - virology                                           |      1|none  |     0|acc                    |↑  | 0.5060|±  |0.0389|
| - social sciences                                     |      2|none  |      |acc                    |↑  | 0.7130|±  |0.0080|
|  - econometrics                                       |      1|none  |     0|acc                    |↑  | 0.5351|±  |0.0469|
|  - high_school_geography                              |      1|none  |     0|acc                    |↑  | 0.8030|±  |0.0283|
|  - high_school_government_and_politics                |      1|none  |     0|acc                    |↑  | 0.7979|±  |0.0290|
|  - high_school_macroeconomics                         |      1|none  |     0|acc                    |↑  | 0.6564|±  |0.0241|
|  - high_school_microeconomics                         |      1|none  |     0|acc                    |↑  | 0.7311|±  |0.0288|
|  - high_school_psychology                             |      1|none  |     0|acc                    |↑  | 0.8330|±  |0.0160|
|  - human_sexuality                                    |      1|none  |     0|acc                    |↑  | 0.7252|±  |0.0392|
|  - professional_psychology                            |      1|none  |     0|acc                    |↑  | 0.6176|±  |0.0197|
|  - public_relations                                   |      1|none  |     0|acc                    |↑  | 0.6000|±  |0.0469|
|  - security_studies                                   |      1|none  |     0|acc                    |↑  | 0.6571|±  |0.0304|
|  - sociology                                          |      1|none  |     0|acc                    |↑  | 0.7662|±  |0.0299|
|  - us_foreign_policy                                  |      1|none  |     0|acc                    |↑  | 0.8200|±  |0.0386|
| - stem                                                |      2|none  |      |acc                    |↑  | 0.5839|±  |0.0085|
|  - abstract_algebra                                   |      1|none  |     0|acc                    |↑  | 0.3500|±  |0.0479|
|  - anatomy                                            |      1|none  |     0|acc                    |↑  | 0.5926|±  |0.0424|
|  - astronomy                                          |      1|none  |     0|acc                    |↑  | 0.7171|±  |0.0367|
|  - college_biology                                    |      1|none  |     0|acc                    |↑  | 0.7431|±  |0.0365|
|  - college_chemistry                                  |      1|none  |     0|acc                    |↑  | 0.4900|±  |0.0502|
|  - college_computer_science                           |      1|none  |     0|acc                    |↑  | 0.5500|±  |0.0500|
|  - college_mathematics                                |      1|none  |     0|acc                    |↑  | 0.4500|±  |0.0500|
|  - college_physics                                    |      1|none  |     0|acc                    |↑  | 0.4706|±  |0.0497|
|  - computer_security                                  |      1|none  |     0|acc                    |↑  | 0.7900|±  |0.0409|
|  - conceptual_physics                                 |      1|none  |     0|acc                    |↑  | 0.6340|±  |0.0315|
|  - electrical_engineering                             |      1|none  |     0|acc                    |↑  | 0.6207|±  |0.0404|
|  - elementary_mathematics                             |      1|none  |     0|acc                    |↑  | 0.5529|±  |0.0256|
|  - high_school_biology                                |      1|none  |     0|acc                    |↑  | 0.7935|±  |0.0230|
|  - high_school_chemistry                              |      1|none  |     0|acc                    |↑  | 0.6059|±  |0.0344|
|  - high_school_computer_science                       |      1|none  |     0|acc                    |↑  | 0.7200|±  |0.0451|
|  - high_school_mathematics                            |      1|none  |     0|acc                    |↑  | 0.3963|±  |0.0298|
|  - high_school_physics                                |      1|none  |     0|acc                    |↑  | 0.4702|±  |0.0408|
|  - high_school_statistics                             |      1|none  |     0|acc                    |↑  | 0.5417|±  |0.0340|
|  - machine_learning                                   |      1|none  |     0|acc                    |↑  | 0.4464|±  |0.0472|
|openbookqa                                             |      1|none  |     0|acc                    |↑  | 0.3000|±  |0.0205|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3920|±  |0.0219|
|piqa                                                   |      1|none  |     0|acc                    |↑  | 0.7557|±  |0.0100|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7552|±  |0.0100|
|social_iqa                                             |      0|none  |     0|acc                    |↑  | 0.4841|±  |0.0113|
|truthfulqa_gen                                         |      3|none  |     0|bleu_acc               |↑  | 0.0526|±  |0.0078|
|                                                       |       |none  |     0|bleu_diff              |↑  |-0.0803|±  |0.0273|
|                                                       |       |none  |     0|bleu_max               |↑  | 0.2757|±  |0.0400|
|                                                       |       |none  |     0|rouge1_acc             |↑  | 0.0612|±  |0.0084|
|                                                       |       |none  |     0|rouge1_diff            |↑  |-0.1183|±  |0.0544|
|                                                       |       |none  |     0|rouge1_max             |↑  | 1.1804|±  |0.1339|
|                                                       |       |none  |     0|rouge2_acc             |↑  | 0.0428|±  |0.0071|
|                                                       |       |none  |     0|rouge2_diff            |↑  |-0.2008|±  |0.0563|
|                                                       |       |none  |     0|rouge2_max             |↑  | 0.6279|±  |0.0799|
|                                                       |       |none  |     0|rougeL_acc             |↑  | 0.0539|±  |0.0079|
|                                                       |       |none  |     0|rougeL_diff            |↑  |-0.1430|±  |0.0557|
|                                                       |       |none  |     0|rougeL_max             |↑  | 1.0829|±  |0.1217|
|truthfulqa_mc1                                         |      2|none  |     0|acc                    |↑  | 0.3170|±  |0.0163|
|truthfulqa_mc2                                         |      3|none  |     0|acc                    |↑  | 0.4856|±  |0.0147|
|winogrande                                             |      1|none  |     0|acc                    |↑  | 0.6401|±  |0.0135|
|niah_single_1|      1|none  |     0|  1024|   |1.000|±  |     0|
|             |       |none  |     0| 16384|↑  |0.186|±  |   N/A|
|             |       |none  |     0|  2048|   |1.000|±  |     0|
|             |       |none  |     0| 32768|↑  |0.082|±  |   N/A|
|             |       |none  |     0|  4096|↑  |0.998|±  |   N/A|
|             |       |none  |     0|  8192|↑  |0.712|±  |   N/A|
|niah_single_2|      1|none  |     0|  1024|   |1.000|±  |0.0000|
|             |       |none  |     0| 16384|↑  |0.304|±  |   N/A|
|             |       |none  |     0|  2048|   |0.998|±  |0.0020|
|             |       |none  |     0| 32768|↑  |0.160|±  |   N/A|
|             |       |none  |     0|  4096|↑  |0.936|±  |   N/A|
|             |       |none  |     0|  8192|↑  |0.542|±  |   N/A|
|niah_single_3|      1|none  |     0|  1024|   |1.000|±  |0.0000|
|             |       |none  |     0| 16384|↑  |0.246|±  |   N/A|
|             |       |none  |     0|  2048|   |1.000|±  |0.0000|
|             |       |none  |     0| 32768|↑  |0.096|±  |   N/A|
|             |       |none  |     0|  4096|↑  |0.976|±  |   N/A|
|             |       |none  |     0|  8192|↑  |0.408|±  |   N/A|
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