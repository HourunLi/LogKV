import pandas as pd
import numpy as np
import io

# 把你的各个模型的原始输出放在这里。这里以 Model_A 为例（填入你提供的原数据）
model_outputs = {
    "base-prolong": """
|                         Tasks                         |Version|Filter|n-shot|        Metric         |   | Value |   |Stderr|
|-------------------------------------------------------|------:|------|-----:|-----------------------|---|------:|---|------|
|arc_challenge                                          |      1|none  |     0|acc                    |↑  | 0.4445|±  |0.0145|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4701|±  |0.0146|
|arc_easy                                               |      1|none  |     0|acc                    |↑  | 0.7731|±  |0.0086|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7580|±  |0.0088|
|boolq                                                  |      2|none  |     0|acc                    |↑  | 0.7541|±  |0.0075|
|ceval-valid                                            |      2|none  |      |acc                    |↑  | 0.6367|±  |0.0127|
|                                                       |       |none  |      |acc_norm               |↑  | 0.6367|±  |0.0127|
| - ceval-valid_accountant                              |      2|none  |     0|acc                    |↑  | 0.5306|±  |0.0720|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5306|±  |0.0720|
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
| - ceval-valid_college_chemistry                       |      2|none  |     0|acc                    |↑  | 0.5417|±  |0.1039|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5417|±  |0.1039|
| - ceval-valid_college_economics                       |      2|none  |     0|acc                    |↑  | 0.5636|±  |0.0675|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5636|±  |0.0675|
| - ceval-valid_college_physics                         |      2|none  |     0|acc                    |↑  | 0.4737|±  |0.1177|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4737|±  |0.1177|
| - ceval-valid_college_programming                     |      2|none  |     0|acc                    |↑  | 0.6757|±  |0.0780|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6757|±  |0.0780|
| - ceval-valid_computer_architecture                   |      2|none  |     0|acc                    |↑  | 0.7619|±  |0.0952|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7619|±  |0.0952|
| - ceval-valid_computer_network                        |      2|none  |     0|acc                    |↑  | 0.5263|±  |0.1177|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5263|±  |0.1177|
| - ceval-valid_discrete_mathematics                    |      2|none  |     0|acc                    |↑  | 0.2500|±  |0.1118|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.2500|±  |0.1118|
| - ceval-valid_education_science                       |      2|none  |     0|acc                    |↑  | 0.7586|±  |0.0809|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7586|±  |0.0809|
| - ceval-valid_electrical_engineer                     |      2|none  |     0|acc                    |↑  | 0.3784|±  |0.0808|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3784|±  |0.0808|
| - ceval-valid_environmental_impact_assessment_engineer|      2|none  |     0|acc                    |↑  | 0.6774|±  |0.0853|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6774|±  |0.0853|
| - ceval-valid_fire_engineer                           |      2|none  |     0|acc                    |↑  | 0.5484|±  |0.0909|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5484|±  |0.0909|
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
| - ceval-valid_logic                                   |      2|none  |     0|acc                    |↑  | 0.6364|±  |0.1050|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6364|±  |0.1050|
| - ceval-valid_mao_zedong_thought                      |      2|none  |     0|acc                    |↑  | 0.8750|±  |0.0690|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8750|±  |0.0690|
| - ceval-valid_marxism                                 |      2|none  |     0|acc                    |↑  | 0.7895|±  |0.0961|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7895|±  |0.0961|
| - ceval-valid_metrology_engineer                      |      2|none  |     0|acc                    |↑  | 0.7500|±  |0.0903|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7500|±  |0.0903|
| - ceval-valid_middle_school_biology                   |      2|none  |     0|acc                    |↑  | 0.9524|±  |0.0476|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.9524|±  |0.0476|
| - ceval-valid_middle_school_chemistry                 |      2|none  |     0|acc                    |↑  | 0.9000|±  |0.0688|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.9000|±  |0.0688|
| - ceval-valid_middle_school_geography                 |      2|none  |     0|acc                    |↑  | 0.5833|±  |0.1486|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5833|±  |0.1486|
| - ceval-valid_middle_school_history                   |      2|none  |     0|acc                    |↑  | 0.9545|±  |0.0455|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.9545|±  |0.0455|
| - ceval-valid_middle_school_mathematics               |      2|none  |     0|acc                    |↑  | 0.5263|±  |0.1177|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5263|±  |0.1177|
| - ceval-valid_middle_school_physics                   |      2|none  |     0|acc                    |↑  | 0.7368|±  |0.1038|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7368|±  |0.1038|
| - ceval-valid_middle_school_politics                  |      2|none  |     0|acc                    |↑  | 0.8571|±  |0.0782|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8571|±  |0.0782|
| - ceval-valid_modern_chinese_history                  |      2|none  |     0|acc                    |↑  | 0.7826|±  |0.0879|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7826|±  |0.0879|
| - ceval-valid_operating_system                        |      2|none  |     0|acc                    |↑  | 0.6316|±  |0.1137|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6316|±  |0.1137|
| - ceval-valid_physician                               |      2|none  |     0|acc                    |↑  | 0.6735|±  |0.0677|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6735|±  |0.0677|
| - ceval-valid_plant_protection                        |      2|none  |     0|acc                    |↑  | 0.8182|±  |0.0842|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8182|±  |0.0842|
| - ceval-valid_probability_and_statistics              |      2|none  |     0|acc                    |↑  | 0.3333|±  |0.1143|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3333|±  |0.1143|
| - ceval-valid_professional_tour_guide                 |      2|none  |     0|acc                    |↑  | 0.5862|±  |0.0931|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5862|±  |0.0931|
| - ceval-valid_sports_science                          |      2|none  |     0|acc                    |↑  | 0.6842|±  |0.1096|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6842|±  |0.1096|
| - ceval-valid_tax_accountant                          |      2|none  |     0|acc                    |↑  | 0.5714|±  |0.0714|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5714|±  |0.0714|
| - ceval-valid_teacher_qualification                   |      2|none  |     0|acc                    |↑  | 0.7727|±  |0.0639|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7727|±  |0.0639|
| - ceval-valid_urban_and_rural_planner                 |      2|none  |     0|acc                    |↑  | 0.6522|±  |0.0710|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6522|±  |0.0710|
| - ceval-valid_veterinary_medicine                     |      2|none  |     0|acc                    |↑  | 0.6957|±  |0.0981|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6957|±  |0.0981|
|hellaswag                                              |      1|none  |     0|acc                    |↑  | 0.4830|±  |0.0050|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6509|±  |0.0048|
|ifeval                                                 |      4|none  |     0|inst_level_loose_acc   |↑  | 0.2890|±  |   N/A|
|                                                       |       |none  |     0|inst_level_strict_acc  |↑  | 0.2638|±  |   N/A|
|                                                       |       |none  |     0|prompt_level_loose_acc |↑  | 0.1738|±  |0.0163|
|                                                       |       |none  |     0|prompt_level_strict_acc|↑  | 0.1497|±  |0.0154|
|longbench_2wikimqa                                     |      5|none  |     0|qa_f1_score            |↑  | 0.1158|±  |0.0119|
|                                                       |       |none  |     0|score                  |↑  | 0.1158|±  |0.0119|
|longbench_2wikimqa_e                                   |      5|none  |     0|qa_f1_score            |↑  | 0.1061|±  |0.0086|
|                                                       |       |none  |     0|score                  |↑  | 0.1061|±  |0.0086|
|longbench_dureader                                     |      5|none  |     0|rouge_zh_score         |↑  | 0.1844|±  |0.0088|
|                                                       |       |none  |     0|score                  |↑  | 0.1844|±  |0.0088|
|longbench_gov_report                                   |      5|none  |     0|rouge_score            |↑  | 0.2261|±  |0.0063|
|                                                       |       |none  |     0|score                  |↑  | 0.2261|±  |0.0063|
|longbench_gov_report_e                                 |      5|none  |     0|rouge_score            |↑  | 0.2427|±  |0.0057|
|                                                       |       |none  |     0|score                  |↑  | 0.2427|±  |0.0057|
|longbench_hotpotqa                                     |      5|none  |     0|qa_f1_score            |↑  | 0.0840|±  |0.0087|
|                                                       |       |none  |     0|score                  |↑  | 0.0840|±  |0.0087|
|longbench_hotpotqa_e                                   |      5|none  |     0|qa_f1_score            |↑  | 0.0999|±  |0.0096|
|                                                       |       |none  |     0|score                  |↑  | 0.0999|±  |0.0096|
|longbench_lcc                                          |      5|none  |     0|code_sim_score         |↑  | 0.0584|±  |0.0047|
|                                                       |       |none  |     0|score                  |↑  | 0.0584|±  |0.0047|
|longbench_lcc_e                                        |      5|none  |     0|code_sim_score         |↑  | 0.1586|±  |0.0139|
|                                                       |       |none  |     0|score                  |↑  | 0.1586|±  |0.0139|
|longbench_lsht                                         |      5|none  |     0|classification_score   |↑  | 0.2175|±  |0.0267|
|                                                       |       |none  |     0|score                  |↑  | 0.2175|±  |0.0267|
|longbench_multi_news                                   |      5|none  |     0|rouge_score            |↑  | 0.2220|±  |0.0086|
|                                                       |       |none  |     0|score                  |↑  | 0.2220|±  |0.0086|
|longbench_multi_news_e                                 |      5|none  |     0|rouge_score            |↑  | 0.1817|±  |0.0071|
|                                                       |       |none  |     0|score                  |↑  | 0.1817|±  |0.0071|
|longbench_multifieldqa_en                              |      5|none  |     0|qa_f1_score            |↑  | 0.3188|±  |0.0228|
|                                                       |       |none  |     0|score                  |↑  | 0.3188|±  |0.0228|
|longbench_multifieldqa_en_e                            |      5|none  |     0|qa_f1_score            |↑  | 0.3188|±  |0.0228|
|                                                       |       |none  |     0|score                  |↑  | 0.3188|±  |0.0228|
|longbench_multifieldqa_zh                              |      5|none  |     0|qa_f1_zh_score         |↑  | 0.2697|±  |0.0179|
|                                                       |       |none  |     0|score                  |↑  | 0.2697|±  |0.0179|
|longbench_musique                                      |      5|none  |     0|qa_f1_score            |↑  | 0.0432|±  |0.0053|
|                                                       |       |none  |     0|score                  |↑  | 0.0432|±  |0.0053|
|longbench_narrativeqa                                  |      5|none  |     0|qa_f1_score            |↑  | 0.0253|±  |0.0023|
|                                                       |       |none  |     0|score                  |↑  | 0.0253|±  |0.0023|
|longbench_passage_count                                |      5|none  |     0|count_score            |↑  | 0.0151|±  |0.0056|
|                                                       |       |none  |     0|score                  |↑  | 0.0151|±  |0.0056|
|longbench_passage_count_e                              |      5|none  |     0|count_score            |↑  | 0.0453|±  |0.0116|
|                                                       |       |none  |     0|score                  |↑  | 0.0453|±  |0.0116|
|longbench_passage_retrieval_en                         |      5|none  |     0|retrieval_score        |↑  | 0.0967|±  |0.0202|
|                                                       |       |none  |     0|score                  |↑  | 0.0967|±  |0.0202|
|longbench_passage_retrieval_en_e                       |      5|none  |     0|retrieval_score        |↑  | 0.1443|±  |0.0200|
|                                                       |       |none  |     0|score                  |↑  | 0.1443|±  |0.0200|
|longbench_qasper                                       |      5|none  |     0|qa_f1_score            |↑  | 0.1793|±  |0.0141|
|                                                       |       |none  |     0|score                  |↑  | 0.1793|±  |0.0141|
|longbench_qasper_e                                     |      5|none  |     0|qa_f1_score            |↑  | 0.1617|±  |0.0125|
|                                                       |       |none  |     0|score                  |↑  | 0.1617|±  |0.0125|
|longbench_qmsum                                        |      5|none  |     0|rouge_score            |↑  | 0.2173|±  |0.0044|
|                                                       |       |none  |     0|score                  |↑  | 0.2173|±  |0.0044|
|longbench_repobench-p                                  |      5|none  |     0|code_sim_score         |↑  | 0.1593|±  |0.0107|
|                                                       |       |none  |     0|score                  |↑  | 0.1593|±  |0.0107|
|longbench_repobench-p_e                                |      5|none  |     0|code_sim_score         |↑  | 0.1522|±  |0.0132|
|                                                       |       |none  |     0|score                  |↑  | 0.1522|±  |0.0132|
|longbench_samsum                                       |      5|none  |     0|rouge_score            |↑  | 0.3172|±  |0.0085|
|                                                       |       |none  |     0|score                  |↑  | 0.3172|±  |0.0085|
|longbench_samsum_e                                     |      5|none  |     0|rouge_score            |↑  | 0.3075|±  |0.0066|
|                                                       |       |none  |     0|score                  |↑  | 0.3075|±  |0.0066|
|longbench_trec                                         |      5|none  |     0|classification_score   |↑  | 0.4258|±  |0.0230|
|                                                       |       |none  |     0|score                  |↑  | 0.4258|±  |0.0230|
|longbench_trec_e                                       |      5|none  |     0|classification_score   |↑  | 0.3794|±  |0.0190|
|                                                       |       |none  |     0|score                  |↑  | 0.3794|±  |0.0190|
|longbench_triviaqa                                     |      5|none  |     0|qa_f1_score            |↑  | 0.1992|±  |0.0072|
|                                                       |       |none  |     0|score                  |↑  | 0.1992|±  |0.0072|
|longbench_triviaqa_e                                   |      5|none  |     0|qa_f1_score            |↑  | 0.1958|±  |0.0061|
|                                                       |       |none  |     0|score                  |↑  | 0.1958|±  |0.0061|
|longbench_vcsum                                        |      5|none  |     0|rouge_zh_score         |↑  | 0.0574|±  |0.0049|
|                                                       |       |none  |     0|score                  |↑  | 0.0574|±  |0.0049|
|mmlu                                                   |      2|none  |      |acc                    |↑  | 0.6008|±  |0.0039|
| - humanities                                          |      2|none  |      |acc                    |↑  | 0.5116|±  |0.0068|
|  - formal_logic                                       |      1|none  |     0|acc                    |↑  | 0.4921|±  |0.0447|
|  - high_school_european_history                       |      1|none  |     0|acc                    |↑  | 0.6970|±  |0.0359|
|  - high_school_us_history                             |      1|none  |     0|acc                    |↑  | 0.7108|±  |0.0318|
|  - high_school_world_history                          |      1|none  |     0|acc                    |↑  | 0.7806|±  |0.0269|
|  - international_law                                  |      1|none  |     0|acc                    |↑  | 0.7769|±  |0.0380|
|  - jurisprudence                                      |      1|none  |     0|acc                    |↑  | 0.7130|±  |0.0437|
|  - logical_fallacies                                  |      1|none  |     0|acc                    |↑  | 0.7423|±  |0.0344|
|  - moral_disputes                                     |      1|none  |     0|acc                    |↑  | 0.6618|±  |0.0255|
|  - moral_scenarios                                    |      1|none  |     0|acc                    |↑  | 0.2536|±  |0.0146|
|  - philosophy                                         |      1|none  |     0|acc                    |↑  | 0.6431|±  |0.0272|
|  - prehistory                                         |      1|none  |     0|acc                    |↑  | 0.6420|±  |0.0267|
|  - professional_law                                   |      1|none  |     0|acc                    |↑  | 0.4029|±  |0.0125|
|  - world_religions                                    |      1|none  |     0|acc                    |↑  | 0.7368|±  |0.0338|
| - other                                               |      2|none  |      |acc                    |↑  | 0.6482|±  |0.0083|
|  - business_ethics                                    |      1|none  |     0|acc                    |↑  | 0.6300|±  |0.0485|
|  - clinical_knowledge                                 |      1|none  |     0|acc                    |↑  | 0.6377|±  |0.0296|
|  - college_medicine                                   |      1|none  |     0|acc                    |↑  | 0.5954|±  |0.0374|
|  - global_facts                                       |      1|none  |     0|acc                    |↑  | 0.3200|±  |0.0469|
|  - human_aging                                        |      1|none  |     0|acc                    |↑  | 0.6233|±  |0.0325|
|  - management                                         |      1|none  |     0|acc                    |↑  | 0.7670|±  |0.0419|
|  - marketing                                          |      1|none  |     0|acc                    |↑  | 0.8504|±  |0.0234|
|  - medical_genetics                                   |      1|none  |     0|acc                    |↑  | 0.7200|±  |0.0451|
|  - miscellaneous                                      |      1|none  |     0|acc                    |↑  | 0.7344|±  |0.0158|
|  - nutrition                                          |      1|none  |     0|acc                    |↑  | 0.6928|±  |0.0264|
|  - professional_accounting                            |      1|none  |     0|acc                    |↑  | 0.4504|±  |0.0297|
|  - professional_medicine                              |      1|none  |     0|acc                    |↑  | 0.5993|±  |0.0298|
|  - virology                                           |      1|none  |     0|acc                    |↑  | 0.4880|±  |0.0389|
| - social sciences                                     |      2|none  |      |acc                    |↑  | 0.7134|±  |0.0080|
|  - econometrics                                       |      1|none  |     0|acc                    |↑  | 0.4737|±  |0.0470|
|  - high_school_geography                              |      1|none  |     0|acc                    |↑  | 0.7828|±  |0.0294|
|  - high_school_government_and_politics                |      1|none  |     0|acc                    |↑  | 0.8031|±  |0.0287|
|  - high_school_macroeconomics                         |      1|none  |     0|acc                    |↑  | 0.6462|±  |0.0242|
|  - high_school_microeconomics                         |      1|none  |     0|acc                    |↑  | 0.7353|±  |0.0287|
|  - high_school_psychology                             |      1|none  |     0|acc                    |↑  | 0.8275|±  |0.0162|
|  - human_sexuality                                    |      1|none  |     0|acc                    |↑  | 0.7328|±  |0.0388|
|  - professional_psychology                            |      1|none  |     0|acc                    |↑  | 0.6111|±  |0.0197|
|  - public_relations                                   |      1|none  |     0|acc                    |↑  | 0.5909|±  |0.0471|
|  - security_studies                                   |      1|none  |     0|acc                    |↑  | 0.7265|±  |0.0285|
|  - sociology                                          |      1|none  |     0|acc                    |↑  | 0.7861|±  |0.0290|
|  - us_foreign_policy                                  |      1|none  |     0|acc                    |↑  | 0.8200|±  |0.0386|
| - stem                                                |      2|none  |      |acc                    |↑  | 0.5775|±  |0.0085|
|  - abstract_algebra                                   |      1|none  |     0|acc                    |↑  | 0.3500|±  |0.0479|
|  - anatomy                                            |      1|none  |     0|acc                    |↑  | 0.5852|±  |0.0426|
|  - astronomy                                          |      1|none  |     0|acc                    |↑  | 0.7105|±  |0.0369|
|  - college_biology                                    |      1|none  |     0|acc                    |↑  | 0.7569|±  |0.0359|
|  - college_chemistry                                  |      1|none  |     0|acc                    |↑  | 0.4200|±  |0.0496|
|  - college_computer_science                           |      1|none  |     0|acc                    |↑  | 0.6100|±  |0.0490|
|  - college_mathematics                                |      1|none  |     0|acc                    |↑  | 0.3700|±  |0.0485|
|  - college_physics                                    |      1|none  |     0|acc                    |↑  | 0.4412|±  |0.0494|
|  - computer_security                                  |      1|none  |     0|acc                    |↑  | 0.8000|±  |0.0402|
|  - conceptual_physics                                 |      1|none  |     0|acc                    |↑  | 0.6255|±  |0.0316|
|  - electrical_engineering                             |      1|none  |     0|acc                    |↑  | 0.6069|±  |0.0407|
|  - elementary_mathematics                             |      1|none  |     0|acc                    |↑  | 0.5503|±  |0.0256|
|  - high_school_biology                                |      1|none  |     0|acc                    |↑  | 0.7774|±  |0.0237|
|  - high_school_chemistry                              |      1|none  |     0|acc                    |↑  | 0.5911|±  |0.0346|
|  - high_school_computer_science                       |      1|none  |     0|acc                    |↑  | 0.6900|±  |0.0465|
|  - high_school_mathematics                            |      1|none  |     0|acc                    |↑  | 0.4148|±  |0.0300|
|  - high_school_physics                                |      1|none  |     0|acc                    |↑  | 0.4437|±  |0.0406|
|  - high_school_statistics                             |      1|none  |     0|acc                    |↑  | 0.5648|±  |0.0338|
|  - machine_learning                                   |      1|none  |     0|acc                    |↑  | 0.4554|±  |0.0473|
|openbookqa                                             |      1|none  |     0|acc                    |↑  | 0.2980|±  |0.0205|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3920|±  |0.0219|
|piqa                                                   |      1|none  |     0|acc                    |↑  | 0.7524|±  |0.0101|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7595|±  |0.0100|
|social_iqa                                             |      0|none  |     0|acc                    |↑  | 0.4637|±  |0.0113|
|truthfulqa_gen                                         |      3|none  |     0|bleu_acc               |↑  | 0.4064|±  |0.0172|
|                                                       |       |none  |     0|bleu_diff              |↑  |-0.1285|±  |0.0498|
|                                                       |       |none  |     0|bleu_max               |↑  | 1.4129|±  |0.0857|
|                                                       |       |none  |     0|rouge1_acc             |↑  | 0.4443|±  |0.0174|
|                                                       |       |none  |     0|rouge1_diff            |↑  |-0.2295|±  |0.0874|
|                                                       |       |none  |     0|rouge1_max             |↑  | 5.3838|±  |0.1445|
|                                                       |       |none  |     0|rouge2_acc             |↑  | 0.3709|±  |0.0169|
|                                                       |       |none  |     0|rouge2_diff            |↑  |-0.2842|±  |0.0956|
|                                                       |       |none  |     0|rouge2_max             |↑  | 3.2740|±  |0.1307|
|                                                       |       |none  |     0|rougeL_acc             |↑  | 0.4272|±  |0.0173|
|                                                       |       |none  |     0|rougeL_diff            |↑  |-0.2513|±  |0.0872|
|                                                       |       |none  |     0|rougeL_max             |↑  | 5.1340|±  |0.1409|
|truthfulqa_mc1                                         |      2|none  |     0|acc                    |↑  | 0.2962|±  |0.0160|
|truthfulqa_mc2                                         |      3|none  |     0|acc                    |↑  | 0.4509|±  |0.0144|
|winogrande                                             |      1|none  |     0|acc                    |↑  | 0.6338|±  |0.0135|
|niah_single_1|      1|none  |     0|  1024|   |1.000|±  |     0|
|             |       |none  |     0| 16384|↑  |0.128|±  |   N/A|
|             |       |none  |     0|  2048|   |1.000|±  |     0|
|             |       |none  |     0| 32768|↑  |0.070|±  |   N/A|
|             |       |none  |     0|  4096|↑  |0.730|±  |   N/A|
|             |       |none  |     0|  8192|↑  |0.344|±  |   N/A|
|niah_single_2|      1|none  |     0|  1024|   |1.000|±  |0.0000|
|             |       |none  |     0| 16384|↑  |0.206|±  |   N/A|
|             |       |none  |     0|  2048|   |0.998|±  |0.0020|
|             |       |none  |     0| 32768|↑  |0.114|±  |   N/A|
|             |       |none  |     0|  4096|↑  |0.636|±  |   N/A|
|             |       |none  |     0|  8192|↑  |0.346|±  |   N/A|
|niah_single_3|      1|none  |     0|  1024|   |1.000|±  |0.0000|
|             |       |none  |     0| 16384|↑  |0.150|±  |   N/A|
|             |       |none  |     0|  2048|   |1.000|±  |0.0000|
|             |       |none  |     0| 32768|↑  |0.064|±  |   N/A|
|             |       |none  |     0|  4096|↑  |0.550|±  |   N/A|
|             |       |none  |     0|  8192|↑  |0.232|±  |   N/A|
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