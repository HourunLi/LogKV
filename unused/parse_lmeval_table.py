import pandas as pd
import numpy as np
import io

# 把你的各个模型的原始输出放在这里。这里以 Model_A 为例（填入你提供的原数据）
model_outputs = {
    "base-prolong": """
|                         Tasks                         |Version|Filter|n-shot|        Metric         |   | Value |   |Stderr|
|-------------------------------------------------------|------:|------|-----:|-----------------------|---|------:|---|------|
|arc_challenge                                          |      1|none  |     0|acc                    |↑  | 0.3123|±  |0.0135|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3618|±  |0.0140|
|arc_easy                                               |      1|none  |     0|acc                    |↑  | 0.6629|±  |0.0097|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6157|±  |0.0100|
|boolq                                                  |      2|none  |     0|acc                    |↑  | 0.6373|±  |0.0084|
|ceval-valid                                            |      2|none  |      |acc                    |↑  | 0.3098|±  |0.0126|
|                                                       |       |none  |      |acc_norm               |↑  | 0.3098|±  |0.0126|
| - ceval-valid_accountant                              |      2|none  |     0|acc                    |↑  | 0.2245|±  |0.0602|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.2245|±  |0.0602|
| - ceval-valid_advanced_mathematics                    |      2|none  |     0|acc                    |↑  | 0.1579|±  |0.0859|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.1579|±  |0.0859|
| - ceval-valid_art_studies                             |      2|none  |     0|acc                    |↑  | 0.2424|±  |0.0758|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.2424|±  |0.0758|
| - ceval-valid_basic_medicine                          |      2|none  |     0|acc                    |↑  | 0.4211|±  |0.1164|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4211|±  |0.1164|
| - ceval-valid_business_administration                 |      2|none  |     0|acc                    |↑  | 0.3636|±  |0.0850|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3636|±  |0.0850|
| - ceval-valid_chinese_language_and_literature         |      2|none  |     0|acc                    |↑  | 0.1739|±  |0.0808|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.1739|±  |0.0808|
| - ceval-valid_civil_servant                           |      2|none  |     0|acc                    |↑  | 0.2979|±  |0.0674|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.2979|±  |0.0674|
| - ceval-valid_clinical_medicine                       |      2|none  |     0|acc                    |↑  | 0.3182|±  |0.1016|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3182|±  |0.1016|
| - ceval-valid_college_chemistry                       |      2|none  |     0|acc                    |↑  | 0.3750|±  |0.1009|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3750|±  |0.1009|
| - ceval-valid_college_economics                       |      2|none  |     0|acc                    |↑  | 0.2727|±  |0.0606|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.2727|±  |0.0606|
| - ceval-valid_college_physics                         |      2|none  |     0|acc                    |↑  | 0.2105|±  |0.0961|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.2105|±  |0.0961|
| - ceval-valid_college_programming                     |      2|none  |     0|acc                    |↑  | 0.2162|±  |0.0686|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.2162|±  |0.0686|
| - ceval-valid_computer_architecture                   |      2|none  |     0|acc                    |↑  | 0.4286|±  |0.1107|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4286|±  |0.1107|
| - ceval-valid_computer_network                        |      2|none  |     0|acc                    |↑  | 0.4211|±  |0.1164|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4211|±  |0.1164|
| - ceval-valid_discrete_mathematics                    |      2|none  |     0|acc                    |↑  | 0.1875|±  |0.1008|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.1875|±  |0.1008|
| - ceval-valid_education_science                       |      2|none  |     0|acc                    |↑  | 0.4828|±  |0.0944|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4828|±  |0.0944|
| - ceval-valid_electrical_engineer                     |      2|none  |     0|acc                    |↑  | 0.3514|±  |0.0796|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3514|±  |0.0796|
| - ceval-valid_environmental_impact_assessment_engineer|      2|none  |     0|acc                    |↑  | 0.3226|±  |0.0853|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3226|±  |0.0853|
| - ceval-valid_fire_engineer                           |      2|none  |     0|acc                    |↑  | 0.3871|±  |0.0889|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3871|±  |0.0889|
| - ceval-valid_high_school_biology                     |      2|none  |     0|acc                    |↑  | 0.1579|±  |0.0859|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.1579|±  |0.0859|
| - ceval-valid_high_school_chemistry                   |      2|none  |     0|acc                    |↑  | 0.3684|±  |0.1137|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3684|±  |0.1137|
| - ceval-valid_high_school_chinese                     |      2|none  |     0|acc                    |↑  | 0.1579|±  |0.0859|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.1579|±  |0.0859|
| - ceval-valid_high_school_geography                   |      2|none  |     0|acc                    |↑  | 0.3158|±  |0.1096|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3158|±  |0.1096|
| - ceval-valid_high_school_history                     |      2|none  |     0|acc                    |↑  | 0.4000|±  |0.1124|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4000|±  |0.1124|
| - ceval-valid_high_school_mathematics                 |      2|none  |     0|acc                    |↑  | 0.2222|±  |0.1008|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.2222|±  |0.1008|
| - ceval-valid_high_school_physics                     |      2|none  |     0|acc                    |↑  | 0.2105|±  |0.0961|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.2105|±  |0.0961|
| - ceval-valid_high_school_politics                    |      2|none  |     0|acc                    |↑  | 0.3684|±  |0.1137|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3684|±  |0.1137|
| - ceval-valid_ideological_and_moral_cultivation       |      2|none  |     0|acc                    |↑  | 0.5263|±  |0.1177|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5263|±  |0.1177|
| - ceval-valid_law                                     |      2|none  |     0|acc                    |↑  | 0.3333|±  |0.0983|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3333|±  |0.0983|
| - ceval-valid_legal_professional                      |      2|none  |     0|acc                    |↑  | 0.3478|±  |0.1015|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3478|±  |0.1015|
| - ceval-valid_logic                                   |      2|none  |     0|acc                    |↑  | 0.3182|±  |0.1016|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3182|±  |0.1016|
| - ceval-valid_mao_zedong_thought                      |      2|none  |     0|acc                    |↑  | 0.2917|±  |0.0948|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.2917|±  |0.0948|
| - ceval-valid_marxism                                 |      2|none  |     0|acc                    |↑  | 0.4211|±  |0.1164|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4211|±  |0.1164|
| - ceval-valid_metrology_engineer                      |      2|none  |     0|acc                    |↑  | 0.2500|±  |0.0903|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.2500|±  |0.0903|
| - ceval-valid_middle_school_biology                   |      2|none  |     0|acc                    |↑  | 0.3333|±  |0.1054|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3333|±  |0.1054|
| - ceval-valid_middle_school_chemistry                 |      2|none  |     0|acc                    |↑  | 0.4500|±  |0.1141|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4500|±  |0.1141|
| - ceval-valid_middle_school_geography                 |      2|none  |     0|acc                    |↑  | 0.3333|±  |0.1421|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3333|±  |0.1421|
| - ceval-valid_middle_school_history                   |      2|none  |     0|acc                    |↑  | 0.2727|±  |0.0972|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.2727|±  |0.0972|
| - ceval-valid_middle_school_mathematics               |      2|none  |     0|acc                    |↑  | 0.3158|±  |0.1096|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3158|±  |0.1096|
| - ceval-valid_middle_school_physics                   |      2|none  |     0|acc                    |↑  | 0.1579|±  |0.0859|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.1579|±  |0.0859|
| - ceval-valid_middle_school_politics                  |      2|none  |     0|acc                    |↑  | 0.3810|±  |0.1086|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3810|±  |0.1086|
| - ceval-valid_modern_chinese_history                  |      2|none  |     0|acc                    |↑  | 0.3913|±  |0.1041|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3913|±  |0.1041|
| - ceval-valid_operating_system                        |      2|none  |     0|acc                    |↑  | 0.3158|±  |0.1096|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3158|±  |0.1096|
| - ceval-valid_physician                               |      2|none  |     0|acc                    |↑  | 0.3469|±  |0.0687|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3469|±  |0.0687|
| - ceval-valid_plant_protection                        |      2|none  |     0|acc                    |↑  | 0.2727|±  |0.0972|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.2727|±  |0.0972|
| - ceval-valid_probability_and_statistics              |      2|none  |     0|acc                    |↑  | 0.2778|±  |0.1086|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.2778|±  |0.1086|
| - ceval-valid_professional_tour_guide                 |      2|none  |     0|acc                    |↑  | 0.2414|±  |0.0809|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.2414|±  |0.0809|
| - ceval-valid_sports_science                          |      2|none  |     0|acc                    |↑  | 0.3158|±  |0.1096|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3158|±  |0.1096|
| - ceval-valid_tax_accountant                          |      2|none  |     0|acc                    |↑  | 0.2245|±  |0.0602|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.2245|±  |0.0602|
| - ceval-valid_teacher_qualification                   |      2|none  |     0|acc                    |↑  | 0.3409|±  |0.0723|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3409|±  |0.0723|
| - ceval-valid_urban_and_rural_planner                 |      2|none  |     0|acc                    |↑  | 0.2826|±  |0.0671|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.2826|±  |0.0671|
| - ceval-valid_veterinary_medicine                     |      2|none  |     0|acc                    |↑  | 0.4783|±  |0.1065|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4783|±  |0.1065|
|hellaswag                                              |      1|none  |     0|acc                    |↑  | 0.4798|±  |0.0050|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6423|±  |0.0048|
|ifeval                                                 |      4|none  |     0|inst_level_loose_acc   |↑  | 0.1571|±  |   N/A|
|                                                       |       |none  |     0|inst_level_strict_acc  |↑  | 0.1451|±  |   N/A|
|                                                       |       |none  |     0|prompt_level_loose_acc |↑  | 0.1091|±  |0.0134|
|                                                       |       |none  |     0|prompt_level_strict_acc|↑  | 0.0961|±  |0.0127|
|longbench_2wikimqa                                     |      5|none  |     0|qa_f1_score            |↑  | 0.0946|±  |0.0084|
|                                                       |       |none  |     0|score                  |↑  | 0.0946|±  |0.0084|
|longbench_2wikimqa_e                                   |      5|none  |     0|qa_f1_score            |↑  | 0.0880|±  |0.0070|
|                                                       |       |none  |     0|score                  |↑  | 0.0880|±  |0.0070|
|longbench_dureader                                     |      5|none  |     0|rouge_zh_score         |↑  | 0.1609|±  |0.0131|
|                                                       |       |none  |     0|score                  |↑  | 0.1609|±  |0.0131|
|longbench_gov_report                                   |      5|none  |     0|rouge_score            |↑  | 0.2311|±  |0.0106|
|                                                       |       |none  |     0|score                  |↑  | 0.2311|±  |0.0106|
|longbench_gov_report_e                                 |      5|none  |     0|rouge_score            |↑  | 0.2333|±  |0.0082|
|                                                       |       |none  |     0|score                  |↑  | 0.2333|±  |0.0082|
|longbench_hotpotqa                                     |      5|none  |     0|qa_f1_score            |↑  | 0.0692|±  |0.0075|
|                                                       |       |none  |     0|score                  |↑  | 0.0692|±  |0.0075|
|longbench_hotpotqa_e                                   |      5|none  |     0|qa_f1_score            |↑  | 0.0742|±  |0.0057|
|                                                       |       |none  |     0|score                  |↑  | 0.0742|±  |0.0057|
|longbench_lcc                                          |      5|none  |     0|code_sim_score         |↑  | 0.1876|±  |0.0118|
|                                                       |       |none  |     0|score                  |↑  | 0.1876|±  |0.0118|
|longbench_lcc_e                                        |      5|none  |     0|code_sim_score         |↑  | 0.2458|±  |0.0167|
|                                                       |       |none  |     0|score                  |↑  | 0.2458|±  |0.0167|
|longbench_lsht                                         |      5|none  |     0|classification_score   |↑  | 0.3692|±  |0.0335|
|                                                       |       |none  |     0|score                  |↑  | 0.3692|±  |0.0335|
|longbench_multi_news                                   |      5|none  |     0|rouge_score            |↑  | 0.2295|±  |0.0079|
|                                                       |       |none  |     0|score                  |↑  | 0.2295|±  |0.0079|
|longbench_multi_news_e                                 |      5|none  |     0|rouge_score            |↑  | 0.1975|±  |0.0064|
|                                                       |       |none  |     0|score                  |↑  | 0.1975|±  |0.0064|
|longbench_multifieldqa_en                              |      5|none  |     0|qa_f1_score            |↑  | 0.2066|±  |0.0151|
|                                                       |       |none  |     0|score                  |↑  | 0.2066|±  |0.0151|
|longbench_multifieldqa_en_e                            |      5|none  |     0|qa_f1_score            |↑  | 0.2066|±  |0.0151|
|                                                       |       |none  |     0|score                  |↑  | 0.2066|±  |0.0151|
|longbench_multifieldqa_zh                              |      5|none  |     0|qa_f1_zh_score         |↑  | 0.1390|±  |0.0111|
|                                                       |       |none  |     0|score                  |↑  | 0.1390|±  |0.0111|
|longbench_musique                                      |      5|none  |     0|qa_f1_score            |↑  | 0.0510|±  |0.0064|
|                                                       |       |none  |     0|score                  |↑  | 0.0510|±  |0.0064|
|longbench_narrativeqa                                  |      5|none  |     0|qa_f1_score            |↑  | 0.0294|±  |0.0038|
|                                                       |       |none  |     0|score                  |↑  | 0.0294|±  |0.0038|
|longbench_passage_count                                |      5|none  |     0|count_score            |↑  | 0.0076|±  |0.0053|
|                                                       |       |none  |     0|score                  |↑  | 0.0076|±  |0.0053|
|longbench_passage_count_e                              |      5|none  |     0|count_score            |↑  | 0.0318|±  |0.0073|
|                                                       |       |none  |     0|score                  |↑  | 0.0318|±  |0.0073|
|longbench_passage_retrieval_en                         |      5|none  |     0|retrieval_score        |↑  | 0.0475|±  |0.0148|
|                                                       |       |none  |     0|score                  |↑  | 0.0475|±  |0.0148|
|longbench_passage_retrieval_en_e                       |      5|none  |     0|retrieval_score        |↑  | 0.0658|±  |0.0134|
|                                                       |       |none  |     0|score                  |↑  | 0.0658|±  |0.0134|
|longbench_qasper                                       |      5|none  |     0|qa_f1_score            |↑  | 0.1052|±  |0.0092|
|                                                       |       |none  |     0|score                  |↑  | 0.1052|±  |0.0092|
|longbench_qasper_e                                     |      5|none  |     0|qa_f1_score            |↑  | 0.0838|±  |0.0067|
|                                                       |       |none  |     0|score                  |↑  | 0.0838|±  |0.0067|
|longbench_qmsum                                        |      5|none  |     0|rouge_score            |↑  | 0.1876|±  |0.0064|
|                                                       |       |none  |     0|score                  |↑  | 0.1876|±  |0.0064|
|longbench_repobench-p                                  |      5|none  |     0|code_sim_score         |↑  | 0.2750|±  |0.0134|
|                                                       |       |none  |     0|score                  |↑  | 0.2750|±  |0.0134|
|longbench_repobench-p_e                                |      5|none  |     0|code_sim_score         |↑  | 0.2508|±  |0.0170|
|                                                       |       |none  |     0|score                  |↑  | 0.2508|±  |0.0170|
|longbench_samsum                                       |      5|none  |     0|rouge_score            |↑  | 0.2955|±  |0.0097|
|                                                       |       |none  |     0|score                  |↑  | 0.2955|±  |0.0097|
|longbench_samsum_e                                     |      5|none  |     0|rouge_score            |↑  | 0.2880|±  |0.0064|
|                                                       |       |none  |     0|score                  |↑  | 0.2880|±  |0.0064|
|longbench_trec                                         |      5|none  |     0|classification_score   |↑  | 0.3725|±  |0.0206|
|                                                       |       |none  |     0|score                  |↑  | 0.3725|±  |0.0206|
|longbench_trec_e                                       |      5|none  |     0|classification_score   |↑  | 0.3544|±  |0.0173|
|                                                       |       |none  |     0|score                  |↑  | 0.3544|±  |0.0173|
|longbench_triviaqa                                     |      5|none  |     0|qa_f1_score            |↑  | 0.1740|±  |0.0071|
|                                                       |       |none  |     0|score                  |↑  | 0.1740|±  |0.0071|
|longbench_triviaqa_e                                   |      5|none  |     0|qa_f1_score            |↑  | 0.1600|±  |0.0056|
|                                                       |       |none  |     0|score                  |↑  | 0.1600|±  |0.0056|
|longbench_vcsum                                        |      5|none  |     0|rouge_zh_score         |↑  | 0.0080|±  |0.0018|
|                                                       |       |none  |     0|score                  |↑  | 0.0080|±  |0.0018|
|mmlu                                                   |      2|none  |      |acc                    |↑  | 0.3772|±  |0.0040|
| - humanities                                          |      2|none  |      |acc                    |↑  | 0.3498|±  |0.0068|
|  - formal_logic                                       |      1|none  |     0|acc                    |↑  | 0.2778|±  |0.0401|
|  - high_school_european_history                       |      1|none  |     0|acc                    |↑  | 0.4909|±  |0.0390|
|  - high_school_us_history                             |      1|none  |     0|acc                    |↑  | 0.4363|±  |0.0348|
|  - high_school_world_history                          |      1|none  |     0|acc                    |↑  | 0.4684|±  |0.0325|
|  - international_law                                  |      1|none  |     0|acc                    |↑  | 0.5289|±  |0.0456|
|  - jurisprudence                                      |      1|none  |     0|acc                    |↑  | 0.4537|±  |0.0481|
|  - logical_fallacies                                  |      1|none  |     0|acc                    |↑  | 0.3497|±  |0.0375|
|  - moral_disputes                                     |      1|none  |     0|acc                    |↑  | 0.3844|±  |0.0262|
|  - moral_scenarios                                    |      1|none  |     0|acc                    |↑  | 0.2380|±  |0.0142|
|  - philosophy                                         |      1|none  |     0|acc                    |↑  | 0.4180|±  |0.0280|
|  - prehistory                                         |      1|none  |     0|acc                    |↑  | 0.4198|±  |0.0275|
|  - professional_law                                   |      1|none  |     0|acc                    |↑  | 0.2986|±  |0.0117|
|  - world_religions                                    |      1|none  |     0|acc                    |↑  | 0.5263|±  |0.0383|
| - other                                               |      2|none  |      |acc                    |↑  | 0.4284|±  |0.0088|
|  - business_ethics                                    |      1|none  |     0|acc                    |↑  | 0.3700|±  |0.0485|
|  - clinical_knowledge                                 |      1|none  |     0|acc                    |↑  | 0.4038|±  |0.0302|
|  - college_medicine                                   |      1|none  |     0|acc                    |↑  | 0.3468|±  |0.0363|
|  - global_facts                                       |      1|none  |     0|acc                    |↑  | 0.3200|±  |0.0469|
|  - human_aging                                        |      1|none  |     0|acc                    |↑  | 0.4529|±  |0.0334|
|  - management                                         |      1|none  |     0|acc                    |↑  | 0.4466|±  |0.0492|
|  - marketing                                          |      1|none  |     0|acc                    |↑  | 0.5171|±  |0.0327|
|  - medical_genetics                                   |      1|none  |     0|acc                    |↑  | 0.4900|±  |0.0502|
|  - miscellaneous                                      |      1|none  |     0|acc                    |↑  | 0.5134|±  |0.0179|
|  - nutrition                                          |      1|none  |     0|acc                    |↑  | 0.4150|±  |0.0282|
|  - professional_accounting                            |      1|none  |     0|acc                    |↑  | 0.2766|±  |0.0267|
|  - professional_medicine                              |      1|none  |     0|acc                    |↑  | 0.3787|±  |0.0295|
|  - virology                                           |      1|none  |     0|acc                    |↑  | 0.4096|±  |0.0383|
| - social sciences                                     |      2|none  |      |acc                    |↑  | 0.4062|±  |0.0087|
|  - econometrics                                       |      1|none  |     0|acc                    |↑  | 0.1842|±  |0.0365|
|  - high_school_geography                              |      1|none  |     0|acc                    |↑  | 0.4848|±  |0.0356|
|  - high_school_government_and_politics                |      1|none  |     0|acc                    |↑  | 0.4560|±  |0.0359|
|  - high_school_macroeconomics                         |      1|none  |     0|acc                    |↑  | 0.3333|±  |0.0239|
|  - high_school_microeconomics                         |      1|none  |     0|acc                    |↑  | 0.3277|±  |0.0305|
|  - high_school_psychology                             |      1|none  |     0|acc                    |↑  | 0.4826|±  |0.0214|
|  - human_sexuality                                    |      1|none  |     0|acc                    |↑  | 0.4427|±  |0.0436|
|  - professional_psychology                            |      1|none  |     0|acc                    |↑  | 0.3513|±  |0.0193|
|  - public_relations                                   |      1|none  |     0|acc                    |↑  | 0.4182|±  |0.0472|
|  - security_studies                                   |      1|none  |     0|acc                    |↑  | 0.3796|±  |0.0311|
|  - sociology                                          |      1|none  |     0|acc                    |↑  | 0.5473|±  |0.0352|
|  - us_foreign_policy                                  |      1|none  |     0|acc                    |↑  | 0.5200|±  |0.0502|
| - stem                                                |      2|none  |      |acc                    |↑  | 0.3390|±  |0.0083|
|  - abstract_algebra                                   |      1|none  |     0|acc                    |↑  | 0.2800|±  |0.0451|
|  - anatomy                                            |      1|none  |     0|acc                    |↑  | 0.4741|±  |0.0431|
|  - astronomy                                          |      1|none  |     0|acc                    |↑  | 0.4276|±  |0.0403|
|  - college_biology                                    |      1|none  |     0|acc                    |↑  | 0.3889|±  |0.0408|
|  - college_chemistry                                  |      1|none  |     0|acc                    |↑  | 0.2700|±  |0.0446|
|  - college_computer_science                           |      1|none  |     0|acc                    |↑  | 0.3500|±  |0.0479|
|  - college_mathematics                                |      1|none  |     0|acc                    |↑  | 0.3400|±  |0.0476|
|  - college_physics                                    |      1|none  |     0|acc                    |↑  | 0.2647|±  |0.0439|
|  - computer_security                                  |      1|none  |     0|acc                    |↑  | 0.5500|±  |0.0500|
|  - conceptual_physics                                 |      1|none  |     0|acc                    |↑  | 0.3660|±  |0.0315|
|  - electrical_engineering                             |      1|none  |     0|acc                    |↑  | 0.3931|±  |0.0407|
|  - elementary_mathematics                             |      1|none  |     0|acc                    |↑  | 0.2646|±  |0.0227|
|  - high_school_biology                                |      1|none  |     0|acc                    |↑  | 0.4194|±  |0.0281|
|  - high_school_chemistry                              |      1|none  |     0|acc                    |↑  | 0.2808|±  |0.0316|
|  - high_school_computer_science                       |      1|none  |     0|acc                    |↑  | 0.3900|±  |0.0490|
|  - high_school_mathematics                            |      1|none  |     0|acc                    |↑  | 0.2481|±  |0.0263|
|  - high_school_physics                                |      1|none  |     0|acc                    |↑  | 0.2649|±  |0.0360|
|  - high_school_statistics                             |      1|none  |     0|acc                    |↑  | 0.2778|±  |0.0305|
|  - machine_learning                                   |      1|none  |     0|acc                    |↑  | 0.3750|±  |0.0460|
|openbookqa                                             |      1|none  |     0|acc                    |↑  | 0.2740|±  |0.0200|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3680|±  |0.0216|
|piqa                                                   |      1|none  |     0|acc                    |↑  | 0.7524|±  |0.0101|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7443|±  |0.0102|
|social_iqa                                             |      0|none  |     0|acc                    |↑  | 0.4314|±  |0.0112|
|truthfulqa_gen                                         |      3|none  |     0|bleu_acc               |↑  | 0.0612|±  |0.0084|
|                                                       |       |none  |     0|bleu_diff              |↑  |-0.0655|±  |0.0139|
|                                                       |       |none  |     0|bleu_max               |↑  | 0.2000|±  |0.0242|
|                                                       |       |none  |     0|rouge1_acc             |↑  | 0.0759|±  |0.0093|
|                                                       |       |none  |     0|rouge1_diff            |↑  |-0.1028|±  |0.0351|
|                                                       |       |none  |     0|rouge1_max             |↑  | 0.9573|±  |0.0805|
|                                                       |       |none  |     0|rouge2_acc             |↑  | 0.0465|±  |0.0074|
|                                                       |       |none  |     0|rouge2_diff            |↑  |-0.1500|±  |0.0318|
|                                                       |       |none  |     0|rouge2_max             |↑  | 0.4400|±  |0.0549|
|                                                       |       |none  |     0|rougeL_acc             |↑  | 0.0734|±  |0.0091|
|                                                       |       |none  |     0|rougeL_diff            |↑  |-0.1151|±  |0.0334|
|                                                       |       |none  |     0|rougeL_max             |↑  | 0.8797|±  |0.0757|
|truthfulqa_mc1                                         |      2|none  |     0|acc                    |↑  | 0.2350|±  |0.0148|
|truthfulqa_mc2                                         |      3|none  |     0|acc                    |↑  | 0.3848|±  |0.0134|
|winogrande                                             |      1|none  |     0|acc                    |↑  | 0.6077|±  |0.0137|
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
    "WinoGrande": "acc", "ARC-e": "acc_norm", "ARC-c": "acc_norm", "OpenBookQA": "acc_norm",
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