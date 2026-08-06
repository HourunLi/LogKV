import pandas as pd
import numpy as np
import io

# 把你的各个模型的原始输出放在这里。这里以 Model_A 为例（填入你提供的原数据）
model_outputs = {
    "base-prolong": """
|                         Tasks                         |Version|Filter|n-shot|        Metric         |   | Value |   |Stderr|
|-------------------------------------------------------|------:|------|-----:|-----------------------|---|------:|---|------|
|arc_challenge                                          |      1|none  |     0|acc                    |↑  | 0.4249|±  |0.0144|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4505|±  |0.0145|
|arc_easy                                               |      1|none  |     0|acc                    |↑  | 0.7395|±  |0.0090|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7269|±  |0.0091|
|boolq                                                  |      2|none  |     0|acc                    |↑  | 0.7602|±  |0.0075|
|ceval-valid                                            |      2|none  |      |acc                    |↑  | 0.6003|±  |0.0129|
|                                                       |       |none  |      |acc_norm               |↑  | 0.6003|±  |0.0129|
| - ceval-valid_accountant                              |      2|none  |     0|acc                    |↑  | 0.4898|±  |0.0722|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4898|±  |0.0722|
| - ceval-valid_advanced_mathematics                    |      2|none  |     0|acc                    |↑  | 0.5263|±  |0.1177|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5263|±  |0.1177|
| - ceval-valid_art_studies                             |      2|none  |     0|acc                    |↑  | 0.6061|±  |0.0864|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6061|±  |0.0864|
| - ceval-valid_basic_medicine                          |      2|none  |     0|acc                    |↑  | 0.6316|±  |0.1137|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6316|±  |0.1137|
| - ceval-valid_business_administration                 |      2|none  |     0|acc                    |↑  | 0.6061|±  |0.0864|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6061|±  |0.0864|
| - ceval-valid_chinese_language_and_literature         |      2|none  |     0|acc                    |↑  | 0.5217|±  |0.1065|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5217|±  |0.1065|
| - ceval-valid_civil_servant                           |      2|none  |     0|acc                    |↑  | 0.4681|±  |0.0736|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4681|±  |0.0736|
| - ceval-valid_clinical_medicine                       |      2|none  |     0|acc                    |↑  | 0.5909|±  |0.1073|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5909|±  |0.1073|
| - ceval-valid_college_chemistry                       |      2|none  |     0|acc                    |↑  | 0.5833|±  |0.1028|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5833|±  |0.1028|
| - ceval-valid_college_economics                       |      2|none  |     0|acc                    |↑  | 0.5091|±  |0.0680|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5091|±  |0.0680|
| - ceval-valid_college_physics                         |      2|none  |     0|acc                    |↑  | 0.4737|±  |0.1177|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4737|±  |0.1177|
| - ceval-valid_college_programming                     |      2|none  |     0|acc                    |↑  | 0.5135|±  |0.0833|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5135|±  |0.0833|
| - ceval-valid_computer_architecture                   |      2|none  |     0|acc                    |↑  | 0.7619|±  |0.0952|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7619|±  |0.0952|
| - ceval-valid_computer_network                        |      2|none  |     0|acc                    |↑  | 0.5263|±  |0.1177|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5263|±  |0.1177|
| - ceval-valid_discrete_mathematics                    |      2|none  |     0|acc                    |↑  | 0.4375|±  |0.1281|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4375|±  |0.1281|
| - ceval-valid_education_science                       |      2|none  |     0|acc                    |↑  | 0.6207|±  |0.0917|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6207|±  |0.0917|
| - ceval-valid_electrical_engineer                     |      2|none  |     0|acc                    |↑  | 0.2973|±  |0.0762|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.2973|±  |0.0762|
| - ceval-valid_environmental_impact_assessment_engineer|      2|none  |     0|acc                    |↑  | 0.6774|±  |0.0853|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6774|±  |0.0853|
| - ceval-valid_fire_engineer                           |      2|none  |     0|acc                    |↑  | 0.4194|±  |0.0901|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4194|±  |0.0901|
| - ceval-valid_high_school_biology                     |      2|none  |     0|acc                    |↑  | 0.6316|±  |0.1137|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6316|±  |0.1137|
| - ceval-valid_high_school_chemistry                   |      2|none  |     0|acc                    |↑  | 0.4737|±  |0.1177|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4737|±  |0.1177|
| - ceval-valid_high_school_chinese                     |      2|none  |     0|acc                    |↑  | 0.4737|±  |0.1177|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4737|±  |0.1177|
| - ceval-valid_high_school_geography                   |      2|none  |     0|acc                    |↑  | 0.4211|±  |0.1164|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4211|±  |0.1164|
| - ceval-valid_high_school_history                     |      2|none  |     0|acc                    |↑  | 0.7500|±  |0.0993|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7500|±  |0.0993|
| - ceval-valid_high_school_mathematics                 |      2|none  |     0|acc                    |↑  | 0.3889|±  |0.1182|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3889|±  |0.1182|
| - ceval-valid_high_school_physics                     |      2|none  |     0|acc                    |↑  | 0.5263|±  |0.1177|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5263|±  |0.1177|
| - ceval-valid_high_school_politics                    |      2|none  |     0|acc                    |↑  | 0.8421|±  |0.0859|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8421|±  |0.0859|
| - ceval-valid_ideological_and_moral_cultivation       |      2|none  |     0|acc                    |↑  | 0.9474|±  |0.0526|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.9474|±  |0.0526|
| - ceval-valid_law                                     |      2|none  |     0|acc                    |↑  | 0.4167|±  |0.1028|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4167|±  |0.1028|
| - ceval-valid_legal_professional                      |      2|none  |     0|acc                    |↑  | 0.3478|±  |0.1015|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.3478|±  |0.1015|
| - ceval-valid_logic                                   |      2|none  |     0|acc                    |↑  | 0.5455|±  |0.1087|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5455|±  |0.1087|
| - ceval-valid_mao_zedong_thought                      |      2|none  |     0|acc                    |↑  | 0.8333|±  |0.0777|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8333|±  |0.0777|
| - ceval-valid_marxism                                 |      2|none  |     0|acc                    |↑  | 0.9474|±  |0.0526|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.9474|±  |0.0526|
| - ceval-valid_metrology_engineer                      |      2|none  |     0|acc                    |↑  | 0.6250|±  |0.1009|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6250|±  |0.1009|
| - ceval-valid_middle_school_biology                   |      2|none  |     0|acc                    |↑  | 0.9048|±  |0.0656|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.9048|±  |0.0656|
| - ceval-valid_middle_school_chemistry                 |      2|none  |     0|acc                    |↑  | 0.7500|±  |0.0993|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7500|±  |0.0993|
| - ceval-valid_middle_school_geography                 |      2|none  |     0|acc                    |↑  | 0.8333|±  |0.1124|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8333|±  |0.1124|
| - ceval-valid_middle_school_history                   |      2|none  |     0|acc                    |↑  | 0.9091|±  |0.0627|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.9091|±  |0.0627|
| - ceval-valid_middle_school_mathematics               |      2|none  |     0|acc                    |↑  | 0.6842|±  |0.1096|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6842|±  |0.1096|
| - ceval-valid_middle_school_physics                   |      2|none  |     0|acc                    |↑  | 0.7368|±  |0.1038|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7368|±  |0.1038|
| - ceval-valid_middle_school_politics                  |      2|none  |     0|acc                    |↑  | 0.8571|±  |0.0782|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.8571|±  |0.0782|
| - ceval-valid_modern_chinese_history                  |      2|none  |     0|acc                    |↑  | 0.7391|±  |0.0936|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7391|±  |0.0936|
| - ceval-valid_operating_system                        |      2|none  |     0|acc                    |↑  | 0.5789|±  |0.1164|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5789|±  |0.1164|
| - ceval-valid_physician                               |      2|none  |     0|acc                    |↑  | 0.7755|±  |0.0602|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7755|±  |0.0602|
| - ceval-valid_plant_protection                        |      2|none  |     0|acc                    |↑  | 0.7727|±  |0.0914|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7727|±  |0.0914|
| - ceval-valid_probability_and_statistics              |      2|none  |     0|acc                    |↑  | 0.2778|±  |0.1086|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.2778|±  |0.1086|
| - ceval-valid_professional_tour_guide                 |      2|none  |     0|acc                    |↑  | 0.5517|±  |0.0940|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5517|±  |0.0940|
| - ceval-valid_sports_science                          |      2|none  |     0|acc                    |↑  | 0.5263|±  |0.1177|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5263|±  |0.1177|
| - ceval-valid_tax_accountant                          |      2|none  |     0|acc                    |↑  | 0.4694|±  |0.0720|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4694|±  |0.0720|
| - ceval-valid_teacher_qualification                   |      2|none  |     0|acc                    |↑  | 0.7273|±  |0.0679|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7273|±  |0.0679|
| - ceval-valid_urban_and_rural_planner                 |      2|none  |     0|acc                    |↑  | 0.5870|±  |0.0734|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.5870|±  |0.0734|
| - ceval-valid_veterinary_medicine                     |      2|none  |     0|acc                    |↑  | 0.7391|±  |0.0936|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7391|±  |0.0936|
|hellaswag                                              |      1|none  |     0|acc                    |↑  | 0.4860|±  |0.0050|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.6453|±  |0.0048|
|ifeval                                                 |      4|none  |     0|inst_level_loose_acc   |↑  | 0.2554|±  |   N/A|
|                                                       |       |none  |     0|inst_level_strict_acc  |↑  | 0.2458|±  |   N/A|
|                                                       |       |none  |     0|prompt_level_loose_acc |↑  | 0.1590|±  |0.0157|
|                                                       |       |none  |     0|prompt_level_strict_acc|↑  | 0.1479|±  |0.0153|
|longbench_2wikimqa                                     |      5|none  |     0|qa_f1_score            |↑  | 0.0087|±  |0.0046|
|                                                       |       |none  |     0|score                  |↑  | 0.0087|±  |0.0046|
|longbench_2wikimqa_e                                   |      5|none  |     0|qa_f1_score            |↑  | 0.0019|±  |0.0014|
|                                                       |       |none  |     0|score                  |↑  | 0.0019|±  |0.0014|
|longbench_dureader                                     |      5|none  |     0|rouge_zh_score         |↑  | 0.0423|±  |0.0036|
|                                                       |       |none  |     0|score                  |↑  | 0.0423|±  |0.0036|
|longbench_gov_report                                   |      5|none  |     0|rouge_score            |↑  | 0.0119|±  |0.0007|
|                                                       |       |none  |     0|score                  |↑  | 0.0119|±  |0.0007|
|longbench_gov_report_e                                 |      5|none  |     0|rouge_score            |↑  | 0.0169|±  |0.0012|
|                                                       |       |none  |     0|score                  |↑  | 0.0169|±  |0.0012|
|longbench_hotpotqa                                     |      5|none  |     0|qa_f1_score            |↑  | 0.0011|±  |0.0011|
|                                                       |       |none  |     0|score                  |↑  | 0.0011|±  |0.0011|
|longbench_hotpotqa_e                                   |      5|none  |     0|qa_f1_score            |↑  | 0.0037|±  |0.0022|
|                                                       |       |none  |     0|score                  |↑  | 0.0037|±  |0.0022|
|longbench_lcc                                          |      5|none  |     0|code_sim_score         |↑  | 0.1192|±  |0.0045|
|                                                       |       |none  |     0|score                  |↑  | 0.1192|±  |0.0045|
|longbench_lcc_e                                        |      5|none  |     0|code_sim_score         |↑  | 0.1318|±  |0.0058|
|                                                       |       |none  |     0|score                  |↑  | 0.1318|±  |0.0058|
|longbench_lsht                                         |      5|none  |     0|classification_score   |↑  | 0.0000|±  |     0|
|                                                       |       |none  |     0|score                  |↑  | 0.0000|±  |     0|
|longbench_multi_news                                   |      5|none  |     0|rouge_score            |↑  | 0.0478|±  |0.0054|
|                                                       |       |none  |     0|score                  |↑  | 0.0478|±  |0.0054|
|longbench_multi_news_e                                 |      5|none  |     0|rouge_score            |↑  | 0.0306|±  |0.0035|
|                                                       |       |none  |     0|score                  |↑  | 0.0306|±  |0.0035|
|longbench_multifieldqa_en                              |      5|none  |     0|qa_f1_score            |↑  | 0.0121|±  |0.0039|
|                                                       |       |none  |     0|score                  |↑  | 0.0121|±  |0.0039|
|longbench_multifieldqa_en_e                            |      5|none  |     0|qa_f1_score            |↑  | 0.0121|±  |0.0039|
|                                                       |       |none  |     0|score                  |↑  | 0.0121|±  |0.0039|
|longbench_multifieldqa_zh                              |      5|none  |     0|qa_f1_zh_score         |↑  | 0.0124|±  |0.0040|
|                                                       |       |none  |     0|score                  |↑  | 0.0124|±  |0.0040|
|longbench_musique                                      |      5|none  |     0|qa_f1_score            |↑  | 0.0049|±  |0.0029|
|                                                       |       |none  |     0|score                  |↑  | 0.0049|±  |0.0029|
|longbench_narrativeqa                                  |      5|none  |     0|qa_f1_score            |↑  | 0.0014|±  |0.0014|
|                                                       |       |none  |     0|score                  |↑  | 0.0014|±  |0.0014|
|longbench_passage_count                                |      5|none  |     0|count_score            |↑  | 0.0010|±  |0.0010|
|                                                       |       |none  |     0|score                  |↑  | 0.0010|±  |0.0010|
|longbench_passage_count_e                              |      5|none  |     0|count_score            |↑  | 0.0011|±  |0.0008|
|                                                       |       |none  |     0|score                  |↑  | 0.0011|±  |0.0008|
|longbench_passage_retrieval_en                         |      5|none  |     0|retrieval_score        |↑  | 0.0083|±  |0.0060|
|                                                       |       |none  |     0|score                  |↑  | 0.0083|±  |0.0060|
|longbench_passage_retrieval_en_e                       |      5|none  |     0|retrieval_score        |↑  | 0.0121|±  |0.0047|
|                                                       |       |none  |     0|score                  |↑  | 0.0121|±  |0.0047|
|longbench_qasper                                       |      5|none  |     0|qa_f1_score            |↑  | 0.0109|±  |0.0020|
|                                                       |       |none  |     0|score                  |↑  | 0.0109|±  |0.0020|
|longbench_qasper_e                                     |      5|none  |     0|qa_f1_score            |↑  | 0.0119|±  |0.0021|
|                                                       |       |none  |     0|score                  |↑  | 0.0119|±  |0.0021|
|longbench_qmsum                                        |      5|none  |     0|rouge_score            |↑  | 0.0317|±  |0.0012|
|                                                       |       |none  |     0|score                  |↑  | 0.0317|±  |0.0012|
|longbench_repobench-p                                  |      5|none  |     0|code_sim_score         |↑  | 0.1095|±  |0.0034|
|                                                       |       |none  |     0|score                  |↑  | 0.1095|±  |0.0034|
|longbench_repobench-p_e                                |      5|none  |     0|code_sim_score         |↑  | 0.0988|±  |0.0042|
|                                                       |       |none  |     0|score                  |↑  | 0.0988|±  |0.0042|
|longbench_samsum                                       |      5|none  |     0|rouge_score            |↑  | 0.0311|±  |0.0029|
|                                                       |       |none  |     0|score                  |↑  | 0.0311|±  |0.0029|
|longbench_samsum_e                                     |      5|none  |     0|rouge_score            |↑  | 0.0257|±  |0.0021|
|                                                       |       |none  |     0|score                  |↑  | 0.0257|±  |0.0021|
|longbench_trec                                         |      5|none  |     0|classification_score   |↑  | 0.0000|±  |     0|
|                                                       |       |none  |     0|score                  |↑  | 0.0000|±  |     0|
|longbench_trec_e                                       |      5|none  |     0|classification_score   |↑  | 0.0017|±  |0.0017|
|                                                       |       |none  |     0|score                  |↑  | 0.0017|±  |0.0017|
|longbench_triviaqa                                     |      5|none  |     0|qa_f1_score            |↑  | 0.0113|±  |0.0047|
|                                                       |       |none  |     0|score                  |↑  | 0.0113|±  |0.0047|
|longbench_triviaqa_e                                   |      5|none  |     0|qa_f1_score            |↑  | 0.0112|±  |0.0046|
|                                                       |       |none  |     0|score                  |↑  | 0.0112|±  |0.0046|
|longbench_vcsum                                        |      5|none  |     0|rouge_zh_score         |↑  | 0.0384|±  |0.0014|
|                                                       |       |none  |     0|score                  |↑  | 0.0384|±  |0.0014|
|mmlu                                                   |      2|none  |      |acc                    |↑  | 0.5619|±  |0.0040|
| - humanities                                          |      2|none  |      |acc                    |↑  | 0.4801|±  |0.0069|
|  - formal_logic                                       |      1|none  |     0|acc                    |↑  | 0.4762|±  |0.0447|
|  - high_school_european_history                       |      1|none  |     0|acc                    |↑  | 0.6788|±  |0.0365|
|  - high_school_us_history                             |      1|none  |     0|acc                    |↑  | 0.7010|±  |0.0321|
|  - high_school_world_history                          |      1|none  |     0|acc                    |↑  | 0.7595|±  |0.0278|
|  - international_law                                  |      1|none  |     0|acc                    |↑  | 0.5372|±  |0.0455|
|  - jurisprudence                                      |      1|none  |     0|acc                    |↑  | 0.6111|±  |0.0471|
|  - logical_fallacies                                  |      1|none  |     0|acc                    |↑  | 0.7055|±  |0.0358|
|  - moral_disputes                                     |      1|none  |     0|acc                    |↑  | 0.5925|±  |0.0265|
|  - moral_scenarios                                    |      1|none  |     0|acc                    |↑  | 0.2425|±  |0.0143|
|  - philosophy                                         |      1|none  |     0|acc                    |↑  | 0.5659|±  |0.0282|
|  - prehistory                                         |      1|none  |     0|acc                    |↑  | 0.6111|±  |0.0271|
|  - professional_law                                   |      1|none  |     0|acc                    |↑  | 0.3905|±  |0.0125|
|  - world_religions                                    |      1|none  |     0|acc                    |↑  | 0.7193|±  |0.0345|
| - other                                               |      2|none  |      |acc                    |↑  | 0.6125|±  |0.0085|
|  - business_ethics                                    |      1|none  |     0|acc                    |↑  | 0.5600|±  |0.0499|
|  - clinical_knowledge                                 |      1|none  |     0|acc                    |↑  | 0.5962|±  |0.0302|
|  - college_medicine                                   |      1|none  |     0|acc                    |↑  | 0.5607|±  |0.0378|
|  - global_facts                                       |      1|none  |     0|acc                    |↑  | 0.3100|±  |0.0465|
|  - human_aging                                        |      1|none  |     0|acc                    |↑  | 0.6547|±  |0.0319|
|  - management                                         |      1|none  |     0|acc                    |↑  | 0.6990|±  |0.0454|
|  - marketing                                          |      1|none  |     0|acc                    |↑  | 0.8162|±  |0.0254|
|  - medical_genetics                                   |      1|none  |     0|acc                    |↑  | 0.6700|±  |0.0473|
|  - miscellaneous                                      |      1|none  |     0|acc                    |↑  | 0.6960|±  |0.0164|
|  - nutrition                                          |      1|none  |     0|acc                    |↑  | 0.6275|±  |0.0277|
|  - professional_accounting                            |      1|none  |     0|acc                    |↑  | 0.4326|±  |0.0296|
|  - professional_medicine                              |      1|none  |     0|acc                    |↑  | 0.5478|±  |0.0302|
|  - virology                                           |      1|none  |     0|acc                    |↑  | 0.4639|±  |0.0388|
| - social sciences                                     |      2|none  |      |acc                    |↑  | 0.6500|±  |0.0084|
|  - econometrics                                       |      1|none  |     0|acc                    |↑  | 0.4298|±  |0.0466|
|  - high_school_geography                              |      1|none  |     0|acc                    |↑  | 0.7273|±  |0.0317|
|  - high_school_government_and_politics                |      1|none  |     0|acc                    |↑  | 0.7668|±  |0.0305|
|  - high_school_macroeconomics                         |      1|none  |     0|acc                    |↑  | 0.5615|±  |0.0252|
|  - high_school_microeconomics                         |      1|none  |     0|acc                    |↑  | 0.6849|±  |0.0302|
|  - high_school_psychology                             |      1|none  |     0|acc                    |↑  | 0.7963|±  |0.0173|
|  - human_sexuality                                    |      1|none  |     0|acc                    |↑  | 0.6336|±  |0.0423|
|  - professional_psychology                            |      1|none  |     0|acc                    |↑  | 0.5686|±  |0.0200|
|  - public_relations                                   |      1|none  |     0|acc                    |↑  | 0.6182|±  |0.0465|
|  - security_studies                                   |      1|none  |     0|acc                    |↑  | 0.5265|±  |0.0320|
|  - sociology                                          |      1|none  |     0|acc                    |↑  | 0.7065|±  |0.0322|
|  - us_foreign_policy                                  |      1|none  |     0|acc                    |↑  | 0.7300|±  |0.0446|
| - stem                                                |      2|none  |      |acc                    |↑  | 0.5480|±  |0.0086|
|  - abstract_algebra                                   |      1|none  |     0|acc                    |↑  | 0.3700|±  |0.0485|
|  - anatomy                                            |      1|none  |     0|acc                    |↑  | 0.5556|±  |0.0429|
|  - astronomy                                          |      1|none  |     0|acc                    |↑  | 0.6908|±  |0.0376|
|  - college_biology                                    |      1|none  |     0|acc                    |↑  | 0.6458|±  |0.0400|
|  - college_chemistry                                  |      1|none  |     0|acc                    |↑  | 0.4600|±  |0.0501|
|  - college_computer_science                           |      1|none  |     0|acc                    |↑  | 0.5300|±  |0.0502|
|  - college_mathematics                                |      1|none  |     0|acc                    |↑  | 0.4300|±  |0.0498|
|  - college_physics                                    |      1|none  |     0|acc                    |↑  | 0.3627|±  |0.0478|
|  - computer_security                                  |      1|none  |     0|acc                    |↑  | 0.7800|±  |0.0416|
|  - conceptual_physics                                 |      1|none  |     0|acc                    |↑  | 0.6298|±  |0.0316|
|  - electrical_engineering                             |      1|none  |     0|acc                    |↑  | 0.6207|±  |0.0404|
|  - elementary_mathematics                             |      1|none  |     0|acc                    |↑  | 0.5503|±  |0.0256|
|  - high_school_biology                                |      1|none  |     0|acc                    |↑  | 0.7290|±  |0.0253|
|  - high_school_chemistry                              |      1|none  |     0|acc                    |↑  | 0.5025|±  |0.0352|
|  - high_school_computer_science                       |      1|none  |     0|acc                    |↑  | 0.7100|±  |0.0456|
|  - high_school_mathematics                            |      1|none  |     0|acc                    |↑  | 0.3963|±  |0.0298|
|  - high_school_physics                                |      1|none  |     0|acc                    |↑  | 0.4305|±  |0.0404|
|  - high_school_statistics                             |      1|none  |     0|acc                    |↑  | 0.4630|±  |0.0340|
|  - machine_learning                                   |      1|none  |     0|acc                    |↑  | 0.3929|±  |0.0464|
|openbookqa                                             |      1|none  |     0|acc                    |↑  | 0.2940|±  |0.0204|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.4040|±  |0.0220|
|piqa                                                   |      1|none  |     0|acc                    |↑  | 0.7454|±  |0.0102|
|                                                       |       |none  |     0|acc_norm               |↑  | 0.7492|±  |0.0101|
|social_iqa                                             |      0|none  |     0|acc                    |↑  | 0.4795|±  |0.0113|
|truthfulqa_gen                                         |      3|none  |     0|bleu_acc               |↑  | 0.2411|±  |0.0150|
|                                                       |       |none  |     0|bleu_diff              |↑  |-0.1089|±  |0.0809|
|                                                       |       |none  |     0|bleu_max               |↑  | 2.0653|±  |0.2431|
|                                                       |       |none  |     0|rouge1_acc             |↑  | 0.2717|±  |0.0156|
|                                                       |       |none  |     0|rouge1_diff            |↑  |-0.3118|±  |0.1112|
|                                                       |       |none  |     0|rouge1_max             |↑  | 6.2383|±  |0.3799|
|                                                       |       |none  |     0|rouge2_acc             |↑  | 0.2093|±  |0.0142|
|                                                       |       |none  |     0|rouge2_diff            |↑  |-0.4369|±  |0.1204|
|                                                       |       |none  |     0|rouge2_max             |↑  | 3.9686|±  |0.3227|
|                                                       |       |none  |     0|rougeL_acc             |↑  | 0.2595|±  |0.0153|
|                                                       |       |none  |     0|rougeL_diff            |↑  |-0.3870|±  |0.1123|
|                                                       |       |none  |     0|rougeL_max             |↑  | 5.8727|±  |0.3659|
|truthfulqa_mc1                                         |      2|none  |     0|acc                    |↑  | 0.3109|±  |0.0162|
|truthfulqa_mc2                                         |      3|none  |     0|acc                    |↑  | 0.4858|±  |0.0148|
|winogrande                                             |      1|none  |     0|acc                    |↑  | 0.6456|±  |0.0134|
|niah_single_1|      1|none  |     0|  1024|   |    1|±  |     0|
|             |       |none  |     0| 16384|↑  |    0|±  |   N/A|
|             |       |none  |     0|  2048|   |    0|±  |     0|
|             |       |none  |     0| 32768|↑  |    0|±  |   N/A|
|             |       |none  |     0|  4096|↑  |    0|±  |   N/A|
|             |       |none  |     0|  8192|↑  |    0|±  |   N/A|
|niah_single_2|      1|none  |     0|  1024|   |    1|±  |0.0000|
|             |       |none  |     0| 16384|↑  |    0|±  |   N/A|
|             |       |none  |     0|  2048|   |    0|±  |0.0000|
|             |       |none  |     0| 32768|↑  |    0|±  |   N/A|
|             |       |none  |     0|  4096|↑  |    0|±  |   N/A|
|             |       |none  |     0|  8192|↑  |    0|±  |   N/A|
|niah_single_3|      1|none  |     0|  1024|   |    1|±  |0.0000|
|             |       |none  |     0| 16384|↑  |    0|±  |   N/A|
|             |       |none  |     0|  2048|   |    0|±  |0.0000|
|             |       |none  |     0| 32768|↑  |    0|±  |   N/A|
|             |       |none  |     0|  4096|↑  |    0|±  |   N/A|
|             |       |none  |     0|  8192|↑  |    0|±  |   N/A|
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