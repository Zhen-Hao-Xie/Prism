# #!/bin/bash

bash scripts/SAME/Eval_CoIN/1_ScienceQA.sh SAME-task1 ./checkpoints/Task1_llava_lora_ours

bash scripts/SAME/Eval_CoIN/1_ScienceQA.sh SAME-task2 ./checkpoints/Task2_llava_lora_ours
bash scripts/SAME/Eval_CoIN/2_TextVQA.sh SAME-task2 ./checkpoints/Task2_llava_lora_ours

bash scripts/SAME/Eval_CoIN/1_ScienceQA.sh SAME-task3 ./checkpoints/Task3_llava_lora_ours
bash scripts/SAME/Eval_CoIN/2_TextVQA.sh SAME-task3 ./checkpoints/Task3_llava_lora_ours
bash scripts/SAME/Eval_CoIN/3_ImageNet.sh SAME-task3 ./checkpoints/Task3_llava_lora_ours

bash scripts/SAME/Eval_CoIN/1_ScienceQA.sh SAME-task4 ./checkpoints/Task4_llava_lora_ours
bash scripts/SAME/Eval_CoIN/2_TextVQA.sh SAME-task4 ./checkpoints/Task4_llava_lora_ours
bash scripts/SAME/Eval_CoIN/3_ImageNet.sh SAME-task4 ./checkpoints/Task4_llava_lora_ours
bash scripts/SAME/Eval_CoIN/4_GQA.sh SAME-task4 ./checkpoints/Task4_llava_lora_ours

bash scripts/SAME/Eval_CoIN/1_ScienceQA.sh SAME-task5 ./checkpoints/Task5_llava_lora_ours
bash scripts/SAME/Eval_CoIN/2_TextVQA.sh SAME-task5 ./checkpoints/Task5_llava_lora_ours
bash scripts/SAME/Eval_CoIN/3_ImageNet.sh SAME-task5 ./checkpoints/Task5_llava_lora_ours
bash scripts/SAME/Eval_CoIN/4_GQA.sh SAME-task5 ./checkpoints/Task5_llava_lora_ours
bash scripts/SAME/Eval_CoIN/5_VizWiz.sh SAME-task5 ./checkpoints/Task5_llava_lora_ours

bash scripts/SAME/Eval_CoIN/1_ScienceQA.sh SAME-task6 ./checkpoints/Task6_llava_lora_ours
bash scripts/SAME/Eval_CoIN/2_TextVQA.sh SAME-task6 ./checkpoints/Task6_llava_lora_ours
bash scripts/SAME/Eval_CoIN/3_ImageNet.sh SAME-task6 ./checkpoints/Task6_llava_lora_ours
bash scripts/SAME/Eval_CoIN/4_GQA.sh SAME-task6 ./checkpoints/Task6_llava_lora_ours
bash scripts/SAME/Eval_CoIN/5_VizWiz.sh SAME-task6 ./checkpoints/Task6_llava_lora_ours
bash scripts/SAME/Eval_CoIN/6_Grounding.sh SAME-task6 ./checkpoints/Task6_llava_lora_ours

bash scripts/SAME/Eval_CoIN/1_ScienceQA.sh SAME-task7 ./checkpoints/Task7_llava_lora_ours
bash scripts/SAME/Eval_CoIN/2_TextVQA.sh SAME-task7 ./checkpoints/Task7_llava_lora_ours
bash scripts/SAME/Eval_CoIN/3_ImageNet.sh SAME-task7 ./checkpoints/Task7_llava_lora_ours
bash scripts/SAME/Eval_CoIN/4_GQA.sh SAME-task7 ./checkpoints/Task7_llava_lora_ours
bash scripts/SAME/Eval_CoIN/5_VizWiz.sh SAME-task7 ./checkpoints/Task7_llava_lora_ours
bash scripts/SAME/Eval_CoIN/6_Grounding.sh SAME-task7 ./checkpoints/Task7_llava_lora_ours
bash scripts/SAME/Eval_CoIN/7_VQAv2.sh SAME-task7 ./checkpoints/Task7_llava_lora_ours

bash scripts/SAME/Eval_CoIN/1_ScienceQA.sh SAME-task8 ./checkpoints/Task8_llava_lora_ours
bash scripts/SAME/Eval_CoIN/2_TextVQA.sh SAME-task8 ./checkpoints/Task8_llava_lora_ours
bash scripts/SAME/Eval_CoIN/3_ImageNet.sh SAME-task8 ./checkpoints/Task8_llava_lora_ours
bash scripts/SAME/Eval_CoIN/4_GQA.sh SAME-task8 ./checkpoints/Task8_llava_lora_ours
bash scripts/SAME/Eval_CoIN/5_VizWiz.sh SAME-task8 ./checkpoints/Task8_llava_lora_ours
bash scripts/SAME/Eval_CoIN/6_Grounding.sh SAME-task8 ./checkpoints/Task8_llava_lora_ours
bash scripts/SAME/Eval_CoIN/7_VQAv2.sh SAME-task8 ./checkpoints/Task8_llava_lora_ours
bash scripts/SAME/Eval_CoIN/8_OCRVQA.sh SAME-task8 ./checkpoints/Task8_llava_lora_ours