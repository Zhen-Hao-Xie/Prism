1.same
python run.py train 2 3 4 5 6 7 8 9 && python infer 0 1 2 3 4 5 6 7 8 9
2.hide_llava
python run.py train 0 1 2 3 4 5 6 7 8 9 && python infer 0 1 2 3 4 5 6 7 8 9
3.replay_lora
python run.py train 0 1 2 3 4 5 6 7 8 9 && python infer 0 1 2 3 4 5 6 7 8 9
4.ft_lora
python run.py train 0 1 2 3 4 5 6 7 8 9 && python infer 0 1 2 3 4 5 6 7 8 9