train:
	python3 run.py train $(game)

test:
	python3 run.py test $(game) $(human) $(eps) $(num_sim) $(ai_player) $(heuristic_w) $(model_name)

view-log:
	python3 view_log.py $(game) $(log_name)