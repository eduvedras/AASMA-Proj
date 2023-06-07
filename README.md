# Overcooked-AI
We applied two of the most known reinforcement learning algorithms, Q-Learning and SARSA, to the multi-agent environment that is Overcooked-AI.

The experiments are run on `Cramped Room` layout.


## How to Run
### Create and activate virtual environment
We first need to create a virtual environment.
```bash
python3 -m venv venv
```

Now we activate the virtual environment.
```bash
source venv/bin/activate
```

### Give permissions to the script
```bash
chmod +x run_train.sh
```

### Run the bash script
#### For Q-Learning Agents
```bash
./run_train.sh qlearning
```
#### For SARSA Agents
```bash
./run_train.sh sarsa
```

### More configurations
If you want to run with other parameters run the following python command with the flags of those parameters.
Example:
```bash
python train.py episode_length=100 eval_frequency=100 num_eval_episodes=1 num_train_steps=200 agent=sarsa
```
The configurable parameters are in the train.yaml file.

## Acknowledgement
- [Overcooked Environment](https://github.com/HumanCompatibleAI/overcooked_ai)






