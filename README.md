# NeurIPS-23-ORDER
The Official Repository for "Offline RL with Discrete Proxy Representations for Generalizability in POMDPs"

---

### NeurIPS 2023 Accepted Paper

🎉 **News**: Our paper has been accepted at the NeurIPS 2023 conference.


📅 **Expected Release**: We are working hard to ensure that the code is of the highest quality and will announce the release date soon. 

### About the Research

Offline Reinforcement Learning (RL) has demonstrated promising results in various applications by learning policies from previously collected datasets, reducing the need for online exploration and interactions. However, real-world scenarios usually involve partial observability, which brings crucial challenges of the deployment of offline RL methods: i) the policy trained on data with full observability is not robust against the masked observations during execution, and ii) the information of which parts of observations are masked is usually unknown during training. In order to address these challenges, we present Offline RL with DiscrEte pRoxy representations (ORDER), a probabilistic framework which leverages novel state representations to improve the robustness against diverse masked observabilities. Specifically, we propose a discrete representation of the states and use a proxy representation to recover the states from masked partial observable trajectories. The training of ORDER can be compactly described as the following three steps. i) Learning the discrete state representations on data with full observations, ii) Training the decision module based on the discrete representations, and iii) Training the proxy discrete representations on the data with various partial observations, aligning with the discrete representations. We conduct extensive experiments to evaluate ORDER, showcasing its effectiveness in offline RL for diverse partially observable scenarios and highlighting the significance of discrete proxy representations in  generalization performance.
ORDER is a flexible framework to employ any offline RL algorithms and we hope that ORDER can pave the way for the deployment of RL policy against various partial  observabilities in the real world.

 ### You can run the script like so

```
cd ORDER

# train the oracle policy

python main.py --experiment_name train_iql_critic --code_name test_hopper --env_name hopper-medium-v2
 
python run_vae_policy.py --experiment_name train_oracle_policy  --code_name test_hopper --env_name hopper-medium-v2 --device 0  --critic_path ..\Experiments\train_iql_critic\Checkpoints\test_hopper_1234\best.pt # here you may change the critic path as an absolute path

# train the proxy policy

python run_pomdp_encoder.py --experiment_name train_proxy_policy --code_name test_hopper --env_name hopper-medium-v2 --env_code hopper --device 0  --policy_path ../Experiments/train_oracle_policy/Checkpoints/test_hopper_1234/best.pt # Similarly, here you may change the policy path as an absolute path

```

