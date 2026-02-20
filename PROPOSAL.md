# Project Proposal

**Project:** Robustness and Recovery in Offline-to-Online Reinforcement Learning  
**Team:** Anika Fuloria, Nicole Garcia, Helen Li

---

## Abstract

Offline reinforcement learning (RL) learns policies from fixed, pre-collected datasets without further interaction with the environment. While attractive for safety-critical domains, offline methods often suffer from performance degradation when deployed online, due to distributional mismatch between the training data and newly encountered states and actions [1]. Many state-of-the-art algorithms address this issue through pessimistic value estimation to avoid overestimation in poorly covered regions of the dataset. However, excessive conservatism can lead to value collapse or slow adaptation when real environment feedback contradicts pessimistic predictions during online fine-tuning [1][2]. This problem is particularly important for real-world applications such as robotics and healthcare, where historical datasets are inevitably incomplete, biased, or noisy, yet agents must safely adapt online to changing conditions [3][4]. In this project, we investigate two related research questions:

1. How severely do existing offline RL algorithms degrade under realistic dataset flaws such as missing high-quality trajectories or noisy rewards?
2. Can uncertainty-driven exploration mechanisms stabilize the transition from offline to online learning and enable faster performance recovery compared to naive online fine-tuning?

---

## Environment and Data

We plan to evaluate our methods on the D4RL MuJoCo locomotion benchmark, focusing on the HalfCheetah and Hopper tasks. Prior works have emphasized the challenges of incomplete offline data in safety-critical domains like robotics and healthcare applications [3].

In safety-critical domains like robotics, offline datasets often omit crucial trajectories; for example, Luo et al. (2023) emphasize the challenges of incomplete offline data in robotics and healthcare applications [3].

To simulate similar dataset imperfections, we will apply synthetic corruption to D4RL datasets by removing the top-*k* percent of reward trajectories (*k* ∈ {30, 60}), modeling coverage gaps arising from suboptimal or constrained data collection processes. As a stretch goal, we may introduce Gaussian noise to reward signals to study robustness to noise. The project is structured into two phases corresponding to the aforementioned research questions.

---

## Phase I: Characterizing Failure Modes in Offline RL

First, we will train representative offline RL algorithms, including Conservative Q-Learning (CQL) and Implicit Q-Learning (IQL) [5], on both clean and corrupted datasets. We will evaluate their performance using normalized returns as well as stability-related metrics like loss variance and Q-value fluctuations during training. This analysis targets how different forms and degrees of dataset corruption affect offline learning and downstream online performance. Qualitatively, we expect learning curves to exhibit increasing instability and degraded performance as dataset corruption increases, as informed by prior works that linked offline algorithm performance collapse to encounters with novel states [3].

For example, Luo et al. (2023) find that standard finetuning of offline-trained policies can suffer from severe performance drops ("policy collapse") when novel states are encountered [3].

---

## Phase II: Uncertainty-Driven Online Recovery

In the second phase, we investigate whether uncertainty-aware exploration can improve online fine-tuning from corrupted offline models. Specifically, we will fine-tune the trained agents online using an ensemble of Q-networks (3–5 members) and derive an exploration bonus from ensemble disagreement to add to the rewards.

This approach is inspired by McInroe et al. (2024), which framed the offline-to-online transition as an exploration problem and proposed an algorithm (PTGOOD) that targets high-value, out-of-distribution state-action regions during fine-tuning [2].

As an alternative uncertainty signal, we may also evaluate TD-error-based bonuses. We will compare this approach against vanilla online fine-tuning and offline-only baselines to assess the benefits of uncertainty-driven exploration.

Evaluation metrics include sample efficiency (number of online environment steps required to reach a fixed performance threshold), final asymptotic performance, and training stability during the offline-to-online transition. Quantitatively, we will report mean and standard deviation across multiple random seeds and compare methods using learning curves and summary statistics.

---

## Related Work

We build on offline RL foundations, hybrid offline-to-online methods, and uncertainty-based exploration. CQL [6] and IQL [5] established pessimistic value estimation to address extrapolation error in poorly-covered dataset regions. However, Liu et al. (2024) [1] found that both insufficient and excessive pessimism harm policy learning. This tension directly motivates our Phase I investigation: characterizing how existing offline algorithms degrade under dataset corruption.

Several studies have targeted instability in the offline-to-online transition. Zhao et al. (2025) [7] identified performance collapse during fine-tuning and proposed retaining the offline evaluation network until convergence for stability. Luo et al. (2023) [3] documented "policy collapse" when naive fine-tuning encounters novel states. Ball et al. (2023) [8] showed that well-designed hybrid approaches achieve sample-efficient online adaptation. Most relevant to our Phase II, McInroe et al. (2024) [2] framed offline-to-online fine-tuning as an exploration problem, introducing the PTGOOD algorithm to actively seek high-value, out-of-distribution samples during online adaptation. This work directly inspired our uncertainty bonuses approach to guide exploration where corrupted offline data provided insufficient coverage. Other approaches include behavior-constrained updates [9], diffusion-based augmentation [10], experience replay rebalancing [4], and stability-plasticity tradeoffs [11]. Our Phase II also draws on ensemble-based uncertainty quantification from Osband et al. (2016) [12]. Ensemble disagreement measures states where the agent lacks knowledge, which we use for exploration bonuses during fine-tuning. Combining this with PTGOOD's targeted out-of-distribution sampling, we aim to show that uncertainty bonuses counteract excessive conservatism from corrupted offline pre-training, enabling faster recovery than vanilla fine-tuning.

Several offline RL algorithms [6][5] have underscored the value of pessimism in countering challenges related to dataset bias and limited coverage, but Liu et al. (2024) [1] observe that both insufficient and excessive pessimism can harm policy learning, motivating adaptive pessimistic constraints.

Inspired by recent work on hybrid offline-online RL that studies safe and efficient adaptation [8], we propose a method that leverages hybrid approaches for adaptive pessimistic constraints.

For example, Zhao et al. (2025) [7] identified instability during the offline-to-online fine-tuning process and propose a Stable Fine-Tuning (SFT) framework that retains the offline evaluation network until convergence to maintain performance stability.

In addition, several recent studies explicitly target the offline-to-online transition. Su et al. (2025) [4] integrated offline pretraining with online fine-tuning for robot social navigation, using a hybrid experience sampling method to mitigate distribution shifts from dynamic pedestrian interactions. McInroe et al. (2024) [2] treated the offline-to-online fine-tuning as an exploration problem, introducing the PTGOOD algorithm to actively seek out-of-distribution, high-reward samples during online adaptation. Zu et al. (2025) [9] propose Behavior-Adaptive Q-Learning (BAQ), which leverages a behavior cloning model derived from offline data to constrain and gradually relax policy updates, thereby stabilizing early online learning and accelerating adaptation. Huang et al. (2025) [10] leveraged classifier-free diffusion models for data augmentation, generating samples that bridge the offline and online data distributions to improve performance in offline-to-online RL. Lin et al. (2026) [11] introduced a stability-plasticity principle for offline-to-online RL, identifying three regimes of fine-tuning and guiding algorithm design based on the trade-off between preserving offline knowledge and enabling online adaptation. Luo et al. (2023) [3] examined fine-tuning from offline RL and report that naive online finetuning can suffer "policy collapse" at the start, suggesting conservative policy optimization as a remedy.

For uncertainty-driven exploration, we draw on ensemble-based methods originally proposed for deep Q-learning [12]. These techniques inform our use of ensemble disagreement bonuses to guide exploration in the offline-to-online setting.

---

## References

[1] liu2024adaptivepessimism  
[2] mcinroe2024planning  
[3] luo2023finetuning  
[4] su2025social  
[5] kostrikov2021implicit  
[6] kumar2020conservativeqlearningofflinereinforcement  
[7] zhao2025stable  
[8] ball2023efficientonlinereinforcementlearning  
[9] zu2025behavior  
[10] huang2025classifier  
[11] lin2026threeregimes  
[12] osband2016deepexplorationbootstrappeddqn