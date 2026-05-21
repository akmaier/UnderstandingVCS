XDQN: Inherently Interpretable DQN through Mimicking

Andreas Kontogiannis, George Vouros

National Technical University of Athens, Greece (andr.kontog@gmail.com); University of Piraeus, Greece (georgev@unipi.gr)

## Abstract

Although deep reinforcement learning (DRL) methods have been successfully applied in challenging tasks, their application in real-world operational settings is challenged by methods' limited ability to provide explanations. Among the paradigms for explainability in DRL is the interpretable box design paradigm, where interpretable models substitute inner constituent models of the DRL method, thus making the DRL method "inherently" interpretable. In this paper we explore this paradigm and we propose XDQN, an explainable variation of DQN, which uses an interpretable policy model trained through mimicking. XDQN is challenged in a complex, real-world operational multi-agent problem, where agents are independent learners solving congestion problems. Specifically, XDQN is evaluated in three MARL scenarios, pertaining to the demand-capacity balancing problem of air traffic management. XDQN achieves performance similar to that of DQN, while its abilities to provide global models' interpretations and interpretations of local decisions are demonstrated.

**Keywords:** Deep Reinforcement Learning, Mimic Learning, Explainability

## 1 Introduction

Deep Reinforcement Learning (DRL) has mastered decision making policies in various difficult control tasks [11] [18] [15], games [22] [13] and other real-time applications [14] [37]. Despite the remarkable performance of DRL models, the knowledge of mastering these tasks remains implicit in deep neural networks. Thus, its application in real-world operational settings is challenged by methods' limited ability to provide explanations at global (policy) and local (individual decisions) levels. This lack of interpretability makes it difficult to trust DRL for solving safety-critical real-world tasks. However, besides the inability of DRL models to provide interpretations on the selection of actions in specific circumstances, they are also unable to provide information about the evolution of models during the training process. These challenges are naturally further extended to multi-agent settings, in which different agents empowered by multi-agent reinforcement learning (MARL) methods aim at learning a joint optimal policy towards solving a target task.

To address some of the aforementioned challenges, one may follow different paradigms for the provision of explanations: The interpretable box design paradigm is one of them where interpretable models substitute inner components of DRL [35]. Additionally, mimic learning has been proposed, so as to infer interpretable models that mimic the behavior of well-trained deep neural networks [2, 5]. In the DRL case, mimic learning aims to replace the closed-box DRL controller with an interpretable one, able to mimic the decisions made by the former [3, 19, 35]. A mimic learner tries to optimize *fidelity* [35], which is determined by comparing the mimic controller's actions with the actions selected by the DRL model. To extract knowledge from deep neural networks, recent work [3, 19] has applied mimic learning with tree representations, using decision trees: Criteria used for splitting tree nodes provide a tractable way to explain the predictions made by the controller.

Typically, mimic learning approaches require already well-trained complex policy networks (which we refer to as *mature* networks), whose behavior are mimicking to support interpretability. In real-world scenarios, this could be quite impractical, since the training overhead required to train the mimic models can often be a very time-consuming and costly process, especially for large state-action spaces and for multi-agent settings. Another limitation of such approaches is that they solely aim at providing explainability on the predictions of only the mature DRL model, ignoring completely the training process of this model. In other words, in these approaches, the mimic learner can only provide explanations about the policy of the inferred DRL controller, but not about the patterns and behaviors learned throughout the training process.

To deal with these challenges, in this paper we propose *eXplainable Deep Q-Network* (*XDQN*), which is an explainable variation of the well-known DQN [22] method. In XDQN, our goal is to provide inherent explainability of DQN via mimic learning in an online manner, by replacing the complex deep Q-network with an interpretable mimic learner in testing. In so doing, XDQN does not require the existence of a well-trained model to train an interpretable one. In particular, we train a mimic learner in parallel with the deep neural network (Q-network) of DQN in an online setting, where: at a training step the DRL model uses the mimic learner to compute the target values of the Q-network needed for its training, while the mimic learner learns to behave as the DRL model, but in an explainable way. Since the mimic learner is trained and updated while the DQN policy model is trained, we can keep multiple "snapshots" of the model evolution through time, offering interpretability on these intermediate models, and insights about the patterns and behaviors that DQN learns during training.

To evaluate our method's utility in real-world operational settings, XDQN is challenged in a complex, real-world multi-agent problem, where agents solve airspace congestion problems. Agents in this setting are trained via parameter sharing following the centralized training, decentralized execution paradigm. We summarize the main contributions of this paper below:

- To our knowledge, this work is the first that provides DQN with inherent interpretability through mimic learning without requiring the existence of a well-trained DRL model.
- We propose XDQN, an explainable variation of DQN, in which an interpretable mimic learner is trained in parallel with the Q-network of DQN and plays the role of the target Q-network of DQN.
- Experimentally, we show that XDQN can perform similarly to DQN, demonstrating good play performance and fidelity to DQN decisions in complex, real-world operational multi-agent problems.
- We demonstrate the ability of XDQN to provide global (policy) and local (in specific circumstances) explanations regarding agents' decisions, also while models are being trained.

## 2 Background

### 2.1 Markov Decision Process

We consider a sequential decision making setup, in which an agent interacts with an environment $E$ over discrete time steps. At a given timestep, the agent perceives features regarding a state $s_t \in S$, where $S$ is the state space. The agent then chooses an action $a_t$ from a discrete set $A$ and observes a reward $r_t$ generated by the environment.

The agent's behavior is determined by a policy $\pi$, which maps states to a probability distribution over the actions, that is $\pi : S \to P(A)$. Apart from an agent's policy, the environment $E$ may also be stochastic. We model it as a Markov Decision Process (MDP) with a state space $S$, action space $A$, an initial state distribution $p(s_1)$, transition dynamics $p(s_{t+1}|s_t)$ and a reward function $r(s_t, a_t, s_{t+1})$. For brevity, we write $r_t = r(s_t, a_t, s_{t+1})$.

The agent aims to maximize the expected discounted cumulative reward, which is formulated as $G_t = \sum_{\tau=t}^{\infty} \gamma^{\tau-t} r_\tau$. Here, $\gamma \in (0, 1)$ is a discount factor which trades-off the importance of immediate and future rewards. Considering that an agent acts under a stochastic policy $\pi$, the Q-function (state-action value) of a pair $(s, a)$ is defined as follows

$$Q^\pi(s, a) = \mathbb{E}[G_t | s_t = s, a_t = a, \pi] \quad (1)$$

which can also be computed recursively with bootstrapping:

$$Q^\pi(s, a) = \mathbb{E}\left[r_t + \gamma \mathbb{E}_{a \sim \pi(s_{t+1})}[Q^\pi(s_{t+1}, a)] | s_t = s, a_t = a, \pi\right] \quad (2)$$

The Q-function measures the value of choosing a particular action when the agent is in this state. We define the optimal policy $\pi^*$ under which the agent receives the optimal $Q^*(s, a) = \max_\pi Q^\pi(s, a)$. For a given state $s$, under the optimal policy $\pi^*$, the agent selects action $a = \arg\max_{a' \in A} Q^*(s, a')$. Therefore, it follows that the optimal Q-function satisfies the Bellman equation:

$$Q^*(s, a) = \mathbb{E}\left[r_t + \gamma \max_a Q^*(s_{t+1}, a) | s_t = s, a_t = a, \pi\right]. \quad (3)$$

### 2.2 Deep Q-Networks

To deal with a high dimensional state space, the state-action value function can be approximated by an online deep Q-network (DQN [22]); i.e. a deep neural network $Q(s, a; \theta)$ with weight parameters $\theta$. To estimate the parameters $\theta$, at iteration $i$ the expected mean squared loss between the estimated Q-value of a state-action pair and its temporal difference target, produced by a fixed and separate *target* Q-network $Q(s, a; \theta^-)$ with weight parameters $\theta^-$, is minimized. Formally:

$$L_i(\theta_i) = \mathbb{E}\left[Y_i^{DQN} - Q(s, a; \theta)\right], \quad (4)$$

with

$$Y_i^{DQN} = r_t + \gamma \max_{a \in A} Q(s_{t+1}, a; \theta^-) \quad (5)$$

In order to train DQN and estimate $\theta$, we could use the standard Q-learning update algorithm. Nevertheless, the Q-learning estimator performs very poorly in practice. To stabilize the training procedure of DQN, Mnih et. al [22] freezed the parameters, $\theta^-$, of the target Q-network for a fixed number of training iterations while updating the online Q-network with gradient descent steps with respect to $\theta$.

In addition to the target network, during the learning process, DQN uses an experience replay buffer [22], which is an accumulative dataset, $D_t$, of state transitions - in the form of $(s, a, r, s')$ - from past episodes. In a training step, instead of only using the current state transition, the Q-Network is trained by sampling mini-batches of past transitions from $D$ uniformly, at random. Therefore, the loss can be written as follows:

$$L_i(\theta_i) = \mathbb{E}_{(s,a,r,s') \sim U(D)}\left[(Y_i^{DQN} - Q(s, a; \theta))^2\right]. \quad (6)$$

The main advantage of using an experience replay buffer is that uniform sampling reduces the correlation among the experience samples used for training the Q-network. The replay buffer also improves data efficiency through reusing the experience samples in multiple training steps.

Instead of sampling mini-batches of past transitions uniformly from the experience replay buffer, a further improvement over DQN results from using a prioritized experience replay buffer [30]. It aims at increasing the probability of sampling those past transitions from the experience replay that are expected to be more useful in terms of absolute temporal difference error.

### 2.3 Mimic Learning for Deep Reinforcement Learning

Recent work on mimic learning [7, 19] has shown that rule-based models, like decision trees, or shallow feed-forward neural networks can mimic a not linear function inferred by a deep neural network with millions of parameters. We present two known settings for mimicking the Q-function of a DRL model.

#### 2.3.1 Experience Training

In the experience training setting [7, 19], all the state-action pairs $\langle s, a \rangle$ of a DRL training process are collected in a time horizon $T$. Then, to obtain the corresponding Q-values, these pairs are provided as input into a DRL model. The final set $\{(\langle s_1, a_1 \rangle, Q_1), ...(\langle s_T, a_T \rangle, Q_T)\}$ of tuples is used as the experience training dataset.

#### 2.3.2 Active Play

The main problem with the experience training is that suboptimal state-action pairs are collected through training, making it more difficult for a learner to mimic the behavior of the DRL model. To address this challenge, active play [19] uses a mature DRL model to generate state-action pairs to construct the training dataset of an active mimic learner. The training data is collected in an online manner through queries, in which the active learner selects the actions, given the states, and the mature DRL model provides the estimated Q-values. These Q-values are then used to update the active learner's parameters on minibatches of the collected dataset.

## 3 Explainable Deep Q-Network (XDQN)

In this work, we are interested in providing interpretability in deep Q-learning through mimicking the behavior of DQN. To this aim, we propose eXplainable Deep Q-Network (XDQN), which is an explainable variation of DQN [22]. XDQN aims at inferring the parameters of the online Q-network and the parameters of a mimic learner concurrently, in an online manner, with the latter substituting the target Q-network of DQN.

Formally, let $\theta$ be the parameters of the online Q-network and $\phi_X$ be the parameters of the mimic learner. In XDQN, the mimic learner is used to estimate the state-action value function and select the best action for the next state in the XDQN target:

$$Y_i^{XDQN} = r_t + \gamma \max_{a \in A} Q(s_{t+1}, a; \phi_X) \quad (7)$$

Similar to DQN, $\phi_X$ are updated every $T_u$ number of timesteps. The full training procedure of XDQN is presented in Algorithm 1.

In contrast to DQN in which we simply copy the parameters $\theta$ of the online Q-network to update the parameters of the target Q-network, here we perform mimic learning on $Q(s, a, \theta)$ (steps 17-20). To update $\phi_X$ we train the mimic learner on minibatches of the experience replay buffer $B$ by minimizing the Mean Squared Error (MSE) loss function using $Q(s, a, \theta)$ to estimate the soft labels (Q-values) of the state-action pairs in the minibatches. Formally the optimization problem for each update of $\phi_X$ can be written as:

$$\min_{\phi_X} \mathbb{E}_{(s,a) \sim B}\left[(Q(s, a; \phi_X) - Q(s, a; \theta))^2\right] \quad (8)$$

In our experiments, we utilize a prioritized experience replay [30] as the replay buffer $B$, as described in Section 2. Similarly to active play, when updating $\phi_X$, to ensure that the state-action pairs of the minibatches provide up-to-date target values with respect to $\theta$, we use records from the replay buffer that were stored during the $K$ latest training steps.

It is worth noting that at each update of $\phi_X$ the hyperparameter $K$ for past transitions plays a similar role as the discounted factor $\gamma$ plays for future rewards, but from the mimic learner's perspective. Building upon the experience training and active play paradigms, XDQN can leverage the benefits of both of them. In particular, the hyperparameter $K$ manages the trade-off between experience training and active play in XDQN. If $K$ is large, the mimic model learns from state-action pairs that may have been collected through more suboptimal instances of $\theta$; deploying however data-augmented versions of Q-value. On the other hand, if $K$ is small, it learns from the most recent instances of $\theta$; making use of up-to-date Q-values. Nevertheless, opting for very small values of $K$ could lead to less stable mimic training, due to the smaller number of minibatches that can be produced for updating $\phi_X$, while using large $K$ can result in a very slow training process.

From all the above, we note that $\theta$ (Q-network) and $\phi_X$ (mimic learner) are highly dependent. To update $\theta$, Q-network uses the mimic learner model with $\phi_X$ to compute the target soft labels (target Q-values), while to update $\phi_X$ the mimic learner uses the original Q-network with parameters $\theta$ to compute the respective target soft labels (online Q-values). Since XDQN produces different instances of $\phi_X$ throughout training, it can eventually output multiple interpretable mimic learner models (up to the number of $\phi_X$ updates), with each one of them corresponding to a different training timestep. Assuming that all these mimic learner instances are interpretable models, XDQN can also provide explainability on how a DRL model learns to solve the target task.

Finally, after Q-network ($\theta$) and mimic learner ($\phi_X$) have been trained, without requiring to learn $\theta$ before $\phi_X$, we can discard the online Q-network and use the mimic learner model as the controller. Therefore, in testing, given a state, the interpretable mimic learner selects the action that profits the highest Q-value, being also able to provide explainability.

**Algorithm 1** eXplainable Deep Q-Network (XDQN)

1. Initialize replay buffer $B$ with capacity N
2. Initialize $\theta$ and $\phi_X$
3. Initialize timestep count $c = 0$
4. **for** episode 1, M **do**
5. &nbsp;&nbsp;&nbsp;&nbsp;Augment $c = c + 1$
6. &nbsp;&nbsp;&nbsp;&nbsp;Initialize state $s_1$
7. &nbsp;&nbsp;&nbsp;&nbsp;With probability $\epsilon$ select a random action $a_t$, otherwise $a_t = \arg\max_a Q(s_t, a; \theta)$
8. &nbsp;&nbsp;&nbsp;&nbsp;Execute action $a_t$ and observe next state $s_{t+1}$ and reward $r_t$
9. &nbsp;&nbsp;&nbsp;&nbsp;Store transition $(s_t, a_t, s_{t+1}, r_t)$ in $B$
10. &nbsp;&nbsp;&nbsp;&nbsp;Sample a minibatch of transitions $(s_i, a_i, s_{i+1}, r_i)$ from $B$
11. &nbsp;&nbsp;&nbsp;&nbsp;**if** $s_{i+1}$ not terminal **then**
12. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Set $Y_i^{XDQN} = r_i + \gamma \max_{a \in A} Q(s_{i+1}, a; \phi_X)$
13. &nbsp;&nbsp;&nbsp;&nbsp;**else**
14. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Set $Y_i^{XDQN} = r_i$
15. &nbsp;&nbsp;&nbsp;&nbsp;**end if**
16. &nbsp;&nbsp;&nbsp;&nbsp;Perform a gradient descent step on $(Y_i^{XDQN} - Q(s_i, a_i; \theta))^2$ w.r.t. $\theta$
17. &nbsp;&nbsp;&nbsp;&nbsp;**if** $c \mod T_u = 0$ **then**
18. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Initialize $\phi_X$
19. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sample a minibatch of transitions $(s_i, a_i, s_{i+1}, r_i)$ from $B$ that were stored at most $c - K$ timesteps before
20. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Perform mimic learning update on $(Q(s, a; \phi_X) - Q(s, a; \theta))^2$ w.r.t $\phi_X$
21. &nbsp;&nbsp;&nbsp;&nbsp;**end if**
22. **end for**

## 4 Experimental Setup

In this section, we demonstrate the effectiveness of XDQN through experiments on real-world data. In all experiments we utilize a Gradient Boosting Regressor [36] as the mimic learner, so as to exploit its boosting ability to learn effectively by exploiting instances generated by the deep Q-network. Although most decision tree algorithms, being rule-based models, are naturally interpretable models [3, 19], this is not the case for a Gradient Boosting Regressor, since the boosting structure makes it very difficult to provide explainability. However, following the work in [38], we are able to enrich the Gradient Boosting Regressor mimic learner with the ability to provide explainability as follows: Given a state-action pair as an input of the mimic learner, we can measure the contribution of each state feature to the predicted Q-value. Therefore, our mimic learner is expected not only to mimic effectively the behavior of the DRL controller, but also, to give local and global explanations on its decisions.

Overall, we are interested in comparing the performance of XDQN with that of DQN in real-world environments where the latter has been state-of-the-art, and also designing appropriate experimental setups, aiming at studying XDQN interpretability. In so doing, we evaluate XDQN on real-world operational multi-agent experimental scenarios, pertaining to the demand-capacity balancing (DCB) problem of air traffic management (ATM), which we describe next.

### 4.1 Real-world demand-capacity problem setting

The current ATM system is based on time-based operations resulting in DCB [17] problems. To solve the DCB issues at the pre-tactical stage of operations, the ATM system opts for methods that generate delays and costs for the entire system. In ATM, the airspace consists of a set of 3D sectors where each one these is characterized by a specific capacity. This is the number of flights that cross the sector during a specific period (e.g. of 20 minutes). The main challenge of dealing with the DCB problem in ATM is to reduce the number of cases where the demand of airspace use exceeds its capacity. These cases are called *hotspots*.

Recent work has transformed the DCB challenge to a multi-agent RL problem by formulating the setting as a multi-agent MDP [17]. We follow the work and the experimental setup of [15–17, 31, 32] and encourage the reader to see the problem formulation [17] in details. In this setting, we consider a society of agents, where each agent is a flight (related to a specific aircraft) that needs to coordinate its decisions, so as to resolve hotspots that occur, jointly with other society agents. Agents' local states comprise 81 state variables related to: (a) the delay (in the range of 0, ..., maxDelay) set by the referring agent, (b) the number of hotspots in which the agent is involved in, (c) the sectors that it crosses, (d) the minutes that the agent is within each sector it crosses, (e) the periods in which the agent joins in hotspots in sectors, and (f) the minute of the day that the agent takes off. The tuple containing all agents' local states is the joint global state. Q-learning [33] agents has been shown to achieve remarkable performance on this task [15]. In our experiments, all agents share parameters and replay buffer and act independently.

A DCB scenario comprises multiple flights crossing various airspace sectors in a time horizon of 24h. This time horizon is segregated into simulation time steps. At each simulation time step (equal to 10 minutes of real time), given only the local state, each agent selects an action which is related to its preference to add ground delay regulating its flight, in order to resolve hotspots in which it participates. The set of local actions for each agent contains $|\text{maxDelay}+1|$ actions, at each simulation time step. We use maxDelay = 10. The joint (global) action is a tuple of local actions selected by the agents. Similarly, we consider local rewards and joint (global) rewards. The local reward is related to the cost per minute within a hotspot, the total duration of the flight (agent) in hotspots as well as to the delay that a flight has accumulated up to the simulation timestep [15].

### 4.2 Evaluation Metrics and Methods

For the evaluation of the proposed method, first, we make use of two known evaluation metrics: (a) *play performance* [19] of the online deep Q-network, and (b) *fidelity* [35] of the mimic learner. Play performance measures how well the deep Q-network performs with the mimic learner estimating its temporal difference targets, while fidelity measures how well the mimic learner matches the predictions of the online deep Q-network.

As far as play performance is concerned, we aim at minimizing the number of *hotspots*, the *average delay per flight* and the number of *delayed flights*. As for fidelity, we use two metric scores: (a) the *mean absolute error (MAE)* and (b) the *accuracy* score. Given a minibatch of states, we calculate the MAE of this minibatch for any action as the mean absolute difference between the Q-values estimated by the mimic learner and the Q-values estimated by the deep Q-network for that action. More formally, for a minibatch of states $D_s$, the $MAE_i$ of action $a_i$ is denoted as:

$$MAE_i = \frac{1}{|D_s|} \sum_{s \in D_s} |Q(s, a_i; \phi_X) - Q(s, a_i; \theta)| \quad (9)$$

It is worth noting that minimizing the MAE of the mimic learner is very important for training XDQN. Since deep Q-network updates its parameters $\theta$ by using the mimic model to provide the target Q-values, large MAEs can lead deep Q-network to overestimate bad states and understimate the good ones, and thus, find very diverging policies that completely fail to solve the task.

To calculate the accuracy score, again given a minibatch of states, for each state we compare the action selected by the mimic model and the online Q-network. Accuracy measures the percentage of the predictions of the two estimators that agree with each other, considering that both models select the action with the highest estimated Q-value.

Second, we design appropriate experiments and illustrate XDQN's *local* and *global* interpretability. We focus on providing aggregated interpretations, focusing on the contribution of features to local decisions and to the overall policy: This, as suggested by ATM operators, is beneficial towards understanding decisions, helping them to increase their confidence to the solutions proposed, and mastering the inherent complexity in such a multi-agent setting, as solutions may be due to complex phenomena that are hard to be traced [15]. Specifically, in this work, local explainability measures state features' importance on a specific instance (i.e. a single state-action pair), demonstrating which features contribute to the selection of a particular action over the other available ones. Global explainability aggregates feature importance on particular action selections over many different instances and aims to explain the overall policy of mimic learner. Third, we demonstrate global explainability of the DRL model through the whole training process, addressing the question of how a DRL model learns to solve the target task.

### 4.3 Experimental Scenarios and Settings

Experiments were conducted on three in total scenarios. Each of these scenarios corresponds to a date in 2019 with heavy traffic in the Spanish airspace. In particular, the date scenarios, on which we assess our models, are 20190705, 20190708 and 20190714. However, to bootstrap the training process we utilize a deep Q-network pretrained in various scenarios, also including 20190705 and 20190708. In the training process, the deep Q-network is further trained according to the method we propose. The experimental scenarios were selected based on the number of hotspots and the average delay generated in the ATM system within the duration of the day, which shows the difficulty of the scenario. We note that for each scenario we ran five separate experiments and average results.

Table 2 presents information on the three experimental scenarios. In particular, the flights column indicates the total number of flights (represented by agents) during the specific day. The initial hotspots column indicates the number of hotspots appearing in the initial state of the scenario. The flights in hotspots column indicates the number of flights in at least one of the initial hotspots. Note that all three scenarios display populations of agents of similar size, with 20190708 having the smaller population and the least initial hotspots.

**Table 1.** Comparison of testing performance of DQN and XDQN on the three experimental ATM scenarios

| Scenario | DQN Final Hotspots | DQN Average Delay | DQN Delayed Flights | XDQN Final Hotspots | XDQN Average Delay | XDQN Delayed Flights |
|----------|-------------------|-------------------|--------------------|--------------------|-------------------|---------------------|
| 20190705 | 38.4 | 13.04 | 1556.5 | 39.0 | 13.19 | 1618.54 |
| 20190708 | 4.6 | 11.4 | 1387.2 | 6.0 | 11.73 | 1331.58 |
| 20190714 | 4.8 | 10.72 | 1645.2 | 7.0 | 13.46 | 1849.49 |

**Table 2.** The three experimental Air Traffic Management (ATM) scenarios

| Scenario | Flights | Initial Hotspots | Flights in Hotspots |
|----------|---------|------------------|--------------------|
| 20190705 | 6676 | 100 | 2074 |
| 20190708 | 6581 | 79 | 1567 |
| 20190714 | 6773 | 92 | 2004 |

**Figure 1.** Episodic reward in the three evaluated ATM scenarios. Line plot showing reward increasing from approximately 0.4 to 0.8 over 1750 episodes for the three scenarios 20190708, 20190714, and 20190705.

### 4.4 Implementation Details

In our implementation setting we utilize a deep multilayer perceptron as the Q-network. In particular, we use an $\epsilon$-greedy policy, which at the start of exploration has $\epsilon$ equal to 0.9 decaying by 0.01 every 15 episodes until reaching the minimum of 0.04. The total number of episodes are set to 1600 and the update target frequency is set to 9 episodes. In the exploitation mode, we set $\epsilon$ equal to 0.04. We set the maximum depth of the Gradient Boosting Regressor equal to 45 and the number of minimum samples for a split equal to 20. We also use the mean squared error as the splitting criterion. To train a single decision tree for all different actions, we create a non binary splitting rule of the root based on the action size of the task, so that the state-action pairs sharing the same action match the same subtree of the splitting root. Empirically, we set the memory capacity of the experience replay for the mimic learner, i.e. the hyperparameter $K$, equal to the 1/20 of the product of three other hyperparameters, namely the total number of timesteps per episode (set to 1440), the update target frequency (set to 9) and the number of agents (set to 7000). Thus, $K$ is set to 4536000 steps.

### 4.5 Evaluation of play performance

Table 1 demonstrates the performance of DQN and XDQN on the three experimental scenarios. The final hotspots column indicates the number of unresolved hotspots in the final state: It must be noted that these hotspots may have emerged due to delays assigned to flights and may be different than the hotspots at the beginning of each scenario. The average delay per flight column shows the total minutes of delay imposed, divided by the number of flights in the specific scenario. The delayed flights column indicates the number of flights affected by more than four minutes of delay, as it is done by operators.

We observe that XDQN performs similar to DQN in all three evaluated metric scores. In particular, DQN slightly outperforms XDQN in terms of the final hotspots and average delay in all three scenarios. Nonetheless, XDQN achieves to decrease the number of the delayed flights in one scenario, while it demonstrates competitive performance on the others. Figure 1 shows the episodic reward of XDQN over time: XDQN manages to reach convergent behavior in all scenarios by retaining high episodic rewards.

### 4.6 Evaluation of fidelity

As discussed in Subsection 4.2, for the fidelity evaluation, we measure the mean absolute error (MAE) and the accuracy score. Given the DCB experimental scenarios, we train three different mimic models; namely X0705, X0708 and X0714. Table 3 reports the average MAE for each decided action over all mimic learning updates. We observe that all errors range in relatively small quantities, given that in testing, the absolute Q-values hovered around 200. As we highlighted above, this is very important for stabilizing the training process of XDQN, since we need very accurate mimic Q-value predictions, ideally equal to the ones generated by the deep Q-network.

To further assess the fidelity of XDQN mimic learner, Table 4 illustrates the average accuracy scores over all mimic learning updates. Since a Gradient Boosting Regressor mimic learner is a boosting algorithm, it produces sequential decision trees that can successfully seperate the state space and approximate well the predictions of the deep Q-network function. We observe that the mimic learner and the deep Q-network agree with each other to a very good extent; namely from approximately 81% to 91%. Therefore, we expect the mimic learner to be able to accumulate the knowledge from the deep Q-network with high fidelity.

**Table 3.** Evaluation of the average Mean Absolute Errors (MAE) of the trained mimic models over all mimic updates

| Action (Delay Option) | X0705 | X0708 | X0714 |
|----------------------|-------|-------|-------|
| 0 | 0.279 | 0.237 | 0.291 |
| 1 | 1.766 | 1.971 | 1.942 |
| 2 | 0.910 | 0.928 | 1.002 |
| 3 | 0.575 | 0.661 | 0.640 |
| 4 | 0.639 | 0.748 | 0.725 |
| 5 | 1.893 | 2.096 | 2.121 |
| 6 | 1.590 | 1.766 | 1.715 |
| 7 | 1.610 | 1.816 | 1.733 |
| 8 | 0.449 | 0.514 | 0.497 |
| 9 | 0.740 | 0.849 | 0.823 |
| 10 | 1.292 | 1.525 | 1.461 |

**Table 4.** The accuracy scores of mimic models

| Scenario | Accuracy (%) |
|----------|--------------|
| 20190705 | 88.45 |
| 20190708 | 81.89 |
| 20190714 | 90.88 |

### 4.7 Local and Global Explainability

In the DCB setting, it is important for the operator to understand how the system reaches decisions on regulations (i.e. assignment of delays to flights): This, as already pointed out, should be done at a level of abstraction that would allow them to increase their confidence to the solutions proposed, mastering the inherent complexity of the setting. Therefore, we are mainly interested in receiving explanations about which state features contribute to the selection of delay actions over the no-delay action (i.e. action equal to 0).

First, we demonstrate the ability of the mimic learner to provide local explainability. As already said, local explainability involves showing which state features contribute to the selection of a particular action over the other available ones in a specific state. To this aim, we work on pairs of actions - let $a_1$ and $a_2$ - and calculate the differences of feature contributions in selecting $a_1$ and $a_2$ in a single state. To highlight only the most significant differences, we focus only on those features whose differences are above a threshold. Empirically, we set this threshold equal to 0.5. Figure 2 illustrates local explainability on a given state in which action "2" was selected. Figure 2 provides the differences of feature contributions to the estimation of Q-values when selecting action "0" against selecting action "2" (denoted by "0-2"). We observe that the features that contributed more to the selection of the delay action "2" were those with index 32 (i.e. The sector in which the last hotspot occurs), 2 (i.e. the sector in which the first hotpot occurs) and 62 (i.e. the minutes that the flight spends crossing the last sector).

**Figure 2.** Illustration of significant differences of feature contributions to Q-value in selecting action "0" and action "2" in a single state, in which action "2" was selected. Bar chart showing positive and negative contribution differences across state features (indexed by 32, 2, 62, 63, 64, 67, 70, 66, 69, 68, 1, 33, 3, 0). Positive differences mean that the respective state features have a greater contribution to Q-value when action "0" is selected, rather than when action "2" is selected. Negative differences have the opposite meaning.

Finally, we demonstrate XDQN global explainability by aggregating the importance of features on particular action selections over many different state-action instances. In particular, we are interested in measuring the state feature contributions to the selection of delay actions (i.e. actions in the range $[1, 10]$) over the no-delay action (i.e. action "0") in the overall policy. To this aim, we work on all possible pairs of actions, with one action always being the no-delay action and the other one being a delay action, and average the differences of feature contributions to estimating the Q-value in selecting those actions over many different state-action instances with the same selected delay action. Table 5 shows the most significant state features in terms of average contribution difference (ACD) in selecting the no-delay action versus a delay action. To select those features, we initially filter the most significant ones, namely the features whose absolute ACD is greater than a threshold, for each action in the range $[1, 10]$ over the no-delay action (i.e. action "0"), and present the three most common features with positive and negative ACD. We observe that features with index 0, 1 and 3 contribute more to the selection of the no-delay action. On the contrary, features with indexes 64, 63 and 68 contribute more to the selection of a no-delay action.

**Table 5.** Demonstration of the most significant state features in terms of average contribution difference (ACD) in selecting the no-delay action versus a delay action. A positive ACD means that the corresponding state feature on average contributes more to the selection of the no-delay action "0". On the contrary, a negative ACD means that the corresponding state feature on average contributes to the selection of a delay action "1 - 10".

| Feature Index | Feature Meaning | ACD |
|---------------|-----------------|-----|
| 0 | Delay the corresponding flight has accumulated up to this point | Positive |
| 1 | Total number of hotspots the corresponding flight participates in | Positive |
| 3 | The sector in which the second hotspot the corresponding flight participates occurs | Positive |
| 63 | The minute of day the flight takes off given the delay (CTOT) | Negative |
| 64 | The minutes the flight remains in the first sector it crosses | Negative |
| 68 | The minutes the flight remains in the fifth sector it crosses | Negative |

Last but not least, we demonstrate how global explainability evolves through the training process, addressing the question of how a DRL model learns to solve the target task. To this aim, we measure the absolute average feature contribution (AAFC) to Q-value at different training episodes for the selection of each action. Figure 3 illustrates the evolution of global explainability for selecting the no-delay action and a delay action through 5 representative training episodes (360th, 720th, 1100th, 1400th and 1600th), in terms of AAFC to Q-value for the eight features with highest AAFC values in the final model (episode 1600) in the selection of the aforementioned actions. We observe that for both evaluated actions most of the features show an increasing/decreasing trend in their average contribution to Q-value over time, such as those with indices 0, 1 and 63. It is worth noting that although the features with indexes 0 and 1 have been highlighted as the most significant for the selection of the no-delay action, they have also significant but less contribution to a delay action as well.

**Figure 3.** Illustration of the evolution of features' contributions for selecting the no-delay action ("0") and a delay one ("2") through 5 representative training episodes (360th, 720th, 1100th, 1400th and 1600th) in terms of absolute average feature contribution (AAFC) to Q-value for the eight features with highest AAFC values in the final model (episode 1600) in the selection of the aforementioned actions. Two stacked bar charts showing per-episode average contributions for features (0, 1, 2, 62, 3, 32, 64, 33) for action 0 and features (0, 32, 1, 33, 62, 2, 63, 66) for action 2.

## 5 Related Work

Explainability in Deep Reinforcement Learning (DRL) is an emergent area whose necessity is related to the fact that DRL agents solve sequential tasks, acting in the real-world, in operational settings where safety, criticality of decisions and the necessity for transparency (i.e. explainability with respect to real-world pragmatic constraints [35]) is the norm. However, DRL methods use closed-boxes whose functionality is intertwined and are not interpretable: This may hinder DRL methods explainability. In this paper we address this problem by proposing an interpretable DQN method comprising two models which are trained jointly: An interpretable mimicking model and a deep policy model. The later offers training samples to the mimicking one and the former interpretable model offers target action values for the other to improve its predictions. At the end of the training process, the mimicking model has the capacity to provide high-fidelity interpretations to the decisions of the deep policy model. This is a specific example for interpreting DRL methods, according to the interpretable box design paradigm: This paradigm follows the conjecture (stated for instance in [28]) that there is high probability that the accuracy of closed boxes can be approximated by well designed interpretable models. In this work, following this paradigm, we train an interpretable model via mimicking, in parallel to the online Q network. Distillation could be another option [29], but in this work we explore mimicking as a process to train inherently interpretable models, such as decision trees.

There are many proposals for interpreting deep NNs models, through distillation and mimicking approaches. These approaches differ in several dimensions: (a) the targeted representation (e.g., decision trees in DecText [6], logistic model trees (LMTs) in reference [9], or Gradient Boosting Trees in reference [7]), (b) to the different splitting rules used towards learning a comprehensive representation, (c) to the actual method used for building the interpretable model (e.g., [9] uses the LogiBoost method, reference [6] proposes the DecText method, while the approach proposed in reference [7] proposes a pipeline with an external classifier, (d) on the way of generating samples to expand the training dataset. These methods can be used towards interpreting constituent individual DRL models employing (deep) NNs. The interested reader is encouraged to read a thorough review on these methods provided in [4, 12, 24, 28].

For DRL, authors in [19] introduce Linear Model U-trees (LMUTs) to approximate predictions for DRL agents. An LMUT is learned by an on-line algorithm that is well-suited for an active play setting. The use of LMUTs is compared against using CART, M5 with regression tree, Fast Incremental Model Tree (FIMT) and with Adaptive Filters (FIMT-AF). The use of decision trees as interpretable policy models trained through mimicking has been also investigated in [21], in conjunction to using a causal model representing agent's objectives and opportunity chains. However, the decision tree in this work is used to infer the effects of actions approximating the causal model of the environment. Similarly to what we do here, the decision tree policy model is trained concurrently with the RL policy model, assuming a model-free RL algorithm and exploiting state-action samples using an experience replay buffer. In [8] authors illustrate how Soft Decision Trees (SDT) [10] can be used in spatial settings as interpretable policy models. SDT are hybrid classification models of binary trees of predetermined depth, and neural networks. However their inherent interpretability is questioned given their structure. Other approaches train interpretable models other than trees, such as the Abstracted Policy Graphs (APGs) proposed in [34], assuming a well-trained policy model. APGs can offer interpretable representations of policies, concisely summarizing them, so that individual decisions can be explained in the context of expected future transitions.

Approaches following the interpretable box design paradigm also use use attention models for visual agents [1, 23], and interpretable policy models in a rather direct way [25, 27].

In contrast to the above mentioned approaches, XDQN can be applied to any setting with arbitrary state features, where the interpretable model formed using Gradient Boosting Regressors is trained jointly to a deep one through mimicking in an active play setting, following the DQN algorithm. It is worth noting that experimentally, instead of Gradient Boosting Regressors, we also tested naturally interpretable Linear Trees (such as LMUTs [19]); i.e. decision trees with linear models in their leaves). However, such approaches completely failed to solve the task, demonstrating quite low play performance with very large mean absolute errors.

As far as explanations are concerned, we opted for features' contributions to the Q-values, in a rather aggregated way, using the residue of each Gradient Boosting Regressor node, as done in [38]. This approach, as shown in [38], reports advantages over using well known feature importance calculation methods, avoiding linearity assumptions made by LIME [26] and bias in areas where features have high variance, and also avoiding taking all tree paths into account in case of outliers, as done by SHAP [20].

## 6 Conclusion and Future Work

In this work, we address the challenging issue of training interpretable policy models for solving real-world problems, such as the multi-agent demand-capacity balancing problem pertaining to air traffic management. To this aim, we have trained interpretable deep Q-learning models through mimic learning without requiring the existence of already well-trained deep Q-networks. Experimentally, we have shown that the proposed interpretable XDQN method, utilizing a Gradient Boosting Regressor as the mimic learner, performs on a par with DQN in terms of play performance whereas demonstrating high fidelity.

Further work on XDQN is to design, evaluate and compare various explainable mimic models that can effectively substitute the target Q-Network. Moreover, the proposed mimicking paradigm is generic, and can be naturally extended to many well-known DRL algorithms. Thus, future steps should also aim at benchmarking our methodology utilizing state-of-the-art DRL in various experimental settings.

## Acknowledgments

Acknowledgements will appear in the final version of this manuscript.

## References

[1] Raghuram Mandyam Annasamy and Katia Sycara. 2019. Towards Better Interpretability in Deep Q-Networks. *Proceedings of the AAAI Conference on Artificial Intelligence* 33, 01 (Jul. 2019), 4561–4569. https://doi.org/10.1609/aaai.v33i01.33014561

[2] Jimmy Ba and Rich Caruana. 2014. Do Deep Nets Really Need to be Deep?. In *Advances in Neural Information Processing Systems*, Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, and K.Q. Weinberger (Eds.), Vol. 27. Curran Associates, Inc.

[3] Osbert Bastani, Yewen Pu, and Armando Solar-Lezama. 2018. Verifiable Reinforcement Learning via Policy Extraction. In *Proceedings of the 32nd International Conference on Neural Information Processing Systems* (Montréal, Canada) (NIPS'18). Curran Associates Inc., Red Hook, NY, USA, 2499–2509.

[4] Vaishak Belle and Ioannis Papantonis. 2021. Principles and practice of explainable machine learning. *Frontiers in big Data* (2021), 39.

[5] Olcay Boz. 2002. Extracting Decision Trees from Trained Neural Networks. In *Proceedings of the Eighth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (Edmonton, Alberta, Canada) (KDD '02). Association for Computing Machinery, New York, NY, USA, 456–461.

[6] Olcay Boz. 2002. Extracting decision trees from trained neural networks. In *Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining*. 456–461.

[7] Zhengping Che, Sanjay Purushotham, Robinder Khemani, and Yan Liu. 2017. Interpretable Deep Models for ICU Outcome Prediction. *AMIA Annual Symposium Proceedings* 2016 (02 2017), 371–380.

[8] Youri Coppens, Kyriakos Efthymiadis, Tom Lenaerts, Ann Nowé, Tim Miller, Rosina Weber, and Daniele Magazzeni. 2019. Distilling deep reinforcement learning policies in soft decision trees. In *Proceedings of the IJCAI 2019 workshop on explainable artificial intelligence*. 1–6.

[9] Darren Dancey, Zuhair A Bandar, and David McLean. 2007. Logistic model tree extraction from artificial neural networks. *IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics)* 37, 4 (2007), 794–802.

[10] Nicholas Frosst and Geoffrey Hinton. 2017. Distilling a neural network into a soft decision tree. *arXiv preprint arXiv:1711.09784* (2017).

[11] Shixiang Gu, Ethan Holly, Timothy Lillicrap, and Sergey Levine. 2017. Deep reinforcement learning for robotic manipulation with asynchronous off-policy updates. In *2017 IEEE International Conference on Robotics and Automation (ICRA)*. 3389–3396.

[12] Riccardo Guidotti, Anna Monreale, Salvatore Ruggieri, Franco Turini, Fosca Giannotti, and Dino Pedreschi. 2018. A survey of methods for explaining black box models. *ACM computing surveys (CSUR)* 51, 5 (2018), 1–42.

[13] Hado van Hasselt, Arthur Guez, and David Silver. 2016. Deep Reinforcement Learning with Double Q-Learning. In *Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence* (Phoenix, Arizona) (AAAI'16). AAAI Press, 2094–2100.

[14] A. Kontogiannis, Dimitrios Kelesis, Vasilis Pollatos, Georgios Paliouras, and George Giannakopoulos. 2021. Tree-based Focused Web Crawling with Reinforcement Learning. *ArXiv* abs/2112.07620 (2021).

[15] Theocharis Kravaris, Konstantinos Lentzos, Georgios M. Santipantakis, George A. Vouros, Gennady L. Andrienko, Natalia V. Andrienko, Ian Crook, Jose Manuel Cordero Garcia, and Enrique Iglesias Martinez. 2022. Explaining deep reinforcement learning decisions in complex multiagent settings: towards enabling automation in air traffic flow management. *Applied Intelligence (Dordrecht, Netherlands)* (2022), 1–36.

[16] Theocharis Kravaris, Christos Spatharis, Alevizos Bastas, George A. Vouros, Konstantinos Blekas, Gennady L. Andrienko, Natalia V. Andrienko, and Jose Manuel Cordero Garcia. 2019. Resolving Congestions in the Air Traffic Management Domain via Multiagent Reinforcement Learning Methods. *ArXiv* abs/1912.06860 (2019).

[17] Theocharis Kravaris, George A. Vouros, Christos Spatharis, Konstantinos Blekas, Georgios Chalkiadakis, and Jose Manuel Cordero Garcia. 2017. Learning Policies for Resolving Demand-Capacity Imbalances During Pre-tactical Air Traffic Management. In *Multiagent System Technologies*, Jan Ole Berndt, Paolo Petta, and Rainer Unland (Eds.). Springer International Publishing, Cham, 238–255.

[18] Andrew Levy, George Dimitri Konidaris, Robert W. Platt, and Kate Saenko. 2019. Learning Multi-Level Hierarchies with Hindsight. In *ICLR*.

[19] Guiliang Liu, Oliver Schulte, Wang Zhu, and Qingcan Li. 2018. Toward Interpretable Deep Reinforcement Learning with Linear Model U-Trees. In *ECML/PKDD*.

[20] Scott M Lundberg and Su-In Lee. 2017. A unified approach to interpreting model predictions. *Advances in neural information processing systems* 30 (2017).

[21] Prashan Madumal, Tim Miller, Liz Sonenberg, and Frank Vetere. 2020. Explainable reinforcement learning through a causal lens. In *Proceedings of the AAAI conference on artificial intelligence*, Vol. 34. 2493–2500.

[22] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis. 2015. Human-level control through deep reinforcement learning. *Nature* 518, 7540 (Feb. 2015), 529–533. http://dx.doi.org/10.1038/nature14236

[23] Alex Mott, Daniel Zoran, Mike Chrzanowski, Daan Wierstra, and Danilo J. Rezende. 2019. Towards Interpretable Reinforcement Learning Using Attention Augmented Agents. https://doi.org/10.48550/ARXIV.1906.02500

[24] W James Murdoch, Chandan Singh, Karl Kumbier, Reza Abbasi-Asl, and Bin Yu. 2019. Definitions, methods, and applications in interpretable machine learning. *Proceedings of the National Academy of Sciences* 116, 44 (2019), 22071–22080.

[25] Larry D Pyeatt, Adele E Howe, et al. 2001. Decision tree function approximation in reinforcement learning. In *Proceedings of the third international symposium on adaptive systems: evolutionary computation and probabilistic graphical models*, Vol. 2. Cuba, 70–77.

[26] Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. 2016. "Why Should I Trust You?": Explaining the Predictions of Any Classifier. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (San Francisco, California, USA) (KDD '16). Association for Computing Machinery, New York, NY, USA, 1135–1144.

[27] Aaron M Roth, Nicholay Topin, Pooyan Jamshidi, and Manuela Veloso. 2019. Conservative q-improvement: Reinforcement learning for an interpretable decision-tree policy. *arXiv preprint arXiv:1907.01180* (2019).

[28] Cynthia Rudin, Chaofan Chen, Zhi Chen, Haiyang Huang, Lesia Semenova, and Chudi Zhong. 2021. Interpretable Machine Learning: Fundamental Principles and 10 Grand Challenges. (2021). https://doi.org/10.48550/ARXIV.2103.11251

[29] Andrei A. Rusu, Sergio Gomez Colmenarejo, Caglar Gulcehre, Guillaume Desjardins, James Kirkpatrick, Razvan Pascanu, Volodymyr Mnih, Koray Kavukcuoglu, and Raia Hadsell. 2015. Policy Distillation. https://doi.org/10.48550/ARXIV.1511.06295

[30] Tom Schaul, John Quan, Ioannis Antonoglou, and David Silver. 2015. Prioritized Experience Replay. https://doi.org/10.48550/ARXIV.1511.05952

[31] Christos Spatharis, Alevizos Bastas, Theocharis Kravaris, Konstantinos Blekas, George Vouros, and Jose Cordero Garcia. 2021. Hierarchical multiagent reinforcement learning schemes for air traffic management. *Neural Computing and Applications* (02 2021).

[32] Christos Spatharis, Theocharis Kravaris, George A. Vouros, Konstantinos Blekas, Georgios Chalkiadakis, Jose Manuel Cordero Garcia, and Esther Calvo Fernandez. 2018. Multiagent Reinforcement Learning Methods to Resolve Demand Capacity Balance Problems. In *Proceedings of the 10th Hellenic Conference on Artificial Intelligence* (Patras, Greece) (SETN '18). Association for Computing Machinery, New York, NY, USA, Article 2, 9 pages.

[33] Ming Tan. 1993. Multi-Agent Reinforcement Learning: Independent versus Cooperative Agents. In *ICML*.

[34] Nicholay Topin and Manuela Veloso. 2019. Generation of policy-level explanations for reinforcement learning. In *Proceedings of the AAAI Conference on Artificial Intelligence*, Vol. 33. 2514–2521.

[35] George A. Vouros. 2022. Explainable Deep Reinforcement Learning: State of the Art and Challenges. *ACM Comput. Surv.* (mar 2022). https://doi.org/10.1145/3527448 Just Accepted.

[36] Richard S. Zemel and Toniann Pitassi. 2000. A Gradient-Based Boosting Algorithm for Regression Problems. In *Proceedings of the 13th International Conference on Neural Information Processing Systems* (Denver, CO) (NIPS'00). MIT Press, Cambridge, MA, USA, 675–681.

[37] Xiangyu Zhao, Changsheng Gu, Haoshenglun Zhang, Xiwang Yang, Xiaobing Liu, Jiliang Tang, and Hui Liu. 2021. DEAR: Deep Reinforcement Learning for Online Advertising Impression in Recommender Systems. *Proceedings of the AAAI Conference on Artificial Intelligence* 35, 1 (May 2021), 750–758.

[38] Ángel Delgado-Panadero, Beatriz Hernández-Lorca, María Teresa García-Ordás, and José Alberto Benítez-Andrades. 2022. Implementing local-explainability in Gradient Boosting Trees: Feature Contribution. *Information Sciences* 589 (2022), 199–212. https://doi.org/10.1016/j.ins.2021.12.111

---

## BibTeX Citation

```bibtex
@misc{Kontogiannis2023xDQN,
  author        = {Kontogiannis, Andreas and Vouros, George},
  title         = {{XDQN}: Inherently Interpretable {DQN} through Mimicking},
  year          = {2023},
  eprint        = {2301.03043},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2301.03043}
}
```
