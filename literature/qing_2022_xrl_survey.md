REVIEW

# A Survey on Explainable Reinforcement Learning: Concepts, Algorithms, and Challenges

Yunpeng Qing, Shunyu Liu, Jie Song, Yang Zhou, Kaixuan Chen, Huiqiong Wang*, Mingli Song

*Corresponding author.

Y. Qing, J. Song, and H. Wang are with the College of Computer Science and Technology, Zhejiang University, Hangzhou 310027, China (e-mail: qingyunpeng@zju.edu.cn, sjie@zju.edu.cn, huiqiong_wang@zju.edu.cn).

S. Liu is with the College of Computing and Data Science, Nanyang Technological University, 639798, Singapore (e-mail: liushunyu@ntu.edu.sg).

Y. Zhou is with the School of Software Technology, Zhejiang University, Hangzhou 310027, China (e-mail: zmzhouyang@zju.edu.cn).

K. Chen and M. Song are with State Key Laboratory of Blockchain and Security, Zhejiang University, Hangzhou 310027, China (e-mail: chenkx@zju.edu.cn, brooksong@zju.edu.cn).

arXiv:2211.06665v6 [cs.LG] 29 Aug 2025

## Abstract

Reinforcement Learning (RL) is a popular machine learning paradigm where intelligent agents interact with the environment to fulfill a long-term goal. Driven by the resurgence of deep learning, Deep RL (DRL) has witnessed great success over a wide spectrum of complex control tasks. Despite the encouraging results achieved, the deep neural network-based backbone is widely deemed as a black box that impedes practitioners to trust and employ trained agents in realistic scenarios where high security and reliability are essential. To alleviate this issue, a large volume of literature devoted to shedding light on the inner workings of the intelligent agents has been proposed, by constructing intrinsic interpretability or post-hoc explainability. In this study, we conducted a comprehensive review of existing work on explainable RL (XRL) and introduced a new classification scheme, categorizing previous work into several main categories, namely, agent model explanation, reward explanation, state explanation, and task explanation, and further dividing them below. Some challenges and opportunities in XRL are discussed. This survey intends to provide a high-level summarization of XRL and to motivate future research on more effective XRL solutions. Corresponding open source codes are collected and categorized at https://github.com/Plankson/awesome-explainable-reinforcement-learning.

*Index Terms*—Reinforcement Learning, Explainability.

## I. Introduction

Reinforcement Learning (RL) [1] is a computational framework for training autonomous agents to purposefully understand environmental dynamics and influence future events through sequential interactions with the environment. This paradigm is inspired by human trial-and-error learning mechanisms, where interaction with the environment serves as a fundamental approach to learning without external guidance [2, 3, 4, 5]. In Technical, RL learns to map environmental states to actions to maximize cumulative rewards [6].

In recent years, the fast development of deep learning [7, 8] promotes the fusion of deep learning and reinforcement learning. Therefore, Deep Reinforcement Learning (DRL) [9, 10, 11, 12, 13] has emerged as a new RL paradigm. With the powerful representation capability of the deep neural network [14, 15, 16], DRL has achieved considerable performance in many domains, such as games [17, 18] and robotic tasks [19, 20, 21, 22, 23]. However, in complex real-world scenarios like autonomous driving [24, 25, 26, 27, 28] and power system dispatch [29, 30, 31, 32, 33, 34], both high performance and user-oriented explainability should be taken into account to ensure security and reliability. Therefore, the lack of explainability in DRL is one of the main bottleneck for employing DRL in the real world.

The opacity of conventional deep reinforcement learning (DRL) stems from the inherent complexity of deep neural networks (DNNs) [35], where high-dimensional parameter spaces and nonlinear transformations impede traceability. This architectural intricacy prevents the identification of features driving decisions or the mechanisms processing them [36], rendering DRL models functionally opaque "black boxes" [37]. Such opacity engenders two critical challenges. The first one is the trust barrier [38], where agent decisions conflict with human intuition without explainable justifications. For example, In autonomous navigation [39], abrupt route deviations (e.g., avoiding unperceived congestion) may confuse users despite rational objectives. And the second one is the knowledge integration limits [40, 41]. While encoding human expertise, the inability to map DRL representations to human-explainable concepts hinders effective integration [42, 43, 44, 45, 46]. Explainable AI (XAI) techniques, such as saliency maps [47, 48] and symbolic reasoning [49, 50, 51], partially address these issues by linking model internals to human-understandable rationales. Although various XAI techniques have been adopted to computer vision [52, 53, 54, 55] and natural language processing field [56, 57, 58], their direct adaptation to DRL remains challenging due to temporal decision dependencies and partial observability of RL tasks [59].

In the field of eXplainable Reinforcement Learning (XRL), many preliminary studies and surveys [59, 60, 61, 62, 63, 64, 65] have made effort to construct the XRL model and gained certain achievements in producing explanations. Early surveys [61, 64] directly adopted XAI's intrinsic vs. post-hoc dichotomy, categorizing XRL by explanation timing. While effectively distinguishing model-transparent approaches like decision trees [66] from post-hoc analyzers such as casual model [67], this framework fails to address RL-specific components like environment dynamics and reward model [59]. Subsequent works decomposed explanations along RL structural elements, [59] partitioned methods by target MDP components into state features, environment dynamics, and agent mechanism; [60] simply focused on agent preferences and goal influences; and [65] proposed policy, feature, and MDP-centric categories. However, these taxonomies suffer from overlapping boundaries. For example, programmatic policies [68] simultaneously address reward decomposition (feature-level) and long-term goals (policy-level), defying singular categorization. [63] introduced a causal architecture that spans from perceptions to dispositions. Although theoretically comprehensive, current implementations [39] operationalize only perception-action causation, representing a much simpler form of the causal framework. In conclusion, current XRL research faces three systemic challenges: (1) Absence of Standardization: Despite diverse conceptual frameworks mentioned above, the field lacks consensus on foundational elements, including the rigorous definitions of explainability in RL and integral evaluation protocols for XRL methods. (2) Taxonomy-Application Misalignment: Existing categorization schemes exhibit two key mismatches. The intrinsic/post-hoc dichotomies directly taken from the XAI taxonomy inadequately capture RL components, while the current component-based XRL taxonomy creates overlapping categories. This necessitates a unified taxonomy based on MDP formalism, explicitly distinguishing explanations of different MDP component targets from various XRL technical approaches. (3) Human-Centric Knowledge Integration Gap: While recent work demonstrates the efficacy of human-RL cooperation like trajectory annotations [69] and real-time corrections [70], current XRL taxonomies fail to formalize knowledge integration pathways. Standardized frameworks are needed to highlight the effectiveness of human prior knowledge for both high explainability and performance.

To advance the further development of XRL, this survey makes a more comprehensive and specialized review on XRL concepts and algorithms. We first clarify the concepts of RL explainability, then we give a systematic overview of the existing evaluation metrics for XRL, encompassing both subjective and objective assessments. We proposed a new taxonomy that categorizes current XRL works according to the central target of explanation: agent model, reward, state, and task, precisely capturing the central component in RL paradigm. Since making the whole RL paradigm explainable is currently difficult, all of the works turn to get partial explainability directly on components of RL paradigms. This taxonomy is much more specialized than the general coarse-grained intrinsic/post-hoc or global/local taxonomies in XAI, providing clearer distinctions among existing XRL methods and comprehensive illustration of RL decision-making process. Meanwhile, by assigning each method to a specific category aligning with its primary objective and specific implementation details, the taxonomy avoids ambiguity or confusion in the category process. Meanwhile, given that there is currently only a small amount of research on human knowledge-integrated XRL and its importance, we make an attempt on summarizing these works and organizing them into our taxonomy. As we know, few researchers look into this field of integrating human knowledge into XRL. Our work can be summarized below:

- We give a clear definition of RL explainability by summarizing existing literature on explainable RL. What's more, we also propose a systematic evaluation architecture of XRL from objective and subjective aspects.
- To make up for the shortcomings of lacking RL-based architecture in XRL community, we propose a new RL-based taxonomy for current XRL works. The taxonomy is based on the explainability of different central target of the RL framework: agent model, reward, state, and task. The taxonomy can be viewed in Figure 2.
- Recognizing that human knowledge-integrated XRL is an emerging research direction, we provide a systematic review of these approaches based on our new XRL taxonomy. This review illustrates how XRL frameworks incorporate human prior knowledge to enhance performance and improve explainability.

The remaining of this survey is organized as follows. In Section II, we recall the necessary basic knowledge of reinforcement learning. Next, we discuss the definition of RL explainability as well as giving some possible evaluation aspects for explanation and XRL approaches in Section III. In Section III-B2, we describe our categorization as well as provide works of each type and sub-type in detail, the abstract figure of our taxonomy can be viewed in Figure 2. Then we discuss XRL works that are combined with human knowledge according to our taxonomy in Section V. After that, we summarize current challenges and promising future directions of XRL in Section VI. Finally, we give a conclusion of our work in Section VII. The structure of this paper and our taxonomy work is shown in Figure 1.

**Figure 1.** An overview of the survey. We categorize existing explainable reinforcement learning (XRL) approaches into four branches based on the explainability of different parts in RL process: agent model, reward, state, and task. The more fine-grained categorization will be discussed detailedly in later sections. Each category is demonstrated with a part of representative works in the figure with different colors.

*Visual description:* A flowchart-like outline showing the survey structure with sections (Introduction, Background, XRL Definitions and Measurement, Explainability in RL, Human Knowledge for XRL, Challenge and Future Directions, Conclusion). Beneath, four colored columns (red, blue, yellow, purple) list categorized methods under State at Different Timestep (History Trajectory, Current Observation, Future Prediction), Agent Model-explaining (Self-explainable, Explanation-Generating), Reward-explaining (Reward Decomposition, Reward Shaping), and Task-explaining (Whole Top-to-Down Structure, Simple Task Decomposition).

## II. Background

Reinforcement Learning paradigm considers the problem of how an agent interacts with the environment to maximize the cumulative reward, where the reward is a feedback signal according to the response action of the agent in different states. Concretely, the interaction process can be formalized as a Markov Decision Process (MDP) [71]. An MDP is described as a tuple $M = \langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$, where $\mathcal{S}$ is the state space, $\mathcal{A}$ is the action space, $P : \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to [0,1]$ is the state transition function, $R : \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ is the reward function, and $\gamma \in [0,1]$ is a discount factor. At each discrete time step $t$, the agent observes the current state $s_t \in \mathcal{S}$ and chooses an action $a_t \in \mathcal{A}$. This causes a transition to the next state $s_{t+1}$ drawn from the transition function $P(s_{t+1}|s_t, a_a)$. Moreover, the agent can receive a reward signal $r_t$ according to the reward function $R(s_t, a_t)$. The core object of the agent is to learn an optimal policy $\pi^*$ that maximizes the expected discounted return $\mathbb{E}_\pi[G_t] = \mathbb{E}_\pi[\sum_{i=0}^{\infty} \gamma^i r_{t+i}]$. To tackle this problem, existing reinforcement learning methods can be categorized into two classes: value-based methods and policy-based ones.

### A. Value-based Methods

The value-based methods [9] tend to assess the quality of a policy $\pi$ by the action-value function $Q^\pi$ defined as:

$$Q^\pi(s,a) = \mathbb{E}_\pi\left[\sum_{i=0}^{\infty} \gamma^i r_{t+i} | s_t = s, a_t = a\right],$$

which denotes the expected discounted return after the agent executes an action $a$ at state $s$. A policy $\pi^*$ is optimal if:

$$Q^{\pi^*}(s,a) \geq Q^\pi(s,a), \forall \pi, s \in \mathcal{S}, a \in \mathcal{A}.$$

There is always at least one policy that is better than or equal to all other policies [1]. All optimal policies share the same optimal action-value function defined as $Q^*$. It is easy to show that $Q^*$ satisfies the Bellman optimality equation:

$$Q^*(s,a) = \mathbb{E}_{s' \sim P(\cdot|s,a)}\left[R(s,a) + \gamma \max_{a' \in \mathcal{A}} Q^*(s',a')\right].$$

To estimate the optimal action-value function $Q^*$, Deep Q-Networks (DQN) [9] uses a neural network $Q(s,a;\theta)$ with parameters $\theta$ as an approximator. We optimize the model by minimizing the following temporal-difference (TD) loss:

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\left[(y - Q(s,a;\theta))^2\right],$$

where $\mathcal{D}$ is the replay buffer of the transitions, $y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$ and $\theta^-$ represents the parameters of the target network. After the network converges, the final optimal policy can be obtained by a greedy policy $\pi(s) = \arg\max_{a \in \mathcal{A}} Q(s,a;\theta)$. Due to the encouraging results accomplished by DQN, several follow-up works [72, 73, 74, 75, 76, 77, 78, 79, 80, 81] progressively enlarged the family of DQN and has recently demonstrated extraordinary capabilities in multiple domains [82, 83, 84, 85]. However, while these value-based methods can handle high-dimensional observation spaces, they are restricted to problems with discrete and low-dimensional action spaces.

### B. Policy-based Methods

To solve the problems with continuous and high-dimensional action spaces, policy-based methods have been proposed as a competent alternative. One of the conventional policy-based methods is stochastic policy gradient (SPG) [1], which seeks to optimize a policy function $\pi_\phi : \mathcal{S} \times \mathcal{A} \to [0,1]$ with parameters $\phi$. SPG directly maximizes the expected discounted return as the objective $\mathcal{J}(\phi) = \mathbb{E}_{\pi_\phi}[\sum_{t=0}^{\infty} \gamma^t r_t]$. To update the policy parameters $\phi$, we can perform the gradient of this objective as follow:

$$\nabla_\phi \mathcal{J}(\pi_\phi) = \mathbb{E}_{s \sim \rho^\pi, a \sim \pi}\left[\nabla_\phi \log \pi_\phi(a|s) Q^\pi(s,a)\right],$$

where $\rho^\pi(s)$ is the state distribution and $Q^\pi(s,a)$ is the action value. To estimate the action value $Q^\pi(s,a)$, a simple and direct way is to use a sample discounted return $G$. Furthermore, to reduce the high variance of the action-value estimation while keeping the bias unchanged, a general method is to subtract an estimated state-value baseline $V^\pi(s)$ from return [1]. This yields the advantage function $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$, where an approximator $V(s; \theta)$ with parameters $\theta$ is used to estimate the state value. This method can be viewed as an actor-critic architecture where the policy function is the actor and the value function is the critic [10, 11, 86, 87, 88, 89, 90, 91].

On the other hand, the policy in the actor-critic architecture can also be updated through the deterministic policy gradient (DPG) [13, 92, 93] for continuous control:

$$\nabla_\phi \mathcal{J}(\mu_\phi) = \mathbb{E}_{s \sim \rho^\mu}\left[\nabla_a Q^\mu(s,a)|_{a=\mu_\phi(s)} \nabla_\phi \mu_\phi(s)\right].$$

where $\mu_\phi(s) : \mathcal{S} \to \mathcal{A}$ is a deterministic policy. Moreover, we directly instead approximate the action-value function $Q^\mu(s,a)$ with a parameterized critic $Q(s,a;\theta)$, where the parameters $\theta$ are updated using the TD loss analogously to the value-based case. By avoiding a problematic integral over the action space, DPG provides a more efficient policy gradient paradigm than the stochastic counterparts [92].

## III. Explainable RL Definitions and Measurement

This section establishes the foundation for integrating explainability into frameworks. Although numerous studies in this field have attempted to establish a precise definition of explainable RL, neither standardized criteria nor a clear consensus have emerged within the research community. Moreover, many current studies treat explainability as a subjective perception that does not require rigorous analysis. This conceptual ambiguity impedes both the development of the XRL research and the standardized evaluation metrics for XRL frameworks. Following a comprehensive review of existing literature, we provide a detailed conceptual analysis of XRL and categorize current evaluation metrics for XRL systems.

### A. Definition of XRL

In this section, we establish a unified definition of XRL, addressing inconsistencies in existing literature caused by divergent criteria proposed across scholarly works. A review of key studies reveals that many approaches frame explainability through human-centric interactions. For instance, [94] defines RL explainability as the degree to which humans can understand an agent decision, while [95] asserts it as the degree to which humans can consistently predict a model output. Concurrently, other works [64, 65] adopt the XAI taxonomy, distinguishing between interpretability (self-explanatory models) and explainability (user-oriented explanations), yet omit a comprehensive formal definition. Synthesizing these perspectives, XRL seeks to provide transparent explanations of decision-making processes in sequential contexts, enabling human understanding of both the process and outcomes, facilitating behavioral prediction, and ensuring reliable generation of user-aligned explanations. Based on the explanatory focus identified in surveyed studies, current XRL methods can be classified into the following categories:

(1) *Intrinsic Explainability via Agent Architecture*: The internal agent architecture can be designed with inherent transparency. This form of explainability measures the extent to which the decision-making logic and inner mechanisms remain explainable throughout its operational lifecycle (training and deployment). Representative architectures include decision trees [96], hierarchical agents [97], and logic-based rule systems [98], among others.

(2) *Extrinsic Explainability via Post-Hoc Analysis*: This category involves generating supplementary explanations to post-decision to clarify the agent action outputs. These explanations identify factors influencing the agent action selection, such as state-input dependencies or feature contributions. Common techniques include using saliency maps [99], attention distributions [100], *etc.*

Intrinsic explainability is an inherent property established during the agent design phase. In contrast, extrinsic explainability requires not only a fully developed agent but also input data and its execution traces, rendering it a post-hoc characteristic. Thus, the XRL field is fundamentally structured around these two classes of explanations: intrinsic (design-driven) and extrinsic (execution-dependent).

### B. Evaluation Framework

Having established a definition of explainability, we now turn to the evaluation of XRL. However, the field lacks standardized metrics for measuring the explainability of RL frameworks. While scholars have proposed preliminary frameworks, such as [101] introducing a three-tiered evaluation approach including application, human, and function-level criteria, and [102, 103] developing quantifiable metrics for explainable AI, no unified methodology has gained broad acceptance. Building on these efforts, we summarize these contributions into a structured framework tailored to XRL paradigm:

#### 1) Subjective Assessment

Subjective assessment evaluates XRL frameworks by analyzing human-generated mental models, which users construct to interpret the agent decision-making processes and structure [3, 4, 5]. By inversely assessing these mental models, researchers can gauge the effectiveness of explanations. Subjective evaluation seeks to quantify the accuracy of these mental representations; however, directly measuring internal cognitive states remains challenging. Current methodologies rely on indirect human feedback. Key metrics for subjective assessment include the following categories:

(1) *User Prediction (S.UP)*: This metric evaluates the alignment between human predictions and the RL agent actual decisions, serving as a proxy for the fidelity of the mental model derived from explanations. A quantitative way is to make the user predict the agent decision $a_\text{pred}$, compare to the real agent action $a_\text{RL}$, and calculate the hit rate $\sum_N |a_\text{pred} - a_\text{RL}|^2/N$, where N denotes the total number of trials [104, 105, 106, 107, 108]. Questionnaires are also widely applied to quantify participants' self-reported understanding of the agent decision logic and explanations [109, 110, 111, 112], often using Likert-scale responses to assess task-specific comprehension.

(2) *User Confidence (S.UC)*: User confidence measures the perceived reliability of explanations, distinct from user prediction metrics that assume full reliance on RL explanations. It reflects the persuasiveness of explainability in fostering trust and actionable reliance on the agent decisions [113, 114, 115]. Structured questionnaires are widely employed to quantify confidence levels [67, 116, 117, 118, 119]. Many researchers [120, 121] track the actions and intentions of users through interactions to measure their trust and reliance on the explanations. Additionally, response time $\Delta t$ is leveraged as an implicit metric to quantify explanation complexity, where prolonged deliberation suggests reduced clarity [114, 118].

(3) *Descriptivity (S.D)*: Many XRL literature employs descriptive case studies, such as state feature visualizations [48, 122] and programmatic policy examples [68, 123], to demonstrate explainability through illustrative scenarios. While such descriptions enhance persuasiveness by logically contextualizing agent decisions, they inherently lack quantitative validation. Notably, almost all XRL studies adopt this approach to assert explainability [66, 100, 124], which positions descriptiveness as an informal, subjective metric due to its reliance on qualitative evidence rather than empirical measurement.

Subjective assessment leverages human feedback to implicitly gauge the efficacy of explanations. Advances in questionnaire design methodologies [125, 126] now facilitate the development of structured questionnaires that systematically quantify S.UP and S.UC, making them a cornerstone of explainability evaluation [67, 127]. However, these metrics remain vulnerable to inherent biases arising from subjective human interpretation, such as cultural predispositions or task familiarity. To ensure robustness, researchers must implement rigorous controls, such as diverse participant sampling and blinded evaluation protocols, to mitigate biases and ensure equitable assessment conditions.

#### 2) Objective Assessment

Objective assessment focuses on quantitatively evaluating explanations using purely algorithmic outputs, thereby eliminating the need for human feedback. These objective metrics provide quantitative evaluations of the agent effectiveness and avoid the potential biases introduced by subjective human judgment.

(1) *Decision Performance (O.DP)*: Decision performance quantifies the cumulative rewards $G_\pi$ achieved by an RL agent $\pi$. It is crucial not to sacrifice performance in favor of explainability. Consequently, O.DP is a mandatory metric for all XRL paradigms, ensuring explanatory mechanisms do not degrade agent effectiveness [100, 123, 124, 128, 129, 130].

(2) *Fidelity (O.F)*: Fidelity measures the measures faithfulness of explanations to the agent actual behavior, ascertaining the extent to which the explanation accurately portrays the agent behavior [103]. The quantification of fidelity varies depending on the employed methodologies. Concerning the intrinsic explanation via agent architecture, fidelity is assessed by measuring the disparity between the inexplainable policy $\pi_\text{RL}$ and the intrinsic explainable policy $\pi_\text{XRL}$, denoted as $D(\pi_\text{RL}, \pi_\text{XRL})$ [131], where $D$ quantifies discrepancies in action distributions. For extrinsic explanations via post-hoc analysis, fidelity can be assessed by perturbing states highlighted in explanations and measuring the change of returns [132].

(3) *Robustness (O.R)*: Robustness quantifies the stability of explanations under perturbations [133, 134]. The assessment of robustness involves evaluating and validating the generated explanation under different conditions. Current evaluation methods add perturbation on state input [135, 136] or model parameters [137, 138, 139], utilizing the difference of generated explanation to determine whether the explanation remains stable.

Objective assessment provides a systematic framework for evaluating RL explainability by relying on algorithmic outputs. While it ensures reproducibility and mitigates human bias, its exclusion of human feedback may lead to imprecise measurement of explainability correctness. Meanwhile, current objective assessment metrics are limited to the specific type of explanation. The field of XRL currently lacks universally accepted measurement methods, necessitating further research attention to address this gap.

## IV. Explainability in RL

We construct the taxonomy of XRL based on the central target of explanation under the RL paradigm. We find that XRL researches focus on making certain aspects of the RL paradigm understandable while maintaining performance. The underlying model of RL can be segmented into several components, namely state, action, reward, agent model, and task. In our taxonomy, we organize existing explainable RL research based on these components: agent model-explaining methods that show the decision-making mechanism of agent, reward-explaining methods that show how different factors within reward function influence agent policy, state-explaining methods that illustrate the state features at different time stages affecting agent behavior, and task-explaining methods that explain how the agent divide the complex task into subtasks and complete them in long term. These XRL methods are further categorized based on the employed technique. We present this taxonomy in Figure 2. The detailed descriptions of these methods are presented in the following subsection.

**Figure 2.** Diagrams of different types of XRL frameworks. These diagrams illustrate how different types of XRL make different parts of the RL model produce explanations. (a) constructs the agent on an explainable model to illustrate the inner mechanism. (b) reconstructs reward function $r$ towards an explainable one $r'$, which is constructed by quantifying the quantitative impact of various key factors $\{w_i\}$. (c) adds a state analyzer submodule to quantify the influences of state features for each state input $s$. (d) gets an architectural level explainability in complex tasks by task division and subtask signal $g$.

*Visual description:* Four side-by-side schematic diagrams labeled (a) Agent Model-explaining, (b) Reward-explaining, (c) State-explaining, (d) Task-explaining. Each panel shows boxes for an Agent and Environment (Env) with arrows indicating data flow (state $s$, action $a$, reward $r$). The agent-model panel inserts an "Explainable Model" inside the Agent box; reward panel inserts a "Reward Reconstructor" with key-factor weights; state panel inserts a "State Analyzer" with a state-feature quantization grid; task panel splits the agent into High-level Agent and Low-level Agent communicating via subtasks and goal $g$.

### A. Agent Model-explaining

Classical RL frameworks primarily aim to optimize the decision-making capability of agent without focusing on the internal decision-making logic. In contrast, agent model-explaining XRL methods achieve high-performing agents while also extracting the underlying decision-making mechanism of agent model to generate explanations. We categorize current agent model-explaining XRL methods into two types: self-explainable and explanation-generating techniques. Self-explainable methods aim to generate explanations by the transparent inner agent model itself, whereas explanation-generating techniques provide explanations based on predetermined reasoning mechanisms.

#### 1) Self-explainable

A self-explainable model is intentionally designed to be self-explanatory throughout the training process, which is accomplished by imposing limitations on the complexity of the model structure [64, 157]. Such a model is also known as an intrinsic model [64], as it embodies transparency and ease of understanding. The explanation logic is inherently integrated within the agent model itself. Our work provides a comprehensive overview of the current self-explainable agent model in XRL field and categorizes them into two types based on the target of explainable agent model: value and policy. This classification can be found in Table I.

**TABLE I:** Self-explainable agent models in XRL approaches. The venue with * denotes the paper published at the workshop of that venue.

| Type | Explanation | Reference | Year | Venue | Evaluation |
|---|---|---|---|---|---|
| Value-based | Decision Tree | [66] | 2018 | ECML-PKDD | O.DP, S.D |
| Value-based | Formula Expression | [140] | 2012 | DS | O.DP, S.D |
| Policy-based | Programmatic policy | [123] | 2018 | ICML | O.DP, S.D |
| Policy-based | Programmatic policy | [68] | 2019 | NeurIPS | O.DP, S.D |
| Policy-based | Programmatic policy | [141] | 2020 | NeurIPS | O.DP, S.D |
| Policy-based | Programmatic policy | [142] | 2021 | NeurIPS | O.DP, S.UP, S.D |
| Policy-based | Symbolic policy | [143] | 2018 | EAAI | O.DP, S.D |
| Policy-based | Symbolic policy | [144] | 2019 | GECCO | O.DP, S.D |
| Policy-based | Symbolic policy | [145] | 2021 | ICML | O.DP, S.D |
| Policy-based | Symbolic policy | [146] | 2024 | NeurIPS | O.DP, S.D |
| Policy-based | Fuzzy controller | [147] | 2017 | EAAI | O.DP, S.D |
| Policy-based | Fuzzy controller | [148] | 2019 | ECML-PKDD* | O.DP, S.D |
| Policy-based | Fuzzy controller | [149] | 2023 | TFS | O.DP, S.D |
| Policy-based | Logic rule | [98] | 2019 | ICML | O.DP, S.D |
| Policy-based | Logic rule | [150] | 2019 | arXiv | O.DP, S.D |
| Policy-based | Logic rule | [151] | 2020 | arXiv | O.DP, S.D |
| Policy-based | Decision Tree | [152] | 2011 | ICML | O.DP, S.D |
| Policy-based | Decision Tree | [128] | 2018 | NeurIPS | O.DP, O.R, S.D |
| Policy-based | Decision Tree | [153] | 2019 | AAAI | O.DP, O.F, S.D |
| Policy-based | Decision Tree | [154] | 2019 | arXiv | O.DP, S.D |
| Policy-based | Decision Tree | [155] | 2019 | arXiv | O.DP, S.D |
| Policy-based | Decision Tree | [130] | 2021 | AAAI | O.DP, S.D |
| Policy-based | Decision Tree | [156] | 2022 | arXiv | O.DP, S.D |

##### a) Value-based

The Q-value in RL measures the expected discounted sum of rewards that an agent would receive from a given state $(s,a)$. This value can also be employed to construct a deterministic or energy-based policy [9, 12, 93]. Thus many value-based agent model-explaining XRL frameworks primarily concentrate on the Q-value model.

The Linear Model U-Tree (LMUT) [66] combines the concepts of imitation learning (IL) and continuous U-tree (CUT) [158], which can be considered as an advanced version of CUT for value function estimation. Similar to a typical decision tree, LMUT internal nodes store dataset features, while the leaf nodes represent a partition of the input space. However, in LMUT, each leaf node contains a linear model that approximates the Q-value instead of a simple constant. The Q-value approximation, denoted as $Q^{UT}_{N_i}$, is obtained from the linear model within the corresponding LMUT leaf node. This approximation acts as an explanation by quantifying the individual effects of different features in LMUT. The researchers outline the training process for LMUT, which involves two steps: (1) data gathering phase counting all transitions $T$ within LMUT and modifying Q-values; (2) node splitting phase by Stochastic Gradient Descent (SGD). When SGD fails to yield sufficient improvement on specific leaf nodes, the framework splits those leaf nodes to disentangle the mixed features. Experimental results demonstrate that LMUT achieves comparable performance to neural network-based baselines across various environments.

[140] introduced a search algorithm exploring the space of simple closed-form formulas to construct Q-value. The variables within the formula represent the abstractions of state and action components, while the operations performed on these variables are unary and binary mathematical operations. The resulting policy is a greedy deterministic policy that selects the action with the maximum Q-value. The different operations highlight the varying effects of variables on the Q-value, thereby ensuring explainability. However, this method struggles with combinatorial explosion during the search process. Therefore, the total number of variables, constants, and operations is restricted to a small number.

##### b) Policy-based

Policy representation is considered a more direct approach compared to the Q-value, as it explicitly reflects the agent decision-making capability. In this section, we present a comprehensive analysis of potential policy models proposed in the existing literature. Specifically, Figure 3 illustrates several representative approaches of this kind.

**Figure 3.** Examples of Self-Explainable Policy Architectures: (a) Programmatic reinforcement learning frameworks [68, 123, 141, 142]; (b) Decision tree policy construction by transforming [128, 156] or shaping [66, 155].

*Visual description:* Panel (a) shows a Programmatic Policy Evaluator and Explainable Agent (environment loop with state $s_t$, action $a_t$), and a Programmatic Policy Generator featuring an Interpretable Programmatic Policy Space producing program-like code. Panel (b) shows two flows: (top) DT Policy Transform—an Environment to a DNN-based Policy and Decision Tree Training/Extraction; (bottom) DT Policy Shaping—an Environment connected to a DT Policy with Leaf Node Maintain/Update splitting steps that incorporate Q-value updates $Q(L_i, a_t) \leftarrow Q(L_i, a_t) + \Delta Q$.

Programmatic Reinforcement Learning (PRL) involves utilizing a program as the representation of the policy, enabling intrinsic explainability through logic rules within the program [68, 123, 141, 142]. This approach operates through two components, as shown in Figure 3a, programmatic policy generator and programmatic policy evaluator. The former updates the current programmatic policy vector within a fixed programmatic space, generating a programmatic policy through vector decoding. The latter involves simulating the generated programmatic policy to optimize the current policy in a one-step fashion. The main challenge in PRL lies in selecting an interpretable programmatic policy space. [123] constructs programmatic policy using a domain-specific high-level programming language based on historical data utilization, allowing for a quick understanding of past interactions influence. They propose Neurally Directed Program Search (NDPS) to construct such policy. NDPS employs DRL method to find a neural policy that approximates the target policy, followed by iterative policy updates through template enumeration using *Bayesian Optimization* [159] or satisfiability modulo theory to optimize the parameters. However, [68] argue that this method is highly suboptimal and propose a new framework based on mirror descent-based meta-algorithm for policy search in the space combining neural and programmatic representations. For multi-agent communication, [141] synthesize programmatic policies based on the generated communication graph of the agents. Additionally, [142] learn a latent program space to improve the efficiency of programmatic policy search. Furthermore, the learned latent program embedding can be transferred and reused for other tasks.

Formulaic expressions are also able to represent policies instead of value functions. Such policies are referred to as symbolic policies, comprising simple and concise symbolic operations that provide intrinsic explainability through succinct mathematical expressions [143, 144, 160]. However, searching the entire symbolic space to find the best fit is generally considered a computationally complex problem known as NP-hard [160]. To address this challenge, several studies [143, 144] utilize genetic programming for model-based batch RL to maintain a population of symbolic expression individuals as well as evolutionary operations. In contrast to a direct search for a symbolic policy, recent methods [145, 146] utilize inexplicable DNN-based anchor policy to generate an explainable symbolic policy.

The policy can be constructed based on the combination of several fuzzy controllers [59, 147, 148, 149]. Specifically, the agent policy, denoted as $\pi(a|s)$, can be represented as a Gaussian Distribution $\mathbf{N}(a|K\varphi(s), \Sigma)$, with $K$ stacking actions for the cluster centers, $\varphi(s)$ returning a weight vector based on the distance, and $\Sigma$ being a state-independent full variance matrix. By evaluating the distance to cluster centers, the influence of different centers on actions can be analyzed. [148, 149] employ policy gradient method to facilitate training of such policies. Additionally, [147] applies parameter training on a world model to construct fuzzy RL policies.

First-order logic (FOL) serves as a foundational language to depict entities and relationships [161]. It underpins the policy representation in Neural logic RL (NLRL), which fuses policy gradient techniques with differentiable inductive logic programming [162]. The seminal work [98] on NLRL shows the enhancement of explainability through weighted logic rules, clarifying the rationale for action choices. Later advances assign weights to rule atoms, leveraging genetic programming technique for policy formula learning from historical interactions [150, 151]. This evolution positions NLRL as a tool for deriving potent policies with superior explainability and generalizability.

Decision Tree (DT) for XRL has been categorized into policy-based and value-based strategies. While the linear model U-tree stands out as a DT variant in value-based XRL, DT-based policies are utilized to select actions based on distinctive features derived from DTs, thereby providing interpretable observations within RL tasks [128, 130, 152, 153, 156]. Frameworks for policy-based DTs are delineated in Figure 3b. With the efficacy of humans in acquiring insights on DDN via DRL, transforming DNN policies to DT policies is a promising strategy. For this, Verifiability via Iterative Policy Extraction (VIPER) [128] employs model distillation [163] to transmute pre-trained DNN policies to DTs using optimal policy trajectories. Techniques like Q-DAGGER [152] and MAVIPER [156] further refine and extend VIPER to more scenarios like multi-agent settings. Iterative Bounding MDP (IBMDP) [130] and policy summarization [153] also focus on extracting interpretable policies from DNNs. Another avenue pursues direct DT policy shaping. By maintaining weight information at the leaf nodes of DT to approximate Q-value and performing leaf node splits at specific stages, a high-performance DT policy can be obtained. [154] employ evolutionary algorithms to evolve the DT structure while applying Q-learning to the leaf nodes. [155] propose Conservative Q-Improvement (CQI), which uses lazy updating and expands the tree size only when the approximation of future discount rewards exceeds a specified threshold.

#### 2) Explanation-generating

Explanation-generating methods utilize an explicit auxiliary reasoning mechanism to facilitate the automatic generation of explanations. The development of such a mechanism relies on emulating the human cognitive processes involved in learning novel concepts. Below, we present a selection of influential works that demonstrate these types of explainability, summarized in Table II.

**TABLE II:** Explanation-generating agent models in XRL.

| Explanation | Reference | Year | Venue | Evaluation |
|---|---|---|---|---|
| Counteract | [67] | 2020 | AAAI | O.DP, S.UP, S.UC, S.D |
| Counteract | [164] | 2021 | AI | O.DP, O.F, S.UP, S.UC, S.D |
| Counteract | [165] | 2022 | NeurIPS | O.DP, O.F, S.D |
| Counteract | [166] | 2022 | AAAI | O.DP, S.D |
| Counteract | [167] | 2023 | IJCAI | O.DP, O.F, S.D |
| Counteract | [168] | 2024 | NeurIPS | O.DP, O.F, O.R |
| Counteract | [169] | 2024 | AAAI | O.DP, S.UP, S.UC |
| Instruction | [67] | 2020 | HAI | O.DP, S.D |
| Instruction | [165] | 2021 | ICONIP | O.DP, S.D |
| Answer to query | [51] | 2017 | HRI | O.DP, S.D |
| Answer to query | [170] | 2022 | IJCAI | O.DP, S.UP, S.UC, S.D |
| Answer to query | [171] | 2023 | IJCAI | O.DP, S.UP, S.UC, S.D |
| Verify | [172] | 2019 | SIGCOMM* | O.DP, S.D |
| Verify | [173] | 2019 | PLDI | O.DP, O.R, S.D |
| Verify | [174] | 2020 | NeurIPS | O.DP, O.F, S.D |
| Verify | [175] | 2022 | CAV | O.DP, O.R, S.D |

Counterfactual explanations answer the question of "why perform X" by explaining "why not perform Y" (the counterfactual of X) [67, 164, 165, 166, 167, 169]. [164, 167, 168] crafts counterfactual states $s'$ that have minimal divergence from the current state $s$, yet lead to distinct agent actions, while [165] emphasize the Q-value discrepancies between counterfactual action pairs. On the causal front, [67] leverages the causal model to grasp the world through distinct variables and potential interrelationships to further elucidate both action reasons and counterfactuals. However, the rigidity of the causal model hampers its adaptability, and the method can only be implemented in discrete action space. To bridge this gap, [167, 168] melds attention-driven causal techniques, facilitating causal influence quantification in continuous action spaces and illuminating the long-term repercussions of such actions. Meanwhile, [169] improves the explainability of counterfactual explanations by visually comparing the actions chosen by the agent with the counterfactual outcomes.

Instruction-based Behavior Explanation (IBE) [176, 177] enhances explainability with formal agent instructions. In basic IBE [176], the agent acquires the capability to explain the behavior with instruction. The learning process includes estimating the target of the agent actions by simulation and acquiring a mapping from the target of the agent actions to the expressions with a clustering approach. However, it is difficult to divide the state space to assign an explanation signal in much more complex tasks. Consequently, in their advanced IBE approach [177], DNN model is employed to construct the mapping, enabling its adaptability to intricate state space.

With the pre-defined query template, the agent is able to explain its inner mechanism by answering [51, 170, 171]. [51] introduce a method wherein queries are mapped to decision-making statements via templates. They harness a graph search algorithm to pinpoint relevant states and summarize attributes in natural language. Although the generated policy explanations align with the expert expectations, their reliability in more complex tasks remains unverified. To address this, [170] extend this approach to MARL by proposing Multi-agent MDP (MMDP), an abstraction of MARL policy. They first transform the learned policy into an MMDP setting by a specified set of feature predicates to address "When, Why not, What" questions in MARL. However, the question templates ignore the task process. To address this drawback, [171] encodes the temporal query and compares it with the transition model to address temporal queries regarding the task order, resulting in contractive explanations.

Formal verification techniques bolster safety in RL paradigms. Verily [172], for instance, accomplishes verification with the satisfiability modulo theories verification engine for DNN. If the verification result is negative, Verily can generate a counterexample through logical verification to explain the discrepancy. This counterexample can in turn guide the updates of the DNN parameter. [174] adopt a similar approach, employing the idea of mirror descent shared by [68]. They perform updating and projecting steps between the neurosymbolic class and restricted symbolic policy class to enable efficient verification. Furthermore, [173] propose a verification toolchain to ensure the safety of learning DNN-based policies. Likewise, [175] present a verification-in-the-loop framework that iteratively trains and refines the abstracted state space using counterexamples if verification fails.

#### 3) Summarization

Self-explainable methods predominantly utilize S.D as an assessment criterion. These methods leverage the agent model itself to provide explanations and illustrate these explanations through case studies. While case studies showcasing intrinsic explainable policy instances, such as decision tree [66, 153] and programmatic policies [141, 142], are intuitively logical and reasonable, the absence of quantitative measurements limits their persuasiveness in demonstrating explainability. Instead, explanation-generating methods utilize various evaluation assessments such as S.UP, S.UC, O.F, and O.R, which are more formal and quantitative. Self-explainable methods primarily rely on agent architecture to provide intrinsic explanations, which are highly formulaic and objectively described in detail. Therefore, objective assessments of O.F and O.R offer a more precise evaluation of the architecture-based explanation. Conversely, explanation-generating methods need more subjective assessments of S.UP and S.UC from human feedback since the core of explanation-generating methods is the extrinsic reasoning mechanisms conducting logical reasoning, which can be effectively evaluated via human participants with strong inferential abilities.

### B. Reward-explaining

Reward-explaining methods reconstruct an explainable reward function through the quantification of various key factors that are instrumental in accomplishing the task, such as the degree of multi-agent cooperation and the features of the final goal. The reward function plays a crucial role in RL tasks, serving as the primary factor for estimating actions in the short term and in the long term. The reward-explaining method involves explicitly designing an explainable reward function to provide explanations for the critical factors of the task. Building upon this notion, we categorize current reward-based XRL work into two types: reward shaping and reward decomposition. The approaches within each category are summarized in Table III.

**TABLE III:** Reward-explaining methods in XRL.

| Type | Reference | Year | Venue | Evaluation |
|---|---|---|---|---|
| Reward decomposition | [180] | 2018 | AAAI | O.DP, S.D |
| Reward decomposition | [178] | 2019 | IJCAI* | O.DP, S.D |
| Reward decomposition | [181] | 2020 | AAAI | O.DP, S.D |
| Reward decomposition | [182] | 2021 | SIGKDD | O.DP, S.D |
| Reward decomposition | [179] | 2023 | ICLR | O.DP, S.D |
| Reward shaping | [183] | 2019 | HRI | O.DP, S.D |
| Reward shaping | [184] | 2019 | AAAI | O.DP, S.D |
| Reward shaping | [185] | 2020 | AAAI | O.DP, S.D |
| Reward shaping | [186] | 2021 | AAAI | O.DP, S.D |
| Reward shaping | [187] | 2022 | NeurIPS | O.DP, S.D |
| Reward shaping | [188] | 2022 | AAAI | O.DP, S.D |
| Reward shaping | [189] | 2022 | NeurIPS | O.DP, S.D |

#### 1) Reward Decomposition

Reward decomposition methods aim to explain the inexplicable value of a reward function by breaking it down into several distinct parts that represent different aspects. The original reward function is a single scalar value influenced by multiple implicit factors. By decomposing the reward function, we can analyze the influence and relationships among these implicit factors.

Horizontal reward decomposition [178] decompose the reward function in the MDP horizontally as $\vec{R} : \mathcal{S} \times \mathcal{A} \to \mathbb{R}^{|\mathcal{C}|}$, where $\mathcal{C}$ represents the number of reward components. Subsequently, the Q-value is also decomposed as $Q^\pi(s,a) = \sum_{c \in \mathcal{C}} Q^\pi_c(s,a)$. To explain the decomposition, the authors primarily focus on comparing pairwise actions. One straightforward approach is Reward Difference eXplanation (RDX) in the form of $\Delta(s, a_1, a_2) = \vec{Q}(s, a_1) - \vec{Q}(s, a_2)$. RDX informs experts about which components may have an advantage over other factors, but it does not identify the most significant component. Moreover, RDX may offer limited explanations when the number of factors increases. To address this, the authors propose another form of explanation called Minimal Sufficient eXplanation (MSX). MSX is a two-tuple (MSX$^+$, MSX$^-$), with MSX$^+$ selecting the minimal set of components where the total $\Delta(s, a_1, a_2)$ surpasses a dynamic threshold, while MSX$^-$ checks the summation of $-\Delta(s, a_1, a_2)$ with the other threshold. Similarly, in vision-based RL, [179] reconstruct the multidimensional patch reward of image samples by assessing the expertise of each local patch. The patch reward effectively captures features, serving as a fine-grained measure of expertise and a tool for visual explainability.

For multi-agent tasks, the widely adopted paradigm is Centralized Training with Decentralized Execution (CTDE), which allows agents to train based on their local view while a central critic estimates the joint value function. The primary challenge of CTDE lies in assigning credit to each agent. One effective tool for assigning credit to each local agent is the Shapley value [190], which represents the average contribution of an entity (or in the context of multi-agent RL, a single agent) across different scenarios. To compute the Shapley value, we can measure the change in the output when the target feature or agent is considered. Considering that the computational costs growing exponentially with the number of agents make it hard to approximate in complex environments, [180] employs a counterfactual advantage function for local agent training. Nevertheless, this method neglects the correlation and interaction between local agents, leading to failure in intricate tasks. To address this limitation, [181] combine the Shapley value with the Q-value and perform reward decomposition at a higher level in multi-agent tasks to plan global rewards rationally: individual agents with greater contributions receive more rewards. Therefore, this method assigns credit to each agent, enabling an explanation of how the global reward is divided during training and how much each agent contributes. However, this network-based method relies highly on the assumption that local agents take actions sequentially without considering the synchronous running of agents. In contrast, [182] employ counterfactual-based methods to quantify the contribution of each agent, which proves to be more stable.

#### 2) Reward Shaping

Direct synthesis of explainable reward functions provides an alternative pathway, as demonstrated by notable research efforts [183, 184, 185, 186, 187, 188]. These approaches establish reward representations through explicit structural alignment with task objectives, eliminating the need to explain the pre-existing reward function components.

Building upon the interactions between the agent and humans, [187] present a reward-shaping approach that modifies the original sparse rewards with human instruction goals into a dense explainable reward. Similarly, The study conducted by [189] employs a meta-learning approach to acquire multiple goal maps, subsequently modifying the reward function by aggregating diverse goal map weights. [183] propose a framework that employs the Partially Observable MDP (POMDP) model to approximate collaborator understanding of joint tasks. The authors continually modify and correct the reward function in order to achieve this objective. If they discover a more plausible reward function, they evaluate whether the advantage of adopting it outweighs the cost of abandoning the previous function. Subsequently, a repairing representation is generated if the newly found reward function proves beneficial.

To enhance the explainability of complex tasks, the adoption of multi-level rewards is a viable approach. Unlike task decomposition, where decomposed reward reflects the actual rewards from the environment, multi-level reward encompasses both extrinsic rewards from the environment and intrinsic rewards aimed at facilitating comprehension and explanation. [184] introduce a two-level framework comprising extrinsic reward standing for real rewards within RL environments and intrinsic reward representing the achievement of inner task factors. Symbolic Planning approaches are utilized to maximize the intrinsic reward. Meanwhile, compared to [184] utilizing the pre-defined intrinsic reward to generate plans, [188] extend their work by automatically learning the intrinsic reward, enabling faster convergence compared to the original approach. For the task of temporal language bounding in untrimmed videos, [185] propose a tree-structured progressive RL technique: while the leaf policy receives the extrinsic reward from the external environment, the root policy, which does not directly interact with the environment, evaluates rewards intrinsically based on high-level semantic branch selections. Meanwhile, to address challenges with defining intrinsic rewards resulting in inferior performance compared to extrinsic rewards, [186] introduce the concept of intrinsic mega-rewards to enhance the agent individual control abilities, including direct and latent control. A relational transition model is formulated to enable the acquisition of such control abilities, yielding superior performance compared to existing intrinsic reward approaches.

#### 3) Evaluation

All of the surveyed reward-explaining methods utilize S.D to assess RL explainability. These explanations in the original paper case study provide visualizations comparing the influences of different aspects of the task within the reward value, offering illustrative examples for case studies in the paper. In order to perform quantitative analysis of these reward-explaining methods in the future, we posit that objective assessment O.F is more required for future research. Different quantitative influences of factors are the backbone of reward explanation, which should be strictly measured for its fidelity objectively. A possible way to measure O.F is manually updating the reward against the reward explanation to see whether the performance falls rapidly [132].

### C. State-explaining

State-explaining methods generate extrinsic explanations based on observation from the environment, which incorporate a state analyzer that allows for the simultaneous analysis of the different state features significance. According to the time stage of different states to construct the explanation, we divide current state-explaining methods into three types: historical trajectory-based methods focusing on past significant states, current observation-based methods emphasizing important features of current state, and future prediction-based methods inferring future states. We provide a brief review of the relevant literature on state-level explainability in Table IV.

**TABLE IV:** State-explaining methods in XRL.

| Temporal perspective | Reference | Year | Venue | Evaluation |
|---|---|---|---|---|
| Historical trajectory | [191] | 2014 | AAAI | O.DP, O.R, S.D |
| Historical trajectory | [192] | 2018 | SSCI | O.DP, S.D |
| Historical trajectory | [127] | 2018 | AI | O.DP, S.UP, S.UC, S.D |
| Historical trajectory | [193] | 2021 | TCSS | O.DP, S.D |
| Historical trajectory | [132] | 2021 | NeurIPS | O.DP, O.F, O.R, S.D |
| Historical trajectory | [194] | 2021 | CIM | O.DP, S.D |
| Historical trajectory | [195] | 2022 | AAAI | O.DP, S.D |
| Historical trajectory | [196] | 2023 | ICLR | O.DP, S.UP, S.UC, S.D |
| Historical trajectory | [197] | 2023 | ICLR | O.DP, O.R, S.D |
| Historical trajectory | [198] | 2023 | NeurIPS | O.DP, S.UP |
| Current observation | [66] | 2018 | ECML-PKDD | O.DP, S.D |
| Current observation | [122] | 2018 | arXiv | O.DP, S.D |
| Current observation | [199] | 2018 | NeurIPS | O.DP, S.D |
| Current observation | [48] | 2018 | ICML | O.DP, S.D |
| Current observation | [200] | 2018 | ICML | O.DP, S.D |
| Current observation | [47] | 2018 | AIES | O.DP, S.UP |
| Current observation | [100] | 2019 | arXiv | O.DP, S.D |
| Current observation | [201] | 2019 | TVCG | O.DP, S.D |
| Current observation | [202] | 2019 | AAAI | O.DP, O.R, S.D |
| Current observation | [203] | 2020 | GECCO | O.DP, S.D |
| Current observation | [204] | 2020 | NeurIPS | O.DP, S.D |
| Current observation | [205] | 2020 | KDD | O.DP, S.D |
| Current observation | [129] | 2021 | NeurIPS | O.DP, O.R, S.D |
| Current observation | [206] | 2021 | NeurIPS | O.DP, S.UP, S.UC, S.D |
| Current observation | [207] | 2022 | ICML | O.DP, O.F, S.D |
| Current observation | [208] | 2022 | NeurIPS | O.DP, S.D |
| Current observation | [209] | 2022 | NeurIPS | O.DP, S.UP, S.UC, S.D |
| Current observation | [210] | 2023 | ICML | O.DP, O.R, S.D |
| Current observation | [211] | 2024 | TIV | O.DP, O.R, S.D |
| Future prediction | [212] | 2018 | arXiv | O.DP, S.D |
| Future prediction | [213] | 2019 | ICRA | O.DP, S.D |
| Future prediction | [214] | 2019 | ICRA | O.DP, S.D |
| Future prediction | [215] | 2020 | NeurIPS | O.DP, S.D |
| Future prediction | [216] | 2020 | NeurIPS | O.DP, S.D |
| Future prediction | [217] | 2023 | TTE | O.DP, S.D |
| Future prediction | [218] | 2023 | NeurIPS | O.DP, S.D |

#### 1) Historical Trajectory

Starting from the trace of historical decisions, numerous studies aim to estimate the influence of historical observations on future decision-making by agents.

Sparse Bayesian Reinforcement Learning (SBRL) [191] constructs latent space representing past experiences during training to facilitate knowledge transfer and continuous action search. SBRL offers an intuitive explanation for how historical data samples impact the learning process. Another approach, Visual SBRL (V-SBRL) [192], utilizes a sparse filter to maintain the significant past image-based state while discarding the trivial ones, resulting in a sparse image set containing valuable past experience. [127] identify interestingness state elements in historical observations from various aspects, such as the reward outliers and environment dynamics, and present the detected observations in video. [196, 197, 198] denote the important past experiences as prototypes, which are learned by contrastive learning during training. By integrating these prototypes into the policy network, human users are able to observe representative interactions within the task. Meanwhile, [196] generate a broader explanation by comparing current policy output with human-defined prototypes, demonstrating better trustworthiness and performance.

The Shapley value also offers an effective approach for calculating and visualizing the contribution of each feature in prior trajectories. However, the naive computation of the Shapley value faces an exponential complexity. To mitigate this issue, [194] employ Monte Carlo sampling to approximate the Shapley value, while [193] leverage DNN to compute the feature gradients and aggregate them as a Shapley value to develop a 3D feature-time-SHAP map to visualize the significance of each timestep.

Previous surveyed methods only focus on the historical interactions within an episode, [132] extend their horizon by considering interactions across episodes. They incorporate a deep recurrent kernel of the Gaussian Process that takes inputs of timestep embeddings and capture the correlation between timesteps as well as the cumulative impact across episodes. Furthermore, these outputs can be employed for episode-level reward prediction via linear regression analysis. The regression coefficients obtained from the linear regression model can identify important timesteps, thereby enhancing the explainability of the results.

#### 2) Current observation

Numerous studies aim to identify critical features influencing decision-making in the current state, particularly in image-based environments. These approaches offer extrinsic explanations by analyzing the impact of state features on agent behavior. Different methods that fall under this category are depicted in Figure 4.

The Linear Model U-Tree (LMUT) [66] evaluates the importance of an LMUT node and its features based on the certainty of the Q-value and the squared weight of the features. The paper applies LMUT to video games and gets pixels with relatively high influence. The explanation denotes such pixels as "super-pixels", which are crucial for decision-making.

Several studies leverage self-attention, allowing the creation of an attention score matrix, which highlights relationships among input features for improved explainability [47, 48, 100, 122, 129, 202, 203, 204, 209]. In contexts like agent interaction in self-driving scenarios, self-attention-based DNN [100] discerns relations amongst multiple entities. Extending this, [129] use attention neurons for honing in on specific state components. Meanwhile, [202] integrate attention within DNNs to develop auto-encoders for input state reconstruction. Neuroevolution combined with self-attention [203] selects spatial patches over individual pixels, enabling the agent to focus on task-critical areas, thus amplifying efficacy and clarity. [122] introduce a region-sensitive module post-DNN to pinpoint essential input image regions, serving as an explainable plug-in module integrated into classical RL algorithms [10, 11]. Shifting from pixel-centric states, [204] design a hierarchical attention model for text-based games using knowledge graph, capturing state feature relationships. Building on this, [209] enhance explanations by integrating multiple subgraphs with template-filling techniques.

Saliency maps, distinguishing from attention by highlighting specific parts of scenes, like objects or regions, have been adopted to increase explainability in RL agents. These maps showcase pixel influences on outputs through gradient measurements of normalized scores. Several notable studies have contributed to XRL domain [47, 48, 200, 205, 206, 207, 210, 211, 219]. [200] gauged pixel significance by applying a random value mask and evaluating its decision impact, extended by [205] for geographic areas. [48] introduced perturbation-based saliency, perturbing certain features certainties to discern their impacts on policy. This is further employed by [206, 211] to juxtapose human and RL agent action patterns, indicating RL training potential to humanize agents. Improving on this, [208] harnessed unsupervised learning for perturbation-based saliency maps and agent training regularization. Meanwhile, [207] applied CNN for partial future interpretations, whereas [210] leveraged shapley values to analyze the effects of feature removals. Lastly, the object saliency map [47] integrates template matching and enables easier human interpretations by connecting pixel saliency maps with object detection.

In contrast to relying on local spatial information, [199] utilize flow information to capture and segment the moving object in the image. Therefore, the policy can focus on moving objects in a more interpretable manner. Furthermore, [201] propose a specialized framework for visualizing DQN [9] process. This visualization provides insights into the operations performed at each stage and the activation levels of each layer within the deep neural network.

**Figure 4.** Examples of state importance extraction techniques via (a) intrinsic architectures [100, 129, 202, 203, 209] and (b) extrinsic perturbations [47, 48, 122, 200, 208].

*Visual description:* Panel (a) "structure-based importance" shows an Unexplainable State Input fed into a Network Architecture (Convolutional Neural Network and Attention Network), producing a Feature/Region Importance Vector that goes into a DRL Method ($\pi_\phi$, $Q_\theta$) yielding policy $\pi$ for the Human User. Panel (b) "Perturbation-based importance" shows the State Input passed to a Perturbation Generator producing perturbed states $m_1, ..., m_n$ fed through a DRL Policy producing $\pi_\pi$, then a Difference function $D(\pi_\pi, \cdot)$ yields the Feature/Region Importance Vector.

#### 3) Future Prediction

The future prediction method first generates forecasts for future events and then uses these predictions to produce various explanations.

A common approach to predicting the future involves repeated forward simulations from the current state [212]. However, these simulations may be unprecise due to stochastic environmental factors and approximate biases in training [220]. To address this, [215] maintains the discounted expected future state visitations with temporal difference loss to further construct the belief map. The training process of such a belief map is consistent with current value-based inexplainable RL frameworks. This advantage renders it an explainable plug-in for value-based RL methods. Meanwhile, [218] utilize diffusion model to dynamically generate future state sequences conditioned on current states. [216, 217] combine future prediction with multi-goal RL, facilitating trustworthy predictions of goal for the current state. Semantic Predictive Control (SPC) [213] dynamically learns the environment and aggregates multi-scale feature maps to predict future semantic events. Additionally, [214] employ an ensemble of LSTM networks trained using Monte Carlo Dropout and bootstrapping to estimate the probability of future events and predict uncertainty in new observations. Recently,

#### 4) Evaluation

Existing state-explaining methods undergo evaluation through various assessments, encompassing both subjective and objective measures. O.R stands out as the predominantly employed quantitative assessment [129, 202]. To further evaluate the quality of explanations in various aspects, it is necessary to utilize more objective methods of O.F and O.R to assess the accuracy and robustness of the allocation of importance on both temporal and spatial features. However, subjective measurement is not applicable since gathering a significant amount of evaluation data through human feedback would be time-consuming and ineffective.

### D. Task-explaining

Task-explaining method explains how to divide the current complex task into multiple subtasks via the hierarchical agent. In a hierarchical agent, a high-level controller selects options, while several low-level controllers choose primitive actions. The option chosen by the high-level controller acts as a sub-goal for the low-level controllers to accomplish. This division of labor in Hierarchical Reinforcement Learning (HRL) enhances architectural explainability to the aforementioned XRL works, offering insight into how the high-level agent schedules the low-level tasks. In this section, we delve into HRL and categorize its approaches into two parts: the whole top-down structure and simple task decomposition according to the scheduling mechanism of high-level agents. These categorized approaches are presented in Table V.

**TABLE V:** Task-explaining methods in XRL.

| Type | Reference | Year | Venue | Evaluation |
|---|---|---|---|---|
| Whole Top-to-Down structure | [97] | 2018 | ICLR | O.DP, S.D |
| Whole Top-to-Down structure | [221] | 2020 | NeurIPS | O.DP, S.D |
| Simple task division | [222] | 2019 | NeurIPS | O.DP, S.D |
| Simple task division | [223] | 2019 | IROS | O.DP, S.D |
| Simple task division | [224] | 2020 | AAMAS | O.DP, S.D |
| Simple task division | [225] | 2021 | ICML | O.DP, S.D |
| Simple task division | [184] | 2021 | TCSS | O.DP, S.D |

#### 1) Whole Top-to-Down Structure

In hierarchical tasks with this structure, task sets are divided into multiple levels. The low-level task sets are subsets of the high-level task sets, with the latter containing task elements absent from the former. This well-defined and coherent structure enhances explainability, as it aligns with human experiences and allows for the observation of how the high-level agent schedules low-level tasks.

A notable study [97] train a hierarchical policy in Minecraft, an open-world and multi-task environment. The task division sets, denoted as $G_1, G_2, ..., G_k$, follow a hierarchical structure: $G_1 \subset G_2 \subset ... \subset G_k$. At each level, a policy $\pi_k$ comprises four components: a base task set policy $\pi_{k-1}$, an instruction policy $\pi^{inst}_k$ for providing instructions $g$ to guide the execution of base tasks by $\pi_{k-1}$, an augment flat policy $\pi^{Aug}_k$ that directly selects actions for $\pi_k$ instead of relying on base tasks, and a switch policy $\pi^{sw}_k$ that determines whether to choose actions from the base tasks or the augment flat. The state is represented as the pair $(e_t, g_t)$, where $e_t$ signifies time and $g_t$ represents the instruction. To train such a hierarchical policy, a two-step approach is proposed. Firstly, basic skills are learned from $G_{k-1}$ to ensure that the previously acquired policy can be leveraged by instructing the base policy. This stage establishes the connection between the instruction policy and the base policy. Next, samples are collected from $G_k$ to learn new skills and the switch policy. Both steps rely on the classical actor-critic RL algorithm.

Another idea is about the logical combination of base tasks utilizing bool algebra form [221]. This allows for task expressions to employ logical operations such as disjunction, conjunction, and negation. The proposed framework focuses on lifelong learning, which necessitates the utilization of previously acquired skills to solve new tasks. Consequently, the tasks $G_i$ follow a sequential relation: $G_1 \subset G_2... \subset G_{t-1} \subset G_t$. In this framework, the paper initially learns goal-oriented approximations of the value function for each base task and subsequently combines these approximations in a specific manner. By leveraging this framework, it becomes possible to acquire new task skills without the need for additional learning. Additionally, it successfully represents the optimal policy for the current RL task using Boolean algebra.

#### 2) Simple Task Division

In contrast to a strictly top-down structure, where sub-tasks are hierarchically defined, simply divided sub-tasks exhibit equal status and filter out priority over each other. Within the context of multi-task reinforcement learning, an efficient approach is needed for knowledge transfer among tasks. To address this, metadata can serve as a valuable tool for capturing task structures and facilitating knowledge transfer. In a study by [225], the authors leverage metadata to learn explainable contextual representations across a family of tasks. These sub-tasks align with a higher-level overarching goal, leading to the division of tasks into two levels. The low-level tasks typically represent decomposed sub-tasks of the original task, sharing the same status. Conversely, the high-level task focuses on scheduling the sub-tasks within the overall task structure.

Numerous methods involve explicitly dividing tasks and constructing a high-level agent as a scheduler for low-level agents. [222] train the high-level agent to produce language instructions for the low-level agents. During training, the low-level agents employ a condition-RL algorithm, while the high-level agents use a language model-based RL algorithm. All language instructions generated by the high-level agents are comprehensible to humans. The symbolic planning+RL method [184] employs a planner-controller-meta-controller framework to address hierarchical tasks. The planner operates at a higher level, leveraging symbolic knowledge to schedule sub-task sequences. Meanwhile, the controller operates at a lower level, employing traditional DRL methods to solve sub-tasks, and the meta-controller simultaneously providing a new intrinsic target for the planner to guide better task-solving explicitly. In the Dot-to-Dot (D2D) framework [223], the high-level agent constructs the environment dynamics, and utilized it to provides direction to the low-level agents. The low-level agent receives guidance from the high-level agent and solves decomposed, simpler sub-tasks. As a result, the high-level agent can learn an explainable representation of the decision-making process, while the low-level agent effectively learns the larger state and action space.

Unlike the two aforementioned approaches, [224] adopt a different strategy for task division. They utilize a primitive model instead of directly dividing the task. Initially, the primitive model approximates piecewise functional decomposition. Each specialized primitive model focuses on a distinct region, resulting in corresponding sub-policies specialized in those regions. The sub-policies are subsequently transferred to compose the complete policy for the desired tasks. Through the combination of these sub-policies, this framework retains architectural explainability. The efficacy of this explainability is demonstrated on high-dimensional continuous tasks, both in lifelong learning scenarios and single-task learning. However, the use of the primitive model may not be individually effective for learning to decompose mixed tasks.

#### 3) Evaluation

Currently, all task-explaining methods employ S.D for evaluation. Task-explaining methods provide explanation on task decomposition and scheduling. To measure whether the high-level agent scheduling is correct and reasonable, experienced human is able to provide precise criterion. Hence, we propose that subjective assessments of S.UP and S.UC, utilizing the divide-and-conquer approach of human participants as evaluating criteria, are more suitable for judging the effectiveness of task-explaining methods in dividing tasks.

### E. XRL Methodology Selection

Despite comprehensive documentation of our taxonomic framework and extensive literature analysis, persistent uncertainties remain regarding selection criteria among XRL methodologies. To address this gap, we present a systematic comparative analysis of XRL archetypes in Table VI, derived through summarization across the four methodological categories. This analytical framework enables practitioners to align methodological selection with both system requirements and explainable objectives through feature-based evaluation.

**TABLE VI:** Comparison of different types of XRL approaches. "H" and "L" denotes "High" and "Low" respectively.

| Type | Quantification | Fineness | Verifiability | Clarity | Need RL prior |
|---|---|---|---|---|---|
| Agent model-explaining | H | H | L | L | H |
| Reward-explaining | H | H | L | H | H |
| State-explaining | H | H | L | H | H |
| Task-explaining | L | L | H | H | L |

Considering the beginner for a specific task, the intrinsic RL explanation helps the beginner to understand the task and its solving process. Individuals unfamiliar with a particular task can quickly grasp its structure, objectives, and general problem-solving approaches through task-explaining methods [97, 224]. To gain more specific and professional insights for solving the task comprehensively, the agent model-explaining methods can guide human learning by providing a policy sketch that illuminates the internal reasoning of XRL agents [68, 98, 123, 128]. Experienced human users who can only solve the task suboptimally possess a general understanding of the method for solving the task but lack proficiency in the specific behaviors required. Therefore, capturing crucial observational features from state-explaining methods can significantly enhance their short-term decision-making. Simultaneously, the reward-explaining methods highlight various latent aspects in the task that contribute to performance changes, enabling human users to attentively select actions and improve their long-term performance.

For RL researchers, different types of XRL methods help the RL researchers to get insight into the exploration of agents and dynamics of environments. Agent model-explaining methods [67, 226] illustrate how the inner mechanism of the agent changes during training. And task-explaining methods [187, 189] provide valuable insights regarding the task complexity. These insights can be leveraged to guide fine-tuning of agent architecture for improving performance. Meanwhile, by receiving the explanation of how different factors in the reward function affect the policy of agent [181, 193], researchers can gain a better understanding of how to design an effective reward function to enhance the agent performance [124, 227]. State-explaining methods shed light on the dynamic focus of the agent on state features during training, enabling researchers to comprehend how these features affect the agent's decision-making process.

Different human groups can choose different XRL methods based on their specific requirements to augment their comprehension of tasks and successfully complete them.

## V. Human Knowledge for XRL

Incorporating human prior knowledge into XRL enhances agent performance and explainability by aligning learning objectives with domain expertise. While mainstream XRL frameworks often neglect human participation during training, emerging studies demonstrate its benefits in guiding agent behavior and explanatory quality [45, 228, 229, 230, 231, 232]. Given their underrepresentation in current XRL work, we advocate for systematic integration of human knowledge within existing taxonomic frameworks to advance XRL.

### A. Fuzzy Controller Representing Human Knowledge

Fuzzy logic can be utilized to represent human knowledge to further construct the agent policy, obtaining intrinsic RL explainability on agent architecture. Traditional approaches, such as bivalent logic rules, are ill-suited for the representation of vague human knowledge due to the overly deterministic nature. In contrast, fuzzy logic can effectively represent human knowledge in an uncertain and imprecise manner. A notable contribution in this area is the work of [228], who propose the KoGuN policy network utilizing the knowledge controller to integrate human suboptimal knowledge. The knowledge controller utilizes a set of fuzzy rules $\{l_i\}$ translated from human knowledge. Given the state input $s$, these fuzzy rules representing prior human knowledge jointly output a preference action $\mathbf{p}$, which is then fine-tuned by summing with an additional vector $\mathbf{p}'$ produced by a hypernetwork. Meanwhile, to tackle the challenge of possible human knowledge mismatch under different states, trainable rule weights $\beta_i$ are introduced for each rule $l_i$ in order to facilitate adaptation to new tasks and optimize the performance of the knowledge controllers. This policy network, with prior human knowledge, is trained by the conventional PPO method. Although the final action output is slightly fine-tuned by the hypernetwork, the rules weights $\{\beta_i\}$ effectively illustrate the influence of different human knowledge towards agent decision-making, exhibiting the intrinsic explainability of agent architecture. Similarly, [233] incorporate exact traffic laws into fuzzy logic rules to constraint the self-vehicle behaviors and [234] establish fuzzy rules containing human knowledge into hierarchical policy, enhancing both fast training and explainability.

### B. Dense Reward on Human Command

The intrinsic dense reward can not only release the ineffective learning under the sparse reward setting but also align with the human command to provide explanation to indicate the agent motivation. Although sparse reward is frequently employed in real tasks due to its simplicity, learning with sparse reward is challenging. Therefore, efforts have been made to define a dense reward function that provides a reward signal for each action performed. Several studies [235, 236, 237, 238, 239] have introduced dense reward functions that focus on state-based novelty. Yet, these studies often fail to provide a clear explanation of the underlying motivation and logical sequence of the task goal. An innovative approach presented by [229] introduces the LanguagE-Action Reward Network (LEARN) based on natural language command from humans, and LEARN captures the correlation between action and human command. The authors define MDP(+L) as a variant of MDP, denoted as $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma, l \rangle$, where $l$ represents a human-defined language command describing the desired agent behavior, while the other components remain consistent with the elements in MDP. The original reward function in MDP is labeled as $R_\text{ext}$, and the dense reward determined by the language command $l$ is denoted as $R_\text{lan}$. To assess whether the agent is following the language command $l$, LEARN extracts the sequence of past actions $(a_1, a_2, ..., a_{t-1})$ and transforms it into an action-frequency vector $\mathbf{f}$. LEARN takes both $\mathbf{f}$ and the natural language command $l$ as inputs and produces a probability distribution indicating the relevance between the action-frequency vector and the natural language command. This distribution measures the correlation between $\mathbf{a}$ and $l$, which composes the intrinsic language reward $R_\text{lan}$. Therefore, the target optimal policy can be generated based on the new reward function $R_\text{ext} + R_\text{lan}$. The auxiliary reward effectively illustrates the quantitative consideration of accomplishing the task based on human command, presenting extrinsic RL explainability for agent-specific behavior.

### C. Learn Mattered Features from Human Interactions

Learning the significant features directly from human interaction is an effective way to enhance performance as well as extrinsic explainability. In Section IV-C, we discussed the utilization of attention-based techniques to learn important features from input vectors of images or videos. In the context of imitation learning frameworks, there are corresponding approaches to obtain attention. [49] categorizes these methods as learning attention from humans, where human trainers provide explicit weight distribution such as gaze information and attention maps. This kind of explanation can serve as an additional source of evaluative feedback if the RL agent is able to capture it. [240] first generate and open the human interaction data with gaze information in Atari games. [45] enhance human attention data by perturbing irrelevant regions. The saliency map serves as human explanation to guide agent effective learning. [231] perceive gaze as probabilistic variables that can be predicted using stochastic units embedded in DNNs. Guided by this idea, they develop a gaze framework that selects important features and estimates the uncertainty of human gaze supervisory signals. As for enhancing explainability, [230] employ a visual attention model to train a mapping from images to vehicle control signals, which synchronously generates extrinsic explanations on current state components features towards the actions of agent. Meanwhile, from the temporal aspect, many methods find the important states during training to extrinsically explain the task-solving process instead of the internal state features based on human expert demonstrations. [227] introduce the concept of meta-state, which encapsulates significant states for task completion based on expert trajectories. The meta-state is obtained from spectral clustering. Concurrently, [124] detect states in expert trajectories as task-specific subgoals by considering the uncertainty of the agent, which is quantified through the variance of the critic value on expert transitions. Although the agent itself is not explainable, the subgoals effectively represent the task process and provide extrinsic RL explainability for agent behaviors.

### D. Subtask Scheduling with Human Annotation

In terms of task explanation, leveraging human annotation on scheduling subtasks can be utilized to guide hierarchical agent training and enhance intrinsic explainability within the hierarchical agent. In order to expedite the task decomposition process, [232] incorporate human annotation and demonstration to train a high-level language generator to schedule the low-level policies. The generator is trained using imitation learning and consists of LSTM networks, which take the encoded state (containing explicit goal) as input and produce natural language instructions as output. Instead of output language instruction, [69] generate a discrete latent representation of primitive skills in long-term task with the clustering method to guide low-level agents with human-annotated trajectories. Meanwhile, [70] further improves task explainability by collaborating with humans in MOBA games, in which the high-level agent learn to generate explainable meta commands from human. These frameworks utilizing natural language instructions facilitate successful task decomposition and exhibit high generalizability to new tasks. Furthermore, [241] propose an approach to decompose tasks by having humans answer yes-or-no questions regarding the task content. These methods leverage various human annotations to guide high-level agent in producing scheduling signals on specific state inputs. These scheduling signals are inherently explainable to humans and are subsequently captured by low-level agents to generate actual behavior.

## VI. Challenges and Future Directions for XRL

Given the early stage of XRL research, there remain uncertainties regarding aspects such as architecture and evaluation metrics. Drawing upon the reviewed literature on XRL, we present several promising directions for future research.

### A. Human Knowledge in XRL

Human knowledge-intergrated XRL, which incorporates human knowledge as raw explanations or resources, further enhances XRL explainability and efficiency, as highlighted in Section V. However, obtaining certain types of human prior knowledge can be challenging. For example, annotating massive amounts of data manually can be time-consuming [69] and gathering expert trajectory data for dangerous tasks can be difficult [242]. In such cases, only a limited amount of data containing suboptimal human knowledge with varying quality may be available, posing a challenge for XRL agents to acquire high-quality policies and explanations. To address this challenge, several methods have been proposed to effectively utilize available data. [243] introduce an explainable active learning approach, which efficiently learns a teacher model from limited human feedback by providing both predictions and explanations to humans. Preference-based RL (PbRL) [244] is another approach that uses human preferences to train the agent and has shown success in tasks with limited annotation [245]. [246] enhance the explainability of PbRL by simultaneously learning the reward function and state importance. They leverage a perturbation analysis method to quantify the learned state importance, enabling high explainability with minimal human intervention. Future research on XRL should fully utilize human-annotated data to extract human knowledge for high explainability and performance.

### B. Evaluation Methods

Despite discussing the current evaluation methods for XRL in Section III, there is still a lack of a widely accepted approach within the DRL community. This can be attributed to the fact that XRL approaches are highly task-specific, making it challenging to establish a universal measurement method due to the diverse forms of explanations. Furthermore, the notion of explainability is often treated subjectively in many papers, with claims of explainability lacking mathematical formulas or rigorous analysis to support their assertions. The establishment of evaluation methods would enable the comparison of different approaches and identification of the state-of-the-art techniques. For instance, [247] propose a software platform for self-driving that facilitates the comparison of various XRL agents in the same driving scenario and evaluates the precision of explanations provided by the XRL agent. However, in addition to XRL performance and explanation precision, legal and ethical aspects must also be considered when devising the evaluation method to ensure real-world applicability.

### C. Multi-part Explainability

The aforementioned XRL approaches, including our categorization work, primarily focus on making only a single component of the RL framework explainable, resulting in partial explainability and improvements in specific areas. However, a crucial challenge is that the remaining parts of the RL framework continue to lack transparency for experts. Tasks of high complexity, such as self-driving, demand comprehensive explainability for enhanced safety. Consequently, reliance on a single explainable component is insufficient and fails to provide convincing explanations. To address this issue, incorporating multi-part explainability into the MDP process can offer a potential solution for RL agents. One approach involves constructing an integrated method that combines various part-explaining techniques. For instance, [99] merge global explanations based on strategy summaries with local explanations derived from saliency maps, which respectively correspond to agent model-explaining and state-explaining. However, the diverse structures and limited applicability of different part-explaining methods make the combination process challenging. A possible approach could involve abstracting them at a higher level and subsequently integrating them.

### D. Balance of High Explainability and Effective Training

It is feasible to achieve both effective training and high explainability in RL agents. Explainability is regarded as an additional attribute to agent performance, which typically results in XRL being perceived as requiring more computational resources compared to the conventional RL approach [248]. However, contrasting the unexplainable DNN-based agents that exhibit high performance, several researchers have discovered that employing simpler and more explainable agent models can also achieve excellent performance while maintaining high explainability, such as linear models [249, 250] and decision tree [128, 130]. These findings indicate that the trade-off between explainability and performance is not as rigid as initially perceived. Moreover, it is possible to strike a balance between these two factors by exploring alternative techniques [59]. For example, the incorporation of sparsity or explainability constraints into the agent policy [251], can enhance explainability without compromising performance. Therefore, further research is warranted to determine the optimal balance between explainability and well-training.

## VII. Conclusion

Explainability has attracted increasing attention in the RL community due to practical, safe, and trustworthy concerns. It endows the RL agent with the ability to exhibit a well-grounded behavior and further convince the human participants. In this comprehensive survey, we introduce unified concept definitions and taxonomies to summarize and correlate a wide variety of recent advanced XRL approaches. The survey first gives an in-depth introduction to the explainability definition and evaluation metric of XRL. Then we further categorize the related XRL approaches into four branches: (a) Agent model-explaining methods that directly build the agent model as an explainable box. (b) Reward-explaining methods that regularize the reward function to be understandable. (c) State-explaining methods that provide the attention-based explanation of observations. (d) Task-explaining methods that decompose the task to get multi-stage explainability.

Moreover, it is notable that several XRL methods conversely leverage human knowledge to promote the optimization process of learning agents. We additionally discuss and organize these works into our taxonomy structure, while the other XRL surveys pay little attention to it.

We hope that this survey can help newcomers and researchers to understand and exploit the existing methods in the growing XRL field, as well as highlight opportunities and challenges for future research.

## References

[1] R. S. Sutton et al., *Reinforcement learning: An introduction*. MIT press, 2018.

[2] H. Kjellström et al., "Tracking people interacting with objects," in *CVPR*, 2010.

[3] R. V. Yampolskiy et al., "Artificial general intelligence and the human mental model," in *Singularity Hypotheses*, 2012.

[4] D. M. Williamson et al., "'mental model' comparison of automated and human scoring," *Journal of Educational Measurement*, 1999.

[5] A. Powers et al., "The advisor robot: tracing people's mental model from a robot's physical attributes," in *HRI*, 2006.

[6] W. R. Stauffer et al., "Components and characteristics of the dopamine reward utility signal," *Journal of Comparative Neurology*, 2016.

[7] Y. Bengio et al., *Deep Learning*, 2016.

[8] V. Sze et al., "Efficient processing of deep neural networks: A tutorial and survey," *Proc. IEEE*, 2017.

[9] V. Mnih et al., "Playing atari with deep reinforcement learning," *arXiv preprint arXiv:1312.5602*, 2013.

[10] J. Schulman and Wothers, "Proximal policy optimization algorithms," *arXiv preprint arXiv:1707.06347*, 2017.

[11] V. Mnih et al., "Asynchronous methods for deep reinforcement learning," in *ICML*, 2016.

[12] T. Haarnoja et al., "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor," in *ICML*, 2018.

[13] S. Fujimoto et al., "Addressing function approximation error in actor-critic methods," in *ICML*, 2018.

[14] A. Ansuini et al., "Intrinsic dimension of data representations in deep neural networks," in *NeurIPS*, 2019.

[15] J. Yosinski et al., "How transferable are features in deep neural networks?" in *NeurIPS*, 2014.

[16] Z. Goldfeld and Bothers, "Estimating information flow in deep neural networks," *arXiv preprint arXiv:1810.05728*, 2018.

[17] D. Silver et al., "Mastering the game of go without human knowledge," *Nature*, 2017.

[18] C. Berner et al., "Dota 2 with large scale deep reinforcement learning," *arXiv preprint arXiv:1912.06680*, 2019.

[19] Y. Chen et al., "Attention-based hierarchical deep reinforcement learning for lane change behaviors in autonomous driving," in *CVPR*, 2019.

[20] S. Liu et al., "Contrastive identity-aware learning for multi-agent value decomposition," in *AAAI*, 2023.

[21] Y. Qing, S. Liu, J. Cong, K. Chen, Y. Zhou, and M. Song, "A2po: Towards effective offline reinforcement learning from an advantage-aware perspective," *Advances in Neural Information Processing Systems*, vol. 37, pp. 29 064–29 090, 2024.

[22] Y. Qing, S. Chen, Y. Chi, S. Liu, S. Lin, and C. Zou, "Bitrajdiff: Bidirectional trajectory generation with diffusion models for offline reinforcement learning," *arXiv preprint arXiv:2506.05762*, 2025.

[23] Y. Zhou et al., "Is centralized training with decentralized execution framework centralized enough for marl?" *arXiv preprint arXiv:2305.17352*, 2023.

[24] A. R. Fayjie et al., "Driverless car: Autonomous driving using deep reinforcement learning in urban environment," in *UR*, 2018.

[25] S. Wang et al., "Deep reinforcement learning for autonomous driving," *arXiv preprint arXiv:1811.11329*, 2018.

[26] J. Chen et al., "Model-free deep reinforcement learning for urban autonomous driving," in *TITS*, 2019.

[27] C.-J. Hoel et al., "Combining planning and deep reinforcement learning in tactical decision making for autonomous driving," in *TIV*, 2019.

[28] P. Wang et al., "Formulation of deep reinforcement learning architecture toward autonomous driving for on-ramp merge," in *ITSC*, 2017.

[29] K. Chen et al., "Powerformer: A section-adaptive transformer for power flow adjustment," *arXiv preprint arXiv:2401.02771*, 2024.

[30] F. Xu et al., "Temporal prototype-aware learning for active voltage control on power distribution networks," in *KDD*, 2024.

[31] L. Lin et al., "Deep reinforcement learning for economic dispatch of virtual power plant in internet of energy," *IEEE Internet of Things Journal*, 2020.

[32] T. Yang et al., "Dynamic energy dispatch strategy for integrated energy system based on improved deep reinforcement learning," *Energy*, 2021.

[33] S. Liu et al., "Transmission interface power flow adjustment: A deep reinforcement learning approach based on multi-task attribution map," *IEEE TPS*, 2023.

[34] ——, "Progressive decision-making framework for power system topology control," *ESWA*, 2024.

[35] K. He et al., "Deep residual learning for image recognition," in *CVPR*, 2016.

[36] T. Zahavy et al., "Graying the black box: Understanding dqns," in *ICML*, 2016.

[37] T. Jaunet et al., "Drlviz: Understanding decisions and memory in deep reinforcement learning," in *CGF*, 2020.

[38] V. Ivchyk et al., "Overcoming barriers to artificial intelligence adoption," *Three Seas Economic Journal*, 2024.

[39] I. Prasetya et al., "Navigation and exploration in 3d-game automated play testing," in *International Workshop on Automating TEST Case Design, Selection, and Evaluation*, 2020.

[40] X. Han et al., "Improving multi-agent reinforcement learning with imperfect human knowledge," in *ICANN*, 2020.

[41] A. Rosenfeld et al., "Leveraging human knowledge in tabular reinforcement learning: A study of human subjects," *KER*, 2018.

[42] R. Zhang et al., "Leveraging human guidance for deep reinforcement learning tasks," *arXiv preprint arXiv:1909.09906*, 2019.

[43] H. Zhang et al., "Faster and safer training by embedding high-level knowledge into deep reinforcement learning," *arXiv preprint arXiv:1910.09986*, 2019.

[44] L. Guan et al., "Explanation augmented feedback in human-in-the-loop reinforcement learning," *arXiv preprint arXiv:2006.14804*, 2020.

[45] ——, "Widening the pipeline in human-guided reinforcement learning with explanation and context-aware data augmentation," in *NeurIPS*, 2021.

[46] A. Silva et al., "Encoding human domain knowledge to warm start reinforcement learning," in *AAAI*, 2021.

[47] R. Iyer et al., "Transparency and explanation in deep reinforcement learning neural networks," in *AIES*, 2018.

[48] S. Greydanus et al., "Visualizing and understanding atari agents," in *ICML*, 2018.

[49] R. Zhang et al., "Leveraging human guidance for deep reinforcement learning tasks," *arXiv preprint arXiv:1909.09906*, 2019.

[50] U. Ehsan et al., "Automated rationale generation: a technique for explainable ai and its effects on human perceptions," in *IUI*, 2019.

[51] B. Hayes et al., "Improving robot controller transparency through autonomous policy explanation," in *HRI*, 2017.

[52] B. RichardWebster et al., "Visual psychophysics for making face recognition algorithms more explainable," in *ECCV*, 2018.

[53] J. R. Williford et al., "Explainable face recognition," in *ECCV*, 2020.

[54] H. J. andothers, "Explainable face recognition based on accurate facial compositions," in *ICCV*, 2021.

[55] D. Franco et al., "Deep fair models for complex data: Graphs labeling and explainable face recognition," *Neurocomputing*, 2022.

[56] H. Liu et al., "Towards explainable nlp: A generative explanation framework for text classification," *arXiv preprint arXiv:1811.00196*, 2018.

[57] I. Arous et al., "Marta: Leveraging human rationales for explainable text classification," in *AAAI*, 2021.

[58] B. Škrlj et al., "autobot: evolving neuro-symbolic representations for explainable low resource text classification," *ML*, 2021.

[59] C. Glanois et al., "A survey on interpretable reinforcement learning," *arXiv preprint arXiv:2112.13112*, 2021.

[60] G. A. Vouros et al., "Explainable deep reinforcement learning: State of the art and challenges," *CSUR*, 2022.

[61] A. Heuillet et al., "Explainability in deep reinforcement learning," *KBS*, 2021.

[62] L. Wells et al., "Explainable ai and reinforcement learning—a systematic review of current approaches and trends," *Frontiers in Artificial Intelligence*, 2021.

[63] R. Dazeley et al., "Explainable reinforcement learning for broad-xai: A conceptual framework and survey," *arXiv preprint arXiv:2108.09003*, 2021.

[64] E. Puiutta et al., "Explainable reinforcement learning: A survey," in *CD-MAKE*, 2020.

[65] S. Milani et al., "Explainable reinforcement learning: A survey and comparative review," *CSUR*, 2023.

[66] G. Liu et al., "Toward interpretable deep reinforcement learning with linear model u-trees," in *ECML PKDD*, 2018.

[67] P. Madumal et al., "Explainable reinforcement learning through a causal lens," in *AAAI*, 2020.

[68] A. Verma et al., "Imitation-projected programmatic reinforcement learning," in *NeurIPS*, 2019.

[69] D. Garg et al., "Lisa: Learning interpretable skill abstractions from language," in *NeurIPS*, 2022.

[70] Y. Gao et al., "Towards effective and interpretable human-agent collaboration in moba games: A communication perspective," *arXiv preprint arXiv:2304.11632*, 2023.

[71] A. Feinberg, "Markov decision processes: Discrete stochastic dynamic programming (martin l. puterman)," *SIAM Review*, 1996.

[72] H. Van Hasselt et al., "Deep reinforcement learning with double q-learning," in *AAAI*, 2016.

[73] Z. Wang et al., "Dueling network architectures for deep reinforcement learning," in *ICML*, 2016.

[74] V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, 2015.

[75] M. Hausknecht et al., "Deep recurrent q-learning for partially observable mdps," in *AAAI*, 2015.

[76] T. Schaul et al., "Prioritized experience replay," *arXiv preprint arXiv:1511.05952*, 2015.

[77] M. G. Bellemare and Dothers, "A distributional perspective on reinforcement learning," in *ICML*, 2017.

[78] M. Hessel et al., "Rainbow: Combining improvements in deep reinforcement learning," in *AAAI*, 2018.

[79] Q. Chen et al., "Es-dqn: A learning method for vehicle intelligent speed control strategy under uncertain cut-in scenario," *TVT*, 2022.

[80] L. Meng et al., "Improving the diversity of bootstrapped dqn via noisy priors," *arXiv preprint arXiv:2203.01004*, 2022.

[81] A. Chraibi et al., "Makespan optimisation in cloudlet scheduling with improved DQN algorithm in cloud computing," *Scientific Programming*, 2021.

[82] L. Chen et al., "Conditional dqn-based motion planning with fuzzy logic for autonomous driving," *TITS*, 2022.

[83] C. Liu et al., "Forecasting the market with machine learning algorithms: An application of NMC-BERT-LSTM-DQN-X algorithm in quantitative trading," *KDD*, 2022.

[84] S. Park et al., "Applying DQN solutions in fog-based vehicular networks: Scheduling, caching, and collision control," *Vehicular Communications*, 2022.

[85] A. Vashist et al., "Dqn based exit selection in multi-exit deep neural networks for applications targeting situation awareness," in *ICCE*, 2022.

[86] J. Schulman et al., "Trust region policy optimization," in *ICML*, 2015.

[87] N. Heess et al., "Emergence of locomotion behaviours in rich environments," *arXiv preprint arXiv:1707.02286*, 2017.

[88] J. Schulman et al., "High-dimensional continuous control using generalized advantage estimation," *arXiv preprint arXiv:1506.02438*, 2015.

[89] B. Fernández-Gauna et al., "Actor-critic continuous state reinforcement learning for wind-turbine control robust optimization," *Information Sciences*, 2022.

[90] X. Gong et al., "Actor-critic with familiarity-based trajectory experience replay," *Information Sciences*, vol. 582, pp. 633–647, 2022.

[91] X. Xin et al., "Supervised advantage actor-critic for recommender systems," in *CIKM*, 2022.

[92] D. Silver et al., "Deterministic policy gradient algorithms," in *ICML*, 2014.

[93] T. P. Lillicrap et al., "Continuous control with deep reinforcement learning," *arXiv preprint arXiv:1509.02971*, 2015.

[94] T. Miller, "Explanation in artificial intelligence: Insights from the social sciences," *AI*, 2019.

[95] B. Kim et al., "Examples are not enough, learn to criticize! criticism for interpretability," in *NeurIPS*, 2016.

[96] A. Silva et al., "Optimization methods for interpretable differentiable decision trees applied to reinforcement learning," in *AISTATS*, 2020.

[97] T. Shu et al., "Hierarchical and interpretable skill acquisition in multi-task reinforcement learning," in *ICLR*, 2018.

[98] Z. Jiang et al., "Neural logic reinforcement learning," in *ICML*, 2019.

[99] T. Huber et al., "Local and global explanations of agent behavior: Integrating strategy summaries with saliency maps," *AI*, 2021.

[100] E. Leurent et al., "Social attention for autonomous decision-making in dense traffic," *arXiv preprint arXiv:1911.12250*, 2019.

[101] F. Doshi-Velez et al., "A roadmap for a rigorous science of interpretability," *arXiv preprint arXiv:1702.08608*, 2017.

[102] R. R. Hoffman et al., "Metrics for explainable ai: Challenges and prospects," *arXiv preprint arXiv:1812.04608*, 2018.

[103] S. Mohseni et al., "A multidisciplinary survey and framework for design and evaluation of explainable ai systems," *TiiS*, 2021.

[104] M. Kay et al., "When (ish) is my bus? user-centered visualizations of uncertainty in everyday, mobile predictive systems," in *CHI*, 2016.

[105] M. T. Ribeiro et al., "" why should i trust you?" explaining the predictions of any classifier," in *KDD*, 2016.

[106] T. Ribeiro et al., "Anchors: High-precision model-agnostic explanations," in *AAAI*, 2018.

[107] B. Nushi, E. Kamar, and E. Horvitz, "Towards accountable ai: Hybrid human-machine analyses for characterizing system failure," in *HCOMP*, 2018.

[108] G. Bansal et al., "Beyond accuracy: The role of mental models in human-ai team performance," in *HCOMP*, 2019.

[109] B. Kim et al., "Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (tcav)," in *ICML*, 2018.

[110] T. Kulesza et al., "Too much, too little, or just right? ways explanations impact end users' mental models," in *VL/HCC*, 2013.

[111] H. Lakkaraju et al., "Interpretable decision sets: A joint framework for description and prediction," in *KDD*, 2016.

[112] E. Rader et al., "Understanding user beliefs about algorithmic curation in the facebook news feed," in *CHI*, 2015.

[113] M. Bilgic et al., "Explaining recommendations: Satisfaction vs. promotion," in *IUI Workshop*, 2005.

[114] F. Gedikli et al., "How should i explain? a comparison of different explanation types for recommender systems," *IJHCI*, 2014.

[115] I. Lage et al., "Human evaluation of models built for interpretability," in *HCOMP*, 2019.

[116] B. Y. Lim et al., "Assessing demand for intelligibility in context-aware applications," in *IUIC*, 2009.

[117] S. Coppers et al., "Intellingo: An intelligible translation environment," in *CHI*, 2018.

[118] B. Y. Lim et al., "Why and why not explanations improve the intelligibility of context-aware intelligent systems," in *CHI*, 2009.

[119] S. Berkovsky et al., "How to recommend? user trust factors in movie recommender systems," in *IUI*, 2017.

[120] P. Pu and L. Chen, "Trust building with explanation interfaces," in *IUI*, 2006.

[121] F. Nothdurft et al., "Probabilistic human-computer trust handling," in *SIGDIAL*, 2014.

[122] Z. Yang et al., "Learn to interpret atari agents," *arXiv preprint arXiv:1812.11276*, 2018.

[123] A. Verma et al., "Programmatically interpretable reinforcement learning," in *ICML*, 2018.

[124] S. Liu et al., "Curricular subgoals for inverse reinforcement learning," *arXiv preprint arXiv:2306.08232*, 2023.

[125] H. J. Hermans, "A questionnaire measure of achievement motivation," *JAP*, 1970.

[126] P. Lietz, "Research into questionnaire design: A summary of the literature," *IJMR*, 2010.

[127] P. Sequeira et al., "Interestingness elements for explainable reinforcement learning: Understanding agents' capabilities and limitations," *AI*, 2020.

[128] O. Bastani et al., "Verifiable reinforcement learning via policy extraction," in *NeurIPS*, 2018.

[129] Y. Tang et al., "The sensory neuron as a transformer: Permutation-invariant neural networks for reinforcement learning," in *NeurIPS*, 2021.

[130] N. Topin et al., "Iterative bounding mdps: Learning interpretable policies via non-interpretable methods," in *AAAI*, 2021.

[131] X. Liu et al., "Fidelity-induced interpretable policy extraction for reinforcement learning," *arXiv preprint arXiv:2309.06097*, 2023.

[132] W. Guo et al., "Edge: Explaining deep reinforcement learning policies," in *NeurIPS*, 2021.

[133] P.-J. Kindermans et al., "The (un) reliability of saliency methods," in *Explainable AI: Interpreting, Explaining and Visualizing Deep Learning*, 2019.

[134] M. Sundararajan et al., "Axiomatic attribution for deep networks," in *ICML*, 2017.

[135] A. Binder et al., "Analyzing and validating neural networks predictions," in *ICML Workshop*, 2016.

[136] T. T. Nguyen et al., "A model-agnostic approach to quantifying the informativeness of explanation methods for time series classification," in *ECML PKDD Workshop*, 2020.

[137] J. Adebayo et al., "Local explanation methods for deep neural networks lack sensitivity to parameter values," *arXiv preprint arXiv:1810.03307*, 2018.

[138] ——, "Sanity checks for saliency maps," in *NeurIPS*, 2018.

[139] M. Gevrey et al., "Review and comparison of methods to study the contribution of variables in artificial neural network models," *Ecol. Modell.*, 2003.

[140] F. Maes et al., "Policy search in a space of simple closed-form formulas: Towards interpretability of reinforcement learning," in *DS*, 2012.

[141] J. P. Inala et al., "Neurosymbolic transformers for multi-agent communication," in *NeurIPS*, 2020.

[142] D. Trivedi et al., "Learning to synthesize programs as interpretable and generalizable policies," in *NeurIPS*, 2021.

[143] D. Hein et al., "Interpretable policies for reinforcement learning by genetic programming," *EAAI*, 2018.

[144] ——, "Generating interpretable reinforcement learning policies using genetic programming," in *GECCO*, 2019.

[145] M. Landajuela et al., "Discovering symbolic policies with deep reinforcement learning," in *ICML*, 2021.

[146] Q. Delfosse et al., "Interpretable and explainable logical policies via neurally guided symbolic abstraction," *NeurIPS*, 2024.

[147] D. Hein et al., "Particle swarm optimization for generating interpretable fuzzy reinforcement learning policies," *EAAI*, 2017.

[148] R. Akrour et al., "Towards reinforcement learning of human readable policies," in *ECML PKDD workshop*, 2019.

[149] L. Ou et al., "Fuzzy centered explainable network for reinforcement learning," *TFS*, 2023.

[150] A. Payani et al., "Inductive logic programming via differentiable deep neural logic networks," *arXiv preprint arXiv:1906.03523*, 2019.

[151] ——, "Incorporating relational background knowledge into reinforcement learning via differentiable inductive logic programming," *arXiv preprint arXiv:2003.10386*, 2020.

[152] S. Ross et al., "A reduction of imitation learning and structured prediction to no-regret online learning," in *AISTATS*, 2011.

[153] N. Topin et al., "Generation of policy-level explanations for reinforcement learning," in *AAAI*, 2019.

[154] L. L. Custode et al., "Evolutionary learning of interpretable decision trees," in *IJCAI*, 2022.

[155] A. M. Roth et al., "Conservative q-improvement: Reinforcement learning for an interpretable decision-tree policy," *arXiv preprint arXiv:1907.01180*, 2019.

[156] S. Milani et al., "Maviper: Learning decision tree policies for interpretable multi-agent reinforcement learning," *arXiv preprint arXiv:2205.12449*, 2022.

[157] M. Du et al., "Techniques for interpretable machine learning," *Communications of the ACM*, 2019.

[158] W. T. B. Uther et al., "Tree based discretization for continuous state space reinforcement learning," in *AAAI*, 1998.

[159] J. Snoek et al., "Practical bayesian optimization of machine learning algorithms," in *NeurIPS*, 2012.

[160] Q. Lu et al., "Using genetic programming with prior formula knowledge to solve symbolic regression problem," *Computational Intelligence and Neuroscience*, vol. 2016, 2016.

[161] J. Barwise, "An introduction to first-order logic," in *Studies in Logic and the Foundations of Mathematics*, 1977, vol. 90, pp. 5–46.

[162] M. Zimmer et al., "Differentiable logic machines," *arXiv preprint arXiv:2102.11529*, 2021.

[163] G. Hinton, O. Vinyals, J. Dean et al., "Distilling the knowledge in a neural network," *arXiv preprint arXiv:1503.02531*, 2015.

[164] M. L. Olson et al., "Counterfactual state explanations for reinforcement learning agents via generative deep learning," *AI*, 2021.

[165] G. Stein, "Generating high-quality explanations for navigation in partially-revealed environments," in *NeurIPS*, 2021.

[166] Y. Amitai and O. Amir, ""i don't think so": Summarizing policy disagreements for agent comparison," in *AAAI*, 2022.

[167] Z. Yu et al., "Explainable reinforcement learning via a causal world model," *arXiv preprint arXiv:2305.02749*, 2023.

[168] A. Meulemans et al., "Would i have gotten that reward? long-term credit assignment by counterfactual contribution analysis," *NeurIPS*, 2024.

[169] Y. Amitai et al., "Explaining reinforcement learning agents through counterfactual action outcomes," in *AAAI*, 2024.

[170] K. Boggess et al., "Toward policy explanations for multi-agent reinforcement learning," *arXiv preprint arXiv:2204.12568*, 2022.

[171] ——, "Explainable multi-agent reinforcement learning for temporal queries," *arXiv preprint arXiv:2305.10378*, 2023.

[172] Y. Kazak et al., "Verifying deep-rl-driven systems," in *SIGCOMM workshop*, 2019.

[173] H. Zhu et al., "An inductive synthesis framework for verifiable reinforcement learning," in *PLDI*, 2019.

[174] G. Anderson et al., "Neurosymbolic reinforcement learning with formally verified exploration," in *NeurIPS*, 2020.

[175] P. Jin et al., "Trainify: A cegar-driven training and verification framework for safe deep reinforcement learning," in *CAV*, 2022.

[176] Y. Fukuchi et al., "Autonomous self-explanation of behavior for interactive reinforcement learning agents," in *HAI*, 2017.

[177] ——, "Application of instruction-based behavior explanation to a reinforcement learning agent with changing policy," in *NeurIPS*, 2017.

[178] Z. Juozapaitis et al., "Explainable reinforcement learning via reward decomposition," in *IJCAI*, 2019.

[179] M. Liu et al., "Visual imitation learning with patch rewards," *arXiv preprint arXiv:2302.00965*, 2023.

[180] J. Foerster et al., "Counterfactual multi-agent policy gradients," in *AAAI*, 2018.

[181] J. Wang et al., "Shapley q-value: A local reward approach to solve global reward games," in *AAAI*, 2020.

[182] J. Li et al., "Shapley counterfactual credits for multi-agent reinforcement learning," in *KDD*, 2021.

[183] A. Tabrez et al., "Improving human-robot interaction through explainable reinforcement learning," in *HRI*, 2019.

[184] D. Lyu et al., "Sdrl: interpretable and data-efficient deep reinforcement learning leveraging symbolic planning," in *AAAI*, 2019.

[185] J. Wu et al., "Tree-structured policy based progressive reinforcement learning for temporally language grounding in video," in *AAAI*, 2020.

[186] H. Wu et al., "Self-supervised attention-aware reinforcement learning," in *AAAI*, 2021.

[187] S. Mirchandani et al., "Ella: Exploration through learned language abstraction," in *NeurIPS*, 2021.

[188] M. Jin et al., "Creativity of ai: Automatic symbolic option discovery for facilitating deep reinforcement learning," in *AAAI*, 2022.

[189] Z. Ashwood et al., "Dynamic inverse reinforcement learning for characterizing animal behavior," in *NeurIPS*, 2022.

[190] A. E. Roth, "Introduction to the shapley value," *The Shapley Value*, pp. 1–27, 1988.

[191] J. Zheng et al., "Robust bayesian inverse reinforcement learning with sparse behavior noise," in *AAAI*, 2014.

[192] I. Mishra et al., "Visual sparse bayesian reinforcement learning: a framework for interpreting what an agent has learned," in *SSCI*, 2018.

[193] K. Zhang et al., "Explainable ai in deep reinforcement learning models for power system emergency control," *TCSS*, 2021.

[194] A. Heuillet et al., "Collective explainable ai: Explaining cooperative strategies and agent contribution in multiagent reinforcement learning with shapley values," *IEEE Computational Intelligence Magazine*, 2022.

[195] E. M. Kenny et al., "Towards interpretable deep reinforcement learning with human-friendly prototypes," in *ICLR*, 2022.

[196] R. Ragodos et al., "Protox: Explaining a reinforcement learning agent via prototyping," in *NeurIPS*, 2022.

[197] S. V. Deshmukh et al., "Explaining rl decisions with trajectories," *arXiv preprint arXiv:2305.04073*, 2023.

[198] H. Sun et al., "Accountability in offline reinforcement learning: Explaining decisions with a corpus of examples," *NeurIPS*, 2024.

[199] V. Goel et al., "Unsupervised video object segmentation for deep reinforcement learning," in *NeurIPS*, 2018.

[200] V. Petsiuk et al., "Rise: Randomized input sampling for explanation of black-box models," in *BMVC*, 2018.

[201] J. Wang et al., "Dqnviz: A visual analytics approach to understand deep q-networks," *TVCG*, 2019.

[202] R. M. Annasamy et al., "Towards better interpretability in deep q-networks," in *AAAI*, 2019.

[203] Y. Tang et al., "Neuroevolution of self-interpretable agents," in *GECCO*, 2020.

[204] Y. Xu et al., "Deep reinforcement learning with stacked hierarchical attention for text-based games," in *NeurIPS*, 2020.

[205] M. Pan et al., "xgail: Explainable generative adversarial imitation learning for explainable human decision analysis," in *KDD*, 2020.

[206] S. S. Guo et al., "Machine versus human attention in deep reinforcement learning tasks," in *NeurIPS*, 2021.

[207] S. Wäldchen et al., "Training characteristic functions with reinforcement learning: Xai-methods play connect four," in *ICML*, 2022.

[208] D. Bertoin et al., "Look where you look! saliency-guided q-networks for generalization in visual reinforcement learning," in *NeurIPS*, 2022.

[209] X. Peng et al., "Inherently explainable reinforcement learning in natural language," in *NeurIPS*, 2022.

[210] D. Beechey et al., "Explaining reinforcement learning with shapley values," in *ICML*, 2023.

[211] C. Wang et al., "Explainable deep adversarial reinforcement learning approach for robust autonomous driving," *TIV*, 2024.

[212] J. van der Waa et al., "Contrastive explanations for reinforcement learning in terms of expected consequences," *arXiv preprint arXiv:1807.08706*, 2018.

[213] X. Pan et al., "Semantic predictive control for explainable and efficient policy learning," in *ICRA*, 2019.

[214] B. Lütjens et al., "Safe reinforcement learning with model uncertainty estimates," in *ICRA*, 2019.

[215] H. Yau et al., "What did you think would happen? explaining agent behaviour through intended outcomes," in *NeurIPS*, 2020.

[216] L. Lee et al., "Weakly-supervised reinforcement learning for controllable behavior," in *NeurIPS*, 2020.

[217] B. Hu et al., "An explainable and robust motion planning and control approach for autonomous vehicle on-ramping merging task using deep reinforcement learning," in *TTE*, 2023.

[218] K. Lee et al., "Refining diffusion planner for reliable behavior synthesis by automatic detection of infeasible plans," in *NeurIPS*, 2023.

[219] J. Wang et al., "Alphastock: A buying-winners-and-selling-losers investment strategy using interpretable deep reinforcement attention networks," in *KDD*, 2019.

[220] H. Hasselt, "Double q-learning," in *NeurIPS*, 2010.

[221] G. Nangue Tasse et al., "A boolean task algebra for reinforcement learning," in *NeurIPS*, 2020.

[222] Y. Jiang et al., "Language as an abstraction for hierarchical deep reinforcement learning," in *NeurIPS*, 2019.

[223] B. Beyret et al., "Dot-to-dot: Explainable hierarchical reinforcement learning for robotic manipulation," in *IROS*, 2019.

[224] B. Wu et al., "Model primitives for hierarchical lifelong reinforcement learning," *AAMAS*, 2020.

[225] S. Sodhani et al., "Multi-task reinforcement learning with context-based representations," in *ICML*, 2021.

[226] V. Zhong et al., "Improving policy learning via language dynamics distillation," in *NeurIPS*, 2022.

[227] R. Luss et al., "Local explanations for reinforcement learning," in *IJCAI*, 2022.

[228] P. Zhang et al., "Kogun: accelerating deep reinforcement learning via integrating human suboptimal knowledge," *arXiv preprint arXiv:2002.07418*, 2020.

[229] P. Goyal et al., "Using natural language for reward shaping in reinforcement learning," in *IJCAI*, 2019.

[230] J. Kim et al., "Textual explanations for self-driving vehicles," in *ECCV*, 2018.

[231] Y. Li et al., "In the eye of beholder: Joint learning of gaze and actions in first person video," in *ECCV*, 2018.

[232] V. Chen et al., "Ask your humans: Using human instructions to improve generalization in reinforcement learning," in *ICLR*, 2021.

[233] T. Rudolf et al., "Fuzzy action-masked reinforcement learning behavior planning for highly automated driving," in *CCAR*, 2022.

[234] W. Shi et al., "Efficient hierarchical policy network with fuzzy rules," *IJMLC*, 2022.

[235] A. Y. Ng et al., "Policy invariance under reward transformations: Theory and application to reward shaping," in *ICML*, 1999.

[236] Y. Burda et al., "Exploration by random network distillation," *arXiv preprint arXiv:1810.12894*, 2018.

[237] A. P. Badia et al., "Agent57: Outperforming the atari human benchmark," in *ICML*, 2020.

[238] A. Harutyunyan et al., "Hindsight credit assignment," in *NeurIPS*, 2019.

[239] Y. Liu et al., "Sequence modeling of temporal credit assignment for episodic reinforcement learning," *arXiv preprint arXiv:1905.13420*, 2019.

[240] R. Zhang et al., "Atari-head: Atari human eye-tracking and demonstration dataset," in *AAAI*, 2020.

[241] Y. Xu et al., "Perceiving the world: Question-guided reinforcement learning for text-based games," *arXiv preprint arXiv:2204.09597*, 2022.

[242] K. Hitomi et al., "Development of a dangerous driving suppression system using inverse reinforcement learning and blockchain," in *DCAI*, 2022.

[243] B. Ghai et al., "Explainable active learning (xal) toward ai explanations as interfaces for machine teachers," *HCI*, 2021.

[244] C. Wirth et al., "A survey of preference-based reinforcement learning methods," *JMLR*, 2017.

[245] P. F. Christiano et al., "Deep reinforcement learning from human preferences," in *NeurIPS*, 2017.

[246] G. Zhang et al., "Learning state importance for preference-based reinforcement learning," *ML*, 2023.

[247] Y. Shen et al., "Autopreview: A framework for autopilot behavior understanding," in *CHI*, 2021.

[248] M. T. Ribeiro et al., "Model-agnostic interpretability of machine learning," *arXiv preprint arXiv:1606.05386*, 2016.

[249] H. Mania et al., "Simple random search of static linear policies is competitive for reinforcement learning," in *NeurIPS*, 2018.

[250] C. Rudin et al., "The secrets of machine learning: ten things you wish you had known earlier to be more effective at data analysis," in *INFORMS*, 2019.

[251] ——, "Interpretable machine learning: Fundamental principles and 10 grand challenges," *Statistic Surveys*, 2022.

---

## BibTeX Citation

```bibtex
@article{Qing2022XRLSurvey,
  author       = {Qing, Yunpeng and Liu, Shunyu and Song, Jie and Zhou, Yang and Chen, Kaixuan and Wang, Huiqiong and Song, Mingli},
  title        = {A Survey on Explainable Reinforcement Learning: Concepts, Algorithms, and Challenges},
  journal      = {arXiv preprint arXiv:2211.06665},
  year         = {2022},
  eprint       = {2211.06665},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2211.06665},
  note         = {v6, 29 Aug 2025}
}
```
