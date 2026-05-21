A Survey on Explainable Deep Reinforcement Learning

Zelei Cheng, Jiahao Yu, Xinyu Xing

Northwestern University. Correspondence: {zelei.cheng, jiahao.yu, xinyu.xing}@northwestern.edu. These authors (Zelei Cheng and Jiahao Yu) contributed equally to this work.

## Abstract

Deep Reinforcement Learning (DRL) has achieved remarkable success in sequential decision-making tasks across diverse domains, yet its reliance on black-box neural architectures hinders interpretability, trust, and deployment in high-stakes applications. Explainable Deep Reinforcement Learning (XRL) addresses these challenges by enhancing transparency through feature-level, state-level, dataset-level, and model-level explanation techniques. This survey provides a comprehensive review of XRL methods, evaluates their qualitative and quantitative assessment frameworks, and explores their role in policy refinement, adversarial robustness, and security. Additionally, we examine the integration of reinforcement learning with Large Language Models (LLMs), particularly through Reinforcement Learning from Human Feedback (RLHF), which optimizes AI alignment with human preferences. We conclude by highlighting open research challenges and future directions to advance the development of interpretable, reliable, and accountable DRL systems.

## 1 Introduction

Deep Reinforcement Learning (DRL) has emerged as a transformative paradigm for solving complex sequential decision-making problems. By enabling autonomous agents to interact with an environment, receive feedback in the form of rewards, and iteratively refine their policies, DRL has demonstrated remarkable success across a diverse range of domains including games (e.g., Atari [Mnih, 2013; Kaiser et al., 2020], Go [Silver et al., 2018, 2017], and StarCraft II [Vinyals et al., 2019, 2017]), robotics [Kalashnikov et al., 2018], communication networks [Feriani and Hossain, 2021], and finance [Liu et al., 2024]. These successes underscore DRL's capability to surpass traditional rule-based systems, particularly in high-dimensional and dynamically evolving environments.

Despite these advances, a fundamental challenge remains: DRL agents typically rely on deep neural networks, which operate as black-box models, obscuring the rationale behind their decision-making processes. This opacity poses significant barriers to adoption in safety-critical and high-stakes applications, where interpretability is crucial for trust, compliance, and debugging. The lack of transparency in DRL can lead to unreliable decision-making, rendering it unsuitable for domains where explainability is a prerequisite, such as healthcare, autonomous driving, and financial risk assessment.

To address these concerns, the field of Explainable Deep Reinforcement Learning (XRL) has emerged, aiming to develop techniques that enhance the interpretability of DRL policies. XRL seeks to provide insights into an agent's decision-making process, enabling researchers, practitioners, and end-users to understand, validate, and refine learned policies. By facilitating greater transparency, XRL contributes to the development of safer, more robust, and ethically aligned AI systems.

Furthermore, the increasing integration of Reinforcement Learning (RL) with Large Language Models (LLMs) has placed RL at the forefront of natural language processing (NLP) advancements. Methods such as Reinforcement Learning from Human Feedback (RLHF) [Bai et al., 2022; Ouyang et al., 2022] have become essential for aligning LLM outputs with human preferences and ethical guidelines. By treating language generation as a sequential decision-making process, RL-based fine-tuning enables LLMs to optimize for attributes such as factual accuracy, coherence, and user satisfaction, surpassing conventional supervised learning techniques. However, the application of RL in LLM alignment further amplifies the explainability challenge, as the complex interactions between RL updates and neural representations remain poorly understood.

This survey provides a systematic review of explainability methods in DRL, with a particular focus on their integration with LLMs and human-in-the-loop systems. We first introduce fundamental RL concepts and highlight key advances in DRL. We then categorize and analyze existing explanation techniques, encompassing feature-level, state-level, dataset-level, and model-level approaches. Additionally, we discuss methods for evaluating XRL techniques, considering both qualitative and quantitative assessment criteria. Finally, we explore real-world applications of XRL, including policy refinement, adversarial attack mitigation, and emerging challenges in ensuring interpretability in modern AI systems. Through this survey, we aim to provide a comprehensive perspective on the current state of XRL and outline future research directions to advance the development of interpretable and trustworthy DRL models.

## 2 Preliminaries

### 2.1 Reinforcement Learning Foundations

Reinforcement Learning (RL) is a subfield of machine learning that focuses on training agents to make sequential decisions by interacting with an environment. The environment is framed as a Markov Decision Process (MDP) [Sutton and Barto, 2018], specified by the tuple $(\mathcal{S}, \mathcal{A}, P, \rho, R, \gamma)$:

- $\mathcal{S}$: A set of states representing possible configurations of the environment.
- $\mathcal{A}$: A set of actions available to the agent.
- $P(s' \mid s, a)$: The transition probability function describing how actions lead from one state $s$ to another state $s'$.
- $\rho$: the distribution of the initial state $s_0$.
- $R(s, a)$: The immediate reward obtained after executing action $a$ in state $s$.
- $\gamma \in (0, 1)$: A discount factor that balances immediate and future rewards.

The goal of RL is to find an optimal policy $\pi(a|s): (\mathcal{S} \to \mathcal{A})$ which maximizes the agent's long-term reward. Formally, the long-term reward is defined as the *state-value function*

$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s) \left[ R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s'|s, a) V^\pi(s') \right]. \quad (1)$$

Accordingly, the *action-value function* $Q^\pi(s, a)$ is defined as

$$Q^\pi(s, a) = R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s'|s, a) \sum_{a' \in \mathcal{A}} \pi(a'|s') Q^\pi(s', a'). \quad (2)$$

The *advantage function* $A^\pi(s, a)$ is defined as

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s). \quad (3)$$

In reinforcement learning, the state-value function $V^\pi(s)$ represents the expected total reward for an agent starting from state $s$. Slightly different from $V^\pi(s)$, the action-value function $Q^\pi(s, a)$ is the expected total reward for an agent to choose action $a$ while in state $s$. The advantage function measures the expected additional reward for choosing action $a$ over the expected reward of the policy. The expected total reward of a policy $\pi$ is defined as

$$\eta(\pi) = \mathbb{E}_{s_0, a_0, \ldots} \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) \right]. \quad (4)$$

By maximizing the expected total reward, an optimal policy $\pi^*$ can be derived, enabling the agent to receive the maximum rewards in the environment.

Reinforcement learning can be categorized into two primary settings based on the agent's ability to interact with the environment: online RL and offline RL. In online RL, the agent has direct, interactive access to the environment and can continuously collect new experiences by executing and updating its policy in real-time. This setting allows for active exploration and immediate policy adaptation. In contrast, offline RL restricts the agent to learn solely from a fixed dataset of previously collected experiences, without any further environment interaction. This dataset typically consists of state-action-reward trajectories collected by one or multiple behavior policies. The offline setting is particularly relevant in scenarios where environment interaction is expensive, risky, or impractical, such as in healthcare, autonomous driving, or industrial control systems.

There are two types of main-stream algorithms, i.e., value-based methods and policy-based methods. For value-based methods such as Q-learning algorithm [Watkins and Dayan, 1992], the agent estimates $Q(s, a)$ and greedily chooses the optimal action. Regarding policy-based methods, the agent directly optimizes its policy based on the reward feedback (e.g., Policy Gradient methods [Sutton et al., 1999]). Classic algorithms have been effective in relatively small or structured environments. However, their performance may degrade in high-dimensional or unstructured domains due to challenges in representation and exploration.

### 2.2 Deep Reinforcement Learning Advancements

To address the limitations of standard RL in complex or high-dimensional settings, Deep Reinforcement Learning (DRL) integrates neural networks as function approximators for policies or value functions. Two prominent approaches for learning deep reinforcement learning policies are Deep Q-Network (DQN) [Mnih et al., 2015] and Proximal Policy Optimization (PPO) [Schulman et al., 2017]. We provide a brief overview of the foundational principles underlying each of these algorithms.

**Deep Q-Network (DQN).** DQN utilizes a deep neural network to approximate the optimal action-value function (Q function). The network architecture typically processes state inputs $s$ through several layers and outputs Q-values for all possible actions simultaneously. The network is trained by minimizing the temporal difference error between predicted and target Q-values using experience replay and a target network to stabilize training. During execution, the optimal policy is derived by selecting the action with the highest predicted Q-value.

**Proximal Policy Optimization (PPO).** Different from DQN, policy gradient methods directly learn a parameterized policy $\pi_\theta(a|s) = \mathbb{P}(a|s, \theta)$ to maximize the expected total reward. While these methods offer more direct policy optimization, they often suffer from high variance and sensitivity to learning rates, leading to unstable training. PPO addresses these challenges by introducing a clipped surrogate objective function. It constrains policy updates to prevent excessive changes while optimizing performance. By maintaining proximity between consecutive policies and using advantage estimation, PPO achieves more stable training and better sample efficiency compared to traditional policy gradient methods, making it one of the most widely adopted algorithms in practice.

### 2.3 Reinforcement Learning for LLMs

The integration of RL with Large Language Models (LLMs) has emerged as a promising direction for improving the alignment and performance of AI systems. Multiple RL approaches such as PPO, Directed Preference Optimization (DPO) [Rafailov et al., 2023], Reward rAnked FineTuning (RAFT) [Dong et al., 2023] have been used to fine-tune LLMs for specific tasks, such as dialogue generation, summarization, and instruction following. By leveraging reward feedback, RL-based approaches enable LLMs to generate more coherent, contextually appropriate, and user-aligned outputs.

Despite these advancements, the explainability of RL for LLMs remains an open challenge. The complexity of LLMs, combined with the sequential decision-making nature of RL, makes it difficult to interpret how the input data impacts these models to generate outputs. Recent efforts have explored techniques such as data influence functions to enhance the transparency of RL for LLMs. However, there is still a need for more systematic explanation approaches in this domain, particularly for applications involving ethical considerations, bias mitigation, and user trust.

In the subsequent sections, we survey existing methods for providing interpretability in DRL systems as well as LLMs, and discuss how these techniques can be evaluated and applied in practice.

## 3 Explanation Techniques for DRL

Existing approaches to explaining deep reinforcement learning can be broadly categorized into four categories: (1) Feature-level Explanation Methods, which focuses on pinpointing the most important feature in the DRL agent's observation; (2) State-level Explanation Methods, which identifies the most critical steps in the RL trajectory; (3) Dataset-level Explanation Methods, which selects the most influential data in RL; (4) Model-level Explanation Methods, which focuses on the self-explainability of RL policy models. A summary of selected methods is provided in Figure 1.

### 3.1 Feature-level Explanation Methods

Feature-level explanation methods aim to identify the most important features in the agent's observation space that influence its decision-making. These methods are particularly useful for understanding how an agent processes visual inputs.

Zahavy et al. [2016] approximated the behavior of DRL agents via Semi-Aggregated Markov Decision Processes (SAMDPs) and analyzed the high-level temporal structure of the policy with the more interpretable SAMDPs. However, the explanation from SAMDPs is drawn from t-SNE clusters which could be uninformative for users without machine learning backgrounds. To make the explanation more accessible, Greydanus et al. [2018] proposed a feature-level explanation method to visualize the importance of pixels in Atari game frames by perturbing the input and observing changes in the agent's policy.

In addition to perturbation-based saliency methods, some researchers also proposed gradient-based saliency methods that use gradients of the agent's policy or value function to pinpoint the most important feature in DRL agent's observation. Wang et al. [2016] extend gradient-based saliency maps to deep RL by computing the Jacobian of the output logits with respect to a stack of input images. Joo and Kim [2019] leveraged Grad-Cam [Selvaraju et al., 2017] to visualize the important features towards the DRL agent's behavior. Cheng et al. [2024] mentioned that we can also use integrated gradients [Sundararajan et al., 2017] to identify the most important features.

Recent advancements in deep reinforcement learning (DRL) also introduce attention-based mechanisms to enhance feature-level explanations of agent behavior. These methods aim to improve interpretability by enabling agents to focus on task-relevant information within their observation space. For instance, Mott et al. [2019] proposed an attention-augmented agent that employs a soft attention mechanism, allowing the agent to sequentially query its environment and focus on pertinent features during decision-making. This approach not only enhances performance but also provides interpretable attention maps that highlight the areas of the input contributing to the agent's actions. Similarly, Nikulin et al. [2019] introduced a method that integrates an attention module into the agent's architecture, producing saliency maps that visualize the importance of different input regions in the agent's decision process.

These methods provide insights into the agent's perception of the environment but are often limited to explaining low-level features rather than high-level decision-making processes.

### 3.2 State-level Explanation Methods

State-level explanation methods focus on identifying critical states in the agent's trajectory that significantly impact its performance. These methods are useful for understanding the agent's behavior over time and diagnosing failures. We categorize state-level explanation methods into two categories: (1) Explain through offline trajectories; (2) Explain through online interactions.

For the first category, Guo et al. [2021] first proposed EDGE that establishes state-reward relationship by collecting a set of trajectories and then approximating an explanation model offline with the Gaussian Process. Note that, EDGE provides a *global explanation* for the policy network. AIRS [Yu et al., 2023] further introduces a *local explanation* method to identify critical time steps for a given trajectory of interest. AIRS pre-collects a set of trajectories and utilizes a deep neural network to estimate the contribution of each state to the final rewards for each trajectory. Liu et al. [2023] proposed a Deep State Identifier that learns to predict returns from episodes and uses mask-based sensitivity analysis to extract important states. However, the fidelity of these methods is highly related to the quality of the pre-collected trajectories, which limits their ability to measure the importance of "unseen states".

For the second category, Jacq et al. [2022] presented LazyMDP, which extends the action space with a lazy action and learns to switch between the default action and the lazy action. The states where the policy diverges from the default are further interpreted as non-important states. Cheng et al. [2023] proposed StateMask, which online trains a mask network in parallel with the agent's policy network. The mask network learns to "blind" the agent's observations at certain time steps (by taking random actions) while minimizing the impact of blinding to the final reward. The time steps when the agent could be blinded are identified as non-critical steps.

State-level explanations are particularly valuable for debugging and improving RL agents, as they highlight the most influential moments in the agent's decision-making process.

### 3.3 Dataset-level Explanation Methods

Dataset-level explanation methods focus on understanding how specific training examples influence the learned policy of an RL agent. By identifying which data points have the most impact on the policy updates, researchers and practitioners can better diagnose training inefficiencies, detect harmful experiences, and refine data collection strategies. Recent work has highlighted multiple approaches for quantifying this influence:

**Influence Functions.** Originally introduced by Koh and Liang [2017], influence functions estimate how an upweighting or removal of a single training example impacts model parameters. In RL contexts, these techniques can be adapted to analyze individual experiences in a replay buffer, thereby revealing which transitions most critically shape the agent's behavior. When incorporating RL with LLMs, Li et al. [2024]; Matelsky et al. [2024]; Ruis et al. [2024] also investigated the feasibility of leveraging influence functions to identify influential data. However, they found that influence functions show poor performance and the reasons might be (1) inevitable approximation errors when estimating the inverse-Hessian vector products (iHVP) component due to the scale of LLMs, (2) uncertain convergence during fine-tuning, (3) the definition of influential data as changes in model parameters do not necessarily correlate with changes in LLM behavior.

**Data Shapley Values.** Shapley values, proposed by Ghorbani and Zou [2019], offer a game-theoretic metric for attributing credit to each data point. By considering all possible subsets of the training set, Data Shapley Values can rank experiences according to their overall contribution to policy performance. However, the original Data Shapley Values are computationally intensive, Wang et al. [2024a] proposed an approximation method FreeShap for instance attribution based on the neural tangent kernel, which makes this method feasible for explaining LLM predictions.

**Data Masking.** Recent advances have introduced masking as a way to figure out how specific elements of a training dataset shape an agent's learning process [Dong et al., 2024; Lin et al., 2024]. Rather than simply omitting entire experiences, data masking strategically hides or perturbs certain tokens and observes how these modifications affect the LLM's performance. Therefore, researchers can pinpoint the data components most critical to LLM training and can construct a pruned dataset based on the critical data to efficiently train a LLM.

Dataset-level explanations help researchers and practitioners understand the role of training data in shaping the RL agent's behavior and can guide the design of more efficient and effective training schemes.

### 3.4 Model-level Explanation Methods

Model-level explanation methods focus on the self-explainability of RL policy models, aiming to make the agent's decision-making process inherently interpretable. These methods often involve designing transparent architectures (e.g., decision tree [Topin et al., 2021; Ding et al., 2020]) or extracting human-understandable rules [Soares et al., 2020; Likmeta et al., 2020] from the agent's policy. Demircan et al. [2024] utilize sparse autoencoders within the policy network to provide detailed explanations of LLM's behavior, specifically focusing on how the network approximates Q-learning by revealing the underlying structure and decision-making process of the model.

Model-level explanations are particularly valuable for applications requiring high transparency, such as healthcare and autonomous driving, where understanding the agent's reasoning is critical for trust and safety.

**Figure 1.** Taxonomy of DRL Explanation Methods. Hierarchical tree diagram organizing DRL Explanation Methods into four top-level branches: Feature-level (sub-branches: Perturbation-based — Zahavy et al. [2016], Greydanus et al. [2018], Atrey et al. [2020]; Gradient-based — Wang et al. [2016], Selvaraju et al. [2017], Sundararajan et al. [2017]; Attention-based — Mott et al. [2019], Nikulin et al. [2019]); State-level (Offline Trajectories — Guo et al. [2021], Yu et al. [2023], Liu et al. [2023]; Online Interactions — Jacq et al. [2022], Cheng et al. [2023, 2024]); Dataset-level (Influence Functions — Koh and Liang [2017], Li et al. [2024], Matelsky et al. [2024], Ruis et al. [2024]; Data Shapley — Ghorbani and Zou [2019], Wang et al. [2024a], Schoch et al. [2023]; Data Masking — Dong et al. [2024], Lin et al. [2024]); Model-level (Transparent Architectures — Topin et al. [2021], Ding et al. [2020], Demircan et al. [2024]; Rule Extraction — Soares et al. [2020], Likmeta et al. [2020]).

## 4 Measuring XRL

Evaluating the quality of explanations in RL requires a multi-faceted approach that captures both user-centered dimensions and objective metrics. This section outlines two broad categories of assessment, i.e., qualitative and quantitative.

### 4.1 Qualitative Evaluation

**Interpretability and Clarity.** At the heart of XRL is the need for explanations that humans find meaningful and intuitive. Qualitative evaluation often begins with user studies, such as surveys, to gauge how well participants understand the explanation and whether the information provided is perceived as coherent and sufficient for understanding policy decisions. Most researchers provide a visualization of the proposed explanation technique to demonstrate to the participants that the explanation can help them understand the DRL agent's behavior. For feature-level explanations, Greydanus et al. [2018] generated saliency videos to show the feature-level explanations for Atari games and conducted a survey over 31 students at Oregon State University to measure how their visualization helps non-experts with these Atari games. For state-level explanations, Cheng et al. [2023] generated game trajectories with a color bar behind each frame to indicate the importance of each state and invited participants to answer a questionnaire to demonstrate their method StateMask could help humans gain a better understanding of a DRL agent's behavior.

**User-Centered Design Considerations.** The qualitative evaluation also informs iterative refinement of explanation interfaces. By examining user reactions, researchers and designers can identify which presentation formats (e.g., visual overlays, textual rationales, or example-based justifications) are most effective. This feedback loop, encompassing pilot testing and usability reviews, ensures that explanations remain aligned with the domain's practical needs and the target audience's expertise.

### 4.2 Quantitative Evaluation

**Fidelity and Faithfulness.** A key quantitative metric is how closely an explanation reflects the true policy or behavior of the RL agent. To evaluate the fidelity of the explanation in RL, researchers commonly use a perturbation-based approach [Guo et al., 2021; Cheng et al., 2023]. The researchers remove features/states/data points identified as critical in the explanation and check if such a removal substantially degrades the agent's performance. A dual form of this approach is to remove the non-critical features/states/data points and the agent's performance is expected to have limited difference. The fidelity score is further measured as the performance difference of the DRL agent before and after perturbing a fixed number of pixels/states/data points. When perturbing the same number of (critical) pixels/states/data points, a higher performance difference indicates a higher fidelity of the explanation method.

**Downstream Performance Impact.** XRL systems can also be evaluated on whether their explanations enhance agent performance. For instance, Cheng et al. [2024] tested their proposed refining method based on the critical steps identified by different explanation methods and compared the agent's performance after refining to evaluate the quality of these explanation methods.

## 5 Applications of Explanations

With the explanation of RL, there can be different applications of it - they can be leveraged both constructively (for policy refinement and debugging) and potentially destructively (for launching adversarial attacks). These applications demonstrate how fidelity and interpretability impact the effectiveness of explanation-based interventions in real-world scenarios.

**Table 1.** Taxonomy of Explanation-based Interventions in DRL.

| Category | Subcategory | Citation |
|---|---|---|
| Launching Adversarial Attacks | Targeted Attack | [Lin et al., 2020; Guo et al., 2021; Cheng et al., 2023; Wang et al., 2024b] |
| Mitigating Adversarial Attacks | Blinding Observations | [Guo et al., 2021] |
| Mitigating Adversarial Attacks | Shielding Backdoor Triggers | [Yuan et al., 2024] |
| Policy Refinement | Human-in-the-Loop Correction | [Van Waveren et al., 2022; Jiang et al., 2024] |
| Policy Refinement | Automated Policy Refinement | [Guo et al., 2021; Cheng et al., 2023; Yu et al., 2023; Cheng et al., 2024; Liu and Zhu, 2025] |

### 5.1 Launching Adversarial Attacks

Recent work demonstrates that explanations of a DRL agent's policy can be repurposed to compromise the agent's performance. Recent studies have revealed that explanations of a Deep Reinforcement Learning (DRL) agent's policy can be exploited to compromise the agent's performance. For instance, Lin et al. [2020] demonstrated the vulnerability of cooperative Multi-Agent Reinforcement Learning systems to adversarial attacks by introducing perturbations based on feature-level explanations (i.e., saliency) to the state space. They proposed a mechanism where an adversary adds perturbations to the observations of a single agent within a team, leading to a significant decrease in overall team performance.

Besides leveraging feature-level explanations to launch adversarial attacks, researchers also demonstrate that state-level explanations can be utilized to attack DRL agents. EDGE [Guo et al., 2021] proposes a more targeted approach by leveraging explanations to identify critical time steps during an episode. The attacker first collects winning episodes from the victim agent and uses post-hoc explanations to highlight moments where actions strongly contribute to victory. By forcing the agent to take sub-optimal actions at these identified crucial steps, the attack achieves significant performance degradation with minimal intervention.

Subsequent research by Cheng et al. [2023] confirms this explanation-driven attack generalizes across different DRL environments, showing that targeting just 10% of time steps can substantially reduce agent reward. Notably, attacks guided by high-fidelity explanation methods prove more effective than those using lower-fidelity alternatives, highlighting how better interpretability tools can paradoxically increase vulnerability.

In addition to exploiting feature-level and state-level explanations, recent research has explored the use of dataset-level explanations to launch adversarial attacks on LLMs. A study by Wang et al. [2024b] investigates the vulnerabilities of reinforcement learning with human feedback. The researchers employ a gradient-based dataset-level explanation method to identify influential data points within the training set. By poisoning a small percentage of critical data, an adversary can significantly manipulate the LLM's behavior, leading to the elicitation of harmful responses.

This line of work highlights the dual-edged nature of interpretability in RL. While explanations are invaluable for debugging and understanding agent behavior, they can also expose vulnerabilities. By identifying specific moments when an agent's correct actions matter most, adversaries can focus on minimal but high-impact interventions. Consequently, researchers must carefully consider the security implications of providing public or easily accessible explanation systems, especially in safety-critical or competitive domains.

### 5.2 Mitigating Adversarial Attacks

XRL methods not only reveal how adversaries can manipulate agents but can also guide the design of robust policies. By pinpointing which states or actions are most vulnerable, developers can selectively limit or modify the agent's observations and decision pathways at crucial moments, ultimately reducing susceptibility to adversarial inputs.

**Blinding Observations at Critical Time Steps.** Guo et al. [2021] illustrated how explanations of the victim agent's losing episodes in the You-Shall-Not-Pass game [Todorov et al., 2012] uncover the specific times when adversarial actions (e.g., pretending to fall) most effectively mislead the agent. By analyzing contrastive explanations—comparing losing and winning trajectories—it becomes clear that the agent's focus on adversarial cues at certain time steps can trigger sub-optimal responses. The authors proposed "blinding" the victim agent to these cues precisely at those critical moments. Experimental results show that this explanation-driven defense significantly boosts the victim's win rate, highlighting how identifying the root cause of agent failures can lead to targeted and effective countermeasures.

**Detecting and Shielding Backdoor Triggers.** Another line of work focuses on a subtler attack vector: maliciously injected backdoors. Yuan et al. [2024] introduced SHINE, a method to shield a pre-trained agent from both perturbation-based and adversarial-agent attacks in a poisoned environment. SHINE first gathers trajectories and employs a two-stage explanation process to (1) locate states where a backdoor trigger is likely active and (2) isolate the common subset of features critical to the agent's decisions in those states. These features are then treated as the backdoor signature. In the second stage, SHINE retrains the policy to neutralize the trigger's influence while preserving performance in a clean environment. This careful mixture of explanation and policy adjustment provides theoretical guarantees of improved robustness.

These defense mechanisms highlight how explanations serve defensive purposes in adversarial contexts. By precisely identifying *where* and *how* an agent's decision-making is compromised, explanation-guided strategies enable targeted fixes that enhance robustness. This demonstrates that transparency, when properly leveraged, can be a powerful tool for securing DRL agents rather than just exposing their vulnerabilities.

### 5.3 Policy Refinement Through Explanations

To refine the policy of the agents, conventional methods such as continual training [Fickinger et al., 2021] often fall short due to a lack of knowledge of the root causes of errors. There are two categories of methods for policy refinement through explanations:

- **Human-in-the-Loop Correction:** Domain experts or non-experts identify suboptimal actions or critical states, providing corrective demonstrations or reward adjustments.
- **Automated Policy Refinement with Explanation:** Explanation techniques automatically identify pivotal states and refine the target agent's policy based on the explanation.

For the first category, Van Waveren et al. [2022] proposed to utilize human feedback to correct the agent's failures. More specifically, when the agent fails, humans (can be non-experts) are involved to point out how to avoid such a failure (i.e., what action should be done instead, and what action should be forbidden). Based on human feedback, the DRL agent gets retrained by taking the human-refined action in those important time steps and finally obtains the corrected policy. The downside is that it relies on humans to identify critical steps and craft rules for alternative actions. This can be challenging for a large action space, and the retraining process is ad-hoc and time-consuming. To address the challenges of imperfect corrective actions and extensive human labor, Jiang et al. [2024] introduced the Iterative learning from Corrective actions and Proxy rewards (ICoPro) framework. In this approach, human labelers provide corrective actions on the agent's trajectories, which are then incorporated into the Q-function using a margin loss to enforce adherence to the labeler's preferences. The agent undergoes iterative training, balancing learning from both proxy rewards and human feedback. Notably, ICoPro integrates pseudo-labels from the target Q-network to reduce human labor and stabilize training. Experimental results in various tasks, including Atari games and autonomous driving scenarios, demonstrate that ICoPro effectively aligns agent behavior with human preferences, even when both proxy rewards and corrective actions are imperfect.

For the second category, Guo et al. [2021] proposed an explanation-guided policy refinement approach to automatically correct policy errors without relying on explicit human feedback. Their method first identifies losing episodes of the target agent and pinpoints crucial time steps within those episodes using its proposed explanation technique. The authors employ a fixed number of random explorations at the identified critical time steps. Any random actions that transform a losing episode into a win get stored in a look-up table as a remediation policy. When deployed, the agent consults this table at run-time: if the current state matches one of the stored entries, the agent applies the corresponding remediation action; otherwise, it defaults to its original policy. The success of this policy refinement approach depends heavily on the budget of random exploration and the size of the look-up table. Cheng et al. [2023]; Yu et al. [2023] further proposed to use DRL explanation methods to identify critical time steps and refine the agent by resetting the environment to the critical states and subsequently resuming training the DRL agents from these critical states. However, this refining strategy can easily lead to overfitting as evidenced in Cheng et al. [2024] and cannot help the agent escape the local optimal. Cheng et al. [2024] further proposed a novel refining strategy to construct a mixed initial state distribution with both the identified critical states and the default initial states to avoid overfitting and encourage the agent to perform exploration during the refining process. Recently, Liu and Zhu [2025] proposed a novel framework that leverages explainable reinforcement learning (XRL) to enhance policy refinement. This approach addresses the challenges of DRL agents' lack of transparency and suboptimal performance by providing a two-level explanation of the agents' decision-making processes. The framework identifies mistakes made by the DRL agent and formulates a constrained bi-level optimization problem to learn how to best utilize these explanations for policy improvement. The upper level of the optimization learns how to use high-level explanations to shape the reward function, while the lower level solves a constrained RL problem using low-level explanations. The proposed algorithm theoretically guarantees global optimality and has demonstrated superior performance in MuJoCo experiments compared to state-of-the-art baselines.

## 6 Conclusion and Future Directions

This survey has reviewed recent advances in the field of XRL, emphasizing a range of techniques - from feature-level, and state-level to dataset-level approaches, and illustrating their roles in adversarial attacks and mitigation, and policy refinement. Evidence across these methods indicates that effective explanations can significantly enhance trust and debugging efficiency in real-world deployments of deep reinforcement learning. Nonetheless, substantial gaps remain to be addressed, which are summarized as follows.

**User-Oriented Explanations.** Although existing techniques could highlight critical features/states to illustrate an agent's decision-making process, these granular depictions can be difficult for non-expert users to interpret. In the case of critical features, users who lack domain knowledge (e.g., specific familiarity with a particular game environment) may struggle to grasp the significance of highlighted features and how they influence the agent's actions. Meanwhile, understanding critical states often demands that users examine multiple visual frames and then manually summarize what these states imply about the agent's strategy. This process can be cognitively taxing, as it requires piecing together dispersed information and inferring the agent's underlying rationale without clear contextual guidance.

To address these challenges, future research should therefore prioritize strategy-level or narrative-based explanations, which can provide higher-level rationales that are more accessible to general audiences. In particular, leveraging vision–language models or other multimodal architectures could facilitate the presentation of natural language narratives that encapsulate an agent's overarching goals, strategies, and reasoning. These narrative formats have the potential to reduce cognitive load, enabling end users to more intuitively comprehend and trust the agent's behavior.

**Developer-Oriented Explanations.** In contrast, developers and researchers frequently require detailed insights into an agent's decision-making process. Mechanistic interpretation methods, such as sparse autoencoders or network dissection, could illuminate hidden representations and policy structures. These more granular approaches enable targeted policy debugging by pinpointing design flaws or overfitting at the architectural level. Crucially, explanations for developers should be *actionable*, which could be compatible with policy refinement workflows to accelerate iterative improvements.

In addition to improving interpretability, explainability tools offer considerable potential for enhancing policy performance. For instance, in game-theoretic contexts, explanations can help identify equilibrium strategies or support robust multi-agent interactions. In hierarchical reinforcement learning, clarifying subtask transitions can streamline learning in sparse-reward or long-horizon tasks. Similarly, in curriculum learning, highlighting critical states through explanation techniques can aid developers in selecting more effective initial conditions. Moving forward, future research should focus on aligning these interpretability and performance objectives by examining how transparent representations of policy decisions can foster robust learning or facilitate agent learning.

## References

Akanksha Atrey, Kaleigh Clary, and David Jensen. Exploratory not explanatory: Counterfactual analysis of saliency maps for deep reinforcement learning. In Proc. of ICLR, 2020.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Zelei Cheng, Xian Wu, Jiahao Yu, Wenhai Sun, Wenbo Guo, and Xinyu Xing. Statemask: Explaining deep reinforcement learning through state mask. In Proc. of NeurIPS, 2023.

Zelei Cheng, Xian Wu, Jiahao Yu, Sabrina Yang, Gang Wang, and Xinyu Xing. Rice: Breaking through the training bottlenecks of reinforcement learning with explanation. In Proc. of ICML, 2024.

Can Demircan, Tankred Saanum, Akshay K Jagadish, Marcel Binz, and Eric Schulz. Sparse autoencoders reveal temporal difference learning in large language models. arXiv preprint arXiv:2410.01280, 2024.

Zihan Ding, Pablo Hernandez-Leal, Gavin Weiguang Ding, Changjian Li, and Ruitong Huang. Cdt: Cascading decision trees for explainable reinforcement learning. arXiv preprint arXiv:2011.07553, 2020.

Hanze Dong, Wei Xiong, Deepanshu Goyal, Yihan Zhang, Winnie Chow, Rui Pan, Shizhe Diao, Jipeng Zhang, KaShun SHUM, and Tong Zhang. RAFT: Reward ranked finetuning for generative foundation model alignment. Transactions on Machine Learning Research, 2023.

Ximing Dong, Shaowei Wang, Dayi Lin, Gopi Krishnan Rajbahadur, Boquan Zhou, Shichao Liu, and Ahmed E Hassan. Promptexp: Multi-granularity prompt explanation of large language models. arXiv preprint arXiv:2410.13073, 2024.

Amal Feriani and Ekram Hossain. Single and multi-agent deep reinforcement learning for ai-enabled wireless networks: A tutorial. IEEE Communications Surveys & Tutorials, 23(2):1226–1252, 2021.

Arnaud Fickinger, Hengyuan Hu, Brandon Amos, Stuart Russell, and Noam Brown. Scalable online planning via reinforcement learning fine-tuning. In Proc. of NeurIPS, 2021.

Amirata Ghorbani and James Zou. Data shapley: Equitable valuation of data for machine learning. In Proc. of ICML, 2019.

Samuel Greydanus, Anurag Koul, Jonathan Dodge, and Alan Fern. Visualizing and understanding atari agents. In Proc. of ICML, 2018.

Wenbo Guo, Xian Wu, Usmann Khan, and Xinyu Xing. Edge: Explaining deep reinforcement learning policies, 2021.

Alexis Jacq, Johan Ferret, Olivier Pietquin, and Matthieu Geist. Lazy-mdps: Towards interpretable rl by learning when to act. In Proc. of AAMAS, 2022.

Zhaohui Jiang, Xuening Feng, Paul Weng, Yifei Zhu, Yan Song, Tianze Zhou, Yujing Hu, Tangjie Lv, and Changjie Fan. Reinforcement learning from imperfect corrective actions and proxy rewards. arXiv preprint arXiv:2410.05782, 2024.

Ho-Taek Joo and Kyung-Joong Kim. Visualization of deep reinforcement learning using grad-cam: how ai plays atari games? In Proc. of CoG, 2019.

Łukasz Kaiser, Mohammad Babaeizadeh, Piotr Miłos, Błażej Osiński, Roy H Campbell, Konrad Czechowski, Dumitru Erhan, Chelsea Finn, Piotr Kozakowski, Sergey Levine, et al. Model-based reinforcement learning for atari. In Proc. of ICLR, 2020.

Dmitry Kalashnikov, Alex Irpan, Peter Pastor, Julian Ibarz, Alexander Herzog, Eric Jang, Deirdre Quillen, Ethan Holly, Mrinal Kalakrishnan, Vincent Vanhoucke, et al. Scalable deep reinforcement learning for vision-based robotic manipulation. In Proc. of CoRL, 2018.

Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. In Proc. of ICML, 2017.

Zhe Li, Wei Zhao, Yige Li, and Jun Sun. Do influence functions work on large language models? arXiv preprint arXiv:2409.19998, 2024.

Amarildo Likmeta, Alberto Maria Metelli, Andrea Tirinzoni, Riccardo Giol, Marcello Restelli, and Danilo Romano. Combining reinforcement learning with rule-based controllers for transparent and general decision-making in autonomous driving. Robotics and Autonomous Systems, 131:103568, 2020.

Jieyu Lin, Kristina Dzeparoska, Sai Qian Zhang, Alberto Leon-Garcia, and Nicolas Papernot. On the robustness of cooperative multi-agent reinforcement learning. In Proc. of IEEE S&P Workshop, 2020.

Xinyu Lin, Wenjie Wang, Yongqi Li, Shuo Yang, Fuli Feng, Yinwei Wei, and Tat-Seng Chua. Data-efficient fine-tuning for llm-based recommendation. In Proc. of SIGIR, 2024.

Shicheng Liu and Minghui Zhu. Utilizing explainable reinforcement learning to improve reinforcement learning: A theoretical and systematic framework. In Proc. of ICLR, 2025.

Haozhe Liu, Mingchen Zhuge, Bing Li, Yuhui Wang, Francesco Faccio, Bernard Ghanem, and Jürgen Schmidhuber. Learning to identify critical states for reinforcement learning from videos. In Proc. of ICCV, 2023.

Xiao-Yang Liu, Ziyi Xia, Hongyang Yang, Jiechao Gao, Daochen Zha, Ming Zhu, Christina Dan Wang, Zhaoran Wang, and Jian Guo. Dynamic datasets and market environments for financial reinforcement learning. Machine Learning, 113(5):2795–2839, 2024.

Jordan K Matelsky, Lyle Ungar, and Konrad P Kording. Empirical influence functions to understand the logic of finetuning. arXiv preprint arXiv:2406.00509, 2024.

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529–533, 2015.

Volodymyr Mnih. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602, 2013.

Alexander Mott, Daniel Zoran, Mike Chrzanowski, Daan Wierstra, and Danilo Jimenez Rezende. Towards interpretable reinforcement learning using attention augmented agents. In Proc. of NeurIPS, 2019.

Dmitry Nikulin, Anastasia Ianina, Vladimir Aliev, and Sergey Nikolenko. Free-lunch saliency via attention in atari agents. In Proc. of ICCV Workshop, 2019.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. In Proc. of NeurIPS, 2022.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Proc. of NeurIPS, 2023.

Laura Ruis, Maximilian Mozes, Juhan Bae, Siddhartha Rao Kamalakara, Dwarak Talupuru, Acyr Locatelli, Robert Kirk, Tim Rocktäschel, Edward Grefenstette, and Max Bartolo. Procedural knowledge in pretraining drives reasoning in large language models. arXiv preprint arXiv:2411.12580, 2024.

Stephanie Schoch, Ritwick Mishra, and Yangfeng Ji. Data selection for fine-tuning large language models using transferred shapley values. In Proc. of ACL, 2023.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proc. of ICCV, 2017.

David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of go without human knowledge. Nature, 550(7676):354–359, 2017.

David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, et al. A general reinforcement learning algorithm that masters chess, shogi, and go through self-play. Science, 362(6419):1140–1144, 2018.

Eduardo Soares, Plamen P Angelov, Bruno Costa, Marcos P Gerardo Castro, Subramanya Nageshrao, and Dimitar Filev. Explaining deep learning models through rule-based approximation and visualization. IEEE Transactions on Fuzzy Systems, 29(8):2399–2407, 2020.

Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In Proc. of ICML, 2017.

Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. MIT press, 2018.

Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In Proc. of NeurIPS, 1999.

Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In Proc. of IROS, 2012.

Nicholay Topin, Stephanie Milani, Fei Fang, and Manuela Veloso. Iterative bounding mdps: Learning interpretable policies via non-interpretable methods. In Proc. of AAAI, 2021.

Sanne Van Waveren, Christian Pek, Jana Tumova, and Iolanda Leite. Correct me if i'm wrong: Using non-experts to repair reinforcement learning policies. In Proc. of HRI, 2022.

Oriol Vinyals, Timo Ewalds, Sergey Bartunov, Petko Georgiev, Alexander Sasha Vezhnevets, Michelle Yeo, Alireza Makhzani, Heinrich Küttler, John Agapiou, Julian Schrittwieser, et al. Starcraft ii: A new challenge for reinforcement learning. arXiv preprint arXiv:1708.04782, 2017.

Oriol Vinyals, Igor Babuschkin, Wojciech M Czarnecki, Michaël Mathieu, Andrew Dudzik, Junyoung Chung, David H Choi, Richard Powell, Timo Ewalds, Petko Georgiev, et al. Grandmaster level in starcraft ii using multi-agent reinforcement learning. Nature, 575(7782):350–354, 2019.

Ziyu Wang, Tom Schaul, Matteo Hessel, Hado Hasselt, Marc Lanctot, and Nando Freitas. Dueling network architectures for deep reinforcement learning. In Proc. of ICML, 2016.

Jingtan Wang, Xiaoqiang Lin, Rui Qiao, Chuan-Sheng Foo, and Bryan Kian Hsiang Low. Helpful or harmful data? fine-tuning-free shapley attribution for explaining language model predictions. In Proc. of ICML, 2024.

Jiongxiao Wang, Junlin Wu, Muhao Chen, Yevgeniy Vorobeychik, and Chaowei Xiao. Rlhfpoison: Reward poisoning attack for reinforcement learning with human feedback in large language models. In Proc. of ACL, 2024.

Christopher JCH Watkins and Peter Dayan. Q-learning. Machine learning, 8:279–292, 1992.

Jiahao Yu, Wenbo Guo, Qi Qin, Gang Wang, Ting Wang, and Xinyu Xing. Airs: Explanation for deep reinforcement learning based security applications. In Proc. of USENIX Security, 2023.

Zhuowen Yuan, Wenbo Guo, Jinyuan Jia, Bo Li, and Dawn Song. SHINE: Shielding backdoors in deep reinforcement learning. In Proc. of ICML, 2024.

Tom Zahavy, Nir Ben-Zrihem, and Shie Mannor. Graying the black box: Understanding dqns. In Proc. of ICML, 2016.

---

## BibTeX Citation

```bibtex
@article{Cheng2025XRLSurvey,
  author        = {Cheng, Zelei and Yu, Jiahao and Xing, Xinyu},
  title         = {A Survey on Explainable Deep Reinforcement Learning},
  year          = {2025},
  month         = {February},
  eprint        = {2502.06869},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2502.06869},
  note          = {arXiv preprint arXiv:2502.06869v1}
}
```
