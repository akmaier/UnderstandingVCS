A Survey of Explainable Reinforcement Learning: Targets, Methods and Needs

Léo Saulières

IMT Mines Albi, Albi, France

Corresponding author email: saulieres.leo@gmail.com

arXiv:2507.12599v1 [cs.AI] 16 Jul 2025

## Abstract

The success of recent Artificial Intelligence (AI) models has been accompanied by the opacity of their internal mechanisms, due notably to the use of deep neural networks. In order to understand these internal mechanisms and explain the output of these AI models, a set of methods have been proposed, grouped under the domain of eXplainable AI (XAI). This paper focuses on a sub-domain of XAI, called eXplainable Reinforcement Learning (XRL), which aims to explain the actions of an agent that has learned by reinforcement learning. We propose an intuitive taxonomy based on two questions "What" and "How". The first question focuses on the target that the method explains, while the second relates to the way the explanation is provided. We use this taxonomy to provide a state-of-the-art review of over 250 papers. In addition, we present a set of domains close to XRL, which we believe should get attention from the community. Finally, we identify some needs for the field of XRL.

## 1 Introduction

This paper presents a state of the art on explainable and transparent reinforcement learning. It brings together a set of relatively recent works that present new methods specific to Reinforcement Learning (RL) as well as papers using methods derived from the eXplainable Artificial Intelligence (XAI) domain, originally used to explain classifiers, such as LIME [285] or SHAP [234].

RL is a machine learning paradigm where an agent learns to make a sequence of actions within an environment. Given a set of information defined as a state, the agent chooses an action at each time-step, arrives in a new state and receives a reward, determined by the environment's dynamics (transition and reward functions). The agent's goal is to maximize its cumulative reward (also called return) by learning an optimal policy. An RL problem is described by a Markov Decision Process (MDP), which is a tuple $\langle \mathcal{S}, \mathcal{A}, R, p \rangle$, where $\mathcal{S}$ and $\mathcal{A}$ are respectively the state space and action space, $R : \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ is the reward function and $p : \mathcal{S} \times \mathcal{A} \to Pr(\mathcal{S})$ the transition function.

In addition to the eXplainable RL (XRL) methods, we briefly present a set of sub-domains of RL whose main motivation is the performance and generalisation of agent policies. Indeed, they can also be used to explain or make transparent the agent's behavior. For example, Relational RL (RRL) [97] consists of the agent reasoning on the basis of *relations* and *objects* rather than reasoning directly with raw data. The relationships on which the agent relies can then be used to explain its behavior.

This overview is based on a total of 12 states of the art [204, 153, 247, 125, 121, 381, 155, 369, 78, 402, 279, 8] and complementary papers. Among the states of the art, [125] focuses on the interpretability of RL and [121] on counterfactual explanations whereas the others are not specific to one type of explanation. A counterfactual explanation is used to determine which part of the input (e.g. a state in RL) must be modified in order to change the model's output (e.g. the agent's policy). It is interesting to note that this type of method is under-represented in XRL [121].

Our paper has four objectives:

- To reflect the methods designed to explain/make transparent agents that have learned by reinforcement.
- To present works that use model-agnostic methods.
- To briefly describe domains related to XRL on which researchers should focus.
- To draw a list of the needs for the XRL domain.

In this state of the art, we present methods aimed at *providing an explanation* as well as those aimed at *making the agent's behavior transparent* by design. We believe it is important to cover both approaches in order to give the reader an overview of the methods used to make the agent's behavior understandable to the user. By abuse of language, we refer to both approaches when talking about explanations or XRL in this paper.

Among researchers in the field of XRL, there is no consensus on the taxonomy to be used to classify the various works. For example, Puiutta and Veith [279] use the taxonomy of classifiers, distinguishing explanations by their scope and the time at which they are extracted. In terms of scope, a *local* explanation is an agent's choice of action, whereas a *global* explanation describes the agent's policy as a whole. In terms of time, an *intrinsic* explanation means that the model can be interpreted by itself, whereas a *post-hoc* explanation means that the explanation is provided after the agent's learning phase. Milani *et al.* [247] separate the methods using categories that better reflect the aspects of RL. Works are separated into three categories: *feature importance*, which consists of identifying the features of a state that the agent's decision, *learning process and MDP*, which shows the past experiences or the components of the MDP that lead to the agent's behavior and *policy level*, which describes the agent's long-term behavior. As a final example, Dazeley *et al.* [78] use the Causal Explanation Framework [46] to propose a conceptual framework for XRL, called the Causal XRL Framework. A simplified version is used to categorise the methods into two types: *perception* and *action*. The first category includes methods that focus on the impact of the agent's perception on its actions and outcomes. The second category includes methods that focus on the choice of action and its impact on the outcome.

In the absence of a standard taxonomy, we propose in this paper a new taxonomy based simply on 2 questions: *What* and *How*. More specifically, the first question is used to determine what the method is trying to explain: "What does the method want to explain? What does it want to make transparent?". We have identified three elements: the agent's *policy*, a particular *sequence* of interactions between the agent and the environment, and a particular *action* by the agent in a given situation. The second question allows us to refine the taxonomy, by looking at the way in which the explanation is provided: "How is it explained?". Thus, a set of ways of providing explanations are described in this state of the art, for each of the three above-mentioned elements. An overview of the taxonomy is presented in Figure 1, where an illustrative work is associated with each sub-category.

We believe that this taxonomy can help readers to quickly identify a body of work that is relevant to the question they want to answer (and also the way in which they want to answer it). This state of the art also makes it possible to determine trends, whether in terms of elements explained or approaches used.

The rest of the paper is structured as follows. Section 2 presents the methods explaining the agent's policy. Section 3 describes the methods explaining the agent's behavior in a sequence of interactions with the environment. Section 4 presents the methods explaining an agent's choice of action from a state. Section 5 briefly describes domains related to XRL that we think researchers should pay attention to. Section 6 highlights the needs for the XRL domain according to the different surveys we studied.

**Figure 1.** Our taxonomy for Explainable Reinforcement Learning. The figure shows three vertical columns labeled POLICY, SEQUENCE, and ACTION corresponding to Sections 2, 3, and 4. The POLICY column lists Interpretable Policy (Bastani et al. [34]: Learn a surrogate model which takes the form of a Decision Tree), Policy Summary (Amir and Amir [12]: Collect a set of important trajectories as a summary of the agent's behavior), Human-readable MDP (Hayes and Shah [144]: Answer to user question by finding then compacting a state space region into a minimal boolean logic expression), and Visual Analysis (Mishra et al. [250]: Provide several tools to explain the agent's behaviour through a user interface). The SEQUENCE column lists Counterfactual Sequence (Van der Waa et al. [357]: Contrast between consequences of user's query-derived policy and agent's policy), Important Elements (Guo et al. [138]: Identify critical time-steps for the agent's final reward), and Human-readable MDP (Li et al. [216]: Actions correspond to image transformation which are interpretable by design). The ACTION column lists Feature Importance (Greydanus et al. [131]: Display perturbation-based saliency maps for Atari agents) and Expected Outcomes (Juozapaitis et al. [187]: Determine which goal the agent is trying to achieve using reward decomposition).

## 2 Policy-level methods

The methods described in this section explain the agent's policy. A total of four ways of explaining the policy have been identified. Constructing *Interpretable Policies* makes the agent's reasoning transparent. *Summarising Policies* provides an overview of the agent's capabilities. The use of *Human-readable MDP* allows the user to understand the information available to the agent, its actions and the dynamics of the environment. *Visual analysis* helps the user to study the agent's policy visually, using various tools and metrics. The detailed taxonomy of policy-level methods is described in Figure 2.

**Figure 2.** Detailed taxonomy for policy-level methods. The figure is a tree diagram rooted at POLICY. The Interpretable Policy branch splits into Surrogate Model (Decision Tree, Graph, Program, Rules, Equation) and Inherently Understandable (Hierarchical Policy, Decision Tree, Rules, Equation, Program, Graph). The Policy Summary branch splits into Sequences, Critical States, SHAP, and Policy Comparison. The Human-readable MDP branch splits into Surrogate Model (States Clustering, State Transformation) and Inherently Understandable (MDP Representation, Relational RL and MDP). The Visual Analysis branch splits into Understanding DQNs, Visual Toolkit, and Inspection.

### 2.1 Interpretable Policy

A way of making the agent's behavior understandable to the user is to construct interpretable policies, thereby making the agent's decision-making transparent. To do this, a surrogate model is learned to understand the agent's opaque decision-making via this model, or the agent's policy is directly learned in such a way as to be interpretable. To clarify, although these methods enable us to explain the agent's decision making (i.e. its choice of action in a particular situation), we choose to classify this type of approach as policy explanation, because the main result of these methods is an interpretable form of the policy, and the action explanation is simply derived from it.

#### 2.1.1 Surrogate Model

A surrogate model is an interpretable substitute for an agent's opaque policy. It is used to understand the agent's behavior. This section organises the works according to the form taken by the surrogate model.

**Decision Tree** The following works propose a surrogate model in the form of a Decision Tree (DT).

A Deep RL (DRL) method is a method that combines RL and Neural Networks (NN) to obtain an agent's policy. A DRL agent is one that has learned using a DRL method. To extract an interpretable policy from a DRL agent, Dai *et al.* [71] propose an *information gain rate weighted oblique DT* (IGR-WODT). Based on three defined metrics, IGR-WODT provides a policy that requires a much smaller number of parameters (only 1.1% of the number of DRL model parameters). An *oblique DT* is composed of internal nodes that separate data based on linear combinations of features (i.e. of the form $w_1 f_1 + w_2 f_2 + ... + w_n f_n > 0$ where $w_i$ are weights, $f_i$ are state features and $n$ is the number of features). The WODT [389], on which IGR-WODT is based, uses a logistic regression model to determine the probability of an instance belonging to the left or right child node. The extension provided by IGR-WODT consists of using a different objective function inspired by the information gain [281].

In the same vein, a *soft DT* (SDT) is used to approximate the policy of a DRL agent trained on the Mario AI benchmark [191] in [65]. An SDT [112] has a pre-defined depth, where each internal node corresponds to a weighted vector to which a bias is added (i.e. perceptron form) and the leaves correspond to a softmax distribution (which in this case represents the actions distribution). The advantage of this approach is that the weights learned from each perceptron can be displayed as heatmaps of the agent's state (which takes the form of an image). In this way, the user can successively visualise the pixels that do or do not have an impact on the agent's decision-making. A compromise between interpretability and performance is observed: as the depth of the tree increases, the SDT performance gets closer to the performance of the DRL agent, at the expense of interpretability.

The same approach is proposed in [137] for an aircraft separation problem. A visualisation module allows the user to see both the tree plot, where each node is represented by a heatmap, and the trajectory plot, which provides the features that have the greatest impact on the agent's choice of action.

In order to outperform the SDT-based approaches, Ding *et al.* [86] present a *cascading DT*, an architecture comprising two kinds of trees: a forest of learning trees F used to provide a compact representation of features, and a decision-making tree D which uses the features learned by F to learn a policy. This architecture reduces the number of parameters and improves performance compared with other SDT-based methods, e.g. [65].

Another type of DT, the *non linear DT* (NLDT), initially proposed for classification problems [83], allows a DRL agent to be represented by a more interpretable model [84]. Such a tree has internal nodes that take the form of conditions on non-linear functions. Based on a dataset of DRL agent interactions within the environment, the NLDT is learned using an evolutionary approach. This tree is then pruned to give an interpretable NLDT and re-optimised for performance.

With a concern for reliability, the VIPER algorithm is used to learn a "robust" DT that has the same performance as the DRL agent [34]. Given a set of state-action situations, imitation learning (IL) [140] aims to learn a policy that mimics the given decisions and generalizes over unseen situations. Based on IL and model compression methods, VIPER produces a DT used to verify the policy according to certain properties such as *stability*, *robustness* and *correctness*. Stability seeks to determine whether the agent asymptotically achieves its objective. The robustness degree of a policy is defined by comparing the action returned for states close to a state $s$ with the one returned for $s$. Correctness varies according to the environment: for the Atari Pong game, the objective is to prove that the policy never loses, and for CartPole, the objective is to prove that the pole never falls below a certain height. An extension of this work focuses on multi-agent settings. IVIPER [248] produces a good quality DT for each agent, but does not take into account the coordinated behavior of the agents. To overcome this, MAVIPER [248] was introduced. Experiments have shown that MAVIPER generates better and more robust policies than IVIPER and the other baselines tested.

Also, in a sort of multi-agent view, the MoËT method [361] is based on *mixture of experts* [175]. It consists of a set of expert DT's and a gating function that uses weights to define the extent to which an expert's decision is referred to in the agent's choice of action. The DT's are learned by imitating a DRL agent. The softmax function is used as MoËT's gating function. A variant called MoËT$_h$ consists of selecting only one DT using the gating function. This variant is used to transform the learned policy into a Satisfiability Modulo Theories (SMT) formula and thus verify properties using the SMT solver Z3 [79]. MoËT$_h$ outperforms VIPER [34] in terms of policy fidelity and reward obtained.

The method proposed by Sieusahai *et al.* [314] involves learning a DT based not on the image but on a set of *sprites* associated with their position in the image. A sprite is an image representing a character or object within a game. To learn the DT, a function transformation is used to extract a set of sprites with their coordinates from the image representing the state of the DRL agent. The DT learns to approach the DRL agent based on this more compact representation of the agent's state, which makes the internal nodes of the tree interpretable.

In [60], DT's and Random Forests (RF) are learned in The Open Racing Car Simulator (TORCS) environment based on IL. The continuous actions are vectors composed of the steering, acceleration, brake and gear parameters in TORCS. RF generalise better and perform better, at the expense of interpretability, than DT's. In order to learn such models, a number of modifications had to be made to the problem: the decomposition of the multidimensional actions problem into a set of one-dimensional actions and the discretization of the continuous actions.

A more user friendly approach uses a tree of objects [227]. The RAMi framework is used to learn a tree that mimics a DRL agent based on a representation of states by objects, where an object is a sub-part of an image. This representation is learned using the Identifiable Multi-Object Network and then, the tree is learned using a Monte Carlo Regression Tree Search algorithm. It has a good compromise between fidelity and simplicity in terms of the number of nodes in it. In addition, a method is proposed for calculating the importance of features and obtaining causal relationships from the mimic tree.

In the similar aim to produce interpretable nodes, Acharya *et al.* [2] propose an algorithm to generate a set of strategy labels and interpretable experiential features representations, i.e. predicates, from a set of sequences of interactions of the DRL agent within the environment. Using these elements, a DT is constructed where an internal node uses a generated predicate. The tree is used to represent the agent's different strategies. The condition for following a certain strategy is thus represented by a branch (which is a conjunction of interpretable features).

The algorithm based on IL and program induction proposed in [316] is used to cluster agent interaction data into a set of rules expressed by a set of predicates. The output of the algorithm returns a set of DT's, which are displayed to the user in the form of flowcharts. These flowcharts represent the predicates of the internal nodes of the DT's in the form of natural language questions.

Linear Model Trees (LMT) are used to approximate the policy directly. The leaves are linear models representing an action to be taken. LMT's have been compared in [51, 231] with the model-agnostic methods SHAP [234] and LIME [285]. The main advantage of LMT's is the lower computational cost, as the model is learned in advance. To explain an action choice, LMT's allow the influence of features to be extracted directly, whereas other approaches require the influences to be calculated. Also, representing the policy by an LMT makes it directly interpretable, to the extent of the size of the tree. As with most other work, a trade-off between simplicity and fidelity is necessary.

For a state $s$, the value function represents the expected return starting from $s$. For a state-action pair $(s, a)$, the Q function represents the quality, or expected return, of executing $a$ from $s$. Q-learning aims to learn the optimal Q-function by iteratively updating the Q-values based on the Bellman equation. Deep-Q-Network (DQN) [253] is a method that learns to approximate Q-values using NNs. Liu *et al.* [226] introduce Linear Model U-trees (LMUTs) to approximate the Q-values of a DQN agent. This method differs from the other approaches previously presented in this section, where the aim is not to mimic the policy directly. LMUTs is an extension of CUTs [354], which are regression trees for value functions. Compared to CUTS, LMUTs has leaf nodes that contain linear models of state features. The model is interpreted in three ways: by identifying the importance of the splitting features in the LMUT, by extracting the rule that led to the Q-values obtained and by displaying the pixels that have an above-average influence on the Q-values.

With the same idea of approximating Q-values, Jhunjhunwala [181] proposes a new structure, called Q-BSP Tree, which models the Q-value in an interpretable way and an Ordered Sequential Monte Carlo algorithm which learns to approximate the Q-value of the DQN. For a problem comprising $n$ actions, the Q-BSP forest is composed of $n$ Q-BSP trees, each learning the Q-value of one action. This approach outperforms the baselines in terms of performance and fidelity with respect to the DQN agent policy, including LMUT [226].

**Graph** The surrogate models described in this section are an abstract policy represented by a graph (which is not a decision tree). All the approaches below are attempting to build a graph that zooms out the agent's policy.

Topin and Veloso [346] present *abstracted policy graphs* (APG), which are Markov chains of abstract states. In an APG, edges are transitions labelled by their probability of occurring, nodes are abstract states labelled by the agent action performed from that state. An abstract state represents a grouping of similar states that is constructed with respect to the actions and feature importance of the states, calculated via FIRM [416]. The transition probability from node $S$ to node $S'$ is an average of the transition probabilities on the states represented by $S$ and $S'$. To build the APG, a policy, a value function and a set of data on the agent's interactions within the environment are required. An example of APG is described in Figure 3.

**Figure 3.** Example of Abstract Policy Graph [346]. Each vertex is an abstract state with an associated action, and the edges between two vertices represent transitions (weighted by their probability of occurring). The figure shows a directed graph of labeled nodes (b1-b6, l4-l10) with transition probabilities such as 0.2, 0.8, 0.1, 0.9, 0.9, and self-loops with probability 1.0 connecting them in a Markov chain layout.

In the same vein, McCalmone *et al.* [244] propose CAPS, a method that constructs a directed graph composed of abstract states. The state space is abstracted using a decision tree-based clustering algorithm called CLTree [225]. This algorithm returns a hierarchy of cluster configurations, the best of which is extracted by taking into account the interpretability of the clusters and the accuracy of the state transitions. To improve user comprehension, the authors use an approach to highlight the most important abstract states according to the agent's stochastic policy [161] and a method to describe each abstract state by a natural language sentence.

Based on a set of agent's interactions within the environment, the Policy Graph proposed in [91] is a directed graph that represents the policy with states expressed via predicates (a state is a formula in propositional logic). These handcrafted predicates are defined upstream according to the environment. In the graph, the nodes represent the states, the edges represent the transitions with their probability and the agent's action. With this graph, the authors propose to answer three questions (inspired by [144]) using custom algorithms: 'What will you do when you are in state region $X$?', 'When do you perform action $a$?' and 'Why did you not perform action $a$ in state $s$?' This approach was tested on a single-agent and a multi-agent environment.

MARLeME [196] is a library that fits into a Multi-Agent RL (MARL) setting to create different directed graphs from agent policies, each representing a Value-based Argumentation Framework (VAF). A VAF is used to model a set of action arguments and the attack relations between them. These arguments are valued in such a way as to order the arguments according to their usefulness. MARLeME can be used to represent an agent, or a group of agents, in the form of a VAF. To use this library, it is necessary to enter the arguments, their valuations and the attack relations.

In order to verify safety properties of a DRL agent, Vinzent *et al.* [367] construct an abstract graph based on predicates. Abstraction is only performed for observable states, taking into account the agent's policy and the predicates. Abstract states are constructed on the basis of satisfying the set of predicates (so that states satisfying the same predicates belong to the same abstract state). An SMT solver, Z3 [79], is used to define the transitions in the graph.

The following work focuses on transforming a policy that takes the form of a Recurrent NN (RNN) into a Moore Machine (MM) [254], which is a finite state machine whose output depends only on the current state and where each state is labelled by an output value (an action in this context). To do this, Kou *et al.* [203] first propose an auto-encoder called Quantized Bottleneck Network (QBN), which learns to quantize the agent's observations and the RNN's memory states for a MM representation. Next, the QBN is combined with the RNN to create a Moore Machine Network (MMN). Finally, a MM is extracted from the MMN and minimised to be interpretable. In the MM, the nodes represent the states of the memory, and the edges represent the agent's observations. Each node is labelled with the action to be performed from a memory state. In some environments, it has been observed that the number of memory states and observations to represent the agent's policy is small. An extension of this work [72] criticises the interpretability of the MM provided and proposes four ways of reducing the size of the MM in an interpretable way. Indeed, in [203] the minimisation techniques used do not take into account the user's comprehension, for example certain states judged to be different for a user can be merged into a single state. The proposed minimisation methods improve the visualisation of the MM, while preserving the key decision points of the agent's behavior. A tool based on the saliency map principle (presented later in this survey) is proposed as a complement to understand the differences in the agent's choice of action between 2 observations.

Based on a set of sequences of interactions of the agent within the environment, Hüyük *et al.* [167] present a Bayesian algorithm for learning decision dynamics and decision boundaries. Decision dynamics aggregate sequences as beliefs, which correspond to the probability that a state $s$ exists at a time $t$. Decision boundaries represent policy by partitioning decision dynamics into regions where the same action is performed.

The quasi-symbolic (QS) agent [211] is composed of two interpretable networks made up of a single layer: the matching network, which memorises transitions, and the value network, which evaluates them. This agent uses the RL and the environment function to predict a plan of actions that leads to one of the most valuable transitions (in terms of reward). The structure of the QS agent makes its behavior interpretable.

**Program** The agent's policy is substituted by an interpretable program in the following works.

In order to design interpretable and verifiable policies, the *programmatically interpretable reinforcement learning* (PIRL) framework is proposed in [365]. Policies are generated using a domain-specific programming language. To compensate for the size of the policy search-space, the specification of the policy form is used in PIRL through a sketch, which is a restriction of the grammar proposed by the language. The Neurally Directed Program Sketch (NDPS) algorithm is used to locally search for a policy that takes the form of a program based on the NN agent policy, a Partially Observable MDP (POMDP) and a sketch. POMDP [25] is a generalisation of MDP where the agent does not have full visibility of the state it is in. NDPS returns a policy close to that of the DRL agent and which obtains the highest expected aggregated reward among the programs evaluated during the search.

In the same vein, Zhu *et al.* [415] synthesise a program close to the DRL agent policy which avoids reaching unsafe states of the environment. To this end, the *counterexample-guided inductive synthesis* (CEGIS) framework is presented to search for a program parameterised by a sketch (i.e. a high-level description of the sought program). In this approach, a counterfactual example is an initial state in which the program under study has not yet been proven safe. In addition to the interpretability aspect, the program produced is used as a safety shield: it takes the relay when the agent chooses an action that may lead to an unsafe region.

Based on a DRL agent policy, INTERPRETER [201] is used to obtain an interpretable and editable python program. To do this, a regularized oblique tree [257] is learned from the DRL agent policy via IL, then converted into a python program, notably by translating the nodes of the tree using *if-else* statements.

For robot manipulation tasks, a program is extracted from a set of demonstrations in [49]. To do this, a probabilistic generative task model is used to infer from the demonstrations a sequence expressed by interpretable low-level motion primitives. Program induction is then used on this sequence to simplify it into a program.

The PROPEL approach [364] makes it possible to discover policies that are interpretable, verifiable and generalisable. The idea is to iteratively search for a policy represented by a NN using policy-gradient methods and to project the policy into the space of programmatic policies (i.e. policies representable by programs). Two PROPELs searching for policies in different programmatic policy classes have been proposed: PROPELPROG, which uses a domain-specific language, like NDPS [365] and PROPELTREE, which uses tree regression like VIPER [34].

For a detailed overview of program synthesis for interpreting DRL agents, we recommend [33] to the interested reader.

**Rules** The following works present a set of if-then rules that act as a surrogate model.

In [259], a set of fuzzy rules is learned to approximate an agent's policy. Based on a dataset representing the agent's interactions with the environment, the rules are learned using the Evolving Takagi-Sugeno method [21]. To obtain these rules, it is necessary to learn the rule antecedents, which decompose the state space into relevant regions, and to estimate the feature coefficients of the rule consequents, which are linear model of each rule used for determining the action to be taken. This approach allows the policy to be approximated by a set of conditioned interpretable linear models.

Soares *et al.* [319] propose prototype-based fuzzy rules which are then visualised by displaying either features or prototypes. To reduce the number of fuzzy rules generated, the authors propose a hierarchical mechanism to group adjacent prototypes in the data space that have the same consequent (i.e. action).

**Equation** The substitute policy takes the form of equations in the following two works. It is in a context where the agent's action is determined as a function of continuous features of the agent's state. Genetic algorithms are used to find such policies.

Zhang *et al.* [405] use the evolution feature synthesis [24] to generate control policies that mimic the DRL agent policy. A generated policy is a combination of operations (defined on a set of operators O) using reals and features of a state. In order to make the policy interpretable, a method based on the complexity-performance trade-off is proposed.

In the same line, the method presented in [205] is a variant of Single Node Genetic Programming [174], which represents the individuals in a population by a node in a graph. For a robotic task, this approach approximates a policy that produces smooth control by an equation, based on a set of interactions of the agent within the environment.

#### 2.1.2 Inherently Understandable

**Hierarchical policy.** Hierarchical RL (HRL) is a sub-domain of RL consisting in learning a policy which is structured in a hierarchical manner, generally comprising 2 levels, so as to break down the task to be performed into a set of sub-tasks. For each sub-task, a *low-level* policy is learned. A *high-level* policy is learned using the various *low-level* policies as an action. This approach was not originally designed to explain a policy, but to improve the performance and generalisation of policies on tasks that are decomposable. Therefore, in this section we do not intend to provide an overview of HRL, but rather to present work that has been designed for explainability or that seems relevant to this purpose.

The low-level policies are represented using a human language description in the following works. For Minecraft games, agents learn to determine when it is necessary to learn a new policy [313]. If not, the agent chooses one of the policies it has already learned. These policies are interpretable due to their description, such as '*Get object o*'. To learn the dependencies between different tasks/policies (for example, it is necessary to use '*Get object o*' before '*Put object o*'), a stochastic temporal grammar model is learned. In the same vein, Jiang *et al.* [182] present Hierarchical Abstraction with Language using high-level actions, sub-goals, represented by a human language description. The high-level policy and the low-level policy are trained separately, where the low-level policy is learned based on a reward function that depends on an (assumed known) function determining whether a sub-goal is respected or not. The results show that this framework produces interpretable policies that perform well on long-horizon problems with sparse reward.

Model-based RL methods use the dynamics of the environment (i.e. the transition function and the reward function) to obtain an agent's policy. Note that the dynamics are not necessarily known beforehand. In a model-based HRL context, Xu and Fekri [386] propose to learn a transition function for high-level symbolic states (i.e. sub-goals) by modifying DILP [106] so as not to generate meaningless clauses. The transitions generated are logic rules expressing the preconditions and effects of the various sub-goals. In the same context, the Model Primitives HRL framework [384] allows a set of modular interpretable policies to be learned and combined based on a set of imperfect models of the environment. The idea is that approximate models of the environment are specialised in different regions of the environment. In addition to the policies, a gating controller is learned to compose the policies and solve the basic task.

Three works propose to mix Planning and HRL. The Symbolic DRL framework [236] combines a high-level planner that returns a plan composed of symbolic actions with sub-policies learned by DRL for each symbolic action. The framework is made up of three elements: a planner that builds a plan of symbolic actions, a controller that learns a sub-policy and a meta-controller that obtains the quality of the proposed plan and thus helps the planner to propose a better plan. An intrinsic and environment reward are respectively used to learn the sub-goals, i.e. symbolic actions, and to measure the plan quality. The combination of Planning and HRL is also proposed in [55] for a human-UAVs cooperation problem. The plan represented by a tree and the sequential order of the actions to be performed is displayed to the user in such a way as to make the plan interpretable. Jin *et al.* [184] use the option framework [336]. An option consists of pre-conditions, a policy and effects. Option policies are learned in RL and a planner is used to generate a plan. To use this approach, the user requires access to a function defined upstream which extracts a symbolic state from an image state.

For a robotic manipulation task, the Dot-to-Dot method [43] uses HRL to decompose the problem into simpler tasks, assimilated to sub-goals, which the low-level agent learns to perform. The sub-goals are iteratively generated by the high-level agent to guide the low-level agent. The low-level agent's input is its observation and the sub-goal to be achieved. This approach makes it possible to interpret the impact of the sub-goals on the agent's learned Q-values, and to observe that the high-level agent has learned a notion of distance for the given problem.

In [359], a reward function is based on the achievement of a sub-goal for learning. In addition, a curiosity reward is generated using a Generative Adversarial Network (GAN) [128] that generates the state resulting from an action $a$ from a state $s$. The sub-goal is generated by a neural network that selects a feature of the agent's current state as the sub-goal to be achieved. This method is specific to a problem where the state represents a grid in which the agent can move. The sub-goal is transcribed manually using a natural language description. In this work, a single policy is learned and is not technically an HRL method, although the task is decomposed into sub-objectives.

Rietz *et al.* [286] combine HRL with reward decomposition to provide explanations of the agent's decisions. A context is associated with the reward decomposition, providing the user with the sub-goal that the agent is currently aiming to achieve. This method explains the policy with the different sub-goals but also the agent's local decision by providing a reward decomposition.

Using policy sketches, the agent learns to solve multi-task RL problems [19]. These policy sketches are sequences of symbolic labels. Each label is a comprehensible task that is learned by a sub-policy in the form of a neural network.

*Composition* [343], a slightly different approach to HRL, has been used in [340]. It consists of learning a new skill based on old skills that have already been learned. The interpretable aspect presented in this paper is that the new skill is composed without new learning, simply by using a boolean algebra on the tasks (modelled by MDP's) and on the value functions already learned.

In this paragraph, we give a brief presentation of some HRL methods or methods close to the domain which are not focused on explainability, but which are nonetheless worth mentioning. Two works [212, 327] use Answer Set Programming [220] to constrain the agent's behavior [212] or define a high-level plan made up of abstract actions [327]. Furelos-Blanco *et al.* [117] mix Inductive Logic Programming (ILP) and RL by learning an automaton composed of edges which represent a sub-goal by a logic formula. The framework proposed in [391] combines planning and RL to provide robust symbolic plans. In [404], the agent learns a graph, the environment model based on a high-level state representation and uses it to learn a policy. In the same vein, Eysenbach *et al.* [107] build a graph from the sub-goals obtained using the replay buffer and then use Djikstra's algorithm [85] to plan. HRL is combined with the RRL in [95] to learn an efficient policy where each sub-policy is learned by a Q-tree. The hierarchical DQN [206] represents a set of Q-functions in a hierarchical manner, so that the low-level ones learn a sub-policy for a given objective and the high-level ones solve the base problem.

**Rules** The following works represent the policy of an agent by a set of rules.

A model-based approach is presented in [148], entitled Fuzzy Particle Swarm RL. The aim is to learn a set of fuzzy rules offline with particle swarm optimization [197], using NN's to represent world models. The produced rules take the form: '*if s is m then o*' where $s$ is a state, $m$ a Gaussian membership function and $o$ a real number. The number of rules to be produced must be provided prior to learning. Note that only 2 rules are needed to learn a good policy for the Mountain Car and CartPole problem. The rules are visualised by presenting the membership functions and an example state. The learning of world models has an impact on the production of efficient policies: models that do not approximate the real dynamics correctly can cause the agent to learn the wrong policy.

In the same vein, Huang *et al.* [160] propose the interpretable fuzzy RL method to produce an interpretable policy in the form of fuzzy rules. This method is model-free and uses an actor-critic structure to learn the rules. To use this approach, the state-action space is discretised.

As a preliminary work [7], the interpretable policy takes the form of a succession of blocks '*if close(s, center[k]) do action[k]*' where the state space is split into clusters with associated actions. The *close* function, which is assumed to be known, determines whether a state $s$ belongs to a cluster. The number of clusters is limited upstream and the cluster centres are not optimised. Learning a policy is based on the approximate policy iteration framework [39].

With a state represented by an image, the architecture described in [268] breaks down into three parts: a Compact Convolutional Transformer (CCT) [143] extractor which uses an attention mechanism to extract relevant features, a fuzzy-based decision network which learns fuzzy if-then rules from the output of the CCT, and a Convolutional NN (CNN) latent decoder which reconstructs the agent's state. The policy is analysed by visualising the attention map extracted from the CCT and the most influential rules in the agent's choice of action.

In the following works, the rules are expressed in first order logic, each taking the form of a clause $a \leftarrow a_1, a_2, \ldots a_n$ where $a$ is the head atom (i.e. the action to be performed) and $a_1, a_2, \ldots a_n$ the body atoms (i.e. the preconditions to be met). Most of the works [183, 407, 145, 274] is based on DILP [106] to produce interpretable policies. This neuro-symbolic approach makes it possible to learn a set of symbolic rules weighted by a confidence score, while still having the advantages of using NN's.

Jiang and Luo [183] present Neural Logic RL (NLRL) for learning a policy in the form of first-order logic rules. For this purpose, a DILP architecture is proposed as well as an MDP with logic interpretation. For the experiments, the authors provided only minimal atoms as input to describe the agent's states and background. The NLRL policies are interpretable and achieve near-optimal performance on the problems tested.

Also based on DILP, Zhang *et al.* [407] propose an off-policy algorithm for learning a set of rules. This method is designed to use approximate inference to reduce the number of rules and has the same or better performance than [183].

Deep Explainable Relational RL [145] represents an interpretable policy by a set of if-then rules. The objective is to find a policy such that given a state and background knowledge, an action is predicted. An NN is learned to produce rules and semantic constraints are proposed to avoid redundancy between rules. The approach outperforms NLRL [183] in terms of generalisation and computation time.

In [274], the approach is a joint use of RRL [97] and DILP. RRL is used to convert the state of an agent in image form into a set of auxiliary predicates. To do this, a CNN is used to take a low-level representation as input and convert it, using auxiliary predicates, into a high-level representation. The dNL-ILP engine [273] is then used to express the policy in the form of rules.

In comparison with DILP-based methods, Neural Symbolic RL [237] avoids storing all the rules by using an attention module. This framework is divided into three parts: the attention module, the reasoning module and the policy module. This approach provides near-optimal policies composed of chain-like rules. However, the rules generated are not always true and must be selected by an expert.

Neural Logic Machines (NLM) [92] is a neuro-symbolic architecture for learning the underlying logical rules of a problem. To do this, starting from a set of basic predicates lying upstream of the architecture, a succession of first-order rules are applied using Multi-Layer Perceptron's (MLP), to obtain a set of conclusions about the objects. NLM is able to solve simple reinforcement learning tasks. The problem with this approach is that the rules learned are not interpretable.

Based on the NLM architecture [92], Zimmer *et al.* [417] introduce the Differentiable Logic Machine (DLM), a high-performance neuro-logic architecture which outperforms dNL-ILP [273] and NLM in terms of computation time and memory used. Two operations on predicates are added compared to NLM, namely negation and preservation, and MLP's are replaced by logic modules, which correspond to fuzzy AND and OR operations on predicates, to improve the interpretability of the model. A post-process approach is used to extract logical formulas from the model, making its reasoning interpretable. DLM achieves good performance (similar to or better than NLM) while extracting an interpretable set of rules.

**Decision Tree** The interpretable policy learnt in the following works takes the form of a DT.

Conservative Q improvement [290] is an algorithm that manages the trade-off between interpretability and performance during learning. The policy takes the form of a DT where the internal nodes are conditions based on features of a state and the leaves contain the Q-values of possible actions. Using a dynamic threshold, the tree is extended only if this extension results in a sufficient expected discounted future reward.

Differentiable DT (DDT) is used in [315]. The DDT can be used for Q-learning, where each leaf contains the Q-values of the actions, or for a gradient-based approach, where each leaf contains the distribution of the actions. The tree is then discretised to obtain an interpretable DT (an example for the CartPole environment [48] is shown in Figure 4). In addition, a decision list can be extracted. The results show that these trees have a good performance compared to a MLP, while being more interpretable, according to the user study carried out.

**Figure 4.** A discrete decision tree for the CartPole environment [315]. Each internal node of the tree (in blue) represents a condition on a feature of the agent's state, and each leaf of the tree (in green) corresponds to an action. The figure shows a binary tree with root condition "Pole Velocity > 0.44", branching into "Pole Velocity > -0.3" and "Pole Angle > 0.01", with deeper nodes testing "Pole Angle > -0.41" and "Pole Angle > 0.0", and leaves labeled Right, Left, Left, Right, Right, and Left.

Topin *et al.* [345] propose a specific representation of an MDP, called Iterative Bounding MDP (IBMDP), to provide a policy in the form of DT's for the original MDP. This policy is learned using modified versions of either policy-gradient or Q-learning algorithms. The aim is to create a policy that also works for the base MDP. The CUSTARD algorithm presented for solving IMDPs obtains better results than VIPER [34] concerning tree size and policy performance.

An evolutionary approach coupled with Q-learning is presented in [69]. The evolutionary approach, inspired by the Grammatical Evolution algorithm [294], is used to find the tree structure and Q-learning to determine the action to be taken for each leaf of the tree. Once the policy has been obtained, a test phase is carried out on 100 episodes to compress the tree. Nodes are not visited are deleted, as well as internal nodes whose leaves lead to the same action.

Paleja *et al.* [269] present the Interpretable Continuous Control Trees (ICCTs) which represent the policy in a sparse way. The interpretable representation consists of internal nodes whose condition is set only by one feature and leaves which are sparse linear controllers (i.e. based on few features). On a set of control tasks tested, ICCTs achieved a good compromise between performance and interpretability.

The Policy Tree algorithm [139] represents the policy as a tree. This method is applied with agent states described by binary features and the policy is learned by gradient-based methods. The tree returned by the algorithm is a binary DT where each internal node corresponds to a feature and a leaf to a distribution of actions. Starting from any policy and a binary DT containing only one node, two steps are repeated: Parameter Optimisation, which optimises the leaves, and Tree Growth, which extends the tree starting from a leaf.

For a set of autonomous driving scenarios, Likmeta *et al.* [221] propose parametric rule-based policies, visualised in the form of trees. Such a policy is a parametric set of rules which is learned using policy gradients with parameter-based exploration [305].

Although the following works do not aim at interpretability, we think it is interesting to mention them, because of their representation of the Q-table as a tree. Only a few works are presented, to give an overview. Ersnt *et al.* [104] present two methods: extra trees and totally randomized trees. The RL batch mode consists of approximating the Q-function based on a set of agent's interactions within the environment. In this context, the two methods are compared with a set of classical supervised learning methods. In Relational MDP, states, actions and background knowledge are represented by a set of predicates and constants forming ground atoms. Das *et al.* [75] describe a sample efficient approach to approximate the Q-function in the form of a regression tree in a Relational MDP. By providing a set of knowledge about the problem, the Q function and policy are approximated in [64]. The ALKEMY algorithm [262] is used to learn a set of DT's for each agent task. In addition, a P function is deduced from the Q function, allowing a compact Boolean representation of the policy.

**Equation** The policy takes the form of an equation. This form is used when the agent's actions can be expressed as a function of the features of the agent's state.

The work of Hein *et al.* [149] proposes to use a genetic programming (GP) approach to find interpretable policies in a model-based RL setting. To do this, a dataset of interactions is collected (using any policy) to learn a world model $\hat{g}$. This is used to learn a policy via GP. This method is compared with two baselines: a NN model-based policy $\pi_m$ using $\hat{g}$, and a policy learned via GP based on a dataset of interactions generated by $\pi_m$. To measure the complexity of a GP-generated policy represented by a function tree, it is simply necessary to sum the different nodes by weighting them according to pre-defined weights (e.g. an addition costs 1 and the function tanh 4). The policies returned by the algorithm have a similar/better performance to the baselines, while being less complex and therefore more interpretable.

Deep Symbolic Policy learns interpretable policies for environments where the action space is continuous [209]. The framework is split into two parts: the Policy Generator, which uses an RNN to iteratively generate an expression tree representing the policy, and the Policy Evaluator, which evaluates the policy and returns the average episode reward. This is used to train the Policy Generator. Experiments show that the policies of this approach outperform the 7 DRL baselines, such as *Deep Deterministic Policy Gradient* [222] and *Trust Region Policy Optimization* [303]. To handle the multiple action dimensions of certain environments, the authors propose to use a pre-trained policy to 'anchor' the symbolic policy during training.

Finding an efficient and interpretable policy, represented by a small formula, is modelled by a Multi-arm bandit problem in [240]. The space of policies represented by a formula $F$, such that its size does not exceed a certain threshold (i.e. $|F| \leq K$), is approximated by a clustering method. A formula of minimum size is extracted from each cluster, resulting in $N$ potential policies, each of them represented via an arm in the Multi-arm bandit setting. The best performing policy is then extracted by solving this problem. Compared to previous work described in this section, this approach does not take into account a continuous action space.

For a traffic signal control problem, a regulatable precedence function is learned jointly with the DQN [27]. A precedence function is regulatable if it is monotonic on state features. In this work, such a function is assumed to be interpretable. 3 variants of DQN are proposed to approximate such a function in addition to learning Q-values.

**Program** An interpretable policy in the form of a program is proposed in the following works.

The Learning Embeddings for lAtent Program Synthethis (LEAPS) framework [348] synthesises a policy in the form of a program based on a domain-specific language. This framework consists of two stages: firstly, LEAPS learns an encoder-decoder that allows a program to be represented in a latent space, and secondly, an iterative search for a latent program is performed to solve a given task. The latent space is learned in an unsupervised fashion, using randomly generated programs. The aim is to ensure that two programs with similar policies are close in the latent space. Finding a good latent program according to the reward obtained is performed based on the Cross Entropy Method (CEM) [292]. To demonstrate the interpretability of the programs generated, a user study shows that users have succeeded in analysing and modifying the program so as to obtain a better reward.

In [382], simple programs are proposed for playing games in the arcade learning environment [37]. Cartesian Genetic Programming [249] represents the programs by a directed graph indexed by Cartesian coordinates. The input is an image, and the output is a set of scalars used to select the action to be performed. This evolutionary approach relies on a set of functions to build an interpretable policy where each internal node corresponds to a function from this set.

For a traffic signal control task, Gu *et al.* [134] use Monte Carlo tree search to find a program, and Bayesian optimisation [318] to fine tune its parameters. Beforehand, a domain-specific language is defined including control flows, conditions and instructions. In addition, four transformation rules are defined to iteratively build a program. The best programs found by the algorithm are small in terms of number of instructions and are easily interpretable.

**Graph** Two works use a neural network based on the structure of a graph.

Using a total of 20 expert demonstrations, a single Graph Neural Network (GNN) is learned via IL to solve robot manipulation tasks [223]. The GNN is used to model the underlying structure of the task. Based on GNNExplainer [396], the explanation consists of determining the neighbouring nodes and edges of the graph that contributed most to the agent's choice.

In [375], the NerveNet model is proposed to solve continuous control problems. It has been tested on a set of MuJoCo [344] environments. The graph structure used for a given robot is based on its skeleton. Thus, two types of nodes are part of the graph: bodies and joints that connect two bodies. Learned representations are interpreted using different plots, making it possible to analyse the learned policy.

The following three works use abstract machines (or automata), assimilated to graphs, to represent the agent's policy.

The approach proposed by Inala *et al.* [170] consists of learning, in a teacher-student setting, a policy represented in the form of a state machine. This policy can be easily interpreted and modified. The proposed approach is called adaptive teaching. Alternatively, the student learns to imitate the teacher's behavior, and the teacher learns to provide behavior similar to that of the student. The student's policy takes the form of a state machine, which consists of a set of modes (an internal memory state) to which is associated an action function (which is akin to a sub-policy). Transitions between two modes are modelled by switching conditions, which reflect the probability of an agent observation switching from one mode to another. Action functions and switching conditions are represented by programs.

DeepSynth [142] synthesises a set of sequences of objects detected from agent states within an environment. Using an image segmentation method, a set of objects is extracted from a state. With the sequences collected, a deterministic finite automaton (DFA) is produced. This DFA is used to guide the agent in learning its policy. Using both the DFA and the MDP, a Product MDP is generated, where each transition corresponds to a sub-task of the agent.

In [360], a Tsetlin Machine [130] is used to learn a policy which is an alternative to the classical NN which uses Tsetlin Automaton to form clauses. Thus, the Regression Tsetlin Machine method [1] is combined with Q-learning. The results show that too many clauses are needed to have a good policy, making the method uninterpretable.

### 2.2 Policy Summary

In the blue sky papers [13, 14], Amir *et al.* propose a framework for the Strategy Summarization problem that can be decomposed into three parts: World States Representation, Intelligent States Extraction and Strategy Summary Interface. World States Representation is concerned with encoding the state space to make it comprehensible to the user, for example by grouping states into abstract states. Intelligent States Extraction presents different directions for extracting states that are useful for the user's purpose. The summary should be of reasonable length but still provide sufficient information to the user. Strategy Summary Interface provides considerations for user interface design. In addition, this work presents ideas for methods of evaluating summaries, such as domains or metrics.

#### 2.2.1 Sequences

The work presented in this section provides summaries in the form of a set of state-action sequences of the agent behavior within the environment.

The HIGHLIGHTS algorithm [12] provides a set of sequences that summarise the agent's behavior. To select sequences, Amir and Amir focus on important states using the notion of *state importance* [62] (which uses the agent's Q-values). From the important states, sequences are created to provide the user with more context. Thus, a sequence includes states that precede the important state, the state itself, and states that follow it. In the same work, the HIGHLIGHTS-DIV algorithm is proposed to provide more varied sequences as a summary. Several studies have used HIGHLIGHTS to provide explanations [166, 307]. Huber *et al.* [166] use HIGHLIGHTS-DIV coupled with saliency maps to provide additional information. A saliency map makes it possible to determine which part of the state (which takes the form of an image) is responsible for the agent's choice of action. In the same vein, Septon *et al.* [307] combine HIGHLIGHTS-DIV with the reward decomposition approach. This method decomposes the reward function into several interpretable objectives, making it possible to determine which goal(s) the agent is seeking to maximise by performing a certain action. Saliency maps and reward decomposition will be described in more detail in Section 4.1.1 and Section 4.2.2 respectively. Amitai *et al.* [17] propose the CoVIZ algorithm to compare the outcomes between the action $a$ taken by the agent from a state $s$ and a counterfactual action $a'$. A variant of this algorithm is described for extracting a summary of sequences. In this case, the agent's sequences are proposed at the same time as the counterfactuals. These counterfactual sequences start with a counterfactual action $a'$, then follow the agent's policy. The pairs of sequences that contrast the most in terms of outcomes (based on the value function) are displayed to the user.

Sequeira and Gervasio [308] propose a different approach to providing a summary. During the agent's interaction with the environment, a set of data is collected to extract information about, for example, the frequency of situations encountered or the agent's confidence in these decisions. These Interestingness Elements are used to provide the user with a visual summary. In [108], this approach was used in conjunction with the reward decomposition method. Interestingness Elements has been extended by the IxDRL framework [309] which contains new interestingness elements to analyse, a method for clustering trajectories based on interestingness elements and the use of SHAP [234]. This model-agnostic method is used to determine the impact of each feature of a state on the agent's decision. SHAP will be described in more detail in Section 2.2.3.

Another approach [207, 208] shows the impact of the user's mental model and the model used to create the summary on the user's reconstruction of the agent's policy. Lage *et al.* compare IL with Inverse RL (IRL) [261]. As a reminder, given a set of state-actions situations, IL aims to learn a policy that mimics the given decisions and generalizes over unseen situations. IRL consists of extracting a reward function based on a set of sequences of the agent. In order to provide informative sequences to communicate the agent's goal, [162] use an algorithmic teaching approach to model users' mental models.

Frost *et al.* [113] use an exploratory policy to reach a set of various states not seen during the training phase. Sequences are created from each of these states using the agent's policy. The aim is to provide an informative summary of the agent's behavior in the case of a state distribution shift that may occur during the test phase. In the same idea, Altmann *et al.* [10] use an evolutionary algorithm to perturb the initial state and obtain a varied set of sequences.

In order to learn a policy based on a dataset, Gottesman *et al.* [129] propose an efficient approach to identify the transitions in the dataset that are most important in the agent's learning. The idea is that removing an influential transition has a big impact on learning. To determine the influence of a transition $t$, a disparity of predictions is performed between a critical function that has learned with the entire dataset and a critical function that has learned without $t$.

#### 2.2.2 Critical States

An agent's policy can be summarised simply as a set of so-called critical states in the agent's interactions with the environment.

In this sense, Huang *et al.* [161] provide the user with a set of critical states of a policy as an explanation. This provides a restricted view of the agent's capacities on critical states, i.e. states where the Q-value varies greatly between different possible actions.

Based on a generative model producing an agent's state and a user-defined objective function, the method proposed by Rupprecht *et al.* [293] can be used to identify and visualise agent weaknesses. It has been tested on Atari games, where the state is an image. With this approach, it is possible, for example, to look at a given action $a$ by generating states in which the agent assigns a large Q-value to $a$, then seeing whether this action is coherent from the states generated.

In the context of MDP, Sreedharan *et al.* [325] propose a policy summary in the form of milestones, called landmarks, that the agent seeks to reach. The summary is a partial order of landmarks. For the exploding blocks-world problem [398], an ordering of landmarks is described in Figure 5. The identification of landmarks is done using compilation procedures and the ordering between these landmarks using methods for reasoning about task hierarchies in MDP's.

Reward redistribution is the process of creating a dense reward function for a given problem [23] (as opposed to a sparse reward function). Using this, Dinu *et al.* [87] present a method for extracting key events from an agent's behavior. An event corresponds to a cluster of state differences. The agent's policy is then expressed as a sequence of key events.

Other works are particularly interested in summarising the agent's training [251, 74, 73]. The idea is to save relevant tuples of snapshot images (i.e. states), actions, and weight distributions in memory to keep the agent's important learning experiences. The weight distribution associated with each snapshot-action pair corresponds to the impact that snapshot has on the development of the policy. Important snapshots can then be viewed to understand the agent's decisions. In [73], saliency maps are generated to reduce the number of important snapshots using Grad-CAM [306].

As a verification framework for DRL, Verily [195] allows the user to check whether a policy satisfies a certain requirement. This framework combines scalable model checking and formal NN verification methodologies. This verification is carried out at the level of the states of an agent. If the requirement is not met, a counter-example is proposed (in the form of a state).

#### 2.2.3 SHAP

This work is based on the model agnostic method, called SHAP [234]. It is interesting to note that this method is mainly used to explain classifiers. As already mentioned, it consists of determining the impact of each feature of an input on the output of the model. In our context, the input is a state, the output an action, and the model the agent's policy. This method is an approximation to the computation of Shapley values [310]. Originally from the field of game theory, Shapley values are used to determine the extent to which a member of a coalition contributes to the final value obtained. To determine Shapley values, it is necessary to calculate for each player $m$ of the game, the set of possible coalitions $c$ containing $m$, and to see the impact of its removal from $c$ on the final value. SHAP proposes different approximations to Shapley values, which consist of limiting the number of coalition samples. Two approximations have been used in the work presented in this survey: Kernel SHAP, a model-agnostic estimation, and DeepSHAP, an estimation specific to neural networks. Kernel SHAP is based on Linear LIME [285] (this method will be described in Section 2.4.3) and DeepSHAP on DeepLIFT [312]. As this section is concerned with the explanation of policies, SHAP is used globally: it identifies the impact of each feature on the agent's policy.

There are two works that use the method as is: Zhang *et al.* [406] use the DeepSHAP method in the context of power system emergency control and Wang *et al.* [378] use Kernel SHAP for a machine control use case (an example of SHAP is shown in Figure 6).

**Figure 6.** SHAP for a set of states along an episode for a machine control use case [378]. The evolution of the importance score of each feature is represented by a colored curve in the chart, this importance varying over time. The figure shows a line chart with the x-axis labeled time (s) ranging from 0 to about 16, and y-axis labeled Scores ranging from -0.2 to 0.3. Four colored curves represent features x (blue), v (yellow), phi (green), and omega (red).

**Figure 5.** Landmarks and their ordering obtained using TLdR [325] for the exploding blocks-world problem [398]. Vertices are landmarks and each blue edge defines an order between two landmarks. The green nodes are goals that the agent must reach. The figure displays two directed acyclic graphs of labeled ovals (e.g. Achieve holding b7, Achieve clear b8, Achieve on b7 b10) connected by arrows, with the goal nodes highlighted with thicker green outlines.

In the ixDRL framework [309], SHAP values are calculated using XGBoost machines [57] to simplify the analysis. Beechey et al. [35] distinguish their approach from previous work by proposing a general framework for the use of shapley values in the context of RL. They also propose a method for providing global explanations of the agent's performance, i.e. the expected return.

In the context of cooperative MARL, Heuillet *et al.* [154] compute SHAP values using Monte Carlo sampling. This use of SHAP is closer to the basic principle of Shapley values. In fact, this method makes it possible to determine the contribution of each agent. In the same context, Angelotti and Diaz-Rodriguez [20] compare the calculation of Shapley values with that of Myerson values [258], preceded by the construction of a Hierachical Knowledge Graph (HKG) representing the agents and their features. Myerson values are an equivalent of Shapley values specific to cooperative games constrained by a graph. They are therefore calculated on the basis of the HKG. In this work, the contribution of agents' policies and their features are studied jointly. The computation of Myerson values leads to a significant time saving compared to Shapley values.

#### 2.2.4 Policy Comparison

Two works have been found to compare two policies. This type of method makes it possible to evaluate two agents by identifying the differences in their behavior.

The DISAGREEMENTS algorithm [16] shows a visual summary of the most important disagreements between the two agents. The agents' Q-values are used to determine their disagreement. The summary takes the form of a set of various trajectories. This disagreement is measured in terms of ability whereas [122] focuses on a disagreement based on preferences. Indeed, in a problem where different winning strategies may exist, an agent may prefer one strategy over another. Based on a set of disagreements obtained between two policies, the method determines the type of states (in terms of features) that the different agents seek to achieve.

### 2.3 Human-readable MDP

#### 2.3.1 Surrogate Model

**States Clustering.** The following work clusters the states to make the MDP interpretable.

Bekkemoen and Langseth [36] propose the ASAP method, which groups states into clusters representing abstract states. The explanation consists of two parts: a display of a representative state and its associated action for each abstract state identified and a modelling of a policy graph, which is a Markov chain showing the agent's behavior in the abstract state space. Abstract states are identified using attention maps (which can be likened to saliency maps). ASAP is illustrated with the Mountain Car environment [48] in Figure 7.

**Figure 7.** Explanation via ASAP [36] of the agent's policy for the Mountain Car environment. The first line shows a representative state for each hyperstate with its associated action. The second line is divided into two parts. On the left, a policy graph models the policy based on the hyperstates. Each vertex is a hyperstate and each edge represents a transition between two hyperstates (weighted by its probability of occurrence). On the right, the attention maps of each hyperstate are displayed (where an attention score is associated with each feature, determining the impact of the features on the agent's decision-making). The figure shows five hyperstates H0-H4 (each with a Mountain Car snapshot showing velocity arrows and an associated action like "accelerate left/right"), a Markov chain graph of transitions between H0-H4 with probabilities like 0.96, 0.93, 0.99, and a heatmap matrix of attention scores for Position and Velocity across the hyperstates.

In the next two paragraphs, the explanation describes a specific cluster of states in natural language that answers the user's question.

Given a set of communicable predicates for describing states, Hayes and Shah [144] describe a framework for answering different questions about policy, such as '*When do you do action $a$?*'. To do this, the user's question is first identified, then a region of states is extracted based on the question. Using the optimised Quine-McCluskey algorithm [177], this region represented by a logical formula is minimised. Since the predicates are communicable, the resulting compact formula is transposed into a human language description for the user. A complementary study of this approach is carried out in [61] on the CartPole problem. This approach is combined with Reward Decomposition [105] to provide both local and global explanations of agent behavior in [172]. Based on this method, Booth *et al.* [47] carried out a user study to analyse the interpretability of different propositional theories.

Using knowledge compilation techniques, the framework of Wollenstein-Betech *et al.* [383] answers a set of questions relating to the choice of a given agent action. For example, this method can return the probability of doing an action knowing that we are in a state where one of its features is true. To do this, given a set of state-action tuples and a question, a deterministic Decomposable Negation Normal Form (d-DNNF) is constructed, model counting and probabilistic inference are performed on the d-DNNF and the result is translated into natural language.

By collecting all the transitions during agent training, Bewley *et al.* [41] propose a state abstraction as well as a temporal abstraction to analyse agent learning. The state space is decomposed into regions, called abstract states. The characterisation of these states is provided in the form of a tree diagram. The learning phase is divided into several windows, where each window is represented by a transition graph using the abstract states. This reflects the agent's interactions with the environment over a learning period. In addition, two graphs are provided to represent the rate of visits to abstract states as the training progresses, as well as the abstract states in which an agent's episode ends.

TRIPLETREE [40] is an algorithm that constructs a decision tree where the branches represent the action, the value and the state derivative estimate. This method allows the state space to be discretised according to three criteria: agent's action, value function and temporal dynamics. In addition, the transition probabilities between the different regions obtained are computed. A set of visualisation tools is used to analyse the environment and the agent. In addition, with these different regions, a set of explanations can be provided on the different criteria: factual explanations describing the limits of the region or counterfactual explanations based on a principle of minimal change to reach a certain region.

**State Transformation.** Transformations are applied to the states in order to make them interpretable in the work described below.

The field of State Representation Learning (SRL) consists of learning a low-dimensional representation of the state, which evolves over time and reflects the change induced by the agent's actions. Although the main objective of SRL is the performance of the learned policy, proposing a state representation composed of a small number of comprehensible features makes it possible to provide interpretability, as stated in the survey [213]. The objective of SRL is to learn a mapping function which, for a history of observations, returns the state in a low-dimensional representation. To do this, supervised approaches use the true high-dimensional state, while unsupervised approaches (the ones described in the survey) do not. Raffin *et al.* [282] propose a toolbox for SRL methods to compare approaches in different environments, using visualisation and metrics.

In [124], the agent module decomposes into two sequential parts: a neural network that learns to compress a state into a symbolic form, and a Q-learning algorithm that learns to reason based on the symbolic representation. In the tested environment, the neural network represents an image symbolically by a set of objects, their characteristics and interactions. This approach outperforms the compared DQN, and allows the agent's behavior to be interpreted through symbolic states.

D'Avila Garcez *et al.* [76] present a simplified version of the above approach as well as an extension. The symbolic representation of a state is composed of sub-states reflecting the relative position of the agent with respect to an object. Learning and action choice are modified to take account of this representation in the proposed extension. This addition of common sense results in a high-performance policy.

By combining a hierarchical learning architecture with conceptual embedding techniques to embed prior knowledge, the agent learns a more interpretable representation of the state [70]. A saliency map is generated on the hidden layers including the embedding of prior knowledge, to determine the impact of this knowledge on the agent's decision-making.

François-Lavet *et al.* [111] propose to combine the model-based and model-free approaches of RL algorithms to learn an abstract state representation. A specific loss is added for learning the representation of an abstract state so that features of the abstract state are impacted by the agent's actions. This makes the representation interpretable. The method is applied in a deterministic environment, but can be adapted for stochastic environments.

The HRL method described in [404] proposes to learn a representation of states in the form of a set of attributes, as well as a transition function between these attributes. In this work, the binary attributes are learned in a supervised manner and correspond to the relative positions of objects in a 3D environment.

In [56], a latent model of the environment is learned at the same time as the agent's policy in order to reduce sample learning complexity and generate semantic bird-eye masks. For an urban autonomous driving problem, the input consists of a front view image and a lidar image. The generated masks (learned from ground truth) are based on the input to display the map, the route that will be taken, the surrounding objects and the car position.

ReCCoVER [118] detects causal confusion in the agent decision-making on a set of critical states. From each critical state, the policy is tested on a set of alternative environments. These environments consist of imposing a certain value on a feature of the state. To detect causal confusion, it is necessary to learn a set of feature-parameterised policies for each subset of features. Causal confusion occurs when the agent's policy does not perform well in an alternative environment compared with a feature-parameterised policy. ReCCoVER then proposes a correction for these problematic states by providing the user (in this context, a developer/expert) with the feature subset on which the agent should base itself when it encounters this state.

#### 2.3.2 Inherently Understandable

**MDP Representation.** This section groups works that represents the MDP, especially the dynamics of the environment (i.e. the reward function and the transition function) in an interpretable way.

In the following work, interpretability is based solely on the reward function.

In a close air combat context, the reward decomposition [105] is used globally to determine in each tactical region the type of reward that is dominant [295]. A tactical region is defined beforehand and represents a set of states. In addition, a visualisation method is used to analyse the impact of the different types of reward on the different tactical regions. This representation takes the form of heatmaps of normalised Q-values.

The reward function learned in [44] is a weighted sum over potential outcomes, i.e. changes in features of a state. The approach is in an IRL batch setting, where the objective is to learn the reward function based on a dataset of expert policy trajectories. The weights of the different outcomes are used to determine their influence on the agent's choice of action. The main limitation of this approach is that it assumes that the reward function is a linear function over the features.

In [179], the idea is to transform a reward function $r$ into a more interpretable function $r'$. To use this framework, it is necessary to instantiate an equivalence relation between two reward functions and a cost function measuring the interpretability of a function. For example, the cost function can take into account the sparsity or smoothness of a reward function. In a gridworld-type environment, additional pre-processing is carried out to make the generated function more interpretable, which is then displayed in the form of a heatmap.

EXPRD [82] designs interpretable reward functions by making a compromise between informativeness and sparseness. This function is learned from a dense MDP reward function. The interpretable reward function must satisfy an invariance requirement, i.e. the optimal policy obtained with this function must belong to the set of optimal policies of the base MDP. To construct such functions in a large state space, a state abstraction method is proposed.

The Adversarial IRL algorithm (AIRL) [114] is extended by Srinivasan et Doshi-Velez [328] to build more interpretable reward functions. AIRL consists of using a discriminator to help learn a reward function from which the learnt policies are close to expert demonstrations. The interpretability aspect of the reward function is added using the Deep Neural DT architecture [393], which represents in tree structure the different feature valuations (called binning layer in NN), before using a decision layer to provide a reward. This method obtains smooth and sparse reward functions.

Preference-based RL (PbRL) consists of solving a problem modelled by an MDP where the reward function is not provided. A fitness function reflecting the user's preference on pairs of trajectories is used to learn a reward function upstream of the agent's policy. In [42], the proposed PbRL algorithm represents the interpretable reward function as a binary tree structure where each internal node is a test on a state-action pair and each leaf returns a reward. An example is shown in Figure 8.

**Figure 8.** A reward function as a binary tree structure for the original RoboCar environment consisting of a 4-wheeled vehicle whose goal is to reach a certain area while avoiding obstacles [42]. In this reward function, each internal node represents a test on a feature of an agent state, and each leaf a fixed reward. The color of the leaves reflects the amount of reward the agent can obtain (blue means a high amount, red a low one). The figure shows a binary tree rooted at "d >= 1.16" with subsequent splits on features such as "d >= 5.84", "y >= 1.68", "y >= -1.64", "beta >= 2.18", "beta >= -2.15", and "beta >= -0.676", with leaves labeled r_1 through r_8 showing reward values like 0.613, 0.0467, -0.565, -0.458, 0.254, 0.149, -0.41, -0.0497 (with standard deviations), colored from blue (high) to red (low).

The following works do not speak of explainability or interpretability but are mentioned because of the non-opaque form of the reward functions presented. Icarte *et al.* [169] propose to express the reward function in the form of a Finite State Machine as well as an algorithm, Q-learning for Reward Machines, to learn rules efficiently. Based on specifications given in a formal language such as Linear Temporal Logic (LTL), Camacho *et al.* [50] construct a reward machine to represent the different goals and temporal properties. For partially observable environments (i.e. the POMDP setting), [168] learns with the help of experiences a reward machine and [387] jointly learns high level knowledge modelled by a reward machine and the agent's policy. In [123], a model comprising non-Markovian Rewards is transformed into a Markovian model by learning a reward deterministic finite automaton from a set of sequences. The robustness degree [217] is used to extract a reward function from a logical formula expressed in Truncated LTL (TLTL). Such a formula corresponds to a specification of a task to be performed. Based on a set of demonstrations, Kasenberg and Scheutz [194] represent the task specification in the form of an LTL formula, instead of inferring a reward function that cannot be interpreted using IRL techniques.

Only the transition function is interpretable in the three following works.

Using expert knowledge about the environment, Kaiser *et al.* [188] describe a model-based RL method that learns an interpretable transition function. In the illustrative problem, the expert knowledge gives the information that the state $s_{t+1}$ obtained from the state $s_t$ by performing the action $a_t$ depends on three different sources. The transition function is learnt using a Bayesian approach, taking into account a structure including the three sources. The resulting probabilistic model shows the sources impact on the transition for each agent's state-action pair.

Motion-Oriented REinforcement Learning (MOREL) [127] improves the interpretability and, depending on the problem, the sample complexity of the RL algorithm used, by tracking moving objects in states. This approach is tested for a set of 59 Atari Games, where the state of an agent takes the form of an image. First, an unsupervised method is applied to learn to predict object and camera motions as well as object masks using a random policy to generate the data. Then, the agent uses and refines the previously learned model to learn its policy. Users can view the object masks to understand the agent's policy.

Although not presented as XAI papers, we find it interesting to mention the following works. [246] proposes an efficient heuristic called FIGE to represent the transition function by a graph. The SPITI algorithm [80] learns a decision tree for the reward function and one for the transition function using the decision tree induction algorithm ITI [353].

Finkelstein *et al.* [109] set out an approach based on a partial contrastive policy on a subset of states described by the user. The explainer looks for a meaningful transformation sequence to modify the current MDP into a contrastive MDP in which the agent would act according to the user's expected policy. A set of such user-interpretable transformations is provided as input to the problem. An example of such a transformation is the single-outcome determinization transform [397], which makes the transition function deterministic by considering only the most likely transitions. To find a sequence of transformations, a Djikstra-like search is performed.

Lazy-MDP's [176] are MDP's that include a default policy. The agent must then learn to estimate when it is necessary to replace the default policy and what action to take in replacement. Such an approach makes the policy sparse, making it possible to understand the states in which it is critical to take over from the default policy. In this work, the modelling problem and the policy generated make the approach interpretable.

In a human-robot cooperation setting, the framework proposed by Tabrez and Hayes [337] consists of detecting differences between the human and robot task models and proposing an interpretable modification for the human model. To do this, the human's reward function is inferred based on its behavior during cooperation using Hidden Markov Models. A particular POMDP is presented, so that the agent can choose between an action that informs the human or an action related to the task to be performed. This communicative action tells the human about a missing piece of information in its reward function.

**Relational RL and MDP.** The following works provide an interpretable representation of states and transitions and are based on works presenting relational MDPs [135] and Relational RL [97]. As a reminder, the objective is to be able to learn a generalizable and efficient policy based on a representation of the environment by a set of objects and relations between them. In this section, we present only an overview of work in these domains. For more details, we recommend reading the state of the art [358] dating from 2012, which does not therefore include recent work.

The works presented in this paragraph are described briefly because they are not proposed as explainability methods, although the approaches do make the MDP interpretable. Diuk *et al.* [88] introduce the Oriented-Object MDP (OO-MDP) to model the problem based on objects and their interaction and describe an algorithm for solving deterministic OO-MDPs. This algorithm is modified to solve Deictic OO-MDPs [241], which represent relationships between an object and an object class (instead of a simple object-object relationship), enabling better abstraction. Based on this idea, Scholz *et al.* [301] focus on physical domains, and propose two approaches: an extension of OO-MDP and a more efficient approach using a physics engine. In a model-based RL context, Veerapaneni *et al.* [363] approximate the observation model which represents an image (i.e. agent state) by a set of semantic masks representing objects and the dynamics model which describes the temporal evolution of the different objects for a given action. Schema networks [190] is a generative model of an MDP that learns the dynamics of the environment in a setting where the state is represented by a set of objects in an image. An interpretable MDP is built in [371] where abstract states are learned via ILP techniques and transitions and rewards are estimated. The policy is simply learned by value iteration in this MDP. The relational actions schemas class [372] groups together a set of languages that compactly represent the transition function using relational conditions and effects.

An RRL algorithm is proposed in [243] in the context of robotic tasks where the agent can request demonstrations from the expert in order to learn efficiently. This allows the agent to learn in a sample-efficient way and to request a demonstration when an action is unknown to it. In this work, each action is defined by a set of preconditions and effects weighted by probabilities. To guide the expert in generating demonstrations, the excuse principle [126] is used to explain why the plan failed. An excuse principle returns the important predicates that caused the plan to fail.

The framework described in [413] is used to extract objects from an image and learn the transition function. The objective of this framework is to generate the next image (i.e. state) using the information extracted from the image (its background and the various objects) and the action of the agent. To do this, the background extractor learns to extract the background, the object detector learns to produce masks of different types of static or dynamic objects and the dynamics net learns to make the transition. The object masks show that the method correctly learns to distinguish objects in the game tested, even on different levels of the game (i.e. unseen environments). An extension of this work [414] proposes a three-level learning architecture for learning model dynamics from the most to the least abstract. To aid learning, the output of an abstract level is provided to the level below. In order from most to least abstract, motion detection detects regions containing dynamic objects from an image sequence, instance segmentation produces coarse masks of dynamic objects and dynamics learning learns the transition function based on agent action and object relationships (as in [413]).

The Object-Level RL [6] is a framework that produces a high performance policy by learning a representation of states using relative and absolute distances, velocities, acceleration and contacts between the agent and objects in an image. This method was tested on a set of Atari games. During sample-efficient learning, the agent's exploration is based on a prior knowledge, which states that interesting experiences occur when two objects come into contact.

Zambaldi *et al.* [401] describe an architecture where the agent learns to extract objects from the image, determine relations between them and produce the policy and value for a state. For extraction, a CNN is used and a self attention mechanism [362] is used to compute relations between objects. These representations are then given as input to an RL algorithm for learning the policy and value function. The visualization of attention weights on different objects allows to analyze the learned relations between objects and to understand the decision making of the agent.

Symbolic Relation Network [4] takes as input a set of objects which describes the agent's state, represents them as a unary or binary relation and then concatenates them into a relational state used to learn Q-values with a DQN. In the environment used, the learned relations are interpretable: they represent the type of object and the relative location of the objects with respect to the agent. In this work, the object representation of states is not learned.

### 2.4 Visual Analysis

This section brings together a body of work focusing on the extraction of agent policy information to be displayed visually to the user.

#### 2.4.1 Visual Toolkit

A user interface including a set of tools for visual analysis of the agent's behavior is proposed in [250, 178, 373, 147, 245].

Mishra *et al.* [250] propose a total of 8 tools to be used to understand the agent's policy. This interface, which is described in Figure 9, requires the features describing a state to be interpretable. PolicyExplainer provides a global view of the frequency of actions and rewards, the policy, the value function (cf. panels A-C in Figure 9) and can be used to look at particular states and their associated Q-value, or a particular trajectory (cf. panels D-G in Figure 9). In addition, the '*Why?*', '*Why not?*' and '*When?*' questions are answered by providing the state features that have an impact on the choice of action (cf. panel H in Figure 9).

The other visualisation interfaces are designed for experts, to help them with debugging. DRLViz [178] analyses the latent internal memory of a DRL agent in the context of a video-based environment. DQNViz [373] focuses on the interpretation of DQN training and illustrates this using the Atari Breakout game. DynamicsExplorer [147] focuses on a DRL agent that includes a Long-Short-Time-Memory [157] layer for encoding environment dynamics. MDPvis [245] is used to analyse and modify an MDP and optimise the policy.

**Figure 9.** PolicyExplainer visual interface [250]. The different tools are divided into three parts. Panels A-C provide a global view of the frequency of actions and rewards, the policy, the value function. Panels D-G provide a detailed analysis of particular states and their associated Q-value, or a particular trajectory. Panel H answers the questions 'Why?', 'Why not?' and 'When?' using state features. The figure shows a multi-panel dashboard for the Stack Bot domain (with 13872 states and 6 actions) that visualizes Action Summary, Reward Summary, Policy Summary scatter plot, State-Value Overview, State Detail View, Trajectory Overview, Policy Detail View as a colored grid, and an Explanation View tree showing causal/informed/uninfected predicates.

#### 2.4.2 Interpreting DQNs

Three works are interested in visually analysing a DQN that has learned to play Atari games in order to understand its internal representation of states.

Zahavy *et al.* [400] propose to visualise the activations of the DQN by applying t-Distributed Stochastic Neighbor Embedding (t-SNE) [356] to reduce the dimension. Before visualisation, a set of data is collected, including hand-crafted features such as the agent's position, so that clusters can be identified and analysed. In addition, saliency maps are also generated on the different states collected. By analysing the dynamics between clusters, the authors identified that the DQN has learned a hierarchical aggregation of the state space.

Following on from this, Zrihem *et al.* [38] present a method for modelling the t-SNE representation of the DQN by a Semi Aggregated MDP, a human-interpretable approximation of the MDP. Also, clustering algorithms are used to identify the structure of the t-SNE maps, instead of having to construct features by hand. These clusters are states of the Semi-MDP and the probability of moving from one cluster to another defines the transition matrix of the Semi-MDP. An example of Semi-MDP obtained based on the policy of an agent that has learned to play Atari 2600 Breakout [37] is shown in Figure 10.

A specific architecture is presented by Annasamy *et al.* [22] with the aim of interpreting the DQN when the agent's states are images. The agent learns to focus on keys of a store composed of an action and a Q-value. Using deconvolutions, these cluster centres $(a, q)$ can then be assimilated to cluster centres. These keys can be used to understand the important elements of the reconstructed image that lead to the choice of $a$ and a Q-value $q$. This image represents an aggregate of states. For example, in Mrs Pacman, the visualisation of an action-return pair displays the different positions of the yellow blob in which the agent chooses the pair $(a, q)$.

**Figure 10.** Semi-MDP built on top of a t-SNE map for the Atari 2600 Breakout environment [38]. The t-SNE map is the set of points on which the Semi-MDP is based. It is made up of states (or clusters of MDP states), linked by transitions weighted by their probability of occurrence. Each cluster is modelled by an image of the game where the position of the ball in each state of the cluster is displayed in red. The figure shows a colored t-SNE scatter plot in the background with about 20 Breakout game screenshots overlaid as cluster representatives, connected by directed weighted edges (probabilities like 0.24, 0.32, 0.46, 0.56, 0.18, etc.).

#### 2.4.3 Inspection

The next two works simply propose to study the agent's behavior by analysing the metrics and displaying them in the form of plots.

ETeMoX [352] is a framework for keeping track of the evolution of certain metrics as the agent is trained. This framework consists of three parts. The Translator, which transforms the data collected from the agent's interaction with the environment. The Complex Event Processing [232], which filters the data to select only the relevant ones. The Temporal Graph Database, which represents relationships between multiple evolving metrics with time dimension. In this work, three RL algorithms were compared using ETeMoX by monitoring the overall reward, the exploration/exploitation trade-off and a feature specific to the environment used. Dethise *et al.* [81] study the *Pensive* agent by looking at, for example, the probability of action choice or confidence in the choice of actions. In addition, a study of the global importance of the agent's state features is carried out using LIME [285].

This method estimates the influence that each feature of input $x$ has on the model's output. In our context, the input is a state of the agent, the model the agent's policy and the output an action. To calculate the influence of features, a transparent surrogate model is learned locally around $x$. This surrogate model is a linear model. To train it, a set of points is generated by perturbing $x$ and weighted as a function of distance from $x$. After training, the weights of the linear model are used to explain the influence of each feature of $x$ on the output of the model. In [81], LIME is used on a set of states to provide the average contribution of each feature to the agent's choice of actions.

**Table 1.** Interpretable Policy works.

| Type | | Refs |
|------|---|------|
| **Surrogate Model (34)** | Decision Tree (17) | [71], [65], [137], [86], [84], [34], [248], [361], [314], [60], [227], [2], [316], [51], [231], [226], [181] |
| | Graph (9) | [346], [244], [91], [196], [367], [203], [72], [167], [211] |
| | Program (5) | [365], [415], [201], [49], [364] |
| | Rules (2) | [259], [319] |
| | Equation (2) | [405], [205] |
| **Inherently Understandable (54)** | Hierarchical Policy (20) | [313], [182], [386], [384], [236], [55], [184], [43], [359], [286], [19], [340], [212], [327], [117], [391], [404], [107], [95], [206] |
| | Rules (12) | [148], [160], [7], [268], [183], [407], [145], [274], [237], [92], [417], [321] |
| | Decision Tree (10) | [290], [315], [345], [69], [269], [64], [139], [221], [104], [75] |
| | Equation (4) | [149], [209], [240], [27] |
| | Program (3) | [348], [382], [134] |
| | Graph (5) | [223], [375], [170], [142], [360] |

**Table 2.** Policy Summary works.

| Type | Refs |
|------|------|
| Sequences (13) | [12], [166], [307], [17], [308], [108], [309], [207], [208], [162], [113], [10], [129] |
| Critical States (8) | [161], [293], [325], [87], [251], [74], [73], [195] |
| SHAP (6) | [406], [378], [309], [35], [154], [20] |
| Policy Comparison (2) | [16], [122] |

**Table 3.** Human-readable MDP works.

| Type | | Refs |
|------|---|------|
| **Surrogate Model (16)** | States Clustering (8) | [36], [144], [61], [172], [47], [383], [41], [40] |
| | State Transformation (8) | [282], [124], [76], [70], [111], [404], [56], [118] |
| **Inherently Understandable (33)** | MDP Representation (20) | [295], [44], [179], [82], [328], [42], [169], [50], [168], [387], [123], [217], [194], [188], [127], [246], [80], [109], [176], [337] |
| | Relational RL and MDP (13) | [88], [241], [301], [363], [190], [371], [372], [243], [413], [414], [6], [401], [4] |

**Table 4.** Visual Analysis works.

| Type | Refs |
|------|------|
| Visual Toolkit (5) | [250], [178], [373], [147], [245] |
| Interpreting DQNs (3) | [400], [38], [22] |
| Inspection (2) | [352], [81] |

## 3 Sequence-level methods

The methods described in this section explain sequences of interaction between the agent and the environment. A total of three ways of explaining sequences have been identified. *Counterfactual Sequences* allow a comparison of the agent's behavior with a sequence that uses an alternative policy, and thus determine the agent's strengths and weaknesses. The *Important Elements* of a sequence are those that have the greatest impact on the agent's ability to achieve a given objective. The use of *Human-readable MDP* makes it possible to understand, in the context of a sequence, the information available to the agent, the actions and the dynamics of the environment. The detailed taxonomy of sequence-level methods is described in Figure 11.

**Figure 11.** Detailed taxonomy for sequence-level methods. The figure shows a tree rooted at SEQUENCE branching into Counterfactual Sequence, Important Elements, and Human-readable MDP (which further splits into Surrogate Model and Inherently Understandable).

### 3.1 Counterfactual Sequence

A body of work aims to compare the state-action sequence of an agent with a counterfactual sequence where one or more actions differ from the agent's policy [357, 9, 119, 349, 350, 329].

Van der Waa *et al.* [357] propose to create a counterfactual sequence based on the user's query. An alternative policy is obtained by modifying the agent's Q-value so that the policy locally follows the user's contrastive query. The sequences are generated using the most probable transition at each time step (note that the transition function was obtained by training). The explanation consists of describing the sequences with the occurrence of the actions performed and states encountered, as well as the positive and negative outcomes.

In a context where the environment includes different reward classes, Alabdulkarim *et al.* [9] train a set of influence predictor models for each reward type during agent training. The counterfactual sequence is generated by considering the action (or sequence of actions) proposed by the user, then using the agent's policy. The sequences are compared using influence predictors (i.e. a calculation of the average influence of each type of reward in each sequence), the cost of the actions and the final reward obtained. An example showing the interest of this approach is presented in Figure 12.

The evolutionary algorithm, named ACTER [119], generates a diverse set of counterfactual sequences to propose to the user sequences that could have prevented the agent from reaching a state of failure. The algorithm performs multi-criteria optimisation, in accordance with the 5 properties defined upstream. Among these, validity ensures that the counterfactual avoids failure and proximity ensures that the sequence of actions resulting from the agent and the counterfactual one are as similar as possible.

The objective of Tsirtsis *et al.* [349] is to provide a counterfactual sequence that leads to a better outcome under the constraint that at most $k$ actions are modified with respect to the agent's sequence. This work is carried out in a MDP context where the transition function is modelled by the Gumbel-Max structural causal model [266]. The algorithm proposes a polynomial-time solution based on dynamic programming. In the case where the state space is continuous, this problem is NP-hard [350]. Tsirtsis and Gomez-Rodriguez then proposed a search method based on the A* algorithm.

For a specific drone parcel delivery environment, Stefik *et al.* [329] use MSX [105] (which is described later) to explain in natural language an action sequence in terms of risk factors. The risk is decomposed into different factors, so as to be able to understand what types of risk the agent is trying to avoid by performing a certain action. In this paper, the authors answer two questions related to the agent's action sequence: '*Why?*' by comparing two sequences on the different risk factors and '*What?*' by identifying in a sequence the action that leads to the greatest difference in risk.

**Figure 12.** Possible explanations for the sequence of actions that the agent (represented by a robot in cell (3,1)) performs in a 2D environment to reach the green cell [9]. The user's question is written in red. Explanation A (in black and marked by a red cross) is based solely on the agent's policy, while explanation B (in blue and marked with a correct tick) uses influence predictors. According to authors, explanation B is of better quality. The figure shows a small grid with the agent (a person icon) asking "Why did you go down instead of up?", with response A (rejected): "My expected future reward for going down is 0.95, while my future reward for going up is 0.80", and response B (preferred): "If I go up, I will pass through regions influenced by dangerous obstacles, going down feels safer."

### 3.2 Important Elements

The aim of these methods is to determine, within a given sequence, the most important elements for achieving an objective.

The EDGE algorithm is a self-explainable model that predicts, for a given episode, the final agent's reward [138]. The provided explanation takes the form of a set of important time-steps within the agent's interaction episode with the environment. An explanation of a game of Atari 2600 Pong [37] is shown in Figure 13.

**Figure 13.** An explanation of an episode from a game of Atari 2600 Pong [138]. The objective is to determine the important time-steps for the defeat of the agent controlling the green pad. The colored bar below each state of the episode represents its level of importance (yellow represents a low level, orange a medium level and red a high level). The figure shows a horizontal strip of Pong game frames (paddles and ball on a brown background) with a colored importance bar underneath that transitions from yellow through orange to red.

In a video-based environment, Liu *et al.* [228] identify the critical states of a sequence sufficient to predict the final reward obtained by the agent. Two models are learned for this method: the return predictor, which predicts the agent's final reward based on a (partial) video (i.e. a sequence of frames), and the critical state detector, which identifies critical frames by learning to mask non-critical ones.

The History eXplanation based on Predicates (HXP) method [298] and its variant, named Backward-HXP [300] are part of this line of work. The aim of these methods is to provide, within a sequence, the most important actions for the realisation of a predicate. This predicate is respected in the final state of the sequence studied, and may represent the success or failure of an agent, or other characteristics specific to the environment studied.

### 3.3 Human-readable MDP

Few works have been found in this category. As with the policy explanation category, the work can be broken down into two parts: the use of surrogate models to make MDP component(s) interpretable, and the use of MDP component(s) interpretable by design.

#### 3.3.1 Surrogate Model

Relying on a user-defined vocabulary, Sreedharan *et al.* [324] present the idea of learning propositional concepts to explain the agent's action choices in a contrastive manner. With these propositional concepts, actions are represented in a symbolic way by preconditions, effects and their cost. The user proposes an alternative sequence of actions to the agent. The explanation shows that the user's sequence does not lead the agent to achieve the objective, or leads it into an invalid state, or describes that the proposed sequence is more costly than the agent's sequence.

Soni [322] proposes to explain a set of transitions (or sequence) using a sequence explanatory message. Firstly, by collecting a set of data by interacting with various users, the method groups users into user-types and learns their associated labelling function. These functions allow to determine whether an explanation works for a user-type, and is therefore useful for providing personalised explanatory messages. Once this has been done, explanations can be generated for an unknown user. However, the user's type is not given. To provide personalised explanation, the problem consists in solving a POMDP where the hidden part of the state corresponds to the user's type.

#### 3.3.2 Inherently Understandable

In a context of sequential reconstruction of magnetic resonance images, Li *et al.* [216] use RL to successively apply interpretable pixel-wise operations. In this environment, the action space is directly interpretable because each action corresponds to common Computer Vision filters, such as the Sobel filter or the Gaussian filter.

**Table 5.** Sequence-level works.

| Type | | Refs |
|------|---|------|
| Counterfactual Sequence (6) | | [357], [9], [119], [349], [350], [329] |
| Important Elements (2) | | [138], [228] |
| **Human-readable MDP (3)** | Surrogate Model (2) | [324], [322] |
| | Inherently Understandable (1) | [216] |

## 4 Action-level methods

The methods described in this section explain the agent's choice of action. The various methods have been grouped into 2 types of explanation: *Feature Importance*, which determines the features of the agent's state that led it to choose an action, and *Expected Outcomes*, which justifies the agent's decision by providing its future potential impact. The detailed taxonomy of action-level methods is described in Figure 14.

**Figure 14.** Detailed taxonomy for action-level methods. The figure shows a tree rooted at ACTION. Feature Importance splits into Saliency Maps (Perturbation-based Approach, Gradient-based Approach, Attention Mechanism), Model-agnostic Approach (SHAP, LIME), and Counterfactual State. Expected Outcomes splits into State (Probability, Causal Lens, Contrastive), Reward, Sequence, and Action.

### 4.1 Feature Importance

To explain the choice of an action $a$ from a state $s$, the following works describe the importance of the features of $s$ in the agent's choice. Most of this work falls into three categories: saliency maps, model-agnostic approaches and counterfactual states.

#### 4.1.1 Saliency Maps

Saliency maps are used when the state of an agent is represented by an image. This method makes it possible to show pixels, or groups of pixels, that are essential in the agent's decision-making. The major different ways of calculating them are: gradient-based, perturbation-based and attention mechanisms approaches. Based on the image, gradient-based approaches use the gradient of a class (i.e. action) in the last layer of the neural network and recover the information with back-propagation. An example of a well-known method used in certain works (e.g. [185, 264]) is Grad-CAM [306], which will be described later. Perturbation-based approaches perturb the input image and evaluate its impact on the agent's policy. Some works [173, 218, 18] use this approach on groups of pixels identified as objects in the image. Attention mechanisms are modules used to make the agent focus on certain parts of the image. These attention mechanisms are used to generate saliency maps. As this type of approach is fashionable, we do not intend to provide an exhaustive description of the work related to saliency maps in the XRL context. For a detailed analysis of the use of saliency maps for RL agents, see [26]. Most of the experiments on saliency maps were carried out on Atari2600 games using the Arcade Learning Environment [37]. Note that saliency maps can also be referred to as attention maps or attention masks in this section.

**Attention Mechanism** This section focuses on methods providing saliency maps based on attention mechanisms.

In [256], the current state of a Long Short Term Memory is used to make a set of requests, sent to an MLP. Its output is used, together with the agent state, to generate saliency maps. This system is called *attention head*. Experiments have shown that the agent has learned to focus its attention on regions/objects present in the image.

Based on A3C [252], Itaya *et al.* [171] propose two attention masks: one for the policy and one for the state value. These attention masks are generated from the feature map extracted by the first part of the neural network, called the feature extractor, and then used to predict the agent's action and estimate the value of the agent's state.

Two feature extractors are presented in [265]: Sparse FLS and Dense FLS (where FLS stands for Free Lunch Saliency). The main objective of this work is to provide an attention mechanism that makes both visualisations interoperable, without impacting the agent's performance. Both approaches and baselines were tested using Atari-head [410], a dataset containing human gameplay and visual tracking. None of the approaches studied stood out from the others.

The Action Region Scoring (ARS) module proposed in [229] is used both to explain the choice of action and to improve learning. In the CNN, at the output of each convolutional layer to which a ReLU activation has been applied, an ARS module is used to identify the important regions and then combined with the current image representation. This helps learning by indicating the regions to focus on. For the explanation part, the saliency map is produced by retrieving the outputs of the ARS modules and combining them.

In an approach based on fuzzy rule learning, Ou *et al.* [268] propose to use the Compact Convolutional Transformer [143] to provide attention masks. In addition to this visualisation, the choice of agent is explained by providing the influence of each fuzzy rule.

Based on the U-net architecture [288], SSINet takes the form of an encoder-decoder to obtain an attention mask of the input image [311]. SSINet is built in such a way as to respect 2 properties: maximum behavior resemblance and minimum region retaining. The first describes that the agent's prediction must be consistent between the image and the image overlapped with the attention mask and the second describes that the attention mask must focus on as little information as possible, so as to provide sparse explanations.

Tang *et al.* [339] only provide the agent with a subset of patches from the original image for its decision making. Indeed, the image is first cut into a set of patches, which are then evaluated using a self-attention module. Finally, the $k$ most important patches are extracted from this evaluation and given to the agent. This technique makes it possible to understand on which patches of the original image the agent bases its decision.

In [186], different types of input are tested for an unmanned ground vehicle navigation problem. Several attention mechanisms are implemented depending on the features (visual or not). Thus, the self-attention mechanism used to explain the problem is applied to an image or a vector of features.

The DRIVE model [30] is proposed for a traffic accident anticipation problem using videos. Two attention maps are generated for the same frame $f_t$: a bottom-up approach which generates a map based directly on $f_t$ and a top-down approach which applies a transformation to $f_t$ before generating the map. This second approach focuses the attention mechanism on a risky area, using a foveal vision module based on a fixation point p predicted by the agent with the previous frame $f_{t-1}$.

Region-sensitive Rainbow [394] is an enhancement to Rainbow DQN [152] which uses a region-sensitive-module to determine the important regions in which the agent focuses to choose an action. This module is similar to an attention mechanism, and returns importance scores for $k$ regions of an image. Of the three ways of visualising salient regions, the most user-friendly approach has been retained. It simply consists of a binary saliency map which displays just the salient region to the user (replacing the rest of the image with black pixels).

Focusing on the immediate reward, Yang *et al.* [392] propose to generate attention masks that reflect as much as possible the agent's reward. The reward obtained by using the policy from state $s$ is compared with the reward obtained by using it from state $s'$ constructed by combining $s$ and the attention mask. To minimise the absolute difference of these rewards, an RL approach is proposed. An extension of this method is provided to handle multi-step rewards.

The approach proposed by Wang *et al.* [376] is different from the rest of the work. In addition to an attention map, an explanation in the form of natural language is produced. The architecture of the algorithm presented breaks down into three parts: an encoder, which encodes the image into a set of features, an attention mechanism, which returns the salient features and the decoder, which generates a verbal explanation from the salient features. A total of four attention mechanisms have been proposed and compared with the BLEU metric [271] for the textual explanation. The best of these is called *adaptive attention*, which consists of assigning a weight to the features dynamically using the last word generated by the verbal explanation.

Wang *et al.* [379] use an attention mechanism to understand the agent's decision choices. In the same idea, Zhang *et al.* [408] present the Temporal-Adaptive Feature Attention algorithm. This identifies the most relevant features in the agent's choice of action. In MuJoCo [344] environments, the weights of attention show that the features specific to the agent's position are more important than those linked to its velocity in the choice of action throughout the duration of an episode.

The approach proposed by Huber *et al.* [165] does not require any additional module or specific architecture, but generates, as with the attention mechanisms, a saliency map with only one forward-pass. The saliency maps contain only the most important information. To do this, Layer-wise Relevance Propagation [29] is used: this is a general concept that identifies relevant pixels during the forward-pass by looking at the activations of each neuron. To limit the information to the most relevant, the authors use [255] to restrict the number of neurons tracked to 1 per convolutional layer.

**Perturbation-based approach** Works in this section uses a perturbation-based approach to generate saliency maps.

The perturbation method proposed by Greydanus *et al.* [131] involves adding spatial uncertainty around the pixel at coordinates $(i, j)$. This is done by interpolating the original image with a Gaussian blur applied around $(i, j)$. In practice, instead of applying this perturbation for each pixel, it was applied per group of 5 pixels, which reduces the computational cost and produces good saliency maps. Using the A3C learning algorithm, saliency maps are generated and displayed jointly for the agent's policy and for its value function $V$ (i.e. this captures the regions that are important for the agent's choice of action and evaluation of the image).

We have identified three works based on this method. Built on top of the agent implementation, Persiani and Hellström [275] propose a new architecture, called the Mirror Agent Model, to make the agent's behavior interpretable by the user. Both the agent model and the user model are represented by Bayesian networks. A metric is presented to measure the distance between the two models. The explanatory layer added to the agent (which adds a node to the Bayesian network) consists of using saliency maps based on the method of Greydanus *et al.* [131]. This method is also used in [136] to compare the saliency maps explaining the agent with the visual attention of humans. To do this, this study uses the Atari-head dataset [410] and AGIL [409], a model for predicting human visual attention. The authors found that, as learning progressed, the agent's attention maps became closer to human visual attention, which can also be used to understand agent failures. By modifying the method of [131] to apply it for a 2D grid-world environment, Douglas *et al.* [94] propose a display called Towers of Saliency, to interpret the agent's behavior over an entire episode.

Puri *et al.* [280] construct saliency maps based on two properties: specificity and relevance. This method is called Specific and Relevant Feature Attribution (SARFA). The Specificity property defines that the salience of a feature must focus on the perturbation of the action to be explained, $a$. Thus, with $s$ the base state and $s'$ the perturbed state, the salience must be high if $Q(s, a) - Q(s', a)$ is substantially greater than $Q(s, a') - Q(s', a'), \forall a' \neq a$. The Relevance property defines that the salience of a feature should only have an impact on the $Q$ value of $a$. In other words, the salience of a feature must be low if the perturbation also affects the $Q$ values of the other actions.

Rather than directly using the Q function [280] or the V function [131] to measure the impact of a perturbation, Yan *et al.* [388] propose using the Advantage function. Given a state $s$, an action $a$ and a policy $\pi$, the advantage of performing $a$ from $s$ and then following $\pi$ is: $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$. To locally perturb the image, a Gaussian blur is applied (as in [131]).

Huber *et al.* [164] evaluate a total of 5 perturbation-based approaches by inspecting the dependency of the methods on the NN parameters learned by the agent using the *sanity checks* metric [3] as well as their fidelity on the agent's reasoning using the *insertion* metric [276]. Among these approaches, there are the two aforementioned works: SARFA [280] and Noise Sensitivity [131]. *Sanity checks* consist of successively randomising the layers of the neural network and calculating a saliency map each time. These saliency maps are then compared with the original, with the expected positive result being that the maps differ significantly from the original. All the methods depend on the learned parameters, but Noise Sensitivity shows little dependency. The *insertion* metric starts from a noisy image (using black occlusion and uniform random perturbation) and iteratively reconstructs the base image by resetting the correct pixel values, starting with the most salient pixels indicated by the saliency map. The expected positive result is that the agent's action prediction should quickly be similar to that of the agent on the noiseless image. This would mean that the saliency map has put forward the 'critical' pixels in the agent's choice of action. With this metric, SARFA is one of the two best approaches. The authors suggest further research into the generation of perturbation-based saliency maps, by first determining which type of perturbation might be appropriate for the task in hand.

The following papers use saliency maps to focus on objects rather than pixels. The work by Iyer *et al.* [173, 218] uses a computer vision technique called template matching to recognise objects in the image and use them as a basis for saliency maps. To provide the agent with more information for choosing an action, object channels are added to the base image. These object channels are extracted using the recognizer object and are used to determine the positions of the various objects detected in the image. The perturbation in the image $s$ consists of masking an object, resulting in $s_o$ and determining its impact on the choice of action by comparing the $Q$ values. The maps generated highlight the objects that are salient in the agent's decision-making. A comparison between a 'classic' pixel saliency map and an object saliency map, based on a state-action pair from a game of Atari 2600 Ms PacMan [37], is shown in Figure 15. In [18], a user study was carried out comparing object saliency maps with reward decomposition (which is described later), a combination of the two and the control strategy, which simply displays the impact of the action on the state and the score obtained. In summary, participants understood the agent's behavior better when the explanation contained reward decomposition, although this may have led to cognitive overload for some participants.

**Figure 15.** Two saliency maps for a state-action pair from the Ms PacMan environment [173]. The action is '*move right*' and the sub-figure 'a' is the state from which a pixel saliency map (sub-figure 'b') and an object saliency map (sub-figure 'c') are generated. The saliency of an element is represented using a grey scale, where the darker the element, the more salient it is. In sub-figure 'c', we can see that the agent has focused on Ms PacMan and the pills in the right path to choose the '*move right*' action. Conversely, it did not focus on the part of the image that symbolises 1 life for the player, as shown by the white square in the object saliency map. The figure shows three side-by-side panels: a color screenshot of Ms Pacman, a grey pixel saliency map showing diffuse activations, and an object saliency map with discrete grey squares highlighting Ms Pacman and the pills.

**Gradient-based approach** This section focuses on methods providing saliency maps using a gradient-based approach.

Gradient-weighted Class Activation Mapping (Grad-CAM) [306] is a method that can be used on CNN's without having to modify the architecture or re-train the model, in order to create saliency maps. To find out the influence of pixels on a class c for a given image, a forward pass is performed, the gradient of class c is calculated, then the signal is propagated with a backward pass. Importance weights of the feature maps (where a feature map is the output of a convolutional layer of a CNN) are calculated and then the saliency map is generated by applying a ReLU operation on the weighted linear combination of the feature maps. This method has been used in several works to explain RL agents [185, 264, 380, 73].

Joo *et al.* [185] propose an architecture combining A3C [252] with Grad-CAM in order to visualise the salient parts of the image for the agent's choice of action. To describe the behavior of a swarm robotic system, Nie *et al.* [264] combined it with a deconvolutive network to produce saliency maps. Weitkamp *et al.* [380] use it by replacing ReLU activation with ELU activation. To reduce the number of important snapshots from the agent's learning phase, Dao *et al.* [73] use Grad-CAM. Similar states with common attentions are thus grouped together. The SmoothGrad method [317], which adds noise to the image so that the Grad-CAM saliency map is less noisy, was used. Note that in this paper, the method is not used to explain, but to make another explainable approach more sparse.

An in-depth analysis of an agent that has learned in the CoinRun environment [63] is proposed by Hilton *et al.* [156] by analysing a hidden layer of the agent that has learned to recognise objects in CoinRun. The integrated gradient method [334] is used to generate saliency maps.

In order to provide additional information for the visual analysis of a DQN, Zahavy *et al.* [400] calculate the Jacobian of the DQN as a function of the input image.

He *et al.* [146] propose an approach that does not use gradients, but works in a similar way to Grad-CAM. In addition to the saliency maps, textual explanations are provided. The agent uses an image and a feature vector to choose an action $a$. The explanation of $a$ based on the image is produced by a new method using both CAM [411] (on which Grad-CAM is based) and SHAP [234]. The textual explanation of $a$ based on the vector is provided using SHAP. Note that both approaches are also used to provide global explanations of agent behavior.

#### 4.1.2 Model-agnostic Approach

This section focuses on the use of SHAP [234] and LIME [285] and their variants to determine the importance of state features in the agent's choice of action.

**SHAP** Most of the works focus on SHAP, using either Kernel SHAP [287, 378, 51, 231, 284] or DeepSHAP [406, 341, 283, 219, 302].

For example, Rizzo *et al.* [287] use Kernel SHAP for a traffic light control problem and Wang *et al.* [378] for an automatic crane control problem. Two works [51, 231] propose to compare SHAP with linear model trees (LMT), already presented in the policy explanation section. In [51], the author suggests that LMT and SHAP 'capture similar relationships between input features'. Lover *et al.* [231] also compare LIME and use a K-means summarizer to reduce the number of samples to be used for the calculation of SHAP values. Both studies show that LMT is the fastest at providing an explanation, and [231] state that LIME is too slow to be used in real time. KernelSHAP and Causal SHAP [151] are compared in [284]. Causal SHAP is a variant of KernelSHAP which computes SHAP values by taking into account the dependencies between the features of a state. The results highlight the benefits of using Causal SHAP.

DeepSHAP is used in [406] for a power system emergency control problem and in [341] where the explanation in a MARL context consists of an explanation of the decision of a single agent or the decisions of all agents. Remman *et al.* [283] use DeepSHAP to understand the impact of different variables in a robotic lever manipulation problem. Values are calculated and displayed for each agent's decision over an entire episode. Based on the same use, Liessner *et al.* [219] propose to calculate values with DeepSHAP in a longitudinal control task and Shreiber *et al.* [302] in a traffic signal control task.

Within the IxDRL framework [309], SHAP is used both to study the impact of features on the agent's decision-making in a single state and also globally, by displaying the average impact of features on a set of states. In the same vein, [35] propose two methods for explaining locally (i.e. from a given state) and globally the agent's performance, i.e. its expected return.

In addition to proposing a benchmark specific to the XRL domain, Xiong *et al.* [385] present an algorithm entitled TabularSHAP. It is limited to explaining tabular states. To use it, the user first needs a set of interactions of the agent within the environment, which will be used to learn an ensemble tree model. Finally, TreeSHAP [233], which is specific to tree-based models, is used on the basis of the ensemble tree model learned.

**LIME** In order to analyse the behavior of an agent on a heating ventilation and air-conditioning control task, Kotevska *et al.* [202] propose three modules: Model assessment, Model local view and Model global view. The first consists of a probabilistic analysis of the agent's behavior and a statistical analysis of the impact of features on the agent's behavior. The second consists of using LIME for an agent decision and the third consists of using visualisation tools, e.g. a partial independence plot, to understand the impact of features on agent decisions. As already mentioned, [81] uses LIME to explain a single agent's decision, as well as the agent's global behavior. LIME is also used in [214] to add an explanatory layer to an advanced persistent threats defense mechanism.

#### 4.1.3 Counterfactual State

A body of work aims to provide an answer to the question '*Why does the agent perform action $a$ from state $s$ rather than action $a'$?*' by proposing a counterfactual state $s'$, close to $s$, in which the agent would have chosen $a'$.

A number of studies [267, 163, 296] have used GAN [128] for this purpose. As a reminder, this architecture consists of jointly learning a generative model $G$ that creates data and a discriminative model $D$ that determines whether the input data is generated or comes from the training dataset. After learning, only the generative model is used.

In order to provide a counterfactual explanation where the state is an image, Olson *et al.* [267] propose to learn, in addition to these two models, an encoder which encodes the state $s$ in order to omit information from $s$ related to the action choice. $D$ is trained to determine the action distribution based on the encoded state. $G$ is trained to reconstruct $s$ based on the action distribution and the encoded state. For training, it is necessary to get an initial dataset of state-action pairs using the agent to be explained. This method, tested on Atari2600 games, generates a state $s'$, close to $s$, in which the agent performs a different action.

In the same line, GANterfactual-RL [163] is based on the StarGAN architecture [59]. This helps $G$ to construct states in which the agent chooses the desired counterfactual action. Examples of explanations are shown in Figure 16. With the help of a user study and metrics, GANterfactual-RL is more efficient and has better results in terms of metrics than the previous approach mentioned above [267]. Among these metrics, proximity and sparsity measure the distance of the generated counterfactual state $s'$ from the base state $s$, and validity verifies that from $s'$, the action predicts the desired action $a'$.

Another competitive approach is proposed by Samadi *et al.* [296] to generate counterfactual states using saliency maps. The proposed method, called SAFE-RL, is based on attentionGAN [338]. $G$ learns to generate a counterfactual state by taking as input the state, the associated saliency map and a counterfactual action. $D$ learns to differentiate true states from counterfactuals.

With states also corresponding to images, Druce *et al.* [96] present counterfactual states in their interface to help user comprehension. In this work, counterfactual states are obtained by pre-defined interventions on state $s$, such as the player's position in the image. For the same state, several counterfactual states are displayed to see how the agent would have acted according to certain modifications.

The RACCER algorithm [120] is used to identify a sequence of actions that leads to a counterfactual state in which the agent has a high probability of performing the desired action. This method is not image-specific, unlike the previous methods, and provides a recourse for the user to perform a certain action (with a certain probability). However, RACCER requires access to the environment dynamics.

#### 4.1.4 Others

To use the method proposed by Davoodi *et al.* [77], the features of the agent's state must be actionable. A feature is said to be actionable if the user knows which action or sequence of actions will increase/decrease the value of the feature. The aim of the method is to explain, for a given state $s$, the impact of features in reaching risky states. To do this, a transition graph $T$ limited by a number of actions is constructed from $s$, then a linear model is learned on all the states of $T$ to estimate the risk. The model weights describe the importance of the different features in reaching a risky state.

In [103], the aim is to explain a recommended action in an MDP by returning the most relevant feature of the current state $s$, with respect to the value function $V^\pi$ (which represents the expected reward of following $\pi$ from $s$). To determine this feature, we need to measure the impact of the change in valuation of each feature of $s$ on $V^\pi$. Each feature is tested separately, leaving the other features fixed. In addition to the relevant feature, a verbal explanation is proposed based on a knowledge-base dataset.

**Figure 16.** Comparison of counterfactual states in the Atari 2600 Ms PacMan environment [37] generated on the basis of a state-action pair (on the left) [163]. The agent's action is '*move right*' and the counterfactual one is '*move left*'. The state in the middle, where Ms PacMan is missing, is generated using [267] and the state on the right, where Ms PacMan is safe in the left part of the map, is generated using [163]. The first counterfactual state is not a convincing one as it shows that the agent chooses '*move left*' while Ms PacMan does not even appear, whereas in the second one, Ms PacMan is (partially) visible. The figure shows three Ms PacMan game screenshots side by side, with the leftmost being the original 'Move Right' state, the middle showing 'Move Left' generated by [267] (with Pacman absent), and the rightmost showing 'Move Left' generated by [163] (with Pacman visible).

### 4.2 Expected Outcomes

The work presented below focuses on explaining the agent's choice of action by giving and analysing the agent's expected outcomes. These works have been categorised according to the type of information provided to the user.

#### 4.2.1 State

This line of work is interested in providing the future state(s), or features of the state(s), resulting from an action $a$ from a state $s$, as an explanation.

**Causal Lens** In order to focus on causal relationship between action and state variables, Madumal *et al.* [239] build an Action Influence Model (AIM) which is a structural causal model [141] with the addition of actions. This model takes the form of a graph which is used to answer '*Why?*', '*Why not?*' questions by providing a (partial) causal chain describing the impact of the action on the features of the model. An extension of this work [238] introduces a distal explanation model to provide causal chain explanations using a decision tree representing the agent's policy. The shortcoming of this work is that the causal model is hand-crafted.

CauseOccam [368] is a model-based RL framework where the model learned is a sparse causal graph.

Herlau *et al.* [150] also propose learning a sparse causal graph, which the agent uses during training. The search for relevant causal variables is carried out by maximising the '*natural indirect effect*'. These two approaches could serve as a basis for the construction of AIMs.

With the same idea, Yu *et al.* [399] extend the use of AIM to problems with a continuous action space, without the need of prior knowledge of the environment causal structure. Indeed, to represent the dynamics of the environment, the causal model is learned based on the agent's interactions with the environment. An example of a learned causal model for the LunarLander environment [48] is shown in Figure 17.

CAMEL [96] is a framework that uses causal probabilistic models to explain the agent's choice of action.

**Figure 17.** Causal model for the LunarLander environment [399]. The features of a state are in blue, the actions in orange and the outcomes in green. An arrow between two elements describes a causal relation. The figure shows a directed graph with blue nodes (legs, theta, theta-dot, x, x-dot, y, y-dot) connecting to other blue next-state nodes and green outcome nodes (crash, rest, fuel), with orange action nodes (main, lateral) in between.

**Probability** The Memory-based eXplainable Reinforcement Learning approach [67] provides the probability of success (i.e. reaching a goal state) and the number of transitions to reach the goal state. When the agent is trained, the success probabilities $P$ and the number $T$ of transitions leading to a goal state are calculated and updated for each state. The agent can then provide answers to the questions '*Why does the agent choose action $a$?*', '*Why does the agent choose action $a$ and not $b$?*' or '*What is the probability of reaching a goal state in 8 steps from $s$?*'. The problem with this approach is that it is necessary to memorise $P$ and $T$ in order to explain. To overcome this problem, Cruz *et al.* [68] propose 2 additional methods for calculating or approximating $P$: learning-based and introspection-based approaches. The first approach learns a $P$-table of the probability of success in parallel with learning the agent's Q-table. The second requires no additional memory, as the probability of success is calculated directly from the Q-values.

In a Factored MDP context [198], Khan *et al.* introduce Minimal Sufficient Explanations (MSE) to provide a minimal explanation in terms of the number of templates sufficient to explain the agent's action. Templates are phrases used to describe the probability of reaching a certain state (e.g. '[action] is likely to take you to [state description] about [lambda] times').

Dodson *et al.* [90] present a system that provides an explanation of the optimal action via a natural language paragraph in a MDP context. One of the two explanation modules proposed is called Case-based explanation (CBE). CBE explains the usefulness of an action by using a database of previous transitions, called *cases*, to calculate statistics on the *cases* relevant to the recommended action. In terms of probabilities, this explanation makes it possible to explain the influence of an action on future feature values.

**Contrastive** Stein [330] provides counterfactual explanations based on the abstraction of sub-objectives. These sub-goals must be completed in order to achieve the agent's goal. Thus, the user can ask why the agent prefers to perform one sub-goal rather than another. The answer contrasts the two sub-goals with the probability of achieving the final goal (i.e. a state) and the cost of achieving the sub-goal.

Tsuchiya *et al.* [351] present a method similar to reward decomposition (which will be described in the next section), but focused on states. Additional neural networks are learned to estimate the Q-value of each state type. The different state types are provided in advance by the user. A contrastive explanation between the agent's choice of action and that of the user is performed by displaying and then comparing the different Q-values decomposed by state type.

Contrastive explanations are proposed using action-values based on given human-readable features [224]. To this end, deep generalized value functions [335] are learned to predict future feature accumulation. To avoid information overload, the MSX approach [187] (described in the next section) is applied to obtain a clear comparison of the features impacted by the actions. Note that features can represent the state of the agent, but also components of the reward function.

The Moody framework [32] allows the agent to self-assess his decisions. Based on the agent's Q-values, its confidence of reaching the final state is calculated [68], then converted into a pleasure/arousal scale [66] consisting of two dimensions: pleasing/unpleasing and excited/calm. Finally, a Growing-When-Required Network [242] is used to define the agent's current mood. Note that this method can also be used by an agent to evaluate the mood of other agents.

**Others** Interactive RL [342] is a framework in which the agent's learning is accelerated by instructions given by an expert. In this context, the works by Fukuchi *et al.* [115, 116] develop the Instruction-based Behavior Explanation (IBE) framework, which reuses expert instructions to explain the agent's choice of action $a_t$ from state $s_t$. To do this, IBE estimates $\delta(s_t)$ which represents the change in value of the features between state $s_t$ and the state reached $n$ steps later $s_{t+n}$. Then, an expert expression, which is a user interpretable signal, is associated to $\delta(s_t)$, either by a clustering method [115], or by classification using a NN [116].

With a state represented by an image, Pan *et al.* [270] propose to predict the semantics of pixels in future states. Image semantic segmentation consists of labelling each pixel of an image. It is used here in a vehicle driving context, where pixels can be labelled as *road, grass, sky, ...* The agent learns to predict this visual representation of the future state, which acts as an explanation.

#### 4.2.2 Reward

The agent's choice of action can be explained in terms of the reward that the agent is seeking to obtain. The vast majority of works uses the principle of reward decomposition [105, 187, 295, 172, 108, 286, 18, 17].

As a reminder, reward decomposition is used to represent the reward function in an interpretable way: it is decomposed into a set of comprehensible reward types. Thus, to understand the agent's choice of action, it is sufficient to look at the type of reward it seeks to maximise. Based on this method, [105, 187] propose Reward Difference eXplanations (RDX) and Minimal Sufficient eXplanation (MSX) to explain deep adaptive programs learned via RL [105] and agents from the CliffWorld and LunarLander domains [187]. RDX simply compares 2 actions from a state by comparing their Q-values for each type of reward. An example of RDX, from [172], is shown in Figure 18. In accordance with RDX, MSX provides a more compact explanation of an agent's preference for action $a$ over another action $a'$: only the reward types that have the greatest impact (positively and negatively) on the preference of $a$ over $a'$ are displayed to the user.

Several works combine reward decomposition (by simply displaying the different Q-values or by using RDX, MSX) with other methods. Of these, most combine it with approaches that explain the agent's policy. Saldiran *et al.* [295] use a global reward decomposition and a heatmap visualisation of Q-values. Iucci *et al.* [172] use autonomous policy explanation [144], Feit *et al.* [108] the Interestingness Elements framework [308] and Rietz *et al.* [286] the HRL setting. Anderson *et al.* [18] employ object perturbation-based saliency maps, a complementary explanation of the agent's choice of action. Amitai *et al.* [17] present a counterfactual explanation using reward decomposition.

**Figure 18.** RDX for a Human-Robot collaboration task [172]. Two actions (related to robot speed) are compared across four different reward types (on the x-axis of the chart). The y-axis shows the loss and gain for each type of reward when performing the agent's action rather than the other one. The figure is a vertical bar chart with positive bars for Obstacle (orange, about +20), Speed (green, about +18), Goal (red, about +15), and a negative bar for Collision Direction (brown, about -30).

A new estimator proposed in [347] allows the agent to predict the $n$ future expected rewards from a given state. This new estimator makes it possible to explain the agent's choice of action from a given state through three questions: '*What rewards to expect and When?*', '*What observation features are important?*' and '*What is the impact of an action choice?*'. The answers to these questions from a state $s$ are respectively a plot of the $n$ future expected rewards, a use of Grad-CAM [306] to understand the important features for the different future expected rewards and a contrastive explanation by displaying the $n$ future expected rewards for two different actions performed from $s$.

In a human-machine cooperation context, Kampik *et al.* [189] describe different types of goals that the agent could communicate to explain a sympathetic action, i.e. an action that helps the user at the agent's expense. Although not intended for RL, this type of explanation could be useful in a cooperative MARL context, where agents would explain their sympathetic action by describing the expected outcome in terms of reward.

#### 4.2.3 Sequence

The expected outcome of an action here takes the form of a sequence, or a summary of sequences, which the agent can reach by performing an action $a$ from a state $s$.

A belief map is learned together with the agent's Q-values, to give the user the states that the agent is trying to reach by doing $a$ from $s$ [395]. This belief map provides a local representation of the agent's intentions. In the Taxi problem, these expected states form a path, as shown in Figure 19.

The CoViz algorithm [17], already presented in this survey with its policy summary variant, proposes to display to the user two sequences from a state $s$, one in which the action $a$ chosen by the agent is performed, and one in which a different action $a'$ is performed. This enables the user to compare the sequences. In addition, reward decomposition is used.

In [291], the set of possible trajectories from a pair $(s,a)$ is divided into different groups that can be explained in the same way, by training a rule-based classifier. Subsequently, a minimal common explanation is built for each group, which takes the form of an existentially quantified conjunction of literals. In a MDP context, these explanations describe a partial plan made up of actions and/or state features to reach a winning state.

The Scenario eXplanation (SXp) method [299] falls into this category of work. Its objective is to summarise all possible sequences starting with action $a$ from state $s$, by simply providing 3 representative sequences: the best-case, worst-case and most probable scenario.

Luss *et al.* [235] present a method for clustering the state space locally and extracting a set of critical states. Starting from a state $s$, the states to cluster are obtained using the stochastic policy for $x$ steps. Reachable states are grouped into meta-states and strategic states are identified. These states are identified as sub-goals. With the help of visualisation, these sub-goals and clusters are used to understand the agent's future behavior.

**Figure 19.** Explanation of each action in an episode in the Taxi environment [395]. The episode is described on two lines, starting on the left of the first line. In each state, the agent's future positions according to the belief map are displayed by colored cells. The color intensity reflects the agent's confidence in accessing the position (the more intense the colour, the more confident the agent). In this episode, the Taxi picks up the passenger at the position marked with a 'G' and drops him off at the position marked with a 'B'. The figure shows two rows of ASCII-rendered Taxi grids (5x5 cells with cells R, G, Y, B as terminals and a taxi character moving through), with colored highlights (yellow/green) on the future positions in each grid state.

#### 4.2.4 Action

Only one work explains the choice of an action by indicating future actions that could be taken. The work [90] is closely related to the problem studied and so future work could draw on it to explain RL agents (provided that the MDP is known or at least approximated).

Dodson *et al.* [90] provide a natural language explanation for advising students on their choice of courses. This problem, represented by a MDP, is explained using two modules, one of which is called Model-Based Explanation. This module extracts information from the MDP about the usefulness of performing the recommended action in terms of future actions. As an example, if the student takes course $x$, this will enable him to take courses $y, z$ in the future.

The following four works use both types of explanation of action (i.e. *Feature Importance* and *Expected Outcomes*).

The PeCoX framework [260] proposes two types of explanation based on the perceptual and cognitive aspects of the agent. PeCoX models the explanation problem in three parts: explanation generation, communication to the user and reception. The perceptual explanation comprises the calculation of a confidence index allowing the agent to express its confidence in its decision and a contrastive method that identifies an alternative output for comparison. The cognitive explanation consists of providing a restricted set of goals and beliefs for the agent. The authors also suggest modelling emotions to enrich the explanation. Note that the way in which the agent has learned and the explanation methods are not specified in this work, the idea being to provide a general framework for explaining agent behavior.

An approach is presented in [374] for directly extracting explanations based on the POMDP. The idea is to retrieve the various types of information from the POMDP model and translate them into natural language sentences using predefined templates. For example, the agent can use its expected reward outcome to justify its choice of action, the transition function to describe the probability of achieving different outcomes, or its observation to clarify it to the user.

The method used in the work by Ehsan *et al.* [99, 100] consists of collecting a training corpus with the help of users so that the agent can explain its choice of action 'as a human would have done'. So the first step is to collect explanations associated with state-action pairs. Users have to 'think aloud' while performing the agent's task. The second step is to train an encoder-decoder to return, based on a state-action pair, an explanation of the choice of action from the state, in a natural language form.

**RL as an explanation framework** The following section is an aside to the XRL survey. It succinctly presents different works found during our research that provide explanations for AI models using RL.

A body of work uses RL to generate counterfactual inputs to a model that are close to the original input, to answer the question '*Why predict a rather than b from instance $x$?*' In [58], the problem of generating a counterfactual instance of a classification or regression model is described as a decision-making problem, which is solved using a DRL algorithm. Based on the same idea, [297] generates batches of counterfactuals from the instance to be explained. For the Drug Target prediction problem, a multi-agent framework is proposed in [263] to generate counterfactuals by taking into account two distinct inputs, the drug and the target, which are assimilated to two agents with distinct actions. To explain the classification of a point, Lash [210] describes an approach using DRL to find the closest point to the classified point which is located in the decision boundary, based on the user's set of preferred features. The use of two RL agents allows the question '*Why an item is recommended?*' to be answered by a personalised explanation [377]. To explain the choice of the recommendation system $f$, agent 1 learns to provide an explanation consisting of a subset of items and agent 2 learns to use this to predict the output ratings: the explanation is a subset of items sufficient to provide the same output ratings as $f$. Kohler *et al.* [200] use Iterative Bounding MDP (IBMDP) [345] to build compact and efficient DT's for classification tasks. In addition to a classical MDP, IBMDP includes feature bounds, information gathering actions and a reward function that defines the interpretability-performance trade-off. The SXp method [299] learns favorable and hostile agents representing the response of the environment with RL, in order to explain an RL agent.

**Table 6.** Feature Importance works.

| Type | | Refs |
|------|---|------|
| **Saliency Maps (32)** | Attention Mechanism (15) | [256], [171], [265], [229], [268], [311], [339], [186], [30], [394], [392], [376], [379], [408], [165] |
| | Perturbation-based (10) | [131], [275], [136], [94], [280], [388], [164], [173], [218], [18] |
| | Gradient-based (7) | [185], [264], [380], [73], [156], [400], [146] |
| **Model-agnostic Approach (15)** | SHAP (13) | [287], [378], [51], [231], [284], [406], [341], [283], [219], [302], [309], [35], [385] |
| | LIME (3) | [202], [81], [214] |
| Counterfactual State (5) | | [267], [163], [296], [96], [120] |
| Others (2) | | [77], [103] |

**Table 7.** Expected Outcomes works.

| Type | | Refs |
|------|---|------|
| **State (17)** | Causal Lens (6) | [239], [238], [368], [150], [399], [96] |
| | Probability (4) | [67], [68], [198], [90] |
| | Contrastive (4) | [330], [351], [224], [32] |
| | Others (3) | [115], [116], [270] |
| Reward (10) | | [105], [187], [295], [172], [108], [286], [18], [17], [347], [189] |
| Sequence (4) | | [395], [17], [291], [235] |
| Action (1) | | [90] |

## 5 Related domains

**Explainable Planning** Planning proposes a set of methods for providing a plan as a solution to a problem given the initial configuration, the goal configuration and a set of actions (or operators) defined by preconditions and effects. Planning and RL being relatively close domains, explainability methods for planning, grouped under the name XAIP, are of particular interest. We believe that this area should be further explored by XRL researchers. As a first step in this domain, we will briefly describe some XAIP surveys and methods.

A survey and roadmap for XAIP is described by Fox *et al.* [110] in order to obtain more efficient methods. Different questions are posed to guide the search, such as '*Why can't you do that?*' or '*Why do I need to replan at this point?*' in the respective cases where a plan is not found by the algorithm and a failure of the current plan occurs. Chakraborti *et al.* [53] categorise the different XAIP works according to the target of the explanation and the properties of the explanation. The method may explain the planning algorithm, the problem model or the plan. As examples of properties, XAIP methods can use abstractions and/or contrastive approaches. A brief overview is given in [159], with a focus on contrastive explanations. For these explanations, several types of question are interesting to study, such as '*Why action $a$ instead of action $b$?*' or '*Why does the current plan satisfy property $p$ rather than $q$?*'. Cashmore *et al.* [52] describe a framework for answering this type of contrastive questions and the different challenges for effective XAIP methods.

Plan debugging is carried out by finding states $s$ called bugs where there is a difference between the evaluation of the value of $s$ based on the current plan and the optimal plan [331]. This comparison can be performed according to two distinct criteria: a quantitative criterion that considers the cost of the plan, and a qualitative criterion that considers the resolution of the problem. The unsolvability of a planning problem is explained by a hierarchical abstraction of sub-goals [326]. These sub-goals are necessary to solve the problem but are for the most part unreachable. The proposed method consecutively identifies the appropriate level of abstraction for the explanation, the sequence of sub-goals to be achieved and the first unreachable one in the sequence.

The Explaining Robot Action [230] answers a user question in the form of a sentence. To do this, the method determines the information required, selects the template for the answer and sends requests to the world model and the planner to fill in the template.

In a problem where not all goals are feasible, Eifler *et al.* [101] propose two types of explanations based on plan-property entailments. In this work, a plan property $p$, expressed as a Boolean function, entails $q$, which means that all plans satisfied by $p$ are satisfied by $q$. A local explanation for a '*Why not $p$?*' question is to provide a set of undesirable plan properties that would be satisfied by satisfying $p$. A global explanation is a graph that returns all plan-property entailments. This method is used in [102] in an iterative planning context where the user can, via a user interface, ask questions about the different plan-properties.

Seegebarth *et al.* [304] construct an axiomatic system composed of first-order logic formulae and then use it to explain the plan steps and the ordering constraints between plan steps. The explanations take the form of a sequence of applications of the axioms allowing the question about the plan to be answered.

In a problem where the user's mental model differs from that of the agent, the explanation consists of a set of model changes so that the plan is optimal for both the agent and the user [323]. The aim is to provide the user with the minimum number of model changes. This approach provides explanations for an uncertain user model or a several-users model.

RADAR-X [355] is a user interface that allows the user to ask contrastive questions about the plan in an interactive way. The user describes a plan, and the explanation provides a reduced set of information, based on [54], to explain the interest of the plan compared to the one proposed.

Given a set of positive and negative plans, Kim *et al.* [199] generate a set of LTL formulas as an explanation with a Bayesian inference-based method that is robust to noise. In this context, noise corresponds to swapping plans from the two sets. The explanations describe the positive plans and not the negative ones.

In the context of multi-objective probabilistic planning, Sukkerd *et al.* [333, 332] propose to compare an optimal policy with alternative policies based on the different objectives. This contrastive approach takes the form of a verbal explanation where the values of the policies for the different objectives are compared. An algorithm is presented to generate alternative policies that are Pareto optimal according to a certain objective.

Two explanations are described in [98] for a robotic task. This task is solved with a symbolic action planner and a haptic prediction model. The explanation extracted from the symbolic action planner is the symbolic action sequence performed by the robot and the one extracted from the haptic prediction model is a visualisation of the effects of the previous action.

**Model Checking** This domain brings together a set of approaches that analyse a program according to correctness properties (for a survey of the domain, see [180]). Probabilistic (or statistical) model checking is a domain that could be of greater interest to XRL (for a survey, see [5]). It covers a range of methods for checking properties expressed in stochastic temporal logic. These approaches are based on sampling, and return confidence scores associated with the result of the properties studied. In the following, we briefly describe a few works that are more or less close to XRL and that use methods from these domains to provide explanations.

In a MARL context, Boggess *et al.* [45] propose an algorithm that verifies the user request expressed in a probabilistic Computational Tree Logic formula [28] using probabilistic model checking. The user request corresponds to a (partial) plan proposed by the user. The verification is performed on an abstraction of the multi-agent policy, which is updated if the request is not verified. If, despite this, the request is still not verified, the algorithm generates an explanation as to why it has not been respected.

TraceVis [132] makes it possible to combine visualization methods with model checking, carried out with Deep Statistical Model Checking [133]. This approach makes it possible to analyse a policy represented by a NN.

Li *et al.* [215] use a probabilistic model checker to provide a synthesised explanation to the user when necessary in a human-on-the-loop approach. In this work, an explanation is a content, effect and cost triplet.

Model checking and statistical model checking would make it possible to study the agent's policy concerning respecting a set of properties in order to analyse in more detail what the agent has learned. In addition to properties linked to the agent's performance, it would be interesting to consider more varied properties based on, for example, the use of a certain action (or strategy) that is deemed dangerous or costly by the user, or access to certain regions of the state space. Such diversity would help the user decide whether or not to use an agent's policy.

**Algorithmic Recourse** This domain is generally applied to classification or regression models and aims to propose a set of approaches explaining a prediction in a contrastive way. The aim is to answer the question '*Why a rather than b from instance $x$?*' and to provide recommendations for obtaining $b$. These recommendations can be seen as a set of actions to be performed from instance $x$ to reach an instance $x'$ where the output of the model would be $b$. For an overview of the different methods in this domain, we recommend [192].

As an example, the work of Karimi *et al.* [193] presents a formulation of the problem that consists of generating counterfactual instances by minimizing not the change in features of the instance to be explained, but the cost of the actions that lead to a counterfactual instance. This reformulation makes it possible to avoid obtaining counterfactuals whose recommendations are sub-optimal or even infeasible.

Algorithmic recourse seems to be an interesting area to investigate in order to provide XRL methods. Indeed, this approach could be used to explain the choice of one action over another (or the choice of one sequence of actions over another), and provide recommendations for reaching a state (resp. set of states), allowing the contrastive action (resp. sequence of actions) to be performed. This explanation could simply target the understanding of actions or outcomes (e.g. state, reward, respected predicate). Several works come close to this domain by providing *Counterfactual States* [267, 163, 296, 96, 120] or *Counterfactual Sequences* [357, 9, 119, 349, 350, 329] but do not provide recommendations. Properties (e.g. recourse, sparsity) and ideas for counterfactual explanations specific to RL are proposed in [121]. One work has been identified as being directly inspired by the domain: [120] describes a method for providing a counterfactual state reachable by agent actions from the base state.

**RL sub-domains** HRL (cf. Section 2.1.2), RRL and relational MDP (cf. Section 2.3.2) have been the subject of brief domain overviews in this state of the art. These areas bring interpretability by default, hence are interesting to look at in more detail.

Similarly, Causal RL brings together a body of work that combines causality with RL. The aim of this combination is to improve the data efficiency of RL problems while providing interpretability by providing causal relations between states, actions, etc. Although some of the work in this survey falls into this category [239, 238, 368, 150, 399, 96], we recommend the survey [403] which presents a general overview of the advances in this domain by categorising the work into two categories: approaches which must learn the causal information of the environment before the agent is trained, and those which already have this information.

We believe that the domains discussed are of real interest in the future design of XRL methods. We encourage researchers to read the various surveys highlighted [53, 159, 110, 52, 180, 5, 192, 358, 272, 403]. The explainability of agents that have learned by reinforcement is a flourishing domain that requires particular attention on several points, or needs. We will list these needs below, which could also be considered for the XAI domain in general.

## 6 Needs for XRL

**Compare methods** As stated in the surveys [247, 369], it is necessary to compare XRL methods. In this way, for a given target (e.g. action) and for the same type of method (e.g. feature importance), developers could determine which method is best suited to the problem considered. To give a few examples, the works [278, 277, 31, 385] tend to compare explainable methods in different ways.

Pierson *et al.* [277] compare several methods to explain the agent's policy: HIGHLIGHTS [12], a combination of HIGHLIGHTS with a saliency map, graphs summarising the agent's behaviour and a combination of saliency maps and graphs. To do this, a user study is carried out.

In [31], an interesting comparison is made for classifiers using the computational complexity of providing some type of explanation (e.g. returning a minimal set of features of a particular instance sufficient to predict a given class).

A specific methodology for evaluating saliency maps is developed in [26]. This evaluation method is based on interventions applied to the image and allows the authors to conclude that saliency maps should not be used to explain.

Pocius *et al.* [278] provide three tasks, based on the StarCraft II environment [366], for evaluating XRL methods.

In the same vein, a benchmark comprising a set of environments, already implemented XRL methods and metrics is described in [385]. This benchmark focuses on methods that explain the agent's action using a *Feature Importance* approach. The proposed metrics enable the methods to be compared in terms of fidelity, stability and computation time.

**Provide metrics** One of the shortcomings of XRL is the lack of a unified way of comparing methods, whatever their type or target. For a high level overview of metrics used in XRL, we refer the reader to [247]. We believe that a unified assessment of methods is important, but also a more diverse set of metrics for each type of method in order to measure the quality of the explanations. In the following, we present different ways of measuring the quality of explanations and categorizing the evaluation methods in the XAI domain.

Zhou *et al.* [412] present a survey of methods and metrics for evaluating XAI methods. A set of metrics are categorised according to the type of explanation evaluated and the properties of explanations taken into account by the metric (e.g. clarity, parsimony, soundness).

Doshi-Velez and Kim [93] divide the methods for evaluating the interpretability of a model into three parts. Application-grounded evaluation groups together methods that rely on end-task experts to determine its interpretability. For human-grounded evaluations, this is determined using lay persons. Functionally-grounded evaluations are a set of methods that use functions defined by certain properties that act as interpretability metrics, and therefore do not require human experiments.

An overview of evaluation methods for interpretable machine learning models is given in [390]. Three properties are considered in the evaluation methods: generalizability, which determines the extent to which the explanations are specific to the instance, fidelity, which determines the extent to which the explainer's output matches the model's decision-making process, and persuasibility, which determines user satisfaction.

A set of measures is presented in [158] for XAI. These are grouped into four sets: methods that evaluate the quality of explanations, user satisfaction, user understanding of the model and performance according to their mental model.

4 metrics are set out in the blue-sky paper [289]. $D$ quantifies the difference in performance between the agent's opaque model and its surrogate model. $R$ measures the simplicity of the proposed model, $F$ the number of elements useful for generating the explanation and $S$ the stability of the explainer.

Amgoud and Ben-Naim [11] present a total of 10 axioms that explainers should satisfy. These axioms are limited to explainers whose explanation is a subset of the features of an instance. Among the axioms, irreducibility defines that an explainer should only contain useful features and feasibility defines that an explainer is a subset of at least one instance of the classification problem.

Depending on the dataset, Amiri *et al.* [15] propose to generate a ground truth of explanations, which allows comparison with the explanations produced by LIME [285].

In [370], several methods based on the notion of feature importance are compared using an auxiliary task. This task consists of guiding an agent playing Connect4 to win. In this context, the agent has only a partial observation of the board, which is given by a feature importance method. The idea is to measure, over a set of games, whether the agent manages to win based solely on the cells on the board indicated as important by an explainer.

Sokol and Flach [320] present a set of elements to be considered when evaluating XAI methods, which are grouped according to 5 dimensions: functional, operational, usability, safety and validation.

**Perform user studies** Of the various works cited in this survey, a relatively small proportion validate their approach with a user study (which is also noted in [381, 247]). As consumers of explanations of agents' behavior, it makes sense that any XRL method should at least be evaluated by a user study. We believe that taking end-users into consideration in the development process of an explainable method is essential, whether the end-users are domain experts, developers or lay persons. Furthermore, with sufficiently in-depth user studies, it would be possible to identify the strengths and weaknesses of the approach, and then use this feedback to improve the approach. Here is a non-exhaustive list of works described in this state of the art that validate their approach through user studies [308, 120, 12] or simply compare two already existing XRL methods with each other [18, 89, 277].

**Develop user interface** In line with the above need, we believe that it is important to propose, in addition to an XRL method, a dedicated user interface. The construction of a toolbox is suggested as a future direction in [369]. An interface should be intuitive and ergonomic so that the user gets the most out of the explainability method. In user studies, a more or less basic interface is proposed. The various works explaining the agent's policy through Visual Toolkit [250, 178, 373, 147, 245] is a good example of a way of thinking about explainability. In our opinion, it is as important to provide a good XRL method as it is to provide a user interface that improves its usefulness from a user's point of view.

## 7 Conclusion

This paper has categorised recent work in the field of XRL using two questions: '*What?*' and '*How?*'. 3 targets for explanations have been identified, namely the agent's policy, the state-action sequence resulting from the agent's interaction within the environment and the agent's choice of action. Several ways of providing these explanations have been proposed, which can be summarised in three clear ideas. The first idea is to make behavior interpretable by directly influencing its knowledge representation. The second idea is to represent the environment in which the agent evolves in a comprehensible way. The last idea is to integrate methods from outside the RL paradigm to explain the agent's decisions.

This state of the art allows us to highlight the low interest of researchers in explaining sequences of agent interactions: 175 works use methods which are categorised as *Policy-level methods*, 89 as *action-level methods* and only 11 as *sequence-level methods*. We assume that this is due to the fact that XAI for classifiers influences XRL. Indeed, the vast majority of classifier explanation methods simply explain a decision locally or the model globally. We encourage researchers to explain the agent's action choice sequences, to provide better heterogeneity of XRL methods. On the other hand, many methods have been proposed to explain a simple agent decision or policy. Among the different approaches identified, the majority of methods that explain an agent's decision by *Feature Importance* do so using saliency maps, although this approach is limited to agents whose state is an image. For methods based on expected outcomes, the majority return states or features of states as explanations. For policy, the majority of papers fall into the *Interpretable Policy* category, with a predominance of methods proposing directly interpretable policies.

A small proportion of the works presented do not aim at explainability, but rather at performance or policy generalisation. However, we thought it was interesting to include these works, as the methods described make it possible to obtain an interpretable agent or to take an intermediate step between opacity and clarity of the agent's behavior.

This state of the art proposes an intuitive taxonomy that allows us to quickly identify a set of works related to the target we want to explain and the way we want to do it. The different targets are the policy, a sequence of actions and an agent action. This paper has also enabled us to highlight the types of work done to explain or make interpretable one of the targets, to show the use of methods initially intended for the explanation of classifiers in the context of XRL. Moreover, it briefly describes a set of domains that we consider relevant to explore in order to propose new XRL methods and lists several needs for this topic.

## References

[1] Kuruge Darshana Abeyrathna, Ole-Christoffer Granmo, Lei Jiao, and Morten Goodwin. The regression Tsetlin machine: A Tsetlin machine for continuous output problems. In Paulo Moura Oliveira, Paulo Novais, and Luís Paulo Reis, editors, *Progress in Artificial Intelligence, 19th EPIA Conference on Artificial Intelligence, EPIA 2019, Vila Real, Portugal, September 3-6, 2019, Proceedings, Part II*, volume 11805 of *Lecture Notes in Computer Science*, pages 268–280. Springer, 2019.

[2] Aastha Acharya, Rebecca L. Russell, and Nisar R. Ahmed. Explaining conditions for reinforcement learning behaviors from real and imagined data. *CoRR*, abs/2011.09004, 2020.

[3] Julius Adebayo, Justin Gilmer, Michael Muelly, Ian J. Goodfellow, Moritz Hardt, and Been Kim. Sanity checks for saliency maps. In Samy Bengio, Hanna M. Wallach, Hugo Larochelle, Kristen Grauman, Nicolò Cesa-Bianchi, and Roman Garnett, editors, *Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montréal, Canada*, pages 9525–9536, 2018.

[4] Dhaval Adjodah, Tim Klinger, and Joshua Joseph. Symbolic relation networks for reinforcement learning. In *Proceedings of the Workshop on Relational Representation Learning in Conference on Neural Information Processing Systems (NeurIPS)*, 2018.

[5] Gul Agha and Karl Palmskog. A survey of statistical model checking. *ACM Trans. Model. Comput. Simul.*, 28(1):6:1–6:39, 2018.

[6] William Agnew and Pedro Domingos. Unsupervised object-level deep reinforcement learning. In *NeurIPS Workshop on Deep RL*, 2018.

[7] Riad Akrour, Davide Tateo, and Jan Peters. Towards reinforcement learning of human readable policies. In *The European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases: The 1st Workshop on Deep Continuous-Discrete Machine Learning*, 2019.

[8] Maxime Alaarabiou, Nicolas Delestre, and Laurent Vercouter. Explicabilité en apprentissage par renforcement: vers une taxinomie unifiée. *JIAF-JFPDA*, page 90, 2024.

[9] Amal Alabdulkarim and Mark O. Riedl. Experiential explanations for reinforcement learning. *CoRR*, abs/2210.04723, 2022.

[10] Philipp Altmann, Céline Davignon, Maximilian Zorn, Fabian Ritz, Claudia Linnhoff-Popien, and Thomas Gabor. REACT: revealing evolutionary action consequence trajectories for interpretable reinforcement learning. *CoRR*, abs/2404.03359, 2024.

[11] Leila Amgoud and Jonathan Ben-Naim. Axiomatic foundations of explainability. In Luc De Raedt, editor, *Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI 2022, Vienna, Austria, 23-29 July 2022*, pages 636–642. ijcai.org, 2022.

[12] Dan Amir and Ofra Amir. HIGHLIGHTS: summarizing agent behavior to people. In Elisabeth André, Sven Koenig, Mehdi Dastani, and Gita Sukthankar, editors, *Proceedings of the 17th International Conference on Autonomous Agents and MultiAgent Systems, AAMAS*, pages 1168–1176. International Foundation for Autonomous Agents and Multiagent Systems / ACM, 2018.

[13] Ofra Amir, Finale Doshi-Velez, and David Sarne. Agent strategy summarization. In Elisabeth André, Sven Koenig, Mehdi Dastani, and Gita Sukthankar, editors, *Proceedings of the 17th International Conference on Autonomous Agents and MultiAgent Systems, AAMAS 2018, Stockholm, Sweden, July 10-15, 2018*, pages 1203–1207. International Foundation for Autonomous Agents and Multiagent Systems Richland, SC, USA / ACM, 2018.

[14] Ofra Amir, Finale Doshi-Velez, and David Sarne. Summarizing agent strategies. *Auton. Agents Multi Agent Syst.*, 33(5):628–644, 2019.

[15] Shideh Shams Amiri, Rosina O. Weber, Prateek Goel, Owen Brooks, Archer Gandley, Brian Kitchell, and Aaron Zehm. Data representing ground-truth explanations to evaluate XAI methods. *CoRR*, abs/2011.09892, 2020.

[16] Yotam Amitai and Ofra Amir. "I don't think so": Summarizing policy disagreements for agent comparison. In *Thirty-Sixth AAAI Conference on Artificial Intelligence, AAAI 2022, Thirty-Fourth Conference on Innovative Applications of Artificial Intelligence, IAAI 2022, The Twelveth Symposium on Educational Advances in Artificial Intelligence, EAAI 2022 Virtual Event, February 22 - March 1, 2022*, pages 5269–5276. AAAI Press, 2022.

[17] Yotam Amitai, Yael Septon, and Ofra Amir. Explaining reinforcement learning agents through counterfactual action outcomes. In Michael J. Wooldridge, Jennifer G. Dy, and Sriraam Natarajan, editors, *Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2014, February 20-27, 2024, Vancouver, Canada*, pages 10003–10011. AAAI Press, 2024.

[18] Andrew Anderson, Jonathan Dodge, Amrita Sadarangani, Zoe Juozapaitis, Evan Newman, Jed Irvine, Souti Chattopadhyay, Matthew L. Olson, Alan Fern, and Margaret Burnett. Mental models of mere mortals with explanations of reinforcement learning. *ACM Trans. Interact. Intell. Syst.*, 10(2):15:1–15:37, 2020.

[19] Jacob Andreas, Dan Klein, and Sergey Levine. Modular multitask reinforcement learning with policy sketches. In Doina Precup and Yee Whye Teh, editors, *Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017*, volume 70 of *Proceedings of Machine Learning Research*, pages 166–175. PMLR, 2017.

[20] Giorgio Angelotti and Natalia Díaz-Rodríguez. Towards a more efficient computation of individual attribute and policy contribution for post-hoc explanation of cooperative multi-agent systems using Myerson values. *Knowledge-Based Systems*, 260:110189, 2023.

[21] Plamen P. Angelov and Dimitar P. Filev. An approach to online identification of Takagi-Sugeno fuzzy models. *IEEE Trans. Syst. Man Cybern. Part B*, 34(1):484–498, 2004.

[22] Raghuram Mandyam Annasamy and Katia P. Sycara. Towards better interpretability in deep Q-networks. In *The Thirty-Third AAAI Conference on Artificial Intelligence, AAAI 2019, The Thirty-First Innovative Applications of Artificial Intelligence Conference, IAAI 2019, The Ninth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2019, Honolulu, Hawaii, USA, January 27 - February 1, 2019*, pages 4561–4569. AAAI Press, 2019.

[23] Jose A. Arjona-Medina, Michael Gillhofer, Michael Widrich, Thomas Unterthiner, Johannes Brandstetter, and Sepp Hochreiter. RUDDER: return decomposition for delayed rewards. In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence d'Alché-Buc, Emily B. Fox, and Roman Garnett, editors, *Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada*, pages 13544–13555, 2019.

[24] Ignacio Arnaldo, Una-May O'Reilly, and Kalyan Veeramachaneni. Building predictive models via feature synthesis. In Sara Silva and Anna Isabel Esparcia-Alcázar, editors, *Proceedings of the Genetic and Evolutionary Computation Conference, GECCO 2015, Madrid, Spain, July 11-15, 2015*, pages 983–990. ACM, 2015.

[25] Karl Johan Åström. Optimal control of Markov processes with incomplete state information I. *Journal of mathematical analysis and applications*, 10:174–205, 1965.

[26] Akanksha Atrey, Kaleigh Clary, and David D. Jensen. Exploratory not explanatory: Counterfactual analysis of saliency maps for deep reinforcement learning. In *8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020*. OpenReview.net, 2020.

[27] James Ault, Josiah P. Hanna, and Guni Sharon. Learning an interpretable traffic signal control policy. In Amal El Fallah Seghrouchni, Gita Sukthankar, Bo An, and Neil Yorke-Smith, editors, *Proceedings of the 19th International Conference on Autonomous Agents and Multiagent Systems, AAMAS '20, Auckland, New Zealand, May 9-13, 2020*, pages 88–96. International Foundation for Autonomous Agents and Multiagent Systems, 2020.

[28] Adnan Aziz, Vigyan Singhal, and Felice Balarin. It usually works: The temporal logic of stochastic systems. In Pierre Wolper, editor, *Computer Aided Verification, 7th International Conference, Liège, Belgium, July, 3-5, 1995, Proceedings*, volume 939 of *Lecture Notes in Computer Science*, pages 155–165. Springer, 1995.

[29] Sebastian Bach, Alexander Binder, Grégoire Montavon, Frederick Klauschen, Klaus-Robert Müller, and Wojciech Samek. On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation. *PloS one*, 10(7):e0130140, 2015.

[30] Wentao Bao, Qi Yu, and Yu Kong. DRIVE: deep reinforced accident anticipation with visual explanation. In *2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021*, pages 7599–7608. IEEE, 2021.

[31] Pablo Barceló, Mikaël Monet, Jorge Pérez, and Bernardo Subercaseaux. Model interpretability through the lens of computational complexity. In Hugo Larochelle, Marc'Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, editors, *Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual*, 2020.

[32] Pablo V. A. Barros, Ana Tanevska, Francisco Cruz, and Alessandra Sciutti. Moody learners - explaining competitive behaviour of reinforcement learning agents. In *Joint IEEE 10th International Conference on Development and Learning and Epigenetic Robotics, ICDL-EpiRob 2020, Valparaiso, Chile, October 26-30, 2020*, pages 1–8. IEEE, 2020.

[33] Osbert Bastani, Jeevana Priya Inala, and Armando Solar-Lezama. Interpretable, verifiable, and robust reinforcement learning via program synthesis. In Andreas Holzinger, Randy Goebel, Ruth Fong, Taesup Moon, Klaus-Robert Müller, and Wojciech Samek, editors, *xxAI - Beyond Explainable AI - International Workshop, Held in Conjunction with ICML 2020, July 18, 2020, Vienna, Austria, Revised and Extended Papers*, volume 13200 of *Lecture Notes in Computer Science*, pages 207–228. Springer, 2020.

[34] Osbert Bastani, Yewen Pu, and Armando Solar-Lezama. Verifiable reinforcement learning via policy extraction. In Samy Bengio, Hanna M. Wallach, Hugo Larochelle, Kristen Grauman, Nicolò Cesa-Bianchi, and Roman Garnett, editors, *Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montréal, Canada*, pages 2499–2509, 2018.

[35] Daniel Beechey, Thomas M. S. Smith, and Özgür Simsek. Explaining reinforcement learning with shapley values. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, *International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA*, volume 202 of *Proceedings of Machine Learning Research*, pages 2003–2014. PMLR, 2023.

[36] Yanzhe Bekkemoen and Helge Langseth. ASAP: attention-based state space abstraction for policy summarization. In Berrin Yanikoglu and Wray L. Buntine, editors, *Asian Conference on Machine Learning, ACML 2023, 11-14 November 2023, Istanbul, Turkey*, volume 222 of *Proceedings of Machine Learning Research*, pages 137–152. PMLR, 2023.

[37] Marc G. Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling. The arcade learning environment: An evaluation platform for general agents. *J. Artif. Intell. Res.*, 47:253–279, 2013.

[38] Nir Ben-Zrihem, Tom Zahavy, and Shie Mannor. Visualizing dynamics: from t-SNE to SEMI-MDPs. *CoRR*, abs/1606.07112, 2016.

[39] Dimitri P Bertsekas. Approximate policy iteration: A survey and some new methods. *Journal of Control Theory and Applications*, 9(3):310–335, 2011.

[40] Tom Bewley and Jonathan Lawry. Tripletree: A versatile interpretable representation of black box agents and their environments. In *Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, Thirty-Third Conference on Innovative Applications of Artificial Intelligence, IAAI 2021, The Eleventh Symposium on Educational Advances in Artificial Intelligence, EAAI 2021, Virtual Event, February 2-9, 2021*, pages 11415–11422. AAAI Press, 2021.

[41] Tom Bewley, Jonathan Lawry, and Arthur Richards. Summarising and comparing agent dynamics with contrastive spatiotemporal abstraction. *CoRR*, abs/2201.07749, 2022.

[42] Tom Bewley and Freddy Lécué. Interpretable preference-based reinforcement learning with tree-structured reward functions. In Piotr Faliszewski, Viviana Mascardi, Catherine Pelachaud, and Matthew E. Taylor, editors, *21st International Conference on Autonomous Agents and Multiagent Systems, AAMAS 2022, Auckland, New Zealand, May 9-13, 2022*, pages 118–126. International Foundation for Autonomous Agents and Multiagent Systems (IFAAMAS), 2022.

[43] Benjamin Beyret, Ali Shafti, and A. Aldo Faisal. Dot-to-dot: Explainable hierarchical reinforcement learning for robotic manipulation. In *2019 IEEE/RSJ International Conference on Intelligent Robots and Systems, IROS 2019, Macau, SAR, China, November 3-8, 2019*, pages 5014–5019. IEEE, 2019.

[44] Ioana Bica, Daniel Jarrett, Alihan Hüyük, and Mihaela van der Schaar. Learning "what-if" explanations for sequential decision-making. In *9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021*. OpenReview.net, 2021.

[45] Kayla Boggess, Sarit Kraus, and Lu Feng. Explainable multi-agent reinforcement learning for temporal queries. In *Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, IJCAI 2023, 19th-25th August 2023, Macao, SAR, China*, pages 55–63. ijcai.org, 2023.

[46] Gisela Böhm and Hans-Rüdiger Pfister. How people explain their own and others' behavior: a theory of lay causal explanations. *Frontiers in psychology*, 6:109763, 2015.

[47] Serena Booth, Christian Muise, and Julie Shah. Evaluating the interpretability of the knowledge compilation map: Communicating logical statements effectively. In Sarit Kraus, editor, *Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, IJCAI 2019, Macao, China, August 10-16, 2019*, pages 5801–5807. ijcai.org, 2019.

[48] Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. OpenAI gym, 2016.

[49] Michael Burke, Svetlin Penkov, and Subramanian Ramamoorthy. From explanation to synthesis: Compositional program induction for learning from demonstration. In Antonio Bicchi, Hadas Kress-Gazit, and Seth Hutchinson, editors, *Robotics: Science and Systems XV, University of Freiburg, Freiburg im Breisgau, Germany, June 22-26, 2019*, 2019.

[50] Alberto Camacho, Rodrigo Toro Icarte, Toryn Q. Klassen, Richard Anthony Valenzano, and Sheila A. McIlraith. LTL and beyond: Formal languages for reward function specification in reinforcement learning. In Sarit Kraus, editor, *Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, IJCAI 2019, Macao, China, August 10-16, 2019*, pages 6065–6073. ijcai.org, 2019.

[51] Nicolas Blystad Carbone. Explainable AI for path following with model trees. Master's thesis, NTNU, 2020.

[52] Michael Cashmore, Anna Collins, Benjamin Krarup, Senka Krivic, Daniele Magazzeni, and David E. Smith. Towards explainable AI planning as a service. *CoRR*, abs/1908.05059, 2019.

[53] Tathagata Chakraborti, Sarath Sreedharan, and Subbarao Kambhampati. The emerging landscape of explainable automated planning & decision making. In Christian Bessiere, editor, *Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI 2020*, pages 4803–4811. ijcai.org, 2020.

[54] Tathagata Chakraborti, Sarath Sreedharan, Yu Zhang, and Subbarao Kambhampati. Plan explanations as model reconciliation: Moving beyond explanation as soliloquy. In Carles Sierra, editor, *Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence, IJCAI 2017, Melbourne, Australia, August 19-25, 2017*, pages 156–163. ijcai.org, 2017.

[55] Wang Chang, WU Lizhen, YAN Chao, WANG Zhichao, LONG Han, YU Chao. Coactive design of explainable agent-based task planning and deep reinforcement learning for human-UAVs teamwork. *Chinese Journal of Aeronautics*, 33(11):2930–2945, 2020.

[56] Jianyu Chen, Shengbo Eben Li, and Masayoshi Tomizuka. Interpretable end-to-end urban autonomous driving with latent deep reinforcement learning. *IEEE Trans. Intell. Transp. Syst.*, 23(6):5068–5078, 2022.

[57] Tianqi Chen and Carlos Guestrin. Xgboost: A scalable tree boosting system. In Balaji Krishnapuram, Mohak Shah, Alexander J. Smola, Charu C. Aggarwal, Dou Shen, and Rajeev Rastogi, editors, *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, San Francisco, CA, USA, August 13-17, 2016*, pages 785–794. ACM, 2016.

[58] Ziheng Chen, Fabrizio Silvestri, Jia Wang, He Zhu, Hongshik Ahn, and Gabriele Tolomei. Relax: Reinforcement learning agent explainer for arbitrary predictive models. In Mohammad Al Hasan and Li Xiong, editors, *Proceedings of the 31st ACM International Conference on Information & Knowledge Management, Atlanta, GA, USA, October 17-21, 2022*, pages 252–261. ACM, 2022.

[59] Yunjey Choi, Min-Je Choi, Munyoung Kim, Jung-Woo Ha, Sunghun Kim, and Jaegul Choo. Stargan: Unified generative adversarial networks for multi-domain image-to-image translation. In *2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018*, pages 8789–8797. Computer Vision Foundation / IEEE Computer Society, 2018.

[60] Pawel Cichosz and Lukasz Pawelczak. Imitation learning of car driving skills with decision trees and random forests. *Int. J. Appl. Math. Comput. Sci.*, 24(3):579–597, 2014.

[61] Antoni Climent, Dmitry Gnatyshak, and Sergio Álvarez-Napagao. Applying and verifying an explainability method based on policy graphs in the context of reinforcement learning. In Mateu Villaret, Teresa Alsinet, Cèsar Fernández, and Aïda Valls, editors, *Artificial Intelligence Research and Development - Proceedings of the 23rd International Conference of the Catalan Association for Artificial Intelligence, CCIA 2021, Virtual Event, 20-22 October, 2021*, volume 339 of *Frontiers in Artificial Intelligence and Applications*, pages 455–464. IOS Press, 2021.

[62] Jeffery Allen Clouse. *On integrating apprentice learning and reinforcement learning*. University of Massachusetts Amherst, 1996.

[63] Karl Cobbe, Oleg Klimov, Christopher Hesse, Taehoon Kim, and John Schulman. Quantifying generalization in reinforcement learning. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, *Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA*, volume 97 of *Proceedings of Machine Learning Research*, pages 1282–1289. PMLR, 2019.

[64] Joshua Cole, JW Lloyd, and Kee Siong Ng. Symbolic learning for adaptive agents. In *Proceedings of the Annual Partner Conference, Smart Internet Technology Cooperative Research Centre*, 2003.

[65] Youri Coppens, Kyriakos Efthymiadis, Tom Lenaerts, Ann Nowé, Tim Miller, Rosina Weber, and Daniele Magazzeni. Distilling deep reinforcement learning policies in soft decision trees. In *Proceedings of the IJCAI 2019 workshop on explainable artificial intelligence*, pages 1–6, 2019.

[66] Paul T Costa Jr and Robert R McCrae. Mood and personality in adulthood. In *Handbook of emotion, adult development, and aging*, pages 369–383. Elsevier, 1996.

[67] Francisco Cruz, Richard Dazeley, and Peter Vamplew. Memory-based explainable reinforcement learning. In Jixue Liu and James Bailey, editors, *AI 2019: Advances in Artificial Intelligence - 32nd Australasian Joint Conference, Adelaide, SA, Australia, December 2-5, 2019, Proceedings*, volume 11919 of *Lecture Notes in Computer Science*, pages 66–77. Springer, 2019.

[68] Francisco Cruz, Richard Dazeley, Peter Vamplew, and Ithan Moreira. Explainable robotic systems: understanding goal-driven actions in a reinforcement learning scenario. *Neural Comput. Appl.*, 35(25):18113–18130, 2023.

[69] Leonardo Lucio Custode and Giovanni Iacca. Evolutionary learning of interpretable decision trees. *IEEE Access*, 11:6169–6184, 2023.

[70] Yinglong Dai, Haibin Ouyang, Hong Zheng, Han Long, and Xiaojun Duan. Interpreting a deep reinforcement learning model with conceptual embedding and performance analysis. *Appl. Intell.*, 53(6):6936–6952, 2023.

[71] Yuxin Dai, Qimei Chen, Jun Zhang, Xiaohui Wang, Yilin Chen, Tianlu Gao, Peidong Xu, Siyuan Chen, Siyang Liao, Huaiguang Jiang, et al. Enhanced oblique decision tree enabled policy extraction for deep reinforcement learning in power system emergency control. *Electric Power Systems Research*, 209:107932, 2022.

[72] Mohamad H. Danesh, Anurag Koul, Alan Fern, and Saeed Khorram. Re-understanding finite-state representations of recurrent policy networks. In Marina Meila and Tong Zhang, editors, *Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event*, volume 139 of *Proceedings of Machine Learning Research*, pages 2388–2397. PMLR, 2021.

[73] Giang Dao, Wesley Houston Huff, and Minwoo Lee. Learning sparse evidence-driven interpretation to understand deep reinforcement learning agents. In *IEEE Symposium Series on Computational Intelligence, SSCI 2021, Orlando, FL, USA, December 5-7, 2021*, pages 1–7. IEEE, 2021.

[74] Giang Dao, Indrajeet Mishra, and Minwoo Lee. Deep reinforcement learning monitor for snapshot recording. In M. Arif Wani, Mehmed M. Kantardzic, Moamar Sayed Mouchaweh, João Gama, and Edwin Lughofer, editors, *17th IEEE International Conference on Machine Learning and Applications, ICMLA 2018, Orlando, FL, USA, December 17-20, 2018*, pages 591–598. IEEE, 2018.

[75] Srijita Das, Sriraam Natarajan, Kaushik Roy, Ronald Parr, and Kristian Kersting. Fitted Q-learning for relational domains. *CoRR*, abs/2006.05595, 2020.

[76] Artur S. d'Avila Garcez, Aimore Resende Riquetti Dutra, and Eduardo Alonso. Towards symbolic reinforcement learning with common sense. *CoRR*, abs/1804.08597, 2018.

[77] Omid Davoodi and Majid Komeili. Feature-based interpretable reinforcement learning based on state-transition models. In *2021 IEEE International Conference on Systems, Man, and Cybernetics, SMC 2021, Melbourne, Australia, October 17-20, 2021*, pages 301–308. IEEE, 2021.

[78] Richard Dazeley, Peter Vamplew, and Francisco Cruz. Explainable reinforcement learning for broad-XAI: a conceptual framework and survey. *Neural Comput. Appl.*, 35(23):16893–16916, 2023.

[79] Leonardo Mendonça de Moura and Nikolaj S. Bjørner. Z3: an efficient SMT solver. In C. R. Ramakrishnan and Jakob Rehof, editors, *Tools and Algorithms for the Construction and Analysis of Systems, 14th International Conference, TACAS 2008, Held as Part of the Joint European Conferences on Theory and Practice of Software, ETAPS 2008, Budapest, Hungary, March 29-April 6, 2008. Proceedings*, volume 4963 of *Lecture Notes in Computer Science*, pages 337–340. Springer, 2008.

[80] Thomas Degris, Olivier Sigaud, and Pierre-Henri Wuillemin. Learning the structure of factored Markov decision processes in reinforcement learning problems. In William W. Cohen and Andrew W. Moore, editors, *Machine Learning, Proceedings of the Twenty-Third International Conference (ICML 2006), Pittsburgh, Pennsylvania, USA, June 25-29, 2006*, volume 148 of *ACM International Conference Proceeding Series*, pages 257–264. ACM, 2006.

[81] Arnaud Dethise, Marco Chiesa, and Srikanth Kandula. Cracking open the black box: What observations can tell us about reinforcement learning agents. In *Proceedings of the 2019 Workshop on Network Meets AI & ML, NetAI@SIGCOMM 2019, Beijing, China, August 23, 2019*, pages 29–36. ACM, 2019.

[82] Rati Devidze, Goran Radanovic, Parameswaran Kamalaruban, and Adish Singla. Explicable reward design for reinforcement learning agents. In Marc'Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan, editors, *Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual*, pages 20118–20131, 2021.

[83] Yashesh D. Dhebar and Kalyanmoy Deb. Interpretable rule discovery through bilevel optimization of split-rules of nonlinear decision trees for classification problems. *IEEE Trans. Cybern.*, 51(11):5573–5584, 2021.

[84] Yashesh D. Dhebar, Kalyanmoy Deb, Subramanya Nageshrao, Ling Zhu, and Dimitar P. Filev. Interpretable-AI policies using evolutionary nonlinear decision trees for discrete action systems. *CoRR*, abs/2009.09521, 2020.

[85] Edsger W. Dijkstra. A note on two problems in connexion with graphs. *Numerische Mathematik*, 1:269–271, 1959.

[86] Zihan Ding, Pablo Hernandez-Leal, Gavin Weiguang Ding, Changjian Li, and Ruitong Huang. CDT: cascading decision trees for explainable reinforcement learning. *CoRR*, abs/2011.07553, 2020.

[87] Marius-Constantin Dinu, Markus Hofmarcher, Vihang Prakash Patil, Matthias Dorfer, Patrick M. Blies, Johannes Brandstetter, Jose A. Arjona-Medina, and Sepp Hochreiter. XAI and strategy extraction via reward redistribution. In Andreas Holzinger, Randy Goebel, Ruth Fong, Taesup Moon, Klaus-Robert Müller, and Wojciech Samek, editors, *xxAI - Beyond Explainable AI - International Workshop, Held in Conjunction with ICML 2020, July 18, 2020, Vienna, Austria, Revised and Extended Papers*, volume 13200 of *Lecture Notes in Computer Science*, pages 177–205. Springer, 2020.

[88] Carlos Diuk, Andre Cohen, and Michael L. Littman. An object-oriented representation for efficient reinforcement learning. In William W. Cohen, Andrew McCallum, and Sam T. Roweis, editors, *Machine Learning, Proceedings of the Twenty-Fifth International Conference (ICML 2008), Helsinki, Finland, June 5-9, 2008*, volume 307 of *ACM International Conference Proceeding Series*, pages 240–247. ACM, 2008.

[89] Jonathan Dodge, Andrew Anderson, Roli Khanna, Jed Irvine, Rupika Dikkala, Kin-Ho Lam, Delyar Tabatabai, Anita Ruangrotsakun, Zeyad Shureih, Min-suk Kahng, et al. From "no clear winner" to an effective explainable artificial intelligence process: An empirical journey. *Applied AI Letters*, 2(4):e36, 2021.

[90] Thomas Dodson, Nicholas Mattei, and Judy Goldsmith. A natural language argumentation interface for explanation generation in Markov decision processes. In Ronen I. Brafman, Fred S. Roberts, and Alexis Tsoukiàs, editors, *Algorithmic Decision Theory - Second International Conference, ADT 2011, Piscataway, NJ, USA, October 26-28, 2011. Proceedings*, volume 6992 of *Lecture Notes in Computer Science*, pages 42–55. Springer, 2011.

[91] Marc Domenech i Vila, Dmitry Gnatyshak, Adrian Tormos, Victor Gimenez-Abalos, and Sergio Alvarez-Napagao. Explaining the behaviour of reinforcement learning agents in a multi-agent cooperative environment using policy graphs. *Electronics*, 13(3):573, 2024.

[92] Honghua Dong, Jiayuan Mao, Tian Lin, Chong Wang, Lihong Li, and Denny Zhou. Neural logic machines. *CoRR*, abs/1904.11694, 2019.

[93] Finale Doshi-Velez and Been Kim. Towards a rigorous science of interpretable machine learning, 2017.

[94] Nathan Douglas, Dianna Yim, Bilal Kartal, Pablo Hernandez-Leal, Frank Maurer, and Matthew E. Taylor. Towers of saliency: A reinforcement learning visualization using immersive environments. In Bongshin Lee, Geehyuk Lee, Stacey D. Scott, Melanie Tory, and Jeonghyun Kim, editors, *Proceedings of the 2019 ACM International Conference on Interactive Surfaces and Spaces, ISS 2019, Daejeon, South Korea, November 10-13, 2019*, pages 339–342. ACM, 2019.

[95] Kurt Driessens and Hendrik Blockeel. Learning digger using hierarchical reinforcement learning for concurrent goals. In *Proceedings of the European Workshop on Reinforcement Learning*, pages 11–12. CKI Utrecht University, 2001.

[96] Jeff Druce, Michael Harradon, and James Tittle. Explainable artificial intelligence (XAI) for increasing user trust in deep reinforcement learning driven autonomous systems. *CoRR*, abs/2106.03775, 2021.

[97] Saso Dzeroski, Luc De Raedt, and Kurt Driessens. Relational reinforcement learning. *Mach. Learn.*, 43(1/2):7–52, 2001.

[98] Mark Edmonds, Feng Gao, Hangxin Liu, Xu Xie, Siyuan Qi, Brandon Rothrock, Yixin Zhu, Ying Nian Wu, Hongjing Lu, and Song-Chun Zhu. A tale of two explanations: Enhancing human trust by explaining robot behavior. *Sci. Robotics*, 4(37), 2019.

[99] Upol Ehsan, Brent Harrison, Larry Chan, and Mark O. Riedl. Rationalization: A neural machine translation approach to generating natural language explanations. In Jason Furman, Gary E. Marchant, Huw Price, and Francesca Rossi, editors, *Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society, AIES 2018, New Orleans, LA, USA, February 02-03, 2018*, pages 81–87. ACM, 2018.

[100] Upol Ehsan, Pradyumna Tambwekar, Larry Chan, Brent Harrison, and Mark O. Riedl. Automated rationale generation: a technique for explainable AI and its effects on human perceptions. In Wai-Tat Fu, Shimei Pan, Oliver Brdiczka, Polo Chau, and Gaelle Calvary, editors, *Proceedings of the 24th International Conference on Intelligent User Interfaces, IUI 2019, Marina del Ray, CA, USA, March 17-20, 2019*, pages 263–274. ACM, 2019.

[101] Rebecca Eifler, Michael Cashmore, Jörg Hoffmann, Daniele Magazzeni, and Marcel Steinmetz. A new approach to plan-space explanation: Analyzing plan-property dependencies in oversubscription planning. In *The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020*, pages 9818–9826. AAAI Press, 2020.

[102] Rebecca Eifler and Jörg Hoffmann. Iterative planning with plan-space explanations: A tool and user study. *CoRR*, abs/2011.09705, 2020.

[103] Francisco Elizalde, Luis Enrique Sucar, Julieta Noguez, and Alberto Reyes. Generating explanations based on Markov decision processes. In Arturo Hernández Aguirre, Raúl Monroy Borja, and Carlos A. Reyes García, editors, *MICAI 2009: Advances in Artificial Intelligence, 8th Mexican International Conference on Artificial Intelligence, Guanajuato, Mexico, November 9-13, 2009. Proceedings*, volume 5845 of *Lecture Notes in Computer Science*, pages 51–62. Springer, 2009.

[104] Damien Ernst, Pierre Geurts, and Louis Wehenkel. Tree-based batch mode reinforcement learning. *J. Mach. Learn. Res.*, 6:503–556, 2005.

[105] Martin Erwig, Alan Fern, Magesh Murali, and Anurag Koul. Explaining deep adaptive programs via reward decomposition. In *IJCAI/ECAI workshop on explainable artificial intelligence*, 2018.

[106] Richard Evans and Edward Grefenstette. Learning explanatory rules from noisy data. *J. Artif. Intell. Res.*, 61:1–64, 2018.

[107] Ben Eysenbach, Ruslan Salakhutdinov, and Sergey Levine. Search on the replay buffer: Bridging planning and reinforcement learning. In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence d'Alché-Buc, Emily B. Fox, and Roman Garnett, editors, *Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada*, pages 15220–15231, 2019.

[108] Felix Feit, Andreas Metzger, and Klaus Pohl. Explaining online reinforcement learning decisions of self-adaptive systems. In Roberto Casadei, Elisabetta Di Nitto, Ilias Gerostathopoulos, Danilo Pianini, Ivana Dusparic, Timothy Wood, Phyllis R. Nelson, Evangelos Pournaras, Nelly Bencomo, Sebastian Götz, Christian Krupitzer, and Claudia Raibulet, editors, *IEEE International Conference on Autonomic Computing and Self-Organizing Systems, ACSOS 2022, Virtual, CA, USA, September 19-23, 2022*, pages 51–60. IEEE, 2022.

[109] Mira Finkelstein, Lucy Liu, Yoav Kolumbus, David C. Parkes, Jeffrey S. Rosenschein, and Sarah Keren. Reinforcement learning explainability via model transforms (student abstract). In *Thirty-Sixth AAAI Conference on Artificial Intelligence, AAAI 2022, Thirty-Fourth Conference on Innovative Applications of Artificial Intelligence, IAAI 2022, The Twelveth Symposium on Educational Advances in Artificial Intelligence, EAAI 2022 Virtual Event, February 22 - March 1, 2022*, pages 12943–12944. AAAI Press, 2022.

[110] Maria Fox, Derek Long, and Daniele Magazzeni. Explainable planning. *CoRR*, abs/1709.10256, 2017.

[111] Vincent François-Lavet, Yoshua Bengio, Doina Precup, and Joelle Pineau. Combined reinforcement learning via abstract representations. In *The Thirty-Third AAAI Conference on Artificial Intelligence, AAAI 2019, The Thirty-First Innovative Applications of Artificial Intelligence Conference, IAAI 2019, The Ninth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2019, Honolulu, Hawaii, USA, January 27 - February 1, 2019*, pages 3582–3589. AAAI Press, 2019.

[112] Nicholas Frosst and Geoffrey E. Hinton. Distilling a neural network into a soft decision tree. In Tarek R. Besold and Oliver Kutz, editors, *Proceedings of the First International Workshop on Comprehensibility and Explanation in AI and ML 2017 co-located with 16th International Conference of the Italian Association for Artificial Intelligence (AI*IA 2017), Bari, Italy, November 16th and 17th, 2017*, volume 2071 of *CEUR Workshop Proceedings*. CEUR-WS.org, 2017.

[113] Julius Frost, Olivia Watkins, Eric Weiner, Pieter Abbeel, Trevor Darrell, Bryan A. Plummer, and Kate Saenko. Explaining reinforcement learning policies through counterfactual trajectories. *CoRR*, abs/2201.12462, 2022.

[114] Justin Fu, Katie Luo, and Sergey Levine. Learning robust rewards with adversarial inverse reinforcement learning. *CoRR*, abs/1710.11248, 2017.

[115] Yosuke Fukuchi, Masahiko Osawa, Hiroshi Yamakawa, and Michita Imai. Application of instruction-based behavior explanation to a reinforcement learning agent with changing policy. In Derong Liu, Shengli Xie, Yuanqing Li, Dongbin Zhao, and El-Sayed M. El-Alfy, editors, *Neural Information Processing - 24th International Conference, ICONIP 2017, Guangzhou, China, November 14-18, 2017, Proceedings, Part I*, volume 10634 of *Lecture Notes in Computer Science*, pages 100–108. Springer, 2017.

[116] Yosuke Fukuchi, Masahiko Osawa, Hiroshi Yamakawa, and Michita Imai. Autonomous self-explanation of behavior for interactive reinforcement learning agents. In Britta Wrede, Yukie Nagai, Takanori Komatsu, Marc Hanheide, and Lorenzo Natale, editors, *Proceedings of the 5th International Conference on Human Agent Interaction, HAI 2017, Bielefeld, Germany, October 17 - 20, 2017*, pages 97–101. ACM, 2017.

[117] Daniel Furelos-Blanco, Mark Law, Anders Jonsson, Krysia Broda, and Alessandra Russo. Induction and exploitation of subgoal automata for reinforcement learning. *J. Artif. Intell. Res.*, 70:1031–1116, 2021.

[118] Jasmina Gajcin and Ivana Dusparic. ReCCoVER: Detecting causal confusion for explainable reinforcement learning. In Davide Calvaresi, Amro Najjar, Michael Winikoff, and Kary Främling, editors, *Explainable and Transparent AI and Multi-Agent Systems - 4th International Workshop, EXTRAAMAS 2022, Virtual Event, May 9-10, 2022, Revised Selected Papers*, volume 13283 of *Lecture Notes in Computer Science*, pages 38–56. Springer, 2022.

[119] Jasmina Gajcin and Ivana Dusparic. ACTER: diverse and actionable counterfactual sequences for explaining and diagnosing RL policies. *CoRR*, abs/2402.06503, 2024.

[120] Jasmina Gajcin and Ivana Dusparic. RACCER: towards reachable and certain counterfactual explanations for reinforcement learning. In Mehdi Dastani, Jaime Simão Sichman, Natasha Alechina, and Virginia Dignum, editors, *Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems, AAMAS 2024, Auckland, New Zealand, May 6-10, 2024*, pages 632–640. ACM, 2024.

[121] Jasmina Gajcin and Ivana Dusparic. Redefining counterfactual explanations for reinforcement learning: Overview, challenges and opportunities. *ACM Comput. Surv.*, 56(9):219:1–219:33, 2024.

[122] Jasmina Gajcin, Rahul Nair, Tejaswini Pedapati, Radu Marinescu, Elizabeth Daly, and Ivana Dusparic. Contrastive explanations for comparing preferences of reinforcement learning agents. *CoRR*, abs/2112.09462, 2021.

[123] Maor Gaon and Ronen I. Brafman. Reinforcement learning with non-Markovian rewards. In *The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020*, pages 3980–3987. AAAI Press, 2020.

[124] Marta Garnelo, Kai Arulkumaran, and Murray Shanahan. Towards deep symbolic reinforcement learning. *CoRR*, abs/1609.05518, 2016.

[125] Claire Glanois, Paul Weng, Matthieu Zimmer, Dong Li, Tianpei Yang, Jianye Hao, and Wulong Liu. A survey on interpretable reinforcement learning. *Mach. Learn.*, 113(8):5847–5890, 2024.

[126] Moritz Göbelbecker, Thomas Keller, Patrick Eyerich, Michael Brenner, and Bernhard Nebel. Coming up with good excuses: What to do when no plan can be found. In Ronen I. Brafman, Hector Geffner, Jörg Hoffmann, and Henry A. Kautz, editors, *Proceedings of the 20th International Conference on Automated Planning and Scheduling, ICAPS 2010, Toronto, Ontario, Canada, May 12-16, 2010*, pages 81–88. AAAI, 2010.

[127] Vikash Goel, Jameson Weng, and Pascal Poupart. Unsupervised video object segmentation for deep reinforcement learning. In Samy Bengio, Hanna M. Wallach, Hugo Larochelle, Kristen Grauman, Nicolò Cesa-Bianchi, and Roman Garnett, editors, *Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montréal, Canada*, pages 5688–5699, 2018.

[128] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron C. Courville, and Yoshua Bengio. Generative adversarial networks. *CoRR*, abs/1406.2661, 2014.

[129] Omer Gottesman, Joseph Futoma, Yao Liu, Sonali Parbhoo, Leo A. Celi, Emma Brunskill, and Finale Doshi-Velez. Interpretable off-policy evaluation in reinforcement learning by highlighting influential transitions. In *Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event*, volume 119 of *Proceedings of Machine Learning Research*, pages 3658–3667. PMLR, 2020.

[130] Ole-Christoffer Granmo. The Tsetlin machine - A game theoretic bandit driven approach to optimal pattern recognition with propositional logic. *CoRR*, abs/1804.01508, 2018.

[131] Samuel Greydanus, Anurag Koul, Jonathan Dodge, and Alan Fern. Visualizing and understanding Atari agents. In Jennifer G. Dy and Andreas Krause, editors, *Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018*, volume 80 of *Proceedings of Machine Learning Research*, pages 1787–1796. PMLR, 2018.

[132] Timo P. Gros, David Groß, Stefan Gumhold, Jörg Hoffmann, Michaela Klauck, and Marcel Steinmetz. Tracevis: Towards visualization for deep statistical model checking. In Tiziana Margaria and Bernhard Steffen, editors, *Leveraging Applications of Formal Methods, Verification and Validation: Tools and Trends - 9th International Symposium on Leveraging Applications of Formal Methods, ISoLA 2020, Rhodes, Greece, October 20-30, 2020, Proceedings, Part IV*, volume 12479 of *Lecture Notes in Computer Science*, pages 27–46. Springer, 2020.

[133] Timo P. Gros, Holger Hermanns, Jörg Hoffmann, Michaela Klauck, and Marcel Steinmetz. Deep statistical model checking. In Alexey Gotsman and Ana Sokolova, editors, *Formal Techniques for Distributed Objects, Components, and Systems - 40th IFIP WG 6.1 International Conference, FORTE 2020, Held as Part of the 15th International Federated Conference on Distributed Computing Techniques, DisCoTec 2020, Valletta, Malta, June 15-19, 2020, Proceedings*, volume 12136 of *Lecture Notes in Computer Science*, pages 96–114. Springer, 2020.

[134] Yin Gu, Kai Zhang, Qi Liu, Weibo Gao, Longfei Li, and Jun Zhou. $\pi$-light: Programmatic interpretable reinforcement learning for resource-limited traffic signal control. In Michael J. Wooldridge, Jennifer G. Dy, and Sriraam Natarajan, editors, *Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2014, February 20-27, 2024, Vancouver, Canada*, pages 21107–21115. AAAI Press, 2024.

[135] Carlos Guestrin, Daphne Koller, Chris Gearhart, and Neal Kanodia. Generalizing plans to new environments in relational MDPs. In Georg Gottlob and Toby Walsh, editors, *IJCAI-03, Proceedings of the Eighteenth International Joint Conference on Artificial Intelligence, Acapulco, Mexico, August 9-15, 2003*, pages 1003–1010. Morgan Kaufmann, 2003.

[136] Sihang Guo, Ruohan Zhang, Bo Liu, Yifeng Zhu, Dana H. Ballard, Mary M. Hayhoe, and Peter Stone. Machine versus human attention in deep reinforcement learning tasks. In Marc'Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan, editors, *Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual*, pages 25370–25385, 2021.

[137] Wei Guo and Peng Wei. Explainable deep reinforcement learning for aircraft separation assurance. In *2022 IEEE/AIAA 41st Digital Avionics Systems Conference (DASC)*, pages 1–10. IEEE, 2022.

[138] Wenbo Guo, Xian Wu, Usmann Khan, and Xinyu Xing. EDGE: explaining deep reinforcement learning policies. In Marc'Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan, editors, *NeurIPS*, pages 12222–12236, 2021.

[139] Ujjwal Das Gupta, Erik Talvitie, and Michael Bowling. Policy tree: Adaptive representation for policy gradient. In Blai Bonet and Sven Koenig, editors, *Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, January 25-30, 2015, Austin, Texas, USA*, pages 2547–2553. AAAI Press, 2015.

[140] Daniel Conrad Halbert. *Programming by example*. University of California, Berkeley, 1984.

[141] Joseph Y. Halpern and Judea Pearl. Causes and explanations: A structural-model approach - part II: explanations. In Bernhard Nebel, editor, *Proceedings of the Seventeenth International Joint Conference on Artificial Intelligence, IJCAI 2001, Seattle, Washington, USA, August 4-10, 2001*, pages 27–34. Morgan Kaufmann, 2001.

[142] Mohammadhosein Hasanbeig, Natasha Yogananda Jeppu, Alessandro Abate, Tom Melham, and Daniel Kroening. Deepsynth: Automata synthesis for automatic task segmentation in deep reinforcement learning. In *Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, Thirty-Third Conference on Innovative Applications of Artificial Intelligence, IAAI 2021, The Eleventh Symposium on Educational Advances in Artificial Intelligence, EAAI 2021, Virtual Event, February 2-9, 2021*, pages 7647–7656. AAAI Press, 2021.

[143] Ali Hassani, Steven Walton, Nikhil Shah, Abulikemu Abuduweili, Jiachen Li, and Humphrey Shi. Escaping the big data paradigm with compact transformers. *CoRR*, abs/2104.05704, 2021.

[144] Bradley Hayes and Julie A. Shah. Improving robot controller transparency through autonomous policy explanation. In Bilge Mutlu, Manfred Tscheligi, Astrid Weiss, and James E. Young, editors, *Proceedings of the 2017 ACM/IEEE International Conference on Human-Robot Interaction, HRI 2017, Vienna, Austria, March 6-9, 2017*, pages 303–312. ACM, 2017.

[145] Rishi Hazra and Luc De Raedt. Deep explainable relational reinforcement learning: A neuro-symbolic approach. In Danai Koutra, Claudia Plant, Manuel Gomez Rodriguez, Elena Baralis, and Francesco Bonchi, editors, *Machine Learning and Knowledge Discovery in Databases: Research Track - European Conference, ECML PKDD 2023, Turin, Italy, September 18-22, 2023, Proceedings, Part IV*, volume 14172 of *Lecture Notes in Computer Science*, pages 213–229. Springer, 2023.

[146] Lei He, Nabil Aouf, and Bifeng Song. Explainable deep reinforcement learning for UAV autonomous path planning. *Aerospace science and technology*, 118:107052, 2021.

[147] Wenbin He, Teng-Yok Lee, Jeroen van Baar, Kent Wittenburg, and Han-Wei Shen. Dynamicsexplorer: Visual analytics for robot control tasks involving dynamics and LSTM-based control policies. In *2020 IEEE Pacific Visualization Symposium, PacificVis 2020, Tianjin, China, June 3-5, 2020*, pages 36–45. IEEE, 2020.

[148] Daniel Hein, Alexander Hentschel, Thomas A. Runkler, and Steffen Udluft. Particle swarm optimization for generating interpretable fuzzy reinforcement learning policies. *Eng. Appl. Artif. Intell.*, 65:87–98, 2017.

[149] Daniel Hein, Steffen Udluft, and Thomas A. Runkler. Generating interpretable reinforcement learning policies using genetic programming. In Manuel López-Ibáñez, Anne Auger, and Thomas Stützle, editors, *Proceedings of the Genetic and Evolutionary Computation Conference Companion, GECCO 2019, Prague, Czech Republic, July 13-17, 2019*, pages 23–24. ACM, 2019.

[150] Tue Herlau and Rasmus Larsen. Reinforcement learning of causal variables using mediation analysis. In *Thirty-Sixth AAAI Conference on Artificial Intelligence, AAAI 2022, Thirty-Fourth Conference on Innovative Applications of Artificial Intelligence, IAAI 2022, The Twelveth Symposium on Educational Advances in Artificial Intelligence, EAAI 2022 Virtual Event, February 22 - March 1, 2022*, pages 6910–6917. AAAI Press, 2022.

[151] Tom Heskes, Evi Sijben, Ioan Gabriel Bucur, and Tom Claassen. Causal Shapley values: Exploiting causal knowledge to explain individual predictions of complex models. In Hugo Larochelle, Marc'Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, editors, *Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual*, 2020.

[152] Matteo Hessel, Joseph Modayil, Hado van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan, Bilal Piot, Mohammad Gheshlaghi Azar, and David Silver. Rainbow: Combining improvements in deep reinforcement learning. In Sheila A. McIlraith and Kilian Q. Weinberger, editors, *Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence, (AAAI-18), the 30th innovative Applications of Artificial Intelligence (IAAI-18), and the 8th AAAI Symposium on Educational Advances in Artificial Intelligence (EAAI-18), New Orleans, Louisiana, USA, February 2-7, 2018*, pages 3215–3222. AAAI Press, 2018.

[153] Alexandre Heuillet, Fabien Couthouis, and Natalia Díaz Rodríguez. Explainability in deep reinforcement learning. *Knowl. Based Syst.*, 214:106685, 2021.

[154] Alexandre Heuillet, Fabien Couthouis, and Natalia Díaz Rodríguez. Collective explainable AI: explaining cooperative strategies and agent contribution in multiagent reinforcement learning with Shapley values. *IEEE Comput. Intell. Mag.*, 17(1):59–71, 2022.

[155] Thomas Hickling, Abdelhafid Zenati, Nabil Aouf, and Phillippa Spencer. Explainability in deep reinforcement learning: A review into current methods and applications. *ACM Comput. Surv.*, 56(5):125:1–125:35, 2024.

[156] Jacob Hilton, Nick Cammarata, Shan Carter, Gabriel Goh, and Chris Olah. Understanding RL vision. *Distill*, 5(11):e29, 2020.

[157] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. *Neural Comput.*, 9(8):1735–1780, 1997.

[158] Robert R. Hoffman, Shane T. Mueller, Gary Klein, and Jordan Litman. Metrics for explainable AI: challenges and prospects. *CoRR*, abs/1812.04608, 2018.

[159] Jörg Hoffmann and Daniele Magazzeni. Explainable AI planning (XAIP): overview and the case of contrastive explanation (extended abstract). In Markus Krötzsch and Daria Stepanova, editors, *Reasoning Web. Explainable Artificial Intelligence - 15th International Summer School 2019, Bolzano, Italy, September 20-24, 2019, Tutorial Lectures*, volume 11810 of *Lecture Notes in Computer Science*, pages 277–282. Springer, 2019.

[160] Jianfeng Huang, Plamen P. Angelov, and Chengliang Yin. Interpretable policies for reinforcement learning by empirical fuzzy sets. *Eng. Appl. Artif. Intell.*, 91:103559, 2020.

[161] Sandy H. Huang, Kush Bhatia, Pieter Abbeel, and Anca D. Dragan. Establishing appropriate trust via critical states. In *2018 IEEE/RSJ International Conference on Intelligent Robots and Systems, IROS 2018, Madrid, Spain, October 1-5, 2018*, pages 3929–3936. IEEE, 2018.

[162] Sandy H. Huang, David Held, Pieter Abbeel, and Anca D. Dragan. Enabling robots to communicate their objectives. *Auton. Robots*, 43(2):309–326, 2019.

[163] Tobias Huber, Maximilian Demmler, Silvan Mertes, Matthew L. Olson, and Elisabeth André. GANterfactual-RL: Understanding reinforcement learning agents' strategies through visual counterfactual explanations. In Noa Agmon, Bo An, Alessandro Ricci, and William Yeoh, editors, *Proceedings of the 2023 International Conference on Autonomous Agents and Multiagent Systems, AAMAS 2023, London, United Kingdom, 29 May 2023 - 2 June 2023*, pages 1097–1106. ACM, 2023.

[164] Tobias Huber, Benedikt Limmer, and Elisabeth André. Benchmarking perturbation-based saliency maps for explaining Atari agents. *Frontiers Artif. Intell.*, 5, 2022.

[165] Tobias Huber, Dominik Schiller, and Elisabeth André. Enhancing explainability of deep reinforcement learning through selective layer-wise relevance propagation. In Christoph Benzmüller and Heiner Stuckenschmidt, editors, *KI 2019: Advances in Artificial Intelligence - 42nd German Conference on AI, Kassel, Germany, September 23-26, 2019, Proceedings*, volume 11793 of *Lecture Notes in Computer Science*, pages 188–202. Springer, 2019.

[166] Tobias Huber, Katharina Weitz, Elisabeth André, and Ofra Amir. Local and global explanations of agent behavior: Integrating strategy summaries with saliency maps. *Artif. Intell.*, 301:103571, 2021.

[167] Alihan Hüyük, Daniel Jarrett, and Mihaela van der Schaar. Explaining by imitating: Understanding decisions by interpretable policy learning. *CoRR*, abs/2310.19831, 2023.

[168] Rodrigo Toro Icarte, Toryn Q. Klassen, Richard Anthony Valenzano, and Sheila A. McIlraith. Using reward machines for high-level task specification and decomposition in reinforcement learning. In Jennifer G. Dy and Andreas Krause, editors, *Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018*, volume 80 of *Proceedings of Machine Learning Research*, pages 2112–2121. PMLR, 2018.

[169] Rodrigo Toro Icarte, Ethan Waldie, Toryn Q. Klassen, Richard Anthony Valenzano, Margarita P. Castro, and Sheila A. McIlraith. Learning reward machines for partially observable reinforcement learning. In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence d'Alché-Buc, Emily B. Fox, and Roman Garnett, editors, *Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada*, pages 15497–15508, 2019.

[170] Jeevana Priya Inala, Osbert Bastani, Zenna Tavares, and Armando Solar-Lezama. Synthesizing programmatic policies that inductively generalize. In *8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020*. OpenReview.net, 2020.

[171] Hidenori Itaya, Tsubasa Hirakawa, Takayoshi Yamashita, Hironobu Fujiyoshi, and Komei Sugiura. Visual explanation using attention mechanism in actor-critic-based deep reinforcement learning. In *International Joint Conference on Neural Networks, IJCNN 2021, Shenzhen, China, July 18-22, 2021*, pages 1–10. IEEE, 2021.

[172] Alessandro Iucci, Alberto Hata, Ahmad Terra, Rafia Inam, and Iolanda Leite. Explainable reinforcement learning for human-robot collaboration. In *20th International Conference on Advanced Robotics, ICAR 2021, Ljubljana, Slovenia, December 6-10, 2021*, pages 927–934. IEEE, 2021.

[173] Rahul Iyer, Yuezhang Li, Huao Li, Michael Lewis, Ramitha Sundar, and Katia P. Sycara. Transparency and explanation in deep reinforcement learning neural networks. In Jason Furman, Gary E. Marchant, Huw Price, and Francesca Rossi, editors, *Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society, AIES 2018, New Orleans, LA, USA, February 02-03, 2018*, pages 144–150. ACM, 2018.

[174] David Jackson. A new, node-focused model for genetic programming. In Alberto Moraglio, Sara Silva, Krzysztof Krawiec, Penousal Machado, and Carlos Cotta, editors, *Genetic Programming - 15th European Conference, EuroGP 2012, Málaga, Spain, April 11-13, 2012. Proceedings*, volume 7244 of *Lecture Notes in Computer Science*, pages 49–60. Springer, 2012.

[175] Robert A. Jacobs, Michael I. Jordan, Steven J. Nowlan, and Geoffrey E. Hinton. Adaptive mixtures of local experts. *Neural Comput.*, 3(1):79–87, 1991.

[176] Alexis Jacq, Johan Ferret, Olivier Pietquin, and Matthieu Geist. Lazy-MDPs: Towards interpretable reinforcement learning by learning when to act. *CoRR*, abs/2203.08542, 2022.

[177] Tarun Kumar Jain, Dharmender Singh Kushwaha, and Arun Kumar Misra. Optimization of the Quine-McCluskey method for the minimization of the boolean expressions. In *Fourth International Conference on Autonomic and Autonomous Systems, ICAS 2008, 16-21 March 2008, Gosier, Guadeloupe*, pages 165–168. IEEE Computer Society, 2008.

[178] Theo Jaunet, Romain Vuillemot, and Christian Wolf. Drlviz: Understanding decisions and memory in deep reinforcement learning. *Comput. Graph. Forum*, 39(3):49–61, 2020.

[179] Erik Jenner and Adam Gleave. Preprocessing reward functions for interpretability. *CoRR*, abs/2203.13553, 2022.

[180] Ranjit Jhala and Rupak Majumdar. Software model checking. *ACM Comput. Surv.*, 41(4):21:1–21:54, 2009.

[181] Aman Jhunjhunwala, Jaeyoung Lee, Sean Sedwards, Vahdat Abdelzad, and Krzysztof Czarnecki. Improved policy extraction via online Q-value distillation. In *2020 International Joint Conference on Neural Networks, IJCNN 2020, Glasgow, United Kingdom, July 19-24, 2020*, pages 1–8. IEEE, 2020.

[182] Yiding Jiang, Shixiang Gu, Kevin Murphy, and Chelsea Finn. Language as an abstraction for hierarchical deep reinforcement learning. In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence d'Alché-Buc, Emily B. Fox, and Roman Garnett, editors, *Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada*, pages 9414–9426, 2019.

[183] Zhengyao Jiang and Shan Luo. Neural logic reinforcement learning. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, *Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA*, volume 97 of *Proceedings of Machine Learning Research*, pages 3110–3119. PMLR, 2019.

[184] Mu Jin, Zhihao Ma, Kebing Jin, Hankz Hankui Zhuo, Chen Chen, and Chao Yu. Creativity of AI: automatic symbolic option discovery for facilitating deep reinforcement learning. In *Thirty-Sixth AAAI Conference on Artificial Intelligence, AAAI 2022, Thirty-Fourth Conference on Innovative Applications of Artificial Intelligence, IAAI 2022, The Twelveth Symposium on Educational Advances in Artificial Intelligence, EAAI 2022 Virtual Event, February 22 - March 1, 2022*, pages 7042–7050. AAAI Press, 2022.

[185] Ho-Taek Joo and Kyung-Joong Kim. Visualization of deep reinforcement learning using Grad-CAM: How AI plays Atari games? In *IEEE Conference on Games, CoG 2019, London, United Kingdom, August 20-23, 2019*, pages 1–2. IEEE, 2019.

[186] Shirel Josef and Amir Degani. Deep reinforcement learning for safe local planning of a ground vehicle in unknown rough terrain. *IEEE Robotics Autom. Lett.*, 5(4):6748–6755, 2020.

[187] Zoe Juozapaitis, Anurag Koul, Alan Fern, Martin Erwig, and Finale Doshi-Velez. Explainable reinforcement learning via reward decomposition. In *IJCAI/ECAI Workshop on explainable artificial intelligence*, 2019.

[188] Markus Kaiser, Clemens Otte, Thomas A. Runkler, and Carl Henrik Ek. interpretable dynamics models for data-efficient reinforcement learning. In *27th European Symposium on Artificial Neural Networks, ESANN 2019, Bruges, Belgium, April 24-26, 2019*, 2019.

[189] Timotheus Kampik, Juan Carlos Nieves, and Helena Lindgren. Explaining sympathetic actions of rational agents. In Davide Calvaresi, Amro Najjar, Michael Schumacher, and Kary Främling, editors, *Explainable, Transparent Autonomous Agents and Multi-Agent Systems - First International Workshop, EXTRAAMAS 2019, Montreal, QC, Canada, May 13-14, 2019, Revised Selected Papers*, volume 11763 of *Lecture Notes in Computer Science*, pages 59–76. Springer, 2019.

[190] Ken Kansky, Tom Silver, David A. Mély, Mohamed Eldawy, Miguel Lázaro-Gredilla, Xinghua Lou, Nimrod Dorfman, Szymon Sidor, D. Scott Phoenix, and Dileep George. Schema networks: Zero-shot transfer with a generative causal model of intuitive physics. In Doina Precup and Yee Whye Teh, editors, *Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017*, volume 70 of *Proceedings of Machine Learning Research*, pages 1809–1818. PMLR, 2017.

[191] Sergey Karakovskiy and Julian Togelius. The Mario AI benchmark and competitions. *IEEE Trans. Comput. Intell. AI Games*, 4(1):55–67, 2012.

[192] Amir-Hossein Karimi, Gilles Barthe, Bernhard Schölkopf, and Isabel Valera. A survey of algorithmic recourse: definitions, formulations, solutions, and prospects. *CoRR*, abs/2010.04050, 2020.

[193] Amir-Hossein Karimi, Bernhard Schölkopf, and Isabel Valera. Algorithmic recourse: from counterfactual explanations to interventions. In Madeleine Clare Elish, William Isaac, and Richard S. Zemel, editors, *FAccT '21: 2021 ACM Conference on Fairness, Accountability, and Transparency, Virtual Event / Toronto, Canada, March 3-10, 2021*, pages 353–362. ACM, 2021.

[194] Daniel Kasenberg and Matthias Scheutz. Interpretable apprenticeship learning with temporal logic specifications. In *56th IEEE Annual Conference on Decision and Control, CDC 2017, Melbourne, Australia, December 12-15, 2017*, pages 4914–4921. IEEE, 2017.

[195] Yafim Kazak, Clark W. Barrett, Guy Katz, and Michael Schapira. Verifying deep-RL-driven systems. In *Proceedings of the 2019 Workshop on Network Meets AI & ML, NetAI@SIGCOMM 2019, Beijing, China, August 23, 2019*, pages 83–89. ACM, 2019.

[196] Dmitry Kazhdan, Zohreh Shams, and Pietro Liò. MARLeME: a multi-agent reinforcement learning model extraction library. In *2020 International Joint Conference on Neural Networks, IJCNN 2020, Glasgow, United Kingdom, July 19-24, 2020*, pages 1–8. IEEE, 2020.

[197] James Kennedy and Russell Eberhart. Particle swarm optimization. In *Proceedings of International Conference on Neural Networks (ICNN'95), Perth, WA, Australia, November 27 - December 1, 1995*, pages 1942–1948. IEEE, 1995.

[198] Omar Zia Khan, Pascal Poupart, and James P. Black. Minimal sufficient explanations for factored Markov decision processes. In Alfonso Gerevini, Adele E. Howe, Amedeo Cesta, and Ioannis Refanidis, editors, *Proceedings of the 19th International Conference on Automated Planning and Scheduling, ICAPS 2009, Thessaloniki, Greece, September 19-23, 2009*. AAAI, 2009.

[199] Joseph Kim, Christian Muise, Ankit Shah, Shubham Agarwal, and Julie Shah. Bayesian inference of linear temporal logic specifications for contrastive explanations. In Sarit Kraus, editor, *Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, IJCAI 2019, Macao, China, August 10-16, 2019*, pages 5591–5598. ijcai.org, 2019.

[200] Hector Kohler, Riad Akrour, and Philippe Preux. Optimal interpretability-performance trade-off of classification trees with black-box reinforcement learning. *CoRR*, abs/2304.05839, 2023.

[201] Hector Kohler, Quentin Delfosse, Riad Akrour, Kristian Kersting, and Philippe Preux. Interpretable and editable programmatic tree policies for reinforcement learning. *CoRR*, abs/2405.14956, 2024.

[202] Olivera Kotevska, Jeffrey Munk, Kuldeep R. Kurte, Yan Du, Kadir Amasyali, Robert W. Smith, and Helia Zandi. Methodology for interpretable reinforcement learning model for HVAC energy control. In Xintao Wu, Chris Jermaine, Li Xiong, Xiaohua Hu, Olivera Kotevska, Siyuan Lu, Weija Xu, Srinivas Aluru, Chengxiang Zhai, Eyhab Al-Masri, Zhiyuan Chen, and Jeff Saltz, editors, *2020 IEEE International Conference on Big Data (IEEE BigData 2020), Atlanta, GA, USA, December 10-13, 2020*, pages 1555–1564. IEEE, 2020.

[203] Anurag Koul, Alan Fern, and Sam Greydanus. Learning finite state representations of recurrent policy networks. In *7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019*. OpenReview.net, 2019.

[204] Agneza Krajna, Mario Brcic, Tomislav Lipic, and Juraj Doncevic. Explainability in reinforcement learning: perspective and position. *CoRR*, abs/2203.11547, 2022.

[205] Jiří Kubalík, Eduard Alibekov, and Robert Babuška. Optimal control via reinforcement learning with symbolic policy approximation. *IFAC-PapersOnLine*, 50(1):4162–4167, 2017.

[206] Tejas D. Kulkarni, Karthik Narasimhan, Ardavan Saeedi, and Josh Tenenbaum. Hierarchical deep reinforcement learning: Integrating temporal abstraction and intrinsic motivation. In Daniel D. Lee, Masashi Sugiyama, Ulrike von Luxburg, Isabelle Guyon, and Roman Garnett, editors, *Advances in Neural Information Processing Systems 29: Annual Conference on Neural Information Processing Systems 2016, December 5-10, 2016, Barcelona, Spain*, pages 3675–3683, 2016.

[207] Isaac Lage, Daphna Lifschitz, Finale Doshi-Velez, and Ofra Amir. Exploring computational user models for agent policy summarization. In Sarit Kraus, editor, *Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, IJCAI 2019, Macao, China, August 10-16, 2019*, pages 1401–1407. ijcai.org, 2019.

[208] Isaac Lage, Daphna Lifschitz, Finale Doshi-Velez, and Ofra Amir. Toward robust policy summarization. In Edith Elkind, Manuela Veloso, Noa Agmon, and Matthew E. Taylor, editors, *Proceedings of the 18th International Conference on Autonomous Agents and MultiAgent Systems, AAMAS '19, Montreal, QC, Canada, May 13-17, 2019*, pages 2081–2083. International Foundation for Autonomous Agents and Multiagent Systems, 2019.

[209] Mikel Landajuela, Brenden K. Petersen, Sookyung Kim, Cláudio P. Santiago, Ruben Glatt, T. Nathan Mundhenk, Jacob F. Pettit, and Daniel M. Faissol. Discovering symbolic policies with deep reinforcement learning. In Marina Meila and Tong Zhang, editors, *Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event*, volume 139 of *Proceedings of Machine Learning Research*, pages 5979–5989. PMLR, 2021.

[210] Michael T. Lash. HEX: human-in-the-loop explainability via deep reinforcement learning. *CoRR*, abs/2206.01343, 2022.

[211] Jung Hoon Lee. Complementary reinforcement learning towards explainable agents. *CoRR*, abs/1901.00188, 2019.

[212] Matteo Leonetti, Luca Iocchi, and Peter Stone. A synthesis of automated planning and reinforcement learning for efficient, robust decision-making. *Artif. Intell.*, 241:103–130, 2016.

[213] Timothée Lesort, Natalia Díaz Rodríguez, Jean-François Goudou, and David Filliat. State representation learning for control: An overview. *Neural Networks*, 108:379–392, 2018.

[214] Huiling Li, Jun Wu, Hansong Xu, Gaolei Li, and Mohsen Guizani. Explainable intelligence-driven defense mechanism against advanced persistent threats: A joint edge game and AI approach. *IEEE Trans. Dependable Secur. Comput.*, 19(2):757–775, 2022.

[215] Nianyu Li, Sridhar Adepu, Eunsuk Kang, and David Garlan. Explanations for human-on-the-loop: A probabilistic model checking approach. In Shinichi Honiden, Elisabetta Di Nitto, and Radu Calinescu, editors, *SEAMS '20: IEEE/ACM 15th International Symposium on Software Engineering for Adaptive and Self-Managing Systems, Seoul, Republic of Korea, 29 June - 3 July, 2020*, pages 181–187. ACM, 2020.

[216] Wentian Li, Xidong Feng, Haotian An, Xiang Yao Ng, and Yu-Jin Zhang. MRI reconstruction with interpretable pixel-wise operations using reinforcement learning. In *The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020*, pages 792–799. AAAI Press, 2020.

[217] Xiao Li, Cristian Ioan Vasile, and Calin Belta. Reinforcement learning with temporal logic rewards. In *2017 IEEE/RSJ International Conference on Intelligent Robots and Systems, IROS 2017, Vancouver, BC, Canada, September 24-28, 2017*, pages 3834–3839. IEEE, 2017.

[218] Yuezhang Li, Katia P. Sycara, and Rahul Iyer. Object-sensitive deep reinforcement learning. In Christoph Benzmüller, Christine L. Lisetti, and Martin Theobald, editors, *GCAI 2017, 3rd Global Conference on Artificial Intelligence, Miami, FL, USA, 18-22 October 2017*, volume 50 of *EPiC Series in Computing*, pages 20–35. EasyChair, 2017.

[219] Roman Liessner, Jan Dohmen, and Marco A. Wiering. Explainable reinforcement learning for longitudinal control. In Ana Paula Rocha, Luc Steels, and H. Jaap van den Herik, editors, *Proceedings of the 13th International Conference on Agents and Artificial Intelligence, ICAART 2021, Volume 2, Online Streaming, February 4-6, 2021*, pages 874–881. SCITEPRESS, 2021.

[220] Vladimir Lifschitz. What is answer set programming? In Dieter Fox and Carla P. Gomes, editors, *Proceedings of the Twenty-Third AAAI Conference on Artificial Intelligence, AAAI 2008, Chicago, Illinois, USA, July 13-17, 2008*, pages 1594–1597. AAAI Press, 2008.

[221] Amarildo Likmeta, Alberto Maria Metelli, Andrea Tirinzoni, Riccardo Giol, Marcello Restelli, and Danilo Romano. Combining reinforcement learning with rule-based controllers for transparent and general decision-making in autonomous driving. *Robotics Auton. Syst.*, 131:103568, 2020.

[222] Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. In Yoshua Bengio and Yann LeCun, editors, *4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings*, 2016.

[223] Yixin Lin, Austin S. Wang, Eric Undersander, and Akshara Rai. Efficient and interpretable robot manipulation with graph neural networks. *IEEE Robotics Autom. Lett.*, 7(2):2740–2747, 2022.

[224] Zhengxian Lin, Kin-Ho Lam, and Alan Fern. Contrastive explanations for reinforcement learning via embedded self predictions. In *9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021*. OpenReview.net, 2021.

[225] Bing Liu, Yiyuan Xia, and Philip S Yu. Clustering via decision tree construction. *Foundations and advances in data mining*, pages 97–124, 2005.

[226] Guiliang Liu, Oliver Schulte, Wang Zhu, and Qingcan Li. Toward interpretable deep reinforcement learning with linear model U-trees. In Michele Berlingerio, Francesco Bonchi, Thomas Gärtner, Neil Hurley, and Georgiana Ifrim, editors, *Machine Learning and Knowledge Discovery in Databases - European Conference, ECML PKDD 2018, Dublin, Ireland, September 10-14, 2018, Proceedings, Part III*, volume 11052 of *Lecture Notes in Computer Science*, pages 414–429. Springer, 2018.

[227] Guiliang Liu, Xiangyu Sun, Oliver Schulte, and Pascal Poupart. Learning tree interpretation from object representation for deep reinforcement learning. In Marc'Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan, editors, *Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual*, pages 19622–19636, 2021.

[228] Haozhe Liu, Mingchen Zhuge, Bing Li, Yuhui Wang, Francesco Faccio, Bernard Ghanem, and Jürgen Schmidhuber. Learning to identify critical states for reinforcement learning from videos. In *IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023*, pages 1955–1965. IEEE, 2023.

[229] Yang Liu, Xinzhi Wang, Yudong Chang, and Chao Jiang. Towards explainable reinforcement learning using scoring mechanism augmented agents. In Gérard Memmi, Baijian Yang, Linghe Kong, Tianwei Zhang, and Meikang Qiu, editors, *Knowledge Science, Engineering and Management - 15th International Conference, KSEM 2022, Singapore, August 6-8, 2022, Proceedings, Part II*, volume 13369 of *Lecture Notes in Computer Science*, pages 547–558. Springer, 2022.

[230] Meghann Lomas, Robert Chevalier, Ernest Vincent Cross II, Robert Christopher Garrett, John Hoare, and Michael Kopack. Explaining robot actions. In Holly A. Yanco, Aaron Steinfeld, Vanessa Evers, and Odest Chadwicke Jenkins, editors, *International Conference on Human-Robot Interaction, HRI'12, Boston, MA, USA - March 05 - 08, 2012*, pages 187–188. ACM, 2012.

[231] Jakob Løver, Vilde B Gjærum, and Anastasios M Lekkas. Explainable AI methods on a deep reinforcement learning agent for automatic docking. *IFAC-PapersOnLine*, 54(16):146–152, 2021.

[232] David C Luckham and Brian Frasca. Complex event processing in distributed systems. *Computer Systems Laboratory Technical Report CSL-TR-98-754. Stanford University, Stanford*, 28:16, 1998.

[233] Scott M. Lundberg, Gabriel G. Erion, and Su-In Lee. Consistent individualized feature attribution for tree ensembles. *CoRR*, abs/1802.03888, 2018.

[234] Scott M. Lundberg and Su-In Lee. A unified approach to interpreting model predictions. In Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett, editors, *Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA*, pages 4765–4774, 2017.

[235] Ronny Luss, Amit Dhurandhar, and Miao Liu. Local explanations for reinforcement learning. In Brian Williams, Yiling Chen, and Jennifer Neville, editors, *Thirty-Seventh AAAI Conference on Artificial Intelligence, AAAI 2023, Thirty-Fifth Conference on Innovative Applications of Artificial Intelligence, IAAI 2023, Thirteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2023, Washington, DC, USA, February 7-14, 2023*, pages 9002–9010. AAAI Press, 2023.

[236] Daoming Lyu, Fangkai Yang, Bo Liu, and Steven Gustafson. SDRL: interpretable and data-efficient deep reinforcement learning leveraging symbolic planning. In *The Thirty-Third AAAI Conference on Artificial Intelligence, AAAI 2019, The Thirty-First Innovative Applications of Artificial Intelligence Conference, IAAI 2019, The Ninth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2019, Honolulu, Hawaii, USA, January 27 - February 1, 2019*, pages 2970–2977. AAAI Press, 2019.

[237] Zhihao Ma, Yuzheng Zhuang, Paul Weng, Hankz Hankui Zhuo, Dong Li, Wulong Liu, and Jianye Hao. Learning symbolic rules for interpretable deep reinforcement learning. *CoRR*, abs/2103.08228, 2021.

[238] Prashan Madumal, Tim Miller, Liz Sonenberg, and Frank Vetere. Distal explanations for explainable reinforcement learning agents. *CoRR*, abs/2001.10284, 2020.

[239] Prashan Madumal, Tim Miller, Liz Sonenberg, and Frank Vetere. Explainable reinforcement learning through a causal lens. In *The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020*, pages 2493–2500. AAAI Press, 2020.

[240] Francis Maes, Raphaël Fonteneau, Louis Wehenkel, and Damien Ernst. Policy search in a space of simple closed-form formulas: Towards interpretability of reinforcement learning. In Jean-Gabriel Ganascia, Philippe Lenca, and Jean-Marc Petit, editors, *Discovery Science - 15th International Conference, DS 2012, Lyon, France, October 29-31, 2012. Proceedings*, volume 7569 of *Lecture Notes in Computer Science*, pages 37–51. Springer, 2012.

[241] Ofir Marom and Benjamin Rosman. Zero-shot transfer with deictic object-oriented representation in reinforcement learning. In Samy Bengio, Hanna M. Wallach, Hugo Larochelle, Kristen Grauman, Nicolò Cesa-Bianchi, and Roman Garnett, editors, *Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montréal, Canada*, pages 2297–2305, 2018.

[242] Stephen Marsland, Jonathan Shapiro, and Ulrich Nehmzow. A self-organising network that grows when required. *Neural Networks*, 15(8-9):1041–1058, 2002.

[243] David Martínez Martínez, Guillem Alenyà, and Carme Torras. Relational reinforcement learning with guided demonstrations. *Artif. Intell.*, 247:295–312, 2017.

[244] Joe McCalmon, Thai Le, Sarra M. Alqahtani, and Dongwon Lee. CAPS: comprehensible abstract policy summaries for explaining reinforcement learning agents. In Piotr Faliszewski, Viviana Mascardi, Catherine Pelachaud, and Matthew E. Taylor, editors, *21st International Conference on Autonomous Agents and Multiagent Systems, AAMAS 2022, Auckland, New Zealand, May 9-13, 2022*, pages 889–897. International Foundation for Autonomous Agents and Multiagent Systems (IFAAMAS), 2022.

[245] Sean McGregor, Hailey Buckingham, Thomas G. Dietterich, Rachel Houtman, Claire A. Montgomery, and Ronald A. Metoyer. Interactive visualization for testing Markov decision processes: MDPVIS. *J. Vis. Lang. Comput.*, 39:93–106, 2017.

[246] Jan Hendrik Metzen. Learning graph-based representations for continuous reinforcement learning domains. In Hendrik Blockeel, Kristian Kersting, Siegfried Nijssen, and Filip Zelezný, editors, *Machine Learning and Knowledge Discovery in Databases - European Conference, ECML PKDD 2013, Prague, Czech Republic, September 23-27, 2013, Proceedings, Part I*, volume 8188 of *Lecture Notes in Computer Science*, pages 81–96. Springer, 2013.

[247] Stephanie Milani, Nicholay Topin, Manuela Veloso, and Fei Fang. Explainable reinforcement learning: A survey and comparative review. *ACM Comput. Surv.*, 56(7):168:1–168:36, 2024.

[248] Stephanie Milani, Zhicheng Zhang, Nicholay Topin, Zheyuan Ryan Shi, Charles A. Kamhoua, Evangelos E. Papalexakis, and Fei Fang. MAVIPER: learning decision tree policies for interpretable multi-agent reinforcement learning. In Massih-Reza Amini, Stéphane Canu, Asja Fischer, Tias Guns, Petra Kralj Novak, and Grigorios Tsoumakas, editors, *Machine Learning and Knowledge Discovery in Databases - European Conference, ECML PKDD 2022, Grenoble, France, September 19-23, 2022, Proceedings, Part IV*, volume 13716 of *Lecture Notes in Computer Science*, pages 251–266. Springer, 2022.

[249] Julian F. Miller and Peter Thomson. Cartesian genetic programming. In Riccardo Poli, Wolfgang Banzhaf, William B. Langdon, Julian F. Miller, Peter Nordin, and Terence C. Fogarty, editors, *Genetic Programming, European Conference, EuroGP 2000, Edinburgh, Scotland, UK, April 15-16, 2000, Proceedings*, volume 1802 of *Lecture Notes in Computer Science*, pages 121–132. Springer, 2000.

[250] Aditi Mishra, Utkarsh Soni, Jinbin Huang, and Chris Bryan. Why? why not? when? visual explanations of agent behavior in reinforcement learning. *CoRR*, abs/2104.02818, 2021.

[251] Indrajeet Mishra, Giang Dao, and Minwoo Lee. Visual sparse bayesian reinforcement learning: A framework for interpreting what an agent has learned. In *IEEE Symposium Series on Computational Intelligence, SSCI 2018, Bangalore, India, November 18-21, 2018*, pages 1427–1434. IEEE, 2018.

[252] Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In Maria-Florina Balcan and Kilian Q. Weinberger, editors, *Proceedings of the 33nd International Conference on Machine Learning, ICML 2016, New York City, NY, USA, June 19-24, 2016*, volume 48 of *JMLR Workshop and Conference Proceedings*, pages 1928–1937. JMLR.org, 2016.

[253] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin A. Riedmiller, Andreas Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis. Human-level control through deep reinforcement learning. *Nat.*, 518(7540):529–533, 2015.

[254] Edward F Moore. Gedanken-experiments on sequential machines. *Automata studies*, 34:129–153, 1956.

[255] Konda Reddy Mopuri, Utsav Garg, and R. Venkatesh Babu. CNN fixations: An unraveling approach to visualize the discriminative image regions. *IEEE Trans. Image Process.*, 28(5):2116–2125, 2019.

[256] Alexander Mott, Daniel Zoran, Mike Chrzanowski, Daan Wierstra, and Danilo Jimenez Rezende. Towards interpretable reinforcement learning using attention augmented agents. In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence d'Alché-Buc, Emily B. Fox, and Roman Garnett, editors, *Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada*, pages 12329–12338, 2019.

[257] Sreerama K. Murthy, Simon Kasif, and Steven Salzberg. A system for induction of oblique decision trees. *J. Artif. Intell. Res.*, 2:1–32, 1994.

[258] Roger B. Myerson. Graphs and cooperation in games. *Math. Oper. Res.*, 2(3):225–229, 1977.

[259] Subramanya Nageshrao, Bruno Costa, and Dimitar P. Filev. Interpretable approximation of a deep reinforcement learning agent as a set of if-then rules. In M. Arif Wani, Taghi M. Khoshgoftaar, Dingding Wang, Huanjing Wang, and Naeem Seliya, editors, *18th IEEE International Conference On Machine Learning And Applications, ICMLA 2019, Boca Raton, FL, USA, December 16-19, 2019*, pages 216–221. IEEE, 2019.

[260] Mark A. Neerincx, Jasper van der Waa, Frank Kaptein, and Jurriaan van Diggelen. Using perceptual and cognitive explanations for enhanced human-agent team performance. In Don Harris, editor, *Engineering Psychology and Cognitive Ergonomics - 15th International Conference, EPCE 2018, Held as Part of HCI International 2018, Las Vegas, NV, USA, July 15-20, 2018, Proceedings*, volume 10906 of *Lecture Notes in Computer Science*, pages 204–214. Springer, 2018.

[261] Andrew Y. Ng and Stuart Russell. Algorithms for inverse reinforcement learning. In Pat Langley, editor, *Proceedings of the Seventeenth International Conference on Machine Learning (ICML 2000), Stanford University, Stanford, CA, USA, June 29 - July 2, 2000*, pages 663–670. Morgan Kaufmann, 2000.

[262] Kee Siong Ng. Alkemy: A learning system based on an expressive knowledge representation formalism. *submitted for publication*, 2004.

[263] Tri Minh Nguyen, Thomas P. Quinn, Thin Nguyen, and Truyen Tran. Counterfactual explanation with multi-agent reinforcement learning for drug target prediction. *CoRR*, abs/2103.12983, 2021.

[264] Xiaotong Nie, Motoaki Hiraga, and Kazuhiro Ohkura. Visualizing deep Q-learning to understanding behavior of swarm robotic system. In Hiroshi Sato, Saori Iwanaga, and Akira Ishii, editors, *Proceedings of the 23rd Asia Pacific Symposium on Intelligent and Evolutionary Systems, Tottori, Japan, December 6-8, 2019*, pages 118–129. Springer, 2019.

[265] Dmitry Nikulin, Anastasia Ianina, Vladimir Aliev, and Sergey I. Nikolenko. Free-lunch saliency via attention in Atari agents. In *2019 IEEE/CVF International Conference on Computer Vision Workshops, ICCV Workshops 2019, Seoul, Korea (South), October 27-28, 2019*, pages 4240–4249. IEEE, 2019.

[266] Michael Oberst and David A. Sontag. Counterfactual off-policy evaluation with Gumbel-Max structural causal models. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, *Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA*, volume 97 of *Proceedings of Machine Learning Research*, pages 4881–4890. PMLR, 2019.

[267] Matthew L. Olson, Roli Khanna, Lawrence Neal, Fuxin Li, and Weng-Keen Wong. Counterfactual state explanations for reinforcement learning agents via generative deep learning. *Artif. Intell.*, 295:103455, 2021.

[268] Liang Ou, Yu-Cheng Chang, Yu-Kai Wang, and Chin-Teng Lin. Fuzzy centered explainable network for reinforcement learning. *IEEE Trans. Fuzzy Syst.*, 32(1):203–213, 2024.

[269] Rohan R. Paleja, Yaru Niu, Andrew Silva, Chace Ritchie, Sugju Choi, and Matthew C. Gombolay. Learning interpretable, high-performing policies for autonomous driving. In Kris Hauser, Dylan A. Shell, and Shoudong Huang, editors, *Robotics: Science and Systems XVIII, New York City, NY, USA, June 27 - July 1, 2022*, 2022.

[270] Xinlei Pan, Xiangyu Chen, Qi-Zhi Cai, John F. Canny, and Fisher Yu. Semantic predictive control for explainable and efficient policy learning. In *International Conference on Robotics and Automation, ICRA 2019, Montreal, QC, Canada, May 20-24, 2019*, pages 3203–3209. IEEE, 2019.

[271] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In *Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, July 6-12, 2002, Philadelphia, PA, USA*, pages 311–318. ACL, 2002.

[272] Shubham Pateria, Budhitama Subagdja, Ah-Hwee Tan, and Chai Quek. Hierarchical reinforcement learning: A comprehensive survey. *ACM Comput. Surv.*, 54(5):109:1–109:35, 2022.

[273] Ali Payani and Faramarz Fekri. Inductive logic programming via differentiable deep neural logic networks. *CoRR*, abs/1906.03523, 2019.

[274] Ali Payani and Faramarz Fekri. Incorporating relational background knowledge into reinforcement learning via differentiable inductive logic programming. *CoRR*, abs/2003.10386, 2020.

[275] Michele Persiani and Thomas Hellström. The mirror agent model: A bayesian architecture for interpretable agent behavior. In Davide Calvaresi, Amro Najjar, Michael Winikoff, and Kary Främling, editors, *Explainable and Transparent AI and Multi-Agent Systems - 4th International Workshop, EXTRAAMAS 2022, Virtual Event, May 9-10, 2022, Revised Selected Papers*, volume 13283 of *Lecture Notes in Computer Science*, pages 111–123. Springer, 2022.

[276] Vitali Petsiuk, Abir Das, and Kate Saenko. RISE: randomized input sampling for explanation of black-box models. In *British Machine Vision Conference 2018, BMVC 2018, Newcastle, UK, September 3-6, 2018*, page 151. BMVA Press, 2018.

[277] Brittany Davis Pierson, Dustin Arendt, John Miller, and Matthew E. Taylor. Comparing explanations in RL. *Neural Comput. Appl.*, 36(1):505–516, 2024.

[278] Rey Pocius, Lawrence Neal, and Alan Fern. Strategic tasks for explainable reinforcement learning. In *The Thirty-Third AAAI Conference on Artificial Intelligence, AAAI 2019, The Thirty-First Innovative Applications of Artificial Intelligence Conference, IAAI 2019, The Ninth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2019, Honolulu, Hawaii, USA, January 27 - February 1, 2019*, pages 10007–10008. AAAI Press, 2019.

[279] Erika Puiutta and Eric M. S. P. Veith. Explainable reinforcement learning: A survey. In Andreas Holzinger, Peter Kieseberg, A Min Tjoa, and Edgar R. Weippl, editors, *Machine Learning and Knowledge Extraction - 4th IFIP TC 5, TC 12, WG 8.4, WG 8.9, WG 12.9 International Cross-Domain Conference, CD-MAKE 2020, Dublin, Ireland, August 25-28, 2020, Proceedings*, volume 12279 of *Lecture Notes in Computer Science*, pages 77–95. Springer, 2020.

[280] Nikaash Puri, Sukriti Verma, Piyush Gupta, Dhruv Kayastha, Shripad V. Deshmukh, Balaji Krishnamurthy, and Sameer Singh. Explain your move: Understanding agent actions using specific and relevant feature attribution. In *8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020*. OpenReview.net, 2020.

[281] J. Ross Quinlan. Induction of decision trees. *Mach. Learn.*, 1(1):81–106, 1986.

[282] Antonin Raffin, Ashley Hill, René Traoré, Timothée Lesort, Natalia Díaz Rodríguez, and David Filliat. S-RL toolbox: Environments, datasets and evaluation metrics for state representation learning. *CoRR*, abs/1809.09369, 2018.

[283] Sindre Benjamin Remman and Anastasios M. Lekkas. Robotic lever manipulation using hindsight experience replay and Shapley additive explanations. In *2021 European Control Conference, ECC 2021, Virtual Event / Delft, The Netherlands, June 29 - July 2, 2021*, pages 586–593. IEEE, 2021.

[284] Sindre Benjamin Remman, Inga Strümke, and Anastasios M. Lekkas. Causal versus marginal Shapley values for robotic lever manipulation controlled using deep reinforcement learning. In *American Control Conference, ACC 2022, Atlanta, GA, USA, June 8-10, 2022*, pages 2683–2690. IEEE, 2022.

[285] Marco Túlio Ribeiro, Sameer Singh, and Carlos Guestrin. "Why should I trust you?": Explaining the predictions of any classifier. In Balaji Krishnapuram, Mohak Shah, Alexander J. Smola, Charu C. Aggarwal, Dou Shen, and Rajeev Rastogi, editors, *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, San Francisco, CA, USA, August 13-17, 2016*, pages 1135–1144. ACM, 2016.

[286] Finn Rietz, Sven Magg, Fredrik Heintz, Todor Stoyanov, Stefan Wermter, and Johannes A. Stork. Hierarchical goals contextualize local reward decomposition explanations. *Neural Comput. Appl.*, 35(23):16693–16704, 2023.

[287] Stefano Giovanni Rizzo, Giovanna Vantini, and Sanjay Chawla. Reinforcement learning with explainability for traffic signal control. In *2019 IEEE Intelligent Transportation Systems Conference, ITSC 2019, Auckland, New Zealand, October 27-30, 2019*, pages 3567–3572. IEEE, 2019.

[288] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Nassir Navab, Joachim Hornegger, William M. Wells III, and Alejandro F. Frangi, editors, *Medical Image Computing and Computer-Assisted Intervention - MICCAI 2015 - 18th International Conference Munich, Germany, October 5 - 9, 2015, Proceedings, Part III*, volume 9351 of *Lecture Notes in Computer Science*, pages 234–241. Springer, 2015.

[289] Avi Rosenfeld. Better metrics for evaluating explainable artificial intelligence. In Frank Dignum, Alessio Lomuscio, Ulle Endriss, and Ann Nowé, editors, *AAMAS '21: 20th International Conference on Autonomous Agents and Multiagent Systems, Virtual Event, United Kingdom, May 3-7, 2021*, pages 45–50. ACM, 2021.

[290] Aaron M. Roth, Nicholay Topin, Pooyan Jamshidi, and Manuela Veloso. Conservative Q-improvement: Reinforcement learning for an interpretable decision-tree policy. *CoRR*, abs/1907.01180, 2019.

[291] Céline Rouveirol, Malik Kazi Aoual, Henry Soldano, and Véronique Ventos. Explaining optimal trajectories. In Anna Fensel, Ana Ozaki, Dumitru Roman, and Ahmet Soylu, editors, *Rules and Reasoning - 7th International Joint Conference, RuleML+RR 2023, Oslo, Norway, September 18-20, 2023, Proceedings*, volume 14244 of *Lecture Notes in Computer Science*, pages 206–221. Springer, 2023.

[292] Reuven Y Rubinstein. Optimization of computer simulation models with rare events. *European Journal of Operational Research*, 99(1):89–112, 1997.

[293] Christian Rupprecht, Cyril Ibrahim, and Christopher J. Pal. Finding and visualizing weaknesses of deep reinforcement learning agents. In *8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020*. OpenReview.net, 2020.

[294] Conor Ryan, J. J. Collins, and Michael O'Neill. Grammatical evolution: Evolving programs for an arbitrary language. In Wolfgang Banzhaf, Riccardo Poli, Marc Schoenauer, and Terence C. Fogarty, editors, *Genetic Programming, First European Workshop, EuroGP'98, Paris, France, April 14-15, 1998, Proceedings*, volume 1391 of *Lecture Notes in Computer Science*, pages 83–96. Springer, 1998.

[295] Emre Saldiran, Mehmet Hasanzade, Gokhan Inalhan, and Antonios Tsourdos. Towards global explainability of artificial intelligence agent tactics in close air combat. *Aerospace*, 11(6):415, 2024.

[296] Amir Samadi, Konstantinos Koufos, Kurt Debattista, and Mehrdad Dianati. SAFE-RL: saliency-aware counterfactual explainer for deep reinforcement learning policies. *CoRR*, abs/2404.18326, 2024.

[297] Robert-Florian Samoilescu, Arnaud Van Looveren, and Janis Klaise. Model-agnostic and scalable counterfactual explanations via reinforcement learning. *CoRR*, abs/2106.02597, 2021.

[298] Léo Saulières, Martin C. Cooper, and Florence Dupin de Saint-Cyr. Predicate-based explanation of a reinforcement learning agent via action importance evaluation. In Rosa Meo and Fabrizio Silvestri, editors, *Machine Learning and Principles and Practice of Knowledge Discovery in Databases - International Workshops of ECML PKDD 2023, Turin, Italy, September 18-22, 2023, Revised Selected Papers, Part I*, volume 2133 of *Communications in Computer and Information Science*, pages 21–37. Springer, 2023.

[299] Léo Saulières, Martin C. Cooper, and Florence Dupin de Saint-Cyr. Predictive explanations for and by reinforcement learning. In Ana Paula Rocha, Luc Steels, and H. Jaap van den Herik, editors, *Agents and Artificial Intelligence - 15th International Conference, ICAART 2023, Lisbon, Portugal, February 22-24, 2023, Revised Selected Papers*, volume 14546 of *Lecture Notes in Computer Science*, pages 115–140. Springer, 2023.

[300] Léo Saulières, Martin C. Cooper, and Florence Dupin de Saint-Cyr. Backward explanations via redefinition of predicates. In Ulle Endriss, Francisco S. Melo, Kerstin Bach, Alberto José Bugarín Diz, Jose Maria Alonso-Moral, Senén Barro, and Fredrik Heintz, editors, *ECAI 2024 - 27th European Conference on Artificial Intelligence, 19-24 October 2024, Santiago de Compostela, Spain - Including 13th Conference on Prestigious Applications of Intelligent Systems (PAIS 2024)*, volume 392 of *Frontiers in Artificial Intelligence and Applications*, pages 786–793. IOS Press, 2024.

[301] Jonathan Scholz, Martin Levihn, Charles Lee Isbell Jr., and David Wingate. A physics-based model prior for object-oriented MDPs. In Proceedings of the 31th International Conference on Machine Learning, ICML 2014, Beijing, China, 21-26 June 2014, volume 32 of JMLR Workshop and Conference Proceedings, pages 1089–1097. JMLR.org, 2014.

[302] L Schreiber, G de O Ramos, and Ana LC Bazzan. Towards explainable deep reinforcement learning for traffic signal control. In Proc. of LatinX in AI Workshop@ ICML 2021, LXAI. LXIA, 2021.

[303] John Schulman, Sergey Levine, Pieter Abbeel, Michael I. Jordan, and Philipp Moritz. Trust region policy optimization. In Francis R. Bach and David M. Blei, editors, Proceedings of the 32nd International Conference on Machine Learning, ICML 2015, Lille, France, 6-11 July 2015, volume 37 of JMLR Workshop and Conference Proceedings, pages 1889–1897. JMLR.org, 2015.

[304] Bastian Seegebarth, Felix Müller, Bernd Schattenberg, and Susanne Biundo. Making hybrid plans more clear to human users - A formal approach for generating sound explanations. In Lee McCluskey, Brian Charles Williams, José Reinaldo Silva, and Blai Bonet, editors, Proceedings of the Twenty-Second International Conference on Automated Planning and Scheduling, ICAPS 2012, Atibaia, São Paulo, Brazil, June 25-19, 2012. AAAI, 2012.

[305] Frank Sehnke, Christian Osendorfer, Thomas Rückstieß, Alex Graves, Jan Peters, and Jürgen Schmidhuber. Policy gradients with parameter-based exploration for control. In Vera Kurková, Roman Neruda, and Jan Koutnı́k, editors, Artificial Neural Networks - ICANN 2008 , 18th International Conference, Prague, Czech Republic, September 3-6, 2008, Proceedings, Part I, volume 5163 of Lecture Notes in Computer Science, pages 387–396. Springer, 2008.

[306] Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-CAM: Visual explanations from deep networks via gradient-based localization. In IEEE International Conference on Computer Vision, ICCV 2017, Venice, Italy, October 22-29, 2017, pages 618– 626. IEEE Computer Society, 2017.

[307] Yael Septon, Tobias Huber, Elisabeth André, and Ofra Amir. Integrating policy summaries with reward decomposition for explaining reinforcement learning agents. In Philippe Mathieu, Frank Dignum, Paulo Novais, and Fernando de la Prieta, editors, Advances in Practical Applications of Agents, Multi-Agent Systems, and Cognitive Mimetics. The PAAMS Collection - 21st International Conference, PAAMS 2023, Guimarães, Portugal, July 12-14, 2023, Proceedings, volume 13955 of Lecture Notes in Computer Science, pages 320–332. Springer, 2023.

[308] Pedro Sequeira and Melinda T. Gervasio. Interestingness elements for explainable reinforcement learning: Understanding agents’ capabilities and limitations. Artif. Intell., 288:103367, 2020.

[309] Pedro Sequeira and Melinda T. Gervasio. IxDRL: A novel explainable deep reinforcement learning toolkit based on analyses of interestingness. In Luca Longo, editor, Explainable Artificial Intelligence - First World Conference, xAI 2023, Lisbon, Portugal, July 26-28, 2023, Proceedings, Part I, volume 1901 of Communications in Computer and Information Science, pages 373–396. Springer, 2023.

[310] Lloyd S Shapley et al. A value for n-person games. Contributions to the Theory of Games, Volume II, 1953.

[311] Wenjie Shi, Gao Huang, Shiji Song, Zhuoyuan Wang, Tingyu Lin, and Cheng Wu. Self-supervised discovering of interpretable features for reinforcement learning. IEEE Trans. Pattern Anal. Mach. Intell., 44(5):2712– 2724, 2022.

[312] Avanti Shrikumar, Peyton Greenside, Anna Shcherbina, and Anshul Kundaje. Not just a black box: Learning important features through propagating activation differences. CoRR, abs/1605.01713, 2016.

[313] Tianmin Shu, Caiming Xiong, and Richard Socher. Hierarchical and interpretable skill acquisition in multitask reinforcement learning. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net, 2018.

[314] Alexander Sieusahai and Matthew Guzdial. Explaining deep reinforcement learning agents in the Atari domain through a surrogate model. In David Thue and Stephen G. Ware, editors, Proceedings of the Seventeenth AAAI Conference on Artificial Intelligence 62 and Interactive Digital Entertainment, AIIDE 2021, virtual, October 11-15, 2021, pages 82–90. AAAI Press, 2021.

[315] Andrew Silva, Taylor Killian, Ivan Dario Jimenez Rodriguez, Sung-Hyun Son, and Matthew Gombolay. Optimization methods for interpretable differentiable decision trees in reinforcement learning, 2019.

[316] Julian Skirzynski, Frederic Becker, and Falk Lieder. Automatic discovery of interpretable planning strategies. Mach. Learn., 110(9):2641–2683, 2021.

[317] Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda B. Viégas, and Martin Wattenberg. Smoothgrad: removing noise by adding noise. CoRR, abs/1706.03825, 2017.

[318] Jasper Snoek, Hugo Larochelle, and Ryan P. Adams. Practical bayesian optimization of machine learning algorithms. In Peter L. Bartlett, Fernando C. N. Pereira, Christopher J. C. Burges, Léon Bottou, and Kilian Q. Weinberger, editors, Advances in Neural Information Processing Systems 25: 26th Annual Conference on Neural Information Processing Systems 2012. Proceedings of a meeting held December 3-6, 2012, Lake Tahoe, Nevada, United States, pages 2960–2968, 2012.

[319] Eduardo A. Soares, Plamen P. Angelov, Bruno Costa, Marcos Castro, Subramanya Nageshrao, and Dimitar P. Filev. Explaining deep learning models through rulebased approximation and visualization. IEEE Trans. Fuzzy Syst., 29(8):2399–2407, 2021.

[320] Kacper Sokol and Peter A. Flach. Explainability fact sheets: a framework for systematic assessment of explainable approaches. In Mireille Hildebrandt, Carlos Castillo, L. Elisa Celis, Salvatore Ruggieri, Linnet Taylor, and Gabriela Zanfir-Fortuna, editors, FAT* ’20: Conference on Fairness, Accountability, and Transparency, Barcelona, Spain, January 27-30, 2020, pages 56–67. ACM, 2020.

[321] Zhihao Song, Yunpeng Jiang, Jianyi Zhang, Paul Weng, Dong Li, Wulong Liu, and Jianye Hao. An interpretable deep reinforcement learning approach to autonomous driving. In IJCAI Workshop on Artificial Intelligence for Automous Driving, 2022.

[322] Utkarsh Soni. Towards more accessible human-AI interactions in sequential decision-making tasks. Technical report, Arizona State University, 2024.

[323] Sarath Sreedharan, Tathagata Chakraborti, and Subbarao Kambhampati. Handling model uncertainty and multiplicity in explanations via model reconciliation. In Mathijs de Weerdt, Sven Koenig, Gabriele Röger, and Matthijs T. J. Spaan, editors, Proceedings of the Twenty-Eighth International Conference on Automated Planning and Scheduling, ICAPS 2018, Delft, The Netherlands, June 24-29, 2018, pages 518–526. AAAI Press, 2018.

[324] Sarath Sreedharan, Utkarsh Soni, Mudit Verma, Siddharth Srivastava, and Subbarao Kambhampati. Bridging the gap: Providing post-hoc symbolic explanations for sequential decision-making problems with inscrutable representations. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022.

[325] Sarath Sreedharan, Siddharth Srivastava, and Subbarao Kambhampati. Tldr: Policy summarization for factored SSP problems using temporal abstractions. In J. Christopher Beck, Olivier Buffet, Jörg Hoffmann, Erez Karpas, and Shirin Sohrabi, editors, Proceedings of the Thirtieth International Conference on Automated Planning and Scheduling, Nancy, France, October 2630, 2020, pages 272–280. AAAI Press, 2020.

[326] Sarath Sreedharan, Siddharth Srivastava, David E. Smith, and Subbarao Kambhampati. Why can’t you do that HAL? explaining unsolvability of planning tasks. In Sarit Kraus, editor, Proceedings of the TwentyEighth International Joint Conference on Artificial Intelligence, IJCAI 2019, Macao, China, August 1016, 2019, pages 1422–1430. ijcai.org, 2019.

[327] Mohan Sridharan, Michael Gelfond, Shiqi Zhang, and Jeremy L. Wyatt. REBA: A refinement-based architecture for knowledge representation and reasoning in robotics. J. Artif. Intell. Res., 65:87–180, 2019.

[328] Srivatsan Srinivasan and Finale Doshi-Velez. Interpretable batch IRL to extract clinician goals in ICU hypotension management. AMIA Summits on Translational Science Proceedings, 2020:636, 2020.

[329] Mark Stefik, Michael Youngblood, Peter Pirolli, Christian Lebiere, Robert Thomson, Robert Price, Lester D Nelson, Robert Krivacic, Jacob Le, Konstantinos Mitsopoulos, et al. Explaining autonomous drones: An XAI journey. Applied AI Letters, 2(4):e54, 2021.

[330] Gregory J. Stein. Generating high-quality explanations for navigation in partially-revealed environments. In Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan, editors, Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 17493–17506, 2021.

[331] Marcel Steinmetz, Daniel Fiser, Hasan Ferit Eniser, Patrick Ferber, Timo P. Gros, Philippe Heim, Daniel Höller, Xandra Schuler, Valentin Wüstholz, Maria Christakis, and Jörg Hoffmann. Debugging a policy: Automatic action-policy testing in AI planning. In 63 Akshat Kumar, Sylvie Thiébaux, Pradeep Varakantham, and William Yeoh, editors, Proceedings of the Thirty-Second International Conference on Automated Planning and Scheduling, ICAPS 2022, Singapore (virtual), June 13-24, 2022, pages 353–361. AAAI Press, 2022.

[332] Roykrong Sukkerd, Reid G. Simmons, and David Garlan. Towards explainable multi-objective probabilistic planning. In Tomás Bures, John S. Fitzgerald, Bradley R. Schmerl, and Danny Weyns, editors, Proceedings of the 4th International Workshop on Software Engineering for Smart Cyber-Physical Systems, ICSE 2018, Gothenburg, Sweden, May 27, 2018, pages 19–25. ACM, 2018.

[333] Roykrong Sukkerd, Reid G. Simmons, and David Garlan. Tradeoff-focused contrastive explanation for MDP planning. In 29th IEEE International Conference on Robot and Human Interactive Communication, ROMAN 2020, Naples, Italy, August 31 - September 4, 2020, pages 1041–1048. IEEE, 2020.

[334] Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In Doina Precup and Yee Whye Teh, editors, Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017, volume 70 of Proceedings of Machine Learning Research, pages 3319–3328. PMLR, 2017.

[335] Richard S. Sutton, Joseph Modayil, Michael Delp, Thomas Degris, Patrick M. Pilarski, Adam White, and Doina Precup. Horde: a scalable real-time architecture for learning knowledge from unsupervised sensorimotor interaction. In Liz Sonenberg, Peter Stone, Kagan Tumer, and Pinar Yolum, editors, 10th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2011), Taipei, Taiwan, May 2-6, 2011, Volume 1-3, pages 761–768. IFAAMAS, 2011.

[336] Richard S. Sutton, Doina Precup, and Satinder Singh. Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning. Artif. Intell., 112(1-2):181–211, 1999.

[337] Aaquib Tabrez and Bradley Hayes. Improving humanrobot interaction through explainable reinforcement learning. In 14th ACM/IEEE International Conference on Human-Robot Interaction, HRI 2019, Daegu, South Korea, March 11-14, 2019, pages 751–753. IEEE, 2019.

[338] Hao Tang, Hong Liu, Dan Xu, Philip H. S. Torr, and Nicu Sebe. Attentiongan: Unpaired image-to-image translation using attention-guided generative adversarial networks. IEEE Trans. Neural Networks Learn. Syst., 34(4):1972–1987, 2023.

[339] Yujin Tang, Duong Nguyen, and David Ha. Neuroevolution of self-interpretable agents. In Carlos Artemio Coello Coello, editor, GECCO ’20: Genetic and Evolutionary Computation Conference, Cancún Mexico, July 8-12, 2020, pages 414–424. ACM, 2020.

[340] Geraud Nangue Tasse, Steven D. James, and Benjamin Rosman. A boolean task algebra for reinforcement learning. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.

[341] Philipp Theumer, Florian Edenhofner, Roland Zimmermann, and Alexander Zipfel. Explainable deep reinforcement learning for production control. In Proceedings of the Conference on Production Systems and Logistics: CPSL 2022, pages 809–818. Hannover: publishIng., 2022.

[342] Andrea Lockerd Thomaz, Guy Hoffman, and Cynthia Breazeal. Real-time interactive reinforcement learning for robots. In AAAI 2005 workshop on human comprehensible machine learning, volume 3, page 1, 2005.

[343] Emanuel Todorov. Compositionality of optimal control laws. In Yoshua Bengio, Dale Schuurmans, John D. Lafferty, Christopher K. I. Williams, and Aron Culotta, editors, Advances in Neural Information Processing Systems 22: 23rd Annual Conference on Neural Information Processing Systems 2009. Proceedings of a meeting held 7-10 December 2009, Vancouver, British Columbia, Canada, pages 1856–1864. Curran Associates, Inc., 2009.

[344] Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, IROS 2012, Vilamoura, Algarve, Portugal, October 7-12, 2012, pages 5026–5033. IEEE, 2012.

[345] Nicholay Topin, Stephanie Milani, Fei Fang, and Manuela Veloso. Iterative bounding MDPs: Learning interpretable policies via non-interpretable methods. In Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, Thirty-Third Conference on Innovative Applications of Artificial Intelligence, IAAI 2021, The Eleventh Symposium on Educational Advances in Artificial Intelligence, EAAI 2021, Virtual Event, February 2-9, 2021, pages 9923–9931. AAAI Press, 2021.

[346] Nicholay Topin and Manuela Veloso. Generation of policy-level explanations for reinforcement learning. In The Thirty-Third AAAI Conference on Artificial Intelligence, AAAI 2019, The Thirty-First Innovative 64 Applications of Artificial Intelligence Conference, IAAI 2019, The Ninth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2019, Honolulu, Hawaii, USA, January 27 - February 1, 2019, pages 2514–2521. AAAI Press, 2019.

[347] Mark Towers, Yali Du, Christopher T. Freeman, and Timothy J. Norman. Explaining an agent’s future beliefs through temporally decomposing future reward estimators. In Ulle Endriss, Francisco S. Melo, Kerstin Bach, Alberto José Bugarı́n Diz, Jose Maria AlonsoMoral, Senén Barro, and Fredrik Heintz, editors, ECAI 2024 - 27th European Conference on Artificial Intelligence, 19-24 October 2024, Santiago de Compostela, Spain - Including 13th Conference on Prestigious Applications of Intelligent Systems (PAIS 2024), volume 392 of Frontiers in Artificial Intelligence and Applications, pages 2790–2797. IOS Press, 2024.

[348] Dweep Trivedi, Jesse Zhang, Shao-Hua Sun, and Joseph J. Lim. Learning to synthesize programs as interpretable and generalizable policies. In Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan, editors, Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 25146–25163, 2021.

[349] Stratis Tsirtsis, Abir De, and Manuel Rodriguez. Counterfactual explanations in sequential decision making under uncertainty. In Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan, editors, Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 30127–30139, 2021.

[350] Stratis Tsirtsis and Manuel Rodriguez. Finding counterfactually optimal action sequences in continuous state spaces. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023.

[351] Yuta Tsuchiya, Yasuhide Mori, and Masashi Egi. Explainable reinforcement learning based on Q-value decomposition by expected state transitions. In Andreas Martin, Hans-Georg Fill, Aurona Gerber, Knut Hinkelmann, Doug Lenat, Reinhard Stolle, and Frank van Harmelen, editors, Proceedings of the AAAI 2023 Spring Symposium on Challenges Requiring the Combination of Machine Learning and Knowledge Engineering (AAAI-MAKE 2023), Hyatt Regency, San Francisco Airport, California, USA, March 27-29, 2023, volume 3433 of CEUR Workshop Proceedings. CEURWS.org, 2023.

[352] Juan Marcelo Parra Ullauri, Antonio Garcı́aDomı́nguez, Nelly Bencomo, Changgang Zheng, Chen Zhen, Juan Boubeta-Puig, Guadalupe Ortiz, and Shufan Yang. Event-driven temporal models for explanations - ETeMoX: explaining reinforcement learning. Softw. Syst. Model., 21(3):1091–1113, 2022.

[353] Paul E. Utgoff, Neil C. Berkman, and Jeffery A. Clouse. Decision tree induction based on efficient tree restructuring. Mach. Learn., 29(1):5–44, 1997.

[354] William T. B. Uther and Manuela M. Veloso. Tree based discretization for continuous state space reinforcement learning. In Jack Mostow and Chuck Rich, editors, Proceedings of the Fifteenth National Conference on Artificial Intelligence and Tenth Innovative Applications of Artificial Intelligence Conference, AAAI 98, IAAI 98, July 26-30, 1998, Madison, Wisconsin, USA, pages 769–774. AAAI Press / The MIT Press, 1998.

[355] Karthik Valmeekam, Sarath Sreedharan, Sailik Sengupta, and Subbarao Kambhampati. RADAR-X: an interactive interface pairing contrastive explanations with revised plan suggestions. In Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, Thirty-Third Conference on Innovative Applications of Artificial Intelligence, IAAI 2021, The Eleventh Symposium on Educational Advances in Artificial Intelligence, EAAI 2021, Virtual Event, February 2-9, 2021, pages 16051–16053. AAAI Press, 2021.

[356] Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-SNE. Journal of machine learning research, 9(11), 2008.

[357] Jasper van der Waa, Jurriaan van Diggelen, Karel van den Bosch, and Mark A. Neerincx. Contrastive explanations for reinforcement learning in terms of expected consequences. CoRR, abs/1807.08706, 2018.

[358] Martijn van Otterlo. Solving relational and firstorder logical Markov decision processes: A survey. In Marco A. Wiering and Martijn van Otterlo, editors, Reinforcement Learning, volume 12 of Adaptation, Learning, and Optimization, pages 253–292. Springer, 2012.

[359] Connor van Rossum, Candice Feinberg, Adam Abu Shumays, Kyle Baxter, and Benedek Bartha. A novel approach to curiosity and explainable reinforcement learning via interpretable sub-goals. CoRR, abs/2104.06630, 2021.

[360] Varun Ravi Varma. Interpretable reinforcement learning with the regression Tsetlin machine, 2021.

[361] Marko Vasic, Andrija Petrovic, Kaiyuan Wang, Mladen Nikolic, Rishabh Singh, and Sarfraz Khurshid. Moët: Mixture of expert trees and its application to verifiable reinforcement learning. Neural Networks, 151:34–47, 2022.

[362] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett, editors, Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 5998–6008, 2017.

[363] Rishi Veerapaneni, John D. Co-Reyes, Michael Chang, Michael Janner, Chelsea Finn, Jiajun Wu, Joshua B. Tenenbaum, and Sergey Levine. Entity abstraction in visual model-based reinforcement learning. In Leslie Pack Kaelbling, Danica Kragic, and Komei Sugiura, editors, 3rd Annual Conference on Robot Learning, CoRL 2019, Osaka, Japan, October 30 - November 1, 2019, Proceedings, volume 100 of Proceedings of Machine Learning Research, pages 1439–1456. PMLR, 2019.

[364] Abhinav Verma, Hoang Minh Le, Yisong Yue, and Swarat Chaudhuri. Imitation-projected programmatic reinforcement learning. In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence d’Alché-Buc, Emily B. Fox, and Roman Garnett, editors, Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pages 15726–15737, 2019.

[365] Abhinav Verma, Vijayaraghavan Murali, Rishabh Singh, Pushmeet Kohli, and Swarat Chaudhuri. Programmatically interpretable reinforcement learning. In Jennifer G. Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018, volume 80 of Proceedings of Machine Learning Research, pages 5052–5061. PMLR, 2018.

[366] Oriol Vinyals, Timo Ewalds, Sergey Bartunov, Petko Georgiev, Alexander Sasha Vezhnevets, Michelle Yeo, Alireza Makhzani, Heinrich Küttler, John P. Agapiou, Julian Schrittwieser, John Quan, Stephen Gaffney, Stig Petersen, Karen Simonyan, Tom Schaul, Hado van Hasselt, David Silver, Timothy P. Lillicrap, Kevin Calderone, Paul Keet, Anthony Brunasso, David Lawrence, Anders Ekermo, Jacob Repp, and Rodney Tsing. Starcraft II: A new challenge for reinforcement learning. CoRR, abs/1708.04782, 2017.

[367] Marcel Vinzent, Marcel Steinmetz, and Jörg Hoffmann. Neural network action policy verification via predicate abstraction. In Akshat Kumar, Sylvie Thiébaux, Pradeep Varakantham, and William Yeoh, editors, Proceedings of the Thirty-Second International Conference on Automated Planning and Scheduling, ICAPS 2022, Singapore (virtual), June 13-24, 2022, pages 371–379. AAAI Press, 2022.

[368] Sergei Volodin. CauseOccam: Learning interpretable abstract representations in reinforcement learning environments via model sparsity. Master’s thesis, École Polytechnique Fédérale de Lausanne, 2021.

[369] George A. Vouros. Explainable deep reinforcement learning: State of the art and challenges. ACM Comput. Surv., 55(5):92:1–92:39, 2023.

[370] Stephan Wäldchen, Sebastian Pokutta, and Felix Huber. Training characteristic functions with reinforcement learning: XAI-methods play connect four. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvári, Gang Niu, and Sivan Sabato, editors, International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pages 22457–22474. PMLR, 2022.

[371] Trevor Walker, Lisa Torrey, Jude W. Shavlik, and Richard Maclin. Building relational world models for reinforcement learning. In Hendrik Blockeel, Jan Ramon, Jude W. Shavlik, and Prasad Tadepalli, editors, Inductive Logic Programming, 17th International Conference, ILP 2007, Corvallis, OR, USA, June 19-21, 2007, Revised Selected Papers, volume 4894 of Lecture Notes in Computer Science, pages 280–291. Springer, 2007.

[372] Thomas J Walsh. Efficient learning of relational models for sequential decision making. Rutgers The State University of New Jersey, School of Graduate Studies, 2010.

[373] Junpeng Wang, Liang Gou, Han-Wei Shen, and Hao Yang. DQNViz: A visual analytics approach to understand deep Q-networks. IEEE Trans. Vis. Comput. Graph., 25(1):288–298, 2019.

[374] Ning Wang, David V. Pynadath, and Susan G. Hill. The impact of POMDP-generated explanations on trust and performance in human-robot teams. In Catholijn M. Jonker, Stacy Marsella, John Thangarajah, and Karl Tuyls, editors, Proceedings of the 2016 International Conference on Autonomous Agents & Multiagent Systems, Singapore, May 9-13, 2016, pages 997–1005. ACM, 2016.

[375] Tingwu Wang, Renjie Liao, Jimmy Ba, and Sanja Fidler. Nervenet: Learning structured policy with graph neural networks. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net, 2018.

[376] Xinzhi Wang, Shengcheng Yuan, Hui Zhang, Michael Lewis, and Katia P. Sycara. Verbal explanations for deep reinforcement learning neural networks with attention on extracted features. In 28th IEEE International Conference on Robot and Human Interactive Communication, RO-MAN 2019, New Delhi, India, October 14-18, 2019, pages 1–7. IEEE, 2019.

[377] Xiting Wang, Yiru Chen, Jie Yang, Le Wu, Zhengtao Wu, and Xing Xie. A reinforcement learning framework for explainable recommendation. In IEEE International Conference on Data Mining, ICDM 2018, Singapore, November 17-20, 2018, pages 587–596. IEEE Computer Society, 2018.

[378] Yuyao Wang, Masayoshi Mase, and Masashi Egi. Attribution-based salience method towards interpretable reinforcement learning. In Andreas Martin, Knut Hinkelmann, Hans-Georg Fill, Aurona Gerber, Doug Lenat, Reinhard Stolle, and Frank van Harmelen, editors, Proceedings of the AAAI 2020 Spring Symposium on Combining Machine Learning and Knowledge Engineering in Practice, AAAI-MAKE 2020, Palo Alto, CA, USA, March 23-25, 2020, Volume I, volume 2600 of CEUR Workshop Proceedings. CEUR-WS.org, 2020.

[379] Zheng Wang and Shen Wang. Xrouting: Explainable vehicle rerouting for urban road congestion avoidance using deep reinforcement learning. In IEEE International Smart Cities Conference, ISC2 2022, Pafos, Cyprus, September 26-29, 2022, pages 1–7. IEEE, 2022.

[380] Laurens Weitkamp, Elise van der Pol, and Zeynep Akata. Visual rationalizations in deep reinforcement learning for Atarigames. CoRR, abs/1902.00566, 2019.

[381] Lindsay Wells and Tomasz Bednarz. Explainable AI and reinforcement learning - A systematic review of current approaches and trends. Frontiers Artif. Intell., 4:550030, 2021.

[382] Dennis G. Wilson, Sylvain Cussat-Blanc, Hervé Luga, and Julian F. Miller. Evolving simple programs for playing Atari games. In Hernán E. Aguirre and Keiki Takadama, editors, Proceedings of the Genetic and Evolutionary Computation Conference, GECCO 2018, Kyoto, Japan, July 15-19, 2018, pages 229–236. ACM, 2018.

[383] Salomón Wollenstein-Betech, Christian Muise, Christos G. Cassandras, Ioannis Ch. Paschalidis, and Yasaman Khazaeni. Explainability of intelligent transportation systems using knowledge compilation: a traffic light controller case. In 23rd IEEE International Conference on Intelligent Transportation Systems, ITSC 2020, Rhodes, Greece, September 20-23, 2020, pages 1–6. IEEE, 2020.

[384] Bohan Wu, Jayesh K. Gupta, and Mykel J. Kochenderfer. Model primitive hierarchical lifelong reinforcement learning. In Edith Elkind, Manuela Veloso, Noa Agmon, and Matthew E. Taylor, editors, Proceedings of the 18th International Conference on Autonomous Agents and MultiAgent Systems, AAMAS ’19, Montreal, QC, Canada, May 13-17, 2019, pages 34–42. International Foundation for Autonomous Agents and Multiagent Systems, 2019.

[385] Yu Xiong, Zhipeng Hu, Ye Huang, Runze Wu, Kai Guan, Xingchen Fang, Ji Jiang, Tianze Zhou, Yujing Hu, Haoyu Liu, Tangjie Lyu, and Changjie Fan. XRLBench: A benchmark for evaluating and comparing explainable reinforcement learning techniques. CoRR, abs/2402.12685, 2024.

[386] Duo Xu and Faramarz Fekri. Interpretable modelbased hierarchical reinforcement learning using inductive logic programming. CoRR, abs/2106.11417, 2021.

[387] Zhe Xu, Ivan Gavran, Yousef Ahmad, Rupak Majumdar, Daniel Neider, Ufuk Topcu, and Bo Wu. Joint inference of reward machines and policies for reinforcement learning. In J. Christopher Beck, Olivier Buffet, Jörg Hoffmann, Erez Karpas, and Shirin Sohrabi, editors, Proceedings of the Thirtieth International Conference on Automated Planning and Scheduling, Nancy, France, October 26-30, 2020, pages 590–598. AAAI Press, 2020.

[388] Dandan Yan. Research on reinforcement learning explainable strategies based on advantage saliency. Frontiers in Computing and Intelligent Systems, 3(1):124– 129, 2023.

[389] Bin-Bin Yang, Song-Qing Shen, and Wei Gao. Weighted oblique decision trees. In The Thirty-Third AAAI Conference on Artificial Intelligence, AAAI 2019, The Thirty-First Innovative Applications of Artificial Intelligence Conference, IAAI 2019, The Ninth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2019, Honolulu, Hawaii, USA, January 27 - February 1, 2019, pages 5621–5627. AAAI Press, 2019.

[390] Fan Yang, Mengnan Du, and Xia Hu. Evaluating explanation without ground truth in interpretable machine learning. CoRR, abs/1907.06831, 2019.

[391] Fangkai Yang, Daoming Lyu, Bo Liu, and Steven Gustafson. PEORL: integrating symbolic planning and 67 hierarchical reinforcement learning for robust decisionmaking. In Jérôme Lang, editor, Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence, IJCAI 2018, July 13-19, 2018, Stockholm, Sweden, pages 4860–4866. ijcai.org, 2018.

[392] Qisen Yang, Huanqian Wang, Mukun Tong, Wenjie Shi, Gao Huang, and Shiji Song. Leveraging reward consistency for interpretable feature discovery in reinforcement learning. IEEE Trans. Syst. Man Cybern. Syst., 54(2):1014–1025, 2024.

[393] Yongxin Yang, Irene Garcia Morillo, and Timothy M. Hospedales. Deep neural decision trees. CoRR, abs/1806.06988, 2018.

[394] Zhao Yang, Song Bai, Li Zhang, and Philip H. S. Torr. Learn to interpret Atari agents. CoRR, abs/1812.11276, 2018.

[395] Herman Yau, Chris Russell, and Simon Hadfield. What did you think would happen? explaining agent behaviour through intended outcomes. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.

[396] Zhitao Ying, Dylan Bourgeois, Jiaxuan You, Marinka Zitnik, and Jure Leskovec. GNNExplainer: Generating explanations for graph neural networks. In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence d’Alché-Buc, Emily B. Fox, and Roman Garnett, editors, Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pages 9240–9251, 2019.

[397] Sung Wook Yoon, Alan Fern, and Robert Givan. FFReplan: A baseline for probabilistic planning. In Mark S. Boddy, Maria Fox, and Sylvie Thiébaux, editors, Proceedings of the Seventeenth International Conference on Automated Planning and Scheduling, ICAPS 2007, Providence, Rhode Island, USA, September 22-26, 2007, page 352. AAAI, 2007.

[398] Håkan LS Younes. Exploding blocksworld.

[399] Zhongwei Yu, Jingqing Ruan, and Dengpeng Xing. Explainable reinforcement learning via a causal world model. In Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, IJCAI 2023, 19th-25th August 2023, Macao, SAR, China, pages 4540–4548. ijcai.org, 2023.

[400] Tom Zahavy, Nir Ben-Zrihem, and Shie Mannor. Graying the black box: Understanding DQNs. In MariaFlorina Balcan and Kilian Q. Weinberger, editors, Proceedings of the 33nd International Conference on Machine Learning, ICML 2016, New York City, NY, USA, June 19-24, 2016, volume 48 of JMLR Workshop and Conference Proceedings, pages 1899–1908. JMLR.org, 2016.

[401] Vinı́cius Flores Zambaldi, David Raposo, Adam Santoro, Victor Bapst, Yujia Li, Igor Babuschkin, Karl Tuyls, David P. Reichert, Timothy P. Lillicrap, Edward Lockhart, Murray Shanahan, Victoria Langston, Razvan Pascanu, Matthew M. Botvinick, Oriol Vinyals, and Peter W. Battaglia. Relational deep reinforcement learning. CoRR, abs/1806.01830, 2018.

[402] Amber E. Zelvelder, Marcus Westberg, and Kary Främling. Assessing explainability in reinforcement learning. In Davide Calvaresi, Amro Najjar, Michael Winikoff, and Kary Främling, editors, Explainable and Transparent AI and Multi-Agent Systems - Third International Workshop, EXTRAAMAS 2021, Virtual Event, May 3-7, 2021, Revised Selected Papers, volume 12688 of Lecture Notes in Computer Science, pages 223–240. Springer, 2021.

[403] Yan Zeng, Ruichu Cai, Fuchun Sun, Libo Huang, and Zhifeng Hao. A survey on causal reinforcement learning. CoRR, abs/2302.05209, 2023.

[404] Amy Zhang, Sainbayar Sukhbaatar, Adam Lerer, Arthur Szlam, and Rob Fergus. Composable planning with attributes. In Jennifer G. Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018, volume 80 of Proceedings of Machine Learning Research, pages 5837–5846. PMLR, 2018.

[405] Hengzhe Zhang, Aimin Zhou, and Xin Lin. Interpretable policy derivation for reinforcement learning based on evolutionary feature synthesis. Complex & Intelligent Systems, 6:741–753, 2020.

[406] Ke Zhang, Jun Jason Zhang, Peidong Xu, Tianlu Gao, and David Wenzhong Gao. Explainable AI in deep reinforcement learning models for power system emergency control. IEEE Trans. Comput. Soc. Syst., 9(2):419–427, 2022.

[407] Li Zhang, Xin Li, Mingzhong Wang, and Andong Tian. Off-policy differentiable logic reinforcement learning. In Nuria Oliver, Fernando Pérez-Cruz, Stefan Kramer, Jesse Read, and José Antonio Lozano, editors, Machine Learning and Knowledge Discovery in Databases. Research Track - European Conference, ECML PKDD 68 2021, Bilbao, Spain, September 13-17, 2021, Proceedings, Part II, volume 12976 of Lecture Notes in Computer Science, pages 617–632. Springer, 2021.

[408] Qiyuan Zhang, Xiaoteng Ma, Yiqin Yang, Chenghao Li, Jun Yang, Yu Liu, and Bin Liang. Learning to discover task-relevant features for interpretable reinforcement learning. IEEE Robotics Autom. Lett., 6(4):6601–6607, 2021.

[409] Ruohan Zhang, Zhuode Liu, Luxin Zhang, Jake Alden Whritner, Karl S. Muller, Mary M. Hayhoe, and Dana H. Ballard. AGIL: learning attention from human for visuomotor tasks. In Vittorio Ferrari, Martial Hebert, Cristian Sminchisescu, and Yair Weiss, editors, Computer Vision - ECCV 2018 - 15th European Conference, Munich, Germany, September 8-14, 2018, Proceedings, Part XI, volume 11215 of Lecture Notes in Computer Science, pages 692–707. Springer, 2018.

[410] Ruohan Zhang, Calen Walshe, Zhuode Liu, Lin Guan, Karl S. Muller, Jake Alden Whritner, Luxin Zhang, Mary M. Hayhoe, and Dana H. Ballard. Atari-head: Atari human eye-tracking and demonstration dataset. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020, pages 6811– 6820. AAAI Press, 2020.

[411] Bolei Zhou, Aditya Khosla, Àgata Lapedriza, Aude Oliva, and Antonio Torralba. Learning deep features for discriminative localization. In 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016, pages 2921–2929. IEEE Computer Society, 2016.

[412] Jianlong Zhou, Amir H Gandomi, Fang Chen, and Andreas Holzinger. Evaluating the quality of machine learning explanations: A survey on methods and metrics. Electronics, 10(5):593, 2021.

[413] Guangxiang Zhu, Zhiao Huang, and Chongjie Zhang. Object-oriented dynamics predictor. In Samy Bengio, Hanna M. Wallach, Hugo Larochelle, Kristen Grauman, Nicolò Cesa-Bianchi, and Roman Garnett, editors, Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montréal, Canada, pages 9826–9837, 2018.

[414] Guangxiang Zhu, Jianhao Wang, Zhizhou Ren, Zichuan Lin, and Chongjie Zhang. Object-oriented dynamics learning through multi-level abstraction. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020, pages 6989–6998. AAAI Press, 2020.

[415] He Zhu, Zikang Xiong, Stephen Magill, and Suresh Jagannathan. An inductive synthesis framework for verifiable reinforcement learning. In Kathryn S. McKinley and Kathleen Fisher, editors, Proceedings of the 40th ACM SIGPLAN Conference on Programming Language Design and Implementation, PLDI 2019, Phoenix, AZ, USA, June 22-26, 2019, pages 686–701. ACM, 2019.

[416] Alexander Zien, Nicole Krämer, Sören Sonnenburg, and Gunnar Rätsch. The feature importance ranking measure. In Wray L. Buntine, Marko Grobelnik, Dunja Mladenic, and John Shawe-Taylor, editors, Machine Learning and Knowledge Discovery in Databases, European Conference, ECML PKDD 2009, Bled, Slovenia, September 7-11, 2009, Proceedings, Part II, volume 5782 of Lecture Notes in Computer Science, pages 694–709. Springer, 2009.

[417] Matthieu Zimmer, Xuening Feng, Claire Glanois, Zhaohui Jiang, Jianyi Zhang, Paul Weng, Jianye Hao, Dong Li, and Wulong Liu. Differentiable logic machines. CoRR, abs/2102.11529, 2021.
---

## BibTeX Citation

```bibtex
@article{Saulieres2025XRLSurvey,
  author        = {Sauli{\`e}res, L{\'e}o},
  title         = {A Survey of Explainable Reinforcement Learning: Targets, Methods and Needs},
  journal       = {arXiv preprint arXiv:2507.12599},
  year          = {2025},
  eprint        = {2507.12599},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2507.12599}
}
```
