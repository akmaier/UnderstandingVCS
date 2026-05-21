# An Atari Model Zoo for Analyzing, Visualizing, and Comparing Deep Reinforcement Learning Agents

Felipe Petroski Such$^1$, Vashisht Madhavan$^1$, Rosanne Liu$^1$, Rui Wang$^1$, Pablo Samuel Castro$^2$, Yulun Li$^1$, Jiale Zhi$^1$, Ludwig Schubert$^3$, Marc G. Bellemare$^2$, Jeff Clune$^1$, Joel Lehman$^{1*}$

$^1$Uber AI Labs, $^2$Google Brain, $^3$OpenAI

$^*$Corresponding author: `joel.lehman@uber.com`

## Abstract

Much human and computational effort has aimed to improve how deep reinforcement learning (DRL) algorithms perform on benchmarks such as the Atari Learning Environment. Comparatively less effort has focused on understanding what has been learned by such methods, and investigating and comparing the representations learned by different families of DRL algorithms. Sources of friction include the onerous computational requirements, and general logistical and architectural complications for running DRL algorithms at scale. We lessen this friction, by (1) training several algorithms at scale and releasing trained models, (2) integrating with a previous DRL model release, and (3) releasing code that makes it easy for anyone to load, visualize, and analyze such models. This paper introduces the Atari Zoo framework, which contains models trained across benchmark Atari games, in an easy-to-use format, as well as code that implements common modes of analysis and connects such models to a popular neural network visualization library. Further, to demonstrate the potential of this dataset and software package, we show initial quantitative and qualitative comparisons between the performance and representations of several DRL algorithms, highlighting interesting and previously unknown distinctions between them.

## 1 Introduction

Since its introduction the Atari Learning Environment (ALE; [Bellemare *et al.*, 2013]) has been an important reinforcement learning (RL) testbed. It enables easily evaluating algorithms on over 50 emulated Atari games spanning diverse game-play styles, providing a window on such algorithms' generality. Indeed, surprisingly strong results in ALE with deep neural networks (DNNs), published in Nature [Mnih *et al.*, 2015], greatly contributed to the current popularity of deep reinforcement learning (DRL).

Like other machine learning benchmarks, much effort aims to quantitatively improve state-of-the-art (SOTA) scores. As the DRL community grows, a paper pushing SOTA is likely to attract significant interest and accumulate citations. While improving performance is important, it is equally important to understand what DRL algorithms learn, how they process and represent information, and what are their properties, strengths, and weaknesses. These questions cannot be answered through simple quantitative measurements of performance across the ALE suite of games.

Compared to pushing SOTA, much less work has focused on understanding, interpreting, and visualizing *products* of DRL; in particular, little research compares DRL algorithms across dimensions other than performance. This paper thus aims to alleviate the considerable *friction* for those looking to rigorously understand the qualitative behavior of DRL agents. Three main sources of such friction are: (1) the significant computational resources required to run DRL at scale, (2) the logistical tedium of plumbing the products of different DRL algorithms into a common interface, and (3) the wasted effort in re-implementing standard analysis pipelines (like t-SNE embeddings of the state space [Mnih *et al.*, 2015], or activation maximization for visualizing what neurons in a model represent [Erhan *et al.*, 2009; Olah *et al.*, 2018; Nguyen *et al.*, 2017; Simonyan *et al.*, 2013; Yosinski *et al.*, 2015; Mahendran and Vedaldi, 2016]). To address these frictions, this paper introduces the *Atari Zoo*, a release of trained models spanning major families of DRL algorithms, and an accompanying open-source software package$^1$ that enables their easy analysis, comparison, and visualization (and similar analysis of future models). In particular, this package enables easily downloading particular frozen models of interest from the zoo on-demand, further evaluating them in their training environment or modified environments, generating visualizations of their neural activity, exploring compressed visual representations of their behavior, and creating synthetic input patterns that reveal what particular neurons most respond to.

To demonstrate the promise of this model zoo and software, this paper presents an initial analysis of the products of seven DRL algorithms spanning policy gradient, value-based, and evolutionary methods$^2$: A2C (policy-gradient; [Mnih *et al.*, 2016]), IMPALA (policy-gradient; [Espeholt *et al.*, 2018]), DQN (value-based; [Mnih *et al.*, 2015]), Rainbow (value-based; [Hessel *et al.*, 2017]), Ape-X (value-based; [Horgan *et al.*, 2018]), ES (evolutionary; [Salimans *et al.*, 2017]), and Deep GA (evolutionary; [Such *et al.*, 2017]). The analysis illuminates differences in learned policies across methods that are independent of raw score performance, highlighting the benefit of going beyond simple quantitative measures and of having a unifying software framework that enables analyses with multiple, different, complementary techniques and applying them across many RL algorithms.

$^1$https://github.com/uber-research/atari-model-zoo

$^2$While evolutionary algorithms are excluded from some definitions of RL, their inclusion in the zoo can help investigate what distinguishes such black-box optimization from more traditional RL.

## 2 Background

### 2.1 Visualizing Deep Networks

One line of DNN research focuses on visualizing the internal dynamics of a DNN [Yosinski *et al.*, 2015] or examines what particular neurons detect or respond to [Erhan *et al.*, 2009; Zeiler and Fergus, 2014; Olah *et al.*, 2018; Nguyen *et al.*, 2017; Simonyan *et al.*, 2013; Yosinski *et al.*, 2015; Mahendran and Vedaldi, 2016]. The hope is to gain more insight into how DNNs are representing information, motivated both to enable more interpretable decisions from these models [Olah *et al.*, 2018], and to illuminate previously unknown properties about DNNs [Yosinski *et al.*, 2015]. For example, through live visualization of all activations of a vision network responding to different images, Yosinski *et al.* [2015] highlighted that representations were often surprisingly local (as opposed to distributed), e.g. one convolutional filter proved to be a reliable face detector. One practical value of such insights is that they can catalyze future research. The Atari Zoo enables animations in the spirit of Yosinski *et al.* [2015] that show an agent's activations as it interacts with a game, and also enables creating synthetic inputs via activation maximization [Erhan *et al.*, 2009; Zeiler and Fergus, 2014; Olah *et al.*, 2018; Nguyen *et al.*, 2017; Simonyan *et al.*, 2013; Yosinski *et al.*, 2015; Mahendran and Vedaldi, 2016], specifically by connecting DRL agents to the Lucid visualization package [Luc, 2018].

### 2.2 Understanding Deep RL

While much more visualization and understanding work has been done for vision models than for DRL, a few papers directly focus on understanding DRL agents [Greydanus *et al.*, 2017; Zahavy *et al.*, 2016], and many others feature some analysis of DRL agent behavior (often in the form of t-SNE diagrams of the state space; see [Mnih *et al.*, 2015]). One approach to understanding DRL agents is to investigate the learned features of models [Greydanus *et al.*, 2017; Zahavy *et al.*, 2016]. For example, Zahavy *et al.* [2016] visualize what pixels are most important to an agent's decision by using gradients of decisions with respect to pixels. Another approach is to modify the DNN architecture or training procedure such that a trained model will have more interpretable features [Annasamy and Sycara, 2018]. For example, Annasamy and Sycara [2018] augment a model with an attention mechanism and a reconstruction loss, hoping to produce interpretable explanations as a result.

The software package released here is in the spirit of the first paradigm. It facilitates understanding the most commonly applied architectures instead of changing them, although it is designed also to accommodate importing in new vision-based DRL models, and could thus be also used to analyze agents explicitly engineered to be interpretable. In particular, the package enables re-exploring many past DRL analysis techniques at scale, and across algorithms, which were previously applied only for one algorithm and across only a handful of hand-selected games.

### 2.3 Model Zoos

A useful mechanism for reducing friction for analyzing and building upon models is the idea of a *model zoo*, i.e. a repository of pre-trained models that can easily be further investigated, fine-tuned, and/or compared (e.g. by looking at how their high-level representations differ). For example, the Caffe website includes a model zoo with many popular vision models, as do Tensorflow, Keras, and PyTorch. The idea is that training large-scale vision networks (e.g. on the ImageNet dataset) can take weeks with powerful GPUs, and that there is little reason to constantly reduplicate the effort of training. Pre-trained word-embedding models are often released with similar motivation, e.g. for Word2Vec or GLoVE. However, such a practice is much less common in the space of DRL; one reason is that so far, unlike with vision models and word-embedding models, there are few other downstream tasks from which Atari DRL agents provide obvious value. But, if the goal is to better understand these models and algorithm, both to improve them and to use them safely, then there is value in their release.

The recent Dopamine reproducible DRL package [Bellemare *et al.*, 2018] released trained ALE models; it includes final checkpoints of models trained by several DQN variants. However, in general it is non-trivial to extract TensorFlow models from their original context for visualization purposes, and to compare agent behavior across DRL algorithms in the same software framework (e.g. due to slight differences in image preprocessing), or to explore dynamics that take place *over* learning, i.e. from *intermediate* checkpoints. To remedy this, for this paper's accompanying software release, the Dopamine checkpoints were distilled into frozen models that can be easily loaded into the Atari Zoo framework; and for algorithms trained specifically for the Atari Zoo, we distill intermediate checkpoints in addition to final ones.

## 3 Generating the Zoo

The approach is to run several validated implementations of DRL algorithms and to collect and standardize the models and results, such that they can then be easily used for downstream analysis and synthesis. There are many algorithms, implementations of them, and different ways that they could be run (e.g. different hyperparameters, architectures, input representations, stopping criteria, etc.). These choices influence the kind of post-hoc analysis that is possible. For example, Rainbow most often outperforms DQN, but if only final models are released, it is impossible to explore scientific questions where it is important to *control* for performance.

We thus adopted the high level principles that the Atari Zoo should hold as many elements of architecture and experimental design constant across algorithms (e.g. DNN structure, input representation), should enable as many types of downstream analysis as possible (e.g. by releasing checkpoints across training time), and should make reasonable allowances for the particularities of each algorithm (e.g. ensuring hyperparameters are well-fit to the algorithm, and allowing for differences in how policies are encoded or sampled from). The next paragraphs describe specific design choices.

### 3.1 Frozen Model Selection Criteria

To enable the platform to facilitate a variety of explorations, we release multiple frozen models for each run, according to different criteria that may be useful to control for when comparing trained policies. The idea is that depending on the desired analysis, controlling for samples, or for wall-clock, or for performance (i.e. comparing similarly-performing policies) will be more appropriate. In particular, in addition to releasing the final model for each run, additional are models taken over training time (at one, two, four, six, and ten hours); over game frame samples (400 million, and 1 billion frames); over scores (if an algorithm reaches human level performance); and also a model before any training, to enable analysis of how weights change from their random initialization. The hope is that these frozen models will cover a wide spectrum of possible use cases.

### 3.2 Algorithm Choice

One important choice for the Atari Zoo is which DRL algorithms to run and include. The main families of DRL algorithms that have been applied to the ALE are policy gradients methods like A2C [Mnih *et al.*, 2016], value-based methods like DQN [Mnih *et al.*, 2015], and black-box optimization methods like ES [Salimans *et al.*, 2017] and Deep GA [Such *et al.*, 2017]. Based on representativeness and available trusted implementations, the particular algorithms chosen to train included two policy gradients algorithms (A2C [Mnih *et al.*, 2016] and IMPALA [Espeholt *et al.*, 2018]), two evolutionary algorithms (ES [Salimans *et al.*, 2017] and Deep GA [Such *et al.*, 2017]), and one value-function based algorithm (a high-performing DQN variant, Ape-X; [Horgan *et al.*, 2018]). Additionally, models are also imported from the Dopamine release [Bellemare *et al.*, 2018], which include DQN [Mnih *et al.*, 2015] and a sophisticated variant of it called Rainbow [Hessel *et al.*, 2017]. Note that from the Dopamine models, only final models are currently available. Hyperparameters and training details for all algorithms are available in supplemental material section S3. We hope to include models from additional algorithms in future releases.

### 3.3 DNN Architecture and Input Representation

All algorithms are run with the DNN architecture from Mnih *et al.* [2015], which consists of three convolutional layers (with filter size 8x8, 4x4, and 3x3, followed by a fully-connected layer). For most of the explored algorithms, the fully-connected layer connects to an output layer with one neuron per valid action in the underlying Atari game. However, A2C and IMPALA have an additional output that approximates the state value function; Ape-X's architecture features dueling DQN [Wang *et al.*, 2015], which has two separate fully-connected streams; and Rainbow's architecture includes C51 [Bellemare *et al.*, 2017], which uses many outputs to approximate the distribution of expected Q-values.

Atari frames are 210x160 color images (see figure 1a); the canonical DRL representation is a a tensor consisting of the four most recent observation frames, grayscaled and down-sampled to 84x84 (figure 1b). By including some previous frames, the aim is to make the game more fully-observable, to boost performance of the feed-forward architectures that are currently most common in Atari research (although recurrent architectures offer possible improvements [Mnih *et al.*, 2016; Espeholt *et al.*, 2018]). One useful Atari representation that is applied in post-training analysis in this paper, is the Atari RAM state, which is only 1024 bits long but encompasses the true underlying state (figure 1c).

**Figure 1.** *Input and RAM Representation*. (a) One RGB frame of emulated Atari gameplay is shown, which is (b) preprocessed and concatenated with previous frames before being fed as an observation into the DNN agent. A compressed representation of a 2000-step ALE simulation is shown in (c), i.e. the 1024-bit RAM state (horizontal axis) unfurled over frames (vertical axis). *Visual: shows a colored Atari frame, four stacked 84x84 grayscale frames labeled T-3/T-2/T-1/Now, and a tall black-and-white RAM-state image.*

### 3.4 Data Collection

All algorithms are run across 55 Atari games, for at least three independent random weight initializations. Regular checkpoints were taken during training; after training, the checkpoints that best fit each of the desired criteria (e.g. 400 million frames or human-level performance) were frozen and included in the zoo. The advantage of this post-hoc culling is that additional criteria can be added in the future, e.g. if Atari Zoo users introduce a new use case, because the original checkpoints are archived. Log files were stored and converted into a common format that are also released with the models, to aid future performance curve comparisons for other researchers. Each frozen model was run post-hoc in ALE for 2500 timesteps to generate cached behavior of policies in their training environment, which includes the raw game frames, the processed four-frame observations, RAM states, and high-level representations (e.g. neural representations at hidden layers). As a result, it is possible to do meaningful analysis without ever running the models themselves.

## 4 Quantitative Analysis

The open-source software package released with the acceptance of this work provides an interface to the Atari Zoo dataset, and implements several common modes of analysis. Models can be downloaded with a single line of code; and other single-line invocations interface directly with ALE and return the behavioral outcome of executing a model's policy, or create movies of agents superimposed with neural activation, or access convolutional weight tensors. In this section, we demonstrate analyses the Atari Zoo software can facilitate, and highlight some of its built-in features. For many of the analyses below, for computational simplicity we study results in a representative subset of 13 ALE games used by prior research [Such *et al.*, 2017], which we refer to here as the *analysis subset* of games.

### 4.1 Convolutional Filter Analysis

While understanding a DNN only by examining its weights is challenging, weights directly connected to the input can often be interpreted. For example, from visualizing the weights of the first convolutional layer in a vision model, Gabor-like edge detection filters are nearly always present. An interesting question is if Gabor-like features also arise when DRL algorithms are trained from pixel input (as is done here). In visualizing filters across games and DRL algorithms, edge-detector-like features sometimes arise in the gradient-based methods, but they are seemingly never as crisp as in vision models; this may because ALE lacks the visual complexity of natural images. In contrast, the filters in the evolutionary models are less regular. Representative examples across games and algorithms are shown in supplemental figure S1.

Learned filters commonly are tiled similarly across time (i.e. across the four DNN input frames), with past frames having lower-intensity weights. One explanation is that reward gradients are more strongly influenced by present observations. To explore this systematically, across games and algorithms we examined the absolute magnitude of filter weights connected to the present frame versus the past. In contrast to the gradient-based methods the evolutionary methods show no discernable preference across time (supplemental figure S2), again suggesting that their learning differs qualitatively from the gradient-based methods. Interestingly, a rigorous information-theoretic approximation of memory usage is explored by Dann *et al.* [2016] in the context of DQN; our measure well-correlates with theirs despite the relative simplicity of exploring only filter weight strength (supplemental section S1.1).

### 4.2 Robustness to Observation Noise

An important property is how agents perform in slightly out-of-distribution (OOD) situations; ideally they would not catastrophically fail in the face of nominal change. While it is difficult to freely alter the ALE game dynamics (without learning how to program in 6502 assembly code), it is possible to systematically distort observations. Here we explore one simple OOD change to observations by adding increasingly severe noise to the observations input to DNN-based agents, and observe how their evaluated game score degrades. The motivation is to discover whether some learning algorithms are learning more robust policies than others. The results show that with some caveats, methods with a direct representation of the policy appear more robust to observation noise (supplemental figure S4). A similar study conducted for robustness to parameter noise (supplemental section S1.2) tentatively suggests that actor-critic methods are more robust to such noise.

### 4.3 Distinctiveness of Learned Policies

To explore the distinctive signature of solutions discovered by different DRL algorithms, we train image classifiers to identify the generating DRL algorithm given states sampled from independent runs of each algorithm (details are in supplemental section S1.3). Supplemental figure S6 shows the confusion matrix for Seaquest, wherein a cluster of policy search methods (A2C, ES, and GA) have the most inter-class confusion, reflecting (as confirmed in later sections) that these algorithms tend to converge to the same sub-optimal behavior in this game; results are qualitatively similar when tabulated across the analysis subset of games (supplemental figure S7).

## 5 Visualization

We next highlight the Atari Zoo's capabilities to quickly and systematically visualize policies, which broadly can be divided into three categories: Direct policy visualization, dimensionality reduction, and neuron activation maximization.

### 5.1 Animations to Inspect Policy and Activations

To quickly survey the solutions being learned, our software generates grids of videos, where one grid axis spans different DRL algorithms, and the other axis covers independent runs of the algorithm. Such videos can highlight when different algorithms are converging to the same local optimum (e.g. supplemental figure S9 shows a situation where this is the case for A2C, ES, and the GA; video: http://bit.ly/2XpD5kO).

To enable investigating the internal workings of the DNN, our software generates movies that display activations of all neurons alongside animated frames of the agent acting in game. This approach is inspired by the deep visualization toolbox [Yosinski *et al.*, 2015], but put into a DRL context. Supplemental figure S10 shows how this tool can lead to recognizing the functionality of particular high-level features (video: http://bit.ly/2tFHiCU); in particular, it helped to identify a submarine detecting neuron on the third convolution layer of an Ape-X agent. Note that for ES and GA, no such specialized neuron was found; activations seemed qualitatively more distributed for those methods.

**Figure 2.** *A sub-detecting neuron in Seaquest*. Each image represents an observation from an Ape-X agent playing Seaquest. The red square indicates which image patch highly-activated the sub-detecting neuron on the third convolutional layer of the DNN. Having multiple tools (such as this image patch finder, or the activation movies which identified this neuron of interest) enables more easily triangulating and verifying hypotheses about the internals of a DRL agent's neural network. *Visual: six observation frames showing a submarine in undersea Seaquest scenes, each with a red bounding box marking the patch that maximally activated the neuron.*

### 5.2 Image Patches that Maximally Excite Filters

One automated technique for uncovering the functionality of a particular convolutional filter is to find which image patches evoke from it the highest magnitude activations. Given a trained DRL agent and a target convolution filter to analyze, observations from the agent interacting with its ALE training environment are input to the agent's DNN, and resulting maps of activations from the filter of interest are stored. These maps are sorted by the single maximum activation within them, and the geometric location within the map of that maximum activation is recorded. Then, for each of these top-most activations, the specific image patch from the observation that generated it is identified and displayed, by taking the receptive field of the filter into account (i.e. modulated by both the stride and size of the convolutional layers). As a sanity check, we validate that the neuron identified in the previous section does indeed maximally fire for submarines (figure 2).

### 5.3 Dimensionality Reduction

Dimensionality reduction provides another view on agent behavior; often DRL research includes t-SNE plots of agent DNN representations that summarize behavior in the domain [Mnih *et al.*, 2015]. Our software includes such an implementation (supplemental figure S12).

However, such an approach relies on embedding the high-level representation of *one* agent; it is unclear how to apply it to create an embedding appropriate for comparisons of different independent runs of the same algorithm, or runs from different DRL algorithms. As an initial approach, we implement an embedding based on the Atari RAM representation (which is the same across algorithms and runs, but distinct between games). Like the grid view of agent behaviors and the state-distinguishing classifier, this t-SNE tool provides high-level information from which to compare runs of or between different algorithms (figure 3); details of this approach are provided in supplemental section S2.1.

**Figure 3.** *Multiple runs of algorithms and sharing the same RAM-space embedding in Seaquest*. This plot shows one ALE evaluation per model for A2C, ES, and Ape-X, visualized in the same underlying RAM t-SNE embedding. Each dot represents a separate frame from each agent, colored by score (darker color indicates higher score). The plot highlights that in this game, A2C and ES visit similar distributions of states (corresponding to the same sub-optimal behavior), while Ape-X visits a distinct part of the state-space, i.e. matching what could manually be distilled from watching the policy movies in supplemental figure S9. The interface allows clicking on points to observe the corresponding RGB frame, and for toggling different runs of different algorithms for visualization. *Visual: a 2D t-SNE scatter plot with three colored clusters (A2C red, ES blue, Ape-X magenta) surrounded by example Seaquest frames connected by arrows to representative dots.*

**Figure 4.** *Synthesized inputs for output layer neurons in Seaquest*. For a representative run of Rainbow and DQN, inputs are shown optimized to maximize the activation of the first neuron in the output layer of a Seaquest network. Because Rainbow includes C51, its image is in effect optimized to maximize the probability of a low-reward scenario; this neuron appears to be learning interpretable features such as submarine location and the seabed. When maximizing (or minimizing) DQN Q-value outputs (one example shown on left), this qualitative outcome of interpretability was not observed. *Visual: two 4-frame sequences (T-3 to Now) of synthetic black-and-white inputs — DQN's appear high-frequency static-like, Rainbow's show recognizable submarine/seabed shapes.*

### 5.4 Synthesizing Inputs to Understand Neurons

While the previous sections explore DNN activations in the context of an agent's training environment, another approach is to optimize synthetic input images that stimulate particular DNN neurons. Variations on this approach have yielded striking results in vision models [Nguyen *et al.*, 2017; Olah *et al.*, 2018; Simonyan *et al.*, 2013]; the hope is that these techniques could yield an additional view on DRL agents' neural representations. To enable this analysis, we leverage the Lucid visualization library [Luc, 2018]; in particular, we create wrapper classes that enable easy integration of Atari Zoo models into Lucid, and release Jupyter notebooks that generate synthetic inputs for different DRL models.

We now present a series of synthetic inputs generated by the Lucid library across a handful of games that highlight the potential of these kinds of visualizations for DRL understanding (further details of the technique used are described in supplemental section S2.2). We first explore the kinds of features learned across depth. Supplemental figure S13 supports what was learned by visualizing the first-layer filter weights for value-based networks (section 4.1; i.e. showing that first convolution layers in the value-based networks appear to be learning edge-detector features). The activation videos of section 5.1 and the patch-based approach of section 5.2 help to provide grounding, showing that in the context of the game, some first-layer filters detect the edges of the screen, in effect to serve as location anchors, while others encode concepts like blinking objects (see figure S11). Supplemental figure S14 explores visualizing later-layer convolution filters, and figure 4 show inputs synthesized to maximize output neurons, which sometimes yields interpretable features.

**Figure 5.** *Synthesized inputs for fully-connected layer neurons in Freeway*. Inputs synthesized to maximize activations of the first three neurons in the first fully connected layer are shown for a respresentative DQN and Rainbow DNN. One of the Rainbow neurons (in red rectangle) appears to be capturing lane features. *Visual: a 3x2 grid of 4-frame synthetic inputs; one Rainbow row outlined in red shows horizontal stripe-like patterns reminiscent of road lanes.*

Such visualizations can also reveal that critical features are being attended to (figure 5 and supplemental figure S15). Overall, these visualizations demonstrate the potential of this kind of technique, and we believe that many useful further insights may result from a more systematic application and investigation of this and many of the other interesting visualization techniques implemented by Lucid, which can now easily be applied to Atari Zoo models. Also promising would be to further explore regularization to constrain the space of synthetic outputs, e.g. a generative model of Atari frames in the spirit of Nguyen *et al.* [2017] or similar works.

## 6 Discussion and Conclusions

There are many follow-up extensions that the initial explorations of the zoo raise. One natural extension is to include more DRL algorithms (e.g. TRPO or PPO [Schulman *et al.*, 2017]). Beyond algorithms, there are many alternate architectures that might have interesting effects on representation and decision-making, for example recurrent architectures, or architectures that exploit attention. Also intriguing is examining the effect of the incentive driving search: Do auxiliary or substitute objectives qualitatively change DRL representations, e.g. as in UNREAL [Jaderberg *et al.*, 2016], curiosity-driven exploration [Pathak *et al.*, 2017], or novelty search [Conti *et al.*, 2017]? How do the representations and features of meta-learning agents such as MAML [Finn *et al.*, 2017] change as they learn a new task? Finally, there are other analysis tools that could be implemented, which might illuminate other interesting properties of DRL algorithms and learned representation, e.g. the image perturbation analysis of Greydanus *et al.* [2017] or a variety of sophisticated neuron visualization techniques [Nguyen *et al.*, 2017]. We welcome community contributions for these algorithms, models, architectures, incentives, and tools.

While the main motivation for the zoo was to reduce friction for research into understanding and visualizing the behavior of DRL agents, it can also serve as a platform for other research questions. For example, having a zoo of agents trained on individual games, for different amounts of data, also would reduce friction for exploring transfer learning within Atari, i.e. whether experience learned on one game can quickly benefit on another game. Also, by providing a huge library of cached rollouts for agents across algorithms, the zoo may be interesting in the context of learning from demonstrations, or for creating generative models of games. In conclusion, we look forward to seeing how this dataset will be used by the community at large.

## References

Raghuram Mandyam Annasamy and Katia Sycara. Towards better interpretability in deep q-networks. *arXiv preprint arXiv:1809.05630*, 2018.

Marc G Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling. The arcade learning environment: An evaluation platform for general agents. *Journal of Artificial Intelligence Research*, 47:253–279, 2013.

Marc G Bellemare, Will Dabney, and Rémi Munos. A distributional perspective on reinforcement learning. *arXiv preprint arXiv:1707.06887*, 2017.

Marc G. Bellemare, Pablo Samuel Castro, Carles Gelada, Saurabh Kumar, and Subhodeep Moitra. Dopamine. *GitHub, GitHub repository*, 2018.

Edoardo Conti, Vashisht Madhavan, Felipe Petroski Such, Joel Lehman, Kenneth O Stanley, and Jeff Clune. Improving exploration in evolution strategies for deep reinforcement learning via a population of novelty-seeking agents. *arXiv preprint arXiv:1712.06560*, 2017.

Christoph Dann, Katja Hofmann, and Sebastian Nowozin. Memory lens: How much memory does an agent use? *arXiv preprint arXiv:1611.06928*, 2016.

Dumitru Erhan, Yoshua Bengio, Aaron Courville, and Pascal Vincent. Visualizing higher-layer features of a deep network. *University of Montreal*, 1341(3):1, 2009.

Lasse Espeholt, Hubert Soyer, Remi Munos, Karen Simonyan, Volodymir Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, et al. Impala: Scalable distributed deep-rl with importance weighted actor-learner architectures. *arXiv preprint arXiv:1802.01561*, 2018.

Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. *arXiv preprint arXiv:1703.03400*, 2017.

Sam Greydanus, Anurag Koul, Jonathan Dodge, and Alan Fern. Visualizing and understanding atari agents. *arXiv preprint arXiv:1711.00138*, 2017.

Matteo Hessel, Joseph Modayil, Hado Van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan, Bilal Piot, Mohammad Azar, and David Silver. Rainbow: Combining improvements in deep reinforcement learning. *arXiv preprint arXiv:1710.02298*, 2017.

Dan Horgan, John Quan, David Budden, Gabriel Barth-Maron, Matteo Hessel, Hado Van Hasselt, and David Silver. Distributed prioritized experience replay. *arXiv preprint arXiv:1803.00933*, 2018.

Max Jaderberg, Volodymyr Mnih, Wojciech Marian Czarnecki, Tom Schaul, Joel Z Leibo, David Silver, and Koray Kavukcuoglu. Reinforcement learning with unsupervised auxiliary tasks. *arXiv preprint arXiv:1611.05397*, 2016.

Lucid: A collection of infrastructure and tools for research in neural network interpretability. http://http://github.com/tensorflow/lucid, 2018.

Aravindh Mahendran and Andrea Vedaldi. Visualizing deep convolutional neural networks using natural pre-images. *International Journal of Computer Vision*, 120(3):233–255, 2016.

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. *Nature*, 518(7540):529, 2015.

Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In *International conference on machine learning*, pages 1928–1937, 2016.

Anh Nguyen, Jeff Clune, Yoshua Bengio, Alexey Dosovitskiy, and Jason Yosinski. Plug & play generative networks: Conditional iterative generation of images in latent space. In *CVPR*, volume 2, page 7, 2017.

Chris Olah, Arvind Satyanarayan, Ian Johnson, Shan Carter, Ludwig Schubert, Katherine Ye, and Alexander Mordvintsev. The building blocks of interpretability. *Distill*, 3(3):e10, 2018.

Deepak Pathak, Pulkit Agrawal, Alexei A Efros, and Trevor Darrell. Curiosity-driven exploration by self-supervised prediction. In *International Conference on Machine Learning (ICML)*, volume 2017, 2017.

Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor, and Ilya Sutskever. Evolution strategies as a scalable alternative to reinforcement learning. *arXiv preprint arXiv:1703.03864*, 2017.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*, 2017.

Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Deep inside convolutional networks: Visualising image classification models and saliency maps. *arXiv preprint arXiv:1312.6034*, 2013.

Felipe Petroski Such, Vashisht Madhavan, Edoardo Conti, Joel Lehman, Kenneth O Stanley, and Jeff Clune. Deep neuroevolution: genetic algorithms are a competitive alternative for training deep neural networks for reinforcement learning. *arXiv preprint arXiv:1712.06567*, 2017.

Ziyu Wang, Tom Schaul, Matteo Hessel, Hado Van Hasselt, Marc Lanctot, and Nando De Freitas. Dueling network architectures for deep reinforcement learning. *arXiv preprint arXiv:1511.06581*, 2015.

Jason Yosinski, Jeff Clune, Anh Nguyen, Thomas Fuchs, and Hod Lipson. Understanding neural networks through deep visualization. *arXiv preprint arXiv:1506.06579*, 2015.

Tom Zahavy, Nir Ben-Zrihem, and Shie Mannor. Graying the black box: Understanding dqns. In *International Conference on Machine Learning*, pages 1899–1908, 2016.

Matthew D Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. In *European conference on computer vision*, pages 818–833. Springer, 2014.

## Supplementary Material

The following sections contain additional figures, and describe in more detail the experimental setups applied in the paper's experiments.

### S1 Quantitative Analysis Details

Figure S1 shows a sampling of first-layer convolutional filters from final trained models, and figure S2 highlights that such filters often differentially attend to the present over the past.

#### S1.1 Further Study of Temporal Bias in DQN

As an exploration of the connection between the information theoretic measure of memory-dependent action in Dann *et al.* [2016] and the pattern highlighted in this paper (i.e. the strength of filter weights in the first layer of convolutions may highlight a network's reliance on the past), we examined first-layer filters in DQN across all 55 games. A simple metric of present-focus is the ratio of average weight magnitudes for the past three frames to the present frame. When sorted by this metric (see figure S3), there is high agreement with the 8 games identified by Dann *et al.* [2016] that use memory. In particular, three out of the top four games identified by our metric align with theirs; as do six out of the top twelve games, considered among the games that overlap between their 49 and our 55.

#### S1.2 Observation and Parameter Noise Details

Figure S4 shows robustness to observation noise for games in the analysis subset. Beyond observation noise, another interesting property of learning algorithms is the kind of local optimum they find in the parameter space, i.e. whether the learned function is smooth or not in the area of a found solution. One gross tool for examining this property is to test the robustness of policies to parameter perturbations. It is plausible that the evolutionary methods would be more robust in this way, given that they are trained through parameter perturbations. To measure this, we perturb the convolutional weights of each DRL agent with increasingly severe normally-distributed noise. We perturbed the convolutional weights only, because that part of the DNN is identical across agents, whereas the fully-connected layers sometimes vary in structure across DRL algorithms (e.g. value-based algorithms like Rainbow or Ape-X that include features that require post-convolutional architectural changes). Figure S5 shows the results across games and algorithms; while no incontrovertible trend exists, the two policy-gradient methods (A2C and Impala) show greater relative robustness than the other methods.

Note that for both algorithm-best performance plots (i.e. figures S4a and S5a), three games were excluded from analysis because at least one of the DRL algorithms performed worse than a random policy before perturbation; including them would have conflated performance and robustness, which would undermine the purpose of the plot.

#### S1.3 Distinctiveness of Policies Learned by Algorithms

We use only the "present" channel of each gray-scale observation frame (i.e. without the complete four-frame stack) to train a classifier for each game. The classifier consists of two convolution layers and two fully connected layers, and is trained with early stopping to avoid overfitting. For each game, 2501 frames are collected from multiple evaluations by each model. The reported classification results use 20% of the frames as test set. Figure S6 visualizes the confusion matrix for Seaquest frame classification, while figure S7 shows a confusion matrix summed across games.

We also provide summaries of F1 classification performance: Figure S8 summarizes classification performance across DRL algorithms and games, while table S1 shows performance averaged across games.

### S2 Visualization Details

This section provides more details and figures for the visualization portion of the paper's analysis. Figure S9 shows one frame of a collage of simultaneous videos that give a quick high-level comparison of how different algorithms and runs are solving an ALE environment. Figure S9 shows one frame of a video that simultaneously shows a DNN agent acting in an ALE environment and all of the activations of its DNN.

Figure S11 shows a second example of how the image-patch finder can help ground out what particular DNN neurons are learning.

#### S2.1 t-SNE Details

To visualize RAM states and high-level DNN representations in 2D, as is typical in t-SNE analysis PCA is first applied to reduce the number of dimensions (to 50), followed by 3000 t-SNE iterations with perplexity of 30. The dimensionality reduction of RAM states is applied across all available runs of DRL algorithms to be jointly embedded. In contrast, dimensionality reduction of high-level DNN representations is particular to a specific model trained by a single DRL algorithm (i.e. each run of a DRL algorithm learns its own distinct representation).

**Figure S1.** *Learned Convolutional Filters*. Shown are first-layer convolutional filters taken from DNNs trained by each algorithm, as well as random filters drawn from a normal distribution (Random). In games in which they exceed random performance, filters for the gradient-based algorithms often have spatial structure and sometimes resemble edge detectors, and the intensity of weights often degrades into the past (i.e. the left-most patches). This can be seen for all gradient-based methods in (a) Seaquest; when gradient-based methods fail to learn, as DQN and A2C often do in (b) Venture, their filters then appear more random (this effect is consistent across runs). Filters for the evolutionary algorithms appear less regular, even when their performance is competitive with the gradient-based methods. *Visual: large grid of 6x4 small grayscale filter tiles per algorithm (A2C, Impala, DQN, Rainbow, ES, GA, Random, Ape-X) for Seaquest (top) and Venture (bottom).*

**Figure S2.** *Significance of Time Across Models*. Filter weight magnitudes across input patches are shown averaged across a representative sample of ALE games with 3 independent runs each for each DRL algorithm; also included for analysis are random filters drawn from a normal distribution (Random). Before averaging, past-frame weight magnitudes are normalized by that of weights attending to the most recent observation (i.e. the present magnitudes are anchored to 1.0). For the gradient-based DRL algorithms (Ape-X, Rainbow, DQN, & A2C), filter weights are stronger when connected to the current frame than to historical frames. Interestingly, such a trend is not seen for the evolutionary algorithms; note that ES includes L2 regularization, so this effect is not merely an artifact of weight decay being present in the gradient-based methods only. The effect is also present when looking at individual games (data not shown). *Visual: eight bar-chart panels (one per algorithm) plotting normalized weight magnitudes for T-3, T-2, T-1, Now — gradient methods show a rise toward "Now"; evolutionary methods stay flat near 1.0.*

**Figure S3.** *Attention to the past in DQN*. DQN's tendency to focus on the present relative to the past (as measured by filter weight magnitudes from different input frames), is shown across 55 ALE games. From left to right, the amount of present-bias increases, e.g. the games at the left seemingly have greater use for information stored in the past three frames relative to the games on the right. *Visual: a horizontal bar/strip ordering all 55 Atari game names by present-bias (MontezumaRevenge at left, TimePilot at right).*

**Figure S4.** *Robustness to Observation Noise*. How performance of trained policies degrades with increasing severe normally-distributed noise is shown, averaged over three independent runs across the analysis subset of games. The figure shows performance degrades (a) relative to baseline performance by that algorithm on each game, and (b) by the best performance of any algorithm on each game. Zero performance in this chart represents random play. The conclusion is that the policy search algorithms show less steep degradation relative to (a) their own best performance; although this is confounded by (b) the overall better absolute performance of the value-based methods. Follow-up analysis will control for performance, by using the Atari Zoo human-performance frozen models. *Visual: two line plots of episode reward vs observation noise level (log scale) for seven algorithms — value-based methods sharply drop, policy-gradient methods drop more gradually.*

**Figure S5.** *Robustness to Parameter Noise*. How performance of trained policies degrades with increasing severe normally-distributed parameter noise is shown, averaged over three independent runs across the analysis subset of games. The figure shows performance degrades (a) relative to baseline performance by that algorithm on each game, and (b) by the best performance of any algorithm on each game. Zero performance in this chart represents random play. Interestingly, the two policy-gradient methods demonstrate a very similar algorithm-best profile that is more robust than the other methods; our prior hypothesis was that the evolutionary algorithms might exhibit higher robustness by this measure (given that they are trained with parameter perturbations). *Visual: two line plots of episode reward vs parameter noise level (log scale) for seven algorithms.*

**Figure S6.** *Confusion matrix for Seaquest frame classification*. The cell in the $i$th row and the $j$th column denotes the number of frames generated by the algorithm in row $i$ that are predicted to be generated by the algorithm in column $j$. The conclusion is that in this game, there is a cluster of confusion among many of the direct policy search algorithms (ES, A2C, and GA), highlighting that they are converging to similarly sub-optimal behaviors. *Visual: 7x7 blue-heatmap matrix of classifier predictions with high diagonal values and a confusion block among A2C/ES/GA.*

The transcribed values of Figure S6 (rows are true algorithm, columns are predicted):

| | A2C | Ape-X | DQN | ES | GA | IMPALA | Rainbow |
|---|---|---|---|---|---|---|---|
| A2C | 1,156 | 0 | 0 | 183 | 115 | 0 | 0 |
| Ape-X | 22 | 1,438 | 0 | 10 | 8 | 0 | 0 |
| DQN | 0 | 0 | 1,445 | 0 | 0 | 0 | 42 |
| ES | 151 | 0 | 0 | 978 | 361 | 0 | 0 |
| GA | 110 | 3 | 0 | 198 | 1,248 | 0 | 0 |
| IMPALA | 0 | 0 | 0 | 0 | 0 | 1,483 | 0 |
| Rainbow | 0 | 0 | 58 | 0 | 0 | 0 | 1,496 |

**Figure S7.** *Confusion matrix summed across all games*. The cell in the $i$th row and the $j$th column denotes the total number of frames from the rollouts of the algorithm in row $i$ predicted to be from rollouts of the algorithm in column $j$. The true positive predictions are reset to 0 to highlight the false positives. *Visual: 7x7 blue-heatmap matrix with diagonal zeroed; strong off-diagonal blocks among A2C/ES/GA and between DQN and Rainbow.*

The transcribed values of Figure S7:

| | A2C | Ape-X | DQN | ES | GA | IMPALA | Rainbow |
|---|---|---|---|---|---|---|---|
| A2C | 0 | 285 | 0 | 817 | 615 | 0 | 0 |
| Ape-X | 693 | 0 | 0 | 356 | 340 | 0 | 1 |
| DQN | 0 | 0 | 0 | 0 | 0 | 0 | 824 |
| ES | 1,296 | 186 | 1 | 0 | 1,738 | 0 | 0 |
| GA | 1,034 | 155 | 0 | 1,780 | 0 | 0 | 0 |
| IMPALA | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| Rainbow | 0 | 0 | 1,776 | 0 | 0 | 2 | 0 |

**Figure S8.** *F1 scores for frame classification*. F1 score is defined as $2 \times \frac{\text{precision} \times \text{recall}}{\text{precision} + \text{recall}}$. We observe the classifier distinguishes each algorithm in all environments with at least 0.5 score. *Visual: 13x7 blue heatmap of F1 scores across the analysis-subset games (rows) and seven algorithms (columns), mostly between 0.7 and 1.0.*

**Table S1.** *Average F1 scores by game*. The score is an unweighted average of F1 scores across all algorithm. Lower scores indicate games for which different DRL algorithms are less distinguishable from each other.

| Game | Mean F1 |
|---|---|
| Amidar | 0.96 |
| Assault | 0.86 |
| Asterix | 0.96 |
| Asteroids | 0.96 |
| Atlantis | 0.73 |
| Enduro | 0.92 |
| Frostbite | 0.94 |
| Gravitar | 0.92 |
| Kangaroo | 0.93 |
| Seaquest | 0.88 |
| Skiing | 0.9 |
| Venture | 0.96 |
| Zaxxon | 0.96 |

#### S2.2 Synthetic Input Generation Details

We use the lucid library [Luc, 2018] to visualize what types of inputs maximize neuron activations throughout the agents' networks. This study used the trained checkpoints provided by Dopamine [Bellemare *et al.*, 2018] for DQN and Rainbow (although it could be applied to any of the DRL algorithms in the Atari Zoo). These frozen graphs are then loaded as part of a Lucid model and an optimization objective is created.

An input pattern to the network (consisting of a stack of four 84x84 pixel screens) is optimized to maximize the activations of the desired neurons. Initially, the four 84x84 frames are initialized with random noise. The result of optimization ideally yields visualizations that reveal qualitatively what features the neurons have learned to capture. As recommended in Olah *et al.* [2017] and Mahendran and Vedaldi [2016] we apply regularization to produce clearer results; for most images we use only image jitter (i.e. randomly offsetting the input image by a few pixels to encourage local translation invariance). For some images, we found it helpful to add total variation regularization (to encourage local smoothness; see Mahendran and Vedaldi [2016]) and L1 regularization (to encourage pixels that are not contributing to the objective to become zero) on the optimized image.

### S3 DRL Algorithm Details and Hyperparameters

This section describes the implementations and hyperparameters used for training the models released with the zoo. The DQN and Rainbow models come from the Dopamine model release [Bellemare *et al.*, 2018]. The following sections describe the algorithms for the newly-trained models released with this paper.

#### S3.1 A2C

The implementation of A2C [Mnih *et al.*, 2016] that generated the models in this paper was derived from the OpenAI baselines software package [Dhariwal *et al.*, 2017]. It ran with 20 parallel worker threads for 400 million frames; checkpoints occurred every 4 million frames. Hyperparameters are listed in table S2.

**Table S2.** *A2C Hyperparameters*. Population sizes are incremented to account for elites (+1). Many of the unusual numbers were found via preliminary hyperparameter searches in other domains.

| Hyperparameter | Setting |
|---|---|
| Learning Rate | 7e-5 |
| $\tau$ | 1.0 |
| Value Function Loss Coefficient | 0.5 |
| Entropy Loss Coefficient | 0.01 |
| Discount factor | 0.99 |

#### S3.2 Ape-X

The implementation of Ape-X used to generate the models in this paper was based on the one found here: https://github.com/uber-research/ape-x. The hyperparameters are reported in Table S3.

**Table S3.** *Ape-X Hyperparameters*. For more details on what these parameters signify, see [Horgan *et al.*, 2018].

| Hyperparameter | Setting |
|---|---|
| Buffer Size | $2^{21}$ |
| Number of Actors | 384 |
| Batch Size | 512 |
| n-step | 3 |
| gamma | 0.99 |
| gradient clipping | 40 |
| target network period | 2500 |
| Prioritized replay $(\alpha, \beta)$ | (0.6, 0.4) |
| Adam Learning rate | 0.00025 / 4 |

#### S3.3 GA

The implementation of GA used to generate the models in this paper was based on the one found here: https://github.com/uber-research/deep-neuroevolution. The hyperparameters are reported in Table S4 and were found through random search.

**Table S4.** *GA Hyperparameters*. For more details on what these parameters signify, see [Such *et al.*, 2017].

| Hyperparameter | Setting |
|---|---|
| $\sigma$ (Mutation Power) | 0.002 |
| Population Size | 1000 |
| Truncation Size | 20 |

#### S3.4 ES

The implementation of ES used to generate the models in this paper was based on the one found here: https://github.com/uber-research/deep-neuroevolution. The hyperparameters reported in Table S5 were found via preliminary search and are similar to those reported in [Conti *et al.*, 2017].

**Table S5.** *ES Hyperparameters*. For more details on what these parameters signify, see [Salimans *et al.*, 2017; Conti *et al.*, 2017].

| Hyperparameter | Setting |
|---|---|
| $\sigma$ (Mutation Power) | 0.02 |
| Virtual Batch Size | 128 |
| Population Size | 5000 |
| Learning Rate | 0.01 |
| Optimizer | Adam |
| L2 Regularization Coefficient | 0.005 |

#### S3.5 Impala

The implementation of Impala used to generate the models in this paper is based on the one found here: https://github.com/deepmind/scalable_agent. The hyperparameters reported in Table S6 are the same as those reported in [Espeholt *et al.*, 2018].

**Table S6.** *IMPALA Hyperparameters*. For more details on what these parameters signify, see [Espeholt *et al.*, 2018].

| Hyperparameter | Setting |
|---|---|
| Number of actors | 25 |
| Image Width | 84 |
| Image Height | 84 |
| Grayscaling | Yes |
| Action Repetitions | 4 |
| Max-pool over last N action repeat frames | 2 |
| Frame Stacking | 4 |
| End of episode when life lost | Yes |
| Reward Clipping | [-1, 1] |
| Unroll Length (n) | 20 |
| Batch size | 32 |
| Discount ($\gamma$) | 0.99 |
| Baseline loss scaling | 0.5 |
| Entropy Regularizer | 0.01 |
| RMSProp momentum | 0.0 |
| RMSProp $\epsilon$ | 0.01 |
| Learning rate | 0.0006 |
| Clip global gradient norm | 40.0 |
| Learning rate schedule | Anneal linearly to 0 from beginning to end of training |

**Figure S9.** *Grid of Rollout Videos in Seaquest*. The vertical axis represents different independently-trained models, while the horizontal axis represents the DRL algorithms included in the Atari Zoo. In Seaquest, one objective is to control a submarine to shoot fish without getting hit by them, and another is to avoid running out of oxygen by intermittently resurfacing. All three independent runs of A2C, GA, and ES converge to the same sub-optimal behavior: They dive to the bottom of the ocean, and shoot fish until they run out of oxygen. The value-function based methods exhibit more sophisticated behavior, highlighting that in this game, greedy policy searches may often converge to sub-optimal solutions, while learning the value of state-action pairs can avoid this pathology. Video is available at: http://bit.ly/2XpD5kO. *Visual: 3x7 grid of Seaquest screenshots — rows are independent runs (R1/R2/R3), columns are algorithms (a2c, impala, dqn, rainbow, apex, ga, es); most show a submarine deep underwater.*

**Figure S10.** *Policy and activation visualization*. The figure shows a still frame from a video of an Ape-X agent acting in the Seaquest environment (full video can be accessed at http://bit.ly/2tFHiCU). On the left, the RGB frame is shown, while from top to bottom on the right are: the processed observations, and then the activations for the convolutional layers, the fully connected layer, and finally, the Q-value outputs. From watching the video, it is apparent that the brightest neuron in the third convolutional layer tracks the position of the submarine. This shows that like in vision DNNs, sometimes important features are represented in a local, rather than distributed fashion [Yosinski *et al.*, 2015]. *Visual: an RGB Seaquest frame alongside stacked grids of red-tinted activation maps for obs, conv1, conv2, conv3, fc, and output layers.*

**Figure S11.** *Location-anchor and oxygen-detector in a Rainbow agent in Seaquest*. The top three images show image patches (red square) that highly-activate a first-layer convolution filter of a Rainbow agent; this filter always activates maximally in the same geometric location, potentially serving as a geometric anchor for localization by down-stream filters. The bottom three images show images patches that highly-activate a separate first-layer filter in the same agent. It detects blinking objects; the submarine can blink before it runs out of oxygen, and the oxygen meter itself blinks when it is running low. *Visual: six 84x84 Seaquest frames in a 2x3 grid with red bounding boxes — top row's box is fixed near the bottom-center; bottom row's box surrounds the blinking sub/oxygen meter.*

**Figure S12.** *Comparing high-level DNN representations through separate t-SNE embeddings*. The figure shows separate t-SNE embeddings of high-level representations for DNNs trained to play Seaquest by A2C and Ape-X. Each dot corresponds to a specific frame in a rollout, and darker shades indicate higher scores. Embeddings that represent similar frames cluster together, indicating states with different positions of the submarine, and objects of various numbers, categories and colors. Representative frames for selected clusters are displayed. For example, in the left figure (A2C), the top-left cluster represents terminated states, and the bottom-left cluster corresponds to the situation of oxygen depletion, while in the right figure (Ape-X), bottom-right cluster corresponds to a repeated series of actions that the agent takes to surface and refill its oxygen. *Visual: two side-by-side t-SNE scatter plots (red for A2C, blue for Ape-X) surrounded by example Seaquest frames pointing to clusters.*

**Figure S13.** *Synthesized inputs for neurons in the first convolutional layer in Seaquest*. Inputs optimized to activate the first three neurons in the first convolutional layer are shown for representative runs of DQN and Rainbow. These neurons appear to be learning 'edge-detector' style features. *Visual: a 3x2 grid of 4-frame black-and-white synthetic inputs (DQN vs Rainbow) showing high-frequency stripe and grid patterns characteristic of edge detectors.*

**Figure S14.** *Synthesized inputs for neurons in the third convolutional layer in Seaquest*. Inputs optimized to activate the first four neurons in the last (third) convolutional layer in Seaquest are shown for a representative run of DQN and Rainbow (hyperparameters for regularization, e.g. a total variation penalty, were optimized by hand to improve image quality). Both networks appear to focus on particular styles of objects, combinations of them, and animation-related features such as blinking. Some synthetic inputs make sense in the context of other investigatory tools; e.g. Rainbow's first neuron's synthetic input includes objects that blink between frames, and when explored with the patch activation technique, is seen responding most intensely when the sub is blinking and about to explode from running out of oxygen. However, for some features it is unclear how the synthetic input is to be interpreted without further investigation, e.g. the patch activation technique shows that Rainbow's third neuron responds most when the sub is nearing the top border of the water. Further experimentation with regularization within Lucid, or employing more sophisticated techniques, may help to improve these initial results. *Visual: a 4x2 grid of 4-frame black-and-white synthetic inputs (DQN vs Rainbow) with more spatially structured patterns than figure S13.*

**Figure S15.** *Synthesized inputs for neurons in the third convolutional layer in Pong*. Inputs optimized to activate the first four neurons in the last (third) convolutional layer in Pong are shown for a representative run of DQN and Rainbow. Both networks seem to learn qualitatively similar features, with images featuring vertical lines reminiscent of patterns and smaller objects reminiscent of balls. Further exploration is needed to ground out these evocative appearances. *Visual: a 4x2 grid of 4-frame black-and-white synthetic inputs (DQN vs Rainbow) showing vertical-stripe Pong-like motifs.*

### Supplementary References

Marc G. Bellemare, Pablo Samuel Castro, Carles Gelada, Saurabh Kumar, and Subhodeep Moitra. Dopamine. *GitHub, GitHub repository*, 2018.

Edoardo Conti, Vashisht Madhavan, Felipe Petroski Such, Joel Lehman, Kenneth O Stanley, and Jeff Clune. Improving exploration in evolution strategies for deep reinforcement learning via a population of novelty-seeking agents. *arXiv preprint arXiv:1712.06560*, 2017.

Christoph Dann, Katja Hofmann, and Sebastian Nowozin. Memory lens: How much memory does an agent use? *arXiv preprint arXiv:1611.06928*, 2016.

Prafulla Dhariwal, Christopher Hesse, Oleg Klimov, Alex Nichol, Matthias Plappert, Alec Radford, John Schulman, Szymon Sidor, and Yuhuai Wu. Openai baselines. *GitHub, GitHub repository*, 2017.

Lasse Espeholt, Hubert Soyer, Remi Munos, Karen Simonyan, Volodymir Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, et al. Impala: Scalable distributed deep-rl with importance weighted actor-learner architectures. *arXiv preprint arXiv:1802.01561*, 2018.

Dan Horgan, John Quan, David Budden, Gabriel Barth-Maron, Matteo Hessel, Hado Van Hasselt, and David Silver. Distributed prioritized experience replay. *arXiv preprint arXiv:1803.00933*, 2018.

Lucid: A collection of infrastructure and tools for research in neural network interpretability. http://http://github.com/tensorflow/lucid, 2018.

Aravindh Mahendran and Andrea Vedaldi. Visualizing deep convolutional neural networks using natural pre-images. *International Journal of Computer Vision*, 120(3):233–255, 2016.

Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In *International conference on machine learning*, pages 1928–1937, 2016.

Chris Olah, Alexander Mordvintsev, and Ludwig Schubert. Feature visualization. *Distill*, 2017. https://distill.pub/2017/feature-visualization.

Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor, and Ilya Sutskever. Evolution strategies as a scalable alternative to reinforcement learning. *arXiv preprint arXiv:1703.03864*, 2017.

Felipe Petroski Such, Vashisht Madhavan, Edoardo Conti, Joel Lehman, Kenneth O Stanley, and Jeff Clune. Deep neuroevolution: genetic algorithms are a competitive alternative for training deep neural networks for reinforcement learning. *arXiv preprint arXiv:1712.06567*, 2017.

Jason Yosinski, Jeff Clune, Anh Nguyen, Thomas Fuchs, and Hod Lipson. Understanding neural networks through deep visualization. *arXiv preprint arXiv:1506.06579*, 2015.

---

## BibTeX Citation

```bibtex
@inproceedings{SuchEtAl2019AtariZoo,
  author    = {Such, Felipe Petroski and Madhavan, Vashisht and Liu, Rosanne and Wang, Rui and Castro, Pablo Samuel and Li, Yulun and Zhi, Jiale and Schubert, Ludwig and Bellemare, Marc G. and Clune, Jeff and Lehman, Joel},
  title     = {An {Atari} Model Zoo for Analyzing, Visualizing, and Comparing Deep Reinforcement Learning Agents},
  booktitle = {Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI-19)},
  year      = {2019},
  pages     = {3260--3267},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  doi       = {10.24963/ijcai.2019/452},
  url       = {https://www.ijcai.org/proceedings/2019/452},
  eprint    = {1812.07069},
  archivePrefix = {arXiv},
  primaryClass = {cs.NE}
}
```
