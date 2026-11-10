# 7. Explain the training run you already launched

Open the first run's log. Find its environment count, rollout horizon, and iteration number. Predict its collected-transition count before calculating it.

PPO repeats four operations: collect a recent rollout with the current stochastic actor; estimate returns and advantages with the critic; update actor and critic for a limited number of minibatch epochs; collect fresh experience.

An advantage estimates whether an action worked better than expected from its state. Positive advantage encourages that action; negative advantage discourages it. The probability ratio compares its likelihood under the candidate and rollout policies. PPO clips the surrogate objective to limit the incentive for large changes. Clipping is not a guarantee that every action probability stays inside a hard interval.

The value loss trains the critic; entropy encourages exploration; gradient clipping limits an optimizer step. The configured horizon and number of parallel environments determine the rollout size. Epochs/minibatches reuse that recent rollout without creating additional environment transitions.

The maintained PPO recipe uses 24 control steps per environment, discount 0.99, GAE lambda 0.95, clip parameter 0.2, five learning epochs, and four minibatches. A rollout of 128 × 24 contains 3,072 transitions. Fifty iterations collect 153,600 transitions, regardless of how many optimizer minibatches reuse them.

**Experiment:** change only the number of environments in a new short run. Predict the new experience budget and memory use direction. A changed transition budget is a changed experiment; do not call it an equal-data comparison.

**Check yourself:** why do we collect fresh data after the PPO updates? PPO learns from recent experience sampled by its rollout policy; indefinitely reusing old data changes that relationship.
