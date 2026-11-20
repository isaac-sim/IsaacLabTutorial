# Train a Microduck

This branch is the starter code for **Train a Microduck** in the
[Getting Started With Isaac Lab](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/index.html)
course. Build a simulated two-legged robot that balances and follows walking
commands. Then change the rewards and compare how the policies walk.

## Set Up the Project

Follow the Microduck module’s **Project Setup** lesson to check the requirements,
fetch the pinned starter revision, and create a local `microduck-course` branch.
If you’ve already set up [Train a Robot Arm](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/train-a-robot-arm/index.html),
reuse the same `IsaacLabTutorial` directory. The robot arm exercises use
`learning-path`. Commit your lesson edits before switching branches.

Git Large File Storage (Git LFS) downloads the assets and saved policies in
[`checkpoints/microduck/`](checkpoints/microduck/). Follow the lesson’s Git LFS
instructions before loading a checkpoint.

## Work Through the Lessons

Start with the robot and its sensors, then assemble the environment and
register tasks as the lessons introduce them. Work through balance and walking
before comparing rewards and evaluating the policies. You can train your own
policies or use the supplied checkpoints once you’ve set up the matching tasks.

The [`lp/adding-microduck` branch](https://github.com/anthonyjclark/IsaacLabTutorial/tree/lp/adding-microduck)
contains the completed Microduck tasks and evaluation scripts. For the robot arm
reference, see the [SO-101 guide](https://github.com/isaac-sim/IsaacLabTutorial#so-101-vial-placement).
