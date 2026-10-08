# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["lerobot[feetech]==0.6.1", "torch==2.10.0", "torchvision==0.25.0"]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# torchvision = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Run interactive LeRobot calibration in an isolated CPU environment.

Pass the usual lerobot-calibrate arguments. This writes motor calibration and requires
the operator to support the arm and manually sweep its joints when prompted.
"""

from lerobot.scripts.lerobot_calibrate import main

if __name__ == "__main__":
    main()
