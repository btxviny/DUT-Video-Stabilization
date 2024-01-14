# DUT-Video-Stabilization

This my implementation of the paper [DUT:Learning Video Stabilization By Simply Watching Unstable Videos](https://arxiv.org/pdf/2011.14574.pdf).
It requires less GPU memory than the [original] repository, it uses [RAFT]() optical flow estimator which requires no custom layers and I also provide an alternative warping method which achieves the same stability scores with less black borders.
