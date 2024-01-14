# DUT-Video-Stabilization
![image](https://github.com/btxviny/DUT-Video-Stabilization/blob/main/image.png).

This my implementation of the paper [DUT:Learning Video Stabilization By Simply Watching Unstable Videos](https://arxiv.org/pdf/2011.14574.pdf).
It requires less GPU memory than the [original](https://github.com/Annbless/DUTCode) repository. I  use the [RAFT](https://arxiv.org/abs/2003.12039) optical flow estimator which requires no custom layers as opposed to PWCNet and I also provide an alternative warping method which achieves the same stability scores with less black borders. The warping method is based on [PCA-Flow](http://openaccess.thecvf.com/content_cvpr_2015/papers/Wulff_Efficient_Sparse-to-Dense_Optical_2015_CVPR_paper.pdf).

1. **Download the pretrained models [here](https://drive.google.com/drive/folders/15T8Wwf1OL99AKDGTgECzwubwTqbkmGn6).**
2. **Run the Stabilization Script:**
   - Run the following command for the pca warping method:
     ```bash
     python stabilizer_pca.py --in_path unstable_video_path --out_path result_path
     ```
     or this command for the original mosaic multi homography method:
     ```bash
     python stabilizer_mosaic.py --in_path unstable_video_path --out_path result_path
     ```
   - Replace `unstable_video_path` with the path to your input unstable video.
   - Replace `result_path` with the desired path for the stabilized output video.
