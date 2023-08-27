## Dual Attention Networks for Few-Shot Fine-Grained Recognition

This is the MindSpore implementation of the paper "[Dual Attention Networks for Few-Shot Fine-Grained Recognition](https://ojs.aaai.org/index.php/AAAI/article/view/20196)".

Abstract: The task of few-shot fine-grained recognition is to classify images belonging to subordinate categories merely depending on few examples. Due to the fine-grained nature, it is desirable to capture subtle but discriminative part-level patterns from limited training data, which makes it a challenging problem. In this paper, to generate fine-grained tailored representations for few-shot recognition, we propose a Dual Attention Network (DUAL ATT-NET) consisting of two dual branches of both hard- and soft-attentions. Specifically, by producing attention guidance from deep activations of input images, our hardattention is realized by keeping a few useful deep descriptors and forming them as a bag of multi-instance learning. Since these deep descriptors could correspond to objects’ parts, the advantage of modeling as a multi-instance bag is able to exploit inherent correlation of these fine-grained parts. On the other side, a soft attended activation representation can be obtained by applying attention guidance upon original activations, which brings comprehensive attention information as the counterpart of hard-attention. After that, both outputs of dual branches are aggregated as a holistic image embedding w.r.t. input images. By performing meta-learning, we can learn a powerful image embedding in such a metric space to generalize to novel classes. Experiments on three popular fine-grained benchmark datasets show that our DUAL ATT-NET obviously outperforms other existing state-of-the-art methods.

## Preparing the datasets

We provide three datasets in this repo: CUB Birds, Stanford Cars, and Stanford Dogs.

The detailed information of these datasets are shown as follows:

| Datasets         | CUB Birds | Stanford Cars | Stanford Dogs |
| ---------------- | --------- | ------------- | ------------- |
| Images           | 11,788    | 16,185        | 20,580        |
| Classes          | 200       | 196           | 120           |
| Training classes | 150       | 147           | 90            |
| Testing classes  | 50        | 49            | 30            |

For each dataset, we follow ([Wei et al. 2019](https://arxiv.org/abs/1805.04288); [Huang et al. 2021](https://arxiv.org/abs/1904.03580)) to randomly split its original image categories into two disjoint subsets.

## The comparison between the baseline results using our codes and the references

- The results are reported with mean accuracy with 0.95 confidence intervals over sampled 2000 testing sets.
- We apply data augmentation, which includes random crops, random horizontal flips, and color jitter at the meta-training stage, as well as center crops at the testing stage, in all implemented experiments.
- Our backbone is CNN-4 ([Shell, Swersky, and Zemel 2017](https://arxiv.org/abs/1703.05175); [Zhu, Liu, and Jiang 2020](https://www.ijcai.org/proceedings/2020/0152.pdf)), which is composed of four convolutional blocks and each block comprises a 64-filter 3 × 3 convolution, a batch normalization layer and a ReLU nonlinearity. The first three blocks contain a 2 × 2 max-pooling layer. The input of this network is 84×84, and obtain Ti with 10 × 10 × 64 elements.
- For hyperparameters, we set $\delta=0.6$ in Eq. (4), and $k=\sum_{u=1}^{n_i}{\sum_{v=1}^{n_i}{A_{u,v}}}=\lceil \frac{n_i^2}{t} \rceil$ in Eq. (5).
- We use the Adam optimizer with initial learning rate of 0.001. The total number of episode is 200,000 and the learning rate is of reduced as 1/2 after each 50,000 episodes. 

| Methods     | Type | Published in  | CUB 1-shot | CUB 5-shot | Cars 1-shot | Cars 5-shot | Dogs 1-shot | Dogs 5-shot |
| ----------- | ---- | ------------- | ---------- | ---------- | ----------- | ----------- | ----------- | ----------- |
| MatchingNet | FS   | NeurIPS’16    | 57.59±0.74 | 70.57±0.62 | 48.03±0.60  | 64.22±0.59  | 45.05±0.66  | 60.60±0.62  |
| ProtoNet    | FS   | NeurIPS’17    | 53.88±0.72 | 70.85±0.63 | 45.27±0.61  | 64.24±0.61  | 42.58±0.63  | 59.49±0.65  |
| RelationNet | FS   | CVPR’18       | 59.82±0.77 | 71.83±0.61 | 56.02±0.74  | 66.93±0.63  | 44.75±0.70  | 58.36±0.66  |
| DN4         | FS   | CVPR’19       | 53.15±0.84 | 81.90±0.60 | 61.51±0.85  | 89.60±0.44  | 45.73±0.76  | 66.33±0.66  |
| LRPABN      | FSFG | IEEE TMM’19   | 67.97±0.44 | 78.26±0.22 | 63.11±0.46  | 74.66±0.22  | 54.82±0.46  | 67.12±0.23  |
| BSNet       | FSFG | IEEE TIP’20   | 65.20±0.92 | 84.18±0.64 | 61.41±0.92  | 86.68±0.54  | 51.06±0.94  | 71.90±0.68  |
| MattML      | FSFG | IJCAI’20      | 66.29±0.56 | 80.34±0.30 | 66.11±0.54  | 82.80±0.28  | 54.84±0.53  | 71.34±0.38  |
| TOAN        | FSFG | IEEE TCSVT’21 | 65.34±0.75 | 80.43±0.60 | 65.90±0.72  | 84.24±0.48  | 49.30±0.77  | 67.16±0.49  |
| Ours        | FSFG | This paper    | 72.89±0.50 | 86.60±0.31 | 70.21±0.50  | 85.55±0.31  | 59.81±0.50  | 77.19±0.35  |

## Citation

```
@inproceedings{zhang2021tricks,
  author    = {Shu{-}Lin Xu and Faen Zhang and Xiu{-}Shen Wei and Jianhua Wang},
  title     = {Dual Attention Networks for Few-Shot Fine-Grained Recognition},
  pages = {2911-2919},
  booktitle = {AAAI},
  year      = {2022},
}
```
