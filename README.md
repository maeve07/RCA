# Regional Semantic Contrast and Aggregation for Weakly Supervised Semantic Segmentation

> [**Regional Semantic Contrast and Aggregation for Weakly Supervised Semantic Segmentation**](https://arxiv.org/abs/2203.09653),            
> [Tianfei Zhou](https://www.tfzhou.com/), Meijie Zhang, Fang Zhao, Jianwu Li <br>
> *To be appeared in CVPR 2022*

## News

* [2022-03-16] Repo created. Code will come soon.

## Abstract

Learning semantic segmentation from weakly-labeled (e.g., image tags only) data is challenging since it is hard to infer dense object regions from sparse semantic tags. Despite being broadly studied, most current efforts directly learn from limited semantic annotations carried by individual image or image pairs, and struggle to obtain integral localization maps. Our work alleviates this from a novel perspective, by exploring rich semantic contexts synergistically among abundant weakly-labeled training data for network learning and inference. In particular, we propose regional semantic contrast and aggregation (RCA) . RCA is equipped with a regional memory bank to store massive, diverse object patterns appearing in training data, which acts as strong support for exploration of dataset-level semantic structure. Particularly, we propose i) semantic contrast to drive network learning by contrasting massive categorical object regions, leading to a more holistic object pattern understanding, and ii) semantic aggregation to gather diverse relational contexts in the memory to enrich semantic representations. In this manner, RCA earns a strong capability of fine-grained semantic understanding, and eventually establishes new state-of-the-art results on two popular benchmarks, i.e., PASCAL VOC 2012 and COCO 2014.

## Citation
```
@inproceedings{zhou2022regional,
    author    = {Zhou, Tianfei and Zhang, Meijie and Zhao, Fang and Li, Jianwu},
    title     = {Regional Semantic Contrast and Aggregation for Weakly Supervised Semantic Segmentation},
    booktitle = {CVPR},
    year      = {2022}
}
```
