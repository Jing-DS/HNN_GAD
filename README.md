# Hyperbolic Neural Networks in Node-Level Graph Anomaly Detection

This repository provides official implementation of the model from the following paper.

>Three Revisits to Node-Level Graph Anomaly Detection: Outliers, Message Passing and Hyperbolic Neural Networks
>
>Jing Gu and Dongmian Zou, Duke Kunshan University, 2023
>
>OpenReview: https://openreview.net/forum?id=fNsU9gi1Fy&noteId=fNsU9gi1Fy

## Training

In the folder `HNN_GAD`, can specify parameters in `config.py` and run the model via
```
python run.py
```

## Environment

- torch==1.9.1+cu111
- torch_sparse==0.6.12
- torch_scatter==2.0.9
- torch_geometric==2.1.0
- python==3.7.13
- scikit-learn
- networkx
- ogb
- geoopt
- jupyter
- nb_conda_kernels

For more specific information, please see `environment.yml`.

## Citation

```
Gu, Jing, and Dongmian Zou. "Three Revisits to Node-Level Graph Anomaly Detection: Outliers, Message Passing and Hyperbolic Neural Networks." Proceedings of the Second Learning on Graphs Conference (LoG 2023), PMLR 231, Virtual Event, November 27–30, 2023.
```

or

```
@inproceedings{gu2023three,
  title={Three Revisits to Node-Level Graph Anomaly Detection: Outliers, Message Passing and Hyperbolic Neural Networks},
  author={Gu, Jing and Zou, Dongmian},
  booktitle={The Second Learning on Graphs Conference},
  year={2023}
}
```

## Reference
For construction of hyperbolic models, we utilized code available at https://github.com/HazyResearch/hgcn.

