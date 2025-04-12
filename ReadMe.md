# A Light Spatial-Frequency Network for Robust Iris Segmentation and Localization

## Pipleline
![fig1](https://github.com/user-attachments/assets/2c4829e4-8da2-4233-bed7-6c54d2392932)

## Requirement

    pytorch
    python
    ptflops
    pandas
    numpy

## Testing

    python test-dis-SFNet.py

## Model Evaluation

### Complexity

Model Complexity: SFNet contains only 0.38M parameters, the computational complexity is 5.57 GFLOPs. This low-complexity design enables efficient deployment for mobile iris recognition, achieving real-time inference speeds of 57 FPS on 448×576 resolution images and 85 FPS on 384×384 resolution images.

| Method | Params(M) | FLOPs(G) |
| :----- | :-------- | :------- |
| SFNet   | 0.38      | 5.57     |

### Performance

Performance Evaluation: To comprehensively evaluate SFNet, we conducted experiments on six benchmark datasets including Lamp, Thousand, MICHE-I, CASIA-Iris-Distance, UBIRIS.v2 and M1. Segmentation performance was measured using  average segmentation error rate (E1) and the F-Measure (F1) , with boundary localization accuracy measured using three dedicated metrics: Outer mHdis (mean Hausdorff Distance) for outer boundaries, Inner mHdis for inner boundaries, and Overall mHdis for comprehensive boundary assessment.

|         Data        | E1(%) |  F1(%)  | Outer mHdis(%) | Inner mHdis(%) | Overall mHdis(%) |
| :-----------------: | :---: | :-----: | :------------: | :------------: | :--------------: |
|         Lamp        |  0.16 |  98.58  |     0.5402     |     0.4384     |      0.4893      |
|       Thousand      |  0.27 |  97.37  |     0.4291     |     0.5783     |      0.5037      |
|       MICHE-1       |  0.65 |  93.28  |     1.3293     |     1.0763     |      1.2028      |
| CASIA-Iris-Distance |  0.37 |  94.89  |     0.9673     |     0.5179     |      0.7426      |
|      UBIRIS.v2      |  0.79 |  92.45  |     1.3393     |     1.1288     |      1.2341      |
|          M1         |  0.59 | 93.9637 |     0.5456     |     0.4540     |      0.4998      |

## Citation

If you use our code or models in your research, please cite with:

    title = {A light spatial-frequency network for robust iris segmentation and localization},
    journal = {Applied Soft Computing},
    volume = {175},
    pages = {113009},
    year = {2025},
    issn = {1568-4946},
    author = {Qi Wang and Chun Xia and Yue Yan and Rui Zhang and Yang Liu}

