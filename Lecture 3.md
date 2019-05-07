## Lecture 3

[TOC]

### Loss Function

![cs231n_2017_lecture310](assets/cs231n_2017_lecture310.png)

![cs231n_2017_lecture312](assets/cs231n_2017_lecture312.png)

![cs231n_2017_lecture320](assets/cs231n_2017_lecture320.png)

At initialization W is small so all s ≈ 0, $L_i=k-1$.

#### Hinge Loss VS. Squared Hinge Loss

Loss Function: Quantify how bad the mistakes made by classifiers are.

- Hinge Loss: we don’t want any wrong (increse wrong)
- Squared Hinge Loss: ignore a little bit wrong

### Regularization

![cs231n_2017_lecture333](assets/cs231n_2017_lecture333.png)

![cs231n_2017_lecture334](assets/cs231n_2017_lecture334.png)

### Softmax

![cs231n_2017_lecture346](assets/cs231n_2017_lecture346.png)

At initialization W is small so all s ≈ 0, $L_i=ln\frac{1}{k}$.

#### SVM VS. Softmax

![cs231n_2017_lecture349](assets/cs231n_2017_lecture349.png)

- SVM: ignore correct
- Softmax: want to be better

### SGD

![cs231n_2017_lecture372](assets/cs231n_2017_lecture372.png)

![cs231n_2017_lecture376](assets/cs231n_2017_lecture376.png)