---
layout: post
title:  "What learning rate should I use?"
---

One of the first decisions you make when starting to train a neural network is: **what to set the learning rate to?** This is often times one of those things in deep learning people lump into the "alchemy" category because there really isn't a one size-fits-all answer to this. Instead it takes tinkering on the part of the researcher to find the most appropriate number for that given problem. This post tries to provide a little more intuition around picking learning rates.

Depending on the shape of the loss surface, the optimizer, and your choice for learning rate, $\eta$, will determine how fast (and 'if') you can converge to the target minimum using gradient decent: 

$$
w \leftarrow w - \eta \frac{d\mathcal{L}}{dw},
$$

![]({{site.url}}/assets/learning-rate/lr-types.png)

  - A learning rate that is too low will take a long time to converge. This is especially true if there are a lot of saddle points in the loss-space. Along a saddle point $d \mathcal{L} / dw$ will be close to zero in many directions. If the learning rate $\eta$ is also very low, it can slow down the learning substantially. 
  - A learning rate that is too high can "jump" over the best configurations
  - A learning rate that is much too high can lead to divergence


## Optimizers

One of the choices you'll make before picking a learning rate is "What optimizer should I use?". I'm not going to dive into this, Because there's so much other good literature out. Instead I'll just cite these links which I've found helpful.

 - [Why Momentum Really Works](https://distill.pub/2017/momentum/)
 - [c231](http://cs231n.github.io/neural-networks-3/)
 - [An overview of gradient descent optimization algorithms](http://ruder.io/optimizing-gradient-descent/)
 - [AdamW and Super-convergence is now the fastest way to train neural nets](https://www.fast.ai/2018/07/02/adam-weight-decay/)

For this post, I'll only be talking about SGD. But be aware that you're choice of optimizer will also effect the learning rate.

## Visualizing the loss surface

https://www.cs.umd.edu/~tomg/projects/landscapes/

The architecture of a neural network can have significant effects on the structure of the loss-landscape. Because these neural networks are highly-non-linear, there is no grantee the loss surface is convex. 




## Local minima and deep learning

Finally, we prove that recovering the global minimum becomes harder as the network size increases and that it is in practice irrelevant as global minimum often leads to overfitting."


Here we argue, based on results from statistical physics, random matrix theory, neural network theory, and empirical evidence, that a deeper and more profound difficulty originates from the proliferation of saddle points, not local minima, especially in high dimensional problems of practical interest. Such saddle points are surrounded by high error plateaus that can dramatically slow down learning, and give the illusory impression of the existence of a local minimum.

https://arxiv.org/pdf/1406.2572.pdf

https://datascience.stackexchange.com/questions/22853/local-minima-vs-saddle-points-in-deep-learning


## Learning rate and batch size 

Batch size is another one of those things you'll set initially in your problem. Usually this doesn't require too much thinking, and you just increase the batch size until you get an OOM error on your GPU. 

But lets say you want to scale up and start doing distributed training. 


When the minibatch size is multiplied by k, multiply the learning rate by k (All other hyperparameters are kept unchanged (weight decay, etc)

$$
\begin{align*}
w_{t+k} &= w_t - \eta \frac{1}{n}\sum_{j<k}\sum_{x\ni  B} \nabla  l(x, w_{t+j}) \\
\hat{w}_{t+k} &= w_t - \hat{\eta} \frac{1}{kn}\sum_{j<k}\sum_{x\ni  B} \nabla  l(x, w_{t+j})
\end{align*}
$$

With 

$$
\hat{\eta} = k\eta
$$

To improve initial stability with larger learning rate: Train with a lower learning rate of increment by a constant amount at each iterator to target value of  over 5 epochs

References
 - https://arxiv.org/abs/1706.02677
 - https://arxiv.org/pdf/1804.07612.pdf
 - https://openreview.net/pdf?id=B1Yy1BxCZ


## Picking the best learning rate

### Learning rate finder

If you want some convincing of this method, this is a simple implementation on a linear regression problem:

It tries to find the simple line:

$$
y=2.5x
$$

If you had some incorrect value for w there would be some loss associated with that, where

$$
L =  \sum_i (y_i-wx_i)^2
$$

![]({{site.url}}/assets/learning-rate/loss.png)

To find the optimal learning rate, we randomly select a value for $w$. Lets say we pick "8". For $W=8$ there's a loss associated with that, ~1e6, given by the above graph.

We take an tiny step, which is determined by the smallest learning rate we want to use (min_lr in the example implementation code) and recalculate our loss. We didn't move very far, so our loss is about the same, ~ 1e6.

We keep doing that, slowly increasing the LR each time. Eventually, we'll find a point where we start moving down the error function faster. We can then plot our loss vs the lr during this exploration.

![]({{site.url}}/assets/learning-rate/loss_v_lr.png)

We're interested in the region where we're moving down the error function the fastest, i.e. the region of the largest change in loss for a given lr. So, we select ~1e-4 as our optimal learning rate.

### Experiment

## Reduce LR on Plateau

### Cyclical Learning Rates

## Transfer learning and Freezing Layers

## Differential Learning Rates



~~~python
import numpy as np
import matplotlib.pyplot as plt

BASE_LR = 1e-7
EPOCHS = 500
TRIALS = 3


def lr_scanner(method, factor):
    def _increase_lr(lr):
        if method == 'linear':
            lr += factor 
        elif method == 'exp':
            lr *= factor
        return lr
    return _increase_lr

def calculate_loss(x, y):
    def _calculate_loss(w):
        return np.mean((y-w*x)**2)
    return _calculate_loss


# You can uncomment one-or-the-other to try different learning rate searchers

increase_lr = lr_scanner(method='linear', factor=5e-4)
# increase_lr = lr_scanner(method='exp', factor=1.1)

if __name__ == "__main__":
    plt.ion()
    plt.close('all')

    # Equation to find, want to find w, such that y = y_
    # y_ = wx
    x = np.linspace(0, 10, 100)
    y = 2.5*x
    plt.figure()
    plt.plot(x, y, 'o')
    plt.xlabel("x")
    plt.ylabel("y")

    # Plot the loss, the mean squared error:  (y-y_)**2
    # for various weights, w
    w = np.linspace(-5, 10, 200)
    loss = np.mean((y - np.outer(w, x))**2, axis=1)
    plt.figure()
    plt.plot(w, loss)
    plt.xlabel("weights")
    plt.ylabel("loss")

    # Find the best learning rate to use with gradient decent to find the 
    # optimal value w
    # w <- w - lr * dl / dw

    loss_fn = calculate_loss(x, y)

    plt.figure()

    for trial in range(TRIALS):

        lr = BASE_LR

        # randomly select a starting value for w
        # compute the loss associated with this weight 
        w_ = np.random.uniform(w.min(), w.max())
        loss_ = loss_fn(w_)

        weights = [w_]
        losses = [loss_]
        lrs = [lr]
        dldw = 1

        # For each epoch, update the weight using gradient decent, and increase
        # the learning rate
        for _ in range(EPOCHS):

            # update weight and calculate new loss (error)
            w_ -= lr * dldw
            loss_ = loss_fn(w_)
            
            # Take gradient 
            dldw = (loss_ - losses[-1])/(w_ - weights[-1])

            # Store the values 
            lrs.append(lr)
            losses.append(loss_)
            weights.append(w_)

            # Increase the learning rate
            lr = increase_lr(lr) 

            # once we start to diverge, go ahead and break out of the loop, there's
            # not coming back
            if loss_ > 1e7:
                break

        plt.plot(lrs, losses, label=f"trial: {trial}")

    plt.xlabel("learning rate")
    plt.ylabel("loss")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
~~~
