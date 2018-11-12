---
layout: post
title:  "Learning Rate Finder"
---

## Theory

If you want some convincing of this method, this is a simple implementation on a linear regression problem:

It tries to find the simple line:

$$
y=2.5x
$$

If you had some incorrect value for w there would be some loss associated with that, where

[fix indices in this]

$$
L =  \sum_i (y_i-w_ix)^2
$$

![]({{site.url}}/assets/learning-rate/loss.png)

To find the optimal learning rate, we randomly select a value for $w$. Lets say we pick "8". For $W=8$ there's a loss associated with that, ~1e6, given by the above graph.

We take an tiny step, which is determined by the smallest learning rate we want to use (min_lr in the example implementation code) and recalculate our loss. We didn't move very far, so our loss is about the same, ~ 1e6.

We keep doing that, slowly increasing the LR each time. Eventually, we'll find a point where we start moving down the error function faster. We can then plot our loss vs the lr during this exploration.

![]({{site.url}}/assets/learning-rate/loss_v_lr.png)

We're interested in the region where we're moving down the error function the fastest, i.e. the region of the largest change in loss for a given lr. So, we select ~1e-4 as our optimal learning rate.

## Experiment

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
