---
layout: post
title:  "Maximum Likelihood Estimation and Maximum A Posterior Estimation"
description: A quick description on the difference between MLE and MAP and how to use them
image: assets/mle-map/apple-scale.png
tags: [machine learning, data science]
---

## What is the difference between MLE and MAP?

Maximum likelihood is a special case of Maximum A Posterior estimation. To be specific, MLE is what you get when you do MAP estimation using a _uniform_ prior. Both methods come about when we want to answer a question of the form: "What the probability of scenario $Y$ given some data, $X$ i.e. $P(Y\|X)$.

A question of this form is commonly answered using Bayes' Law.

$$
\underbrace{P(Y|X)}_{\rm posterior} = \frac{\overbrace{P(X|Y)}^{\rm likelihood} \overbrace{P(Y)}^{\rm prior}}{\underbrace{P(X)}_\text{probability of seeing the data}}.
$$

**MLE**  
If we're doing Maximum Likelihood Estimation, we do not consider prior information (this is another way of saying "we have a uniform prior"). In this case, the above equation reduces to.

$$
P(Y|X) \propto P(X|Y)
$$

In this scenario, we can fit a statistical model to correctly predict the posterior, $P(Y\|X)$, by maximizing the likelihood, $P(X\|Y)$. Hence "Maximum Likelihood Estimation."

**MAP**   
If we know something about the probability of $Y$, we can incorporate it into the equation in the form of the prior, $P(Y)$. In This case, Bayes' laws has it's original form.

$$
P(Y|X) \propto P(X|Y)P(Y)
$$

We then find the posterior by taking into account the likelihood and our prior belief about $Y$. Hence "Maximum A Posterior".

## Using MLE or MAP estimation to answer a question

Let's say you have a barrel of apples that are all different sizes. You pick an apple at random, and you want to know its weight. Unfortunately, all you have is a broken scale.

![Apple on scale]({{ site.url}}/assets/mle-map/apple-scale.png)

For the sake of this example, lets say you know the scale returns the weight of the object with an error of +/- a standard deviation of 10g (later, we'll talk about what happens when you don't know the error). We can describe this mathematically as:

$$
{\rm measurment} = {\rm weight} + \mathcal{N}(0, 10g)
$$

Let's also say we can weigh the apple as many times as we want, so we'll weigh it 100 times.  

We can look at our measurements by plotting them with a histogram

![data histogram]({{ site.url}}/assets/mle-map/hist.png)

Now, with this many data points we could just take the average and be done with it

$$
\begin{align}
\mu &= \frac{1}{N} \sum_i^N x_i \\
{\rm SE} &= \frac{\sigma}{\sqrt{N}}
\end{align}
$$

`The weight of the apple is (69.62 +/- 1.03) g`

<small> If the $\sqrt{N}$ doesnt look familiar, this is the "standard error"</small>

No surprises, nothing fancy. But the whole reason we're doing this is we wont always be able to measure the apple 100 times - perhaps the scale were using requires us to pay every time we use it. or maybe we're at a carnival, and were trying to win the grand prize by guessing the weight and we can only take 5 measurements. Whatever the reason, there will always be instances in the real world where we don't have enough data. To flush out the mechanics, we'll use this number of samples for now, and we'll look at what happens when we have less data later.


### Maximum Likelihood Estimation using a grid approximation
Just as a quick refresher: Our end goal is to find the weight of the apple, given the data we have. To formulate it in a Bayesian way: We'll ask what is the _probability_ of the apple having weight, $w$, given the measurements we took, $X$. And, because were formulating this in a Bayesian way, we use Bayes' Law to find the answer:

$$
P(w|X) = \frac{P(X|w)P(w)}{P(X)}
$$

If we make no assumptions about the initial weight of our apple, then we can drop $P(w)$. We'll say all sizes of apples are equally likely (we'll revisit this assumption in the MAP approximation).

Furthermore, we'll drop $P(X)$ - the probability of seeing our data. This is a normalization constant and will be important if we _do_ want to know the probabilities of apple weights. But, for right now, our end goal is to only to find the _most_ probable weight. P(X) is independent of $w$, so we can drop it if we're doing relative comparisons.

This leaves us with $P(X\|w)$, our likelihood, as in, what is the likelihood that we would see the data, $X$, given an apple of weight $w$. If we maximize this, we maximize the probability that we will guess the right weight.

#### Enter the grid approximation:

The grid approximation is probably the dumbest (simplest) way to do this. Basically, we'll systematically step through different weight guesses, and compare what it would look like if this hypothetical weight were to generate data. We'll compare this hypothetical data to our real data and pick the one the matches the best.

**To put this illustratively:**

![Illustrative MLE]({{ site.url}}/assets/mle-map/mle-grid.png)

**To formulate this mathematically:**

For each of these guesses, we're asking "what is the probability that the data we have, came from the distribution that our weight guess would generate". Because each measurement is independent from another, we can break the above equation down into finding the probability on a per measurement basis

$$
P(X|w) = \prod_{i}^{N} p(x_i | w)
$$

So, if we multiply the probability that we would see each individual data point - given our weight guess - then we can find one number comparing our weight guess to all of our data. We can then plot this:

![Likelihood]({{ site.url}}/assets/mle-map/likelihood.png)

There you have it, we see a peak in the likelihood right around the weight of the apple. 

But, you'll notice that the units on the y-axis are in the range of 1e-164. This is because we took the product of a whole bunch of numbers less that 1. If we were to collect even more data, we would end up fighting numerical instabilities because we just cannot represent numbers that small on the computer. To make life computationally easier, we'll use the logarithm trick.

$$
\begin{align}
\log P(X|w) &= \log \prod_{i}^{N} p(x_i | w) \\
&= \sum_{i}^{N} \log p(d_i | w)
\end{align}
$$

This is the _log likelihood_. We can do this because the likelihood is a [monotonically increasing function](https://en.wikipedia.org/wiki/Monotonic_function)

![Log Likelihood]({{ site.url}}/assets/mle-map/log-likelihood.png)

These numbers are much more reasonable, and our peak is guaranteed in the same place.

**To implement this in code**

Implementing this in code is very simple. The python snipped below accomplished the what we want to do.

```Python
from scipy.stats import norm
import numpy as np

weight_grid = np.linspace(0, 100)

likelihoods = [
  np.sum(norm(weight_guess, 10).logpdf(DATA))
  for weight_guess in wieght_grid
]
weight = weight_guess[np.argmax(likelihoods)]
 ```

### MLE grid approximation for multiple parameters

Now lets say we don't know the error of the scale. We know that its additive random normal, but we don't know what the standard deviation is

$$
{\rm measurment} = {\rm weight} + \mathcal{N}(0, \sigma)
$$

We can use the exact same mechanics, but now we need to consider a new degree of freedom.

In other words, we want to find the weight of the apple and the error of the scale

$$
P(w, \sigma |X) \propto P(X|w, \sigma)
$$

![MLE 2D Grid]({{ site.url}}/assets/mle-map/mle-grid-2d.png)

Comparing log likelihoods like we did above, we come out with a 2D heat map

![MLE heat map]({{ site.url}}/assets/mle-map/likelihood-2d.png)

The maximum point will then give us both our value for the apple's weight and the error in the scale.

### Maximum A Posterior with a grid approximation

In the above examples we made the assumption that all apple weights were equally likely. This simplified Bayes' law so that we only needed to maximize the likelihood

$$
P(w, \sigma|X) \propto P(X|w, \sigma)
$$

However, not knowing anything about apples isn't really true. We know an apple probably isn't as small as 10g, and probably not as big as 500g. In fact, a quick internet search will tell us that the average apple is between 70-100g. So, we can use this information to our advantage, and we encode it into our problem in the form of the prior.

$$
P(w, \sigma|X) \propto P(X|w, \sigma)P(w, \sigma)
$$

By recognizing that weight is independent of scale error, we can simplify things a bit. So we split our prior up

$$
P(w, \sigma) = P(w)P(\sigma)
$$

Like we just saw, an apple is around 70-100g so maybe we'd pick the prior

$$
P(w) = \mathcal{N}(85, 40)
$$

![weight prior]({{ site.url}}/assets/mle-map/prior_w.png)

Likewise, we can pick a prior for our scale error. We're going to assume that broken scale is more likely to be a little wrong as opposed to very wrong

$$
P(\sigma) = {\rm Inv{-}Gamma}(.05)

$$

![weight prior]({{ site.url}}/assets/mle-map/prior_err.png)

With these two together, we build up a grid of our prior using the same grid discretization steps as our likelihood. We then weight our likelihood with this prior via element-wise multiplication. 

![likelihood times prior]({{ site.url}}/assets/mle-map/likelihoodprior.png)

From this, we obtain our Posterior.

![Posterior]({{ site.url}}/assets/mle-map/posterior.png)

In this example, the answer we get from the MAP method is almost equivalent to our answer from MLE. This is because we have so many data points that it dominates any prior information we put in. But I encourage you to play with the example code at the bottom of this post to explore when each method is the most appropriate.

## Example code

Play around with the code and try to answer the following questions

 1. How sensitive is the MAP measurement to the choice of prior?
 2. How sensitive is the MLE and MAP answer to the grid size?
 3. How many measurements do you need to take before all three methods return the same (or close tot the same) answers?

<script src="https://gist.github.com/bdhammel/8c95b028442e507f961e557c57d65be8.js"></script>


