---
layout: post
title:  "ML Interview Prep: Loss functions"
date:   2018-10-05 13:48:19 -0700
---

These series of posts are designed to be a quick overview of each machine learning model. The target audience is people with some ML background who want a quick reference or refresher. The following questions are compiled out of common things brought up in interviews.

1. Top-level
	1. What is the high-level version, explain in layman's terms
	2. What scenario should you use it in (classification vs regression, noisy data vs clean data)?
	3. What assumptions does the model make about the data? (Linear, etc)?
	4. When does the model break/fail (adv & dis-advantages)? What are alternatives?

2. A bit more detail  
	1. How do you normalize the data for the model, if you need to? How does this deal with outliers? Skewed data?  
	2. What's the loss function used?  
	3. What's the complexity - runtime, parameters? How does it scale with # of features or input data  

3. In-depth
	1. Probabilistic interpretation
	2. Derivation
	3. Simple implementation

4. More on training the model (not model-specific, this should be common for most of the models):
	1. How can you validate the model?
	2. How do you deal with over-fitting?
	3. How to deal with imbalanced data?
	4. How do you regularize the model? Trade-offs?
	5. Does the model emphasize Type 1 or Type 2 errors?


---

## 1. Top-level

### 1.1 High-level Explanation
### 1.2 What scenario should you use linear regression


### 3.1 Probabilistic interpretation
We are trying to find the line, $\hat{y} = XW$, which maximizes the probability that a given point $(y_i, x_i)$ will fall on that line. To say that another way, "what is the probability that our best-fit-line is correct, given the data we have: $P( \hat{y}_i \| y_i)$." From Bayes' Law we know that this can be described as

$$
P(\hat{y}_i | y_i) = \frac{ P(\hat{y}_i) P(y_i|\hat{y}_i) }{P(y_i)}.
$$

We want to maximize the likelihood, $P( y_i \| \hat{y}_i )$, that a single datapoint, $y_i$, in our dataset will come from a distribution given by our best-fit-line, $\hat{y}_i$.

To do this, we make the assumption that the probability of a given value, $y_i$, would fall within a normal distribution for a specific, in which the value $\hat{y}_i$ is the mean of the distribution, $\mu$.

Modified from C. Bishop "Pattern Recognition and Machine learning"We are trying to predict the mean of the Gaussian distribution that best describes the points at $x_i$, so, we will use $\mu$ in place of our line-of-best-fit, $\hat{y}$. The equation for the Gaussian take on this form:

We take the $\log$ of P because it is less computationally expensive than having to calculate the exponent,

and drop the constant terms:

At this point, we've arrived at our L2 error function; it can be seen that maximizing the log-likelihood is equivalent to minimizing the squared error:
