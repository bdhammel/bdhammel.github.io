---
layout: post
title:  "ML Interview Prep: Naive Bayes"
date:   2018-10-09 13:48:19 -0700
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
Naive Bayes is a generative model. That is, instead of finding a decision boundary, it builds up a probability map to predict a classification of input features.

![http://www.inf.ed.ac.uk/teaching/courses/iaml/2011/slides/naive.pdf]({{ site.url}}/assets/ml-naive-bayes/generative_model.png)

The model's prediction of the probability that the data belongs to a class is given by Bayes' Law:

$$
P(Y|X) = \frac{P(X|Y) P(Y)}{P(X)}
$$

Wherein $P(Y)$ is defined as our prior belief of the probability of occurrence of the class $Y$; $P(X\|Y)$ is the likelihood that features $X$ will occur given that we are looking at class $Y$; and $P(X)$ is the overall probability occurrences of the features.

This is easier to understand with an example:

Lets consider a Naive Bayes approach to the MNIST dataset. We select an image of a number, 'two', and flatten it to a vector with the intensity values plotted.

![]({{ site.url}}/assets/ml-naive-bayes/two.png)

We can do this for every single 'two', and build up an understanding of the average intensity at each pixel (mean), as well as what pixels see the widest variety of intensities (standard deviation).

![]({{ site.url}}/assets/ml-naive-bayes/all_twos.png)

We make the fundamental assumption now that the occurrence of intensities follows a normal distribution. Doing so, we can generate a probability map based on these two parameters ($\mu$ and $\sigma$).


![Likelihood of pixel occurrence for each digit, 0–9]({{ site.url}}/assets/ml-naive-bayes/two_prob_map.png)

Each one of these probability maps acts as a thumb-print to describe a class, in this case the class is the digit. Mathematically, we've built up the probability of occurrences of features given a class,

$$
P(X|Y)
$$

this is defined as the likelihood, e.g. What is the *likelihood* that a given pixel will have a value of 255 *given* the digit is a 2.

### 1.2 What scenario should you use it in (classification vs regression, noisy data vs clean data)?

#### 1.2.1 Classification vs Regression

NB is a generative model for classification

#### 1.2.2 Noisy data

NB is insensitive to small changes in data. 

Noisy data will hurt the models performance; however, NB handles the input noise better than any other available models.

Consider two cases of noise: background and incorrect classification.

An example of background noise might be stop words, like "the", "is", "and", "at", or "which". If you were building a spam/not-spam classifier for emails. Then these words would show up with the same regularity in each classification model, that is, the likelihood of seeing "the" is the same for an email that is spam and one that is not spam:

