---
layout: post
title:  "ML Interview Prep: Naive Bayes"
description: These posts are designed to be a quick overview of each machine learning model
image: /assets/ml-naive-bayes/two_prob_map.png
tags: [machine learning, interviews, data science]
---

***Under construction***

These posts are designed to be a quick overview of each machine learning model. The target audience is people with some ML background who want a quick reference or refresher. The following questions are compiled out of common things brought up in interviews.

1. Top-level
	1. What is the high-level version, explain in layman's terms
	2. What scenario should you use it in (classification vs regression, noisy data vs clean data)?
	3. What assumptions does the model make about the data?
	4. When does the model break/fail (adv & dis-advantages)? What are alternatives?

2. A bit more detail  
	1. How do you normalize the data for the model?
	2. What's the loss function used?  
	3. What's the complexity?

3. In-depth
	1. Probabilistic interpretation
	2. Derivation
	3. Simple implementation

4. More on training the model
	1. How can you validate the model?
	2. How do you deal with over-fitting?
	3. How to deal with imbalanced data?

---

## 1. Top-level

### 1.1 High-level Explanation

Naive Bayes is a generative model for classification. Generative models are called generative because they generate a probability distribution which describes a given class given the occurrence of data. Generative models differ from discriminative models which find a hard decision boundary that separates the target classes. Examples of discriminative modes would be [logistic regression]({{site.url}}/ml-interview-prep-logistic-regression/) or Decision Trees.

![http://www.inf.ed.ac.uk/teaching/courses/iaml/2011/slides/naive.pdf]({{site.url}}/assets/ml-naive-bayes/generative_model.png)
<center><small>Image credit: <a href='#ref'>V. Lavrenko</a></small></center><br/>

Unsurprisingly, Naive Bayes is governed by Bayes' Law:

$$
\underbrace{P(Y|X)}_{\rm posterior} = \frac{\overbrace{P(X|Y)}^{\rm likelihood} \overbrace{P(Y)}^{\rm prior}}{\underbrace{P(X)}_\text{probability of seeing the data}}.
$$

Our goal is to obtain an understanding of the probability that a given selection of data, $X$, will indicate the presence of a class $Y$, i.e. $P(Y\|X)$. Our ability to find this relation depends on our prior belief about the probability that we will see class $Y$, $P(Y)$; the likelihood that the data $X$ will occur given that we are looking at class $Y$, $P(X\|Y)$; and the overall probabilities of seeing the occurrences of the data $P(X)$.

This is easier to understand with an example:

Let's consider a Naive Bayes approach to the MNIST dataset. We select an image of the number 'two' and flatten it to a vector. We then plot the intensity for each pixel vs. the pixel number.

![]({{site.url}}/assets/ml-naive-bayes/two.png)

We can do this for every 'two' and build up an understanding of the average intensity at each pixel (mean, $\mu$), as well as what pixels see the widest variety of intensities (standard deviation, $\sigma$).

![]({{site.url}}/assets/ml-naive-bayes/all_twos.png)

We now have to make an explicit assumption about what probability function we expect the data to fall into. For this we will use a normal distribution. By doing this, we can generate a probability map based on these two parameters ($\mu$ and $\sigma$).

We now repeat this for all other classes (numbers) in the dataset.

![Likelihood of pixel occurrence for each digit, 0–9]({{site.url}}/assets/ml-naive-bayes/two_prob_map.png)

Each of these probability maps acts as a thumb-print to describe a class. Mathematically, we've built up the **probability of seeing a set of features (pixel intensities) given a class**,

$$
P(X|Y).
$$

This is defined as the likelihood, e.g. What is the *likelihood* that a given pixel will have a value of 255 *given* the digit is a 2.

Next we need to encode our **prior knowledge about the occurrence of the classes** $Y$. For digits, we'll say all values $\\{0, \cdots, 9\\}$ are equally likely. i.e. $P(Y) = 1/10$. However, if we knew this was not the case, we can encode this information as well. For example, if we knew we were classifying binary values we would set $P(Y{=}0) = P(Y{=}1) = .5$ and $P(Y{=}2, \cdots, 9) = 0$.

Now, if all were doing is picking the most likely class, we can drop the denominator $P(X)$. This is the **probability of features in the dataset** and only serves as a normalization constant, so it will be constant for each class. We can drop it and still obtain the most likely class:

$$
\begin{align*}
\hat{y} &= \arg \max_i \left \{ P(Y{=}y_i | X)  \right \} \\
&= \arg \max_i \left \{ P(X | Y{=}y_i) P(Y{=}y_i) \right \}.
\end{align*}
$$

If we want the true probability, then we have to calculate $P(X)$

$$
P(X) = \sum_{i} P(X|Y{=}y_i)P(Y{=}y_i)
$$


### 1.2 What scenario should you use it in?

Naive Bayes can work particularly well in scenarios where the datasets contains noisy data, missing data, a large number of outliers, or where there is a large class imbalance. This is assuming an accurate model for the prior and likelihood can be assumed, e.g. Is the data normal distributed? Laplacian?

#### 1.2.2 Noisy data

NB is insensitive to small changes in data. Noisy data will hurt the models performance; however, NB handles the input noise better than any other available models [[T. Pang-Ning. Section 5.3]](#ref).

If we were building an email spam detector, consider two cases of noise: **background (or irrelevant) data** and **incorrect classification**.

An example of **background** noise might be stop words, like "the", "is", "and", "at", or "which". In our spam/not-spam classifier for emails, these words would show up with the same regularity in each classification model. That is, the *likelihood* of seeing "the" is the same for an email that is spam and one that is not spam:

$$
P( X{=}\text{the} \,|\, \text{spam} ) = P( X{=}\text{the} \, | \, \neg \text{spam} ).
$$

In this case, the likelihood for spam/not-spam would both increase by the same constant, making this feature irrelevant for the classification decision. So we don't need to clean these features from our dataset [[cite this]](#ref).

In the case of noise introduced by **incorrect classifications**. NB is the best model to use [[cite this]](#ref).

#### 1.2.3 Missing values

Naive Bayes is very capable of handling missing values during inference. In this case, only the likelihoods based on the observed values are calculated [[cite this]](#ref). 

$$
P(x_1 + \cdots + x_j + \cdots + x_d | y) = \prod^d_{i \neq j} P(x_i | y)
$$

Wherein $x_j$ is a value missing during inference.

Training a NB model with missing values can be problematic in some situations, see the section on the zero-frequency problem for a discussion of this.

#### 1.2.4 Outliers

Outliers will not skew the learning model, because of the assumption of the data distribution. In the below example, the outlier does not significantly skew the PDF because its location can be described by $P(x)$ [[cite this]](#ref)

![]({{site.url}}/assets/ml-naive-bayes/gauss.png)

For the same reason, Naive Bayes is not used for outlier detection.

#### 1.2.5 Minority Class



### 1.3 What assumptions does the model make about the data?

This model makes the fundamental assumption that the data points are distributed by the probability function select to represent the likelihood. In the example below, NB will classify the data as described by a Normal distribution. It will make this assumption even if the sample histogram does not immediately mimic the assume PDF.

![]({{site.url}}/assets/ml-naive-bayes/pdf.png)
<center><small>Image credit: <a href='#ref'>V. Lavrenko</a></small></center><br/>

Additionally, NB will assume all of the features are conditionally independent. 

### 1.4 When does the model break / fail?

Naive Bayes fails when independence is not true between the input features. For example: consider classifying the phrase: "Chicago Bulls". Naive Bayes will classify this as a 'location' and an 'animal'. However, we know from context that this is neither of these things, and the input should instead be classified as 'basketball team'.

Another failure point with Naive Bayes is it's inability to separate classes when the only thing distinguishing them is their correlation. Because it is inherently taking a naive approach, it cannot distinguish between the two examples in the below image, as the probability distribution functions are completely overlapped. 

![]({{site.url}}/assets/ml-naive-bayes/correlation.png)
<center><small>Image credit: <a href='#ref'>V. Lavrenko</a></small></center><br/>

 
**Zero-frequency occurrence**

Because the finally probability is a function of the products of the likelihood, if an occurrence has never been seen before, the predicted probability will be 0.

$$
P(x_1 + \cdots + x_j + \cdots + x_d | y) = P(x_1|y) \times \cdots \times \underbrace{P(x_j|y)}_0 \times \cdots \times P(x_d|y)
$$

To account for this a distribution must be assumed. This allows occurrence to be interpolated. However, doing so encode another prior assumption about the data. If this prior is incorrect, this will effect the models ability to accurately classify the data.

![machine learning cheat sheet](https://blogs.sas.com/content/subconsciousmusings/files/2017/04/machine-learning-cheet-sheet.png)

## 2. A bit more detail

### 2.1 How do you normalize the data for the model, if you need to?

Although it is not strictly necessary to normalize the data going into a NB classifier, not doing so can add unnecessary complexity: 

During training of the model, a small number, $\epsilon$, is added to the standard deviation as a smoothing parameter to avoid division by zero. i.e. $ \exp \\{  (\mu-x)^2 / 2(\sigma + \epsilon)^2 \\}$

If the features varied in range, then the smoothing parameter would have to change to reflect this. This value can have a significant effect on the accuracy of the model. To convince yourself of this, try changing $\epsilon$ in the code below.
 
### 2.2  What's the complexity — runtime, parameters?

NB is one of the fastest learning methods. 

For number of classes $c$, number of instances $n$, number of dimensions $d$; the time complexity at training will be of order

$$
\mathcal{O}(nd+cd) \approx \mathcal{O}(nd)
$$

For a single input at inference, the complexity is

$$
\mathcal{O}(dc)
$$

## 3. In-depth

### 3.1 Probabilistic Interpretation

The model assumes that the features, $X$, are *conditionally* independent from one another. For example, in a data set it might appear that there is a correlation between the occurrences of $B$ and $C$. However, if it can be assumed that $B$ and $C$ are actually *mutually* independent then the correlation can be attributed to the existence of an external factor, $A$, then we can apply Naive Bayes.

![]({{site.url}}/assets/ml-naive-bayes/cond_indp.png)

As an example: if one were to look at the rate of heat stroke and the action of going to the beach, there might be a correlation. However, there is nothing intrinsic about going to the beach, that causes heat stroke. So if we consider an external factor, the temperature, we can model these features as mutually independent. Such that you're more likely to go to the beach when its hot, and your more likely to get heatstroke when its hot. In a NB classifier, the predicted class is this hidden dependence. Such that:

$$
P(Y{=}A | x_1{=}B, x_2{=}C)
$$

### 3.2 Derivation

The probability of a event $A$ and $B$ occurring, with the assumption of conditional independence is 

$$
P(A \cap B) = P(A|B)P(A).
$$

Likewise, the probability of a event $B$ and $A$ occurring is 

$$
P(B \cap A) = P(B|A)P(B).
$$

Because $P(A \cap B) = P(B \cap A)$ we can set the two equations equal to each other and find a description for the probability of $A$ occurring, given $B$ occurring:

$$
P(A|B) = \cfrac{P(B|A)P(B)}{P(A)}.
$$

I don't find the terms $A$ and $B$ particularly illuminating, so we can rewrite this in the Diachronic form: describing the probability of a hypothesis, $H$, being true, given some evidence, $E$, existing. 

$$
P(H|E) = \cfrac{P(E|H)P(H)}{P(E)}.
$$

Furthermore, the probability of an event, $P(E)$, is not always clear. I believe is it more obvious to write this in terms of $P(E)$ as a normalization constant:

$$
P(H|E) = \cfrac{P(E|H)P(H)}{\sum_{H'} P(E|H')P(H')}.
$$

If we are only concerned with "What is the most probably hypothesis that describes evidence $E$". We can drop the normalization, because it is constant across all predictions, and just take the hypothesis with the maximum value. Additionally, we take the log of the probabilities, to reduce the complexity of calculating the exponent in the Gaussian PDF.  

$$
\hat{y} = \arg \max_i \left \{ \log P(y_i | X)  \right \}
$$

such that

$$
\log P(y_i | X) \propto \log P(X|y_i) + \log P(y_i)
$$

and $\hat{y}$ is equal to the most likely hypothesis for multiple evidence, $X$.


### 3.3 Simple Implementation

~~~python
class NaiveBayes:
    
    def fit(self, X, Y, epsilon=1e-2):
        self.params = {}
        
        for c in np.unique(Y):
            current_x = X[Y==c]
            
            self.params[int(c)] = {
                'means':current_x.mean(axis=0),
                'vars':current_x.var(axis=0) + epsilon,
                'prior':len(Y[Y==c])/len(Y)
            }
                
    def predict(self, X):
        N, D = X.shape
        K = len(self.params)
        P = np.zeros((N,K))
        
        for c, p in self.params.items():
            P[:,c] = mvn.logpdf(X, mean=p['means'], cov=p['vars']) + np.log(p['prior'])

        return np.argmax(P, axis=1)
    
    def evaluate(self, X, Y):
        P = self.predict(X)
        return np.mean(Y == P)

# Normalize with min-max scaling. The data doesn not need to be normalized;
# however, the smoothing parameter in training will have to change to
# compinsate for this. If not normalizing, try epsilon = 255
data =  mnist.data
data = data - data.min()
data = data / data.max()

xtrain, xtest, ytrain, ytest = train_test_split(data, mnist.target)

nb = NaiveBayes()
nb.fit(xtrain, ytrain)
print("Accuracy on MNIST classification: {:.2f}%".format(100*nb.evaluate(xtest, ytest)))
~~~

`Accuracy on MNIST classification: 80.66%`

## 4. More on training the model

### 4.1 How can you validate the model?

Accuracy, F1, precision, recall, some other stuff. 

Must be evaluated against the base performance, i.e. the prior.  

### 4.2 How do you deal with over-fitting?

Naive bays is a high-bias model. For the most part, you don't have to worry about over fitting because it only has a few parameters, $\mathcal{O}(dc)$ [[Murphy, 3.5]](#ref). However, the method in which NB is trained will effect is susceptibility to over fit. If the model is trained with a maximum likelihood procedure, then the likelihood is generated directly from the occurrences in the data. This can then creates the zero-frequency problem, discussed above. Under these circumstances, a distribution for the likelihood must be assumed.

### 4.3  How to deal with imbalanced data?

As long as you can build a descriptive distribution for the likelihood, it is appropriate model to use [[cite this]](#ref).

<div id='ref'></div>
## 5. References

The notes above have been compiled from a variety of sources:

 1. [K. P. Murphy. Machine Learning: A Probabilistic Perspective. The MIT Press, 2012.](https://mitpress.mit.edu/books/machine-learning-1)
 2. [V. Lavrenko. Naive Bayes Classifier, 2015](https://www.youtube.com/playlist?list=PLBv09BD7ez_6CxkuiFTbL3jsn2Qd1IU7B)
 3. [T. Pang-Ning. Introduction to data mining. 2005.](https://www.youtube.com/playlist?list=PLBv09BD7ez_6CxkuiFTbL3jsn2Qd1IU7B)

----

 1. http://www.inf.ed.ac.uk/teaching/courses/iaml/slides/naive-2x2.pdf
 1. [G. James, D. Witten, T. Hastie, and R. Tibshirani. An Introduction to Statistical Learning. Springer, 2017.](https://www.amazon.com/Introduction-Statistical-Learning-Applications-Statistics/dp/1461471370)
 2. [C. M. Bishop. Pattern Recognition and Machine Learning. Springer, 2011.](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
 3. [K. P. Murphy. Machine Learning: A Probabilistic Perspective. The MIT Press, 2012.](https://mitpress.mit.edu/books/machine-learning-1)
 4. [T. Hastie, R. Tibshirani, and J. Friedman. The Elements of Statistical Learning, Second Edition. Springer, 2016.](https://web.stanford.edu/~hastie/Papers/ESLII.pdf)
 5. [I. Goodfellow, Y. Bengio, and A. Courville. Deep Learning. MIT Press, 2016.](https://www.deeplearningbook.org/)
 6. [A. Ng, CS229 Lecture notes, 2018](http://cs229.stanford.edu/notes/cs229-notes1.pdf)
