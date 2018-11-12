---
layout: post
title:  "ML Interview Prep: Logistic Regression"
---

***Under construction***

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

### 1.1 High-level explanation

Logistic regression is a binary classification model (it can be extended to [multi-class classification](https://www.quora.com/Can-you-do-multiclass-classification-with-logistic-regression), but it is only a special treatment of the binary classification model).

The approach makes the explicit assumption that your data can be separated by a line or hyper-plane. Like linear regression, logistic regression is a linear model. It is defined by the functional form:

$$
\begin{align*}
y = f(x_i) &= \sigma(w^Tx+b) \\
&= \frac{1}{1+\exp\left\{ -(w^Tx+b) \right \}}
\end{align*}
$$

Consider the example below, of two Gaussian clouds centered at (1,1) and (-1,-1):

![]({{ site.url}}/assets/ml-logistic-regress/fig1.png)

It should be obvious that the dividing line, $x_2=-x_1$, separates the two classes, but lets explore this mathematically.

First, lets drop the sigmoid from the equation for classification (above). Using this, we can describe the system as:

$$
f(x_1, x_2) = x_2 + x_1 = 0
$$

such that $w_1=1$ and $w_2=1$.

Taking the point $(-1,-1)$ and plugging it in to the above equation yields:

$$
\begin{align*}
f(-1,-1) &= -2\\
-2 < 0 \therefore (-1,-1) &= {\rm Blue}.
\end{align*}
$$

Whereas, the point (1,1) yields:

$$
\begin{align*}
f(1,1) &= 2 \\
2 >0 \therefore (1,1) &= {\rm Red}.
\end{align*}
$$

Now, the values 2 or -2 are not particularly illuminating. So, we use the sigmoid function to squash the output into a probability, such that the output $y<.5\equiv{\rm Blue}$ and $y>.5\equiv{\rm Red}$. To put this graphically:

![]({{ site.url}}/assets/ml-logistic-regress/fig2.png)

### 1.2 What scenario should you use logistic regression?

When you want to have a fast, explainable model. 

### 1.3 Assumptions of Linear Regression

Similar to linear regression, logistic regression is an appropriate choice when classifying data that is linearly separable. This either has to be innately true about the data or carefully constructed into custom 'features' - using sufficient domain knowledge. For example, consider the "donut" problem below:

![]({{ site.url}}/assets/ml-logistic-regress/fig3.png)

This is not a linear classification problem - no straight line will separate these classes. However, similarly to the polynomial problem in my [post on linear regression]({{site.url}}/2018/10/16/ml-interview-prep-linear-regression.html), we can construct a custom representation of the data by adding an extra dimension:

$$
x_3 = \sqrt{x_2^2 + x_1^2}
$$ 

![]({{ site.url}}/assets/ml-logistic-regress/fig4.png)

These clusters can now be separated with a hyperplane in $x_1, x_2$-space, at $x_3 \approx 2$.

### 1.4 When the model breaks & what's a good backup?

The model will break when there is not a linear decision boundary the separates the two classes.

![](https://blogs.sas.com/content/subconsciousmusings/files/2017/04/machine-learning-cheet-sheet.png)

## 2. A bit more detail

### 2.1 Normalization of data 

### 2.1 Loss function

The loss function for binary logistic regression is the *cross-entropy error*

$$
\mathcal{L} = - \left \{ t\log(y) + (1-t)\log(1-y) \right \}
$$

wherein $t$ is the target class, and $y$ is the class predicted by the model.

It can be helpful to notice that only one of these terms will matter at a time. For example given a target class of $t=0$ only the second term will matter, $-\log(1-y)$. Whereas if $t=1$ only the first term will matter, $-\log(y)$.

### 2.3 What's the complexity

The complexity for logistic regression is the same as for linear regression:

**training**: $\mathcal{O}(p^2n+p^3) $  
**prediction**: $\mathcal{O}(p)$

Wherein $n$ is the number of training sample and $p$ is the number of features

[https://www.thekerneltrip.com/machine/learning/computational-complexity-learning-algorithms/](https://www.thekerneltrip.com/machine/learning/computational-complexity-learning-algorithms/)

## 3. In-depth

### 3.1 Probabilistic interpretation 

#### 3.1.1 Understanding the loss function

Logistic regression operates on the fundamental assumption that the data falls into a binomial distribution and each of the data points are independent from one-another. We approach this derivation this using [Bayes' Law](https://en.wikipedia.org/wiki/Bayes%27_theorem). For a justification of the Bayesian approach, check out my [3.1 probablistic interpretation section of my post on linear regression]({{site.url}}/2018/10/16/ml-interview-prep-linear-regression.html).

$$
P(y|X) = \frac{P(y) P(X|y)}{P(X)} 
$$

The term in this that we have control over is the likelihood, $P(X\|y)$. We want to maximize this term during training, thereby maximizing the probability that a data point falls into the correct class, $P(y\|X)$.

We define the likelihood based on out prior belief that the data will fall into a binomial distribution.

$$
\begin{align*}
P(X | y) &= P(x_1, x_2, \cdot, x_n | y) \\ 
&= \prod_{n}^{i} P(x_i | y) \\
&= \prod_{n}^{i} y_i^{t_i}(1-y_i)^{t_i}
\end{align*}
$$

Taking the negative log of this, to reduce the complexity of calculating exponents, renders the cross-entropy loss function. 

$$
- \log P(X|y) = \sum_i \left \{ t_i\log y_i + (1-t_i)\log (1-y_i) \right \}
$$

To summarize, logistic regression fundamentally assumes the data falls into a binomial distribution, and by **maximizing** the log of the likelihood (log-likelihood) we **minimizing** the cross-entropy error. 


#### 3.1.2 Understanding connection to linear regression

We have been writing the logistic regression equation as $y=\sigma(w^Tx)$ but in actuality what we are describing the is *probability* that a data point is in a class, 0 or 1:

$$
\begin{align*}
P(y=1 | x) &= \sigma(w^Tx) \\
P(y=0 | x) &= 1 - \sigma(w^Tx)
\end{align*}
$$

We can also think about this in terms of "what are the odds that data point $x_i$ falls into class 0 or class 1", in which case we take the ratio of these probabilities. 

$$
\begin{align*}
\frac{P(y=1|x)}{P(y=0|x)} &= \cfrac{\cfrac{1}{1+\exp(-w^Tx)}}{\cfrac{\exp(-w^Tx)}{1+\exp(-w^Tx)}} \\
&= \frac{1}{\exp(-w^Tx)} \\
&= \exp(w^Tx) \\
\end{align*}
$$

Taking the log of this equation we recover the description of linear regression:

$$
\log \left ( \frac{P(y=1|x)}{P(y=0|x)} \right ) = w^Tx
$$

So, when we find the optimal decision boundary in logistic regression what we're actually doing is finding a best-fit-line that minimizes the log-odds.

### 3.2 Derivation of the analytic solution

A closed-form solution exists for logistic regression. It makes my head hurt. I'll try to add it later. Similar to Linear regression, gradient decent is used.

**[Need to treat indices rigorously, and convert sum to matrix]**

As with most of these ML problems, we can use gradient decent to find the solution numerically.

$$
w \leftarrow w - \frac{d}{dw}\mathcal{L} 
$$

With our definition of loss being the cross-entropy, we find it's derivative w.r.t. the model weights using the chain rule:

$$
\frac{d}{dw}\mathcal{L} = \frac{d\mathcal{L}}{dy}\frac{dy}{dz}\frac{dz}{dw}
$$

where

$$
\begin{align*}
y &= \sigma(z) \\
z &= w^Tx \\ \\[5mm]
\end{align*}
$$

$$
\begin{align*}
\frac{d\mathcal{L}}{dy} &= -\frac{d}{dy} \sum \left \{ t\log(y) + (1-t)\log(1-y) \right \} \\ 
&= - t\frac{1}{y} - (1-t)\frac{1}{1-y} \\ \\[2mm]
\frac{dy}{dz} &= \frac{d}{dz} \frac{1}{1+\exp(-z)} \\
&= \frac{1}{1+\exp(-z)}\frac{\exp(-z)}{1+\exp(-z)} \\ 
&= y(1-y) \\ \\[2mm]
\frac{dz}{dw} &= \frac{d}{dw} w^Tx \\
&= x
\end{align*}
$$

Therefore,

$$
\begin{align*}
\frac{d}{dw}\mathcal{L} = (y-t)x
\end{align*}
$$

### 3.3 Simple Implementation

~~~python
class LogisticRegression:
    
    def __init__(self, order):
        self.W = np.random.randn((order+1))
    
    def fit(self, X, Y, alpha=1e-1, epochs=1000):
        N, D = X.shape
        X = np.hstack((X, np.ones(shape=(N,1))))
                      
        for _ in range(epochs):
            Y_hat = self.perdict(X)
            dL = X.T.dot(Y_hat-Y)  # X^T (Y_hat - Y)
            self.W -= alpha*dL                  
            
    def perdict(self, X):
        return sigmoid(X.dot(self.W))
    
    def coeff(self):
        return self.W.flatten()


X = np.random.randn(200,2)/3
X[100:,:] += 1
X[:100,:] -= 1
Y = np.zeros(200)
Y[100:] = 1

plt.figure(figsize=(5,5))
plt.scatter(X[Y==0,0], X[Y==0,1], color='b', alpha=.5)
plt.scatter(X[Y==1,0], X[Y==1,1], color='r', alpha=.5)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.show()

lr = LogisticRegression(order=2)
lr.fit(X, Y)

w1, w2, b = lr.coeff()
w = -w1/w2
b = (b-.5)/w2

print("The decision boundary is along: y={:.0f}x+{:.0f}".format(w, b))
plt.plot(x1, x2, '--')
~~~

![]({{ site.url}}/assets/ml-logistic-regress/fig5.png)

## 4. Training the model:
### 4.1 How can you validate the model?
### 4.2 How do you deal with over-fitting?
### 4.3 How to deal with imbalanced data?
### 4.4 How do you regularize the model? Trade-offs?
### 4.5 Does the model emphasize Type 1 or Type 2 errors?

## 5. References

The notes above have been compiled from a variety of sources:
 -  [G. James, D. Witten, T. Hastie, and R. Tibshirani. An Introduction to Statistical Learning: with Appli- cations in R (Springer Texts in Statistics). Springer, 2017.](https://www.amazon.com/Introduction-Statistical-Learning-Applications-Statistics/dp/1461471370)
 -  [C. M. Bishop. Pattern Recognition and Machine Learning (Information Science and Statistics). Springer, 2011.](https://www.amazon.com/Pattern-Recognition-Learning-Information-Statistics/dp/0387310738)
 -  [K. P. Murphy. Machine Learning: A Probabilistic Perspective (Adaptive Computation and Machine Learning). The MIT Press, 2012.](https://www.amazon.com/Machine-Learning-Probabilistic-Perspective-Computation/dp/0262018020)
 -  [deeplearningcourses.com](https://deeplearningcourses.com/)
 -  [Victor Lavrenko youtube channel](https://www.youtube.com/user/victorlavrenko/playlists)
