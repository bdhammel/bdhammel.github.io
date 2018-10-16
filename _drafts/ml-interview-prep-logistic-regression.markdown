---
layout: post
title:  "ML Interview Prep: Logistic Regression"
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

### 1.1 High-level explanation

Logistic regression makes the explicit assumption that your data can be separated by a line or hyper-plane. Like linear regression, logistic regression is a linear model.

$$
\begin{align*}
y = f(x_i) &= \sigma(w^Tx+b) \\
&= \frac{1}{1+\exp\left\{ -(w^Tx+b) \right \}}
\end{align*}
$$

It is the internal term to $\sigma$ that gives logistic regression it's linear dependency. Although this terms looks like linear regression, $y=mw+b$, it's important to remember that what gives linear regression it's characteristics is the loss function, i.e. MSE. For logistic regression, we will define a new loss function to obtain the desired behavior.

It's first important to discuss the functionality of the sigmoid function, $\sigma$.

Consider the example below, of two Gaussian clouds located at (1,1) and (-1,-1):

It should be obvious that the dividing line, $x_2=-x_1$, separates the two classes, but lets explore this mathematically without the sigmoid.

$$
f(x_1, x_2) = x_2 + x_1 = 0
$$

taking the point $(-1,-1)$

$$
\begin{align*}
f(-1,-1) &= -2 < 0 \\
\therefore (-1,-1) &= {\rm Blue}.
\end{align*}
$$

Whereas, the point (1,1):

$$
\begin{align*}
f(1,1) &= 2 > 0 \\
\therefore (1,1) &= {\rm Red}.
\end{align*}
$$

Now, the values 2 or -2 are not particularly illuminating. So we use the sigmoid function to squash the output into a probability, such that the output $y=0={\rm Blue}$ and $y=1={\rm Red}$. To put this graphically:

### 1.2 What scenario should you use logistic regression?

When you want to have a fast, explainable model

### 1.3 What types of features does the model use?

#### 1.3.1 Linear dependence

Similar to linear regression, logistic regression is an appropriate choice when classifying data that is linearly separable. This either has to be assumed initially or constructed using sufficient domain knowledge. Consider the "donut" problem below:


This is not a linear classification problem - no straight line will separate these classes. However, similarly to the polynomial problem in my [notebook on linear regression](linear_regression.ipynb), we can add an extra dimension. such that

$$
x_3 = \sqrt{x_2^2 + x_1^2}
$$

These clusters can now be separated with a hyperplane in $x_1, x_2$-space, at $x_3 \approx 2$.


## 2. A bit more detail

### 2.1 Normalization of data 

### 2.1 Loss function

The loss function for binary logistic regression is the *cross-entropy error*

$$
\mathcal{L} = - \left \{ t\log(y) + (1-t)\log(1-y) \right \}
$$

wherein $t$ is the target class, and $y$ is the class predicted by the model.

It can be helpful to notice that only one of these terms will matter at a time. For example given a target class of $t=0$ only the second term will matter, $-\log(1-y)$. Whereas if $t=1$ only the first term will matter, $-\log(y)$.

## 3. In-depth

### 3.1 Probabilistic interpretation

#### 3.1.1 Understanding the loss function

Logistic regression operates on the fundamental assumption that the data falls into a binomial distribution, and each of the data points are independent from one-another. We describe this mathematically this using [Bayes' Law](naive_bayes.ipynb).

$$
P(y|X) = \frac{P(y) P(X|y)}{P(X)}
$$

The term in this that we have control over is the likelihood, $P(X|y)$. We want to maximize this term during training, thereby maximizing the probability that a data point falls into the correct class, $P(y|X)$.

As stated above, we define the likelihood using a binomial distribution, and assume independence.

$$
\begin{align*}
P(X|y) &= P(x_1, x_2, \cdot, x_n | y) \\
&= \prod_{n}^{i} P(x_i|y) \\
&= \prod_{n}^{i} y_i^{t_i}(1-y_i)^{t_i}
\end{align*}
$$

Taking the negative log of this, to reduce the complexity of calculating exponents, renders the cross-entropy loss function.

$$
- \log P(X|y) = \sum_i \left \{ t_i\log y_i + (1-t_i)\log (1-y_i) \right \}
$$

To summarize, logistic regression fundamentally assumes the data falls into a binomial distribution. And by **maximizing** the log of the likelihood (log-likelihood) we **minimizing** the cross-entropy.


#### 3.1.2 Understanding connection to linear regression

We have been writing the logistic regression equation as $y=\sigma(w^Tx)$ but in actuality what we are describing the is *probability* that a data point is in a class, 0 or 1:

$$
\begin{align*}
P(y=1 | x) &= \sigma(w^Tx) \\
P(y=0 | x) &= 1 - \sigma(w^Tx)
\end{align*}
$$

We can also think about this in terms of: "what are the odds that data point $x_i$ falls into class 0 or class 1", in which case we take the ratio of these probabilities.

$$
\begin{align*}
\frac{P(y=1|x)}{P(y=0|x)} &= \cfrac{\cfrac{1}{1+\exp(-w^Tx)}}{\cfrac{\exp(-w^Tx)}{1+\exp(-w^Tx)}} \\
&= \frac{1}{\exp(-w^Tx)} \\
&= \exp(w^Tx) \\
\end{align*}
$$

Taking the log of this equation we recover the a description of linear regression:

$$
\log \left ( \frac{P(y=1|x)}{P(y=0|x)} \right ) = w^Tx
$$

So, when we find the optimal decision boundary in logistic regression what we're actually doing is finding a best-fit-line that minimizes the log-odds, using linear regression.

### 3.2 Derivation of the analytic solution

A closed-form solution exists for logistic regression. It makes my head hurt. I'll try to add it later.

### 3.3 Gradient decent

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

### 3.4 Simple Implementation

```python
class LogisticRegression:

    def __init__(self, order):
        self.W = np.random.randn((order+1))

    def fit(self, X, Y, alpha=1e-1, epochs=1000):
        N, D = X.shape
        X = np.hstack((X, np.ones(shape=(N,1))))

        for _ in range(epochs):
            Y_hat = self.perdict(X)
            dL = X.T.dot(Y_hat-Y)                #   X^T (Y_hat - Y)
            self.W -= alpha*dL

    def perdict(self, X):
        return sigmoid(X.dot(self.W))

    def coeff(self):
        return self.W.flatten()


if __name__ == '__main__':

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
x1 = np.linspace(-2,2,100)
x2 = w * x1 - b
plt.figure(figsize=(5,5))
plt.scatter(X[Y==0,0], X[Y==0,1], color='b', alpha=.5)
plt.scatter(X[Y==1,0], X[Y==1,1], color='r', alpha=.5)
plt.plot(x1, x2, '--')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.show()
```
