---
layout: post
title:  "ML Interview Prep: K-means"
description: This post is designed to be a quick overview of the K-means machine learning model
image: /assets/ml-naive-bayes/two_prob_map.png
tags: [machine learning, interviews, data science]
---

These posts are designed to be a quick overview of each machine learning model. The target audience is people with some ML background who want a quick reference or refresher. The following questions are compiled out of common things brought up in interviews.

1. Top-level
	1. What is the high-level version, explain in layman's terms
	2. What scenario should you use it in (classification vs regression, noisy data vs clean data)?
	3. What assumptions does the model make about the data?
	4. When does the model break/fail (adv & dis-advantages)? What are alternatives?

2. A bit more detail  
	1. How do you normalize the data for the model?
	2. What's the complexity?

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

K means is an unsupervised clustering algorithm which will attempt to find the center of "like features" based on the summed euclidean distance of all data given. This algorithm is greedy in the sense that its decision to group things is absolute. Consider the example below of two Gaussian clouds. We assume that a K means algorithm found the center of the clusters to be $(-2,-2)={\rm Blue}$, and $(2,2)={\rm Red}$. We then get a new data point at location $(0,0.1)$, shown in black, and we ask "What cluster does this point belong to?"

To find which cluster this new data point would belong to, we find the euclidean distance to the centers.

$$
\begin{align*}
\Delta_b &=  \sqrt { \sum_i \left( c^b_i - x_i \right)^2 } \\
&= 2.9 \\ \\[5mm]
\Delta_r &=  \sqrt { \sum_i \left( c^r_i - x_i \right)^2 } \\ 
&= 2.76
\end{align*}
$$

Because $\Delta_r < \Delta_b$ we say (with 100% confidence) that the new data point belongs to the ${\rm Red}$ cluster.

### 1.2 What scenario should you use K means?

K-means is a general-purpose clustering approach (the fastest). It can be used when even-sized spherical clusters can be assumed. It preforms best when there are only a few clusters.

In industry, k-means is used in: User segmentation (based on behaviors, like purchase history, interests). Grouping inventory based on sales activity. Detecting bots from humans. Seeing if a tracked point is changing groups over time. Detect activity types in motion sensors

### 1.3 How does this deal with outliers? Skewed data?

It is sensitive to outliers. (k-medians will be less sensitive.)

With squared error metric, outliers will influence cluster formation.  Resulting clusters may not be truly representative of what they should be, SSE metric is higher as well.  Depending on application, it can be useful to discover and remove outliers beforehand.  Alternatively, outliers can be removed during preprocessing.  Keep track of SSE contributed by each point, and eliminate those points with unusually high contributions, especially over multiple runs.  

During post-processing, can also eliminate small clusters since they can frequently represent groups of outliers.

### 1.4 What types of features does the model use?

The model makes the assumption that the data is clustered into spherical groups: Text, and continuous data. It doesn't work with categorical... but there are some ways to do it)

- Assumes Numeric data. Doesn't work with Categorical (Cardinal or Ordinal) data. 
- To avoid this, feature engineering can be done: 
- 1. Ordinal data can be replaced with arithmetic sequence of appropriate difference. Say, small/large can be replaced by 5/10. Because of unit normalization, difference doesn’t really matter.
- 2. For cardinal data the values can be converted to binary values. However, this increases the dimensions (leading to the curse of dimensionality). So another approach is to use a different distance for categorical data; like Gower distance in R, Hamming distance in k-modes. (Gower distance is a dissimilarity measure.)

### 1.4 When does  the model break?
The model will break when the data does not fall into spherical groups. For example, consider two elliptical cluster:

The "X-marker" marks the predicted centers of the ${\rm Red}$ and ${\rm Blue}$ clusters. We can see it's converged to an incorrect location, despite there being an obvious separation of the data.  

### 1.5 What to use when it breaks? Whats a good back up?

It depends on how the method breaks:

Sklearn has a good table on the use cases and fall-back models for clustering:

http://scikit-learn.org/stable/modules/clustering.html

DB scan seems pretty cool...

One fix that is worth talking about in more detail is to use "soft kmeans," also called "Fuzzy Clustering". Soft kmeans will take into account the distance of a point from the centroid, thereby assigning a "confidence" to the value belonging to that cluster. This can be important for handling outliers, as in the first example. Soft kmeans is described more below.

## 2. A bit more detail

### 2.1 Normalize  of data

Because we're concerned with euclidean distance,

$$
\Delta = \sqrt{ (x-\mu_x)^2 + (y-\mu_y)^2 },
$$

if the coordinates have a different scaling, then k means will preferentially cluster the point on the axis with shorter distance. Therefore should scale the features to, make 0 mean and unit variance (from -1 to 1, along each dimension).

In the end this depends on the data: latitude and longitude should not scaled, because this will cause distortions.


### 2.2 How to initialize parameters?

The parameter to be initialized is the k (number of clusters).

Choosing the k centroids:
- The most basic method is to choose k random samples from the dataset. [According to Andrew Ng, run k means multiple times, and choose the one with the lowest cost.]
- - However, to avoid local minimum, one method is to use the k-means++ initialization scheme. This initializes the centroids to be (generally) distant from each other, leading to provably better results than random initialization. This is implemented in scikit-learn.
### 2.3 What's the loss function used?

Soft and hard Kmeans algorithms implement coordinate descent (not gradient descent), such that the center of the centroid is updated based on the mean (or weighted mean) of the data points assigned to that cluster. Because both of these functions are monotonically decreasing, kmeans is guaranteed to converge; however, it will probably converge to a local minima.

#### 2.3.1 Hard Kmeans

The loss function for hard kmeans is the 

$$
\mu_i \leftarrow \cfrac{\sum_i x_i}{\sum_i i}
$$


#### 2.3.1 Soft Kmeans

In soft k means, the loss function is weighted by a probability that the data point belongs to that cluster, denoted the 'responsibility', $r$.

$$
\mu_i \leftarrow \cfrac{\sum_n r_i^n x^n}{\sum_n r_i^n }
$$

with

$$
r_{i}^n = \cfrac{ \exp \left \{- \beta \delta(\mu_i, x^n) \right \}}{\sum_j \exp \left \{- \beta \delta(\mu_j, x^n) \right \} }
$$

$\delta$ being the euclidean distance between a point x_i and the closest centroid, $\mu$, and $\beta$ is a weighting constant (I think usually it's just set to 1) **Look this up**.

This method alleviates the ambiguity of a point belonging to a certain class, as in the first example of a point landing in the middle of two clusters.

#### 2.3.3 Visulization of Soft and Hard Kmeans

Consider the two gaussian clouds in 1D below

### 2.4 What's the complexity? Does it scale?

The complexity of the problem is of the order 

$$
\mathcal{O}(I\cdot N\cdot D \cdot C)
$$

wherein $I$ is iterations; $N$ is number of data points; $D$ is dimensions; and $C$ is the number of clusters. Some short cuts can be taken, such as only taken a small sample of the total number of data points (mini-batch). K-means will still struggle with large datasets, but it does better than the other options.



## 3. In-depth

### 3.1 Derive the math

The math behind kmeans is quite simple, and there's nothing really to derive, so we'll just do a simple step-by-step example. Say we have data that we're trying to cluster, we drop two centroids onto the data at random locations:

To converge on the clusters we will do the following steps:

 1. Calculate the distance from every single data point to centroid 1 and centroid 2
 2. Assign each data point to the centroid that it is closer to
 3. For every single data point assigned to a cluster, take the mean value $\frac{1}{N}\sum_i (x_i, y_i)$
 4. Move the centroid to this coordinate
 5. Repeat

### 3.2 Simple implementation


## 4. More on training the model

### 4.1 How to deal with imbalanced data?

k-means does not care about cluster cardinalities

### 4.2 How well does it generalize to unseen data (over-fitting vs under-fitting)?

overfits if u set k=n, underfits if k = 1

### 4.3 What if you have MANY more features than sample points? Vice versa? (A variation of the above over/under fitting)

Curse of dimensionality. In high dimensions the (euclidean) distances are similar, making them pretty much useless.

### 4.4 How can you validate the model?

#### 4.4.1 Purity

External validation method:

$$
P = \frac{1}{N} \sum_k^K \max_{j=1...K} \left | c_k \cap  t_j \right |
$$

#### 4.4.2 Davis-Bouldin Index

Internal validation method:

$$
DBI = \frac{1}{K}\sum_i^K \max_{j \neq k} \left [ \frac{\sigma_j + \sigma_k}{\delta(c_j, c_k)} \right ]
$$

with $\sigma$ = average distance between each data point and the cluster center, i.e.

$$
\sigma_j = \frac{1}{N}\sum_i \delta(c_j, x_i)
$$

### 4.5  Does the model emphasize Type 1 or Type 2 errors?

For anomaly detection, its shown to have LOW type 1 errors (false positives). We can use precision and recall (though not widely reused), but of course we need to know the labels

