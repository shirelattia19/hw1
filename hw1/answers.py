r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**
1. **False:** the in-sample is the error rate we get on the same data set we used to build our predictor and not on a 
new dataset such as the test set.
2. **False:** the proportion we chose for the train/test sets has an impact on the goodness of our model. if the train set 
is too small, we could get an over-fitted model with lack in generalization. So the way we split the data and the ratio 
has a direct impact on our model and bad split dataset could result to a less performant model.  
3. **True:** the test set is our way to test our model. the test set cannot be used during any part of the process until we 
get to the point we want to test the model. we cannot extract any information from it. 
4. **True:** indeed, after each fold when performing cross-validation, we use the validation-set performance as a proxy 
for the model's generalization error. but at the end, this is the test set that allows us to compare the different 
models on data that were not used in any part of the training/hyperparameter selection process. 
**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**

The approach of this friend isn't justified. As we explained before, the validation set is the one dedicated to test 
the hyper-parameters. The test should not be used in any step of the model implementation process and is only used at 
the end to test its performance.

**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**

Increasing k can lead to improved generalization until a certain limit. Generalization by giving less power to outliers
is very important and by increasing the k, outliers has less impact. On the other hand, a too big k lead to a too 
much generalized model, means the model cannot make good decision based on the nearest neighbours. A good example to 
illustrate this situation is a KNN with unweighted dataset: when the dataset is unweighted means there are much more 
samples labeled with a specific category, the knn model with a too big K is skewed since the recurrent label will always
have more impact on the decision.
Each dataset has his own optimized k: k is an hyperparameter and should be found using cross-validation/tuning.

**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**
Using k-fold CV, as detailed above, is better than:

1. because using the same dataset for training and validating could lead to an non-generalized and over-fitted model. 
We need to create a model that can generalize also on unseen data.

2. as explained before, the dataset cannot be used on any step of the model implementation and should be used only to 
get model's results when the model is already trained and validated. The hyper-parameters are evaluated with an other 
unseen dataset: the validation set.

**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**

$$
L(\mat{W}) =
\frac{1}{N} \sum_{i=1}^{N} \sum_{j \neq y_i} \max\left(0, \Delta+ \vectr{w_j} \vec{x_i} - \vectr{w_{y_i}} \vec{x_i}\right)
+
\frac{\lambda}{2} \norm{\mat{W}}^2
$$

Since ${\lambda}$ and $\Delta$ are both balancing the equation, we can fix an arbitrary $\Delta$ and define ${\lambda}$ 
as hyperparameter. It would mean that for each $\Delta$ there is an optimum ${\lambda}$ for the model. 

**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**
 
1. When looking at the weights as images, it looks like the model tends to learn the shape structure of the digits by 
producing heatmaps. as we can see, some part are very dark and i presume that the pixels of the heat map are 
negative unlike the illuminated areas that are positive pixels. The pixels/heat maps are the factor that allows the 
model to take a decision. 
As we can see in the visualization of some test-set examples and the model's predictions for them, the model 
gets particularly wrong answers with 6 and 9 (predicts 6/9 when it is not and wrong predictions when expected 6/9).
Indeed, the 6 and 9 structure are badly defined and the illuminated areas could be confounded with illuminated areas of 
other digits. 

2. The Knn is different from the SVM in the fact that the SVM actually update weights according to the loss. 
**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**

1. When looking at the loss graph, when can see the function converges and stagnates after a certain number of epochs (~20).
If the learning rate was too low or too high, the function could not converge. thus, we can assume the learning rate is 
good. If the learning rate was to low, the graph was continually decreasing very very slowly and never converge. If the 
learning rate was to high, the graph would diverge (with oscillations).

2. We can say that the model is Slightly over-fitted to the training set. Indeed, the graph shows that the training 
accuracy keeps increasing when the validation accuracy almost converges. The difference between them get larger and 
larger means that the model is over-fitting. (It is not Highly overfitted since the accuracy of the validation is not 
yet decreasing).

**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**

The ideal pattern for the residual plot should be the concentration of the points around the zero-line and into the 
margin lines when the data form a +/- linear shape. 
The shape we got before was linear than the last one and the points were more dispersed. Finally, on the plot after CV, 
we can see that the  points of the test are very centered into the margin and the blue outlier from the train set not 
really impacted the final result means we didn't get an over-fitted model.
**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q2 = r"""
**

1. The model is still linear since the points are separated by a linear classifier/hyperplane. 

2. In the theory, if we knows exactly which function is mapped,  we could compute the value and then fit the function but 
there exist many complicated function and since all the purpose is to predict a function we don't know, this approach 
isn't exploitable for any non linear function.

3. Yes it will still be an hyperplane and by feature mapping we will get more features and we could be able to better 
separate the different points and then get better results.
**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q3 = r"""
**

1. The numpy logspace() function is used to create an array of equally spaced values between two numbers on the 
logarithmic scale contrary to the  numpy linspace() function which creates an array of equally spaced values on the 
linear scale.
The $\lambda$ hyperparameter has a multiplication on the learning process then for this kind of hyperparameters, 
it is better to use np.logspace than np.linspace.

2. On the hyperparameters CV, the the data was fitted:

len(degree_range) * len(lambda_range) * k_folds = 20 * 3 * 3 = 180

**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
