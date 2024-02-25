# ML Regression Models

## Simple Linear Regression

- The Simple Linear Regression (SLR) model is used to model the relationship between a single independent variable $x$ and a dependent variable $y$ by fitting a linear equation to the observed data.

- A ***training*** set of data is used and from that the learning algorithm will produce a function $f$. This function can then be used to predict an output $\hat{y}$ for values of $x$ which are not a part of the training set.


### The linear function $f$

$$ f_{w, b} = w*x + b $$

### The cost function $J$

The cost function measures the ***error*** between the predicted values $\hat{y}$ and the actual values of $y$. The goal Linear Regression is to minimise the cost function.

$$ J_{w, b} = \frac{1}{2m} \sum_{i=1}^{m}(\hat{y} - y)^2  $$

*Where* $m$ *is the number of training examples*

### Gradient Descent

Now that we have the cost function, we find the values of $w, b$ such that our cost is minimised. Here we use the Gradient Descent Algorithm:

**Repeat until convergence:**

$$
\begin{aligned}
w &:= w - \alpha \frac{d}{dw} J(w, b) \\
b &:= b - \alpha \frac{d}{db} J(w, b)
\end{aligned}
$$

where...

$$
\begin{aligned}
\frac{d}{dw} J(w, b) &= \frac{1}{m} \sum_{i=1}^m (f_{w, b}(x^{(i)}) - y^{(i)})x^{(i)} \\
\frac{d}{dw} J(w, b) &= \frac{1}{m} \sum_{i=1}^m (f_{w, b}(x^{(i)}) - y^{(i)})
\end{aligned}
$$


*Where* $\alpha$ *is the learning rate (controls how much parameters* $w,b$ *are adjusted with each step*

*Note:* $w$ *and* $b$ *must be updated simultaneously*

After applying the Gradient Descent algorithm we now have new values for the parameters $w$ and $b$. We can now plot a line of best fit using these parameters and predict our own outputs for new inputs.

## Swedish Auto Insurance Data Set

This dataset contains the following data:
- x = number of claims
- y = total payment for all the claims in thousands of Swedish Kronor

Running `swedish_auto_insurance.py` gives us the following plots:

