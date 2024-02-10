Normalization Methods
=====================

Normalization, scaling, and transformation methods are essential in data
fusion as they help to standardize and harmonize data from different
sources, making it more comparable and suitable for further analysis.
Normalization adjusts the values in the dataset to a common scale
without distorting the differences in the range of values or losing
information. Scaling modifies the range of the data to ensure that
certain features do not dominate others due to their larger scales.
Transformation methods, on the other hand, can make the data more
suitable for a specific analysis or model by adjusting the distribution
or relationship between variables. These methods collectively enhance
the quality and reliability of the fused data, leading to improved
insights and predictions.

Normalization
-------------

1. **Constant Sum Normalization**: This method normalizes data such that
   the sum of values for each observation remains constant. It ensures
   that the relative contributions of individual features to the total
   sum are compared rather than the absolute values. This normalization
   technique is beneficial for comparing samples with different total
   magnitudes but similar distributions. Mathematically, each
   observation is normalized by dividing it by the sum of its values,
   then multiplying by a constant factor to achieve the desired sum.

2. **L1 Normalization (Lasso Norm or Manhattan Norm)**: Also known as
   Lasso Norm or Manhattan Norm, this method rescales each observation
   vector by dividing each element by the L1-norm of the vector. The
   L1-norm of a vector is the sum of the absolute values of its
   components. Mathematically, for a vector x, L1 normalization is given
   by: After L1 normalization, the sum of the absolute values of the
   elements in each vector becomes 1. This method is widely used in
   machine learning tasks such as Lasso regression to encourage sparsity
   in the solution.

   .. math::


       L1−Norm=∑∣xi∣
       

3. **L2 Normalization (Ridge Norm or Euclidean Norm)**: Also known as
   Ridge Norm or Euclidean Norm, this method rescales each observation
   vector by dividing each element by the L2-norm of the vector. The
   L2-norm of a vector is the square root of the sum of the squares of
   its components. Mathematically, for a vector x, L2 normalization is
   given by: After L2 normalization, the Euclidean distance (or the
   magnitude) of each vector becomes 1. This method is widely used in
   various machine learning algorithms such as logistic regression,
   support vector machines, and neural networks.

   .. math::


       L2−Norm=∑xi2
       

4. **Max Normalization (Maximum Normalization)**: This method scales
   each feature in the dataset by dividing it by the maximum absolute
   value of that feature across all observations. It ensures that each
   feature’s values are within the range [-1, 1] or [0, 1] depending on
   whether negative values are present or not. This method is useful
   when the ranges of features in the dataset are significantly
   different, preventing certain features from dominating the learning
   process due to their larger scales. It is commonly used in neural
   networks and deep learning models as part of the data preprocessing
   step.

Scaling
-------

1. **Standardize**: This method applies the standard scaler method to
   each column of the dataframe. It transforms the data to have a mean
   of zero and a standard deviation of one. This is useful for reducing
   the effect of outliers and making the data more normally distributed.
   Mathematically, for a given feature x, standardization is given by:

   .. math::


       std(x)x−mean(x)
       

2. **Min-Max**: This method applies the min-max scaler method to each
   column of the dataframe. It transforms the data to have a minimum
   value of zero and a maximum value of one. This is useful for making
   the data more comparable and preserving the original distribution.
   Mathematically, for a given feature x, min-max scaling is given by:

   .. math::


       max(x)−min(x)x−min(x)
       

3. **Robust**: This method applies the robust scaler method to each
   column of the dataframe. It transforms the data using the median and
   the interquartile range. This is useful for reducing the effect of
   outliers and making the data more robust to noise. Mathematically,
   for a given feature x, robust scaling is given by: where IQR is the
   interquartile range.

   .. math::


       IQR(x)x−median(x)
       

4. **Pareto**: This method applies the pareto scaling method to each
   column of the dataframe. It divides each element by the square root
   of the standard deviation of the column. This is useful for making
   the data more homogeneous and reducing the effect of skewness.
   Mathematically, for a given feature x, pareto scaling is given by:

   .. math::


       std(x)x
       

Transforming
------------

1. **Cube Root Transformation**: This transformation applies the cube
   root function to each element of the data. It is useful for reducing
   the effect of extreme values or outliers.
2. **Log10 Transformation**: This transformation applies the base 10
   logarithm function to each element of the data. It is useful for
   reducing the effect of exponential growth or multiplicative factors.
3. **Natural Log Transformation**: This transformation applies the
   natural logarithm function to each element of the data. It is useful
   for reducing the effect of exponential growth or multiplicative
   factors.
4. **Log2 Transformation**: This transformation applies the base 2
   logarithm function to each element of the data. It is useful for
   reducing the effect of exponential growth or multiplicative factors.
5. **Square Root Transformation**: This transformation applies the
   square root function to each element of the data. It is useful for
   reducing the effect of square laws or quadratic growth.
6. **Power Transformer**: This transformation applies the power
   transformer method to the data. This method transforms the data to
   make it more Gaussian-like. It supports two
   methods: **``yeo-johnson``** and **``box-cox``**.
   The **``yeo-johnson``** method can handle both positive and negative
   values. The **``box-cox``** method can only handle positive values.
7. **Quantile Transformer**: This transformation applies the quantile
   transformer method to the data. This method transforms the data to
   follow a uniform or a normal distribution. It supports two output
   distributions: **``uniform``** and **``normal``**.
