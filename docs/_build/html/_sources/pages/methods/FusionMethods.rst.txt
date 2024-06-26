Fusion Methods
==============

Fusion methods play a crucial role in data fusion, a process that
integrates multiple data sources to produce more consistent,
accurate, and useful information than that provided by any
individual data source. These methods, which include various
dimensionality reduction techniques and statistical models, help to
uncover the underlying structure and relationships in the data,
reduce noise and redundancy, and transform the data into a suitable
form for further analysis or decision-making. By effectively
combining data from different sources, fusion methods can enhance
the quality and reliability of the resulting information, leading to
improved insights and predictions. This is particularly important in
fields such as machine learning, signal processing, and data mining,
where the ability to extract meaningful information from large and
complex datasets is key.


Linear methods
--------------

1. **PCA (Principal Component Analysis)**: PCA is a linear
   dimensionality reduction technique that projects the data onto a
   lower-dimensional subspace that maximizes the variance. It does so by
   calculating the eigenvectors of the covariance matrix of the data,
   which correspond to the directions (or principal components) in which
   the data varies the most.
2. **ICA (Independent Component Analysis)**: ICA is a linear
   dimensionality reduction technique that separates the data into
   independent sources based on the assumption of statistical
   independence. It is often used in signal processing to separate mixed
   signals into their original sources.
3. **IPCA (Incremental Principal Component Analysis)**: IPCA is a
   variant of PCA that allows for online updates of the components
   without requiring access to the entire dataset. In traditional PCA, 
   the entire dataset must be available and fit into memory, which might 
   not be feasible with large datasets. IPCA, on the other hand, updates 
   the components incrementally as new data comes in, without needing to 
   access the entire dataset. This makes it a more memory-efficient 
   alternative to PCA when dealing with large datasets. This makes it
   suitable for large datasets or streaming data.
4. **CCA (Canonical Correlation Analysis)**: CCA is a linear
   dimensionality reduction technique that finds the linear combinations
   of two sets of variables that are maximally correlated with each
   other. It is often used in multivariate data analysis to understand
   the relationships between two sets of variables.

Non linear methods
------------------

1. **t-SNE (t-distributed Stochastic Neighbor Embedding)**: t-SNE is a
   non-linear dimensionality reduction technique that preserves the
   local structure of the data by embedding it in a lower-dimensional
   space with a probability distribution that matches the pairwise
   distances in the original space.
2. **KPCA (Kernel Principal Component Analysis)**: KPCA is a non-linear
   extension of PCA that uses a kernel function to map the data into a
   higher-dimensional feature space where it is linearly separable. This
   allows KPCA to capture complex, non-linear relationships in the data.
3. **RKS (Random Kitchen Sinks)**: RKS is a non-linear dimensionality
   reduction technique that uses random projections to approximate a
   kernel function and map the data into a lower-dimensional feature
   space. It is often used in machine learning to speed up kernel
   methods.
4. **SEM (Structural Equation Modeling)**: SEM is a statistical
   technique that tests the causal relationships between multiple
   variables using a combination of regression and factor analysis. It
   is often used in social sciences to test theoretical models.
5. **Isomap (Isometric Mapping)**: Isomap is a non-linear dimensionality
   reduction technique that preserves the global structure of the data
   by embedding it in a lower-dimensional space with a geodesic distance
   that approximates the shortest path between points in the original
   space.
6. **LLE (Locally Linear Embedding)**: LLE is a non-linear
   dimensionality reduction technique that preserves the local structure
   of the data by reconstructing each point as a linear combination of
   its neighbors and embedding it in a lower-dimensional space that
   minimizes the reconstruction error.
7. **Autoencoder**: An autoencoder is a type of neural network that
   learns to encode the input data into a lower-dimensional latent
   representation and decode it back to the original data, minimizing
   the reconstruction error. It is often used in unsupervised learning
   for dimensionality reduction or feature extraction.
8. **PLSDA (Partial Least Squares Discriminant Analysis)**: PLSDA is a
   supervised dimensionality reduction technique that finds the linear
   combinations of the features that best explain both the variance and
   the correlation with the target variable. It is often used in
   chemometrics for classification or regression problems.
9. **Tensor Decomposition**: Tensor Decomposition is a technique that
   decomposes a multi-dimensional array (tensor) into a set of
   lower-dimensional arrays (factors) that capture the latent structure
   and interactions of the data. It is often used in machine learning to
   model high-dimensional data with complex interactions.
