Introduction
============

FeatureDice presents an innovative paradigm shift in cheminformatics and
bioinformatics, leveraging advanced feature fusion methodologies to
transcend traditional data silos. By ingeniously amalgamating disparate
data modalities—ranging from chemical descriptors and molecular
structures to omics data, images, and phenotype information—FeatureDice
pioneers a transformative approach. Through a rich arsenal of techniques
including PCA, ICA, IPCA, CCA, t-SNE, KPCA, RKS, SEM, Autoencoders,
Tensor Decomposition, and PLSDA, FeatureDice unlocks novel insights by
unraveling the intricate interplay between chemical and biological
entities within complex datasets. Its intuitive interface and
comprehensive toolkit empower researchers to seamlessly navigate through
the complexities of integrative analyses, heralding a new era of
interdisciplinary research and discovery.


Installation
============

To use the **FeatureDice** package, you need to install it along
with its dependencies. You can install FeatureDice and its
dependencies using the following command:

.. code:: bash

   pip install -i https://test.pypi.org/simple/ FeatureDice==1.0.1 scikit-learn torch tensorly

This command installs Featuredice along with the required
dependencies **scikit-learn**, **torch**, and **tensorly**.
Make sure to have the appropriate versions of these packages compatible
with Featuredice version 1.0.0.

Import
------

.. code:: python

   from FeatureDice.fusionData import fusionData
   from FeatureDice.plot_data import plot_model_metrics
   from FeatureDice.plot_data import plot_model_boxplot

Reading Data
------------

Define data path dictionary with name of dataset and csv file path. The
csv file should contain ID and prediction_labels column along with
features columns. If these columns named properly (ID and
prediction_labels) you can provide\ ``id_column`` and
``prediction_label_column`` argument during initialization of
``fusionData``.

.. code:: python

   data_paths = {
       "tabular1":"data/Chemberta_embeddings.csv",
       "tabular2":"data/graph_embeddings.csv",
       "tabular3":"data/mopac_descriptors.csv",
       "tabular4":"data/mordred_descriptors.csv",
       "tabular5":"data/signaturizer_descriptors.csv"
   }

loading data from csv files and creating ``fusionData`` object.

.. code:: python

   fusiondata = fusionData(data_paths = data_paths)

After loading data, you can use ``fusionData`` object to access your by
``dataframes`` dictionary in fusion data object. For example to get
tabular1 dataframe by the following code. This is important to look at
the datasets before doing any analysis.

.. code:: python

   fusiondata.dataframes['tabular1']

Data Cleaning
-------------

Common samples
~~~~~~~~~~~~~~

Keep only samples (rows) that are common across dataset. This is
important if there is difference in set of samples across datasets.

.. code:: python

   fusiondata.keep_common_samples()

Empty Features removal
~~~~~~~~~~~~~~~~~~~~~~

Features in data should be removed if there is higher percentage of
missing values. Remove columns with more than a certain percentage of
missing values from dataframes can solve this. The percentage threshold
of missing values to drop a column. ``threshold`` should be between 0
and 100. ``ShowMissingValues`` is function which prints the count of
missing values in each dataset.

.. code:: python


   fusiondata.ShowMissingValues()
   fusiondata.remove_empty_features(threshold=20)
   fusiondata.ShowMissingValues()

Imputation/Remove features
~~~~~~~~~~~~~~~~~~~~~~~~~~

Imputation of data if the data have low percentage of missing values.
``ImputeData`` is a function with takes a single argument which is
method to be used for imputation. The ``method`` can be “knn”, “mean”,
“mode”, “median”, and “interpolate”.

.. code:: python

   # Imputing values with missing values
   fusiondata.ShowMissingValues()
   fusiondata.ImputeData(method="knn")
   fusiondata.ShowMissingValues()

Data Normalization
------------------

Normalization/Standardization/Transformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data should be normalized before we proceed to fusion. There are three
functions which can be used for data normalization ``scale_data``,
``normalize_data`` and ``transform_data``. These functions takes single
argument that is type of scaling/normalization/transformation.

.. code:: python

   # Standardize data
   fusiondata.scale_data(scaling_type = 'standardize')

scaling type can be one of these ‘minmax’ , ‘minmax’ ‘robust’ or
‘pareto’

.. code:: python

   # Normalize data
   fusiondata.normalize_data(normalization_type ='constant_sum')

normalization types can be one of these ‘constant_sum’, ‘L1’ ,‘L2’ or
‘max’

.. code:: python

   # Transform data
   fusiondata.transform_df(transformation_type ='log')

transformation_type can be one of these ‘cubicroot’, ‘log10’, ‘log’,
‘log2’, ‘sqrt’, ‘powertransformer’, or ‘quantiletransformer’.

Data Fusion
-----------

Data fusion will take all the data that is normalized in previous step
and make a single fused data. This will result in a single dataframe 
``fusedData`` in the ``fusionData`` object.

::

   # fusing features in different data
   fusiondata.fuseFeatures(n_components = 10,  method="plsda")
   fused_dataframe = fusiondata.fusedData

Other methods available for fusing data are ‘pca’, ‘ica’, ‘ipca’, ‘cca’,
‘tsne’, ‘kpca’, ‘rks’, ‘SEM’, ‘autoencoder’, and ‘tensordecompose’, The
argument number of components can be provided to by ``n_components``
parameter to ``fuseFeature`` function.

Evaluation of fusion methods
----------------------------

Simple evaluation
~~~~~~~~~~~~~~~~~

Data fused by different methods can be evaluated using different machine
learning models using ``evaluate_fusion_models`` function. This function
takes normalized data, split the data into test and train dataset and
after that makes different ML model from fusion of training data and
then evaluate the models by fusion of testing data. It also takes
argument ``methods`` a list of fusion methods to evaluate. Optional
arguments is ``n_components`` the number of components use for the
fusion which is 10 by default.

.. code:: python

   # evaluate all models
   fusiondata.evaluate_fusion_models(n_components=10, methods= ['pca','cca'])

Metrics of all the models can be accessed by ``Accuracy_metrics`` in
``fusionData`` object.

.. code:: python

   ## Accuracy metrics all models
   fusiondata.Accuracy_metrics
   #top 10 models 
   top_models = fuseiondata.Accuracy_metrics.iloc[0:10,:]

Plotting the ``Accuracy_metrics`` can done by the following function.

.. code:: python

   # give top_model dataframe & output directory name for saving plots
   plot_model_metrics(top_models, save_dir = "output_plots")

Cross validation
~~~~~~~~~~~~~~~~

The function ``evaluate_fusion_model_nfold`` can do n fold cross
validation for evaluation of fusion methods, it takes Optional 
arguments ``methods`` list argument to evaluate the fusion model and
``n_components`` the number of components use for the fusion and the
number of folds to use for cross-validation.

.. code:: python

   # evaluate all models
   fusiondata.evaluate_fusion_models_nfold(n_components=10,
                                             methods= ['pca','cca'],
                                             n_folds = 10)

Metrics of all the models can be accessed by ``Accuracy_metrics`` in
``fusionData`` object.

.. code:: python

   ## Accuracy metrics all models
   fusiondata.Accuracy_metrics
   #top 10 models 
   top_models = fuseiondata.Accuracy_metrics.iloc[0:10,:]

Plotting of the ``Accuracy_metrics`` can done by the following function.

.. code:: python

   # give top model dataframe & output directory name for saving box plots
   plot_model_boxplot(top_models, save_dir ='outputs')
