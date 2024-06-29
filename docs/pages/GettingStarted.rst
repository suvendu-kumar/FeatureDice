Getting started
===============

Calculation of descriptors
--------------------------

All the descriptors can be using the following line of code

.. code:: python

   # create a directory for storing descriptors filefrom ChemicalDice 
   import smiles_preprocess, bioactivity, chemberta, Grover, ImageMol, chemical, quantum
   import os
   os.mkdir("Chemicaldice_data")
   # download prerequisites for quantum, grover and ImageMol
   quantum.get_mopac_prerequisites()
   # input file containing SMILES and labels
   input_file = "your_file_name.csv"
   # preprocessing of smiles to different formats
   smiles_preprocess.add_canonical_smiles(input_file)
   smiles_preprocess.create_mol2_files(input_file)
   smiles_preprocess.create_sdf_files(input_file)
   # calculation of all descriptors
   quantum.descriptor_calculator(input_file, output_file="Chemicaldice_data/mopac.csv")
   Grover.get_embeddings(input_file,  output_file_name="Chemicaldice_data/Grover.csv")
   ImageMol.image_to_embeddings(input_file, output_file_name="Chemicaldice_data/ImageMol.csv")
   chemberta.smiles_to_embeddings(input_file, output_file = "Chemicaldice_data/Chemberta.csv")
   bioactivity.calculate_descriptors(input_file, output_file = "Chemicaldice_data/Signaturizer.csv")
   chemical.descriptor_calculator(input_file, output_file="Chemicaldice_data/mordred.csv")

Quantum descriptors
~~~~~~~~~~~~~~~~~~~

To calculate quantum descriptors first we need to generate 3D structure
of molecule. This will save mol2 file in a directory temp_data.

.. code:: python

   smiles_preprocess.create_mol2_files(input_file = "freesolv.csv")

For quantum descriptors calculation we need MOPAC(Molecular Orbital
PACkage). The function ``quantum.get_mopac_prerequisites`` will download
the mopac executable.

.. code:: python

   quantum.get_mopac_prerequisites()

Create a directory where we can store our descriptors files.

.. code:: python

   import os
   os.mkdir("data")

Now we set for the calculation of quantum descriptors. The function
``quantum.descriptor_calculator`` takes two arguments input file path
and output file path.

.. code:: python

   quantum.descriptor_calculator(input_file = "freesolv.csv", output_file="data/mopac.csv")

Mordred Descriptors
~~~~~~~~~~~~~~~~~~~

Mordred descriptors needs sdf files to calculate descriptors. The
``smiles_preprocess`` will create sdf file from mol2 files.

.. code:: python

   smiles_preprocess.create_sdf_files(input_file = "freesolv.csv")

The function ``chemical.descriptor_calculator`` calculates modred
descriptors.

.. code:: python

   chemical.descriptor_calculator(input_file = "freesolv.csv", output_file="data/mordred.csv")

ChemBERTa embeddings
~~~~~~~~~~~~~~~~~~~~

The large language model ChemBERTa embeddings needs canonical SMILES,
the function ``smiles_preprocess.add_canonical_smiles`` adds canonical
smiles to input file.

.. code:: python

   smiles_preprocess.add_canonical_smiles(input_file = "freesolv.csv")

The function ``chemberta.smiles_to_embeddings`` generates embeddings
from the canonical SMILES.

.. code:: python

   chemberta.smiles_to_embeddings(input_file = "freesolv.csv", output_file = "data/Chemberta.csv")

Signaturizer bioactivity Signatures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The function ``bioactivity.calculate_descriptors`` generates bioactivity
signatures from canonical SMILES.

.. code:: python

   bioactivity.calculate_descriptors(input_file = "freesolv.csv", output_file = "data/Signaturizer.csv")

ImageMol embeddings
~~~~~~~~~~~~~~~~~~~

The function ``ImageMol.image_to_embeddings`` function generates 2D
images and then uses ImageMol model to gererate embeddings.

.. code:: python

   ImageMol.image_to_embeddings(input_file = "freesolv.csv", output_file_name="data/ImageMol.csv")

Grover embeddings
~~~~~~~~~~~~~~~~~

The function ``Grover.get_embeddings`` generates graph embeddings using
canonical smiles.

.. code:: python

   Grover.get_embeddings(input_file = "freesolv.csv",  output_file_name="data/Grover.csv")

Reading Data
------------

Define data path dictionary with name of dataset and csv file path. The
csv file should contain ID column along with features columns. Label
file should contain id and labels. If these columns not named id and
labels you can provide\ ``id_column`` and ``label_column`` argument
during initialization of ``fusionData``.

.. code:: python

   from ChemicalDice.fusionData import fusionData
   data_paths = {
      "Chemberta":"Chemicaldice_data/Chemberta.csv",
      "Grover":"Chemicaldice_data/Grover.csv",
      "mopac":"Chemicaldice_data/mopac.csv",
      "mordred":"Chemicaldice_data/mordred.csv",
      "Signaturizer":"Chemicaldice_data/Signaturizer.csv",
      "ImageMol": "Chemicaldice_data/ImageMol.csv"
   }

loading data from csv files and creating ``fusionData`` object.

.. code:: python

   fusiondata = fusionData(data_paths = data_paths, label_file_path="freesolv.csv", label_column="labels", id_column="id")

After loading data, you can use ``fusionData`` object to access your
data by ``dataframes`` dictionary in fusion data object. This is
important to look at the datasets before doing any analysis. For example
to get all dataframes use the following code.

.. code:: python

   fusiondata.dataframes

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
``ImputeData`` is a function which takes a single argument which is
method to be used for imputation. The ``method`` can be “knn”, “mean”,
“mode”, “median”, and “interpolate”.

.. code:: python

   # Imputing values with missing valuesfusiondata.ShowMissingValues()
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

``scaling_type`` can be one of these ‘minmax’ , ‘minmax’ ‘robust’ or
‘pareto’

.. code:: python

   # Normalize data
   fusiondata.normalize_data(normalization_type ='constant_sum')

``normalization_types`` can be one of these ‘constant_sum’, ‘L1’ ,‘L2’
or ‘max’

.. code:: python

   # Transform data
   fusiondata.transform_df(transformation_type ='log')

``transformation_type`` can be one of these ‘cubicroot’, ‘log10’, ‘log’,
‘log2’, ‘sqrt’, ‘powertransformer’, or ‘quantiletransformer’.

**Data Fusion**
---------------

Data fusion will take all the data that is normalized in previous step
and make a single fused data. The ``fuseFeatures`` method can be used to
fuse the data and save it in a csv file. The fusion methods to use given
by methods argument. Methods available for fusing data are ‘AER’, ‘pca’,
‘ica’, ‘ipca’, ‘cca’, ‘tsne’, ‘kpca’, ‘rks’, ‘SEM’, ‘autoencoder’, and
‘tensordecompose’. The components to keep from different data can be
provided by ``n_components``\ aggumrent. Reduced dimensions to use for
Autoencoder Reconstruction can be provided by ``AER_dim`` argument.
Argument ``save_dir`` can be used to specify directory for saving the
fused data.

.. code:: python

   # fusing features in different data
   fusiondata.fuseFeatures(n_components=10,
                     methods= ['pca','tensordecompose','plsda','AER'],
                     AER_dim= [4096,8192],
                     save_dir = "ChemicalDice_fusedData")

**Evaluation of Fusion Methods**
--------------------------------

**Cross Validation**
~~~~~~~~~~~~~~~~~~~~

The method ``evaluate_fusion_model_nfold`` can perform n-fold cross
validation for the evaluation of fusion methods. It takes
the ``nfold`` argument for the number of folds to use for
cross-validation, the ``task_type`` argument for classification or
regression problems, and the ``fused_data_path`` directory that contains
the fused data as CSV files generated in the feature fusion step.

.. code:: python

   # Evaluate all models using 10-fold cross-validation for regression tasks
   fusiondata.evaluate_fusion_models_nfold(folds=10,
                                           task_type="regression",
                                           fused_data_path="ChemicalDice_fusedData")

Metrics for all the models can be accessed using
the ``get_accuracy_metrics`` method, which takes
the ``result_dir`` argument for the directory containing CSV files from
n-fold cross-validation. The outputs are
dataframes ``mean_accuracy_metrics`` and ``accuracy_metrics``, along
with boxplots for the top models for each fusion method saved
in ``result_dir``.

::

   ## Accuracy metrics for all models
   mean_accuracy_metrics, accuracy_metrics = fusiondata.get_accuracy_metrics(result_dir='10_fold_CV_results')

**Scaffold Splitting**
~~~~~~~~~~~~~~~~~~~~~~

The method ``evaluate_fusion_models_scaffold_split`` can perform
scaffold splitting for the evaluation of fusion methods. It takes the
arguments ``split_type`` (“random” for random scaffold splitting,
“balanced” for balanced scaffold splitting, and “simple” for just
scaffold splitting), ``task_type`` for “classification” or “regression”
problems, and the ``fused_data_path`` directory that contains the fused
data as CSV files generated in the feature fusion step.

.. code:: python

   # Evaluate all models using random scaffold splitting for regression tasks
   fusiondata.evaluate_fusion_models_scaffold_split(split_type="random",
                                                    task_type="regression",
                                                    fused_data_path="ChemicalDice_fusedData")

Metrics for all the models can be accessed using
the ``get_accuracy_metrics`` method, which takes
the ``result_dir`` argument for the directory containing CSV files from
scaffold splitting. The outputs are
dataframes ``test_metrics``, ``train_metrics``, and ``val_metrics``,
along with bar plots for the top models for each fusion method saved
in ``result_dir``.

.. code:: python

   ## Accuracy metrics for all models
   test_metrics, train_metrics, val_metrics = fusiondata.get_accuracy_metrics(result_dir='scaffold_split_results')
