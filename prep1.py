# Use the Azure Machine Learning data preparation package
from azureml.dataprep import package
import os
cwd = os.getcwd()
print(cwd)

# Use the Azure Machine Learning data collector to log various metrics
# from azureml.logging import get_azureml_logger
# logger = get_azureml_logger()

# This call will load the referenced package and return a DataFrame.
# If run in a PySpark environment, this call returns a
# Spark DataFrame. If not, it will return a Pandas DataFrame.
df = package.run('prep1.dprep', dataflow_idx=0,spark=False)

# Remove this line and add code that uses the DataFrame
print(df.head(10))

