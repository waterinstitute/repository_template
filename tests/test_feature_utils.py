# Databricks notebook source
# MAGIC %md 
# MAGIC #Tests for feature utils function 
# MAGIC 
# MAGIC Testing within datbricks using the nutter package: https://github.com/microsoft/nutter

# COMMAND ----------

!pip install nutter

# COMMAND ----------

# MAGIC %run ../src/features/utils

# COMMAND ----------

from runtime.nutterfixture import NutterFixture, tag

is_job = dbutils.notebook.entry_point.getDbutils().notebook().getContext().currentRunId().isDefined()

# COMMAND ----------

class TestMountStorage(NutterFixture):
  
  def assertion_mount_valid_account(self):
      """ test we can mount a valid blob storage account """
      mnt = mount_storage_account(
        "flood-ml-data", 
        "insightmlstorage", 
        "insightml", 
        "insightmlstorage"
      )
      assert mnt == "/mnt/flood-ml-data"

result = TestMountStorage().execute_tests()
print(result.to_string())

if is_job:
  result.exit(dbutils)

# COMMAND ----------

import numpy as np
import pandas as pd

class TestCreateLabel(NutterFixture):
  def before_all(self):
    """ set up dataframe"""
    data = np.array([0.1, 0.005, 0.2, 0.05, 0.21])
    pd_df = pd.DataFrame(data, columns=['heatmap_pluvial'])
    self.df = spark.createDataFrame(pd_df)
      
  def assertion_create_label(self):
    """ test label is created as expected """
    df = self.df
    df_w_label = create_label(df, 10)
    labels = np.array(df_w_label.select('label').collect()).flatten()
    assert all(labels == ["0", "0", "1", "0", "1"])
      
  def assertion_create_label_diff_col(self):
    """ test label is created as expected with column name is diff """
    df = self.df.select(col("heatmap_pluvial").alias("test"))
    df_w_label = create_label(df, 10, "test")
    labels = np.array(df_w_label.select('label').collect()).flatten()
    assert all(labels == ["0", "0", "1", "0", "1"])
      
result = TestCreateLabel().execute_tests()
print(result.to_string())

if is_job:
  result.exit(dbutils)

# COMMAND ----------

import itertools 
import numpy as np 

class TestCreateInteractions(NutterFixture):
  def before_all(self):
    """ set up dataframe"""
    data = {
      "a" : np.array([0.1, 0.2, 0.3, 0.4, 0.5]), 
      "b" : np.array([0,   1,   2,   3,   4])
    }
    pd_df = pd.DataFrame(data)
    self.df = spark.createDataFrame(pd_df)
      
  def assertion_create_interactions(self):
    df_result = create_interactions(self.df, ["a", "b"])
    interaction_result = np.array(df_result.select('a_b').collect()).flatten()
    expected_result = np.array([0.,0.2,0.6,1.2,2.])
    np.testing.assert_almost_equal(expected_result, interaction_result)

result = TestCreateInteractions().execute_tests()
print(result.to_string())

if is_job:
  result.exit(dbutils)

# COMMAND ----------

import itertools 
import numpy as np 

class TestCreateSquares(NutterFixture):
  def before_all(self):
    """ set up dataframe"""
    data = {
      "a" : np.array([0.1, 0.2, 0.3, 0.4, 0.5]), 
      "b" : np.array([0,   1,   2,   3,   4])
    }
    pd_df = pd.DataFrame(data)
    self.df = spark.createDataFrame(pd_df)
      
  def assertion_create_interactions(self):
    df_result = create_squares(self.df, ["a", "b"])
    a2_result = np.array(df_result.select('a_2').collect()).flatten()
    b2_result = np.array(df_result.select('b_2').collect()).flatten()
    expected_a2_result = np.array([0.01, 0.04, 0.09, 0.16, 0.25])
    expected_b2_result = np.array([0,1,4,9,16])
    np.testing.assert_almost_equal(expected_a2_result, a2_result)
    np.testing.assert_almost_equal(expected_b2_result, b2_result)

result = TestCreateSquares().execute_tests()
print(result.to_string())

if is_job:
  result.exit(dbutils)
