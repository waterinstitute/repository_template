# Databricks notebook source
def mount_storage_account(container, account, secret_scope, secret_key):
  # mount storage account 
  mnt = "/mnt/{}".format(container)

  if not any(mount.mountPoint == mnt for mount in dbutils.fs.mounts()):  
    dbutils.fs.mount(
      source = "wasbs://{}@{}.blob.core.windows.net".format(container, account),
      mount_point = mnt,
      extra_configs = {
        "fs.azure.account.key.{}.blob.core.windows.net".format(account): 
        dbutils.secrets.get(scope = secret_scope, key = secret_key)
      })
    
  return mnt

# COMMAND ----------

def read_delta_file(full_delta_path, version_to_load="latest"):
  if version_to_load == "latest":
    delta_table = DeltaTable.forPath(spark, full_delta_path)
    version_to_load  = delta_table.history(1).select("version").collect()[0].version 
  
  return (spark.read.format("delta").option("versionAsOf", version_to_load).load(full_delta_path), version_to_load)

# COMMAND ----------

def create_label(df, return_interval, heatmap_col="heatmap_pluvial"):
  """given a dataframe with heatmap column, create a label column """
  df_with_label = df \
    .withColumn("label", when(df[heatmap_col] > 1/return_interval, "1").otherwise("0")) 
  return df_with_label

# COMMAND ----------

def multiply_cols(col1, col2):
  return col1 * col2
  
multiply_cols = udf(multiply_cols)

def create_interactions(df, cols_to_interact):
  for combo in itertools.combinations(cols_to_interact, 2):
    new_feat = combo[0] + "_" + combo[1]
    df = df.withColumn(new_feat, multiply_cols(combo[0], combo[1]))
    df = df.withColumn(new_feat, col(new_feat).cast(FloatType()))
  
  return df

def create_squares(df, cols_to_square):
  for c in cols_to_square:
    new_feat = c + "_2"
    df = df.withColumn(new_feat, multiply_cols(c, c))
    df = df.withColumn(new_feat, col(new_feat).cast(FloatType()))
  
  return df

# COMMAND ----------

def build_features(df_raw, layers, heatmap_col="heatmap_pluvial", return_with_null=False):
  """ Build features to use for training and prediction. """
  df = expand_df(df_raw, layers)
  df_notnull = df.na.drop(how="all", subset=layers)
  df_null = df.exceptAll(df_notnull)
  layers = [c for c in layers if c != heatmap_col]
  df_notnull = create_interactions(df_notnull, layers)
  df_notnull = create_squares(df_notnull, layers)  

  if return_with_null:
    for c in df_notnull.columns:
      if c not in df_null.columns:
        df_null = df_null.withColumn(c, lit(None).cast(FloatType()))
    df_full = df_notnull.union(df_null)
    return df_full
  else:
    return df_notnull

# COMMAND ----------

def get_dims(df):
  """Returns dimensions of 2d array of each HUC"""
  rows = df.where(df.idx_1 == 0).groupBy("huc").agg(sum(df.dims[0]).alias("rows"))
  cols = df.where(df.idx_0 == 0).groupBy("huc").agg(sum(df.dims[1]).alias("cols"))
  total_dims = rows.join(cols, on=['huc'], how='left')
  
  return total_dims

def expand_df(df, layers, return_dims=False):
  """ Takes in raw dataframe, split into tiles and a list of layers to 
  expand and returns an expanded, long format dataframe.
  """
  # collect tile sizes used for each huc
  tile_sizes = df \
    .where((df.idx_1 == 0) & (df.idx_0 == 0)) \
    .withColumn("tile_rows", df.dims[0]) \
    .withColumn("tile_cols", df.dims[1]) \
    .select("huc", "tile_rows", "tile_cols")
    
  # collect & join total dimensions for each HUC
  total_dims = get_dims(df)
  
  all_dims = tile_sizes.join(total_dims, on=['huc'], how='left')
  df = df.join(broadcast(all_dims), on=['huc'], how='left')
  
  # calculate row0 and col0 to represent the number of cols and cols to upper left corner of each tile 
  # and calculate row_i and col_i to represent the number of rows and cols within the tileto each point
  df_exploded = df.select("huc", "dims", "rows", "cols", 
            (df.idx_0*df.tile_rows).alias('row0'),
            (df.idx_1*df.tile_cols).alias('col0'),
            posexplode(arrays_zip(*layers)).alias("local_pos","x")  
           ) \
    .withColumn('row_i',floor(col('local_pos') / col('dims')[1])) \
    .withColumn('col_i',col('local_pos') % col('dims')[1] )\
    .withColumn('pos', (col('row0')+col('row_i'))*col("cols")+col('col0')+col('col_i')) 
  
  # unnest exploded columns
  df_clean = df_exploded \
    .select("*", "x.*") \
    .drop("x", "dims", "local_pos", "row0", "col0", "row_i", "col_i")
  
  return df_clean

# COMMAND ----------

def get_layer_array(df, huc_dims, huc, layer_name):
  """ Plots an layer from a given dataframe in long format """
  
  rows = [row["rows"] for row in (huc_dims.where(huc_dims.huc == huc).select("rows").collect())][0]
  cols = [row["cols"] for row in (huc_dims.where(huc_dims.huc == huc).select("cols").collect())][0]
  df_filt = df.where(df.huc == huc)
  df_order = df_filt.select(layer_name, "pos").orderBy("pos")
  arr_flat = [row[layer_name] for row in df_order.select(layer_name).collect()]
  arr_flat = np.array(arr_flat, dtype = np.float64)
  arr=arr_flat.reshape(rows, cols)
  return arr 

# COMMAND ----------

# Mark's GEV functions for estimating return interval from runoff

def Norm_Constant_GEV(x: np.ndarray, PMP: float) -> float:
    """
    Constant for distribution truncation at the PMP value.
    PMP is the Probable Maximum Precipitation
    x is a list of 3 elements:
    - x[0]: loc
    - x[1]: scale
    - x[2]: shape parameter c
    """ 
    return 1.0/stats.genextreme.cdf(PMP, x[2], x[0], x[1])


def Norm_Constant_LN(SD: float, mu: float, PMP: float) -> float:
    """
    Constant for distribution truncation at the PMP value. 
    """ 
    return 1.0/stats.lognorm.cdf(PMP, SD, scale = np.exp(mu))


def PDF_GEV(R: np.ndarray, x: np.ndarray, PMP: float) -> np.ndarray:
    """
    Computes the pdf of the GEV at runoff value R given truncation value PMP (Probable Maximum Precipitation)
    x is a list of 3 elements:
    - x[0]: loc
    - x[1]: scale
    - x[2]: shape parameter c
    """
    return Norm_Constant_GEV(x, PMP)*stats.genextreme.pdf(R, x[2], x[0], x[1])


def CDF_GEV(R: np.ndarray, x: np.ndarray, PMP: float) -> float:
    """
    Computes the cdf of the GEV at runoff value R given truncation value PMP (Probable Maximum Precipitation)
    x is a list of 3 elements:
    - x[0]: loc
    - x[1]: scale
    - x[2]: shape parameter c
    """
    return Norm_Constant_GEV(x, PMP)*stats.genextreme.cdf(R, x[2], x[0], x[1])


def PPF_GEV(P: np.ndarray, x: np.ndarray, PMP: float) -> np.ndarray:
    """
    Computes the ppf (percent point function; inverse of cdf) of the GEV at runoff value R given truncation value PMP (Probable Maximum Precipitation)
    x is a list of 3 elements:
    - x[0]: loc
    - x[1]: scale
    - x[2]: shape parameter c
    """
    return stats.genextreme.ppf(P/Norm_Constant_GEV(x, PMP), x[2], x[0], x[1])


def GEV_Parameters(df: pd.DataFrame, GEV_Parameters: np.ndarray, bounds: tuple, huc: str, PMP: float, Atom_Prob: float) -> pd.DataFrame:
    """Function defines an objective function for finding the GEV parameters and then determines the best GEV parameters 
       that minimize the difference between the GEV and comparison data.
    """    
    def objective_func_GEV(x: np.ndarray) -> float: 
        """Calculates the sum of the squared residuals between the return interval and return interval calculated from 
           the GEV CDF with the differences normalized by the return interval. 
        """ 
        return np.array([np.square((RI-1/(1-Atom_Prob-(1-Atom_Prob)*CDF_GEV(row[huc], x, PMP)))/RI) for RI, row in df.iterrows()]).sum()
    
    solution = minimize(objective_func_GEV, GEV_Parameters, method='SLSQP', bounds=bounds, options={'disp': False})
    df_GEV_parameters = pd.DataFrame(data=solution.x, index=["mu", "sigma", "xi"], columns=["{}".format(huc)])
    
    return df_GEV_parameters


def GEV_parameters_Fit(raw_precip: pd.DataFrame, huc: str, PMP: float, Atom_Prob:float) -> pd.DataFrame:
    """
    This function provides initial value for finding the GEV parameters and then finds the best GEV parameters using 
       the function GEV_parameters.
    """
    year = raw_precip.index.values
    weights = np.append(1/year[:-1]-1/year[1:], 1/year[-1])
    Avg = (weights*raw_precip[huc]).sum()
    GEV_parameters = np.array([Avg*0.8, .5, -0.4])
    bounds = ((Avg*.1, Avg*1.0), (0.01, 3.1), (-0.5, 0))
    df_GEV_parameters = GEV_Parameters(raw_precip, GEV_parameters, bounds, huc, PMP,Atom_Prob)
    return df_GEV_parameters

def plot_GEV_precip_curves(precip_data: pd.DataFrame, df_GEV_parameters: pd.DataFrame, PMP: float, Atom_Prob:float,
                           Label1: str='') -> None:
    """This functions plots the GEV distributions and also associated GEV return frequency curves on top of the 
       precpitation curve data taken either from NOAA Atlas 14 or from the mean precipitation curve output.
    """
    color = ['r', 'k', 'k']
    _, ax = plt.subplots(1, 2, figsize=(10,4))
    for i, (_, columndata) in enumerate(df_GEV_parameters.iteritems()):
        Precip = np.linspace(PPF_GEV(1e-100, columndata.values, PMP), PPF_GEV(0.999, columndata.values, PMP), 1000)
        Return_Period = 1.0/(1-Atom_Prob-(1-Atom_Prob)*CDF_GEV(Precip, columndata.values, PMP))
        ax[0].plot(Precip, PDF_GEV(Precip, columndata.values, PMP), color[i] , lw=2, alpha=0.6)
        ax[1].plot(Return_Period, Precip, color[i], lw=2.5, alpha=0.6)
    for _, columndata in precip_data.iteritems():
        columndata.plot(style=['+-', 'o-', '.--'], logx=True)
    ax[0].set_xlabel(f'{Label1} [inches]')
    ax[0].set_ylabel('GEV PDF $p_R(R)$')
    ax[0].set_title(f'24-hour {Label1}')
    ax[1].set_xscale('log')
    ax[1].set_xlabel('Return Period [years]')
    ax[1].set_ylabel(f'{Label1} [inches]')
    ax[1].set_title(f'24-hour {Label1}')    
    return None

# COMMAND ----------


