from tigerforecast.utils.download_tools import *
from typing import List, Tuple
import random
from pathlib import Path, PosixPath
import pandas as pd
import sqlite3
import h5py
import numpy as np
import itertools
from numba import njit

GLOBAL_SETTINGS = {
    'batch_size': 256,
    'clip_norm': True,
    'clip_value': 1,
    'dropout': 0.4,
    'epochs': 30,
    'hidden_size': 256,
    'initial_forget_gate_bias': 5,
    'log_interval': 50,
    'learning_rate': 1e-3,
    'seq_length': 270,
    'train_start': pd.to_datetime('01101999', format='%d%m%Y'),
    'train_end': pd.to_datetime('30092008', format='%d%m%Y'),
    'val_start': pd.to_datetime('01101989', format='%d%m%Y'),
    'val_end': pd.to_datetime('30091999', format='%d%m%Y')
}

INVALID_ATTR = [
    'gauge_name', 'area_geospa_fabric', 'geol_1st_class', 'glim_1st_class_frac', 'geol_2nd_class',
    'glim_2nd_class_frac', 'dom_land_cover_frac', 'dom_land_cover', 'high_prec_timing',
    'low_prec_timing', 'huc', 'q_mean', 'runoff_ratio', 'stream_elas', 'slope_fdc',
    'baseflow_index', 'hfd_mean', 'q5', 'q95', 'high_q_freq', 'high_q_dur', 'low_q_freq',
    'low_q_dur', 'zero_q_freq', 'geol_porostiy', 'root_depth_50', 'root_depth_99', 'organic_frac',
    'water_frac', 'other_frac'
]

# Maurer mean/std calculated over all basins in period 01.10.1999 until 30.09.2008
SCALER = {
    'input_means': np.array([3.17563234, 372.01003929, 17.31934062, 3.97393362, 924.98004197]),
    'input_stds': np.array([6.94344737, 131.63560881, 10.86689718, 10.3940032, 629.44576432]),
    'output_mean': np.array([1.49996196]),
    'output_std': np.array([3.62443672])
}

def rescale_features(feature: np.ndarray, variable: str) -> np.ndarray:
    """Rescale features using global pre-computed statistics.

    Parameters
    ----------
    feature : np.ndarray
        Data to rescale
    variable : str
        One of ['inputs', 'output'], where `inputs` mean, that the `feature` input are the model
        inputs (meteorological forcing data) and `output` that the `feature` input are discharge
        values.

    Returns
    -------
    np.ndarray
        Rescaled features

    Raises
    ------
    RuntimeError
        If `variable` is neither 'inputs' nor 'output'
    """
    if variable == 'inputs':
        feature = feature * SCALER["input_stds"] + SCALER["input_means"]
    elif variable == 'output':
        feature = feature * SCALER["output_std"] + SCALER["output_mean"]
    else:
        raise RuntimeError(f"Unknown variable type {variable}")
    return feature

def load_attributes(
                db_path: str,
                basins: List,
                drop_lat_lon: bool = True,
                keep_features: List = None) -> pd.DataFrame:
    """Load attributes from database file into DataFrame

    Parameters
    ----------
    db_path : str
        Path to sqlite3 database file
    basins : List
        List containing the 8-digit USGS gauge id
    drop_lat_lon : bool
        If True, drops latitude and longitude column from final data frame, by default True
    keep_features : List
        If a list is passed, a pd.DataFrame containing these features will be returned. By default,
        returns a pd.DataFrame containing the features used for training.

    Returns
    -------
    pd.DataFrame
        Attributes in a pandas DataFrame. Index is USGS gauge id. Latitude and Longitude are
        transformed to x, y, z on a unit sphere.
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM 'basin_attributes'", conn, index_col='gauge_id')

    # drop rows of basins not contained in data set
    drop_basins = [b for b in df.index if b not in basins]
    df = df.drop(drop_basins, axis=0)

    # drop lat/lon col
    if drop_lat_lon:
        df = df.drop(['gauge_lat', 'gauge_lon'], axis=1)

    # drop invalid attributes
    if keep_features is not None:
        drop_names = [c for c in df.columns if c not in keep_features]
    else:
        drop_names = [c for c in df.columns if c in INVALID_ATTR]

    df = df.drop(drop_names, axis=1)

    return df

def get_basin_list(basin_path):
    """Read list of basins from text file.
    
    Returns
    -------
    List
        List containing the 8-digit basin code of all basins
    """
    # basin_file = Path(__file__).absolute().parent.parent / "data/basin_list.txt"
    f = open(basin_path)
    basins = f.readlines()
    # with self.basin_path.open('r') as fp:
    #    basins = fp.readlines()
    basins = [basin.strip() for basin in basins]
    f.close()
    return basins

class CamelsTXT():
    """PyTorch data set to work with the raw text files in the CAMELS data set.
       
    Parameters
    ----------
    camels_root : PosixPath
        Path to the main directory of the CAMELS data set
    basin : str
        8-digit usgs-id of the basin
    dates : List
        Start and end date of the period.
    is_train : bool
        If True, discharge observations are normalized and invalid discharge samples are removed
    seq_length : int, optional
        Length of the input sequence, by default 270
    with_attributes : bool, optional
        If True, loads and returns addtionaly attributes, by default False
    attribute_means : pd.Series, optional
        Means of catchment characteristics, used to normalize during inference, by default None
    attribute_stds : pd.Series, optional
        Stds of catchment characteristics, used to normalize during inference,, by default None
    concat_static : bool, optional
        If true, adds catchment characteristics at each time step to the meteorological forcing
        input data, by default False
    db_path : str, optional
        Path to sqlite3 database file, containing the catchment characteristics, by default None
    """

    def __init__(self,
                 basin: str,
                 is_train: bool = False,
                 dates: List = None,
                 seq_length: int = 270,
                 with_attributes: bool = True,
                 attribute_means: pd.Series = None,
                 attribute_stds: pd.Series = None,
                 concat_static: bool = False,
                 db_path: str = None):
        tigerforecast_dir = get_tigerforecast_dir()
        CAMELS_ROOT = os.path.join(tigerforecast_dir, 'data/usgs_flood/basin_dataset_public_v1p2')
        self.camels_root = CAMELS_ROOT
        self.basin = basin
        self.seq_length = seq_length
        self.is_train = is_train
        if dates == None:
            self.dates = [GLOBAL_SETTINGS['val_start'], GLOBAL_SETTINGS['val_end']]
        else:
            self.dates = dates
        self.with_attributes = with_attributes

        DB_PATH = os.path.join(tigerforecast_dir, 'data/usgs_flood/attributes.db')
        BASIN_PATH = os.path.join(tigerforecast_dir, 'data/usgs_flood/basin_list.txt')
        self.basins = get_basin_list(BASIN_PATH)
        attributes = load_attributes(db_path=DB_PATH, 
                                     basins=self.basins,
                                     drop_lat_lon=True)
        means = attributes.mean()
        stds = attributes.std()

        if attribute_means == None:
            self.attribute_means = means
        else:
            self.attribute_means = attribute_means
        if attribute_stds == None:
            self.attribute_stds = stds
        else:
            self.attribute_stds = attribute_stds
        self.concat_static = concat_static

        if db_path == None:
            self.db_path = DB_PATH
        else:
            self.db_path = db_path

        # placeholder to store std of discharge, used for rescaling losses during training
        self.q_std = None

        # placeholder to store start and end date of entire period (incl warmup)
        self.period_start = None
        self.period_end = None
        self.attribute_names = None

        self.x, self.y = self._load_data()

        if self.with_attributes:
            self.attributes = self._load_attributes()

        self.num_samples = self.x.shape[0]
        self.num_features = 32

        # self.target_lag = 1

    def __len(self):
        return self.num_samples

    def __getitem(self, idx: int):
        if self.with_attributes:
            if self.concat_static:
                # x = torch.cat([self.x[idx], self.attributes.repeat((self.seq_length, 1))], dim=-1)
                tiled_attributes = np.tile(self.attributes, (self.seq_length, 1))
                # print("self.x[idx].shape = " + str(self.x[idx].shape))
                # print("tiled_attributes.shape = " + str(tiled_attributes.shape))
                x = np.concatenate([self.x[idx], tiled_attributes], axis=1).astype(np.float32)
                return x, self.y[idx]
            else:
                return self.x[idx], self.attributes, self.y[idx]
        else:
            return self.x[idx], self.y[idx]

    def load_forcing(self, camels_root: PosixPath, basin: str) -> Tuple[pd.DataFrame, int]:
        """Load Maurer forcing data from text files.

        Parameters
        ----------
        camels_root : PosixPath
            Path to the main directory of the CAMELS data set
        basin : str
            8-digit USGS gauge id

        Returns
        -------
        df : pd.DataFrame
            DataFrame containing the Maurer forcing
        area: int
            Catchment area (read-out from the header of the forcing file)

        Raises
        ------
        RuntimeError
            If not forcing file was found.
        """
        forcing_path = Path(camels_root) / 'basin_mean_forcing' / 'maurer_extended'
        files = list(forcing_path.glob('**/*_forcing_leap.txt'))
        file_path = [f for f in files if f.name[:8] == basin]
        if len(file_path) == 0:
            raise RuntimeError(f'No file for Basin {basin} at {file_path}')
        else:
            file_path = file_path[0]

        df = pd.read_csv(file_path, sep='\s+', header=3)
        dates = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str))
        df.index = pd.to_datetime(dates, format="%Y/%m/%d")

        # load area from header
        with open(file_path, 'r') as fp:
            content = fp.readlines()
            area = int(content[2])

        return df, area

    def load_discharge(self, camels_root: PosixPath, basin: str, area: int) -> pd.Series:
        """[summary]

        Parameters
        ----------
        camels_root : PosixPath
            Path to the main directory of the CAMELS data set
        basin : str
            8-digit USGS gauge id
        area : int
            Catchment area, used to normalize the discharge to mm/day

        Returns
        -------
        pd.Series
            A Series containing the discharge values.

        Raises
        ------
        RuntimeError
            If no discharge file was found.
        """
        discharge_path = Path(camels_root) / 'usgs_streamflow'
        files = list(discharge_path.glob('**/*_streamflow_qc.txt'))
        file_path = [f for f in files if f.name[:8] == basin]
        if len(file_path) == 0:
            raise RuntimeError(f'No file for Basin {basin} at {file_path}')
        else:
            file_path = file_path[0]

        col_names = ['basin', 'Year', 'Mnth', 'Day', 'QObs', 'flag']
        df = pd.read_csv(file_path, sep='\s+', header=None, names=col_names)
        dates = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str))
        df.index = pd.to_datetime(dates, format="%Y/%m/%d")

        # normalize discharge from cubic feed per second to mm per day
        df.QObs = 28316846.592 * df.QObs * 86400 / (area * 10**6)

        return df.QObs

    def normalize_features(self, feature: np.ndarray, variable: str) -> np.ndarray:
        """Normalize features using global pre-computed statistics.

        Parameters
        ----------
        feature : np.ndarray
            Data to normalize
        variable : str
            One of ['inputs', 'output'], where `inputs` mean, that the `feature` input are the model
            inputs (meteorological forcing data) and `output` that the `feature` input are discharge
            values.

        Returns
        -------
        np.ndarray
            Normalized features

        Raises
        ------
        RuntimeError
            If `variable` is neither 'inputs' nor 'output'
        """

        if variable == 'inputs':
            feature = (feature - SCALER["input_means"]) / SCALER["input_stds"]
        elif variable == 'output':
            feature = (feature - SCALER["output_mean"]) / SCALER["output_std"]
        else:
            raise RuntimeError(f"Unknown variable type {variable}")
        return feature

    # @njit
    def reshape_data(self, x: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Reshape data into LSTM many-to-one input samples

        Parameters
        ----------
        x : np.ndarray
            Input features of shape [num_samples, num_features]
        y : np.ndarray
            Output feature of shape [num_samples, 1]
        seq_length : int
            Length of the requested input sequences.

        Returns
        -------
        x_new: np.ndarray
            Reshaped input features of shape [num_samples*, seq_length, num_features], where 
            num_samples* is equal to num_samples - seq_length + 1, due to the need of a warm start at
            the beginning
        y_new: np.ndarray
            The target value for each sample in x_new
        """
        num_samples, num_features = x.shape

        x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
        y_new = np.zeros((num_samples - seq_length + 1, 1))

        for i in range(0, x_new.shape[0]):
            x_new[i, :, :num_features] = x[i:i + seq_length, :]
            y_new[i, :] = y[i + seq_length - 1, 0]

        return x_new, y_new

    def sequential_batches(self, batch_size, num_batches=None):
        # generator for sequential batches

        batch_data = np.zeros([batch_size, self.seq_length, self.num_features])
        batch_targets = np.zeros([batch_size, self.seq_length])

        # for now, truncate the end

        # first = self.seq_length + self.target_lag - 1 + skip_first
        # if not num_batches:
        #    num_batches = (self.data_length[site_idx] - first) // batch_size
        if not num_batches:
            if self.num_samples % batch_size == 0:
                num_batches = self.num_samples // batch_size
            else:
                num_batches = self.num_samples // batch_size + 1
            # num_batches = self.num_samples // batch_size
        # print("sequential self.num_samples = " + str(self.num_samples))
        for k in range(num_batches):
            for i in range(batch_size):
                # t = first + k*batch_size + i
                # print(site_idx, t)
                # data, target = self.featurize(site_idx, t)
                sample_idx = k*batch_size + i
                if sample_idx == self.num_samples:
                    yield batch_data[:i], batch_targets[:i]
                    return
                data, target = self.__getitem(sample_idx)
                batch_data[i] = data
                batch_targets[i] = target
            yield batch_data, batch_targets

    def _load_data(self):
        """Load input and output data from text files."""
        df, area = self.load_forcing(self.camels_root, self.basin)
        df['QObs(mm/d)'] = self.load_discharge(self.camels_root, self.basin, area)

        # we use (seq_len) time steps before start for warmup
        start_date = self.dates[0] - pd.DateOffset(days=self.seq_length - 1)
        end_date = self.dates[1]
        df = df[start_date:end_date]

        # store first and last date of the selected period (including warm_start)
        self.period_start = df.index[0]
        self.period_end = df.index[-1]

        # use all meteorological variables as inputs
        x = np.array([
            df['prcp(mm/day)'].values, df['srad(W/m2)'].values, df['tmax(C)'].values,
            df['tmin(C)'].values, df['vp(Pa)'].values
        ]).T

        y = np.array([df['QObs(mm/d)'].values]).T

        # normalize data, reshape for LSTM training and remove invalid samples
        x = self.normalize_features(x, variable='inputs')

        x, y = self.reshape_data(x, y, self.seq_length)

        if self.is_train:
            # Deletes all records, where no discharge was measured (-999)
            x = np.delete(x, np.argwhere(y < 0)[:, 0], axis=0)
            y = np.delete(y, np.argwhere(y < 0)[:, 0], axis=0)

            # Delete all samples, where discharge is NaN
            if np.sum(np.isnan(y)) > 0:
                print(f"Deleted some records because of NaNs in basin {self.basin}")
                x = np.delete(x, np.argwhere(np.isnan(y)), axis=0)
                y = np.delete(y, np.argwhere(np.isnan(y)), axis=0)

            # store std of discharge before normalization
            self.q_std = np.std(y)

            y = normalize_features(y, variable='output')

        # convert arrays to torch tensors
        # x = torch.from_numpy(x.astype(np.float32))
        # y = torch.from_numpy(y.astype(np.float32))

        return x.astype(np.float32), y.astype(np.float32)

    def _load_attributes(self):
        df = load_attributes(self.db_path, [self.basin], drop_lat_lon=True)

        # normalize data
        df = (df - self.attribute_means) / self.attribute_stds

        # store attribute names
        self.attribute_names = df.columns

        # store feature as PyTorch Tensor
        attributes = df.loc[df.index == self.basin].values
        return attributes.astype(np.float32)
        # return torch.from_numpy(attributes.astype(np.float32))

class CamelsH5():
    """PyTorch data set to work with pre-packed hdf5 data base files.

    Should be used only in combination with the files processed from `create_h5_files` in the 
    `papercode.utils` module.

    Parameters
    ----------
    h5_file : PosixPath
        Path to hdf5 file, containing the bundled data
    basins : List
        List containing the 8-digit USGS gauge id
    db_path : str
        Path to sqlite3 database file, containing the catchment characteristics
    concat_static : bool
        If true, adds catchment characteristics at each time step to the meteorological forcing
        input data, by default False
    cache : bool, optional
        If True, loads the entire data into memory, by default False
    no_static : bool, optional
        If True, no catchment attributes are added to the inputs, by default False
    """

    def __init__(self,
                 h5_file: PosixPath=None, 
                 db_path: str=None,
                 concat_static: bool = False,
                 cache: bool = False,
                 no_static: bool = False):
        # self.h5_file = h5_file
        tigerforecast_dir = get_tigerforecast_dir()
        DB_PATH = os.path.join(tigerforecast_dir, 'data/usgs_flood/attributes.db')
        H5_PATH = os.path.join(tigerforecast_dir, 'data/usgs_flood/train_data.h5')
        BASIN_PATH = os.path.join(tigerforecast_dir, 'data/usgs_flood/basin_list.txt')
        if h5_file == None:
            self.h5_file = H5_PATH
        else:
            self.h5_file = h5_file
        self.basin_path = BASIN_PATH
        self.basins = get_basin_list(self.basin_path)
        # self.db_path = db_path
        if db_path == None:
            self.db_path = DB_PATH
        else:
            self.db_path = db_path
        self.concat_static = concat_static
        self.cache = cache
        self.no_static = no_static

        # Placeholder for catchment attributes stats
        self.df = None
        self.attribute_means = None
        self.attribute_stds = None
        self.attribute_names = None

        self.seq_length = 270
        self.num_features = 32

        # preload data if cached is true
        if self.cache:
            (self.x, self.y, self.sample_2_basin, self.q_stds) = self._preload_data()

        # load attributes into data frame
        self._load_attributes()

        # determine number of samples once
        if self.cache:
            self.num_samples = self.y.shape[0]
        else:
            with h5py.File(self.h5_file, 'r') as f:
                self.num_samples = f["target_data"].shape[0]

    def __len(self):
        return self.num_samples

    def __getitem(self, idx: int):
        if self.cache:
            x = self.x[idx]
            y = self.y[idx]
            basin = self.sample_2_basin[idx]
            q_std = self.q_stds[idx]

        else:
            with h5py.File(self.h5_file, 'r') as f:
                x = f["input_data"][idx]
                y = f["target_data"][idx]
                basin = f["sample_2_basin"][idx]
                basin = basin.decode("ascii")
                q_std = f["q_stds"][idx]

        if not self.no_static:
            # get attributes from data frame and create 2d array with copies
            attributes = self.df.loc[self.df.index == basin].values

            if self.concat_static:
                attributes = np.repeat(attributes, repeats=x.shape[0], axis=0)
                # combine meteorological obs with static attributes
                x = np.concatenate([x, attributes], axis=1).astype(np.float32)
            else:
                attributes = torch.from_numpy(attributes.astype(np.float32))

        # convert to torch tensors
        # x = torch.from_numpy(x.astype(np.float32))
        # y = torch.from_numpy(y.astype(np.float32))
        # q_std = torch.from_numpy(q_std)

        if self.no_static:
            return x, y, q_std
        else:
            if self.concat_static:
                return x, y, q_std
            else:
                return x, attributes, y, q_std

    def random_sample_idx(self):
        # generates a random site index
        return random.randint(0, self.num_samples-1)

    def random_batches(self, batch_size, num_batches=None):
        # generator for random batches
        # default num_batches: infinity

        batch_data = np.zeros([batch_size, self.seq_length, self.num_features])
        batch_targets = np.zeros([batch_size, self.seq_length])
        print("self.num_samples = " + str(self.num_samples))

        for _ in itertools.repeat(None, num_batches):
            for i in range(batch_size):
                rand_idx = self.random_sample_idx()
                # rand_t = self.random_valid_time(rand_idx)
                # print(rand_idx, rand_t)
                # data, target = self.featurize(rand_idx, rand_t)
                data, target, q_std = self.__getitem(rand_idx)
                batch_data[i] = data
                batch_targets[i] = target
            yield batch_data, batch_targets

    def _preload_data(self):
        with h5py.File(self.h5_file, 'r') as f:
            x = f["input_data"][:]
            y = f["target_data"][:]
            str_arr = f["sample_2_basin"][:]
            str_arr = [x.decode("ascii") for x in str_arr]
            q_stds = f["q_stds"][:]
        return x, y, str_arr, q_stds

    def _get_basins(self):
        if self.cache:
            basins = list(set(self.sample_2_basin))
        else:
            with h5py.File(self.h5_file, 'r') as f:
                str_arr = f["sample_2_basin"][:]
            str_arr = [x.decode("ascii") for x in str_arr]
            basins = list(set(str_arr))
        return basins

    def _load_attributes(self):
        df = load_attributes(self.db_path, self.basins, drop_lat_lon=True)

        # store means and stds
        self.attribute_means = df.mean()
        self.attribute_stds = df.std()

        # normalize data
        df = (df - self.attribute_means) / self.attribute_stds

        self.attribute_names = df.columns
        self.df = df

    def get_attribute_means(self) -> pd.Series:
        """Return means of catchment attributes
        
        Returns
        -------
        pd.Series
            Contains the means of each catchment attribute
        """
        return self.attribute_means

    def get_attribute_stds(self) -> pd.Series:
        """Return standard deviation of catchment attributes
        
        Returns
        -------
        pd.Series
            Contains the stds of each catchment attribute
        """
        return self.attribute_stds

'''
dataloader = CamelsH5(concat_static=True)
# dataloader.random_batches(batch_size=1024, num_batches=1)
for i, (data, targets) in enumerate( dataloader.random_batches(batch_size=1024, num_batches=1)):
    print("data.shape = " + str(data.shape))
    print("targets.shape = " + str(targets.shape))

print("-----------------------------")'''
tigerforecast_dir = get_tigerforecast_dir()
BASIN_PATH = os.path.join(tigerforecast_dir, 'data/usgs_flood/basin_list.txt')
basin_list = get_basin_list(BASIN_PATH)
eval_dataloader = CamelsTXT(basin=basin_list[0], concat_static=True)
for i, (data, targets) in enumerate( eval_dataloader.sequential_batches(batch_size=1024)):
    print("data.shape = " + str(data.shape))
    print("targets.shape = " + str(targets.shape))

