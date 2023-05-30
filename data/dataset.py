import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from pathlib import Path
from bisect import bisect_right
from abc import ABCMeta, abstractmethod


class AbstractReader(metaclass=ABCMeta):
    """Abstract data reader.

    Its subclasses need to ensure conversion from raw data to pandas.DataFrame,
    and process invalid data item
    """

    @abstractmethod
    def _load_data(self, *args, **kwargs):
        """
        Subclasses must implement loading inputs and target data.
        """
        pass

    @abstractmethod
    def _process_invalid_data(self, *args, **kwargs):
        """
        Subclasses must implement how to process invalid data item.
        """
        pass

    @abstractmethod
    def get_df_x(self):
        """
        Subclasses must return inputs data with a form of pandas.DataFrame.
        """
        pass

    @abstractmethod
    def get_df_y(self):
        """
        Subclasses must return target data with a form of pandas.DataFrame.
        """
        pass


class AbstractStaticReader:
    """Abstract static data reader.

    1. Reads data from a static attributes file (.csv).
    2. Select used attributes and do normalization.
    3. Need to ensure conversion from static attributes to pandas.DataFrame.
    """

    @abstractmethod
    def get_df_static(self, basin):
        """
        Subclasses must return static data with a form of pandas.DataFrame for a specific basin
        """
        pass


class DaymetHydroReader(AbstractReader):
    camels_root = None  # needs to be set in class method "init_root"
    forcing_root = None
    discharge_root = None
    forcing_cols = ["Year", "Mnth", "Day", "Hr", "dayl(s)", "prcp(mm/day)", "srad(W/m2)", "swe(mm)", "tmax(C)",
                    "tmin(C)", "vp(Pa)"]
    features = ["prcp(mm/day)", "srad(W/m2)", "tmax(C)", "tmin(C)", "vp(Pa)"]
    discharge_cols = ["basin", "Year", "Mnth", "Day", "QObs", "flag"]
    target = ["QObs(mm/d)"]

    @classmethod
    def init_root(cls, camels_root):  # often be rewritten
        cls.camels_root = Path(camels_root)
        cls.forcing_root = cls.camels_root / "basin_mean_forcing" / "daymet"
        cls.discharge_root = cls.camels_root / "usgs_streamflow"

    def __init__(self, basin: str):
        self.basin = basin
        self.area = None
        df = self._load_data()
        df = self._process_invalid_data(df)
        self.df_x = df[self.features]  # Datetime as index
        self.df_y = df[self.target]  # Datetime as index

    def get_df_x(self):
        return self.df_x

    def get_df_y(self):
        return self.df_y

    def _load_data(self):
        df_forcing = self._load_forcing()
        df_discharge = self._load_discharge()
        df = pd.concat([df_forcing, df_discharge], axis=1)

        return df

    # Loading meteorological data
    def _load_forcing(self):
        files = list(self.forcing_root.glob(f"**/{self.basin}_*.txt"))
        if len(files) == 0:
            raise RuntimeError(f"No forcing file found for Basin {self.basin}")
        elif len(files) >= 2:
            raise RuntimeError(f"Redundant forcing files found for Basin {self.basin}")
        else:
            file_path = files[0]

        # read-in data and convert date to datetime index
        df = pd.read_csv(file_path, sep=r"\s+", header=3)  # \s+ means matching any whitespace character
        dates = df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str)
        df.index = pd.to_datetime(dates, format="%Y/%m/%d")

        # Line 2 (starting at 0) of the file is the area value
        with open(file_path) as fp:
            # readline is faster than readines, if only read two lines
            fp.readline()
            fp.readline()
            content = fp.readline().strip()
            area = int(content)
        self.area = area

        return df[self.features]

    # Loading runoff data
    def _load_discharge(self):
        files = list(self.discharge_root.glob(f"**/{self.basin}_*.txt"))
        if len(files) == 0:
            raise RuntimeError(f"No discharge file found for Basin {self.basin}")
        elif len(files) >= 2:
            raise RuntimeError(f"Redundant discharge files found for Basin {self.basin}")
        else:
            file_path = files[0]

        df = pd.read_csv(file_path, sep=r"\s+", header=None, names=self.discharge_cols)
        dates = df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str)
        df.index = pd.to_datetime(dates, format="%Y/%m/%d")

        # normalize discharge from cubic feed per second to mm per day
        assert len(self.target) == 1
        df[self.target[0]] = 28316846.592 * df["QObs"] * 86400 / (self.area * 10 ** 6)

        return df[self.target]

    # Processing invalid data
    def _process_invalid_data(self, df: pd.DataFrame):
        # Delete all row, where exits NaN (only discharge has NaN in this dataset)
        len_raw = len(df)
        df = df.dropna()
        len_drop_nan = len(df)
        if len_raw > len_drop_nan:
            print(f"Deleted {len_raw - len_drop_nan} records because of NaNs {self.basin}")

        # Deletes all records, where no discharge was measured (-999)
        df = df.drop((df[df['QObs(mm/d)'] < 0]).index)
        len_drop_neg = len(df)
        if len_drop_nan > len_drop_neg:
            print(f"Deleted {len_drop_nan - len_drop_neg} records because of negative discharge {self.basin}")

        return df


class MaurerExtHydroReader(DaymetHydroReader):
    camels_root = None  # needs to be set in class method "init_root"
    forcing_root = None
    discharge_root = None

    @classmethod
    def init_root(cls, camels_root):
        cls.camels_root = Path(camels_root)
        cls.forcing_root = cls.camels_root / "basin_mean_forcing" / "maurer_extended"
        cls.discharge_root = cls.camels_root / "usgs_streamflow"

    def __init__(self, basin: str):
        super().__init__(basin)


class NldasExtHydroReader(DaymetHydroReader):
    camels_root = None  # needs to be set in class method "init_root"
    forcing_root = None
    discharge_root = None
    forcing_cols = ["Year", "Mnth", "Day", "Hr", "Dayl(s)", "PRCP(mm/day)", "SRAD(W/m2)", "SWE(mm)", "Tmax(C)",
                    "Tmin(C)", "Vp(Pa)"]
    features = ["PRCP(mm/day)", "SRAD(W/m2)", "Tmax(C)", "Tmin(C)", "Vp(Pa)"]

    @classmethod
    def init_root(cls, camels_root):
        cls.camels_root = Path(camels_root)
        cls.forcing_root = cls.camels_root / "basin_mean_forcing" / "nldas_extended"
        cls.discharge_root = cls.camels_root / "usgs_streamflow"

    def __init__(self, basin: str):
        super().__init__(basin)


class HydroReaderFactory:
    """
    Simple factory for producing HydroReader
    """

    @staticmethod
    def get_hydro_reader(camels_root, forcing_type, basin):
        if forcing_type == "daymet":
            DaymetHydroReader.init_root(camels_root)
            reader = DaymetHydroReader(basin)
        elif forcing_type == "maurer_extended":
            MaurerExtHydroReader.init_root(camels_root)
            reader = MaurerExtHydroReader(basin)
        elif forcing_type == "nldas_extended":
            NldasExtHydroReader.init_root(camels_root)
            reader = NldasExtHydroReader(basin)
        else:
            raise RuntimeError(f"No such hydro reader type: {forcing_type}")

        return reader


class CamelsDataset(Dataset):
    """CAMELS dataset working with subclasses of AbstractHydroReader.

    It works in a list way: the model trains, validates and tests with all of basins in attribute:basins_list.

    Attributes:
        camels_root: str
            The root of CAMELS dataset.
        basins_list: list of str
            A list contains all needed basins-ids (8-digit code).
        past_len: int
            Length of the past time steps for discharge data.
        pred_len: int
            Length of the predicting time steps for discharge data.
            And it is worth noting that the used length of meteorological data is (past_len + :pred_len).
        stage: str
            One of ['train', 'val', 'test'], decide whether calculating mean and std or not.
            Calculate mean and std in training stage.
        dates: List of pd.DateTimes
            Means the date range that is used, containing two elements, i.e, start date and end date.
        x_dict: dict as {basin: np.ndarray}
             Mapping a basin to its corresponding meteorological data.
        y_dict: dict as {basin: np.ndarray}
             Mapping a basin to its corresponding discharge data.
        length_ls: list of int
            Contains number of serialized sequences of each basin corresponding to basins_list.
        index_ls: list of int
            Created from length_ls, used in __getitem__ method.
        num_samples: int
            Number of serialized sequences of all basins.
        x_mean: numpy.ndarray
            Mean of input features derived from the training stage.
            Has to be provided for 'val' or 'test' stage.
            Can be retrieved if calling .get_means() on the data set.
        y_mean: numpy.ndarray
            Mean of output features derived from the training stage.
            Has to be provided for 'val' or 'test' stage.
            Can be retrieved if calling .get_means() on the data set.
        x_std: numpy.ndarray
            Std of input features derived from the training stage.
            Has to be provided for 'val' or 'test' stage.
            Can be retrieved if calling .get_stds() on the data set.
        y_std: numpy.ndarray
            Std of output features derived from the training stage.
            Has to be provided for 'val' or 'test' stage.
            Can be retrieved if calling .get_stds() on the data set.
    """

    def __init__(self, camels_root: str, forcing_type: str, basins_list: list, past_len: int, pred_len: int, stage: str,
                 dates: list, x_mean=None, y_mean=None, x_std=None, y_std=None, y_stds_dict=None):
        """Initialization

        x_mean, y_mean, x_std, y_std should be provided if stage != "train".
        """
        self.camels_root = camels_root
        self.basins_list = basins_list
        self.past_len = past_len
        self.pred_len = pred_len
        self.stage = stage
        self.dates = dates
        self.x_dict = dict()
        self.y_dict = dict()
        self.date_index_dict = dict()
        self.length_ls = list()

        if y_stds_dict is None:
            self.y_stds_dict = dict()
        else:
            self.y_stds_dict = y_stds_dict

        self._load_data(forcing_type)
        # Calculate mean and std
        if self.stage == 'train':
            self.x_mean, self.x_std = self.calc_mean_and_std(self.x_dict)
            self.y_mean, self.y_std = self.calc_mean_and_std(self.y_dict)
        else:
            self.x_mean = x_mean
            self.y_mean = y_mean
            self.x_std = x_std
            self.y_std = y_std
        self.normalize_data()

        self.num_samples = 0
        for item in self.length_ls:
            self.num_samples += item

        self.index_ls = [0]
        for i in range(len(self.length_ls)):
            v = self.index_ls[i] + self.length_ls[i]
            self.index_ls.append(v)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        basin_idx = bisect_right(self.index_ls, idx) - 1
        local_idx = idx - self.index_ls[basin_idx]
        basin = self.basins_list[basin_idx]
        x_seq = self.x_dict[basin][local_idx: local_idx + self.past_len + self.pred_len, :]
        y_seq_past = self.y_dict[basin][local_idx: local_idx + self.past_len, :]
        y_seq_future = self.y_dict[basin][local_idx + self.past_len: local_idx + self.past_len + self.pred_len, :]

        return x_seq, y_seq_past, y_seq_future, self.y_stds_dict[basin]

    def _load_data(self, forcing_type):
        # Loading vanilla data
        basin_number = len(self.basins_list)
        for idx, basin in enumerate(self.basins_list):
            print(self.stage, f"{basin}: loading data %.4f" % (idx / basin_number))
            reader = HydroReaderFactory.get_hydro_reader(self.camels_root, forcing_type, basin)
            df_x = reader.get_df_x()
            df_y = reader.get_df_y()

            # Select date
            df_x = df_x[self.dates[0]:self.dates[1]]
            df_y = df_y[self.dates[0]:self.dates[1]]
            assert len(df_x) == len(df_y)
            self.date_index_dict[basin] = df_x.index

            # Select used features and discharge
            x = df_x.values.astype("float32")
            y = df_y.values.astype("float32")
            self.x_dict[basin] = x
            self.y_dict[basin] = y

            self.length_ls.append(len(x) - self.past_len - self.pred_len + 1)
            # Calculate mean and std in training stage
            if self.stage == 'train':
                self.y_stds_dict[basin] = y.std(axis=0).item()

    @staticmethod
    def calc_mean_and_std(data_dict):
        data_all = np.concatenate(list(data_dict.values()), axis=0)  # CAN NOT serializable
        nan_mean = np.nanmean(data_all, axis=0)
        nan_std = np.nanstd(data_all, axis=0)
        return nan_mean, nan_std

    def _local_normalization(self, feature: np.ndarray, variable: str) -> np.ndarray:
        if variable == 'inputs':
            feature = (feature - self.x_mean) / self.x_std
        elif variable == 'output':
            feature = (feature - self.y_mean) / self.y_std
        else:
            raise RuntimeError(f"Unknown variable type {variable}")
        return feature

    def normalize_data(self):
        # Normalize data
        for idx, basin in enumerate(self.basins_list):
            print(self.stage, "Normalizing %.4f" % (idx / len(self.basins_list)))
            x = self.x_dict[basin]
            y = self.y_dict[basin]
            # Normalize data
            x_norm = self._local_normalization(x, variable='inputs')
            y_norm = self._local_normalization(y, variable='output')
            self.x_dict[basin] = x_norm
            self.y_dict[basin] = y_norm

    def local_rescale(self, feature: np.ndarray, variable: str) -> np.ndarray:
        if variable == 'inputs':
            feature = feature * self.x_std + self.x_mean
        elif variable == 'output':
            feature = feature * self.y_std + self.y_mean
        else:
            raise RuntimeError(f"Unknown variable type {variable}")
        return feature

    def get_means(self):
        return self.x_mean, self.y_mean

    def get_stds(self):
        return self.x_std, self.y_std

    @classmethod
    def get_instance(cls, past_len: int, pred_len: int, stage: str, specific_cfg: dict,
                     x_mean=None, y_mean=None, x_std=None, y_std=None, y_stds_dict=None):
        final_data_path = specific_cfg["final_data_path"]
        camels_root = specific_cfg["camels_root"]
        basins_list = specific_cfg["basins_list"]
        forcing_type = specific_cfg["forcing_type"]
        start_date = specific_cfg["start_date"]
        end_date = specific_cfg["end_date"]
        if final_data_path is None:
            dates = [start_date, end_date]
            instance = cls(camels_root, forcing_type, basins_list, past_len, pred_len, stage,
                           dates, x_mean, y_mean, x_std, y_std, y_stds_dict)
            return instance
        else:
            if final_data_path.exists():
                instance = torch.load(final_data_path)
                return instance
            else:
                dates = [start_date, end_date]
                instance = cls(camels_root, forcing_type, basins_list, past_len, pred_len, stage,
                               dates, x_mean, y_mean, x_std, y_std, y_stds_dict)
                final_data_path.parent.mkdir(exist_ok=True, parents=True)
                torch.save(instance, final_data_path)
                return instance


class StaticReader(AbstractStaticReader):
    """Static hydrological data reader.

    Reads data from a selected norm static attributes file (.csv).
    Need to ensure conversion from static attributes to pandas.DataFrame.
    """

    def __init__(self, camels_root):
        self.camels_root = Path(camels_root)
        self.static_file_path = Path(
            "/data1/du/CAMELS/CAMELS-US") / "camels_attributes_v2.0" / "selected_norm_static_attributes.csv"
        self.df_static = pd.read_csv(self.static_file_path, header=0, dtype={"gauge_id": str}).set_index("gauge_id")
        self.df_static = self.df_static.astype("float32")

    def get_df_static(self, basin):
        return self.df_static.loc[[basin]].values


class CamelsDatasetWithStatic(CamelsDataset):
    """CAMELS dataset with static attributes injected into serialized sequences.

    Inherited from NullableCamelsDataset

    """

    def __init__(self, camels_root: str, forcing_type: str, basins_list: list, past_len: int, pred_len: int, stage: str,
                 dates: list, x_mean=None, y_mean=None, x_std=None, y_std=None, y_stds_dict=None):
        self.static_reader = StaticReader(camels_root)
        self.norm_static_fea = dict()
        super().__init__(camels_root, forcing_type, basins_list, past_len, pred_len, stage,
                         dates, x_mean, y_mean, x_std, y_std, y_stds_dict)

    def _load_data(self, forcing_type):
        # Loading vanilla data
        basin_number = len(self.basins_list)
        for idx, basin in enumerate(self.basins_list):
            print(self.stage, f"{basin}: loading data %.4f" % (idx / basin_number))
            reader = HydroReaderFactory.get_hydro_reader(self.camels_root, forcing_type, basin)
            df_x = reader.get_df_x()
            df_y = reader.get_df_y()

            # Select date
            df_x = df_x[self.dates[0]:self.dates[1]]
            df_y = df_y[self.dates[0]:self.dates[1]]
            assert len(df_x) == len(df_y)
            self.date_index_dict[basin] = df_x.index

            # Select used features and discharge
            x = df_x.values.astype("float32")
            y = df_y.values.astype("float32")

            self.x_dict[basin] = x
            self.y_dict[basin] = y

            self.length_ls.append(len(x) - self.past_len - self.pred_len + 1)
            # adding static attributes
            self.norm_static_fea[basin] = self.static_reader.get_df_static(basin)

            # Calculate mean and std in training stage
            if self.stage == 'train':
                self.y_stds_dict[basin] = y.std(axis=0).item()

    def normalize_data(self):
        # Normalize data
        for idx, basin in enumerate(self.basins_list):
            print(self.stage, "Normalizing %.4f" % (idx / len(self.basins_list)))
            x = self.x_dict[basin]
            y = self.y_dict[basin]
            # Normalize data
            x_norm = self._local_normalization(x, variable='inputs')
            y_norm = self._local_normalization(y, variable='output')
            norm_static_fea = self.norm_static_fea[basin].repeat(x_norm.shape[0], axis=0)
            x_norm_static = np.concatenate([x_norm, norm_static_fea], axis=1)
            self.x_dict[basin] = x_norm_static
            self.y_dict[basin] = y_norm

    def __getitem__(self, idx: int):
        basin_idx = bisect_right(self.index_ls, idx) - 1
        local_idx = idx - self.index_ls[basin_idx]
        basin = self.basins_list[basin_idx]
        x_seq = self.x_dict[basin][local_idx: local_idx + self.past_len + self.pred_len, :]
        y_seq_past = self.y_dict[basin][local_idx: local_idx + self.past_len, :]
        y_seq_future = self.y_dict[basin][local_idx + self.past_len: local_idx + self.past_len + self.pred_len, :]

        return x_seq, y_seq_past, y_seq_future, self.y_stds_dict[basin]



class CamelsDatasetLimited(CamelsDataset):
    def __getitem__(self, idx: int):
        basin_idx = bisect_right(self.index_ls, idx) - 1
        local_idx = idx - self.index_ls[basin_idx]
        basin = self.basins_list[basin_idx]
        x_seq = self.x_dict[basin][local_idx: local_idx + self.past_len, :]
        y_seq_past = self.y_dict[basin][local_idx: local_idx + self.past_len, :]
        y_seq_future = self.y_dict[basin][local_idx + self.past_len: local_idx + self.past_len + self.pred_len, :]

        return x_seq, y_seq_past, y_seq_future, self.y_stds_dict[basin]


class DatasetFactory:
    @staticmethod
    def get_dataset_type(use_future_fea, use_static):
        if (not use_future_fea) and use_static:
            raise RuntimeError("No implemented yet.")
        elif not use_future_fea:
            ds = CamelsDatasetLimited
        elif use_static:
            ds = CamelsDatasetWithStatic
        else:
            ds = CamelsDataset
        return ds
