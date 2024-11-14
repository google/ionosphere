# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Code used to fit ionosphere maps from phone measurements."""
import cartopy
import dataclasses
import datetime
import functools
from typing import Callable, Iterable, Optional, Tuple, Union
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import s2sphere as s2
import scipy.sparse as sps
import sksparse.cholmod
import warnings

jax.config.update('jax_enable_x64', True)

# Handle differently named methods in s2sphere vs google s2 library
s2.S2CellId = s2.CellId
s2.S2LatLng = s2.LatLng
s2.S2CellId.parent_at_level = s2.CellId.parent
s2.S2CellId.token = s2.CellId.to_token
s2.S2Cell = s2.Cell
s2.S2Cell.vertex = s2.Cell.get_vertex


# A constant used to convert from STEC in TECU to the associated GNSS error in
# meters. Units are m^3 / s^2
ALPHA = 40.3 * 10**16
SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
NS_PER_S = 10**9
L1_HZ = 1_575_420_032
L5_HZ = 1_176_450_048
IONOSPHERE_S2_LEVEL = 7
RECEIVER_S2_LEVEL = 10


def raw_stec_from_received_sv_time_nanos_diff(
    high_freq_hz, low_freq_hz, low_minus_high_received_sv_time_nanos
):
  """Returns raw STEC (no biases removed) for dual frequency time difference.

  Args:
    high_freq_hz: The higher carrier frequency in Hz.
    low_freq_hz: The lower carrier frequency in Hz.
    low_minus_high_received_sv_time_nanos: The difference in received SV time
      between the low and high frequency signals. The received sv time is the
      received GNSS satellite time, at the measurement time, in nanoseconds. It
      is given in the GNSS Measurement in the Android API.
      https://developer.android.com/reference/android/location/GnssMeasurement

  Returns:
    The raw STEC in TECU.
  """
  pseudorange_diff_m = (
      SPEED_OF_LIGHT_M_PER_S * low_minus_high_received_sv_time_nanos / NS_PER_S
  )
  return raw_stec_from_pseudorange_diff(
      high_freq_hz, low_freq_hz, pseudorange_diff_m
  )


def raw_stec_from_pseudorange_diff(
    high_freq_hz, low_freq_hz, high_minus_low_pseudorange_diff_m
):
  """Returns raw STEC (no biases removed) for dual freq pseudorange diff."""
  beta = _calc_beta(high_freq_hz, low_freq_hz)
  return high_minus_low_pseudorange_diff_m / beta


def _calc_beta(high_freq_hz, low_freq_hz):
  # Multiply by 1.0 to ensure floats are used. numpy int32 arrays overflow.
  squared_high_freq_hz = (1.0 * high_freq_hz) ** 2
  squared_low_freq_hz = (1.0 * low_freq_hz) ** 2
  beta = ALPHA * (1 / squared_high_freq_hz - 1 / squared_low_freq_hz)
  return beta


def diff_nanos_from_raw_stec(high_freq_hz, low_freq_hz, stec):
  """Returns the travel time difference in nanos for the given STEC."""
  beta = _calc_beta(high_freq_hz, low_freq_hz)
  pr_diff_m = stec * beta
  diff_nanos = pr_diff_m / (SPEED_OF_LIGHT_M_PER_S / NS_PER_S)
  return diff_nanos


def coarse_id_from_lat_lng_deg(lat_deg, lng_deg):
  lat_lng = s2.S2LatLng.from_degrees(lat_deg, lng_deg)
  return s2.S2CellId.from_lat_lng(lat_lng).parent_at_level(RECEIVER_S2_LEVEL).id()


def _cell_id_to_vertices(cell_id: s2.S2CellId) -> np.ndarray:
  lng = []
  lat = []
  cell = s2.S2Cell(cell_id)
  for j in range(4):
    lat_lng = s2.S2LatLng.from_point(cell.vertex(j))
    lng.append(lat_lng.lng().degrees)
    lat.append(lat_lng.lat().degrees)
  return np.array([lng, lat]).T


DATETIME_FORMAT = '%Y_%m_%d_T_%H_%M_%S'


def string_from_utc_sec(utc_sec: int) -> str:
  dt = datetime.datetime.fromtimestamp(utc_sec, tz=datetime.timezone.utc)
  return dt.strftime(DATETIME_FORMAT)


def utc_sec_from_string(datetime_string: str) -> int:
  with_utc_offset = datetime_string + '+0000'
  dt = datetime.datetime.strptime(with_utc_offset, DATETIME_FORMAT + '%z')
  return int(dt.timestamp())


Array = np.ndarray | jax.Array


SEP = '|'
RCV_ID_COL = 'receiver_id'
PIERCE_S2_COL = 'pierce_s2'


EXPECTED_COLUMNS = [
    UTC_SEC_COL := 'utc_sec',
    PIERCE_LNG_COL := 'pierce_lng',
    PIERCE_LAT_COL := 'pierce_lat',
    SLANT_FACTOR_COL := 'slant_factor',
    HIGH_FREQ_COL := 'higher_frequency_hz',
    LOW_FREQ_COL := 'lower_frequency_hz',
    CONSTELLATION_COL := 'constellation',
    SOURCE_ID_COL := 'source_id',
    SV_COL := 'sv',
    COARSE_RCV_S2_ID := 'coarse_rcv_s2_id',
    STEC_COL := 'stec',
    STEC_STDDEV_COL := 'stec_stddev',
]


def _assert_one_nonzero_per_row(x: sps.spmatrix):
  nz_per_row = (x != 0).sum(1)
  if (nz_per_row != 1).any():
    raise ValueError(
        'Expected matrix to have exactly one nonzero entry per row.'
    )


@dataclasses.dataclass(frozen=True)
class Solver:
  """Solver for the equation `Mx = z` where `M` has a specific block structure.

  `M` must be a symmetric matrix of the form

  M = A B
      C D

  where `C = B.T`. It should also be the case that `D` is "easy" to invert, as
  the class requires the inverse `Di`.

  Attributes:
    A: the symmetric upper-left block of the matrix.
    B: the upper-right block of the matrix.
    Di: the inverse of the lower-right block of the matrix.
    beta: optional regularization value, added to the diagonal of the Schur
      complement when solving.
    schur_complement: `A - B @ Di @ B.T`.
    schur_factor: `sksparse.cholmod.cholesky(schur_complement)`.
    a: size of `A`.
    d: size of `D`.
  """

  A: sps.spmatrix
  B: sps.spmatrix
  Di: sps.spmatrix
  beta: float = 0

  @functools.cached_property
  def schur_complement(self) -> sps.spmatrix:
    return (self.A - self.B @ self.Di @ self.B.T).tocsc()

  @functools.cached_property
  def schur_factor(self) -> sksparse.cholmod.Factor:
    return sksparse.cholmod.cholesky(self.schur_complement, beta=self.beta)

  @property
  def a(self) -> int:
    return self.A.shape[0]

  @property
  def d(self) -> int:
    return self.Di.shape[0]

  def solve(self, z: Array) -> Tuple[Array, Array]:
    """Solves the equation `Mx = z`.

    Args:
      z: a vector with shape `[A.shape[0] + Di.shape[0]]`.

    Returns:
      Two vectors xA and xD with sizes `A.shape[0]` and `Di.shape[0]`
      respectively. Concatenated together, they form the solution `x` to
      `Mx = z`.
    """
    z0 = z[: self.a]
    z1 = z[self.a :]

    xA = self.schur_factor(z0 - self.B @ self.Di @ z1)
    xD = self.Di @ z1 - self.Di @ self.B.T @ xA
    return xA, xD

  def schur_inverse_diagonal(self, rng: np.random.RandomState, n: int) -> Array:
    """Approximates the diagonal of the inverse of the Schur complement.

    This function uses a generalization of Hutchinson trace estimation to
    approximate the diagonal of the inverse.

    Args:
      rng: numpy random state for generating probe vectors.
      n: the number of probe vectors used in the approximation.

    Returns:
      An unbiased estimate of the diagonal of `inv(M)`.
    """
    v = rng.randint(0, 2, size=[self.a, n]) * 2 - 1
    Av = self.schur_factor(v)
    return (v * Av).mean(1)

  @classmethod
  def from_RS(
      cls,
      R: sps.spmatrix,
      S: sps.spmatrix,
      R_regularizer: Array | None = None,
      S_regularizer: Array | None = None,
  ):
    """Constructs a solver used for WLS defined by `R` and `S`.

    The returned solver has `A = R.T @ R`, `B = R.T @ S` and
    `Di = inv(S.T @ S)`. We require that `S` has exactly one non-zero entry per
    row, so that `D` is diagonal with no zeros on the diagonal, making it
    trivial to invert.

    Args:
      R: a sparse matrix.
      S: a sparse matrix with exactly one nonzero entry per row.
      R_regularizer: an array of size `R.shape[1]` that will be added to the
        diagonal of `A = R.T @ R`. This applies regularization to the
        coefficients corresponding to `R`.
      S_regularizer: an array of size `S.shape[1]` that will be added to the
        diagonal of `D = S.T @ S`. This applies regularization to the
        coefficients corresponding to `s`.

    Returns:
      A solver with `A = R.T @ R`, `B = R.T @ S` and `Di = inv(S.T @ S)`.
    """
    _assert_one_nonzero_per_row(S)

    A = R.T @ R
    if R_regularizer is not None:
      A = A + sps.diags(R_regularizer)

    B = R.T @ S

    D = S.T @ S
    diag = D.diagonal()
    if S_regularizer is not None:
      diag = diag + S_regularizer
    assert D.data.size == diag.size, 'D is not diagonal.'
    Di = sps.diags(1 / diag)

    return Solver(A, B, Di)


def to_rcv_id(df: pd.DataFrame) -> pd.Series:
  return df[SOURCE_ID_COL] + SEP + df[CONSTELLATION_COL]


def from_rcv_id(rcv_id: Iterable[str]) -> pd.DataFrame:
  source_id, constellation = zip(*[id.split(SEP) for id in rcv_id])
  return pd.DataFrame(
      {SOURCE_ID_COL: source_id, CONSTELLATION_COL: constellation}
  )


def single_lng_lat_to_token_at_level(
    lng_deg: float, lat_deg: float, s2_level: int
) -> str:
  token = (
      s2.S2CellId.from_lat_lng(s2.S2LatLng.from_degrees(lat_deg, lng_deg))
      .parent_at_level(s2_level)
      .token()
  )
  if isinstance(token, str):
    return token
  return token.decode('utf-8')


def _assert_lng_lat_valid(lngs_deg: pd.Series, lats_deg: pd.Series):
  if (lngs_deg < -180).any() or (lngs_deg > 180).any():
    raise ValueError('lngs_deg must be between -180 and 180.')
  if (lats_deg < -90).any() or (lats_deg > 90).any():
    raise ValueError('lats_deg must be between -90 and 90.')
  if len(lngs_deg) != len(lats_deg):
    raise ValueError('lngs_deg and lats_deg must have the same length.')


def lng_lat_to_token_at_level(
    lngs_deg: pd.Series, lats_deg: pd.Series, s2_level: int
) -> Iterable[str]:
  _assert_lng_lat_valid(lngs_deg, lats_deg)
  return [
      single_lng_lat_to_token_at_level(*lnglat, s2_level=s2_level)
      for lnglat in tuple(zip(lngs_deg, lats_deg, strict=True))
  ]


def _sparse_placeholder_matrix(
    feature: pd.Series, data: Array | pd.Series | None
) -> Tuple[Array, sps.spmatrix]:
  values = np.array(sorted(feature.unique()))
  feature_cat = pd.api.types.CategoricalDtype(values, ordered=True)
  col = feature.astype(feature_cat).cat.codes.values
  row = np.arange(len(col))
  if data is None:
    data = np.ones_like(col)
  shape = (len(col), len(values))
  return values, sps.csr_array((data, (row, col)), shape=shape)


def _prepare_dataframe(
    df: pd.DataFrame,
    s2_level: int,
    min_measurements_per_cell: int = 1,
    min_measurements_per_rcv: int = 2,
) -> pd.DataFrame:
  """Adds receiver and satellite ID columns and filters measurements."""
  df = df[EXPECTED_COLUMNS]
  df = df.dropna().copy()

  if not (
      (df[HIGH_FREQ_COL] == L1_HZ).all() & (df[LOW_FREQ_COL] == L5_HZ).all()
  ):
    raise ValueError('All measurements should be L1 L5.')

  if not df[CONSTELLATION_COL].isin(['GPS', 'GALILEO']).all():
    raise ValueError('All measurements should be GPS or GALILEO.')

  df[RCV_ID_COL] = to_rcv_id(df)
  df[PIERCE_S2_COL] = lng_lat_to_token_at_level(
      df[PIERCE_LNG_COL], df[PIERCE_LAT_COL], s2_level
  )
  cell_count = df.groupby([PIERCE_S2_COL])[PIERCE_S2_COL].transform('count')
  measurements_per_cell_too_low = cell_count < min_measurements_per_cell
  df = df[~measurements_per_cell_too_low]
  receiver_count = df.groupby(RCV_ID_COL)[RCV_ID_COL].transform('count')
  measurements_per_rcv_too_low = receiver_count < min_measurements_per_rcv
  df = df[~measurements_per_rcv_too_low]
  return df


def _keep_constrained(meas_df):
  """Removes measurements for unconstrained receivers or pierce points."""
  df = meas_df[[PIERCE_S2_COL, RCV_ID_COL, COARSE_RCV_S2_ID]].drop_duplicates()

  d = df
  prev_len = len(d) + 1
  while len(d) < prev_len:
    prev_len = len(d)

    # Drop cells which only see one of our remaining locations.
    s2_counts = (
        d[[PIERCE_S2_COL, COARSE_RCV_S2_ID]]
        .drop_duplicates()
        .groupby(PIERCE_S2_COL)
        .size()
    )
    d = d[~d[PIERCE_S2_COL].isin(s2_counts[s2_counts <= 1].index)]

    # Drop receivers which only see one of our remaining cells.
    rcv_counts = (
        d[[PIERCE_S2_COL, RCV_ID_COL]]
        .drop_duplicates()
        .groupby(RCV_ID_COL)
        .size()
    )
    d = d[~d[RCV_ID_COL].isin(rcv_counts[rcv_counts <= 1].index)]

  good_s2 = set(d[PIERCE_S2_COL].unique())
  good_rcv = set(d[RCV_ID_COL].unique())
  # Picked good_s2 and good_rcv so that each of our cells sees at least two
  # locations and each of our receivers sees at least two of our s2 cells.
  # These are guaranteed to be constrained.

  # Now find everything connected to that set of cells and receivers.
  while True:
    new_good_s2 = set(
        df[df[RCV_ID_COL].isin(good_rcv)][PIERCE_S2_COL].unique()
    ) - set(good_s2)
    good_s2.update(new_good_s2)
    new_good_rcv = set(
        df[df[PIERCE_S2_COL].isin(good_s2)][RCV_ID_COL].unique()
    ) - set(good_rcv)
    good_rcv.update(new_good_rcv)
    if not new_good_s2 and not new_good_rcv:
      # No new cells
      break

  return meas_df[
      meas_df[PIERCE_S2_COL].isin(good_s2) & meas_df[RCV_ID_COL].isin(good_rcv)
  ].copy()


def _filter_high_stddev(
    s2_tokens: Array, vtec: Array, vtec_variance: Array, max_stddev: float
) -> Tuple[Array, Array, Array]:
  stddev_ok = np.sqrt(vtec_variance) <= max_stddev
  return s2_tokens[stddev_ok], vtec[stddev_ok], vtec_variance[stddev_ok]


def fit_ionosphere_from_measurements(
    df: pd.DataFrame,
    s2_level: int = IONOSPHERE_S2_LEVEL,
    max_stddev_for_export: float | None = 50.0**0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Solves for VTEC and receiver bias using an S2 basis.

  Uses a sparse blockwise solver and an unbiased estimator of the variance.

  Args:
    df: dataframe of measurements.
    s2_level: the level of S2 cells to use for the model basis.
    max_stddev_for_export: predictions with standard deviation larger than this
      will not be returned. Measured in TECU. None means no filtering.

  Returns:
    Two dataframes, the first containing estimated VTEC and the second
    containing receiver bias estimates.
  """
  df = _prepare_dataframe(df, s2_level)
  if df.empty:
    raise ValueError('No measurements remained for timestamp.')

  df = _keep_constrained(df)
  if df.empty:
    raise ValueError('After keep_constrained, no measurements remained.')

  prd_tecu = df[STEC_COL].values
  prd_stddev_tecu = df[STEC_STDDEV_COL].values
  slant_factor_values = df[SLANT_FACTOR_COL].values

  for name, col in (
      (STEC_COL, prd_tecu),
      (SLANT_FACTOR_COL, prd_stddev_tecu),
      ('slant_factor', slant_factor_values),
  ):
    if np.isnan(col).any():
      raise ValueError(f'{name} contained NaNs.')

  y = prd_tecu / prd_stddev_tecu

  # These are the non-zero values in the design matrix of the linear system.
  stec_matrix_values = slant_factor_values / prd_stddev_tecu
  rcv_dcb_values = 1 / prd_stddev_tecu

  s2_idx, stec_matrix = _sparse_placeholder_matrix(
      df[PIERCE_S2_COL], stec_matrix_values
  )

  rcv_idx, rcv_dcb_matrix = _sparse_placeholder_matrix(
      df[RCV_ID_COL], rcv_dcb_values
  )

  z0 = stec_matrix.T @ y
  z1 = rcv_dcb_matrix.T @ y
  z = np.concatenate([z0, z1])

  solver = Solver.from_RS(stec_matrix, rcv_dcb_matrix)

  x0, x1 = solver.solve(z)

  x0_variance = solver.schur_inverse_diagonal(np.random.RandomState(0), n=1000)

  num_cells = len(s2_idx)
  vtec = x0
  rcv_dcb_tecu = x1

  # Because the solver returns an unbiased estimate of variance, it is
  # possible to get negative variance.
  vtec_stddev = np.sqrt(np.maximum(x0_variance[:num_cells], 1e-3))

  vtec_df = pd.DataFrame(
      dict(
          pierce_s2_token=s2_idx,
          vtec=vtec,
          vtec_stddev=vtec_stddev,
      )
  )
  # The model tends to predict a few very large VTECs with very high
  # uncertainty. Here, we filter these out.
  if max_stddev_for_export is not None:
    stddev_ok = vtec_df.vtec_stddev <= max_stddev_for_export
    vtec_df = vtec_df[stddev_ok].copy()

  rcv_dcb_df = from_rcv_id(rcv_idx)
  rcv_dcb_df['rcv_dcb_tecu'] = rcv_dcb_tecu

  return vtec_df, rcv_dcb_df


def process_station_measurements(
    raw_station_measurements_df,
    satellites,
    high_code,
    low_code,
    bias_df,
    station_name,
    constellation_letter,
):
  # Limit to just the selected satellites.
  df = raw_station_measurements_df.query('sv in @satellites').copy()

  # Subtract two pseudorange measurements to get the difference.
  df['pseudorange_diff_m'] = df[high_code] - df[low_code]

  # Keep only the needed columns.
  df = df.reset_index()[['time', 'sv', 'pseudorange_diff_m']]

  # Look up the satellite biases and join with the measurements.
  satellite_biases = bias_df.query((
      'STATION__.isnull() and '
      'OBS1==@high_code and '
      'OBS2==@low_code and '
      'PRN in @satellites'
  ))
  satellite_biases = satellite_biases.set_index('PRN')
  satellite_biases.rename(
      columns={'__ESTIMATED_VALUE____': 'sat_bias_nanos'}, inplace=True
  )
  df = df.join(satellite_biases['sat_bias_nanos'], on='sv')

  # Look up the station receiver bias to join with the measurements.
  station_bias_row = bias_df.query((
      'STATION__==@station_name and '
      'OBS1==@high_code and '
      'OBS2==@low_code and '
      'PRN ==@constellation_letter'
  ))
  assert len(station_bias_row) == 1
  station_bias = station_bias_row['__ESTIMATED_VALUE____'].iloc[0]
  df['rcv_bias_nanos'] = station_bias

  # Calculate the de-biased pseudorange difference.
  df['bias_m'] = (
      (df['rcv_bias_nanos'] + df['sat_bias_nanos'])
      / NS_PER_S
      * SPEED_OF_LIGHT_M_PER_S
  )
  df['debiased_pseudorange_diff_m'] = df['pseudorange_diff_m'] - df['bias_m']
  # Calculate the measured Slant Total Electron Content (STEC).
  df['debiased_stec'] = df['debiased_pseudorange_diff_m'] / (
      ALPHA * (1 / float(L1_HZ) ** 2 - 1 / float(L5_HZ) ** 2)
  )
  return df


# Scale factor to get the right number of receivers per minute.
# I looked at the number of receivers in one minute of real data in an S2 cell
# in London at noon vs how many the code would have generated and use this to
# scale the number of receivers appropriately. The local hour scale factor is
# 1.0 at noon UTC.
_PER_MINUTE_SCALE = 0.016

# The number of measurements varies based on local time of day.
_LOCAL_HOUR_SCALE = {
    0: 0.0,
    1: 0.0,
    2: 0.0,
    3: 0.0,
    4: 0.0,
    5: 0.06,
    6: 0.38,
    7: 1.01,
    8: 1.12,
    9: 0.61,
    10: 0.49,
    11: 0.67,
    12: 1.0,
    13: 0.9,
    14: 0.46,
    15: 0.36,
    16: 0.37,
    17: 0.55,
    18: 0.56,
    19: 0.43,
    20: 0.23,
    21: 0.14,
    22: 0.07,
    23: 0.0,
}


def to_local_hour(rcv_lng, hour):
  """Gets local hour from longitude and UTC hour."""
  return ((rcv_lng * 12.0 / 180.0).round() + hour).astype(int) % 24


def generate_phones_at_locations(
    utc_sec, population_density_df, dcb_clusters_df, rng: np.random.Generator
) -> pd.DataFrame:
  """Returns a dataframe of phone locations for the given time."""
  # Load the global daily count of receivers.
  rcv_count_df = population_density_df.copy()
  cells = [s2.S2CellId.from_token(x) for x in rcv_count_df['token']]
  rcv_count_df['coarse_rcv_s2_id'] = [c.id() for c in cells]
  ll = [c.to_lat_lng() for c in cells]
  rcv_count_df['rcv_lng'] = [lat_lng.lng().degrees for lat_lng in ll]
  rcv_count_df['rcv_lat'] = [lat_lng.lat().degrees for lat_lng in ll]

  # Scale the expected number receiver count based on the time of day.
  dt = datetime.datetime.fromtimestamp(utc_sec, tz=datetime.timezone.utc)
  hour = dt.hour
  local_hour = to_local_hour(rcv_count_df['rcv_lng'], hour)
  local_hour_scale = local_hour.map(_LOCAL_HOUR_SCALE)
  rcv_count_df['num_rcv'] = (
      rcv_count_df['population_per_km2'] * local_hour_scale * _PER_MINUTE_SCALE
  )
  rcv_count_df = rcv_count_df.drop(columns=['population_per_km2', 'token'])

  # Sample an actual number of receivers for each location.
  rcv_count_df['num_rcv'] = rng.poisson(rcv_count_df['num_rcv'])

  # Repeat each row based on the number of receivers.
  rcv_count_df = rcv_count_df.reset_index(drop=True)
  df = rcv_count_df.reindex(rcv_count_df.index.repeat(rcv_count_df.num_rcv))
  df = df.reset_index(drop=True)
  # Put the time in the source_id so it does not clash with a later one.
  df['source_id'] = ('@%d_' % utc_sec) + df.index.astype(str)
  df = df.drop(columns=['num_rcv'])

  # Set the receiver DCB cluster while we have one row per receiver.
  df['rcv_dcb_cluster'] = rng.integers(1, 5, len(df))
  df = pd.merge(df, dcb_clusters_df, on='rcv_dcb_cluster')
  df['GPS_rcv_dcb_tecu_true'] = rng.normal(
      df['GPS_rcv_dcb_tecu_mean'], df['GPS_rcv_dcb_tecu_stddev']
  )
  df['GALILEO_rcv_dcb_tecu_true'] = rng.normal(
      df['GALILEO_rcv_dcb_tecu_mean'], df['GALILEO_rcv_dcb_tecu_stddev']
  )

  return df


# Minimum elevation angle of the satellite above the horizon.
_MIN_ELEVATION_RAD = np.deg2rad(5)

Array = Union[np.ndarray, jax.Array]
Scalar = Union[Array, float]


class PrecisionWarning(Warning):
  pass


def require_float64(func: Callable) -> Callable:
  """A decorator which makes a function complain if float64s aren't enabled."""

  @functools.wraps(func)
  def wrapped(*args, **kwargs):
    if not jax.config.jax_enable_x64:
      warnings.warn(
          (
              'jax.config.jax_enable_x64 is False\n'
              f"It's recommended you run {func.__name__} with 64 bit floats by "
              'adding this to your binary or notebook:\n'
              'jax.config.update("jax_enable_x64", True)\n\n'
              "If you're confident 32 bit floats are safe, wrap your code in "
              'this context manager:\n'
              'import warnings\n'
              'with warnings.catch_warnings(category=PrecisionWarning):\n'
              "  warnings.simplefilter('ignore', PrecisionWarning)\n"
              '  # your code here'
          ),
          PrecisionWarning,
      )
    return func(*args, **kwargs)

  return wrapped


def xyz_cols(prefix: str) -> list[str]:
  return [f'{prefix}_{x}' for x in ['x', 'y', 'z']]


def normalized(x: Array, axis: Optional[int] = -1) -> Array:
  """Normalizes along the given axis."""
  return x / jnp.linalg.norm(x, axis=axis, keepdims=True)


def _inner(x: Array, y: Array) -> Array:
  return jnp.sum(x * y, -1)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class Ellipsoid:
  """A class for describing ellipsoids.

  Distances are expressed in meters.

  Attributes:
    semimajor_axis: the distance from the ellipsoid's center of mass to the
      'equator,' where latitude is zero.
    semiminor_axis: the distance from the ellipsoid's center of mass to the
      'north pole,' where latitude is 90 degrees.
  """

  semimajor_axis: Scalar
  semiminor_axis: Scalar

  @property
  def linear_eccentricity(self) -> Array:
    """See https://en.wikipedia.org/wiki/Eccentricity_(mathematics)."""
    return jnp.sqrt(self.semimajor_axis**2 - self.semiminor_axis**2)

  @property
  def eccentricity(self) -> Array:
    """See https://en.wikipedia.org/wiki/Eccentricity_(mathematics)."""
    return self.linear_eccentricity / self.semimajor_axis

  @property
  def a(self) -> Scalar:
    """Alias for `semimajor_axis`."""
    return self.semimajor_axis

  @property
  def b(self) -> Scalar:
    """Alias for `semiminor_axis`."""
    return self.semiminor_axis

  @property
  def c(self) -> Array:
    """Alias for `linear_eccentricity`."""
    return self.linear_eccentricity

  @property
  def axes(self) -> Array:
    """Equivalent to `jnp.array([self.a, self.a, self.b])`."""
    return jnp.array([self.a, self.a, self.b])

  def normal_at(self, xyz: Array) -> Array:
    """Computes the normal to the ellipsoid at `xyz`.

    Args:
      xyz: a point on the ellipsoid at which to compute the normal. Note that
        there is no enforcement of `xyz` being a point on the ellipsoid.

    Returns:
      The unit normal to the ellipsoid at `xyz`.
    """
    return normalized(xyz / self.axes**2, axis=-1)

  def offset(
      self, semimajor_offset: Scalar, semiminor_offset: Optional[Scalar] = None
  ):
    """Returns `Ellipsoid(a + semimajor_offset, b + semiminor_offset)`.

    Args:
      semimajor_offset: the amount to offset the semimajor axis of the new
        `Ellipsoid`.
      semiminor_offset: the amount to offset the semiminor axis of the new
        `Ellipsoid`. If not specified, `semimajor_offset` is used for both.

    Returns:
      A new ellipsoid with axes `(a + semimajor_offset, b + semiminor_offset)`.
    """
    if semiminor_offset is None:
      semiminor_offset = semimajor_offset
    return self.__class__(
        semimajor_axis=self.a + semimajor_offset,
        semiminor_axis=self.b + semiminor_offset,
    )

  def prime_vertical_radius(self, lat: Array) -> Array:
    """The prime vertical radius of curvature at `lat`.

    See https://en.wikipedia.org/wiki/Geodetic_coordinates#Conversion.

    Args:
      lat: the latitude at which to compute the prime vertical radius of the
        ellipse.

    Returns:
      The length of the prime vertical radius at `lat`.
    """
    lat_radians = jnp.deg2rad(lat)
    return self.a / jnp.sqrt(
        1 - (self.eccentricity * jnp.sin(lat_radians)) ** 2
    )

  @classmethod
  def wgs84(cls):
    """https://en.wikipedia.org/wiki/World_Geodetic_System#WGS84."""
    return cls(semimajor_axis=6_378_137.0, semiminor_axis=6_356_752.314_245_18)

  # Methods below for JAX pytree compatibility.

  def tree_flatten(self) -> Tuple[Tuple[Array, Array], None]:
    return dataclasses.astuple(self), None

  @classmethod
  def tree_unflatten(cls, _, fields: Tuple[Array, Array]):
    return cls(*fields)


@require_float64
def lng_lat_elevation_to_xyz(
    lng: Scalar,
    lat: Scalar,
    elevation: Scalar,
    ellipsoid: Ellipsoid,
) -> Array:
  """Converts geodetic coordinates to ECEF coordinates.

  See https://en.wikipedia.org/wiki/Geodetic_coordinates#Conversion.

  Args:
    lng: the longitude, in degrees.
    lat: the latitude, in degrees.
    elevation: the eleveation above the surface of `Ellipsoid`, in meters.
    ellipsoid: the reference ellipsoid from which `elevation` is calculated.

  Returns:
    An array of shape `lng.shape + [3]` representing the same point(s) in
    geocentric Cartesian coordinates.
  """
  n = ellipsoid.prime_vertical_radius(lat)

  lng_radians = jnp.deg2rad(lng)
  lat_radians = jnp.deg2rad(lat)

  equatorial_radius = (n + elevation) * jnp.cos(lat_radians)

  # https://en.wikipedia.org/wiki/Geodetic_coordinates#Conversion
  x = equatorial_radius * jnp.cos(lng_radians)
  y = equatorial_radius * jnp.sin(lng_radians)
  z = (
      n * (ellipsoid.semiminor_axis / ellipsoid.semimajor_axis) ** 2 + elevation
  ) * jnp.sin(lat_radians)
  return jnp.stack((x, y, z), axis=-1)


@require_float64
def elevation_and_azimuth_radians(
    rcv_xyz: Array, sat_xyz: Array, ellipsoid: Ellipsoid = Ellipsoid.wgs84()
) -> Tuple[Array, Array]:
  """Elevation angle and azimuth of the satellite."""
  unit_to_sat = normalized(sat_xyz - rcv_xyz)
  unit_zenith = ellipsoid.normal_at(rcv_xyz)
  # Clip in case the dot product exceeds 1.0 because of floating point nonsense.
  cos_zenith_angle = jnp.clip(_inner(unit_zenith, unit_to_sat), -1, 1)
  elevation_radians = jnp.arcsin(cos_zenith_angle)
  unit_east = normalized(jnp.cross(jnp.array([0, 0, 1]), unit_zenith))
  unit_north = normalized(jnp.cross(unit_zenith, unit_east))
  azimuth_radians = jnp.arctan2(
      _inner(unit_to_sat, unit_east), _inner(unit_to_sat, unit_north)
  )
  return elevation_radians, azimuth_radians


@require_float64
def all_pierce_points(
    x: Array, y: Array, ellipsoid: Ellipsoid
) -> Tuple[Array, Array, Array, Array]:
  """Computes intersection points of the line defined by `x, y` with `ellipse`.

  Args:
    x: an array of shape `(..., 3)`, generally indicating receiver positions.
    y: an array that is broadcastable to the shape of `x`, generally indicating
      satellite positions.
    ellipsoid: the ellipsoid at which the pierce point will be computed.

  Returns:
    x_pierce_point: the pierce point closer to `x` than `y`. Returns `nan` if
    the line defined by `x, y` does not intersect `ellipsoid`.
    y_pierce_point: the pierce point closer to `y` than `x`. Returns `nan` if
      the line defined by `x, y` does not intersect `ellipsoid`.
    x_pierce_point_valid: a boolean array of shape `x.shape[:-1]` indicating
      whether `x_pierce_point` lies between `x` and `y`.
    y_pierce_point_valid: a boolean array of shape `x.shape[:-1]` indicating
      whether `x_pierce_point` lies between `x` and `y`.
  """
  x, y = jnp.broadcast_arrays(x, y)
  scaled_satellite = y / ellipsoid.axes
  scaled_receiver = x / ellipsoid.axes
  diff = scaled_satellite - scaled_receiver
  # Quadratic `a * t**2 + b * t + c`` with
  # `a = diff @ diff``
  # `b = 2 * rcv @ diff``
  # `c = rcv @ rcv - 1``
  a = _inner(diff, diff)
  b = 2 * _inner(scaled_receiver, diff)
  c = _inner(scaled_receiver, scaled_receiver) - 1
  discriminant = jnp.sqrt(b**2 - 4 * a * c)

  x_solution = (-b - discriminant) / 2 / a
  x_pierce_point_valid = (
      ~jnp.isnan(x_solution) & (x_solution >= 0) & (x_solution <= 1)
  )
  x_solution = x_solution[..., jnp.newaxis]
  x_pierce_point = x_solution * y + (1 - x_solution) * x

  y_solution = (-b + discriminant) / 2 / a
  y_pierce_point_valid = (
      ~jnp.isnan(y_solution) & (y_solution >= 0) & (y_solution <= 1)
  )
  y_solution = y_solution[..., jnp.newaxis]
  y_pierce_point = y_solution * y + (1 - y_solution) * x
  return (
      x_pierce_point,
      y_pierce_point,
      x_pierce_point_valid,
      y_pierce_point_valid,
  )


def pierce_point(x: Array, y: Array, ellipsoid: Ellipsoid) -> jnp.ndarray:
  """Pierce point between two points.

  Args:
    x: an array of shape `(..., 3)`, generally indicating receiver positions.
    y: an array that is broadcastable to the shape of `x`, generally indicating
      satellite positions.
    ellipsoid: the ellipsoid at which the pierce point will be computed.

  Returns:
    An array representing the point where the line segment from `x` to `y`
    intersects the `ellipsoid`. If the line segment intersects the ellipsoid
    twice, the pierce point closer to `y` is returned. If the line segment does
    not intersect the ellipse, `nan` is returned.
  """
  (
      x_pierce_point,
      y_pierce_point,
      x_pierce_point_valid,
      y_pierce_point_valid,
  ) = all_pierce_points(x, y, ellipsoid)
  nan = jnp.nan * x_pierce_point
  return jnp.where(
      y_pierce_point_valid[..., jnp.newaxis],
      y_pierce_point,
      jnp.where(x_pierce_point_valid[..., jnp.newaxis], x_pierce_point, nan),
  )


# We always want to perform `einsum` with maximum precision.
einsum = functools.partial(jnp.einsum, precision=jax.lax.Precision.HIGHEST)


@require_float64
def slant_factor(
    pierce_position: jnp.ndarray,
    satellite_position: jnp.ndarray,
    ellipsoid: jnp.ndarray,
) -> jnp.ndarray:
  """STEC/VTEC for the ray from the satellite through the pierce position.

  Args:
    pierce_position: position of the pierce point through the ionosphere. The
      slant factor is based on the angle from vertical at this location.
    satellite_position: must be broadcastable with pierce_position.
    ellipsoid: the ellipsoid that is being pierced. Note that `pierce_position`
      should lie on this ellipse, though this requirement is not enforced.

  Returns:
    An array of ratios STEC/VTEC for a ray passing through the ionosphere at
    pierce_position from a satellite at satellite_position.
  """
  normal_at_pierce = ellipsoid.normal_at(pierce_position)
  normalized_pierce_to_satellite = normalized(
      satellite_position - pierce_position, -1
  )
  cos_theta = einsum(
      '...a,...a->...', normal_at_pierce, normalized_pierce_to_satellite
  )
  return 1 / cos_theta


@require_float64
def xyz_to_lng_lat_elevation(
    xyz: Array, ellipsoid: Ellipsoid
) -> Tuple[Array, Array, Array]:
  """Converts ECEF coordinates to geodetic coordinates.

  See https://en.wikipedia.org/wiki/Geodetic_coordinates#Conversion.

  Implementation based on

  http://google3/third_party/py/pymap3d/ecef.py;l=99;rcl=363470010

  and in turn on the approximation described in

  "Transformation of Cartesian to geodetic coordinates without iterations."
  Journal of Surveying Engineering 126, no. 1 (2000): 1-7.

  Note that this approximation is bad when the eccentricity of the ellipse is
  very large. However, it provides centimeter-level accuracy for the WGS84
  ellipse and altitudes relevant to the ionosphere.

  Args:
    xyz: an array with shape `[..., 3]` representing as set of points in
      geocentric Cartesian coordinates.
    ellipsoid: the `Ellipsoid` for which the conversion is performed. It is
      recommended that the eccentricity of the ellipsoid does not exceed 0.8
      when using this method of coordinate conversion.

  Returns:
    Three arrays, (lng, lat, elevation), representing the same points in
    geodetic coordinates.
  """
  x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

  r = jnp.sqrt(jnp.square(xyz).sum(-1))
  e = jnp.sqrt(ellipsoid.semimajor_axis**2 - ellipsoid.semiminor_axis**2)

  r2 = r**2
  e2 = e**2

  u = jnp.sqrt((r2 - e2) / 2 + jnp.sqrt((r2 - e2) ** 2 + 4 * e2 * z**2) / 2)
  q = jnp.hypot(x, y)
  hue = jnp.hypot(u, e)

  beta = jnp.where(
      q > 0,
      jnp.arctan(hue / u * z / q),
      jnp.where(z >= 0, jnp.pi / 2, -jnp.pi / 2),
  )

  eps = (
      (ellipsoid.semiminor_axis * u - ellipsoid.semimajor_axis * hue + e**2)
      * jnp.sin(beta)
      / (
          ellipsoid.semimajor_axis * hue * 1 / jnp.cos(beta)
          - e**2 * jnp.cos(beta)
      )
  )

  beta = beta + eps

  lat_radians = jnp.arctan(
      ellipsoid.semimajor_axis / ellipsoid.semiminor_axis * jnp.tan(beta)
  )
  lng_radians = jnp.arctan2(y, x)

  elevation = jnp.hypot(
      z - ellipsoid.semiminor_axis * jnp.sin(beta),
      q - ellipsoid.semimajor_axis * jnp.cos(beta),
  )
  inside = 1 > (
      (x / ellipsoid.semimajor_axis) ** 2
      + (y / ellipsoid.semimajor_axis) ** 2
      + (z / ellipsoid.semiminor_axis) ** 2
  )
  elevation = jnp.where(inside, -elevation, elevation)

  return jnp.rad2deg(lng_radians), jnp.rad2deg(lat_radians), elevation


# Thin shell located at an offset from the WGS84 ellipsoid.
IONOSPHERE_OFFSET_M = 350_000
IONOSPHERE_ELLIPSOID = Ellipsoid.wgs84().offset(IONOSPHERE_OFFSET_M)


def get_los_df(sat_xyz_df, coarse_rcv_s2_id) -> pd.DataFrame:
  """Makes a dataframe of lines-of-sight from satellites to receivers."""
  rcv_df = (
      pd.DataFrame(
          dict(
              coarse_rcv_s2_id=coarse_rcv_s2_id,
          )
      )
      .drop_duplicates()
      .copy()
  )
  cells = [s2.S2CellId(int(x)) for x in rcv_df.coarse_rcv_s2_id]
  ll = [c.to_lat_lng() for c in cells]
  rcv_lng = np.array([lat_lng.lng().degrees for lat_lng in ll])
  rcv_lat = np.array([lat_lng.lat().degrees for lat_lng in ll])
  rcv_df[xyz_cols('rcv')] = lng_lat_elevation_to_xyz(
      rcv_lng, rcv_lat, 0, Ellipsoid.wgs84())
  df = pd.merge(
      sat_xyz_df.reset_index().assign(join_key=0),
      rcv_df.assign(join_key=0),
      on='join_key',
      how='outer',
  ).drop(columns='join_key')
  obs_elev_rad, _ = elevation_and_azimuth_radians(
      df[xyz_cols('rcv')].values, df[xyz_cols('sat')].values
  )
  df['elev_angle_deg'] = np.rad2deg(obs_elev_rad)
  sat_xyz = df[xyz_cols('sat')].values
  df['sat_lng'], df['sat_lat'], _ = xyz_to_lng_lat_elevation(
      sat_xyz, Ellipsoid.wgs84())
  visible_mask = np.array(obs_elev_rad > _MIN_ELEVATION_RAD)
  df = df[visible_mask].copy()
  return df


def set_pierce_point_cols(los_df):
  los_df = los_df.copy()
  pierce_xyz = pierce_point(
      los_df[xyz_cols('rcv')].values,
      los_df[xyz_cols('sat')].values,
      IONOSPHERE_ELLIPSOID,
  )
  pierce_lng, pierce_lat, _ = xyz_to_lng_lat_elevation(
      pierce_xyz, IONOSPHERE_ELLIPSOID
  )
  slant_factor_values = slant_factor(
      pierce_xyz, los_df[xyz_cols('sat')].values, IONOSPHERE_ELLIPSOID
  )
  los_df['pierce_lng'] = pierce_lng
  los_df['pierce_lat'] = pierce_lat
  los_df['slant_factor'] = slant_factor_values
  return los_df


def set_stec_true(vtec_df, los_df, s2_level=IONOSPHERE_S2_LEVEL):
  vtec_df = vtec_df.rename(columns={'vtec': 'vtec_true'})
  vtec_df = vtec_df.set_index('pierce_s2_token', verify_integrity=True)
  vtec_df = vtec_df[['vtec_true']]

  los_df = los_df.copy()
  los_df['pierce_s2_token'] = lng_lat_to_token_at_level(
      los_df.pierce_lng, los_df.pierce_lat, s2_level=s2_level
  )
  los_df = pd.merge(los_df, vtec_df, on='pierce_s2_token', how='inner')
  los_df['stec_true'] = los_df['vtec_true'] * los_df['slant_factor']
  los_df = los_df.dropna(subset=['stec_true'])
  return los_df


# Standard deviation of the noise in the STEC measurement.
# 70 TECU for each measurement but we average 60 samples in a minute.
_STEC_NOISE_STDDEV_TECU = 70.0 / np.sqrt(60)


def set_measurement_values(
    meas_df: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
  """Sets the measurement values in the dataframe."""
  meas_df['higher_frequency_hz'] = L1_HZ
  meas_df['lower_frequency_hz'] = L5_HZ

  constellation_letter = meas_df['sv'].str.slice(0, 1)
  meas_df['constellation'] = constellation_letter.map(
      dict(G='GPS', E='GALILEO')
  )

  meas_df['stec_noise_true'] = rng.normal(
      0, _STEC_NOISE_STDDEV_TECU, len(meas_df)
  )

  # The receiver DCB is different for GPS and Galileo.
  is_gps = meas_df['constellation'] == 'GPS'
  meas_df['rcv_dcb_tecu_true'] = np.where(
      is_gps,
      meas_df['GPS_rcv_dcb_tecu_true'],
      meas_df['GALILEO_rcv_dcb_tecu_true'],
  )
  rcv_dcb_stddev_tecu = np.where(
      is_gps,
      meas_df['GPS_rcv_dcb_tecu_stddev'],
      meas_df['GALILEO_rcv_dcb_tecu_stddev'],
  )
  meas_df = meas_df.drop(
      columns=[
          'GPS_rcv_dcb_tecu_mean',
          'GPS_rcv_dcb_tecu_stddev',
          'GPS_rcv_dcb_tecu_true',
          'GALILEO_rcv_dcb_tecu_mean',
          'GALILEO_rcv_dcb_tecu_stddev',
          'GALILEO_rcv_dcb_tecu_true',
      ]
  )

  meas_df['stec_stddev'] = np.hypot(
      _STEC_NOISE_STDDEV_TECU, rcv_dcb_stddev_tecu
  )

  meas_df['stec'] = (
      meas_df['stec_true']
      + meas_df['stec_noise_true']
      + meas_df['rcv_dcb_tecu_true']
  )
  return meas_df


def plot_line_of_sight(fig, one_location_meas_df):
  # Interleave the receiver location with each satellite location so the
  # line-of-sight line goes up and down between them.
  line_of_sight_df = pd.concat([
      one_location_meas_df[xyz_cols('sat')]
      .rename(columns={'sat_x': 'x', 'sat_y': 'y', 'sat_z': 'z'})
      .reset_index(drop=True),
      one_location_meas_df[xyz_cols('rcv')]
      .rename(columns={'rcv_x': 'x', 'rcv_y': 'y', 'rcv_z': 'z'})
      .reset_index(drop=True),
  ])
  line_of_sight_df = line_of_sight_df.sort_index(kind='mergesort').reset_index(
      drop=True
  )

  fig.add_trace(
      go.Scatter3d(
          x=line_of_sight_df.x,
          y=line_of_sight_df.y,
          z=line_of_sight_df.z,
          name='Lines of Sight',
          mode='lines',
          line=dict(color='rgb(0,0,0)'),
          hoverinfo='skip',
      )
  )


EARTH_EQUATORIAL_RADIUS_M = 6_378_137.0


def plot_sphere_back(fig):
  clor = 'rgb(255, 255, 255)'
  # Use sperical earth for 3d interactive plot.
  R = np.sqrt(EARTH_EQUATORIAL_RADIUS_M)
  u_angle = np.linspace(0, np.pi, 25)
  v_angle = np.linspace(0, np.pi, 25)
  x_dir = np.outer(R * np.cos(u_angle), R * np.sin(v_angle))
  y_dir = np.outer(R * np.sin(u_angle), R * np.sin(v_angle))
  z_dir = np.outer(R * np.ones(u_angle.shape[0]), R * np.cos(v_angle))
  fig.add_surface(
      z=z_dir,
      x=x_dir,
      y=y_dir,
      colorscale=[[0, clor], [1, clor]],
      opacity=1.0,
      showlegend=False,
      hoverinfo='skip',
      lighting=dict(diffuse=0.1),
      showscale=False,
  )


def plot_sphere_front(fig):
  clor = 'rgb(255, 255, 255)'
  # Use sperical earth for 3d interactive plot.
  R = np.sqrt(EARTH_EQUATORIAL_RADIUS_M)
  u_angle = np.linspace(-np.pi, 0, 25)
  v_angle = np.linspace(0, np.pi, 25)
  x_dir = np.outer(R * np.cos(u_angle), R * np.sin(v_angle))
  y_dir = np.outer(R * np.sin(u_angle), R * np.sin(v_angle))
  z_dir = np.outer(R * np.ones(u_angle.shape[0]), R * np.cos(v_angle))
  fig.add_surface(
      z=z_dir,
      x=x_dir,
      y=y_dir,
      colorscale=[[0, clor], [1, clor]],
      opacity=1.0,
      showlegend=False,
      hoverinfo='skip',
      lighting=dict(diffuse=0.1),
      showscale=False,
  )


def plot_linestring(poly):
  xy_coords = poly.coords.xy
  lng_rad = np.deg2rad(np.array(xy_coords[0]))
  lat_rad = np.deg2rad(np.array(xy_coords[1]))
  # Use sperical earth for 3d interactive plot.
  x = EARTH_EQUATORIAL_RADIUS_M * np.cos(lat_rad) * np.cos(lng_rad)
  y = EARTH_EQUATORIAL_RADIUS_M * np.cos(lat_rad) * np.sin(lng_rad)
  z = EARTH_EQUATORIAL_RADIUS_M * np.sin(lat_rad)
  return x, y, z


def plot_coastlines(fig):
  for polys in cartopy.feature.COASTLINE.geometries():
    x, y, z = plot_linestring(polys)
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='lines',
            line=dict(color='rgb(0,0,0)'),
            showlegend=False,
            hoverinfo='skip',
        )
    )


def plot_satellite_positions(fig, sat_xyz_df):
  fig.add_trace(
      go.Scatter3d(
          x=sat_xyz_df.sat_x,
          y=sat_xyz_df.sat_y,
          z=sat_xyz_df.sat_z,
          name='Satellite Positions',
          mode='markers',
          text=sat_xyz_df.index,
          hovertemplate='<b>Satellite %{text}</b><extra></extra>',
          marker_symbol='diamond',
          marker_line_color='midnightblue',
          marker_color='lightskyblue',
          marker_line_width=2,
      )
  )


def plot_ionosphere_vtec_3d(fig, vtec_df):
  xyz_vtec_df = vtec_df.copy()
  cells = [s2.S2CellId.from_token(x) for x in xyz_vtec_df.pierce_s2_token]
  ll = [x.to_lat_lng() for x in cells]
  xyz_vtec_df['pierce_lat'] = [lat_lng.lat().degrees for lat_lng in ll]
  xyz_vtec_df['pierce_lng'] = [lat_lng.lng().degrees for lat_lng in ll]
  xyz_vtec_df[xyz_cols('pierce')] = lng_lat_elevation_to_xyz(
      xyz_vtec_df.pierce_lng.values,
      xyz_vtec_df.pierce_lat.values,
      0,
      ellipsoid=IONOSPHERE_ELLIPSOID,
  )
  fig.add_trace(
      go.Scatter3d(
          x=xyz_vtec_df.pierce_x,
          y=xyz_vtec_df.pierce_y,
          z=xyz_vtec_df.pierce_z,
          text=xyz_vtec_df.vtec,
          hovertemplate='VTEC: %{text:.2f}<extra></extra>',
          mode='markers',
          showlegend=True,
          name='Ionosphere VTEC',
          marker=dict(
              size=5,
              color=xyz_vtec_df.vtec,
              colorscale='Plasma',
              cmin=0,
              cmax=90,
              opacity=0.2,
              colorbar=dict(title='VTEC'),
          ),
      )
  )
