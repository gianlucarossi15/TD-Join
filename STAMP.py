# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import numpy as np
from datetime import timedelta
from dateutil.relativedelta import relativedelta

from stumpy import core


def _mass_PI(
        Q,
        T,
        M_T,
        Σ_T,
        trivial_idx=None,
        excl_zone=0,
        left=False,
        right=False,
        T_subseq_isconstant=None,
        Q_subseq_isconstant=None,
):
    """
    Compute "Mueen's Algorithm for Similarity Search" (MASS)

    Parameters
    ----------
    Q : numpy.ndarray
        Query array or subsequence

    T : numpy.ndarray
        Time series array or sequence

    M_T : numpy.ndarray
        Sliding mean for `T`

    Σ_T : numpy.ndarray
        Sliding standard deviation for `T`

    T_subseq_isconstant : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T` is constant (True)

    trivial_idx : int, default None
        Index for the start of the trivial self-join

    excl_zone : int, default 0
        The half width for the exclusion zone relative to the `trivial_idx`.
        If the `trivial_idx` is `None` then this parameter is ignored.

    left : bool, default False
        Return the left matrix profile indices if `True`. If `right` is True
        then this parameter is ignored.

    right : bool, default False
        Return the right matrix profiles indices if `True`

    T_subseq_isconstant : numpy.ndarray or function, default None
        A boolean array that indicates whether a subsequence in `T` is constant
        (True). Alternatively, a custom, user-defined function that returns a
        boolean array that indicates whether a subsequence in `T` is constant
        (True). The function must only take two arguments, `a`, a 1-D array,
        and `w`, the window size, while additional arguments may be specified
        by currying the user-defined function using `functools.partial`. Any
        subsequence with at least one np.nan/np.inf will automatically have its
        corresponding value set to False in this boolean array.

    Q_subseq_isconstant : numpy.ndarray or function, default None
        A boolean array that indicates whether the subsequence in `Q` is constant
        (True). Alternatively, a custom, user-defined function that returns a
        boolean array that indicates whether a subsequence in `Q` is constant
        (True). The function must only take two arguments, `a`, a 1-D array,
        and `w`, the window size, while additional arguments may be specified
        by currying the user-defined function using `functools.partial`. Any
        subsequence with at least one np.nan/np.inf will automatically have its
        corresponding value set to False in this boolean array.

    Returns
    -------
    P : numpy.ndarray
        Matrix profile

    I : numpy.ndarray
        Matrix profile indices
    """
    # Distance profile for subsequence Q with every subsequence in T
    D = core.mass(
        Q,
        T,
        M_T,
        Σ_T,
        T_subseq_isconstant=T_subseq_isconstant,
        Q_subseq_isconstant=Q_subseq_isconstant,
    )

    if trivial_idx is not None:
        core.apply_exclusion_zone(D, trivial_idx, excl_zone, np.inf)

        # Get left and right matrix profiles
        IL = -1
        PL = np.inf
        if D[:trivial_idx].size:
            IL = np.argmin(D[:trivial_idx])
            PL = D[IL]
        if PL == np.inf:
            IL = -1

        IR = -1
        PR = np.inf
        if D[trivial_idx + 1 :].size:
            IR = trivial_idx + 1 + np.argmin(D[trivial_idx + 1 :])
            PR = D[IR]
        if PR == np.inf:
            IR = -1

    # Element-wise Min
    I = np.argmin(D)
    P = D[I]
    if P == np.inf:
        I = -1

    if trivial_idx is not None and left:
        I = IL
        P = PL

    if trivial_idx is not None and right:
        I = IR
        P = PR

    return P, I


def stamp(
        T_A,
        T_B,
        m,
        Allen_relation=None,
        ignore_trivial=False,
        T_A_subseq_isconstant=None,
        T_B_subseq_isconstant=None,
):
    """
    Compute matrix profile and indices using the "Scalable Time series
    Anytime Matrix Profile" (STAMP) algorithm and MASS (2017 - with FFT).

    Parameters
    ----------
    T_A : numpy.ndarray
        The time series or sequence for which the matrix profile will be returned

    T_B : numpy.ndarray
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in T_A, its nearest neighbor in T_B will be recorded.

    m : int
        Window size

    ignore_trivial : bool, default False
        `True` if this is a self join and `False` otherwise (i.e., AB-join).

    T_A_subseq_isconstant : numpy.ndarray or function, default None
        A boolean array that indicates whether a subsequence in `T_A` is constant
        (True). Alternatively, a custom, user-defined function that returns a
        boolean array that indicates whether a subsequence in `T_A` is constant
        (True). The function must only take two arguments, `a`, a 1-D array,
        and `w`, the window size, while additional arguments may be specified
        by currying the user-defined function using `functools.partial`. Any
        subsequence with at least one np.nan/np.inf will automatically have its
        corresponding value set to False in this boolean array.

    T_B_subseq_isconstant : numpy.ndarray or function, default None
        A boolean array that indicates whether a subsequence in `T_B` is constant
        (True). Alternatively, a custom, user-defined function that returns a
        boolean array that indicates whether a subsequence in `T_B` is constant
        (True). The function must only take two arguments, `a`, a 1-D array,
        and `w`, the window size, while additional arguments may be specified
        by currying the user-defined function using `functools.partial`. Any
        subsequence with at least one np.nan/np.inf will automatically have its
        corresponding value set to False in this boolean array.

    Returns
    -------
    out : numpy.ndarray
        Two column numpy array where the first column is the matrix profile
        and the second column is the matrix profile indices

    Notes
    -----
    `DOI: 10.1109/ICDM.2016.0179 \
    <https://www.cs.ucr.edu/~eamonn/PID4481997_extend_Matrix%20Profile_I.pdf>`__

    See Table III

    Timeseries, T_A, will be annotated with the distance location
    (or index) of all its subsequences in another times series, T_B.

    For every subsequence, Q, in T_A, you will get a distance and index for
    the closest subsequence in T_B. Thus, the array returned will have length
    T_A.shape[0]-m+1
    """
    T_A_subseq_isconstant = core.rolling_isconstant(T_A, m, T_A_subseq_isconstant)
    T_B, M_T, Σ_T, T_B_subseq_isconstant = core.preprocess(
        T_B, m, T_subseq_isconstant=T_B_subseq_isconstant
    )

    T_A = T_A.copy()
    T_A[np.isinf(T_A)] = np.nan

    if T_A.ndim != 1:  # pragma: no cover
        raise ValueError(f"T_A is {T_A.ndim}-dimensional and must be 1-dimensional. ")

    if T_B.ndim != 1:  # pragma: no cover
        raise ValueError(f"T_B is {T_B.ndim}-dimensional and must be 1-dimensional. ")

    if Allen_relation is not None and Allen_relation not in {"Before","Meets","Equals","Overlaps"}:
        raise ValueError(f"{Allen_relation} is not a valid Allen's relation. Please choose from 'Before', 'Meets', 'Equals', or 'Overlaps'.")

    core.check_window_size(m, max_size=min(T_A.shape[0], T_B.shape[0]))

    subseq_T_A = core.rolling_window(T_A, m)
    excl_zone = int(np.ceil(m / 2))

    # Add exclusionary zone
    if ignore_trivial:
        out = [
            _mass_PI(
                subseq,
                T_B,
                M_T,
                Σ_T,
                i,
                excl_zone,
                T_subseq_isconstant=T_B_subseq_isconstant,
                Q_subseq_isconstant=T_A_subseq_isconstant[[i]],
            )
            for i, subseq in enumerate(subseq_T_A)
        ]
    else: # AB join
        out = [
            _mass_PI(
                subseq,
                T_B,
                M_T,
                Σ_T,
                T_subseq_isconstant=T_B_subseq_isconstant,
                Q_subseq_isconstant=T_A_subseq_isconstant[[i]],
            )
            for i, subseq in enumerate(subseq_T_A)
        ]
        # if Allen_relation == "Meets":
        #     for i, subseq in enumerate(subseq_T_A):
        #             if len(T_B) >= i+2*m:
        #                 print("sub T_A ",subseq, (T_B[i+m: i+2*m])," portion of T_B")
        #     out = [
        #         _mass_PI(
        #             subseq,
        #             T_B[i+m: i+2*m],
        #             M_T,
        #             Σ_T,
        #             T_subseq_isconstant=T_B_subseq_isconstant,
        #             Q_subseq_isconstant=T_A_subseq_isconstant[[i]],
        #         )
        #         for i, subseq in enumerate(subseq_T_A)
        #         if len(T_B) >= i+2*m
        #     ]

        # elif Allen_relation == "Before":
        # elif Allen_relation == "Equals":
        # elif Allen_relation == "Overlaps":
    out = np.array(out, dtype=object)

    return out

def z_normalized_euclidean_distance(a, b):
    a_z = (a - np.mean(a)) / np.std(a)
    b_z = (b - np.mean(b)) / np.std(b)
    return np.linalg.norm(a_z - b_z)

def rolling_window(array, window_size):
    """
    Create a rolling window view of the list of DataPoint objects.
    """
    n = len(array) - window_size + 1
    if n <= 0:
        raise ValueError("Window size must be smaller than or equal to the length of the array.")

    result = [array[i:i + window_size] for i in range(n)]
    return result



def get_timedelta(timestamp):
    """
    Determines the granularity of the timestamp based on trailing zeros.
    Returns the coarsest unit where all finer-grained components are zero.
    """
    if timestamp.second != 0:
        return timedelta(seconds=1)
    elif timestamp.minute != 0:
        return timedelta(minutes=1)
    elif timestamp.hour != 0:
        return timedelta(hours=1)
    elif timestamp.day != 1:
        return timedelta(days=1)
    elif timestamp.month != 1:
        return relativedelta(months=1)
    else:
        return relativedelta(years=1)