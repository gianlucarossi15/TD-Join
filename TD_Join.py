import warnings

import numpy as np

from stumpy import core
from datetime import timedelta, datetime


def TD_Join(T_A, T_B, m, Allen_relation=None):
    """
    Compute Time Dependent Matrix Profile TDMP.

    Parameters
    ----------
    T_A : list of DataPoint
        The time series or sequence for which the matrix profile will be returned

    T_B : list of DataPoint
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in T_A, its nearest neighbor in T_B will be recorded.

    m : int
        Window size


    Allen_relation : str, default None
        Allen's relation to be computed. Choose from 'before', 'meets', 'equal' or 'overlaps'.
        If None, the complete Allen profile will be computed.

    Returns
    -------
    out : dict
    A dictionary where keys are Allen's relations and values are lists of pairs [index, distance].
    """

    T_A = T_A.copy()
    for i in range(len(T_A)):
        if np.isinf(T_A[i].get_value()):
            T_A[i].set_value(np.nan)

    # if T_A.ndim != 2:  # pragma: no cover
    #     raise ValueError(f"T_A is {T_A.ndim}-dimensional and must be 2-dimensional. ")
    #
    # if T_B.ndim != 2:  # pragma: no cover
    #     raise ValueError(f"T_B is {T_B.ndim}-dimensional and must be 2-dimensional. ")

    if Allen_relation is not None and Allen_relation not in {"before","meets","equal","overlaps"}:
        raise ValueError(f"{Allen_relation} is not a valid Allen's relation. Please choose from 'before', 'meets', 'equal', or 'overlaps'.")

    core.check_window_size(m, max_size=min(len(T_A), len(T_B)))
    subseq_T_A = rolling_window(T_A, m)
    subseq_T_B = rolling_window(T_B,  m)


    dict = {}
    dict['before'] = []
    dict['meets'] = []
    dict['equal'] = []
    dict['overlaps'] = []
    if Allen_relation is None:
        warnings.warn("No Allen's relation is provided. Computing the complete Allen profile.")
        for i, seq_A in enumerate(subseq_T_A):
            list_O = [] #list containing the overlapping susbsequences for every subsequence seq_A
            list_B = [] #list containing the before susbsequences for every subsequence seq_A

            sub_T_A_values = [point.get_value() for point in seq_A]
            for j, seq_B in enumerate(subseq_T_B):
                sub_T_B_values = [point.get_value() for point in seq_B]
                first_instant_T_A = datetime.fromisoformat(str(seq_A[0].get_timestamp()))
                second_instant_T_A = datetime.fromisoformat(str(seq_A[m - 1].get_timestamp()))
                first_instant_T_B = datetime.fromisoformat(str(seq_B[0].get_timestamp()))
                second_instant_T_B = datetime.fromisoformat(str(seq_B[m - 1].get_timestamp()))

                if first_instant_T_A == first_instant_T_B and second_instant_T_A == second_instant_T_B: #equal
                    dist = round(z_normalized_euclidean_distance(sub_T_A_values, sub_T_B_values), 5)
                    dict['equal'].append([i, j, dist])

                elif second_instant_T_A + get_timedelta(second_instant_T_A) == first_instant_T_B:   #meets
                    dist = round(z_normalized_euclidean_distance(sub_T_A_values, sub_T_B_values), 5)
                    dict['meets'].append([i, j, dist])

                elif (first_instant_T_A != first_instant_T_B and
                        second_instant_T_A != second_instant_T_B and
                        first_instant_T_A < second_instant_T_B and
                        second_instant_T_A > first_instant_T_B): #overlaps
                    dist = round(z_normalized_euclidean_distance(sub_T_A_values, sub_T_B_values), 5)
                    list_O.append([i,j, dist])

                elif second_instant_T_A < first_instant_T_B and second_instant_T_A + get_timedelta(second_instant_T_A) != first_instant_T_B: #before
                    dist = round(z_normalized_euclidean_distance(sub_T_A_values, sub_T_B_values), 5)
                    list_B.append([i, j, dist])

            if list_O:
                min_distance_point = min(list_O, key=lambda x: x[2])
                dict['overlaps'].append([min_distance_point[0], min_distance_point[1], round(min_distance_point[2], 5)])
            if list_B:
                min_distance_point = min(list_B, key=lambda x: x[2])
                dict['before'].append([min_distance_point[0], min_distance_point[1], round(min_distance_point[2], 5)])


    elif Allen_relation=="before": #ok

        for i, seq_A in enumerate(subseq_T_A):
            list = []
            for j, seq_B in enumerate(subseq_T_B):
                # Compare timestamps instead of indices
                second_instant_T_A = datetime.fromisoformat(str(seq_A[m - 1].get_timestamp()))
                first_instant_T_B = datetime.fromisoformat(str(seq_B[0].get_timestamp()))
                if second_instant_T_A < first_instant_T_B and second_instant_T_A + get_timedelta(second_instant_T_A)  != first_instant_T_B:
                    sub_T_A_values = [point.get_value() for point in seq_A]
                    sub_T_B_values = [point.get_value() for point in seq_B]
                    dist = round(z_normalized_euclidean_distance(sub_T_A_values, sub_T_B_values), 5)
                    list.append([i, j, dist])
            if list:
                min_distance_point = min(list, key=lambda x: x[2])
                dict['before'].append([min_distance_point[0], min_distance_point[1], round(min_distance_point[2], 5)])
        if len(dict['before']) == 0:
            warnings.warn(f"There are no {Allen_relation} relation and window size {m}."
                          f"Please try a different Allen relation or window size.")

    elif Allen_relation == "meets":
        for i, seq_A in enumerate(subseq_T_A):
            for j, seq_B in enumerate(subseq_T_B):
                second_instant_T_A = datetime.fromisoformat(str(seq_A[m - 1].get_timestamp()))
                first_instant_T_B = datetime.fromisoformat(str(seq_B[0].get_timestamp()))
                if second_instant_T_A + get_timedelta(second_instant_T_A) == first_instant_T_B:
                    sub_T_A_values = [point.get_value() for point in seq_A]
                    sub_T_B_values = [point.get_value() for point in seq_B]
                    dist = round(z_normalized_euclidean_distance(sub_T_A_values, sub_T_B_values), 5)
                    dict['meets'].append([i, j, dist])
        if len(dict['meets']) == 0:
            warnings.warn(f"There are no {Allen_relation} relation and window size {m}."
                          f"Please try a different Allen relation or window size.")


    elif Allen_relation == "equal":
        for i, seq_A in enumerate(subseq_T_A):
            sub_T_A_values = [point.get_value() for point in seq_A]
            for j, seq_B in enumerate(subseq_T_B):
                first_instant_T_A = datetime.fromisoformat(str(seq_A[0].get_timestamp()))
                second_instant_T_A = datetime.fromisoformat(str(seq_A[m - 1].get_timestamp()))
                first_instant_T_B = datetime.fromisoformat(str(seq_B[0].get_timestamp()))
                second_instant_T_B = datetime.fromisoformat(str(seq_B[m - 1].get_timestamp()))

                if first_instant_T_A == first_instant_T_B and second_instant_T_A == second_instant_T_B:
                    sub_T_B_values = [point.get_value() for point in seq_B]
                    dist = round(z_normalized_euclidean_distance(sub_T_A_values, sub_T_B_values), 6)
                    dict['equal'].append([i, j, dist])

        if len(dict['equal']) == 0:
            warnings.warn(f"There are no {Allen_relation} relation and window size {m}."
                          f"Please try a different Allen relation or window size.")

    elif Allen_relation == "overlaps":
        for i, seq_A in enumerate(subseq_T_A):
            list = []
            for j, seq_B in enumerate(subseq_T_B):
                first_instant_T_A = datetime.fromisoformat(str(seq_A[0].get_timestamp()))
                second_instant_T_A = datetime.fromisoformat(str(seq_A[m - 1].get_timestamp()))
                first_instant_T_B = datetime.fromisoformat(str(seq_B[0].get_timestamp()))
                second_instant_T_B = datetime.fromisoformat(str(seq_B[m - 1].get_timestamp()))

                if (first_instant_T_A != first_instant_T_B and
                        second_instant_T_A != second_instant_T_B and
                        first_instant_T_A < second_instant_T_B and
                        second_instant_T_A > first_instant_T_B):

                    sub_T_A_values = [point.get_value() for point in seq_A]
                    sub_T_B_values = [point.get_value() for point in seq_B]
                    dist = round(z_normalized_euclidean_distance(sub_T_A_values, sub_T_B_values), 5)
                    list.append([i, j, dist])

            if list:
                min_distance_point = min(list, key=lambda x: x[2])
                dict['overlaps'].append([min_distance_point[0], min_distance_point[1], round(min_distance_point[2], 5)])

        if len(dict['overlaps']) == 0:
            warnings.warn(f"There are no {Allen_relation} relation and window size {m}."
                          f"Please try a different Allen relation or window size.")

    return dict


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

from datetime import timedelta
from dateutil.relativedelta import relativedelta

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
