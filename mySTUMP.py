# This code uses STUMP procedure present in the STUMPY library.
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.
import warnings
from datetime import datetime, timedelta

import numpy as np
import stumpy
from dateutil.relativedelta import relativedelta


#equals

def MYSTUMP(T_A, T_B, m, Allen_relation=None):
    subseq_T_A = rolling_window(T_A, m)
    subseq_T_B = rolling_window(T_B, m)

    dict = {}
    dict['before'] = []
    dict['meets'] = []
    dict['equal'] = []
    dict['overlaps'] = []
    if Allen_relation is  None:
        for i, seq_A in enumerate(subseq_T_A):
            list_O = [] #list containing the overlapping susbsequences for every subsequence seq_A
            list_B = [] #list containing the before susbsequences for every subsequence seq_A
            for j, seq_B in enumerate(subseq_T_B):
                mp = stumpy.stump(T_A = seq_A,
                                  m = m,
                                  T_B = seq_B,
                                  ignore_trivial = False)
                ta_index = mp[:, 0].argmin()
                tb_index = mp[ta_index, 1]
                if i == j: #equal
                    dist = round(float(z_normalized_euclidean_distance(np.array(subseq_T_A[ta_index: ta_index+m]), np.array(subseq_T_B[tb_index: tb_index+m]))),5)
                    dict['equal'].append([i, j, dist])

                elif i+m == j:   #meets
                    dist = round(float(z_normalized_euclidean_distance(np.array(subseq_T_A[ta_index: ta_index+m]), np.array(subseq_T_B[tb_index: tb_index+m]))),5)
                    dict['meets'].append([i, j, dist])

                elif max(i, j)  <= min(i + m - 1, j + m - 1) and (i != j):
                    dist = round(float(z_normalized_euclidean_distance(np.array(subseq_T_A[ta_index: ta_index+m]), np.array(subseq_T_B[tb_index: tb_index+m]))),5)
                    list_O.append([i, j, dist])

                elif i+m < j: #before
                    dist = round(float(z_normalized_euclidean_distance(np.array(subseq_T_A[ta_index: ta_index+m]), np.array(subseq_T_B[tb_index: tb_index+m]))),5)
                    list_B.append([i, j, dist])

            if list_O:
                min_distance_point = min(list_O, key=lambda x: x[2])
                dict['overlaps'].append([min_distance_point[0], min_distance_point[1], min_distance_point[2]])
            if list_B:
                min_distance_point = min(list_B, key=lambda x: x[2])
                dict['before'].append([min_distance_point[0], min_distance_point[1], min_distance_point[2]])
    elif Allen_relation=="overlaps":
        for i, seq_A in enumerate(subseq_T_A):
            list_O = [] #list containing the overlapping susbsequences for every subsequence seq_A
            for j, seq_B in enumerate(subseq_T_B):
                #overlaps
                if max(i, j)  <= min(i + m - 1, j + m - 1) and (i != j):
                    mp = stumpy.stump(T_A = seq_A,
                                      m = m,
                                      T_B = seq_B,
                                      ignore_trivial = False)
                    ta_index = mp[:, 0].argmin()
                    tb_index = mp[ta_index, 1]
                    dist = round(float(z_normalized_euclidean_distance(np.array(subseq_T_A[ta_index: ta_index+m]), np.array(subseq_T_B[tb_index: tb_index+m]))),5)
                    list_O.append([i, j, dist])
            if list_O:
                min_distance_point = min(list_O, key=lambda x: x[2])
                dict['overlaps'].append([min_distance_point[0], min_distance_point[1], min_distance_point[2]])
        if len(dict['overlaps']) == 0:
            warnings.warn(f"There are no {Allen_relation} relation and window size {m}."
                          f"Please try a different Allen relation or window size.")

    elif Allen_relation=="equal":
        for i, seq_A in enumerate(subseq_T_A):
            for j, seq_B in enumerate(subseq_T_B):
                if i == j:
                    mp = stumpy.stump(T_A = seq_A,
                                      m = m,
                                      T_B = seq_B,
                                      ignore_trivial = False)
                    ta_index = mp[:, 0].argmin()
                    tb_index = mp[ta_index, 1]
                    dist = round(float(z_normalized_euclidean_distance(np.array(subseq_T_A[ta_index: ta_index+m]), np.array(subseq_T_B[tb_index: tb_index+m]))),5)
                    dict['equal'].append([i, j, dist])
        if len(dict['equal']) == 0:
            warnings.warn(f"There are no {Allen_relation} relation and window size {m}."
                          f"Please try a different Allen relation or window size.")

    elif Allen_relation=="meets":
        for i, seq_A in enumerate(subseq_T_A):
            for j, seq_B in enumerate(subseq_T_B):
                if i+m == j:
                    mp = stumpy.stump(T_A = seq_A,
                                      m = m,
                                      T_B = seq_B,
                                      ignore_trivial = False)
                    ta_index = mp[:, 0].argmin()
                    tb_index = mp[ta_index, 1]
                    dist = round(float(z_normalized_euclidean_distance(np.array(subseq_T_A[ta_index: ta_index+m]), np.array(subseq_T_B[tb_index: tb_index+m]))),5)
                    dict['meets'].append([i, j, dist])
        if len(dict['meets']) == 0:
            warnings.warn(f"There are no {Allen_relation} relation and window size {m}."
                          f"Please try a different Allen relation or window size.")

    elif Allen_relation=="before":
        for i, seq_A in enumerate(subseq_T_A):
            list_B = [] #list containing the before susbsequences for every subsequence seq_A
            for j, seq_B in enumerate(subseq_T_B):
                if i+m < j:
                    mp = stumpy.stump(T_A = seq_A,
                                      m = m,
                                      T_B = seq_B,
                                      ignore_trivial = False)
                    ta_index = mp[:, 0].argmin()
                    tb_index = mp[ta_index, 1]

                    dist = round(float(z_normalized_euclidean_distance(np.array(subseq_T_A[ta_index: ta_index+m]), np.array(subseq_T_B[tb_index: tb_index+m]))),5)
                    list_B.append([i, j, dist])

            if list_B:
                min_distance_point = min(list_B, key=lambda x: x[2])
                dict['before'].append([min_distance_point[0], min_distance_point[1],min_distance_point[2]])
        if len(dict['before']) == 0:
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