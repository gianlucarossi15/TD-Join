import warnings

import numpy as np

from stumpy import core

def TD_Join(T_A, T_B, m, Allen_relation=None):
    """
    Compute Time Dependent Matrix Profile TDMP.

    Parameters
    ----------
    T_A : numpy.ndarray
        The time series or sequence for which the matrix profile will be returned

    T_B : numpy.ndarray
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in T_A, its nearest neighbor in T_B will be recorded.

    m : int
        Window size


    Allen_relation : str, default None
        Allen's relation to be computed. Choose from 'Before', 'Meets', 'Equals', or 'Overlaps'.
        If None, the complete Allen profile will be computed.

    Returns
    -------
    out : dict
    A dictionary where keys are Allen's relations and values are lists of pairs [index, distance].
    """

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
    subseq_T_B = core.rolling_window(T_B,  m)

    result = []
    dict = {}
    dict['Before'] = []
    dict['Meets'] = []
    dict['Equals'] = []
    dict['Overlaps'] = []
    if Allen_relation is None:
        warnings.warn("No Allen's relation is provided. Computing the complete Allen profile.")
        for i, seq_A in enumerate(subseq_T_A):
            list_O = [] #list containing the overlapping susbsequences for every subsequence seq_A
            list_B = [] #list containing the before susbsequences for every subsequence seq_A
            for j, seq_B in enumerate(subseq_T_B):
                if i == j: #Equals
                    dist = round(float(z_normalized_euclidean_distance(seq_A, seq_B)),5)
                    dict['Equals'] += [[j, dist]]

                elif i+m == j: #Meets
                    dist = round(float(z_normalized_euclidean_distance(seq_A, seq_B)),5)
                    dict['Meets'] += [[j, dist]]

                elif max(i, j) <= min(i + m - 1, j + m - 1) and (i != j): #Overlaps
                    list_O.append([j,float(z_normalized_euclidean_distance(seq_A, seq_B))])

                elif i+m < j: #Before
                    list_B.append([j,float(z_normalized_euclidean_distance(seq_A, seq_B))])

            for index in range(len(list_O)):
                values = [list_O[index][1] for index in range(len(list_O))]
            for index in range(len(list_O)):
                if list_O[index][1] == np.min(values):
                    dist = round(np.min(values),5)
                    dict['Overlaps'] += [[list_O[index][0], float(dist)]]
                    break
            for index in range(len(list_B)):
                values = [list_B[index][1] for index in range(len(list_B))]
            for index in range(len(list_B)):
                if list_B[index][1] == np.min(values):
                    dist = round(np.min(values),5)
                    dict['Before'] += [[list_B[index][0], float(dist)]]
                    break


    elif Allen_relation=="Before": #ok

        for i, seq_A in enumerate(subseq_T_A):
            list = []
            for j, seq_B in enumerate(subseq_T_B):
                # if upper hand of seq is before of lower hand of seq_B
                if i+m < j:
                    list.append([j,float(z_normalized_euclidean_distance(seq_A, seq_B))])
            for index in range(len(list)):
                values = [list[index][1] for index in range(len(list))]
            for index in range(len(list)):
                if list[index][1] == np.min(values):
                    dist = round(np.min(values),5)
                    dict['Before'] += [[list[index][0], float(dist)]]
                    break
        if len(dict['Before']) == 0:
            warnings.warn(f"There are no {Allen_relation} relation and window size {m}."
                          f"Please try a different Allen relation or window size.")

    elif Allen_relation=="Meets": #ok
        for i, seq_A in enumerate(subseq_T_A):
            for j, seq_B in enumerate(subseq_T_B):
                # if upper hand of seq is equal to lower hand of seq_B
                if i+m == j:
                    dist = round(float(z_normalized_euclidean_distance(seq_A, seq_B)),5)
                    dict['Meets'] += [[j, dist]]
        if len(dict['Meets']) == 0:
            warnings.warn(f"There are no {Allen_relation} relation and window size {m}."
                          f"Please try a different Allen relation or window size.")

    elif Allen_relation=="Equals": #ok
        for i, seq_A in enumerate(subseq_T_A):
            for j, seq_B in enumerate(subseq_T_B):
                # if seq upper hand is equal to seq_B lower hand
                if i == j:
                    dist = round(float(z_normalized_euclidean_distance(seq_A, seq_B)),5)
                    dict['Equals'] += [[j, dist]]

    elif Allen_relation=="Overlaps": #ok
        for i, seq_A in enumerate(subseq_T_A):
            list = []
            for j, seq_B in enumerate(subseq_T_B):
                if max(i, j) <= min(i + m - 1, j + m - 1) and (i != j):
                    list.append([j,float(z_normalized_euclidean_distance(seq_A, seq_B))])
            for index in range(len(list)):
                values = [list[index][1] for index in range(len(list))]
            for index in range(len(list)):
                if list[index][1] == np.min(values):
                    dist = round(np.min(values),5)
                    dict['Overlaps'] += [[list[index][0], float(dist)]]
                    break
        if len(dict['Before']) == 0:
            warnings.warn(f"There are no {Allen_relation} relation and window size {m}."
                          f"Please try a different Allen relation or window size.")
    return dict


def z_normalized_euclidean_distance(a, b):
    a_z = (a - np.mean(a)) / np.std(a)
    b_z = (b - np.mean(b)) / np.std(b)
    return np.linalg.norm(a_z - b_z)