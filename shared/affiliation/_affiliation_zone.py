#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from affiliation._integral_interval import interval_intersection

def t_start(j, Js = [(1,2),(3,4),(5,6)], Trange = (1,10)):
    """
    Helper for `E_gt_func`
    
    :param j: index from 0 to len(Js) (included) on which to get the start
    :param Js: ground truth events, as a list of couples
    :param Trange: range of the series where Js is included
    :return: generalized start such that the middle of t_start and t_stop 
    always gives the affiliation zone
    """
    b = max(Trange)
    n = len(Js)
    if j == n:
        return(2*b - t_stop(n-1, Js, Trange))
    else:
        return(Js[j][0])

def t_stop(j, Js = [(1,2),(3,4),(5,6)], Trange = (1,10)):
    """
    Helper for `E_gt_func`
    
    :param j: index from 0 to len(Js) (included) on which to get the stop
    :param Js: ground truth events, as a list of couples
    :param Trange: range of the series where Js is included
    :return: generalized stop such that the middle of t_start and t_stop 
    always gives the affiliation zone
    """
    if j == -1:
        a = min(Trange)
        return(2*a - t_start(0, Js, Trange))
    else:
        return(Js[j][1])

def E_gt_func(j, Js, Trange):
    """
    Get the affiliation zone of element j of the ground truth
    
    :param j: index from 0 to len(Js) (excluded) on which to get the zone
    :param Js: ground truth events, as a list of couples
    :param Trange: range of the series where Js is included, can 
    be (-math.inf, math.inf) for distance measures
    :return: affiliation zone of element j of the ground truth represented
    as a couple
    """
    range_left = (t_stop(j-1, Js, Trange) + t_start(j, Js, Trange))/2
    range_right = (t_stop(j, Js, Trange) + t_start(j+1, Js, Trange))/2
    return((range_left, range_right))

def get_all_E_gt_func(Js, Trange):
    """
    Get the affiliation partition from the ground truth point of view
    
    :param Js: ground truth events, as a list of couples
    :param Trange: range of the series where Js is included, can 
    be (-math.inf, math.inf) for distance measures
    :return: affiliation partition of the events
    """
    # E_gt is the limit of affiliation/attraction for each ground truth event
    E_gt = [E_gt_func(j, Js, Trange) for j in range(len(Js))]
    return(E_gt)

def affiliation_partition(Is = [(1,1.5),(2,5),(5,6),(8,9)], E_gt = [(1,2.5),(2.5,4.5),(4.5,10)]):
    """
    Cut the events into the affiliation zones
    The presentation given here is from the ground truth point of view,
    but it is also used in the reversed direction in the main function.
    
    :param Is: events as a list of couples
    :param E_gt: range of the affiliation zones
    :return: a list of list of intervals (each interval represented by either 
    a couple or None for empty interval). The outer list is indexed by each
    affiliation zone of `E_gt`. The inner list is indexed by the events of `Is`.
    """
    out = [None] * len(E_gt)
    # Small Is (e.g. the single-event recursive calls from _single_ground_truth_event):
    # keep the original O(|E_gt|*|Is|) form — it is already cheap and avoids numpy setup.
    if len(Is) <= 64:
        for j in range(len(E_gt)):
            out[j] = [interval_intersection(I, E_gt[j]) for I in Is]
        return(out)

    # Large Is (the top-level events_pred call): the original is O(|E_gt|*|Is|) and
    # blows up when predictions fragment into hundreds of thousands of events. Both Is
    # and E_gt are sorted, so use searchsorted to grab, per zone, the contiguous slice
    # of events that can possibly overlap it, then keep only real (non-None)
    # intersections. This is a SPARSE partition (drops the None entries the original
    # kept), which yields identical downstream results: precision sums treat None as a
    # zero contribution and recall filters None out. Verified against the original.
    starts = np.fromiter((I[0] for I in Is), float, len(Is))
    ends   = np.fromiter((I[1] for I in Is), float, len(Is))
    for j in range(len(E_gt)):
        a, b = E_gt[j]
        lo = int(np.searchsorted(ends,   a, side="left"))   # first I with end   >= a
        hi = int(np.searchsorted(starts, b, side="right"))  # first I with start >  b
        seg = [interval_intersection(Is[k], E_gt[j]) for k in range(lo, hi)]
        out[j] = [s for s in seg if s is not None]
    return(out)
