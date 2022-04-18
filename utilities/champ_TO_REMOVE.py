"""
This file contains the last function required from the upstream package champ.
    champ.get_intersection
which we defer to for the multilayer case.

You can see that the champ implementation spans 300 lines while our
implementation for the singlelayer case is ~50.

We are removing/refactoring this to improve performance, maintainability, and
fix a few latent bugs.
"""
from collections import defaultdict
import numpy as np
from numpy.random import choice
from scipy.spatial import HalfspaceIntersection
from scipy.spatial.qhull import QhullError
from scipy.optimize import linprog
import warnings


def get_intersection(coef_array, max_pt=None):
    '''
    Calculate the intersection of the halfspaces (planes) that form the convex hull

   :param coef_array: NxM array of M coefficients across each row representing N partitions
   :type coef_array: array
   :param max_pt: Upper bound for the domains (in the xy plane). This will restrict the convex hull \
    to be within the specified range of gamma/omega (such as the range of parameters originally searched using Louvain).
   :type max_pt: (float,float) or float
   :return: dictionary mapping the index of the elements in the convex hull to the points defining the boundary
    of the domain
    '''

    halfspaces = create_halfspaces_from_array(coef_array)
    num_input_halfspaces = len(halfspaces)

    singlelayer = False
    if halfspaces.shape[1] - 1 == 2:  # 2D case, halfspaces.shape is (number of halfspaces, dimension+1)
        singlelayer = True

    # Create Boundary Halfspaces - These will always be included in the convex hull
    # and need to be removed before returning dictionary

    boundary_halfspaces = []
    num_boundary = 0
    if not singlelayer:
        # origin boundaries
        boundary_halfspaces.extend([np.array([0, -1.0, 0, 0]), np.array([-1.0, 0, 0, 0])])
        num_boundary += 2
        if max_pt is not None:
            boundary_halfspaces.extend([np.array([0, 1.0, 0, -1.0 * max_pt[0]]),
                                        np.array([1.0, 0, 0, -1.0 * max_pt[1]])])
            num_boundary += 2
    else:

        boundary_halfspaces.extend([np.array([-1.0, 0, 0]),  # y-axis
                                    np.array([0, -1.0, 0])])  # x-axis
        num_boundary += 2

        if max_pt is not None:
            boundary_halfspaces.append(np.array([1.0, 0, -1.0 * max_pt]))
            num_boundary += 1

    # We expect infinite vertices in the halfspace intersection, so we can ignore numpy's floating point warnings
    old_settings = np.seterr(divide='ignore', invalid='ignore')

    halfspaces = np.vstack((halfspaces,) + tuple(boundary_halfspaces))

    if max_pt is None:
        if not singlelayer:
            # in this case, we will calculate max boundary planes later, so we'll impose x, y <= 10.0
            # for the interior point calculation here.
            interior_pt = get_interior_point(np.vstack((halfspaces,) +
                                                       (np.array([0, 1.0, 0, -10.0]), np.array([1.0, 0, 0, -10.0]))),
                                             num_bound=num_boundary)
        else:
            # similarly, in the 2D case, we impose x <= 10.0 for the interior point calculation
            interior_pt = get_interior_point(np.vstack((halfspaces,) + (np.array([1.0, 0, -10.0]),)),
                                             num_bound=num_boundary)
    else:
        interior_pt = get_interior_point(halfspaces, num_bound=num_boundary)

    # Find boundary intersection of half spaces
    joggled = False
    try:
        hs_inter = HalfspaceIntersection(halfspaces, interior_pt)
    except QhullError as e:
        # print(e)
        warnings.warn("Qhull input might be sub-dimensional, attempting to fix...", RuntimeWarning)

        # move the offset of the the first two boundary halfspaces (x >= 0 and y >= 0) so that
        # the joggled intersections are not outside our boundaries.
        joggled = True
        halfspaces[num_input_halfspaces][-1] = -1e-5
        halfspaces[num_input_halfspaces + 1][-1] = -1e-5
        hs_inter = HalfspaceIntersection(halfspaces, interior_pt,
                                         qhull_options="QJ")

    non_inf_vert = np.array([v for v in hs_inter.intersections if np.isfinite(v).all()])
    mx = np.max(non_inf_vert, axis=0)

    if joggled:
        # find largest (x,y) values of halfspace intersections and refuse to continue if too close to (0,0)
        max_xy_intersections = mx[:2]
        if max(max_xy_intersections) < 1e-2:
            raise ValueError("All intersections are less than ({:.3f},{:.3f}). "
                             "Invalid input set, try setting max_pt.".format(*max_xy_intersections))

    # max intersection on y-axis (x=0) implies there are no intersections in gamma direction.
    if np.abs(mx[0]) < np.power(10.0, -15) and np.abs(mx[1]) < np.power(10.0, -15):
        raise ValueError("Max intersection detected at (0,0).  Invalid input set.")

    if np.abs(mx[1]) < np.power(10.0, -15):
        mx[1] = mx[0]
    if np.abs(mx[0]) < np.power(10.0, -15):
        mx[0] = mx[1]

    # At this point we include max boundary planes and recalculate the intersection
    # to correct inf points.  We only do this for single layer
    if max_pt is None:
        if not singlelayer:
            boundary_halfspaces.extend([np.array([0, 1.0, 0, -1.0 * mx[1]]),
                                        np.array([1.0, 0, 0, -1.0 * mx[0]])])
            halfspaces = np.vstack((halfspaces,) + tuple(boundary_halfspaces[-2:]))

    if not singlelayer:
        # Find boundary intersection of half spaces
        interior_pt = get_interior_point(halfspaces, num_bound=num_boundary)
        hs_inter = HalfspaceIntersection(halfspaces, interior_pt)

    # revert numpy floating point warnings
    np.seterr(**old_settings)

    # scipy does not support facets by halfspace directly, so we must compute them
    facets_by_halfspace = defaultdict(list)
    for v, idx in zip(hs_inter.intersections, hs_inter.dual_facets):
        if np.isfinite(v).all():
            for i in idx:
                facets_by_halfspace[i].append(v)

    ind_2_domain = {}
    dimension = 2 if singlelayer else 3

    for i, vlist in facets_by_halfspace.items():
        # Empty domains
        if len(vlist) == 0:
            continue

        # these are the boundary planes appended on end
        if not i < num_input_halfspaces:
            continue

        pts = sort_points(vlist)
        pt2rm = []
        for j in range(len(pts) - 1):
            if comp_points(pts[j], pts[j + 1]):
                pt2rm.append(j)
        pt2rm.reverse()
        for j in pt2rm:
            pts.pop(j)
        if len(pts) >= dimension:  # must be at least 2 pts in 2D, 3 pt in 3D, etc.
            ind_2_domain[i] = pts

    # use non-inf vertices to return
    return ind_2_domain


def create_halfspaces_from_array(coef_array):
    '''
    create a list of halfspaces from an array of coefficent.  Each half space is defined by\
     the inequality\:
    :math:`normal\\dot point + offset \\le 0`

    Where each row represents the coefficients for a particular partition.
    For single Layer network, omit C_i's.

    :return: list of halfspaces.
    '''

    singlelayer = False
    if coef_array.shape[1] == 2:
        singlelayer = True

    cconsts = coef_array[:, 0]
    cgammas = coef_array[:, 1]
    if not singlelayer:
        comegas = coef_array[:, 2]

    if singlelayer:
        nvs = np.vstack((cgammas, np.ones(coef_array.shape[0])))
        pts = np.vstack((np.zeros(coef_array.shape[0]), cconsts))
    else:
        nvs = np.vstack((cgammas, -comegas, np.ones(coef_array.shape[0])))
        pts = np.vstack((np.zeros(coef_array.shape[0]), np.zeros(coef_array.shape[0]), cconsts))

    nvs = nvs / np.linalg.norm(nvs, axis=0)
    offs = np.sum(nvs * pts, axis=0)  # dot product on each column

    # array of shape (number of halfspaces, dimension+1)
    # Each row represents a halfspace by [normal; offset]
    # I.e. Ax + b <= 0 is represented by [A; b]
    return np.vstack((-nvs, offs)).T


def sort_points(points):
    '''For 2D case we sort the points along the gamma axis in assending order. \
    For the 3D case we sort the points clockwise around the center of mass .

    :param points:
    :return:
    '''
    if len(points[0]) > 2:  # pts are 3D
        cent = (sum([p[0] for p in points]) / len(points), sum([p[1] for p in points]) / len(points))
        points.sort(key=lambda x: np.arctan2(x[1] - cent[1], x[0] - cent[0]))
    else:
        points.sort(key=lambda x: x[0])  # just sort along x-axis

    return points


def comp_points(pt1, pt2):
    '''
    check for equality within certain tolerance
    :param pt1:
    :param pt2:
    :return:

    '''
    for i in range(len(pt1)):
        if np.abs(pt1[i] - pt2[i]) > np.power(10.0, -15):
            return False

    return True


def get_interior_point(hs_list, num_bound):
    '''
    Find interior point to calculate intersections
    :param hs_list: list of halfspaces
    :return: an approximation to the point most interior to the halfspace intersection polyhedron (Chebyshev center) if this computation succeeds. Otherwise, a point a small step towards the interior from the first plane in hs_list.
    '''

    normals, offsets = np.split(hs_list, [-1], axis=1)
    # in our case, the last num_bound halfspaces are boundary halfspaces

    if num_bound > 0:
        interior_hs, boundaries = np.split(hs_list, [-num_bound], axis=0)
    else:
        interior_hs = hs_list
        boundaries = None

    # randomly sample up to 50 of the halfspaces
    sample_len = min(50, len(interior_hs))
    if num_bound > 0:
        sampled_hs = np.vstack((interior_hs[choice(interior_hs.shape[0], sample_len, replace=False)], boundaries))
    else:
        sampled_hs = interior_hs[choice(interior_hs.shape[0], sample_len, replace=False)]

    # compute the Chebyshev center of the sampled halfspaces' intersection
    norm_vector = np.reshape(np.linalg.norm(sampled_hs[:, :-1], axis=1), (sampled_hs.shape[0], 1))
    c = np.zeros((sampled_hs.shape[1],))
    c[-1] = -1
    A = np.hstack((sampled_hs[:, :-1], norm_vector))
    b = -sampled_hs[:, -1:]

    manual = False
    try:
        res = linprog(c, A_ub=A, b_ub=b, bounds=None)

        # For some reason linprog raise error if fails on windows?

        if res.status == 0:
            intpt = res.x[:-1]  # res.x contains [interior_point, distance to enclosing polyhedron]

            # ensure that the computed point is actually interior to all halfspaces
            if (np.dot(normals, intpt) + np.transpose(offsets) < 0).all() and res.success:
                return intpt
        else:
            warnings.warn({1: "Interior point calculation: scipy.optimize.linprog exceeded iteration limit",
                           2: "Interior point calculation: scipy.optimize.linprog problem is infeasible. "
                              "Fallback will fail.",
                           3: "Interior point calculation: scipy.optimize.linprog problem is unbounded",
                           4: "Numerical Difficulties encountered"}.get(res.status,
                                                                        "unknown error with scipy.optimize"),
                          RuntimeWarning)
    except ValueError:
        pass

    warnings.warn("Interior point calculation: falling back to 'small step' approach.", RuntimeWarning)

    z_vals = [-1.0 * offset / normal[-1] for normal, offset in zip(normals, offsets) if
              np.abs(normal[-1]) > np.power(10.0, -15)]

    # take a small step into interior from 1st plane.
    dim = hs_list.shape[1] - 1  # hs_list has shape (number of halfspaces, dimension+1)
    intpt = np.array([0 for _ in range(dim - 1)] + [np.max(z_vals)])
    internal_step = np.array([.000001 for _ in range(dim)])
    return intpt + internal_step
