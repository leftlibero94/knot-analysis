import numpy as np
import sys
from numba import njit
from tqdm import tqdm

def gen_array(filename, bp, iters):
# this functions takes the trajectory files generated and formats them into Numpy arrays, depending on length and number of recorded frames
    datarr=np.empty((bp*iters,3))
    with open(filename, 'r') as f:
        lines=f.readlines()
    lines=lines[bp+3:]
    lines=[x.strip() for x in lines]
    ctr=0
    for idx, line in enumerate(lines):
        line=line.replace('-', ' -')
        temp=line.split()
        if len(temp)==3:
            datarr[ctr]=np.array([float(x) for x in temp])
            ctr += 1
    return datarr

@njit
def check_int_lines(p1, p2, p3, p4):
# this function checks if there is an intersection between the line connecting points p1, p2 and the line connecting
# the points p3, p4, assuming this is in 2D, and gives the intersection point as well.
    flag=0
    den=(p4[1]-p3[1])*(p2[0]-p1[0]) - (p4[0]-p3[0])*(p2[1]-p1[1])
    if den==0:
        return 0, 0, 0
    else:
        ua=((p4[0]-p3[0])*(p1[1]-p3[1]) - (p4[1]-p3[1])*(p1[0]-p3[0]))/den
        ub=((p2[0]-p1[0])*(p1[1]-p3[1]) - (p2[1]-p1[1])*(p1[0]-p3[0]))/den
        if (ua>0 and ua<1) and (ub>0 and ub<1):
            flag=1
            x, y = p1[0]+ua*(p2[0]-p1[0]), p1[1]+ua*(p2[1]-p1[1])
        return flag, x, y

@njit
def contour_dist(arr):
# calculates the distances along the contour for all the points/beads making up the chain, provides unambiguous measure of
# sequence, since the 3D positions and relative orientation of points can change, but their positions along the contour don't.
    n=len(arr)
    dists=np.zeros(n, dtype=float)
    for i in range(1,n):
        dists[i]=dists[i-1] + np.linalg.norm(arr[i]-arr[i-1])
    return dists

@njit
def fcjit(arr):
# fc stands for find_crossings, and jit is for the Numba Just-In-Time flag. This is the main engine of the Alexander polynomial
# algorithm. It tries to first find and label all crossings in the 2D projection/ "knot diagram", and then find an unambiguous
# labelling of all arcs that contribute to it, and finally create an array knot_info which contains the labels for the over-arc,
# and the two under-arcs (incoming and outgoing) for every crossing, as well as the orientation of the crossing.
#This knot_info array can then be used to construct the Alexander polynomial by a subsequent function.
    n = len(arr)
    cross_n=int((n-1)*(n-2)*0.5)
    contour_lengths = contour_dist(arr)
    knot_info = np.zeros((cross_n, 5), dtype='int32')  # Explicitly use 'int32' to match Numba expectations
    cross_info = np.zeros((cross_n, 2), dtype='float32')  # 'float32' is more efficient
    ctr = 0
    proj = arr[:, 1:] # takes 2D projection on YZ plane -- this generates the knot diagram.
    for i in range(n - 3):  # loop only goes up to n-3 since it would take at least three consecutive bonds between beads i and j to make a crossing, reduces iterations
        for j in range(i + 2, n - 1):  # similarly, inner loop starts from i+2, since it would take at least three bonds, this might be taken to i+3 too, not sure
            p1, p2, p3, p4 = proj[i], proj[i + 1], proj[j], proj[j + 1]
            flag, y, z = check_int_lines(p1, p2, p3, p4)
            if flag:
                r1, r2, r3, r4 = arr[i], arr[i + 1], arr[j], arr[j + 1]
                slope1, slope2 = r2 - r1, r4 - r3
                if slope1[1] == 0:
                    slope1[1] = 0.01
                if slope2[1] == 0:
                    slope2[1] = 0.01
                x1 = r1[0] + (y - r1[1]) * slope1[0] / slope1[1]
                x2 = r3[0] + (y - r3[1]) * slope2[0] / slope2[1]  # if there is a crossing between i, i+1 and j, j+1, this gives X-coords corresponding to knot diagram on YZ plane
                if x1 > x2:
                    overidx, underidx = i, j
                    over, under = np.array([x1, y, z]), np.array([x2, y, z])
                    orientation = np.sign(np.cross(slope1, slope2))[0]
                else:
                    overidx, underidx = j, i
                    over, under = np.array([x2, y, z]), np.array([x1, y, z])
                    orientation = np.sign(np.cross(slope2, slope1))[0]
                overdist = contour_lengths[overidx] + np.linalg.norm(over - arr[overidx])
                underdist = contour_lengths[underidx] + np.linalg.norm(under - arr[underidx])
                cross_info[ctr] = overdist, underdist
                knot_info[ctr, :2] = ctr, int(orientation)  # Explicitly convert to int32
    # once the over and under points of intersection are identified, overdist and underdist calculate the contour distances of these points and stores them in cross_info.
                ctr += 1
    # Continue with the rest of the function (arcs construction)
    arcpts = np.append(cross_info[:ctr, 1], [contour_lengths[0] - 1.0, contour_lengths[-1] + 1.0])
    arcpts = np.sort(np.unique(arcpts)) # this takes all the crossings, plus two extra points at either edge of the open chain, and sorts the unique ones (the contour distances give an unambiguous metric)
    n_arcs = int(len(arcpts) - 1)  # no. of unique arcs
    arcs = np.zeros((n_arcs, 2))
    for i in range(n_arcs):
        arcs[i] = arcpts[i], arcpts[i + 1] # an array with all these arcs, between consecutive arcpoints
    for i in range(ctr):
# this loop will associate over (idx_i), under_incoming (idx_j), under_outgoing (idx_k) arcs to every crossing
        idx_j = np.where(arcs[:, 1] == cross_info[i, 1])[0][0]  # identifying the under_incoming arcs from cross_info second column, the underdists
        if idx_j == n_arcs - 2:
            idx_k = 0   # this is part of the novel algebraic closure method -- if the under_outgoing arc is the last one, it is labelled as the first one (n_arcs -2, because we added two extra points at the beginning and end)
        else:
            idx_k = idx_j + 1
        for j in range(n_arcs):
            if (cross_info[i, 0] > arcs[j, 0]) and (cross_info[i, 0] < arcs[j, 1]): # identifying the over arcs from cross_info first column (overdists)
                idx_i = j if j != n_arcs - 1 else 0
                break
            elif cross_info[i, 0] == arcs[j, 0]:
                idx_i = j
                break
        knot_info[i, 2:] = int(idx_i), int(idx_j), int(idx_k)
    return knot_info[:ctr]

@njit
def compute_alexpoly(arr, t):
# standard method to compute the Alexander polynomial from the knot_info array at a given argument value t, we will use -1.1, but not important here
    n=len(arr)
    alexpoly=np.zeros((n,n))
    elems_pos=np.array([1-t, -1, t])
    elems_neg=np.array([1-t, t, -1])
    for idx in range(n):
        orientn, i, j, k = arr[idx, 1:]
        if orientn>0:
            if i==j:
                alexpoly[idx,i]=elems_pos[0]+elems_pos[1]
                alexpoly[idx,k]=elems_pos[2]
            elif j==k:
                alexpoly[idx,j]=elems_pos[1]+elems_pos[2]
                alexpoly[idx,i]=elems_pos[0]
            elif k==i:
                alexpoly[idx,i]=elems_pos[0]+elems_pos[2]
                alexpoly[idx,j]=elems_pos[1]
            else:
                alexpoly[idx,i]=elems_pos[0]
                alexpoly[idx,j]=elems_pos[1]
                alexpoly[idx,k]=elems_pos[2]
        else:
            if i==j:
                alexpoly[idx,i]=elems_neg[0]+elems_neg[1]
                alexpoly[idx,k]=elems_neg[2]
            elif j==k:
                alexpoly[idx,j]=elems_neg[1]+elems_neg[2]
                alexpoly[idx,i]=elems_neg[0]
            elif k==i:
                alexpoly[idx,i]=elems_neg[0]+elems_neg[2]
                alexpoly[idx,j]=elems_neg[1]
            else:
                alexpoly[idx,i]=elems_neg[0]
                alexpoly[idx,j]=elems_neg[1]
                alexpoly[idx,k]=elems_neg[2]
    alexpoly=alexpoly[0:-1, 0:-1]
    alexdet=np.linalg.det(alexpoly)
    return alexpoly, alexdet

@njit
def calc_rg(arr):
# this function calculates the radius of gyration (R_g) of the knot
    n=len(arr)
    com=np.zeros(3)
    for i in range(3):
        com[i]=np.mean(arr[:,i])
    radgyr=0.0
    for i in range(n):
        dist=np.linalg.norm(arr[i]-com)
        radgyr+=dist**2
    radgyr/=n
    return np.sqrt(radgyr)

@njit
def rotyaxis(arr,theta):
# this function rotates the knotted chain around the Y-axis, given the angle theta
    rotation = np.array([[np.cos(theta), 0.0, np.sin(theta)],[0.0, 1.0, 0.0], [-np.sin(theta), 0.0, np.cos(theta)]])
    n=len(arr)
    com=np.sum(arr, axis=0)/len(arr)
    newmat=arr-com
    for i in range(n):
        newmat[i]=np.dot(rotation,arr[i])
    return rotation, newmat

@njit
def check_knot_yaxis(arr, alexpol):
# this function generates ten randomly oriented projections of the knotted chain
# if in >= 5 of these projections, we get the same Alexander polynomial (alexpol),
# then the function returns us alexpol
# otherwise the function returns us 1 (unknot)
# this function will be used to locate the composite knot
    ctr=0
    angles=np.zeros(10)
    for i in range(10):
        theta = -180.0+(360.0*np.random.rand())
        angles[i] = theta
        rot_matrix, rot_piece = rotyaxis(arr, theta)
        knot1 = fcjit(rot_piece)
        alex_poly = np.round(compute_alexpoly(knot1, -1.1)[1]*compute_alexpoly(knot1, -1/1.1)[1],5)
        if (np.abs((alex_poly-alexpol)/alexpol)<0.001):
            final_alexpoly=alex_poly
            ctr+=1
    if ctr>=5:
        return angles, final_alexpoly
    else:
        return angles, 1.0

@njit
def check_multiknot_yaxis(arr, alex1, alex2):
# this identifies which topology corresponds to a chain (or sub-chain), from 10 random projections
# if alex1, alex2 are the alexpolys corresponding to knot1 and knot2, then we have four options for any subchain,
# unknot, knot1, knot2, or composite knot (comb_alex=alex1*alex2)
    comb_alex=alex1*alex2
    ctr = np.array([0,0,0,0]) # counter for which topology is seen how many times in different projections
    alexs=np.array([alex1, alex2, comb_alex, 1.0])
    angles=np.zeros(10)
    for i in range(10):
        theta = -180.0+(360.0*np.random.rand())  # one of 10 random rotation angles
        angles[i] = theta
        rot_matrix, rot_piece = rotyaxis(arr, theta)
        knot1 = fcjit(rot_piece)
        alex_poly = np.round(compute_alexpoly(knot1, -1.1)[1]*compute_alexpoly(knot1, -1/1.1)[1],5)
        if (np.abs((alex_poly-alex1)/alex1)<0.001): #if chain corresponds to knot1
            ctr[0]+=1
        elif (np.abs((alex_poly-alex2)/alex2)<0.001): #if chain corresponds to knot2
            ctr[1]+=1
        elif (np.abs((alex_poly-comb_alex)/comb_alex)<0.001): #if chain corresponds to comb_knot
            ctr[2]+=1
        else:  # if chain corresponds to unknot
            ctr[3]+=1
    return np.amax(ctr), alexs[np.argmax(ctr)]

def loc_knots_trefcinq(arr, bp, iters, alex1, alex2):
# this is the main tracking function to track the knot1, knot2 edges
    knotsizes=np.zeros((iters,14))
    comb_alex=alex1*alex2
    for frame in tqdm(range(iters)):
        piece = arr[bp*frame:bp*(frame+1)]  # loads the trajectory snapshot in "frame"
        dists=contour_dist(piece)

        idx1, idx2 = 0, bp-1
        cut_piece=piece[idx1:]
        _, alex_poly = check_knot_yaxis(cut_piece, comb_alex)
        while (np.abs((alex_poly-comb_alex)/comb_alex)<0.001):
            idx1+=1
            cut_piece=piece[idx1:]
            _, alex_poly = check_knot_yaxis(cut_piece, comb_alex)
        if idx1>=5:
            idx1-=5 ### this section isolates the left edge of the composite knot
### once it finds that edge, just for safety, we move it left by another 5 beads if there is space

        cut_piece=piece[:idx2]
        _, alex_poly = check_knot_yaxis(cut_piece, comb_alex)
        while (np.abs((alex_poly-comb_alex)/comb_alex)<0.001):
            idx2-=1
            cut_piece=piece[:idx2]
            _, alex_poly = check_knot_yaxis(cut_piece, comb_alex)
        if idx2<=bp-6:
            idx2+=5 ### this finds the right edge of the composite knot
### once it finds that edge, we move it right by another 5 beads if there is space

### now we move on to individual knot tracking, now we will only focus on the region [idx1,idx2]
### here we will use a bottom-up search method, calculating alexpolys for all subchains within this region

        n=idx2-idx1+1
        n_arcs=int((n-9)*(n-8)/2) # we focus on subchains that are at least 9 beads long, just for efficiency
        arc_array=np.zeros((n_arcs,3))
        ctr=0
        for i in range(idx1,idx2-8):
            for j in range(i+9,idx2+1):
                arc_array[ctr,:2]=i, j
                ctr+=1
        for idx in range(n_arcs):
            i, j = int(arc_array[idx,0]), int(arc_array[idx,1])
            cut_piece = piece[i:j]
            _, alex_poly = check_multiknot_yaxis(cut_piece, alex1, alex2)
            arc_array[idx,2]=alex_poly
# the arc_array variable stores all sub-chain edges and their alexpolys

        subset1 = arc_array[arc_array[:, 2] == alex1] # subset of arc_array corresponding to knot1, if any
        subset2 = arc_array[arc_array[:, 2] == alex2] # subset of arc_array corresponding to knot2, if any

        if (len(subset1)==0) and (len(subset2)==0): # if no sub-chains can be found specifically for knot 1 or knot2
            trefid1, trefid2, cinqid1, cinqid2 = int(idx1), int(idx2), int(idx1), int(idx2)
        elif (len(subset1)!=0) and (len(subset2)==0): # knot1 found, knot2 not found (mixed state, knot1 inside knot2)
            cinqid1, cinqid2 = int(idx1), int(idx2)  # knot2 is therefore localized in the whole composite knot region (idx1, idx2)
            if not (np.any(subset1[:, 0][:, None] > subset1[:, 1][None, :])):
                starts = subset1[:, 0]
                ends = subset1[:, 1]
                arc_lengths = ends - starts
                min_idx = np.argmin(arc_lengths) # identifying the smallest arc with knot1 topology
                idx3, idx4, _ = subset1[min_idx]
                trefid1, trefid2 = int(idx3), int(idx4)
            else:
                i, j = np.where(subset1[:, 0][:, None] > subset1[:, 1][None, :])
                subset2, subset3 = subset1[np.unique(i)], subset1[np.unique(j)]
                starts2 = subset2[:, 0]
                ends2 = subset2[:, 1]
                arc_lengths2 = ends2 - starts2
                min_idx2 = np.argmin(arc_lengths2)
                starts3 = subset3[:, 0]
                ends3 = subset3[:, 1]
                arc_lengths3 = ends3 - starts3
                min_idx3 = np.argmin(arc_lengths3)
                rmid2a, rmid2b = int(subset2[min_idx2, 0]), int(subset2[min_idx2,1])
                rmid3a, rmid3b = int(subset3[min_idx3, 0]), int(subset3[min_idx3,1])
                newchain2=np.delete(piece, np.s_[rmid2a:rmid2b+1], axis=0)
                _, alex_poly2 = check_multiknot_yaxis(newchain2, alex1, alex2)
                newchain3=np.delete(piece, np.s_[rmid3a:rmid3b+1], axis=0)
                _, alex_poly3 = check_multiknot_yaxis(newchain3, alex1, alex2)
                if (alex_poly2==alex2):
                    trefid1, trefid2 = rmid2a, rmid2b
                elif (alex_poly3==alex2):
                    trefid1, trefid2 = rmid3a, rmid3b
                else:
                    trefid1, trefid2 = int(idx1), int(idx2)
        else:  # two options: knot1, knot2 both found or knot2 inside knot1
            starts = subset2[:, 0]
            ends = subset2[:, 1]
            arc_lengths = ends - starts
            min_idx = np.argmin(arc_lengths)
            idx3, idx4, _ = subset2[min_idx] # identified smallest arc with knot2 topology
            cinqid1, cinqid2 = int(idx3), int(idx4)

            starts = subset1[:, 0]
            ends = subset1[:, 1]
            non_overlapping = (ends < idx3+10) | (starts > idx4-10) # all knot1 arcs that do not overlap or partially overlap with knot2 small arc
            subset3=subset1[non_overlapping]

            if len(subset3)==0:  # if no non-overlapping knot1 arcs found, then knot2 inside knot1 (mixed)
                trefid1, trefid2 = int(idx1), int(idx2)

            else:  # if non overlapping knot1 arc found
                starts = subset3[:, 0]
                ends = subset3[:, 1]
                arc_lengths = ends - starts
                min_idx = np.argmin(arc_lengths)
                idx5, idx6, _ = subset3[min_idx]
                trefid1, trefid2 = int(idx5), int(idx6) # identified smallest non overlapping knot1 arc

        midpt1, midpt2 = (trefid1+trefid2)//2, (cinqid1+cinqid2)//2
        rg1, rg2=calc_rg(piece[trefid1:trefid2]), calc_rg(piece[cinqid1:cinqid2])
        knotsizes[frame] = float(trefid1), float(trefid2), float(midpt1), dists[trefid1], dists[trefid2], dists[midpt1], rg1, float(cinqid1), float(cinqid2), float(midpt2), dists[cinqid1], dists[cinqid2], dists[midpt2], rg2
    return knotsizes

length, iters = 483, 2000
alex1, alex2 = 9.05463, 25.45745 # corresponding to trefoil and cinquefoil

trajdat=gen_array('trajfile_demix.dat', length, iters)
knotdat=loc_knots_trefcinq(trajdat, length, 30, alex1, alex2) # the "30" should be replaced by "iters" for full analysis of the whole file
np.savetxt('knotdat_mix.txt', knotdat)
