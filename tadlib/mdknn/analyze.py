# Author: XiaoTao Wang, Weize Xu
# Organization: HuaZhong Agricultural University

from __future__ import division
import logging
import numpy as np
from scipy.stats import poisson, itemfreq
from scipy.stats import ks_2samp
import scipy.special as special
from sklearn import cluster
from sklearn.neighbors import KDTree

from tadlib.calfea.analyze import load_TAD, _fitting, manipulation


## Customize the logger
log = logging.getLogger(__name__)


class Core(object):
    """
    Interaction analysis at TAD level.
    
    High IFs off the diagonal region can be identified using
    :py:meth:`tadlib.calfea.analyze.Core.longrange`. :py:meth:`tadlib.calfea.analyze.Core.DBSCAN`
    performs a density-based clustering algorithm to detect aggregation patterns
    in those IFs. Furthermore, two structural features, called AP
    (Aggregation Preference) and Coverage in our original research, can be
    calculated by :py:meth:`tadlib.calfea.analyze.Core.gdensity` and
    :py:meth:`tadlib.calfea.analyze.Core.totalCover` respectively.
    
    Parameters
    ----------
    matrix : numpy.ndarray, (ndim = 2)
        Interaction matrix of a TAD.

    k : int, optional
        The number of nearest neighbors for calculate MDKNN. default 3.

    left : int, optional
        Starting point of TAD. For example, if the bin size is 10kb,
        ``left = 50`` means position 500000(bp) on the genome.
    
    Attributes
    ----------
    newM : numpy.ndarray, (ndim = 2)
        Gap-free interaction matrix.
    
    convert : list
        Information required for converting *newM* to *matrix*.
    
    cEM : numpy.ndarray, (ndim = 2)
        Expected interaction matrix. An upper triangular matrix. Value in each
        entry will be used to construct a Poisson Model for statistical
        significance calculation.
    
    fE : numpy.ndarray, (ndim = 2)
        An upper triangular matrix. Each entry represents the fold enrichment
        of corresponding observed interaction frequency.
    
    Ps : numpy.ndarray, (ndim = 2)
        An upper triangular matrix. Value in each entry indicates the p-value
        under corresponding Poisson Model.
    
    pos : numpy.ndarray, (shape = (N, 2))
        Coordinates of the selected IFs in *newM*.
    
    Np : int
        Number of the selected IFs.

    mean_dist : numpy.ndarray, (ndim = 1)
        Mean distance of center to k nearest neighbors.

    local_ap : numpy.ndarray, (ndim = 1)
        Local aggregate preference of each sinificant interaction point,
        equal to the reciprocal of mean distance.
    
    """
    def __init__(self, matrix, k=3, left = 0):
        matrix[np.isnan(matrix)] = 0
        self.k = k

        self.matrix = matrix

        # rescale matrix
        nonzero = matrix[matrix.nonzero()]
        if np.median(nonzero) < 1:
            min_nonzero = nonzero.min()
            scale = 1 / min_nonzero
            matrix = matrix * scale
        
        # Manipulation, remove vacant rows and columns
        self.newM, self.convert = manipulation(matrix, left)
        self._convert = np.array(list(self.convert[0].values()))
        
        ## Determine proper off-diagonal level
        Len = self.newM.shape[0]
        idiag = np.arange(0, Len)
        iIFs = []
        for i in idiag:
            temp = np.diagonal(self.newM, offset = i)
            iIFs.append(temp.mean())
        iIFs = np.array(iIFs)
        
        idx = np.where(iIFs > 0)[0][0]
        
        self._start = idx
        IFs = iIFs[idx:]
        diag = idiag[idx:]
        
        self._Ed = _fitting(diag, IFs)
        
    def longrange(self, pw = 2, ww = 5, top = 0.7, ratio = 0.05):
        """
        Select statistically significant interactions of the TAD. Both
        genomic distance and local interaction background are taken into
        account.
        
        Parameters
        ----------
        pw : int
            Width of the peak region. Default: 2
        
        ww : int
            Width of the donut. Default: 5
        
        top : float, [0.5, 1]
            Parameter for noisy interaction filtering. Default: 0.7
        
        ratio : float, [0.01, 0.1]
            Specifies the sample size of significant interactions.
            Default: 0.05
        
        Notes
        -----
        *pw* and *ww* are sensitive to data resolution. It performs well
        when we set *pw* to 4 and *ww* to 7 at 5 kb, and (2, 5) at 10 kb. [1]_
        
        References
        ----------
        .. [1] Rao, S.S., Huntley, M.H., Durand, N.C. et al. A 3D map of the
           human genome at kilobase resolution reveals principles of chromatin
           looping. Cell, 2014, 159: 1665-1680.
        
        """
        dim = self.newM.shape[0]
        
        ps = 2 * pw + 1 # Peak Size
        ws = 2 * ww + 1 # Initial window size
        bs = 2 * pw + 1 # B -- Blurry
        
        start = ww if (ww > self._start) else self._start
        # Upper triangular matrix
        upper = np.triu(self.newM, k = start)
        bUpper = np.triu(self.newM, k = 0)
        
        # Expanded Matrix
        expM = np.zeros((dim + ww*2, dim + ww*2))
        expBM = np.zeros((dim + ww*2, dim + ww*2))
        expM[ww:-ww, ww:-ww] = upper
        expBM[ww:-ww, ww:-ww] = bUpper
        
        tm = np.all((expBM == 0), axis = 0)
        Mask = np.zeros((dim + ww*2, dim + ww*2), dtype = bool)
        Mask[:,tm] = True
        Mask[tm,:] = True
        expCM = np.ones_like(expM, dtype = int)
        expCM[Mask] = 0
        
        ## Expected matrix
        EM_idx = np.triu_indices(dim, k = start)
        EM_value = self._Ed[EM_idx[1] - EM_idx[0] - self._start]
        EM = np.zeros((dim, dim))
        EM[EM_idx] = EM_value
        ## Expanded Expected Matrix
        expEM = np.zeros((dim + ww*2, dim + ww*2))
        expEM[ww:-ww, ww:-ww] = EM
        
        ## Construct pool of matrices for speed
        # Window
        OPool_w = {}
        EPool_w = {}
        ss = range(ws)
        for i in ss:
            for j in ss:
                OPool_w[(i,j)] = expM[i:(dim+i), j:(dim+j)]
                EPool_w[(i,j)] = expEM[i:(dim+i), j:(dim+j)]
        # Peak
        OPool_p = {}
        EPool_p = {}
        ss = range(ww-pw, ps+ww-pw)
        for i in ss:
            for j in ss:
                OPool_p[(i,j)] = expM[i:(dim+i), j:(dim+j)]
                EPool_p[(i,j)] = expEM[i:(dim+i), j:(dim+j)]
        
        # For Blurry Matrix
        OPool_b = {}
        OPool_bc = {}
        ss = range(ww-pw, bs+ww-pw)
        for i in ss:
            for j in ss:
                OPool_b[(i,j)] = expBM[i:(dim+i), j:(dim+j)]
                OPool_bc[(i,j)] = expCM[i:(dim+i), j:(dim+j)]
        
        ## Background Strength  --> Background Ratio
        bS = np.zeros((dim, dim))
        bE = np.zeros((dim, dim))
        for w in OPool_w:
            if (w[0] != ww) and (w[1] != ww):
                bS += OPool_w[w]
                bE += EPool_w[w]
        for p in OPool_p:
            if (p[0] != ww) and (p[1] != ww):
                bS -= OPool_p[p]
                bE -= EPool_p[p]
                
        bE[bE==0] = 1
        bR = bS / bE
        
        ## Corrected Expected Matrix
        cEM = EM * bR
        self.cEM = cEM
        
        ## Contruct the Blurry Matrix
        BM = np.zeros((dim, dim))
        CM = np.zeros((dim, dim), dtype = int)
        
        for b in OPool_b:
            BM += OPool_b[b]
            CM += OPool_bc[b]
        
        mBM = np.zeros_like(BM)
        Mask = CM != 0
        mBM[Mask] = BM[Mask] / CM[Mask]
        
        ## Fold Enrichments
        self.fE = np.zeros_like(self.cEM)
        mask = self.cEM != 0
        self.fE[mask] = upper[mask] / self.cEM[mask]
        
        ## Poisson Models
        Poisses = poisson(cEM)
        Ps = 1 - Poisses.cdf(upper)
        self.Ps = Ps
        rmBM = mBM[EM_idx] # Raveled array
        # Only consider the top x%
        top_idx = np.argsort(rmBM)[np.int(np.floor((1-top)*rmBM.size)):]
        # The most significant ones
        rPs = Ps[EM_idx][top_idx]
        Rnan = np.logical_not(np.isnan(rPs)) # Remove any invalid entry
        RrPs = rPs[Rnan]
        sig_idx = np.argsort(RrPs)[:np.int(np.ceil(ratio/top*RrPs.size))]
        if sig_idx.size > 0:
            self.pos = np.r_['1,2,0', EM_idx[0][top_idx][Rnan][sig_idx], EM_idx[1][top_idx][Rnan][sig_idx]]
            self.pos = self.convertPos(self.pos)
        else:
            self.pos = np.array([])
            
        self.Np = len(self.pos)
        self._area = EM_idx[0].size
        
    def convertMatrix(self, M):
        """
        Convert an internal gap-free matrix(e.g., newM, cEM, fE, and Ps)
        into a new matrix with the same shape as the original interaction
        matrix by using the recorded index map(see the *convert* attribute).
        """
        idx = sorted(self.convert[0].values())
        newM = np.zeros((self.convert[1], self.convert[1]), dtype=M.dtype)
        y,x = np.meshgrid(idx, idx)
        newM[x,y] = M
            
        return newM

    def convertPos(self, pos):
        """
        Convert the coordinate of the points in the gap-free matrix
        into the coordinate in the original matrix.
        """
        new_x = self._convert[self.pos[:, 0]]
        new_y = self._convert[self.pos[:, 1]]
        new_pos = np.c_[new_x, new_y]
        return new_pos

    def MDKNN(self):
        """Cauculate MDKNN(Mean Distance of k Nearest Neighbors) of selected interactions.
        KD Tree is used for speed up nearest neighbor searching.

        See Also
        --------
        sklearn.neighbors.KDTree : an implementation of KDTree.
        """
        k = self.k
        if self.Np < k + 5: # Lower bound for input
            self.mean_dist = np.nan
            self.mean_dist_all = np.nan
            self.AP = np.nan
            self.local_ap = np.nan
            return
        
        self._kdtree = KDTree(self.pos)
        dist, ind = self._kdtree.query(self.pos, k=k+1)
        self._DKNN = dist[:, 1:]
        self._KNN = ind[:, 1:]

        self.mean_dist_all = self._DKNN.mean(axis=1)
        self.mean_dist = self.mean_dist_all.mean()
        self.local_ap = 1 / self.mean_dist_all
        self.AP = self.local_ap.mean()


class Compare(object):
    """Compare 2 sample, calculate the p-value and the difference(sample2 - sample1) of MDKNN.
    Statistical test using the two-sided Kolmogorov-Smirnov Test on 2 samples.

    Parameters
    ----------
    core1 : `tadlib.mdknn.analyze.Core`
        Core of sample1.

    core2 : `tadlib.mdknn.analyze.Core`
        Core of sample2.

    """
    def __init__(self, core1, core2):
        self.core1 = core1
        self.core2 = core2

    def compare(self):
        """
        Perform two sample KS-Test calculate p-value.

        See Also
        --------
        scipy.stats.ks_2samp : an implementation of KS-Test
        """
        dist1 = self.core1.local_ap
        dist2 = self.core2.local_ap
        D, pvalue = ks_2samp(dist1, dist2)
        diff = dist2.mean() - dist1.mean()
        self.D = D
        self.pvalue = pvalue
        self.diff = diff


def getmatrix(inter, l_bin, r_bin):
    """Extract regional interaction data and place it into a matrix.
    
    Parameters
    ----------
    inter : numpy structured array
        Three fields are required, "bin1", "bin2" and "IF", data types of
        which are int, int and float, respectively.
    
    l_bin : int
        Left bin index of the region.
        
    r_bin : int
        Right bin index of the region.
        
    Returns
    -------
    inter_matrix : numpy.ndarray
        The value of each entry is the interaction frequency between
        corresponding two bins.
        
    """
    # Construct a matrix
    inter_matrix = np.zeros((r_bin - l_bin, r_bin - l_bin), dtype = float)
    # Extract the regional data
    mask = (inter['bin1'] >= l_bin) & (inter['bin1'] < r_bin) & \
           (inter['bin2'] >= l_bin) & (inter['bin2'] < r_bin)
    inter_extract = inter[mask]
    
    # Fill the matrix
    for i in inter_extract:
        # Off-diagonal parts
        if i['bin1'] != i['bin2']:
            inter_matrix[i['bin1'] - l_bin][i['bin2'] - l_bin] += i['IF']
            inter_matrix[i['bin2'] - l_bin][i['bin1'] - l_bin] += i['IF']
        else:
            # Diagonal part
            inter_matrix[i['bin1'] - l_bin][i['bin2'] - l_bin] += i['IF']
    
    return inter_matrix

