import torch
import numpy as np
import xgboost as xgb

from Tools import *

DataParentFolder = os.getcwd() + '/../h5/Ideal_Reweighted_Latent_Data_Coefficients/'
GeneralParentFolder = os.getcwd() + '/../'

def LoadDibosonData(DataSize=int(1e7)):
    DataFilePath   = DataParentFolder + "/trainingSampleReweightedLarge_Latent.h5"
    WeightFilePath = DataParentFolder + "/trainingSampleReweightedLarge_Latent_Benchmark.h5"
    td = CombinedDataFile(DataFilePath, WeightFilePath, verbose=True, NReadData=DataSize)

    with h5py.File(DataParentFolder + "/trainingSampleReweightedLarge_Latent.h5", "r") as vdfile:
            td.ReweightCoeffs = torch.Tensor(vdfile['ReweightCoeffs'][()])[:DataSize]

    DataFilePath   = DataParentFolder + "/testingSampleReweightedLarge_Latent.h5"
    WeightFilePath = DataParentFolder + "/testingSampleReweightedLarge_Latent_Benchmark.h5"
    tdtest = CombinedDataFile(DataFilePath, WeightFilePath, verbose=True, NReadData=DataSize)

    with h5py.File(DataParentFolder + "/testingSampleReweightedLarge_Latent.h5", "r") as vdfile:
            tdtest.ReweightCoeffs = torch.Tensor(vdfile['ReweightCoeffs'][()])[:DataSize]
    
    return td, tdtest


def ConstructDMatrixData(td, tdtest, gw=0.01):
    N = td.Data.size(0)
    
    rwval     = torch.Tensor([0., gw])
    wilson    = torch.cat([rwval, rwval**2, rwval.prod().reshape(1)]).reshape(-1, 1)
    weights   = td.Weights
    reweights = weights + td.ReweightCoeffs.mm(wilson).flatten()
    
    data      = torch.cat([td.Data, td.Data]).numpy()
    
    sample_weight = torch.cat([weights, reweights]).numpy()
    sample_weight = sample_weight/sample_weight.mean()

    label     = torch.cat([torch.zeros(N), torch.ones(N)]).numpy()
    
    dtrain    = xgb.DMatrix(data, label=label, weight=sample_weight)
    
    datatest  = tdtest.Data.numpy()
    dtest     = xgb.DMatrix(datatest)
    
    smweights = tdtest.Weights
    bsmweights = tdtest.ReweightCoeffs.mm(wilson).flatten()
    
    return dtrain, dtest, smweights, bsmweights

def NeymanPearsonTestXGBoost(Predt, SMWeights, BSMweights, Luminosity, compute_error = True):
    Rho = Predt/(1.-Predt)
    Rho = torch.Tensor(Rho)
        
    logrho = torch.log(Rho)
        
    # compute moments for SM and BSM    
    SMmoments, SMmomentsCov = ComputeMomentsWithCovariance(logrho, SMWeights)
    BSMmoments, BSMmomentsCov = ComputeMomentsWithCovariance(logrho, SMWeights + BSMweights)
        
    # compute Nsm and Nbsm with their errors
        
    Nsm = Luminosity * SMWeights.sum(0)
    DNbsm = Luminosity * BSMweights.sum(0)
    Ncov = torch.cuda.FloatTensor([[Nsm, 0.],[0., abs(DNbsm)]]) * SMWeights.size(0)**(-0.5)
        
    # compute central value for the p-value
              
    try:
        pValue = pValueFromMoments(SMmoments, Nsm, BSMmoments, Nsm + DNbsm)
    except Exception:
        return torch.cuda.FloatTensor([0.5, -1.0])
        
    if not compute_error:
        return torch.cuda.FloatTensor([pValue, -1.0])
        
    # randomly generate many toys using the central values and covariances
    # was 1000 on Mathematica, but it's slow
    n_tests = 50

    Nloc = torch.cuda.FloatTensor([Nsm, DNbsm])
    NDist = torch.distributions.multivariate_normal.MultivariateNormal(Nloc, Ncov).sample((n_tests,))
    SMmomentsDist = torch.distributions.multivariate_normal.MultivariateNormal(SMmoments, SMmomentsCov).sample((n_tests,))
    BSMmomentsDist = torch.distributions.multivariate_normal.MultivariateNormal(BSMmoments, BSMmomentsCov).sample((n_tests,))

    # compute p-value in each toy
    # this is a bit slow
    pList = torch.empty(n_tests).cuda()
    for i in range(n_tests):
        try:
            pList[i] = pValueFromMoments(SMmomentsDist[i], NDist[i,0], BSMmomentsDist[i], NDist[i,0] + NDist[i,1])
        except Exception:
            pList[i] = -1.0

    # return pvalue and standard deviation of toys
       
    return torch.cuda.FloatTensor([pValue, pList[pList > 0].std()])