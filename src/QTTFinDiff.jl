# TODO: 
# 1. Code up Laplacian
# 2. Add options for boundary conditions
# 3. Extend to higher dimensions
# 4. See if there is a more efficient memory layout

module QTTFinDiff

import ITensors, ITensorMPS

using ITensors, ITensorMPS

include("QuanticsSite.jl")
include("PrimitiveOperators.jl")

export shiftOp1d, gradFwd1d, gradBwd1d, laplacian1d

function shiftOp1d(sites::Vector{Index{Int}})
    nSites = length(sites)
    links = [Index(2, "l" * string(i)) for i in 1:nSites-1]
    shiftOp = MPO(nSites)
    if nSites == 1
        shiftOp[1] = itensor(J, (sites[1], sites[1]'))
        return shiftOp
    end

    S1_ = zeros(2, 2, 2)
    S2_ = zeros(2, 2, 2, 2)
    S3_ = zeros(2, 2, 2)

    S1_[1, :, :] = J'
    S1_[2, :, :] = J

    S2_[1, 1, :, :] = I
    S2_[1, 2, :, :] = Zero
    S2_[2, 1, :, :] = J'
    S2_[2, 2, :, :] = J

    S3_[1, :, :] = I
    S3_[2, :, :] = J'

    shiftOp[1] = itensor(S1_, (links[1], sites[1], sites[1]'))
    for i in 2:nSites-1
        shiftOp[i] = itensor(S2_, (links[i-1], links[i], sites[i], sites[i]'))
    end
    shiftOp[nSites] = itensor(S3_, (links[nSites-1], sites[nSites], sites[nSites]'))
    return shiftOp
end

function gradFwd1d(sites::Vector{Index{Int}})
    nSites = length(sites)
    links = [Index(2, "l" * string(i)) for i in 1:nSites-1]
    grad = MPO(nSites)
    if nSites == 1
        grad[1] = itensor(I - J, (sites[1], sites[1]'))
        return grad
    end

    G1_ = zeros(2, 2, 2)
    G2_ = zeros(2, 2, 2, 2)
    G3_ = zeros(2, 2, 2)

    G1_[1, :, :] = I - J'
    G1_[2, :, :] = -J

    G2_[1, 1, :, :] = I
    G2_[1, 2, :, :] = Zero
    G2_[2, 1, :, :] = J'
    G2_[2, 2, :, :] = J

    G3_[1, :, :] = I
    G3_[2, :, :] = J'

    grad[1] = itensor(G1_, (links[1], sites[1], sites[1]'))
    for i in 2:nSites-1
        grad[i] = itensor(G2_, (links[i-1], links[i], sites[i], sites[i]'))
    end
    grad[nSites] = itensor(G3_, (links[nSites-1], sites[nSites], sites[nSites]'))
    return grad
end

function gradBwd1d(sites::Vector{Index{Int}})
    nSites = length(sites)
    links = [Index(2, "l" * string(i)) for i in 1:nSites-1]
    grad = MPO(nSites)
    if nSites == 1
        grad[1] = itensor(I - J', (sites[1], sites[1]'))
        return grad
    end

    G1_ = zeros(2, 2, 2)
    G2_ = zeros(2, 2, 2, 2)
    G3_ = zeros(2, 2, 2)

    G1_[1, :, :] = I - J
    G1_[2, :, :] = -J'

    G2_[1, 1, :, :] = I
    G2_[1, 2, :, :] = Zero
    G2_[2, 1, :, :] = J
    G2_[2, 2, :, :] = J'

    G3_[1, :, :] = I
    G3_[2, :, :] = J

    grad[1] = itensor(G1_, (links[1], sites[1], sites[1]'))
    for i in 2:nSites-1
        grad[i] = itensor(G2_, (links[i-1], links[i], sites[i], sites[i]'))
    end
    grad[nSites] = itensor(G3_, (links[nSites-1], sites[nSites], sites[nSites]'))
    return grad
end

function laplacian1d(sites::Vector{Index{Int}})
    nSites = length(sites)
    links = [Index(3, "l" * string(i)) for i in 1:nSites-1]
    laplacian = MPO(nSites)

    if nSites == 1
        laplacian[1] = itensor(2I - J' - J, (sites[1], sites[1]'))
        return laplacian
    end

    L1_ = zeros(3, 2, 2)
    L2_ = zeros(3, 3, 2, 2)
    L3_ = zeros(3, 2, 2)

    L1_[1, :, :] = 2I - J' - J
    L1_[2, :, :] = -J'
    L1_[3, :, :] = -J

    L2_[1, 1, :, :] = I
    L2_[1, 2, :, :] = Zero
    L2_[1, 3, :, :] = Zero
    L2_[2, 1, :, :] = J
    L2_[2, 2, :, :] = J'
    L2_[2, 3, :, :] = Zero
    L2_[3, 1, :, :] = J'
    L2_[3, 2, :, :] = Zero
    L2_[3, 3, :, :] = J

    L3_[1, :, :] = I
    L3_[2, :, :] = J
    L3_[3, :, :] = J'

    laplacian[1] = itensor(L1_, (links[1], sites[1], sites[1]'))
    for i in 2:nSites-1
        laplacian[i] = itensor(L2_, (links[i-1], links[i], sites[i], sites[i]'))
    end
    laplacian[nSites] = itensor(L3_, (links[nSites-1], sites[nSites], sites[nSites]'))

    return laplacian
end

end
