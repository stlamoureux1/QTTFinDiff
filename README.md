# Quantics format for classical problems

This package provides methods for constructing finite difference operators in quantics tensor train format (aka quantics / quantized matrix product operator), based on the constructions given in Kazeev & Khoromskij (2012).

# Operators available

- forward shift (`shiftOp1d()`)
- backward / forward difference (`gradBwd1d()`, `gradFwd1d()`)
- 2-point laplacian (`laplacian1d()`)

all in 1 spatial dimension with Dirichlet boundary conditions. More to come soon.

# Info

The operators are built using the ITensors / ITensorMPS libraries. The output of the methods is an `ITensorMPS.MPO`, i.e. a matrix product operator with tensor cores in `ITensor` format. 

# Usage example

```julia
include("QTTFinDiff.jl")

# Instantiate site indices of type "QTT"
# Here, we use 5 sites, corresponding to 2^5 data points
sites = siteinds("QTT", 5)

# instantiate 1d laplacian operator with 5 sites
L = laplacian1d(sites)
```

It is important to note that, in order to evaluate this operator as applied to an MPS, the two must share physical indices. That is, always instantiate your `sites` and use these for the physical dimensions of operators and operands. 
