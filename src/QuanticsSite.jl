import ITensors, ITensorMPS

using ITensors, ITensorMPS

include("PrimitiveOperators.jl")

ITensors.space(::SiteType"QTT") = 2

ITensors.op(::OpName"I", ::SiteType"QTT") = I

# TODO: MPO construction via OpSum requires an "Id" 
# operator. For now, just define both -- I like "I" 
# for usability but it doesn't matter.
ITensors.op(::OpName"Id", ::SiteType"QTT") = I

ITensors.op(::OpName"J", ::SiteType"QTT") = J

ITensors.op(::OpName"I1", ::SiteType"QTT") = I1

ITensors.op(::OpName"I2", ::SiteType"QTT") = I2

ITensors.op(::OpName"P", ::SiteType"QTT") = P

ITensors.op(::OpName"E", ::SiteType"QTT") = E

ITensors.op(::OpName"F", ::SiteType"QTT") = F

ITensors.op(::OpName"K", ::SiteType"QTT") = K

ITensors.op(::OpName"L", ::SiteType"QTT") = L

function ITensors.op(::OpName{N}, st::SiteType"QTT") where {N}
    name_str = String(N)
    if startswith(name_str, "-")
        base_name = name_str[2:end]
        return -op(base_name, st)
    end
    if endswith(name_str, "'")
        base_name = name_str[1:end-1]
        return op(base_name, st)'
    end
    error("Operator $N not defined for SiteType QTT")
end


