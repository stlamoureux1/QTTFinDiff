using Test
using QTTFinDiff

import ITensors, ITensorMPS

using ITensors, ITensorMPS

s2 = siteinds("QTT", 2)
s3 = siteinds("QTT", 3)
s4 = siteinds("QTT", 4)
s5 = siteinds("QTT", 5)
s10 = siteinds("QTT", 10)

u2_ones_ = ones(2^2)
u2_ones = MPS(u2_ones_, s2; maxdim=1)

u3_ones_ = ones(2^3)
u3_ones = MPS(u3_ones_, s3; maxdim=1)

u4_ones_ = ones(2^4)
u4_ones = MPS(u4_ones_, s4; maxdim=1)

u5_ones_ = ones(2^5)
u5_ones = MPS(u5_ones_, s5; maxdim=1)

function contracted_array(mpo, mps)
    n = length(mpo.data)
    return reshape(array(contract(mpo * mps)), 2^n)
end

@testset "shiftOp1d" begin
    @test begin
        S = shiftOp1d(s2)
        res = ones(2^2)
        res[2^2] = 0
        contracted_array(S, u2_ones) ≈ res
    end

    @test begin
        S = shiftOp1d(s3)
        res = ones(8)
        res[8] = 0
        contracted_array(S, u3_ones) ≈ res
    end

    @test begin
        S = shiftOp1d(s4)
        res = ones(2^4)
        res[2^4] = 0
        contracted_array(S, u4_ones) ≈ res
    end
end

@testset "gradFwd1d" begin
    @test begin
        Gf = gradFwd1d(s2)
        res = zeros(4)
        res[4] = 1
        contracted_array(Gf, u2_ones) ≈ res
    end

    @test begin
        Gf = gradFwd1d(s3)
        res = zeros(8)
        res[8] = 1
        contracted_array(Gf, u3_ones) ≈ res
    end

    @test begin
        Gf = gradFwd1d(s4)
        res = zeros(2^4)
        res[2^4] = 1
        contracted_array(Gf, u4_ones) ≈ res
    end
end

@testset "gradBwd1d" begin
    @test begin
        Gb = gradBwd1d(s2)
        res = zeros(4)
        res[1] = 1
        contracted_array(Gb, u2_ones) ≈ res
    end

    @test begin
        Gb = gradBwd1d(s3)
        res = zeros(8)
        res[1] = 1
        contracted_array(Gb, u3_ones) ≈ res
    end

    @test begin
        Gb = gradBwd1d(s4)
        res = zeros(16)
        res[1] = 1
        contracted_array(Gb, u4_ones) ≈ res
    end
end

@testset "laplacian1d" begin
    @test begin
        L = laplacian1d(s2)
        res = zeros(4)
        res[1] = 1
        res[4] = 1
        contracted_array(L, u2_ones) ≈ res
    end

    @test begin
        L = laplacian1d(s3)
        res = zeros(8)
        res[1] = 1
        res[8] = 1
        contracted_array(L, u3_ones) ≈ res
    end

    @test begin
        L = laplacian1d(s4)
        res = zeros(16)
        res[1] = 1
        res[16] = 1
        contracted_array(L, u4_ones) ≈ res
    end

    @test begin
        L = laplacian1d(s3)
        u_ = collect(1:8)
        u = MPS(u_, s3)
        res = zeros(8)
        res[8] = 9.0
        contracted_array(L, u) ≈ res
    end

    @test begin
        L = laplacian1d(s4)
        u_ = collect(1:16)
        u = MPS(u_, s4)
        res = zeros(16)
        res[16] = 17
        contracted_array(L, u) ≈ res
    end

    @test begin
        M = LinearAlgebra.SymTridiagonal(2 * ones(8), -ones(7))
        u_ = rand(8) .* 10
        L = laplacian1d(s3)
        res = M * u_
        u = MPS(u_, s3)
        contracted_array(L, u) ≈ res
    end

    @test begin
        M = LinearAlgebra.SymTridiagonal(2 * ones(16), -ones(15))
        u_ = rand(16) .* 10
        L = laplacian1d(s4)
        res = M * u_
        u = MPS(u_, s4)
        contracted_array(L, u) ≈ res
    end

end
