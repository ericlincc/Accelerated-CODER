using Dates
include("record.jl")


# the basic goal is to make the coder impl available for both min-max and min settings
# In all codes, we treat the primal and dual variables in the same way for algorithm;
# the difference is hidden in the underlying impl
function coder(x0::Vector{Float64}, grad:: Function, coor_grad::Function, prox_op::Function,
                                            L::Float64, γ::Float64, K::Int64,
                                            loss::Function, regularizer::Function, permutation_order, dim::Int64)::Record

    # obj value
    obj(x) = loss(x[1:dim]) + regularizer(x[1:dim])
    clock_start = now()
    clock_cnt() = now() - clock_start
    # Record
    func_value = zeros(0)
    time = zeros(0)

    #
    a = 0.0
    prev_A = 0.0
    A = 0.0
    # A = 1.0/(2 * L)
    α = 0
    d = size(x0)[1]
    m = d
    ∇ = zeros(Float64, d)
    ∇_prev = zeros(Float64, d)
    x = deepcopy(x0)
    y = deepcopy(x0)
    po = permutation_order

    p = deepcopy(x0)
    p_prev = deepcopy(x0)
    q = deepcopy(x0)
    z = -deepcopy(x0)
    for k = 1:K
        prev_a = a
        a = (1 + prev_A * γ)/(2.0 * L)
        # now we do not consider the strongly convex setting
        A = A + a
        prev_A = A
        β = prev_a / a

        ∇_prev = grad(x)
        Δx = 0.0
        # in each computation of function value, we assume that
        # A_x_b is already computed by the grad function
        value1  = obj(x)
        value2  = Dates.value(clock_cnt())/1000.
        @info "func: $value1"
        push!(func_value, value1)
        push!(time, value2)
        for j = 1:m
            if j == 1
                p[po[j]] = coor_grad(po[j])
            else
                p[po[j]] = coor_grad(po[j], po[j-1], Δx)
            end
            q[po[j]] = p[po[j]] + β * (∇_prev[po[j]] - p_prev[po[j]])
            z[po[j]] = z[po[j]] + a * q[po[j]]
            tmp = x[po[j]]
            x[po[j]] = prox_op(-z[po[j]], A, po[j], dim)
            # update p_prev, Δx_coor
            p_prev[po[j]] = p[po[j]]
            Δx = x[po[j]] - tmp
        end
    end
    return Record(func_value, time)
end
