"""
Implementation of ACDM basic version.

Yu Nesterov. Efficiency of coordinate descent methods on huge-scale optimization problems. SIAM Journal on Optimization, 22(2):341–362, 2012.
"""


struct ACDMParams
    Ls::Vector{Float64}  # Lipschitz constants of each block
    σ::Float64  # convexity parameter
end


function acdm_stepsize(a, b, σ, m)
    _b = - 1 / m + (σ * a^2) / (m * b)
    _c = - a^2 / b^2
    return 0.5 * (-_b + sqrt(_b^2 - 4 * _c))
end

function acdm(
    problem::CompositeFunc,
    exitcriterion::ExitCriterion,
    parameters::ACDMParams;
    x₀=nothing
    )

    # Init of ACDM
    Ls = parameters.Ls
    σ = parameters.σ
    if isnothing(x₀)
        x₀ = zeros(problem.d)
    end
    v, x, y = copy(x₀), copy(x₀), copy(x₀)
    m = problem.d  # Assume that each block is simply a coordinate for now
    a, b = 1 / m, 2.0
    β, γ = 1.0, 1.0

    # Run init
    iteration = 0
    lastloggediter = 0
    exitflag = false
    starttime = time()
    results = Results()
    init_optmeasure = problem.func_value(x₀)
    logresult!(results, 1, 0.0, init_optmeasure)

    while !exitflag
        for _ in 1:m
            b = b / sqrt(β); a = γ * b
            γ = max(1 / m, acdm_stepsize(a, b, σ, m))
            α, β = (m - γ * σ) / (γ * (m^2 - σ)), 1 - γ * σ / m
            y .= α * v + (1 - α) * x 
            j = rand(1:problem.d)  # TODO: Implement probability weight-based sampling
            grad_j = problem.grad_block(y, j)
            x[j] = y[j] - grad_j / Ls[j]
            v .= β * v + (1 - β) * y
            v[j] = v[j] - γ / Ls[j] * grad_j
        end

        iteration += 1
        if (iteration - lastloggediter) >= (exitcriterion.loggingfreq)
            lastloggediter = iteration
            elapsedtime = time() - starttime
            optmeasure = problem.func_value(x)
            @info "elapsedtime: $elapsedtime, iteration: $(iteration), optmeasure: $(optmeasure)"
            logresult!(results, iteration, elapsedtime, optmeasure)
            exitflag = checkexitcondition(exitcriterion, iteration, elapsedtime, optmeasure)
        end
    end

    results, x
end