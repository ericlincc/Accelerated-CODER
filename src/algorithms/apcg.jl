"""
Implementation of APCG basic version, Algorithm 1.

Lin, Q., Lu, Z., & Xiao, L. (2014). An Accelerated Proximal Coordinate Gradient Method and its Application to Regularized Empirical Risk Minimization. arXiv: Optimization and Control.
"""


struct APCGParams
    Ls::Vector{Float64}  # Lipschitz constants of each block
    μ::Float64  # Convexity parameter w.r.t. norm-L. See Assumption 2
    γ₀::Float64  # Choose γ₀ in [μ, 1]
end


function compute_stepsizes(m, γ, μ)
    _inside_sqrt = (γ - μ)^2 + 4 * m^2 * γ
    _α = (- (γ - μ) + sqrt(_inside_sqrt)) / (2 * m^2)
    α = min(_α, 1 / m)
    γ₊₁ = (1 - α) * γ + α * μ
    β = α * μ / γ₊₁
    return α, γ₊₁, β
end


function lipschitz_weightednorm(Ls, x)
    sqrt(sum(Ls .* (x .^ 2)))
end


function apcg(
    problem::CompositeFunc,
    exitcriterion::ExitCriterion,
    parameters::APCGParams;
    x₀=nothing
    )

    # Init of APCG
    Ls = parameters.Ls
    μ = parameters.μ
    γ = parameters.γ₀
    if isnothing(x₀)
        x₀ = zeros(problem.d)
    end
    x, z = copy(x₀), copy(x₀)
    m = problem.d  # Assume that each block is simply a coordinate for now

    # Run init
    iteration = 0
    exitflag = false
    starttime = time()
    results = Results()
    init_optmeasure = problem.func_value(x₀)
    logresult!(results, 1, 0.0, init_optmeasure)

    while !exitflag
        # Step 1
        α, β, γ₊₁ = compute_stepsizes(m, γ, μ)

        # Step 2
        y = (α * γ * z + γ₊₁ * x) / (α * γ + γ₊₁)

        # Step 3
        j = rand(1:problem.d)
        grad_j = problem.loss_func.grad_block(y, j)
        z₊₁ = (1 - β) * z + β * y
        z₊₁[j] = problem.reg_func.prox_opr_block(z[j] - grad_j / (m * α), 1 / (m * α))

        # Step 4
        x₊₁ = y + m * α * (z₊₁ - z) + μ / m * (z - y)

        x, z = x₊₁, z₊₁
        iteration += 1
        if iteration % (m * exitcriterion.loggingfreq) == 0
            elapsedtime = time() - starttime
            optmeasure = problem.func_value(x)
            @info "elapsedtime: $elapsedtime, iteration: $(iteration), optmeasure: $(optmeasure)"
            logresult!(results, iteration, elapsedtime, optmeasure)
            exitflag = checkexitcondition(exitcriterion, iteration, elapsedtime, optmeasure)
        end
    end

    results
end
