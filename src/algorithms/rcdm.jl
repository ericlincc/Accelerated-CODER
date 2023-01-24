"""
Implementation of RCDM basic version.

Yu Nesterov. Efficiency of coordinate descent methods on huge-scale optimization problems. SIAM Journal on Optimization, 22(2):341–362, 2012.
"""


struct RCDMParams
    Ls::Vector{Float64}  # Lipschitz constants of each block
    α::Float64  # Probability weight parameter
end


function get_L_probs(Ls, α)
    _sum_L = 0.0
    for _L in Ls
        _sum_L += _L^α
    end
    _probs = zeros(length(Ls))
    for (i, _L) in enumerate(Ls)
        _probs[i] = _L^α / _sum_L
    end
    Categorical(_probs)
end


function rcdm(
    problem::CompositeFunc,
    exitcriterion::ExitCriterion,
    parameters::RCDMParams;
    x₀=nothing
    )

    # Init of RCDM
    Ls = parameters.Ls
    α = parameters.α
    if isnothing(x₀)
        x₀ = zeros(problem.d)
    end
    x = copy(x₀)
    m = problem.d  # Assume that each block is simply a coordinate for now

    # Pre-compute sampling probabilities
    L_probs = get_L_probs(Ls, α)

    # Run init
    iteration = 0
    exitflag = false
    starttime = time()
    results = Results()
    init_optmeasure = problem.func_value(x₀)
    logresult!(results, 1, 0.0, init_optmeasure)

    while !exitflag
        j = rand(L_probs)
        grad_j = problem.grad_block(x, j)
        x[j] = x[j] - grad_j / Ls[j]
        
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


function rcdm_adapative(
    problem::CompositeFunc,
    exitcriterion::ExitCriterion,
    parameters::RCDMParams;
    x₀=nothing
    )

    # Init of RCDM
    Ls = copy(parameters.Ls)
    α = parameters.α
    @assert(α == 0)
    if isnothing(x₀)
        x₀ = zeros(problem.d)
    end
    x = copy(x₀)
    m = problem.d  # Assume that each block is simply a coordinate for now

    # Pre-compute sampling probabilities
    L_probs = get_L_probs(Ls, α)

    # Run init
    iteration = 0
    lastloggediter = 0
    exitflag = false
    starttime = time()
    results = Results()
    init_optmeasure = problem.func_value(x₀)
    logresult!(results, 1, 0.0, init_optmeasure)

    while !exitflag
        j = rand(L_probs)

        # If smoothness parameter gets to zeros, skip block
        if Ls[j] == 0.0
            continue
        end

        grad_j = problem.grad_block(x, j)
        _x_j = x[j]
        x[j] = _x_j - grad_j / Ls[j]
        _grad_j = problem.grad_block(x, j)

        while grad_j * _grad_j < 0
            Ls[j] = 2 * Ls[j]
            x[j] = _x_j - grad_j / Ls[j]
            _grad_j = problem.grad_block(x, j)
            iteration += 1
        end
        Ls[j] = 0.5 * Ls[j]
        
        iteration += 2
        if (iteration - lastloggediter) >= (m * exitcriterion.loggingfreq)
            lastloggediter = iteration
            elapsedtime = time() - starttime
            optmeasure = problem.func_value(x)
            @info "elapsedtime: $elapsedtime, iteration: $(iteration), optmeasure: $(optmeasure)"
            logresult!(results, iteration, elapsedtime, optmeasure)
            exitflag = checkexitcondition(exitcriterion, iteration, elapsedtime, optmeasure)
        end
    end

    print(Ls)
    results
end