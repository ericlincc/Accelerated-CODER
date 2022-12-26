"""
Implementation of RCDM basic version.

Yu Nesterov. Efficiency of coordinate descent methods on huge-scale optimization problems. SIAM Journal on Optimization, 22(2):341–362, 2012.
"""


struct RCDMParams
    Ls::Vector{Float64}  # Lipschitz constants of each block
    α::Float64  # Probability weight parameter
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

    # Run init
    iteration = 0
    exitflag = false
    starttime = time()
    results = Results()
    init_optmeasure = problem.func_value(x₀)
    logresult!(results, 1, 0.0, init_optmeasure)

    while !exitflag
        j = rand(1:problem.d)  # TODO: Implement probability weight-based sampling
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
