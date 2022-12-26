"""
Implementation of classic gradient descent.
"""


struct GDParams
    L::Float64
end


function gd(
    problem::CompositeFunc,
    exitcriterion::ExitCriterion,
    parameters::GDParams;
    x₀=nothing
    )

    # Init of GD
    L = parameters.L
    if isnothing(x₀)
        x₀ = zeros(problem.d)
    end
    x = copy(x₀)

    # Run init
    iteration = 0
    exitflag = false
    starttime = time()
    results = Results()
    init_optmeasure = problem.func_value(x₀)
    logresult!(results, 1, 0.0, init_optmeasure)

    while !exitflag
        x = x - problem.grad(x) / L
        
        iteration += 1
        if iteration % (exitcriterion.loggingfreq) == 0
            elapsedtime = time() - starttime
            optmeasure = problem.func_value(x)
            @info "elapsedtime: $elapsedtime, iteration: $(iteration), optmeasure: $(optmeasure)"
            logresult!(results, iteration, elapsedtime, optmeasure)
            exitflag = checkexitcondition(exitcriterion, iteration, elapsedtime, optmeasure)
        end
    end

    results
end
