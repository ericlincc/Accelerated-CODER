struct ExitCriterion
    # A data structure to store exit conditions.

    maxiter::Int              # Max #iterations allowed
    maxtime::Float64          # Max execution time allowed
    targetaccuracy::Float64   # Target accuracy to halt algorithm
    loggingfreq::Int          # #datapass between logging
end

"""
Check if the given exit criterion has been satisfied. Returns true if satisfied else returns false.
"""
function checkexitcondition(
    exitcriterion::ExitCriterion,
    currentiter::Integer,
    elapsedtime,
    measure,
)::Bool
    # A function determine if it's time to halt execution

    if currentiter >= exitcriterion.maxiter
        return true
    elseif elapsedtime >= exitcriterion.maxtime
        return true
    elseif measure <= exitcriterion.targetaccuracy
        return true
    end

    false
end
