struct Results
    iterations::Vector{Float64}
    times::Vector{Float64}
    optmeasures::Vector{Float64}
    
    function Results()
        new(Array{Int64}([]), Array{Float64}([]), Array{Float64}([]))
    end
end


"""Append execution measures to Results."""
function logresult!(r::Results, currentiter, elapsedtime, optmeasure)

    push!(r.iterations, currentiter)
    push!(r.times, elapsedtime)
    push!(r.optmeasures, optmeasure)
    return
end
