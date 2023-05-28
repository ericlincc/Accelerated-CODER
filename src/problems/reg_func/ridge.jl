struct Ridge
    λ::Float64
    func_value::Function
    grad::Function
    grad_block::Function
    prox_opr_block::Function

    function Ridge(λ)

        function func_value(x::Vector{Float64})
            0.5 * λ * norm(x)^2
        end

        function grad(x::Vector{Float64})
            λ * x
        end

        function grad_block(x::Vector{Float64}, j)
            λ * x[j]
        end

        function prox_opr_block(u, τ)
            u / (τ * λ + 1)
        end
        
        new(λ, func_value, grad, grad_block, prox_opr_block)
    end
end
