"""Implementation of the regularizer function, g(x), with elastic net."""


function _prox_func(_x0, p1, p2)
    _value = 0.0
    if _x0 > p1
        _value = p2 * (_x0 - p1)
    elseif _x0 < -p1
        _value = p2 * (_x0 + p1)
    end
    _value
end


struct ElasticNet
    λ₁::Float64
    λ₂::Float64
    func_value::Function
    grad::Function
    grad_block::Function
    prox_opr_block::Function

    function ElasticNet(λ₁, λ₂)
        # if λ₁ <= 0 || λ₂ <= 0
        #     error("lambda 1 and lambda 2 must be strictly positive.")
        # end
        
        function func_value(x::Vector{Float64})
            sum(@.(λ₁ * abs(x) + λ₂ / 2 * x^2))
        end

        function grad(x::Vector{Float64})
            @.(λ₁ * sign(x) + λ₂ * x)
        end

        function grad_block(x::Vector{Float64}, j)
            λ₁ * sign(x[j]) + λ₂ * x[j]
        end

        function prox_opr_block(u, τ)
            p1 = τ * λ₁
            p2 = 1.0 / (1.0 + τ * λ₂)
            return _prox_func(u, p1, p2)
        end

        new(λ₁, λ₂, func_value, grad, grad_block, prox_opr_block)
    end
end
