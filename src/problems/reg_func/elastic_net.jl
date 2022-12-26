struct ElasticNet
    λ₁::Float64
    λ₂::Float64
    γ::Float64
    func_value::Function
    prox_opr::Function

    function ElasticNet()
        if λ₁ <= 0 || λ₂ <= 0
            error("lambda 1 and lambda 2 must be strictly positive.")
        end
        

    end
end

