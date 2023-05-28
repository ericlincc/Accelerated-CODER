struct LogisticLoss
    n
    d
    func_value::Function
    grad::Function
    grad_block::Function
    grad_block_sample::Function
    grad_block_update!::Function

    function LogisticLoss(data::Data)
        n = length(data.values)
        d = size(data.features)[2]
        A = data.features
        b = data.values

        function func_value(x::Vector{Float64})
            A_x = A * x
            b_A_x = b .* A_x
            tmp = @. (log(exp(-b_A_x) + 1))
            return sum(tmp) / n
        end

        function grad(x::Vector{Float64})
            A_x = A * x
            b_A_x = b .* A_x
            tmp = @. - b * exp(-b_A_x) / (exp(-b_A_x) + 1) / n
            return transpose(transpose(tmp) * A)
        end
        
        function grad_block(x::Vector{Float64}, j::Int64)
            return grad(x)[j]
        end

        function grad_block_update!(
            x::Vector{Float64}
            )
            A_x = A * x
            b_A_x = b .* A_x
            tmp = @. - b * exp(-b_A_x) / (exp(-b_A_x) + 1) / n
            return transpose(transpose(tmp) * A), b_A_x
        end

        function grad_block_update!(
            x::Vector{Float64},
            j::Int64
            )
            A_x = A * x
            b_A_x = b .* A_x
            tmp = @. - b * exp(-b_A_x) / (exp(-b_A_x) + 1) / n
            return transpose(transpose(tmp) * A)[j], b_A_x
        end

        function grad_block_update!(
            b_A_x::Vector{Float64},
            update_x::Tuple{Int64, Float64},
            j::Int64,
            )
            i, Δxⁱ = update_x
            b_A_x .+= b .* Δxⁱ .* A[:, i]
            grad_block_x = sum(@. - b * exp(-b_A_x) / (1 + exp(-b_A_x)) * A[:, j]) / n
            return grad_block_x, b_A_x
        end

        function grad_block_sample(x::Vector{Float64}, j::Int64, t::Int64)
            a_x = 0.0
            for i in 1:d
                a_x += A[t, i] * x[i]
            end
            b_a_x = b[t] * a_x
            return - b[t] * exp(-b_a_x) / (1 + exp(-b_a_x)) * A[t, j]
        end
        
        new(n, d, func_value, grad, grad_block, grad_block_sample, grad_block_update!)
    end
end
