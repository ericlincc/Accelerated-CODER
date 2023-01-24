struct LogisticLoss
    n
    d
    func_value::Function
    grad::Function
    grad_block::Function
    grad_block_sample::Function

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
        
        function grad_block(x::Vector{Float64}, j)
            return grad(x)[j]
        end

        function grad_block_sample(x::Vector{Float64}, j, t)
            a_x = 0.0
            for i in 1:d
                a_x += A[t, i] * x[i]
            end
            b_a_x = b[t] * a_x
            return - b[t] * exp(-b_a_x) / (1 + exp(-b_a_x)) * A[t, j]
        end

        # TODO: What is this implementation?
        function coord_grad(grad_idx::Int64, update_idx::Int64 = -1, Δx::Float64 = 0.0)
            A_x = A * x
            b_A_x = b .* A_x
            if update_idx !=  -1
                b_A_x += Δx * (A[:, update_idx] .* b)
            end
            # vector
            exp_A_x_b = @. exp(-b_A_x)
            tmp = @. exp_A_x_b / (exp_A_x_b +1)
            return transpose(A[:, grad_idx] .* b) * tmp / n
        end
        
        new(n, d, func_value, grad, grad_block, grad_block_sample)
    end
end
