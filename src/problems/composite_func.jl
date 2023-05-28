struct CompositeFunc
    d
    loss_func
    reg_func
    func_value
    grad
    grad_block

    function CompositeFunc(loss_func, reg_func)

        function func_value(x::Vector{Float64})
            return loss_func.func_value(x) + reg_func.func_value(x)
        end

        function grad(x::Vector{Float64})
            return loss_func.grad(x) + reg_func.grad(x)
        end

        function grad_block(x::Vector{Float64}, j)
            return loss_func.grad_block(x, j) + reg_func.grad_block(x, j)
        end

        new(loss_func.d, loss_func, reg_func, func_value, grad, grad_block)
    end
end
