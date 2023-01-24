"""
Implementation of A-CODER basic version.
"""


struct ACODERParams
    L::Float64
    γ::Float64
end


function acoder_stepsize(a₋₁, A₋₁, L, γ)
    # Largest value s.t. Line 4 of Algo is satisfied
    _r = 0.4 * (1 + A₋₁ * γ) / L
    _b, _c = - _r, - A₋₁ * _r
    0.5 * (- _b + sqrt(_b^2 - 4 * _c))
end


function acoder(
    problem::CompositeFunc,
    exitcriterion::ExitCriterion,
    parameters::ACODERParams;
    x₀=nothing
    )

    # Init of ACODER
    L, γ = parameters.L, parameters.γ
    a, A = 0, 0
    if isnothing(x₀)
        x₀ = zeros(problem.d)
    end
    v, x, y = copy(x₀), copy(x₀), copy(x₀)
    v₋₁, x₋₁, y₋₁ = copy(x₀), copy(x₀), copy(x₀)
    q, w = zeros(length(x₀)), zeros(length(x₀))
    p = problem.loss_func.grad(x₀); p₋₁ = copy(p)
    z = zeros(problem.d); z₋₁ = copy(z)
    m = problem.d  # Assume that each block is simply a coordinate for now
    
    # Run init
    iteration = 0
    lastloggediter = 0
    exitflag = false
    starttime = time()
    results = Results()
    init_optmeasure = problem.func_value(x₀)
    logresult!(results, 1, 0.0, init_optmeasure)

    while !exitflag

        v₋₁ .= v; x₋₁ .= x; y₋₁ .= y
        p₋₁ .= p; z₋₁ .= z
        
        # Step 4
        A₋₁ = A
        a₋₁ = a
        a = acoder_stepsize(a, A₋₁, L, γ)
        A = A₋₁ + a

        # Step 5
        x = A₋₁ / A * y + a / A * v

        w .= x
        loss_func_grad_x₋₁ = problem.loss_func.grad(x₋₁)
        for j in m:-1:1
            # Step 7
            if j <= m - 1
                w[j+1] = y[j+1]
            end
            p[j] = problem.loss_func.grad_block(w, j)

            # Step 8
            q[j] = p[j] + a₋₁ / a * (loss_func_grad_x₋₁[j] - p₋₁[j])
            
            # Step 9
            z[j] = z₋₁[j] + a * q[j]

            # Step 10
            v[j] = problem.reg_func.prox_opr_block(x₀[j] - z[j], A)

            # Step 11
            y[j] = A₋₁ / A * y₋₁[j] + a / A * v[j]
        end
        
        iteration += m
        if (iteration - lastloggediter) >= (m * exitcriterion.loggingfreq)
            lastloggediter = iteration
            elapsedtime = time() - starttime
            optmeasure = problem.func_value(y)
            @info "elapsedtime: $elapsedtime, iteration: $(iteration), optmeasure: $(optmeasure)"
            logresult!(results, iteration, elapsedtime, optmeasure)
            exitflag = checkexitcondition(exitcriterion, iteration, elapsedtime, optmeasure)
        end
    end

    results
end


function acoder_eff(
    problem::CompositeFunc,
    exitcriterion::ExitCriterion,
    parameters::ACODERParams;
    x₀=nothing
    )

    # Init of ACODER
    L, γ = parameters.L, parameters.γ
    a, A = 0, 0
    if isnothing(x₀)
        x₀ = zeros(problem.d)
    end
    v, x, y = copy(x₀), copy(x₀), copy(x₀)
    v₋₁, x₋₁, y₋₁ = copy(x₀), copy(x₀), copy(x₀)
    q, w = zeros(length(x₀)), zeros(length(x₀))
    p = problem.loss_func.grad(x₀); p₋₁ = copy(p)
    z = zeros(problem.d); z₋₁ = copy(z)
    b_A_x = zeros(problem.loss_func.n)
    m = problem.d  # Assume that each block is simply a coordinate for now
    
    # Run init
    iteration = 0
    lastloggediter = 0
    exitflag = false
    starttime = time()
    results = Results()
    init_optmeasure = problem.func_value(x₀)
    logresult!(results, 1, 0.0, init_optmeasure)

    while !exitflag

        v₋₁ .= v; x₋₁ .= x; y₋₁ .= y
        p₋₁ .= p; z₋₁ .= z
        
        # Step 4
        A₋₁ = A
        a₋₁ = a
        a = acoder_stepsize(a, A₋₁, L, γ)
        A = A₋₁ + a

        # Step 5
        x = A₋₁ / A * y + a / A * v

        w .= x
        loss_func_grad_x₋₁ = problem.loss_func.grad(x₋₁)
        for j in m:-1:1
            # Step 7
            if j == m
                p[j], b_A_x = problem.loss_func.grad_block_update!(x, j)
            else
                p[j], b_A_x = problem.loss_func.grad_block_update!(b_A_x, (j+1, y[j+1] - x[j+1]), j)
            end

            # Step 8
            q[j] = p[j] + a₋₁ / a * (loss_func_grad_x₋₁[j] - p₋₁[j])
            
            # Step 9
            z[j] = z₋₁[j] + a * q[j]

            # Step 10
            v[j] = problem.reg_func.prox_opr_block(x₀[j] - z[j], A)

            # Step 11
            y[j] = A₋₁ / A * y₋₁[j] + a / A * v[j]
        end
        
        iteration += 2
        if (iteration - lastloggediter) >= (m * exitcriterion.loggingfreq)
            lastloggediter = iteration
            elapsedtime = time() - starttime
            optmeasure = problem.func_value(y)
            @info "elapsedtime: $elapsedtime, iteration: $(iteration), optmeasure: $(optmeasure)"
            logresult!(results, iteration, elapsedtime, optmeasure)
            exitflag = checkexitcondition(exitcriterion, iteration, elapsedtime, optmeasure)
        end
    end

    results
end


function acoder_adaptive(
    problem::CompositeFunc,
    exitcriterion::ExitCriterion,
    parameters::ACODERParams;
    x₀=nothing
    )

    # Init of ACODER
    L, γ = parameters.L, parameters.γ
    a, A = 0, 0
    if isnothing(x₀)
        x₀ = zeros(problem.d)
    end
    v, x, y = copy(x₀), copy(x₀), copy(x₀)
    v₋₁, x₋₁, y₋₁ = copy(x₀), copy(x₀), copy(x₀)
    q, w = zeros(length(x₀)), zeros(length(x₀))
    p = problem.loss_func.grad(x₀); p₋₁ = copy(p)
    z = zeros(problem.d); z₋₁ = copy(z)
    m = problem.d  # Assume that each block is simply a coordinate for now
    
    # Run init
    iteration = 0
    lastloggediter = 0
    exitflag = false
    starttime = time()
    results = Results()
    init_optmeasure = problem.func_value(x₀)
    logresult!(results, 1, 0.0, init_optmeasure)

    while !exitflag

        v₋₁ .= v; x₋₁ .= x; y₋₁ .= y
        p₋₁ .= p; z₋₁ .= z
        
        # Step 4
        A₋₁ = A
        a₋₁ = a
        a = acoder_stepsize(a, A₋₁, L, γ)
        A = A₋₁ + a

        # Step 5
        x = A₋₁ / A * y + a / A * v

        w .= x
        loss_func_grad_x₋₁ = problem.loss_func.grad(x₋₁)
        for j in m:-1:1
            # Step 7
            if j <= m - 1
                w[j+1] = y[j+1]
            end
            p[j] = problem.loss_func.grad_block(w, j)

            # Step 8
            q[j] = p[j] + a₋₁ / a * (loss_func_grad_x₋₁[j] - p₋₁[j])
            
            # Step 9
            z[j] = z₋₁[j] + a * q[j]

            # Step 10
            v[j] = problem.reg_func.prox_opr_block(x₀[j] - z[j], A)

            # Step 11
            y[j] = A₋₁ / A * y₋₁[j] + a / A * v[j]
        end
        
        iteration += m
        if (iteration - lastloggediter) >= (m * exitcriterion.loggingfreq)
            lastloggediter = iteration
            elapsedtime = time() - starttime
            optmeasure = problem.func_value(y)
            @info "elapsedtime: $elapsedtime, iteration: $(iteration), optmeasure: $(optmeasure)"
            logresult!(results, iteration, elapsedtime, optmeasure)
            exitflag = checkexitcondition(exitcriterion, iteration, elapsedtime, optmeasure)
        end
    end

    results
end
