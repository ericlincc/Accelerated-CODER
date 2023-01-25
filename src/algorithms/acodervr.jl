"""
Implementation of A-CODER-VR version.
"""


struct ACODERVRParams
    L::Float64
    γ::Float64
    K::Int64
end


function acodervr_stepsize(A₋₁, L, γ, K)
    _ret = K * A₋₁ * (1 + A₋₁ * γ) / (8 * L)
    sqrt(_ret)
end


function acodervr(
    problem::CompositeFunc,
    exitcriterion::ExitCriterion,
    parameters::ACODERVRParams;
    x₀=nothing
    )

    # Short names
    ∇ₜʲf = problem.loss_func.grad_block_sample
    ∇f= problem.loss_func.grad
    prox_opr_block_g = problem.reg_func.prox_opr_block

    # Init of ACODERVR
    L, γ, K = parameters.L, parameters.γ, parameters.K
    a₋₁, A₋₁ = 0.0, 0.0; a, A = 1 / (4 * L), 1 / (4 * L)
    if isnothing(x₀)
        x₀ = zeros(problem.d)
    end
    w, v, x, y = copy(x₀), copy(x₀), copy(x₀), copy(x₀)
    w₋₁, v₋₁, x₋₁, y₋₁ = copy(x₀), copy(x₀), copy(x₀), copy(x₀)
    q, z₋₁, z = zeros(problem.d), zeros(problem.d), zeros(problem.d)
    ỹ₋₁, ỹ, ỹ_sum  = copy(x₀), copy(x₀), zeros(problem.d)
    m = problem.d  # Assume that each block is simply a coordinate for now
    n = 1:problem.loss_func.n

    # Seeding of ACODERVR
    z .= z₋₁ + ∇f(x₀)
    v = prox_opr_block_g.(x₀ - z, A)
    ỹ .= v; y .= v
    
    # Run init
    iteration = 0
    lastloggediter = 0
    exitflag = false
    starttime = time()
    results = Results()
    init_optmeasure = problem.func_value(x₀)
    logresult!(results, 1, 0.0, init_optmeasure)

    while !exitflag
        ỹ₋₁ .= ỹ
        ỹ_sum .= 0.0

        # Step 9 and 10
        a₋₁ = a; A₋₁ = A
        a = acodervr_stepsize(A₋₁, L, γ, K)
        A = A₋₁ + a

        # Step 12
        μ = ∇f(ỹ₋₁)

        for k in 1:K
            v₋₁ .= v; x₋₁ .= x; y₋₁ .= y; z₋₁ .= z

            # Step 14
            x .= (A₋₁ * ỹ₋₁ + a * v₋₁) / A

            w .= x
            w₋₁ .= x₋₁
            for j in m:-1:1
                # Step 16
                if j <= m - 1
                    w[j+1] = y[j+1]
                    w₋₁[j+1] = y₋₁[j+1]
                end

                # Step 17
                t = rand(n)

                # Step 18 and 19
                if k == 1
                    a₀₋₁ = a₋₁
                else
                    a₀₋₁ = a
                end
                _grad_vr = ∇ₜʲf(w, j, t) - ∇ₜʲf(ỹ₋₁, j, t) + μ[j]
                q[j] = _grad_vr + a₀₋₁ / a * (∇ₜʲf(x₋₁, j, t) - ∇ₜʲf(w₋₁, j, t))

                # Step 20
                z[j] = z₋₁[j] + a * q[j]

                # Step 21
                v[j] = prox_opr_block_g(x₀[j] - z[j] / K, A₋₁ + a * k / K)

                # Step 22
                y[j] = (A₋₁ * ỹ₋₁[j] + a * v[j]) / A
            end

            ỹ_sum .+= y
        end

        ỹ = ỹ_sum / K
        
        iteration += 5  # TODO: Check and verify
        if (iteration - lastloggediter) >= (exitcriterion.loggingfreq)
            lastloggediter = iteration
            elapsedtime = time() - starttime
            optmeasure = problem.func_value(ỹ)
            @info "elapsedtime: $elapsedtime, iteration: $(iteration), optmeasure: $(optmeasure)"
            logresult!(results, iteration, elapsedtime, optmeasure)
            exitflag = checkexitcondition(exitcriterion, iteration, elapsedtime, optmeasure)
        end
    end

    results, v, ỹ
end
