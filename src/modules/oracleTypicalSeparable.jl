export SeparableOracle, SeparableOracleParam

mutable struct SeparableOracleParam <: AbstractOracleParam
    # may contain parameters for scenario handling.
    enable_parallel::Bool
    max_threads::Union{Int,Nothing}
    
    function SeparableOracleParam(;
        enable_parallel::Bool = false,
        max_threads::Union{Int,Nothing} = nothing
    )
        new(enable_parallel, max_threads)
    end
end

mutable struct SeparableOracle <: AbstractTypicalOracle
    oracle_param::SeparableOracleParam 

    oracles::Vector{AbstractTypicalOracle}
    N::Int

    function SeparableOracle(data::Data, 
                            oracle::T, 
                            N::Int; 
                            solver_param::Dict{String,Any} = Dict("solver" => "CPLEX", "CPX_PARAM_EPRHS" => 1e-9, "CPX_PARAM_NUMERICALEMPHASIS" => 1, "CPX_PARAM_EPOPT" => 1e-9), 
                            sub_oracle_param::AbstractOracleParam = EmptyOracleParam(),
                            oracle_param::SeparableOracleParam = SeparableOracleParam()) where {T<:AbstractTypicalOracle}
        @debug "Building classical separable oracle"
        # assume each oracle is associated with a single t, that is dim_t = N
        oracles = typeof(sub_oracle_param) != EmptyOracleParam ? [T(data, scen_idx=j, solver_param = solver_param, oracle_param = sub_oracle_param) for j=1:N] : [T(data, scen_idx=j, solver_param = solver_param) for j=1:N]

        new(oracle_param, oracles, N)
    end
end

function generate_cuts(oracle::SeparableOracle, x_value::Vector{Float64}, t_value::Vector{Float64}; tol = 1e-9, time_limit = 3600.0)
    tic = time()
    N = oracle.N
    is_in_L = Vector{Bool}(undef,N)
    sub_obj_val = Vector{Vector{Float64}}(undef,N)
    hyperplanes = Vector{Vector{Hyperplane}}(undef,N)

    # Calculate remaining time once for parallel execution
    remaining_time = get_sec_remaining(tic, time_limit)
    
    if oracle.oracle_param.enable_parallel
        # Determine number of threads to use
        max_threads = oracle.oracle_param.max_threads === nothing ? 
                     min(N, Threads.nthreads()) : 
                     min(N, oracle.oracle_param.max_threads)
        
        # Parallel execution
        Threads.@threads for j=1:N
            is_in_L[j], hyperplanes[j], sub_obj_val[j] = generate_cuts(oracle.oracles[j], x_value, [t_value[j]]; tol=tol, time_limit=remaining_time)
        end
        
        # Parallel post-processing of hyperplanes
        Threads.@threads for j=1:N
            # correct dimension for t_j's
            for h in hyperplanes[j]
                coeff_t = h.a_t[1]
                h.a_t = spzeros(length(t_value)) 
                h.a_t[j] = coeff_t
            end
        end
    else
        # Serial execution (original logic)
        for j=1:N
            is_in_L[j], hyperplanes[j], sub_obj_val[j] = generate_cuts(oracle.oracles[j], x_value, [t_value[j]]; tol=tol, time_limit=get_sec_remaining(tic, time_limit))

            # correct dimension for t_j's
            for h in hyperplanes[j]
                coeff_t = h.a_t[1]
                h.a_t = spzeros(length(t_value)) 
                h.a_t[j] = coeff_t
            end
        end
    end

    if any(.!is_in_L)
        return false, reduce(vcat, hyperplanes), reduce(vcat, sub_obj_val)
    else
        return true, [Hyperplane(length(x_value), length(t_value))], deepcopy(t_value)
    end
end






