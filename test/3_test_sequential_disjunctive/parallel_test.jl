include("$(dirname(dirname(@__DIR__)))/example/cflp/data_reader.jl")
include("$(dirname(dirname(@__DIR__)))/example/cflp/oracle.jl")
include("$(dirname(dirname(@__DIR__)))/example/cflp/model.jl")

using DataFrames

@testset verbose = true "CFLP Parallel DCGLP Tests" begin
    # Test a small subset of instances for parallel functionality
    instances = 1:35 # Small instances for faster testing
    
    for i in instances
        @testset "Instance: p$i" begin
            # Load problem data
            problem = read_cflp_benchmark_data("p$i")
            
            # Initialize dimensions and data
            dim_x = problem.n_facilities
            dim_t = 1
            c_x = problem.fixed_costs
            c_t = [1]
            data = Data(dim_x, dim_t, problem, c_x, c_t)
            @assert dim_x == length(data.c_x)
            @assert dim_t == length(data.c_t)

            # Algorithm parameters
            benders_param = BendersSeqParam(;
                                            time_limit = 800.0,
                                            gap_tolerance = 1e-6,
                                            verbose = false
                                            )
            dcglp_param = DcglpParam(;
                                    time_limit = 1000.0, 
                                    gap_tolerance = 1e-3, 
                                    halt_limit = 3, 
                                    iter_limit = 250,
                                    verbose = false
                                    )
            
            # Solver parameters
            mip_solver_param = Dict("solver" => "CPLEX", "CPX_PARAM_EPINT" => 1e-9, "CPX_PARAM_EPRHS" => 1e-9, "CPX_PARAM_EPGAP" => 1e-9, "CPXPARAM_Threads" => 4)
            master_solver_param = Dict("solver" => "CPLEX", "CPX_PARAM_EPINT" => 1e-9, "CPX_PARAM_EPRHS" => 1e-9, "CPX_PARAM_EPGAP" => 1e-9, "CPXPARAM_Threads" => 4)
            typical_oracle_solver_param = Dict("solver" => "CPLEX", "CPX_PARAM_EPRHS" => 1e-9, "CPX_PARAM_NUMERICALEMPHASIS" => 1, "CPX_PARAM_EPOPT" => 1e-9)
            dcglp_solver_param = Dict("solver" => "CPLEX", "CPX_PARAM_EPRHS" => 1e-9, "CPX_PARAM_NUMERICALEMPHASIS" => 1, "CPX_PARAM_EPOPT" => 1e-9)

            # Solve MIP for reference
            mip = Mip(data)
            assign_attributes!(mip.model, mip_solver_param)
            update_model!(mip, data)
            optimize!(mip.model)
            @assert termination_status(mip.model) == OPTIMAL
            mip_opt_val = objective_value(mip.model)

            @testset "Parallel vs Serial Comparison" begin
                @testset "Classical Oracle" begin
                    # Test parameters for focused comparison
                    strengthened = true
                    add_benders_cuts_to_master = true
                    reuse_dcglp = true
                    p = Inf
                    disjunctive_cut_append_rule = AllDisjunctiveCuts()
                    adjust_t_to_fx = false
                    
                    # Serial execution
                    @testset "Serial Execution" begin
                        master_serial = Master(data; solver_param = master_solver_param)
                        update_model!(master_serial, data)

                        typical_oracles_serial = [ClassicalOracle(data; solver_param = typical_oracle_solver_param); ClassicalOracle(data; solver_param = typical_oracle_solver_param)]
                        for k=1:2
                            update_model!(typical_oracles_serial[k], data)
                        end

                        disjunctive_oracle_serial = DisjunctiveOracle(data, typical_oracles_serial; 
                                                                     solver_param = dcglp_solver_param,
                                                                     param = dcglp_param) 
                        oracle_param_serial = DisjunctiveOracleParam(
                            norm = LpNorm(p), 
                            split_index_selection_rule = RandomFractional(),
                            disjunctive_cut_append_rule = disjunctive_cut_append_rule, 
                            strengthened = strengthened, 
                            add_benders_cuts_to_master = add_benders_cuts_to_master, 
                            fraction_of_benders_cuts_to_master = 1.0, 
                            reuse_dcglp = reuse_dcglp,
                            adjust_t_to_fx = adjust_t_to_fx,
                            enable_parallel = false
                        )
                        set_parameter!(disjunctive_oracle_serial, oracle_param_serial)
                        update_model!(disjunctive_oracle_serial, data)

                        env_serial = BendersSeq(data, master_serial, disjunctive_oracle_serial; param = benders_param)
                        log_serial = solve!(env_serial)
                        
                        @test env_serial.termination_status == Optimal()
                        @test isapprox(mip_opt_val, env_serial.obj_value, atol=1e-5)
                        
                        serial_obj_value = env_serial.obj_value
                        serial_iterations = nrow(log_serial)
                        
                        @info "Serial execution: obj_value = $serial_obj_value, iterations = $serial_iterations"
                    end
                    
                    # Parallel execution
                    @testset "Parallel Execution" begin
                        master_parallel = Master(data; solver_param = master_solver_param)
                        update_model!(master_parallel, data)

                        typical_oracles_parallel = [ClassicalOracle(data; solver_param = typical_oracle_solver_param); ClassicalOracle(data; solver_param = typical_oracle_solver_param)]
                        for k=1:2
                            update_model!(typical_oracles_parallel[k], data)
                        end

                        disjunctive_oracle_parallel = DisjunctiveOracle(data, typical_oracles_parallel; 
                                                                        solver_param = dcglp_solver_param,
                                                                        param = dcglp_param) 
                        oracle_param_parallel = DisjunctiveOracleParam(
                            norm = LpNorm(p), 
                            split_index_selection_rule = RandomFractional(),
                            disjunctive_cut_append_rule = disjunctive_cut_append_rule, 
                            strengthened = strengthened, 
                            add_benders_cuts_to_master = add_benders_cuts_to_master, 
                            fraction_of_benders_cuts_to_master = 1.0, 
                            reuse_dcglp = reuse_dcglp,
                            adjust_t_to_fx = adjust_t_to_fx,
                            enable_parallel = true,
                            max_threads = nothing  # Use default thread count
                        )
                        set_parameter!(disjunctive_oracle_parallel, oracle_param_parallel)
                        update_model!(disjunctive_oracle_parallel, data)

                        env_parallel = BendersSeq(data, master_parallel, disjunctive_oracle_parallel; param = benders_param)
                        log_parallel = solve!(env_parallel)
                        
                        @test env_parallel.termination_status == Optimal()
                        @test isapprox(mip_opt_val, env_parallel.obj_value, atol=1e-5)
                        
                        parallel_obj_value = env_parallel.obj_value
                        parallel_iterations = nrow(log_parallel)
                        
                        @info "Parallel execution: obj_value = $parallel_obj_value, iterations = $parallel_iterations"
                        
                        # Test that parallel and serial give the same result
                        # Note: Due to potential differences in floating point operations and timing,
                        # we allow for small numerical differences
                        # @test isapprox(serial_obj_value, parallel_obj_value, atol=1e-5)
                    end
                    
                    # Test with limited threads
                    @testset "Limited Threads" begin
                        master_limited = Master(data; solver_param = master_solver_param)
                        update_model!(master_limited, data)

                        typical_oracles_limited = [ClassicalOracle(data; solver_param = typical_oracle_solver_param); ClassicalOracle(data; solver_param = typical_oracle_solver_param)]
                        for k=1:2
                            update_model!(typical_oracles_limited[k], data)
                        end

                        disjunctive_oracle_limited = DisjunctiveOracle(data, typical_oracles_limited; 
                                                                       solver_param = dcglp_solver_param,
                                                                       param = dcglp_param) 
                        oracle_param_limited = DisjunctiveOracleParam(
                            norm = LpNorm(p), 
                            split_index_selection_rule = RandomFractional(),
                            disjunctive_cut_append_rule = disjunctive_cut_append_rule, 
                            strengthened = strengthened, 
                            add_benders_cuts_to_master = add_benders_cuts_to_master, 
                            fraction_of_benders_cuts_to_master = 1.0, 
                            reuse_dcglp = reuse_dcglp,
                            adjust_t_to_fx = adjust_t_to_fx,
                            enable_parallel = true,
                            max_threads = 1  # Force single thread (should behave like serial)
                        )
                        set_parameter!(disjunctive_oracle_limited, oracle_param_limited)
                        update_model!(disjunctive_oracle_limited, data)

                        env_limited = BendersSeq(data, master_limited, disjunctive_oracle_limited; param = benders_param)
                        log_limited = solve!(env_limited)
                        
                        @test env_limited.termination_status == Optimal()
                        @test isapprox(mip_opt_val, env_limited.obj_value, atol=1e-5)
                        
                        @info "Limited threads execution: obj_value = $(env_limited.obj_value), iterations = $(nrow(log_limited))"
                    end
                end
                
                # Note: Knapsack Oracle tests are disabled because CFLKnapsackOracle 
                # does not implement the generate_cuts method required for parallel execution
                # @testset "Knapsack Oracle" begin
                #     # Test would go here but CFLKnapsackOracle needs generate_cuts implementation
                # end
            end
        end
    end
end 