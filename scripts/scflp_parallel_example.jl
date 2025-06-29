using JuMP, DataFrames, Logging, CSV
using BendersDecomposition
using Printf  
using Statistics  
import BendersDecomposition: generate_cuts
include("$(dirname(@__DIR__))/example/scflp/data_reader.jl")
include("$(dirname(@__DIR__))/example/scflp/model.jl")

# Example script demonstrating parallel processing with SeparableOracle
# This script compares serial vs parallel execution for SCFLP problems
# With proper warm-up to account for Julia's JIT compilation

println("=== SCFLP Parallel Processing Example (Fair Comparison) ===")

# Load problem data
problem = read_stochastic_capacited_facility_location_problem("f25-c50-s128-r10-1")

# Initialize data object
dim_x = problem.n_facilities
dim_t = problem.n_scenarios
c_x = problem.fixed_costs
c_t = fill(1/problem.n_scenarios, problem.n_scenarios)
data = Data(dim_x, dim_t, problem, c_x, c_t)

println("Problem size: $(dim_x) facilities, $(dim_t) scenarios")

# Algorithm parameters
benders_param = BendersSeqParam(;
    time_limit = 200.0,
    gap_tolerance = 1e-6,
    verbose = false  # Turn off verbose for warm-up
)

# Solver parameters
master_solver_param = Dict("solver" => "CPLEX", "CPX_PARAM_EPINT" => 1e-9, "CPX_PARAM_EPRHS" => 1e-9, "CPX_PARAM_EPGAP" => 1e-9, "CPXPARAM_Threads" => 1)
typical_oracle_solver_param = Dict("solver" => "CPLEX", "CPX_PARAM_EPRHS" => 1e-9, "CPX_PARAM_NUMERICALEMPHASIS" => 1, "CPX_PARAM_EPOPT" => 1e-9)

# Function to run a single test
function run_test(test_name, enable_parallel, max_threads)
    master = Master(data; solver_param = master_solver_param)
    update_model!(master, data)
    
    if enable_parallel
        oracle_param = SeparableOracleParam(enable_parallel=true, max_threads=max_threads)
        oracle = SeparableOracle(data, ClassicalOracle(), data.problem.n_scenarios; 
                                solver_param = typical_oracle_solver_param,
                                oracle_param = oracle_param)
    else
        oracle = SeparableOracle(data, ClassicalOracle(), data.problem.n_scenarios; 
                                solver_param = typical_oracle_solver_param)
    end
    
    for j=1:oracle.N
        update_model!(oracle.oracles[j], data, j)
    end
    
    env = BendersSeq(data, master, oracle; param = benders_param)
    
    start_time = time()
    log = solve!(env)
    elapsed_time = time() - start_time
    
    return elapsed_time, env.obj_value, env.termination_status
end

# Warm-up phase
println("\n=== Warm-up Phase (JIT Compilation) ===")
println("Running warm-up for both serial and parallel modes...")

# Warm-up serial
print("Serial warm-up...")
run_test("Serial Warm-up", false, nothing)
println(" ✓")

# Warm-up parallel
print("Parallel warm-up...")
run_test("Parallel Warm-up", true, 4)
println(" ✓")

println("Warm-up completed. Starting fair performance comparison...")

# Enable verbose output for actual tests
benders_param_verbose = BendersSeqParam(;
    time_limit = 200.0,
    gap_tolerance = 1e-6,
    verbose = true
)

# Function to run test with verbose output
function run_test_verbose(test_name, enable_parallel, max_threads)
    master = Master(data; solver_param = master_solver_param)
    update_model!(master, data)
    
    if enable_parallel
        oracle_param = SeparableOracleParam(enable_parallel=true, max_threads=max_threads)
        oracle = SeparableOracle(data, ClassicalOracle(), data.problem.n_scenarios; 
                                solver_param = typical_oracle_solver_param,
                                oracle_param = oracle_param)
    else
        oracle = SeparableOracle(data, ClassicalOracle(), data.problem.n_scenarios; 
                                solver_param = typical_oracle_solver_param)
    end
    
    for j=1:oracle.N
        update_model!(oracle.oracles[j], data, j)
    end
    
    env = BendersSeq(data, master, oracle; param = benders_param_verbose)
    
    start_time = time()
    log = solve!(env)
    elapsed_time = time() - start_time
    
    return elapsed_time, env.obj_value, env.termination_status
end

# Actual Performance Tests (Multiple Runs for Statistical Significance)
num_runs = 3
println("\n=== Performance Tests ($(num_runs) runs each) ===")

# Serial execution tests
println("\n--- Serial Execution Tests ---")
serial_times = Float64[]
serial_objs = Float64[]

for i in 1:num_runs
    println("Serial Run $i:")
    time_taken, obj_val, status = run_test_verbose("Serial Run $i", false, nothing)
    push!(serial_times, time_taken)
    push!(serial_objs, obj_val)
    println("Serial Run $i completed in $(round(time_taken, digits=2)) seconds")
end

# Parallel execution tests  
println("\n--- Parallel Execution Tests ---")
parallel_times = Float64[]
parallel_objs = Float64[]

for i in 1:num_runs
    println("Parallel Run $i:")
    time_taken, obj_val, status = run_test_verbose("Parallel Run $i", true, 4)
    push!(parallel_times, time_taken)
    push!(parallel_objs, obj_val)
    println("Parallel Run $i completed in $(round(time_taken, digits=2)) seconds")
end

# Statistical Analysis
println("\n=== Statistical Results Summary ===")
serial_mean = mean(serial_times)
serial_std = std(serial_times)
parallel_mean = mean(parallel_times)
parallel_std = std(parallel_times)
speedup = serial_mean / parallel_mean
obj_diff = abs(mean(serial_objs) - mean(parallel_objs))

println("Available threads: $(Threads.nthreads())")
println("\nSerial Execution:")
println("  Mean time: $(round(serial_mean, digits=2))s ± $(round(serial_std, digits=2))s")
println("  Times: $(round.(serial_times, digits=2))")
println("  Mean objective: $(round(mean(serial_objs), digits=6))")

println("\nParallel Execution:")
println("  Mean time: $(round(parallel_mean, digits=2))s ± $(round(parallel_std, digits=2))s") 
println("  Times: $(round.(parallel_times, digits=2))")
println("  Mean objective: $(round(mean(parallel_objs), digits=6))")

println("\nPerformance Comparison:")
println("  Speedup: $(round(speedup, digits=2))x")
println("  Objective difference: $(round(obj_diff, digits=8)) (should be ~0)")
println("  Parallel efficiency: $(round(100 * speedup / min(4, Threads.nthreads()), digits=1))%")

if obj_diff < 1e-6
    println("✓ Parallel and serial results match")
else
    println("⚠ Warning: Parallel and serial results differ significantly")
end

if speedup > 1.1
    println("✓ Parallel execution provided meaningful speedup")
    if speedup > 2.0
        println("🚀 Excellent speedup achieved!")
    end
else
    println("⚠ Limited speedup observed")
    println("  This may be due to:")
    println("  - Small problem size (insufficient parallelization benefit)")
    println("  - Limited available threads")
    println("  - Solver overhead dominating computation time")
end
