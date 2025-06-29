# Disjunctive Benders Decomposition

[![Julia](https://img.shields.io/badge/julia-v1.10.4-blue.svg)](https://julialang.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A Julia implementation of disjunctive Benders decomposition algorithms for solving mixed-integer programming problems, developed as part of research on "Disjunctive Benders Decomposition".

## Overview

This repository contains the source code and computational experiments for disjunctive Benders decomposition methods. The implementation extends classical Benders decomposition by incorporating disjunctive cuts to improve convergence and solution quality for mixed-integer programming problems.

## Key Features

- **Multiple Algorithm Variants**: Implementation of sequential and callback-based Benders decomposition algorithms
- **Disjunctive Cuts**: Integration of disjunctive programming techniques for enhanced cut generation
- **Flexible Oracle System**: Modular oracle design supporting different subproblem types (typical, disjunctive, separable)
- **Parallel Processing**: Multi-threaded execution support for separable oracle problems using `Threads.@threads`
- **Comprehensive Testing**: Extensive test suite with multiple problem instances
- **Multiple Problem Types**: Support for facility location problems (UFLP, CFLP, SCFLP), network design (MCNDP), and other optimization problems

## Algorithm Implementations

### Core Algorithms
- `BendersSeq`: Sequential Benders decomposition
- `BendersSeqInOut`: Sequential variant with in-out technique
- `BendersBnB`: Branch-and-bound Benders decomposition  
- `Dcglp`: Disjunctive Cut Generating Linear Program
- `SpecializedBendersSeq`: Specialized sequential implementation

### Oracle Types
- `ClassicalOracle`: Traditional Benders subproblem oracle
- `KnapsackOracle`: Knapsack technique based oracle
- `DisjunctiveOracle`: Disjunctive programming-based oracle
- `SeparableOracle`: Oracle for separable subproblems

## Problem Examples

The `example/` directory contains implementations for several classic optimization problems featured in the paper:
- **UFLP**: Uncapacitated Facility Location Problem
- **SNIP**: Stochastic Network Interdiction Problem

We are actively developing this repository into a professional package. The following implementations are currently in progress:
- **CFLP**: Capacitated Facility Location Problem  
- **SCFLP**: Stochastic Capacitated Facility Location Problem
- **MCNDP**: Multi-Commodity Network Design Problem

## Installation

To set up the project:
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Usage

We provide several scripts to run the algorithms on different problem instances. Please refer to the `scripts/` directory for more details.

### Parallel Processing

For problems with separable subproblems, you can enable parallel processing to accelerate computation:

```julia
# Create a parallel-enabled separable oracle
oracle_param_parallel = SeparableOracleParam(enable_parallel=true, max_threads=4)
oracle = SeparableOracle(data, ClassicalOracle(), n_scenarios; 
                        solver_param = solver_param,
                        oracle_param = oracle_param_parallel)

# The oracle will now use up to 4 threads for parallel subproblem solving
```

**Performance Notes**: 
- Parallel processing is most beneficial for problems with many scenarios (hundreds) and significant per-scenario computation time (several seconds)
- Set `max_threads=nothing` to use all available CPU cores
- Ensure your solver (e.g., CPLEX) supports concurrent usage across threads

## Testing

Run the test suite:
```bash
julia test/runtests.jl
```

Or run specific tests:
```bash
./test/runtests.sh
```

The test suite includes:
1. Sequential typical Benders decomposition
2. Sequential in-out typical Benders decomposition  
3. Sequential disjunctive Benders decomposition
4. Callback typical Benders decomposition
5. Callback disjunctive Benders decomposition
6. Specialized sequential Benders decomposition

## Project Structure

```
├── src/
│   ├── algorithms/          # Core decomposition algorithms
│   ├── modules/            # Oracle implementations and components
│   ├── utils/              # Utility functions and helpers
│   └── types.jl            # Type definitions and exports
├── test/                   # Comprehensive test suite
├── example/                # Problem-specific implementations
│   ├── uflp/              # Uncapacitated facility location
│   ├── cflp/              # Capacitated facility location
│   ├── scflp/             # Stochastic capacitated facility location
│   ├── mcndp/             # Multi-commodity network design
│   └── snip/              # Stochastic network interdiction
└── Project.toml           # Julia project configuration
```

## Contributing

This repository is actively under development and we welcome contributions! Feel free to submit issues for bugs or feature requests, and pull requests for code changes. For major modifications, please open an issue first to discuss your proposal. We appreciate all contributions, from bug fixes to documentation improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.








