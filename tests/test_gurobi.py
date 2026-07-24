import pulp

# 1. Define a simple optimization problem (Maximize 3x + 2y)
prob = pulp.LpProblem("Test_Gurobi_Integration", pulp.LpMaximize)

# 2. Create continuous decision variables
x = pulp.LpVariable('x', lowBound=0)
y = pulp.LpVariable('y', lowBound=0)

# 3. Add objective function
prob += 3 * x + 2 * y, "Objective_Function"

# 4. Add constraints
prob += x + y <= 4, "Constraint_1"
prob += 2 * x + y <= 5, "Constraint_2"

print("--- Attempting to solve using GUROBI API ---")
try:
    # Use the native Gurobi Python API interface (recommended)
    solver = pulp.GUROBI(msg=True)
    prob.solve(solver)
    
    # Print the resulting status and variable values
    print(f"Solver Status: {pulp.LpStatus[prob.status]}")
    if prob.status == pulp.LpStatusOptimal:
        print(f"Optimal Value for x: {pulp.value(x)}")
        print(f"Optimal Value for y: {pulp.value(y)}")
        print(f"Maximized Objective: {pulp.value(prob.objective)}")
    else:
        print("Gurobi was called but did not find an optimal solution.")

except pulp.PulpSolverError:
    print("\n[ERROR]: Direct GUROBI API solver not available.")
    print("Attempting fallback to command-line interface (GUROBI_CMD)...")
    
    try:
        # Fallback if the direct API link fails but the CLI is in your system PATH
        prob.solve(pulp.GUROBI_CMD(msg=True))
        print(f"Solver Status (CMD): {pulp.LpStatus[prob.status]}")
        print(f"Maximized Objective (CMD): {pulp.value(prob.objective)}")
    except pulp.PulpSolverError:
        print("\n[CRITICAL]: Both GUROBI and GUROBI_CMD interfaces failed.")
        print("Please check that your Gurobi license is valid and environment variables are set correctly.")

