import gurobipy as gp
m = gp.Model()         # should NOT throw a license error
m.setParam("OutputFlag", 0)
x = m.addVar(name="x")
m.setObjective(x, gp.GRB.MAXIMIZE)
m.optimize()
print("OK")            # if you get here, the license is visible
