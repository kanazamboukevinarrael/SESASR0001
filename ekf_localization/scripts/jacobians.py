
import sympy as sp

# State variables
x, y, theta = sp.symbols("x y theta")

# Control variables
v, omega = sp.symbols("v omega")

# Time interval
dt = sp.symbols("dt", positive=True)

# State and control vectors
state = sp.Matrix([x, y, theta])
control = sp.Matrix([v, omega])

# Deterministic velocity motion model
g = sp.Matrix([
    x - (v / omega) * sp.sin(theta)
      + (v / omega) * sp.sin(theta + omega * dt),

    y + (v / omega) * sp.cos(theta)
      - (v / omega) * sp.cos(theta + omega * dt),

    theta + omega * dt
])

# Jacobian with respect to the state
G = g.jacobian(state)

# Jacobian with respect to the control
V = g.jacobian(control)

print("Motion model g(x,u):")
sp.pprint(g)

print("\nJacobian G:")
sp.pprint(sp.simplify(G))

print("\nJacobian V:")
sp.pprint(sp.simplify(V))

# Landmark measurement model: range and bearing
mx, my = sp.symbols("m_x m_y")
h = sp.Matrix([
    sp.sqrt((mx - x) ** 2 + (my - y) ** 2),
    sp.atan2(my - y, mx - x) - theta,
])
H = h.jacobian(state)

print("\nMeasurement model h(x):")
sp.pprint(h)
print("\nJacobian H:")
sp.pprint(sp.simplify(H))
