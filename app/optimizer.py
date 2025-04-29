from scipy.optimize import linprog
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from fastapi.responses import StreamingResponse
from scipy.spatial import ConvexHull



def simplex_solver(data):

    c = data["c"]
    A = data["A"]
    b = data["b"]
    operator = data["operador"]
    maximize = data["maximize"]
    print(operator)

    if maximize:
        c = [-x for x in c]

    A_ineq = []
    b_ineq = []

    for i in range(len(A)):
        for j in range(len(operator)):
            if operator[j] == 0:  # <=
                A_ineq.append(A[i])
                b_ineq.append(b[i])
            elif operator[j] == 1:  # >=
                A_ineq.append([-x for x in A[i]])
                b_ineq.append(-b[i])
            elif operator[j] == 2:  # =
                A_ineq.append(A[i])
                b_ineq.append(b[i])
                A_ineq.append([-x for x in A[i]])
                b_ineq.append(-b[i])


    result = linprog(c, A_ub=A_ineq, b_ub=b_ineq, method='simplex')

    if result.success:
        return {
            "status": "success",
            "solution": result.x.tolist(),
            "optimal_value": -result.fun if maximize else result.fun,
            "message": result.message
        }
    else:
        return {
            "status": "failure",
            "solution": None,
            "optimal_value": None,
            "message": result.message
        }



def solve_graphic(problem):

    # Converter input para arrays
    A = np.array(problem['A'], dtype=float)
    b = np.array(problem['b'], dtype=float)
    c = np.array(problem['c'], dtype=float)
    operadores = problem['operador']
    maximize = problem['maximize']

    fig, ax = plt.subplots()
    x_lim = np.max(b) * 1.2 if len(b) > 0 else 10
    x = np.linspace(0, x_lim, 400)

    for i, (row, bi) in enumerate(zip(A, b)):
        a1, a2 = row
        if abs(a2) > 1e-8:
            y = (bi - a1 * x) / a2
            ax.plot(x, y, label=f'Restrição {i+1}')
        else:
            xv = bi / a1
            ax.axvline(x=xv, label=f'Restrição {i+1}', linestyle='--')

    def viavel(p):
        if p[0] < -1e-6 or p[1] < -1e-6:
            return False
        for (row, bi, op) in zip(A, b, operadores):
            val = row.dot(p)
            if op == 0 and val > bi + 1e-6:  # <=
                return False
            if op == 1 and val < bi - 1e-6:  # >=
                return False
            if op == 2 and abs(val - bi) > 1e-6:  # =
                return False
        return True

    pontos = []
    n = len(A)
    for i in range(n):
        for j in range(i+1, n):
            M = A[[i, j]]
            if np.linalg.matrix_rank(M) == 2:
                rhs = np.array([b[i], b[j]])
                sol = np.linalg.solve(M, rhs)
                if viavel(sol):
                    pontos.append(sol)
    for i, (row, bi) in enumerate(zip(A, b)):
        a1, a2 = row
        if abs(a1) > 1e-8:
            sol = np.array([bi / a1, 0.0])
            if viavel(sol): pontos.append(sol)
        if abs(a2) > 1e-8:
            sol = np.array([0.0, bi / a2])
            if viavel(sol): pontos.append(sol)

    origem = np.array([0.0, 0.0])
    if viavel(origem):
        pontos.append(origem)

    if len(pontos) == 0:
        pontos = np.zeros((0,2))
    else:
        pontos = np.unique(np.array(pontos), axis=0)

    if pontos.shape[0] >= 3:
        hull = ConvexHull(pontos)
        poly = pontos[hull.vertices]
        ax.fill(poly[:, 0], poly[:, 1], color='lightblue', alpha=0.5, label='Região Viável')
        for simplex in hull.simplices:
            ax.plot(pontos[simplex, 0], pontos[simplex, 1], 'k-')
        hull_pts = pontos[hull.vertices]
    else:

        hull_pts = pontos


    if hull_pts.shape[0] > 0:

        valores = hull_pts.dot(c)
        idx = np.argmax(valores) if maximize else np.argmin(valores)
        p_opt = hull_pts[idx]
        z_opt = valores[idx]
        ax.plot(p_opt[0], p_opt[1], 'ro', label='Ótimo')
        ax.text(p_opt[0] + 0.2, p_opt[1] + 0.2,
                f'Ótimo ({p_opt[0]:.2f}, {p_opt[1]:.2f})', color='red')


    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Região de Soluções')
    ax.legend()
    ax.grid(True)


    all_x = np.concatenate([pontos[:,0], [0]]) if pontos.size else [0]
    all_y = np.concatenate([pontos[:,1], [0]]) if pontos.size else [0]
    max_lim = max(np.max(all_x), np.max(all_y), 1)
    ax.set_xlim(0, max_lim * 1.1)
    ax.set_ylim(0, max_lim * 1.1)


    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return StreamingResponse(buf, media_type='image/png')


