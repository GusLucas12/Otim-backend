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
    A = np.array(problem['A'])
    b = np.array(problem['b'])

    fig, ax = plt.subplots()

    x = np.linspace(0, 30, 400)
    restricoes = []

    for i in range(len(A)):
        if A[i][1] != 0:
            y = (b[i] - A[i][0] * x) / A[i][1]
            restricoes.append((x, y))
            ax.plot(x, y, label=f'Restrição {i + 1}')
        else:
            x_val = b[i] / A[i][0]
            ax.axvline(x=x_val, label=f'Restrição {i + 1}', linestyle='--')

    pontos = []
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            A_eq = np.array([A[i], A[j]])
            b_eq = np.array([b[i], b[j]])
            if np.linalg.matrix_rank(A_eq) == 2:
                ponto = np.linalg.solve(A_eq, b_eq)
                if all(ponto >= 0) and all(np.matmul(A, ponto) <= b + 1e-5):
                    pontos.append(ponto)

    for i in range(len(A)):
        if A[i][1] != 0:
            y_intercept = b[i] / A[i][1]
            if y_intercept >= 0:
                p = np.array([0, y_intercept])
                if all(np.matmul(A, p) <= b + 1e-5):
                    pontos.append(p)
        if A[i][0] != 0:
            x_intercept = b[i] / A[i][0]
            if x_intercept >= 0:
                p = np.array([x_intercept, 0])
                if all(np.matmul(A, p) <= b + 1e-5):
                    pontos.append(p)

    if pontos:
        pontos = np.array(pontos)
        if len(pontos) >= 3:
            hull = ConvexHull(pontos)
            for simplex in hull.simplices:
                ax.plot(pontos[simplex, 0], pontos[simplex, 1], 'k-')
            ax.fill(pontos[hull.vertices, 0], pontos[hull.vertices, 1], 'lightblue', alpha=0.5, label='Região Viável')
        else:
            ax.plot(pontos[:, 0], pontos[:, 1], 'o', color='lightblue')

    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.legend()
    ax.grid(True)
    ax.set_title('Região de Soluções')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")



