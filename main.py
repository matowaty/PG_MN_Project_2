from Matrix import Matrix
import matplotlib.pyplot as plt

def print_results_with_res_err(method, iteration, delta_time, res_err):
    print("metoda ", method)
    print("     liczba iteracji: ", iteration)
    print("     czas [s]: ", delta_time)
    print("     wartość błędu rezydualnego: ", res_err)

def residual_err_plot(method_0, method_1, res_arr, save_name):
    plt.title("Wartości błędu rezydualnego dla kolejnych iteracji:  " + method_0)
    plt.xlabel("numer iteracji")
    plt.ylabel("wartość błędu rezydualnego")

    plt.plot(res_arr, label=method_1)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_name)
    plt.show()



if __name__ == '__main__':
    #920 X 920 matrix

    size = 100
    A = Matrix(size)
    A.initialize_std_mat()

    b = Matrix(size)
    b.initialize_std_vec()

#zad B
    print("---- Zadanie B ----")
    iterations, time, res_norm, res_J = A.jacobi_solve(b)
    print_results_with_res_err("Jacobi", iterations, time, res_norm)
    plt.semilogy(res_J, label="Jacobi")
    iterations, time, res_norm, res_G = A.gauss_solve(b)
    print_results_with_res_err("Gauss-Seidl ", iterations, time, res_norm)
    plt.semilogy(res_G, label="Gauss-Seidle")
    plt.grid()
    plt.legend()
    plt.title("Porównanie normy rezidium w kolejnych iteracjach")
    plt.xlabel("numer iteracji")
    plt.ylabel("wartość błędu rezydualnego")
    plt.savefig("zad_B_Jac_Gaus_semilogy.png")
    plt.show()
    residual_err_plot("Jacobi", "Jacobi", res_J, "zad_B_Jac.png")
    residual_err_plot("Gauss-Seidl", "Gauss-Seidl", res_G, "zad_B_Gaus.png")

#zad C

    A = Matrix(size)
    A.initialize_std_mat(-2)
    print("---- Zadanie C ----")
    iterations, time, res_norm, res = A.jacobi_solve(b, max_iterations=100)
    print_results_with_res_err("Jacobiego", iterations, time, res_norm)
    plt.semilogy(res, label="Jacobi")

    iterations, time, res_norm, res = A.gauss_solve(b, max_iterations=100)
    print_results_with_res_err("Gauss-Seidla ", iterations, time, res_norm)
    plt.semilogy(res, label="Gauss-Seidle")
    plt.grid()
    plt.legend()
    plt.title("Wartości błędu rezydualnego dla kolejnych iteracji")
    plt.xlabel("numer iteracji")
    plt.ylabel("wartość błędu rezydualnego")
    plt.tight_layout()
    plt.savefig("zad_C_Jac_Gaus_inf.png")
    plt.show()

#zad D
    print("---- Zadanie D ----")
    time, norm = A.lu_solve(b)
    print("metoda   LU")
    print("     czas [s]: ", time)
    print("     wartość błędu rezydualnego: ", norm)

#zad E
    print("---- Zadanie E ----")
    N = [100, 500, 1000, 2000, 3000]
    sum_time_jac = []
    sum_time_gas = []
    sum_time_LU = []
    for i in N:
        A = Matrix(i)
        A.initialize_std_mat()

        b = Matrix(i)
        b.initialize_std_vec()

        iterations_J, time_J, res_norm_J, res_J = A.jacobi_solve(b)
        sum_time_jac.append(time_J)


        iterations_G, time_G, res_norm_G, res_G = A.gauss_solve(b)
        sum_time_gas.append(time_G)

        time_LU, norm_LU = A.lu_solve(b)
        sum_time_LU.append(time_LU)

    plt.title("Czas trwania poszczególnych algorytmów od liczby niewiadomych N")
    plt.xlabel("Liczba niewiadomych N")
    plt.ylabel("czas trwania [s]")
    plt.plot(N, sum_time_jac, label="Jacobi")
    plt.plot(N, sum_time_gas, label="Gauss-Seidl")
    plt.plot(N, sum_time_LU, label='LU')
    plt.legend()
    plt.grid()
    plt.savefig("zad_e_time_all.png")
    plt.show()


    plt.plot(N, sum_time_jac, label="Jacobi")
    plt.plot(N, sum_time_gas, label="Gauss-Seidl")
    plt.title("Czas trwania bez metody LU")
    plt.xlabel("Liczba niewiadomych N")
    plt.ylabel("czas trwania [s]")
    plt.legend()
    plt.grid()
    plt.savefig("zad_e_time_J_G.png")
    plt.show()

    print("---- Czasy ----")

    print("Jacobi: ")
    for i in range(len(N)):
        print("     iteracja " + str(i), "     czas [s]: ", sum_time_jac[i])

    print("Gauss: ")
    for i in range(len(N)):
        print("     iteracja " + str(i), "     czas [s]: ", sum_time_gas[i])

    print("LU: ")
    for i in range(len(N)):
        print("     iteracja " + str(i), "     czas [s]: ", sum_time_LU[i])





