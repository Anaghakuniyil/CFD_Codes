#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h> // Include OpenMP header

void write_1darr_to_screen(double *arr, int n)
{
    int i;
    for(i = 0; i < n; i++)
        printf("%.2e    ", arr[i]);
}

double calc_Jfun(double *y, double *yhat, int m)
{
    int i;
    double Jfun = 0.0;

    #pragma omp parallel for reduction(+:Jfun) // Parallelize with reduction
    for(i = 0; i < m; i++)
    {
        Jfun += (y[i] - yhat[i]) * (y[i] - yhat[i]);
    }
    Jfun = 0.5 * Jfun / (double)m;
    return Jfun;
}

void calc_yhat(double **x, double *theta, double *yhat, int m, int n)
{
    int i, j;

    #pragma omp parallel for private(j) // Parallelize outer loop
    for(i = 0; i < m; i++)
    {
        yhat[i] = theta[0];
        for(j = 1; j < n + 1; j++)
        {
            yhat[i] += theta[j] * x[i][j];
        }
    }
}

void calc_new_theta(double **x, double *theta, double *yhat, double *y, double learning_rate, int m, int n)
{
    int i, j;
    double gradJ;

    #pragma omp parallel for private(i, gradJ) // Parallelize outer loop
    for(j = 0; j < n + 1; j++)
    {
        gradJ = 0.0;
        for(i = 0; i < m; i++)
        {
            gradJ += (yhat[i] - y[i]) * x[i][j];
        }
        theta[j] = theta[j] - learning_rate * gradJ;
    }
}

void read_data_from_file(char *fname, double **x, double *y, int m, int n)
{
    FILE *fid;
    int i, j;

    fid = fopen("Student_Performance.csv", "r");
    for(i = 0; i < m; i++)
    {
        x[i][0] = 1.0; // dummy feature
        for(j = 1; j < n + 1; j++)
        {
            fscanf(fid, "%lf,", &x[i][j]);
        }
        fscanf(fid, "%lf\n", &y[i]);
    }
    fclose(fid);
}

void scale_and_translate_data(double **x, double *y, double *means, double *sdevs, int m, int n)
{
    int i, j;

    // Calculate means and standard deviations
    #pragma omp parallel for private(j) // Parallelize outer loop
    for(j = 0; j < n + 2; j++)
    {
        means[j] = 0.0;
        sdevs[j] = 0.0;
    }

    #pragma omp parallel for private(j) reduction(+:means[:n+2], sdevs[:n+2]) // Parallelize with reduction
    for(i = 0; i < m; i++)
    {
        for(j = 0; j < n + 1; j++)
        {
            means[j] += x[i][j];
            sdevs[j] += x[i][j] * x[i][j];
        }
        means[n + 1] += y[i];
        sdevs[n + 1] += y[i] * y[i];
    }

    for(j = 0; j < n + 2; j++)
    {
        means[j] = means[j] / (double)m;
        sdevs[j] = sqrt(sdevs[j] / (double)m - means[j] * means[j]);
    }

    // Normalize the data
    #pragma omp parallel for private(j) // Parallelize outer loop
    for(i = 0; i < m; i++)
    {
        for(j = 1; j < n + 1; j++) // Do not modify the dummy feature
        {
            x[i][j] = (x[i][j] - means[j]) / sdevs[j];
        }
        y[i] = (y[i] - means[n + 1]) / sdevs[n + 1];
    }
}

void undo_scale_and_translate_data(double **x, double *y, double *means, double *sdevs, int m, int n)
{
    int i, j;

    #pragma omp parallel for private(j) // Parallelize outer loop
    for(i = 0; i < m; i++)
    {
        for(j = 1; j < n + 1; j++) // Do not modify the dummy feature
        {
            x[i][j] = x[i][j] * sdevs[j] + means[j];
        }
        y[i] = y[i] * sdevs[n + 1] + means[n + 1];
    }
}

int main()
{
    int m, n, i, j, k, max_iter;
    double **x, *y, *yhat, *theta, tol, learning_rate;
    double *means, *sdevs, Jfun0, Jfun, Jfun_old;
    FILE *fid;

    // Read data
    m = 50; // Size of data
    n = 4;  // Number of features

    // Allocate memory
    x = (double **)malloc(m * sizeof(double *));
    for(i = 0; i < m; i++)
        x[i] = (double *)malloc((n + 1) * sizeof(double));

    y = (double *)malloc(m * sizeof(double));
    means = (double *)malloc((n + 2) * sizeof(double));
    sdevs = (double *)malloc((n + 2) * sizeof(double));
    theta = (double *)malloc((n + 1) * sizeof(double));
    yhat = (double *)malloc(m * sizeof(double));

    // Initialize parameters
    for(j = 0; j < n + 1; j++)
        theta[j] = 0.0;

    // Read and preprocess data
    read_data_from_file("Student_Performance.csv", x, y, m, n);
    scale_and_translate_data(x, y, means, sdevs, m, n);

    // Gradient descent parameters
    max_iter = 40;
    tol = 1.0e-6;
    learning_rate = 0.01;

    // Calculate initial cost
    calc_yhat(x, theta, yhat, m, n);
    Jfun_old = calc_Jfun(y, yhat, m);
    Jfun0 = Jfun_old;
    printf("\nIter: %03d: Jfun = %e", -1, Jfun_old);

    // Gradient descent loop
    for(k = 0; k < max_iter; k++)
    {
        calc_new_theta(x, theta, yhat, y, learning_rate, m, n);
        calc_yhat(x, theta, yhat, m, n);
        Jfun = calc_Jfun(y, yhat, m);
        printf("\nIter: %03d: Jfun = %0.12e", k, Jfun);
        if(fabs(Jfun - Jfun_old) < tol)
            break;
        else
        {
            Jfun_old = Jfun;
        }
    }

    // Free memory
    for(i = 0; i < m; i++)
        free(x[i]);
    free(x);
    free(y);
    free(yhat);
    free(theta);
    free(means);
    free(sdevs);

    printf("\n");
    return 0;
}