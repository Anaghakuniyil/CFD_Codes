#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>  // Include OpenMP header

int main()
{
  int n, i, j, **flag, err, p;
  double L, dx, dy, dA, area_cir, area_sq, *xg, *yg, pi_err, pi_estimate;
  double xcen, ycen, rad, rad_sq, dist_sq;
  FILE *fid;
  struct timeval tv1, tv2;

  // length and area of circumscribing square
  L = 2.0;  area_sq = L*L;

  // number of grid points
  n = 16;

  // allocate grid
  xg = (double*) malloc(n*sizeof(double));
  yg = (double*) malloc(n*sizeof(double));

  dx = L/((double) n);
  for (i=0; i<n; i++)
    xg[i] = (i+0.5)*dx;

  dy = L/((double) n);
  for (i=0; i<n; i++)
    yg[i] = (i+0.5)*dy;

  // prepare to compute area of the circle
  dA = dx*dy; // area of one grid element
  xcen = 0.5*L; ycen = 0.5*L; rad = 0.5*L; // define center and radius of the circle
  rad_sq = rad*rad;
  area_cir = 0.0;

  // allocate flag array -- to be used for visualization
  flag = (int**) malloc(n*sizeof(int*));
  for(i=0; i<n; i++)
    flag[i] = (int*) malloc(n*sizeof(int));

  // initialize flag array to 0
  for(i=0; i<n; i++)
   for(j=0; j<n; j++)
     flag[i][j] = 0;

  // Start timing
  gettimeofday(&tv1, NULL);

  // Parallelize the main loop using OpenMP
  #pragma omp parallel for private(i, j, dist_sq) reduction(+:area_cir)
  for(i=0; i<n; i++)
  {
    for(j=0; j<n; j++)
    {
      dist_sq = (xg[i]-xcen)*(xg[i]-xcen) + (yg[j]-ycen)*(yg[j]-ycen);
      // check if (xg[i], yg[j]) is inside the circle
      if(dist_sq <= rad_sq)
      {
        area_cir += dA;
        flag[i][j] = 1;
      }
    }
  }

  // End timing
  gettimeofday(&tv2, NULL);

  pi_estimate = 4.0 * area_cir / area_sq;
  pi_err = fabs(M_PI - pi_estimate);

  // Calculate elapsed time
  long secs = tv2.tv_sec - tv1.tv_sec;
  long usecs = tv2.tv_usec - tv1.tv_usec;
  long mtime = ((secs)*1000 + usecs/1000) + 0.5;

  printf("\n Estimate of pi with %05d points is %.12e", n, pi_estimate);
  printf("\n Error    in pi with %05d points is %.12e", n, pi_err);
  printf("\n Elapsed time is %ld millisecs\n", mtime);

  // Write grid and flag data to files
  fid = fopen("xgrid.dat","w");
  for(i=0; i<n; i++)
    fprintf(fid, "%12.5e ", xg[i]);
  fclose(fid);

  fid = fopen("ygrid.dat","w");
  for(j=0; j<n; j++)
    fprintf(fid, "%12.5e ", yg[j]);
  fclose(fid);

  fid = fopen("flags.dat","w");
  for(i=0; i<n; i++)
  {
   fprintf(fid, "\n");
   for(j=0; j<n; j++)
    fprintf(fid, "%d ", flag[i][j]);
  }
  fclose(fid);

  printf("\n");

  // Free allocated memory
  for(i=0; i<n; i++)
    free(flag[i]);
  free(flag);
  free(xg);
  free(yg);

  return 0;
}