// [[Rcpp::plugins(openmp)]]
// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
NumericVector vec_add_omp(NumericVector A, NumericVector B){
  int N = A.size();
  NumericVector C(N);
  #pragma omp parallel for
  for(int i=0;i<N;i++) C[i] = A[i] + B[i];
  return C;
}
