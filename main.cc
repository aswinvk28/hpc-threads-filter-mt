#include <cstdlib>
#include <cstdio>
#include <omp.h>
#include <mkl.h>
#include <vector>
#include <algorithm>

void filter(const long n, const long m, float *data, const float threshold, std::vector<long> &result_row_ind, int argc, char** argv);

//reference function to verify data
void filter_ref(const long n, const long m, float *data, const float threshold, std::vector<long> &result_row_ind) {
  float sum;
  for(long i = 0; i < n; i++){
    sum = 0.0f;
    for(long j = 0; j < m; j++) {
      sum+=data[i*m+j];
    } 
    if(sum > threshold) 
      result_row_ind.push_back(i);
  }
  std::sort(result_row_ind.begin(),result_row_ind.end());
}

int main(int argc, char** argv) {
  float threshold = 0.5;
  if(argc < 2) {
    threshold = 0.5;
  } else {
    threshold = atof(argv[1]);
  } 

  int _n_shift = atoi(argv[2]);
  int _m_shift = atoi(argv[3]);

  long _n = 1<<_n_shift;
  long _m = 1<<_m_shift;
  
  const long n = 1UL<<10; //rows
  const long m = 1UL<<10; //columns
  
  float *data = (float *) malloc((long long)sizeof(float)*_n*_m);
  long random_seed = (long)(omp_get_wtime()*1000.0) % 1000L;
  VSLStreamStatePtr rnStream;
  vslNewStream( &rnStream, VSL_BRNG_MT19937, random_seed);
  vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rnStream, _m*_n, &data[0], -1.0, 1.0);

  //initialize 2D data
// #pragma omp parallel for 
//   for(long i =0; i < n; i++)
    // vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rnStream, m*n, &data[0], -1.0, 1.0);

  std::vector<long> ref_result_row_ind; 
  //compute the refernce data using unoptimized refernce function defined above
  filter_ref(_n, _m, data, threshold, ref_result_row_ind);
  
  //compute actual data using the function defined in worker.cc and get the timing
  std::vector<long> result_row_ind; 
  const double t0 = omp_get_wtime();
  filter(_n, _m, data, threshold, result_row_ind, argc, argv);
  const double t1 = omp_get_wtime();
  
  //verify the actual data and the refernce data
  if(ref_result_row_ind.size() != result_row_ind.size()) {
    // Result sizes did not match
    printf("Error: The reference and result vectors have different sizes: %ld %ld",ref_result_row_ind.size(), result_row_ind.size());
  } else {
    bool passed = true;
    for(long i = 0; i < ref_result_row_ind.size(); i++) {
      passed &= (ref_result_row_ind[i] == result_row_ind[i]);
    } 
    if(passed) {
      // Printing perf
      printf("Time: %f\n", t1-t0);
    } else  {
      // Results did not match
      printf("Error: The reference and result vectors did not match");
    }
  }
}