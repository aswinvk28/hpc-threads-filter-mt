#include <vector>
#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <mkl.h>
#include <cmath>

using namespace std;

/**
 * (1<<8) section offsets, (1<<4) task offsets, (1<<4) offsets, (1<<2) pointers, 4 sum offsets
 */
struct FrameParams
{
  int num_frames = 1<<6;
  int num_tasks = 1<<4;
  int num_subtasks = 1<<5;

  int num_objects = 1<<3;
  int num_codes = 1<<3;
  int num_pointers = 1<<2;
  int num_offset = 1<<10;

  int n_threads = 1<<6;
  int intersum_size = 512;
  int num_subindex = 0;
  int num_product = 0;
  int num_division = 0;
  int num_index = 0;
  unsigned long num_superindex = 0;
  FrameParams()
	{
    num_subindex = num_tasks * num_subtasks;
    num_product = num_tasks * num_subtasks * num_offset;
    num_division = num_codes * num_pointers;
    num_index = num_division * num_product;
    num_superindex = num_index * num_objects;
	}
};

void calculate_vector_sum(FrameParams * frameParams, float * data, const long p, const int frame_no, vector<vector<float>>& pointers);
void initialise_frames(float * data, FrameParams * frameParams);
void calculate_row_sum(FrameParams * frameParams);
void calculate_odd_row_sum(FrameParams * frameParams);
void calculate_next_odd_row_sum(FrameParams * frameParams);
void calculate_next2_odd_row_sum(FrameParams * frameParams);
void calculate_next3_odd_row_sum(FrameParams * frameParams);
void execute_task_for_sum(FrameParams * frameParams, float * data, int p, int frame_no, vector<vector<float>>& pointers);
void execute_section_for_frame(FrameParams * frameParams, float * data, int frame_no, vector<vector<float>>& pointers);
void execute_section_wise_frames(FrameParams * frameParams, float * data, vector<vector<float>>& pointers);
// void aggregate_sum_result(FrameParams * frameParams, const long n, const long m, 
// float threshold, vector<float>& sum, vector<vector<float>> pointers);
void measure_result(FrameParams * frameParams, const long n, const long m, float threshold, 
vector<float> sum, vector<long> &result_row_ind);
void call_filter(const long n, const long m, float *data, const float threshold, std::vector<long> &result_row_ind);

void call_filter(const long n, const long m, float *data, const float threshold, vector<long> &result_row_ind)
{
  FrameParams * frameParams = new FrameParams;

  vector<vector<float>> pointers = vector<vector<float>>(1, vector<float>(0));
  
  // execution (sum) of 18 members
  execute_section_wise_frames(frameParams, data, pointers);

  // aggregate_sum_result(frameParams, n, m, threshold, sum, pointers);
  measure_result(frameParams, n, m, threshold, pointers[0], result_row_ind);
}

void filter(const long n, const long m, float *data, const float threshold, vector<long> &result_row_ind) {
  
  call_filter(n, m, data, threshold, result_row_ind);

  //sort the values stored in the vector
  std::sort(result_row_ind.begin(),result_row_ind.end());
}

// void aggregate_sum_result(FrameParams * frameParams, const long n, const long m, 
// float threshold, vector<float>& sum, vector<vector<float>> pointers)
// {
//   #pragma ivdep
//   for(long i = 0; i < n; i++) {
//     sum.push_back(pointers[0][i] + pointers[1][i] + 
//     pointers[2][i] + pointers[3][i] + 
//     pointers[4][i] + pointers[5][i] + 
//     pointers[6][i] + pointers[7][i]);
//   }
// }

void measure_result(FrameParams * frameParams, const long n, const long m, 
float threshold, vector<float> sum, vector<long> &result_row_ind)
{
  for(size_t i = 0; i < n; i++) {
    if(sum[i] > threshold) {
      result_row_ind.push_back(i);
    }
  }
}

//helper function , refer instructions on the lab page
void append_vec(std::vector<long> &v1, std::vector<long> &v2) {
  v1.insert(v1.end(),v2.begin(),v2.end());
}

/**
 * (1<<3) offset_equator1, (1<<4) offset_equator2
 */
void initialise_frames(float * data, FrameParams * frameParams)
{
  // fill(pointers[0].begin(), pointers[0].end(), 0.0);
  // fill(pointers[1].begin(), pointers[1].end(), 0.0);
  // fill(pointers[2].begin(), pointers[2].end(), 0.0);
  // fill(pointers[3].begin(), pointers[3].end(), 0.0);
  // fill(pointers[4].begin(), pointers[4].end(), 0.0);
  // fill(pointers[5].begin(), pointers[5].end(), 0.0);
  // fill(pointers[6].begin(), pointers[6].end(), 0.0);
  // fill(pointers[7].begin(), pointers[7].end(), 0.0);
}

/**
 * execution (sum) of 18 members
 */
void calculate_vector_sum(FrameParams * frameParams, float * data, const long p, const int frame_no, vector<vector<float>>& pointers)
{
  const size_t STRIP = frameParams->num_pointers*frameParams->num_offset;
  float intersum = 0.0f;
  for(size_t ii = 0; ii < frameParams->num_codes*STRIP; ii+=STRIP) {
    for(size_t j = ii; j <= ii+frameParams->num_offset*frameParams->num_pointers - frameParams->intersum_size; j+=frameParams->intersum_size) {
      #pragma omp simd reduction(+:intersum)
      #pragma vector nontemporal
      for(ptrdiff_t k = j; k < j+frameParams->intersum_size; k++) {
        intersum += data[k];
      }
    }
  }
  #pragma omp critical
  {
    pointers[0].push_back(intersum);
  }
}

void execute_task_for_sum(FrameParams * frameParams, float * data, int st, const int frame_no, vector<vector<float>>& pointers)
{
  const size_t r = frameParams->num_division*frameParams->num_offset;
  #pragma omp parallel
  {
    #pragma omp master
    {
      for(ptrdiff_t i = st; i < st+frameParams->num_subtasks; i++) {
        #pragma omp task depend(out:data[(i-st)*r:r]) depend(in:data[(i-st+1)*r:r]) final(i==(st+frameParams->num_subtasks-1))
        {
          calculate_vector_sum(frameParams, &data[(i-st)*r], i, frame_no, pointers);
        }
      }
    }
  }
}

void execute_section_for_frame(FrameParams * frameParams, float * data, const int frame_no, vector<vector<float>>& pointers)
{
  const size_t r = frameParams->num_subtasks*frameParams->num_division*frameParams->num_offset;
  #pragma omp parallel
  {
    #pragma omp master
    {
      for(ptrdiff_t i = 1; i <= frameParams->num_tasks; i++) {
        #pragma omp task depend(out:data[i*r:r]) depend(in:data[(i+1)*r:r]) final(i==(frameParams->num_tasks-1))
        {
          execute_task_for_sum(frameParams, &data[(i-1)*r], (i-1)*frameParams->num_subtasks, frame_no, pointers);
        }
      }
    }
  }
}

void execute_task_wise_frames(FrameParams * frameParams, float * data, const int z, vector<vector<float>>& pointers)
{
  for(ptrdiff_t i = (z-1)*frameParams->num_frames/frameParams->n_threads; i < z*frameParams->num_frames/frameParams->n_threads; i++) {
    #pragma omp ordered
    execute_section_for_frame(frameParams, 
    &data[i*frameParams->num_index], i, pointers);
  }
}

void execute_section_wise_frames(FrameParams * frameParams, float * data, vector<vector<float>>& pointers)
{
  #pragma omp parallel num_threads(64)
  {
    int k = omp_get_thread_num() + 1;
    execute_task_wise_frames(frameParams, data, k, pointers);
  }
}