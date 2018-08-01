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
  int num_frames = 1<<9;
  int num_tasks = 1<<2;
  int num_subtasks = 1<<4;

  int num_codes = 1<<8;
  int num_pointers = 1<<4;
  int num_offset = 1<<6;

  int n_threads = 1<<8;
  int intersum_size = 64;
};

void calculate_vector_sum(FrameParams * frameParams, float * data, const long p, const int frame_no, vector<float>& pointers);
void execute_task_for_sum(FrameParams * frameParams, float * data, int p, int frame_no, vector<float>& pointers);
void execute_section_for_frame(FrameParams * frameParams, float * data, int frame_no, vector<float>& pointers);
void execute_section_wise_frames(const long n, const long m, float threshold,
  FrameParams * frameParams, float * data, vector<float>& pointers, vector<long>& result_row_ind);
void measure_result(FrameParams * frameParams, const long n, const long m, float threshold, 
vector<float> sum, vector<long>& result_row_ind);
void call_filter(const long n, const long m, float *data, const float threshold, vector<long> &result_row_ind);

void call_filter(const long n, const long m, float *data, const float threshold, vector<long> &result_row_ind)
{
  FrameParams * frameParams = new FrameParams;

  vector<float> pointers = vector<float>(n);

  omp_set_num_threads(frameParams->n_threads);
  
  // execution (sum) of 18 members
  execute_section_wise_frames(n, m, threshold, frameParams, data, pointers, result_row_ind);
}

void filter(const long n, const long m, float *data, const float threshold, vector<long> &result_row_ind) {
  
  call_filter(n, m, data, threshold, result_row_ind);

  //sort the values stored in the vector
  std::sort(result_row_ind.begin(),result_row_ind.end());
}

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
 * execution (sum) of 18 members
 */
void calculate_vector_sum(FrameParams * frameParams, float * data, const long p, const int frame_no, vector<float>& pointers)
{
  const size_t STRIP = frameParams->num_pointers*frameParams->num_offset;
  register float intersum = 0.0f;
  for(size_t ii = 0; ii < frameParams->num_codes*STRIP; ii+=STRIP) {
    for(size_t j = ii; j <= ii+frameParams->num_offset*frameParams->num_pointers - frameParams->intersum_size; j+=frameParams->intersum_size) {
      #pragma omp simd reduction(+:intersum)
      #pragma vector nontemporal
      #pragma vector aligned
      for(ptrdiff_t k = j; k < j+frameParams->intersum_size; k++) {
        intersum += data[k];
      }
    }
  }
  pointers[p] = intersum;
}

void execute_task_for_sum(FrameParams * frameParams, float * data, int st, const int frame_no, vector<float>& pointers)
{
  const size_t r = frameParams->num_codes*frameParams->num_pointers*frameParams->num_offset;
  #pragma omp parallel
  {
    #pragma omp master
    {
      for(ptrdiff_t i = st; i < st+frameParams->num_subtasks; i++) {
        #pragma omp task depend(out:data[(i-st)*r:r]) depend(in:data[(i-st+1)*r:r]) final(i==(st+frameParams->num_subtasks-1))
        {
          calculate_vector_sum(frameParams, &data[(i-st)*r], 
          frame_no*frameParams->num_tasks*frameParams->num_subtasks + i, frame_no, pointers);
        }
      }
    }
  }
}

void execute_section_for_frame(FrameParams * frameParams, float * data, const int frame_no, vector<float>& pointers)
{
  const size_t r = frameParams->num_subtasks*frameParams->num_codes*frameParams->num_pointers*frameParams->num_offset;
  #pragma omp parallel
  {
    #pragma omp master
    {
      for(ptrdiff_t i = 0; i < frameParams->num_tasks; i++) {
        #pragma omp task depend(out:data[i*r:r]) depend(in:data[(i+1)*r:r]) final(i==(frameParams->num_tasks-1))
        {
          execute_task_for_sum(frameParams, &data[i*r], i*frameParams->num_subtasks, frame_no, pointers);
        }
      }
    }
  }
}

void execute_task_wise_frames(FrameParams * frameParams, float * data, const int z, vector<float>& pointers)
{
  for(ptrdiff_t i = (z-1)*frameParams->num_frames/frameParams->n_threads; i < z*frameParams->num_frames/frameParams->n_threads; i++) {
    #pragma omp ordered
    execute_section_for_frame(frameParams, 
    &data[i*frameParams->num_tasks*frameParams->num_subtasks*
    frameParams->num_codes*frameParams->num_pointers*frameParams->num_offset], i, pointers);
  }
}

void execute_section_wise_frames(const long n, const long m, float threshold,
FrameParams * frameParams, float * data, vector<float>& pointers, 
vector<long>& result_row_ind)
{
  int k = 0;
  #pragma omp parallel num_threads(256) shared(k)
  {
    execute_task_wise_frames(frameParams, data, ++k, pointers);
  }
  // aggregate_sum_result(frameParams, n, m, threshold, sum, pointers);
  measure_result(frameParams, n, m, threshold, pointers, result_row_ind);
}