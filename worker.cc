#include <vector>
#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <mkl.h>

using namespace std;

/**
 * (1<<8) section offsets, (1<<4) task offsets, (1<<4) offsets, (1<<2) pointers, 4 sum offsets
 */
struct FrameParams
{
  int num_frames = 1<<7;
  int num_tasks = 1<<5;
  int num_subtasks = 1<<3;

  int num_codes = 1<<4;
  int num_pointers = 1<<6;
  int num_offset = 1<<8;

  int n_threads = 1<<7;
  int intersum_size = 1<<14;
  vector<float> pointers;
};

/**
 * execution (sum) of 18 members
 */
void calculate_vector_sum(FrameParams * frameParams, float * data, const long p, const int frame_no)
{
  const size_t STRIP = frameParams->num_pointers*frameParams->num_offset;
  register float intersum = 0.0f;
  #pragma vector always
  for(size_t ii = 0; ii < frameParams->num_codes*STRIP; ii+=STRIP) {
    for(size_t j = ii; j <= ii+frameParams->num_offset*frameParams->num_pointers - frameParams->intersum_size; j+=frameParams->intersum_size) {
      #pragma omp simd reduction(+:intersum)
      #pragma vector nontemporal
      #pragma vector aligned
      #pragma prefetch data
      for(ptrdiff_t k = j; k < j+frameParams->intersum_size; k++) {
        intersum += data[k];
      }
    }
  }
  #pragma omp atomic write
  frameParams->pointers[p] = intersum;
}

// aggregate the result from segments of totals
void aggregate_result(const long n, const long m, FrameParams * frameParams, short int k)
{
  #pragma ivdep
  #pragma vector aligned
  #pragma vector nontemporal
  for(int i=0; i < frameParams->num_frames*frameParams->num_tasks*frameParams->num_subtasks; i+=k) {
    frameParams->pointers[i] += frameParams->pointers[i+k/2];
  }
}

// measure the aggregated result
void measure_result(FrameParams * frameParams, const long n, const long m, float threshold, vector<long> &result_row_ind)
{
  #pragma vector aligned
  #pragma vector always
  #pragma omp for ordered
  for(size_t i = 0; i < frameParams->num_frames*frameParams->num_tasks*frameParams->num_subtasks; i++) {
    if(frameParams->pointers[i] > threshold) {
      result_row_ind.push_back(i);
    }
  }
}

void execute_task_for_sum(FrameParams * frameParams, float * data, int st, const int frame_no)
{
  const size_t r = frameParams->num_codes*frameParams->num_pointers*frameParams->num_offset;
  #pragma omp parallel
  {
    #pragma omp single nowait
    {
      #pragma vector always
      for(ptrdiff_t i = st; i < st+frameParams->num_subtasks; i++) {
        #pragma omp task depend(out:data[(i-st)*r:r]) depend(in:data[(i-st+1)*r:r]) final(i==(st+frameParams->num_subtasks-1))
        {
	  calculate_vector_sum(frameParams, &data[(i-st)*r], frame_no*frameParams->num_tasks*frameParams->num_subtasks+i, frame_no);
        }
      }
    }
  }
}

void execute_section_for_frame(FrameParams * frameParams, float * data, const int frame_no)
{
  const size_t r = frameParams->num_subtasks*frameParams->num_codes*frameParams->num_pointers*frameParams->num_offset;
  #pragma omp parallel
  {
    #pragma omp single nowait
    {
      #pragma vector always	
      for(ptrdiff_t i = 1; i <= frameParams->num_tasks; i++) {
        #pragma omp task depend(out:data[i*r:r]) depend(in:data[(i+1)*r:r]) final(i==(frameParams->num_tasks-1))
        {
          execute_task_for_sum(frameParams, &data[(i-1)*r], (i-1)*frameParams->num_subtasks, frame_no);
        }
      }
    }
  }
}

void execute_task_wise_frames(FrameParams * frameParams, float * data, const int z)
{
  #pragma vector always
  for(ptrdiff_t i = (z-1)*frameParams->num_frames/frameParams->n_threads; i < z*frameParams->num_frames/frameParams->n_threads; i++) {
    execute_section_for_frame(frameParams, 
    &data[i*frameParams->num_codes*frameParams->num_pointers*
    frameParams->num_offset*frameParams->num_tasks*frameParams->num_subtasks], i);
  }
}

void execute_section_wise_frames(FrameParams * frameParams, float * data)
{
  #pragma omp parallel num_threads(128)
  {
    int k = omp_get_thread_num() + 1;
    execute_task_wise_frames(frameParams, data, k);
  }
}

// primary function
void filter(const long n, const long m, float *data, const float threshold, vector<long> &result_row_ind) {
  
  FrameParams * frameParams = new FrameParams;

  // initialising pointer totals
  frameParams->pointers = vector<float>(frameParams->num_frames*frameParams->num_tasks*frameParams->num_subtasks);
  
  // execution (sum) of 18 members
  execute_section_wise_frames(frameParams, data);

  measure_result(frameParams, n, m, threshold, result_row_ind);

  //sort the values stored in the vector
  std::sort(result_row_ind.begin(),result_row_ind.end());
}
