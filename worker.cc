#include <vector>
#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <mkl.h>
#include <cmath>
#include <utility>
#include <thread>
#include <chrono>

using namespace std;

/**
 * (1<<8) section offsets, (1<<4) task offsets, (1<<4) offsets, (1<<2) pointers, 4 sum offsets
 */
struct FrameParams
{
  int num_frames;
  int num_vectors;
  int num_codes;
  int num_pointers;
  int num_offset;
  int num_tasks;
  int num_subtasks;
  int num_objects;
  int num_subindex;
  int num_product;
  int num_division;
  int num_index;
  int n_borrow;
  int n_divide;
  int n_threads;
  unsigned long num_superindex;
  FrameParams(int f, int v, int c, 
    int p, int o, int t, int st, int ob, int nb, int thr)
	{
    num_frames = f;
    num_vectors = v;
    num_codes = c;
    num_pointers = p;
    num_offset = o;
    num_tasks = t;
    num_subtasks = st;
    num_objects = ob;
    n_borrow = nb;
    n_threads = thr;
    num_subindex = num_tasks * num_subtasks;
    n_divide = num_subindex / n_borrow;
    num_product = num_tasks * num_subtasks * num_offset / n_borrow;
    num_division = num_vectors * num_codes * num_pointers;
    num_index = num_division * num_product * n_borrow;
    num_superindex = num_index * num_objects;
	}
};

/**
 * (1<<4) objects, (1<<6) vectors, (1<<5) frames = (1<<15) n
 */
class Frame
{
public:
  vector<float> _pointers;
  vector<float> __pointers;
  vector<float> _apointers;
  vector<float> __apointers;
  vector<float> _bpointers;
  vector<float> __bpointers;
  vector<float> _cpointers;
  vector<float> __cpointers;
  void setPointers(int num_vectors, int num_codes, int num_pointers, int n_borrow, int num_frames);
  Frame(FrameParams * frameParams);
  ~Frame();
};

void Frame::setPointers(int num_vectors, int num_codes, int num_pointers, int n_borrow, int num_frames)
{
  _pointers = vector<float>(num_frames*n_borrow*num_vectors);
  __pointers = vector<float>(num_frames*n_borrow*num_vectors);
  _apointers = vector<float>(num_frames*n_borrow*num_vectors);
  __apointers = vector<float>(num_frames*n_borrow*num_vectors);
  _bpointers = vector<float>(num_frames*n_borrow*num_vectors);
  __bpointers = vector<float>(num_frames*n_borrow*num_vectors);
  _cpointers = vector<float>(num_frames*n_borrow*num_vectors);
  __cpointers = vector<float>(num_frames*n_borrow*num_vectors);
}

Frame::Frame(FrameParams * frameParams)
{
  setPointers(frameParams->num_vectors, frameParams->num_codes, frameParams->num_pointers, frameParams->n_borrow, frameParams->num_frames);
}

Frame::~Frame() {
  _pointers.clear();
  _pointers.shrink_to_fit();
  __pointers.clear();
  __pointers.shrink_to_fit();
  _apointers.clear();
  _apointers.shrink_to_fit();
  __apointers.clear();
  __apointers.shrink_to_fit();
  _bpointers.clear();
  _bpointers.shrink_to_fit();
  __bpointers.clear();
  __bpointers.shrink_to_fit();
  _cpointers.clear();
  _cpointers.shrink_to_fit();
  __cpointers.clear();
  __cpointers.shrink_to_fit();
}

void calculate_vector_sum(Frame * frame, FrameParams * frameParams, float * data, long p, int frame_no, vector<vector<float>>& pointers);
void initialise_frames(Frame* frame, float * data, FrameParams * frameParams);
void calculate_row_sum(Frame* frame, FrameParams * frameParams);
void calculate_odd_row_sum(Frame* frame, FrameParams * frameParams);
void calculate_next_odd_row_sum(Frame* frame, FrameParams * frameParams);
void calculate_next2_odd_row_sum(Frame* frame, FrameParams * frameParams);
void calculate_next3_odd_row_sum(Frame* frame, FrameParams * frameParams);
void execute_task_for_sum(Frame * frame, FrameParams * frameParams, float * data, int p, int frame_no, vector<vector<float>>& pointers);
void execute_section_for_frame(Frame * frame, FrameParams * frameParams, float * data, int frame_no, vector<vector<float>>& pointers);
void execute_section_wise_frames(Frame* frame, FrameParams * frameParams, float * data, vector<vector<float>>& pointers);
void aggregate_result(Frame* frame, FrameParams * frameParams, const long n, const long m, 
float threshold, vector<long> &result_row_ind);
void aggregate_sum_result(Frame * frame, FrameParams * frameParams, const long n, const long m, 
float threshold, vector<float>& sum, vector<vector<float>> pointers);
void measure_result(Frame * frame, FrameParams * frameParams, const long n, const long m, float threshold, 
vector<float> sum, vector<long> &result_row_ind);
void call_filter(const long n, const long m, float *data, const float threshold, std::vector<long> &result_row_ind);
void calculate_total_sum(const long n, const long m, Frame * frame, FrameParams * frameParams, 
float * data, float threshold, vector<long> &result_row_ind);

void call_filter(const long n, const long m, float *data, const float threshold, vector<long> &result_row_ind)
{
  // n
  int num_frames = 1<<4;
  int num_vectors = 1<<3;
  int num_tasks = 1<<4;
  int num_subtasks = 1<<4;

  // m
  int num_codes = 1<<2;
  int num_pointers = 1<<4;
  int num_offset = 1<<12;
  int num_objects = 1<<3;

  int n_borrow = 1<<5;

  int n_threads = 1<<3;

  FrameParams * frameParams = new FrameParams
  (num_frames, num_vectors, num_codes, num_pointers, num_offset, num_tasks, num_subtasks, num_objects, n_borrow, n_threads);

  Frame * frame = new Frame(frameParams);
  // initialise_frames(frame, data, frameParams);

  vector<vector<float>> pointers = vector<vector<float>>(8, vector<float>(0));
  vector<float> sum;
  
  // execution (sum) of 18 members
  execute_section_wise_frames(frame, frameParams, data, pointers);

  aggregate_sum_result(frame, frameParams, n, m, threshold, sum, pointers);
  measure_result(frame, frameParams, n, m, threshold, sum, result_row_ind);
}

void filter(const long n, const long m, float *data, const float threshold, vector<long> &result_row_ind) {
  
  call_filter(n, m, data, threshold, result_row_ind);

  //sort the values stored in the vector
  std::sort(result_row_ind.begin(),result_row_ind.end());
}

void aggregate_result(Frame * frame, FrameParams * frameParams, const long n, const long m, 
float threshold, vector<long> &result_row_ind)
{
  for(long i = 0; i < frameParams->num_frames*frameParams->n_borrow*frameParams->num_vectors; i++) {
    if(frame->_pointers[i] > threshold) {
      result_row_ind.emplace_back(8*i);
    }
    if(frame->__pointers[i] > threshold) {
      result_row_ind.emplace_back(8*i+1);
    }
    if(frame->_apointers[i] > threshold) {
      result_row_ind.emplace_back(8*i+2);
    }
    if(frame->__apointers[i] > threshold) {
      result_row_ind.emplace_back(8*i+3);
    }
    if(frame->_bpointers[i] > threshold) {
      result_row_ind.emplace_back(8*i+4);
    }
    if(frame->__bpointers[i] > threshold) {
      result_row_ind.emplace_back(8*i+5);
    }
    if(frame->_cpointers[i] > threshold) {
      result_row_ind.emplace_back(8*i+6);
    }
    if(frame->__cpointers[i] > threshold) {
      result_row_ind.emplace_back(8*i+7);
    }
  }
}

void aggregate_sum_result(Frame * frame, FrameParams * frameParams, const long n, const long m, 
float threshold, vector<float>& sum, vector<vector<float>> pointers)
{
  #pragma ivdep
  for(long i = 0; i < n; i++) {
    sum.push_back(pointers[0][i] + pointers[1][i] + 
    pointers[2][i] + pointers[3][i] + 
    pointers[4][i] + pointers[5][i] + 
    pointers[6][i] + pointers[7][i]);
  }
}

void measure_result(Frame * frame, FrameParams * frameParams, const long n, const long m, 
float threshold, vector<float> sum, vector<long> &result_row_ind)
{
  for(long i = 0; i < n; i++) {
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
void initialise_frames(Frame * frame, float * data, FrameParams * frameParams)
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
void calculate_vector_sum(Frame * frame, FrameParams * frameParams, float * data, const int p, const int frame_no, vector<vector<float>>& pointers)
{
  const int STRIP = frameParams->num_codes*frameParams->num_pointers*frameParams->num_offset;
  const int WIDTH = frameParams->num_vectors*STRIP*(frame_no*frameParams->num_subindex + p);
  for(long ii = WIDTH; ii < WIDTH + frameParams->num_vectors*STRIP; ii+=STRIP) {
    float intersum1, intersum2, intersum3, intersum4, intersum5, intersum6, intersum7, intersum8 = 0.0f;
    float *mptr = &data[ii];
    for(long aa = ii; aa < ii+frameParams->num_pointers*frameParams->num_codes; aa++) {
      float *ptr = &mptr[aa-ii];
      #pragma omp simd reduction(+: intersum1, intersum2, intersum3, intersum4, intersum5, intersum6, intersum7, intersum8)
      for(long j = aa; j < aa+frameParams->num_offset - 8; j+=8) {
        intersum1 += ptr[j-aa];
        intersum2 += ptr[j-aa+1];
        intersum3 += ptr[j-aa+2];
        intersum4 += ptr[j-aa+3];
        intersum5 += ptr[j-aa+4];
        intersum6 += ptr[j-aa+5];
        intersum7 += ptr[j-aa+6];
        intersum8 += ptr[j-aa+7];
      }
    }
    #pragma omp critical
    {
      pointers[0].push_back(intersum1);
      pointers[1].push_back(intersum2);
      pointers[2].push_back(intersum3);
      pointers[3].push_back(intersum4);
      pointers[4].push_back(intersum5);
      pointers[5].push_back(intersum6);
      pointers[6].push_back(intersum7);
      pointers[7].push_back(intersum8);
    }
  }
}

void execute_task_for_sum(Frame * frame, FrameParams * frameParams, float * data, int st, const int frame_no, vector<vector<float>>& pointers)
{
  // #pragma omp single nowait
  // {
    // #pragma omp for
    for(int i = st; i < st+frameParams->num_subtasks; i++) {
      // #pragma omp task
      // {
        calculate_vector_sum(frame, frameParams, data, i, frame_no, pointers);
      // }
    }
  // }
}

void execute_section_for_frame(Frame * frame, FrameParams * frameParams, float * data, const int frame_no, vector<vector<float>>& pointers)
{
  // #pragma omp single nowait
  // {
    // #pragma omp for
    for(int i = 1; i <= frameParams->num_tasks; i++) {
      // #pragma omp task
      // {
        execute_task_for_sum(frame, frameParams, data, (i-1)*frameParams->num_subtasks, frame_no, pointers);
      // }
    }
  // }
}

void execute_task_wise_frames(Frame * frame, FrameParams * frameParams, float * data, const int z, vector<vector<float>>& pointers)
{
  // #pragma omp for
  for(int i = (z-1)*frameParams->num_frames/frameParams->n_threads; i < z*frameParams->num_frames/frameParams->n_threads; i++) {
    execute_section_for_frame(frame, frameParams, data, i, pointers);
  }
}

void execute_section_wise_frames(Frame * frame, FrameParams * frameParams, float * data, vector<vector<float>>& pointers)
{
  // #pragma omp for
  // for(int z = 1; z <= frameParams->n_threads; z++) {
  //   execute_task_wise_frames(frame, frameParams, data, z);
  // }
  #pragma omp parallel num_threads(8)
  {
    int z = omp_get_thread_num();
    execute_task_wise_frames(frame, frameParams, data, z+1, pointers);
  }
}