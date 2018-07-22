#include <vector>
#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <mkl.h>
#include <cmath>

using namespace std;

/**
 * breadth
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
  unsigned long num_superindex;
  FrameParams(int f, int v, int c, 
    int p, int o, int t, int st, int ob, int nb)
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
    num_subindex = num_tasks * num_subtasks;
    n_divide = num_subindex / n_borrow;
    num_product = num_tasks * num_subtasks * num_offset / n_borrow;
    num_division = num_vectors * num_codes * num_pointers;
    num_index = num_division * num_product * n_borrow;
    num_superindex = num_index * num_objects;
	}
};

/**
 * depth
 * (1<<4) objects, (1<<6) vectors, (1<<5) frames = (1<<15) n
 */
class Frame
{
public:
  vector<float> _pointers;
  vector<float> __pointers;
  vector<float> _apointers;
  vector<float> __apointers;
  void setPointers(int num_vectors, int num_codes, int num_pointers, int n_borrow);
  Frame(FrameParams * frameParams);
  ~Frame();
};

void Frame::setPointers(int num_vectors, int num_codes, int num_pointers, int n_borrow)
{
  _pointers = vector<float>(n_borrow*num_vectors);
  __pointers = vector<float>(n_borrow*num_vectors);
  _apointers = vector<float>(n_borrow*num_vectors);
  __apointers = vector<float>(n_borrow*num_vectors);
}

Frame::Frame(FrameParams * frameParams)
{
  setPointers(frameParams->num_vectors, frameParams->num_codes, frameParams->num_pointers, frameParams->n_borrow);
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
}

void calculate_vector_sum(Frame * frame, FrameParams * frameParams, float * data, long p, int frame_no);
void initialise_frames(vector<Frame*>& frames, float * data, FrameParams * frameParams);
void calculate_row_sum(std::vector<Frame*> frames, FrameParams * frameParams);
void calculate_odd_row_sum(std::vector<Frame*> frames, FrameParams * frameParams);
void calculate_next_odd_row_sum(std::vector<Frame*> frames, FrameParams * frameParams);
void calculate_next2_odd_row_sum(std::vector<Frame*> frames, FrameParams * frameParams);
void calculate_next3_odd_row_sum(std::vector<Frame*> frames, FrameParams * frameParams);
void execute_task_for_sum(Frame * frame, FrameParams * frameParams, float * data, int p, int frame_no);
void execute_section_for_frame(Frame * frame, FrameParams * frameParams, float * data, int frame_no);
void execute_section_wise_frames(vector<Frame*> frames, FrameParams * frameParams, float * data);
void aggregate_result(vector<Frame*> frames, FrameParams * frameParams, const long n, const long m, float threshold, std::vector<long> &result_row_ind);
void call_filter(const long n, const long m, float *data, const float threshold, std::vector<long> &result_row_ind, int argc, char** argv);

void call_filter(const long n, const long m, float *data, const float threshold, std::vector<long> &result_row_ind, int argc, char** argv)
{
  int _n_shift = atoi(argv[2]);
  int _m_shift = atoi(argv[3]);
  int frames_shift = atoi(argv[4]);
  int vector_shift = atoi(argv[5]);
  int code_shift = atoi(argv[6]);
  int pointer_shift = atoi(argv[7]);
  int offset_shift = atoi(argv[8]);
  int task_shift = atoi(argv[9]);
  int subtask_shift = atoi(argv[10]);
  int borrow_shift = atoi(argv[11]);
  
  int num_frames = 1<<frames_shift;
  int num_vectors = 1<<vector_shift;
  int num_objects = 1<<2;

  int num_codes = 1<<code_shift;
  int num_pointers = 1<<pointer_shift;
  int num_offset = 1<<offset_shift;
  int num_tasks = 1<<task_shift;
  int num_subtasks = 1<<subtask_shift;

  int n_borrow = 1<<borrow_shift;

  long _n = 1<<_n_shift;
  long _m = 1<<_m_shift;
  
  // long random_seed = (long)(omp_get_wtime()*1000.0) % 1000L;
  // VSLStreamStatePtr rnStream;
  // vslNewStream( &rnStream, VSL_BRNG_MT19937, random_seed);
  // vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rnStream, _m*_n, &data[0], -1.0, 1.0);
  
  // n = 1<<15, m = 1<<18, N = 1<<4, M = 1<<10, factor = 1<<4
  FrameParams * frameParams = new FrameParams
  (num_frames, num_vectors, num_codes, num_pointers, num_offset, num_tasks, num_subtasks, num_objects, n_borrow);

  cout << "FrameParams: SubIndex: " << log2(frameParams->num_subindex) << "Index: " << log2(frameParams->num_index) << "SuperIndex: " << log2(frameParams->num_superindex) 
  << "Product: " << log2(frameParams->num_product) << "Division: " << log2(frameParams->num_division) << "\n";

  vector<Frame*> frames(num_frames);
  const double _t = omp_get_wtime();
  initialise_frames(frames, data, frameParams);
  const double __t = omp_get_wtime();
  printf("Initialise Time %f\t", __t - _t);
  
  const double t3 = omp_get_wtime();

  // execution (sum) of 18 members
  // sections created to test
  execute_section_wise_frames(frames, frameParams, data);

  const double t4 = omp_get_wtime();

  printf("Task Time: %f\t", t4-t3);

  #pragma omp taskwait
  {
    // calculate_row_sum(frames, frameParams);
    // calculate_odd_row_sum(frames, frameParams);
    // calculate_next_odd_row_sum(frames, frameParams);
    // calculate_next2_odd_row_sum(frames, frameParams);
    // calculate_next3_odd_row_sum(frames, frameParams);
    aggregate_result(frames, frameParams, n, m, threshold, result_row_ind);
    const double t5 = omp_get_wtime();
    printf("Task Time: %f\t", t5-t4);
  }

  frames.clear();
  frames.shrink_to_fit();

  printf("Result size: %d\t", result_row_ind.size());
}

void filter(const long n, const long m, float *data, const float threshold, std::vector<long> &result_row_ind, int argc, char** argv) {
  
  // #pragma omp parallel sections
  // {
  //   #pragma omp section
  //   {
      call_filter(n, m, data, threshold, result_row_ind, argc, argv);
  //   }
  // }

  //sort the values stored in the vector
  std::sort(result_row_ind.begin(),
            result_row_ind.end());
}

void aggregate_result(vector<Frame*> frames, FrameParams * frameParams, const long n, const long m, float threshold, std::vector<long> &result_row_ind)
{
  long idx = 0;
  #pragma vector nontemporal
  #pragma vector aligned
  for(int i = 0; i < frames.size(); i++) {
    #pragma vector nontemporal
    #pragma vector aligned
    for(int j = 0; j < frameParams->n_borrow * frameParams->num_vectors; j++) {
      idx = 4*(i*frameParams->n_borrow * frameParams->num_vectors + j);
      if(frames[i]->_pointers[j] > threshold) {
        result_row_ind.emplace_back(idx);
      }
      if(frames[i]->__pointers[j] > threshold) {
        result_row_ind.emplace_back(idx+1);
      }
      if(frames[i]->_apointers[j] > threshold) {
        result_row_ind.emplace_back(idx+2);
      }
      if(frames[i]->_apointers[j] > threshold) {
        result_row_ind.emplace_back(idx+3);
      }
    }
  }
  printf("final value of idx: %d\t", idx);
}

//helper function , refer instructions on the lab page
void append_vec(std::vector<long> &v1, std::vector<long> &v2) {
  v1.insert(v1.end(),v2.begin(),v2.end());
}

/**
 * breadth
 * (1<<4) section offsets, (1<<4) task offsets, (1<<4) offsets, (1<<6) pointers
 * depth
 * (1<<2) pointers, (1<<3) vectors
 * (1<<3) offset_equator1, (1<<4) offset_equator2
 */
void initialise_frames(vector<Frame*>& frames, float * data, FrameParams * frameParams)
{
  #pragma omp parallel for schedule(guided, 4)
  for(int j = 0; j < frames.size(); j++) {
    Frame * frame = new Frame(frameParams);
    fill(frame->_pointers.begin(), frame->_pointers.end(), 0.0);
    fill(frame->__pointers.begin(), frame->__pointers.end(), 0.0);
    fill(frame->_apointers.begin(), frame->_apointers.end(), 0.0);
    fill(frame->__apointers.begin(), frame->__apointers.end(), 0.0);
    frames[j] = frame;
  }
}

/**
 * execution (sum) of 18 members
 * breadth-first search
 * low performance tuning parameters as higher time to traverse through parallely
 */
void calculate_vector_sum(Frame * frame, FrameParams * frameParams, float * data, const int p, const int frame_no) 
{
  #pragma omp parallel
  {
    const int p_index = p/frameParams->n_divide;
    const int p_offset = p*(frameParams->num_offset);
    const int p_max = frameParams->num_subindex * frameParams->num_offset;
    const int d_offset = p*(frameParams->num_division*frameParams->num_offset);
    #pragma omp for firstprivate(frame)
    for(int ii = 0; ii < frameParams->num_vectors; ii++) {
      unsigned int idx = p_index*frameParams->num_vectors + ii;
      float supersum1, supersum2, supersum3, supersum4, supersum5, supersum6, supersum7, supersum8 = 0.0f;
      #pragma vector nontemporal
      #pragma vector aligned
      #pragma omp simd reduction(+: supersum1, supersum2, supersum3, supersum4, supersum5, supersum6, supersum7, supersum8)
      for(int pp = 0; pp < frameParams->num_codes; pp++) {
        float sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8 = 0.0f;
        #pragma vector nontemporal
        #pragma vector aligned
        #pragma omp simd reduction(+: sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8)
        for(int aa = 0; aa < frameParams->num_pointers; aa++) {
          ptrdiff_t offset = ((ii*frameParams->num_codes + pp)*frameParams->num_pointers + aa)*frameParams->num_offset;
          float intersum1, intersum2, intersum3, intersum4, intersum5, intersum6, intersum7, intersum8 = 0.0f;
          #pragma ivdep
          #pragma vector nontemporal
          #pragma vector aligned
          #pragma omp simd reduction(+: intersum1, intersum2, intersum3, intersum4, intersum5, intersum6, intersum7, intersum8)
          for(int j = 0; j <= frameParams->num_offset - 4; j+=4) {
            intersum1 += data[frame_no*frameParams->num_index + d_offset + offset + j];
            intersum2 += data[frame_no*frameParams->num_index + d_offset + offset + j+1];
            intersum3 += data[frame_no*frameParams->num_index + d_offset + offset + j+2];
            intersum4 += data[frame_no*frameParams->num_index + d_offset + offset + j+3];
          }
          sum1 += intersum1;
          sum2 += intersum2;
          sum3 += intersum3;
          sum4 += intersum4;
        }
        supersum1 += sum1;
        supersum2 += sum2;
        supersum3 += sum3;
        supersum4 += sum3;
      }
      frame->_pointers[idx] += supersum1;
      frame->__pointers[idx] += supersum2;
      frame->_apointers[idx] += supersum3;
      frame->__apointers[idx] += supersum4;
    }
  }
}



/**
 * trawl for n (nouns)
 * depth-first search
 * high performance tuning parameters as lower time to sum up the dependencies
 */
void calculate_row_sum(std::vector<Frame*> frames, FrameParams * frameParams) 
{
  #pragma omp parallel for schedule(guided, 4)
  for(int a = 0; a < frames.size(); a++) {
    #pragma ivdep
    #pragma vector nontemporal
    #pragma vector aligned
    #pragma omp simd
    for(int idx = 0; idx < frameParams->num_vectors * frameParams->n_borrow; idx+=2) {
      frames[a]->_pointers[idx] += frames[a]->_pointers[idx+1];
      frames[a]->__pointers[idx] += frames[a]->__pointers[idx+1];
      frames[a]->_apointers[idx] += frames[a]->_apointers[idx+1];
      frames[a]->__apointers[idx] += frames[a]->__apointers[idx+1];
    }
  }
}

/**
 * depth
 */
void calculate_odd_row_sum(std::vector<Frame*> frames, FrameParams * frameParams) 
{
  #pragma omp parallel for schedule(guided, 4)
  for(int a = 0; a < frames.size(); a++) {
    #pragma ivdep
    #pragma vector nontemporal
    #pragma vector aligned
    #pragma omp simd
    for(int idx = 0; idx < frameParams->num_vectors*frameParams->n_borrow; idx+=4) {
      frames[a]->_pointers[idx] += frames[a]->_pointers[idx+2];
      frames[a]->__pointers[idx] += frames[a]->__pointers[idx+2];
      frames[a]->_apointers[idx] += frames[a]->_apointers[idx+2];
      frames[a]->__apointers[idx] += frames[a]->__apointers[idx+2];
    }
  }
}

/**
 * depth
 */
void calculate_next_odd_row_sum(std::vector<Frame*> frames, FrameParams * frameParams) 
{
  #pragma omp parallel for schedule(guided, 4)
  for(int a = 0; a < frames.size(); a++) {
    #pragma ivdep
    #pragma vector nontemporal
    #pragma vector aligned
    #pragma omp simd
    for(int idx = 0; idx < frameParams->num_vectors*frameParams->n_borrow; idx+=8) {
      frames[a]->_pointers[idx] += frames[a]->_pointers[idx+4];
      frames[a]->__pointers[idx] += frames[a]->__pointers[idx+4];
      frames[a]->_apointers[idx] += frames[a]->_apointers[idx+4];
      frames[a]->__apointers[idx] += frames[a]->__apointers[idx+4];
    }
  }
}

/**
 * depth
 */
void calculate_next2_odd_row_sum(std::vector<Frame*> frames, FrameParams * frameParams) 
{
  #pragma omp parallel for schedule(guided, 4)
  for(int a = 0; a < frames.size(); a++) {
    #pragma ivdep
    #pragma vector nontemporal
    #pragma vector aligned
    #pragma omp simd
    for(int idx = 0; idx < frameParams->num_vectors*frameParams->n_borrow; idx+=16) {
      frames[a]->_pointers[idx] += frames[a]->_pointers[idx+8];
      frames[a]->__pointers[idx] += frames[a]->__pointers[idx+8];
      frames[a]->_apointers[idx] += frames[a]->_apointers[idx+8];
      frames[a]->__apointers[idx] += frames[a]->__apointers[idx+8];
    }
  }
}

/**
 * depth
 */
void calculate_next3_odd_row_sum(std::vector<Frame*> frames, FrameParams * frameParams) 
{
  #pragma omp parallel for schedule(guided, 4)
  for(int a = 0; a < frames.size(); a++) {
    #pragma ivdep
    #pragma vector nontemporal
    #pragma vector aligned
    #pragma omp simd
    for(int idx = 0; idx < frameParams->num_vectors*frameParams->n_borrow; idx+=32) {
      frames[a]->_pointers[idx] += frames[a]->_pointers[idx+16];
      frames[a]->__pointers[idx] += frames[a]->__pointers[idx+16];
      frames[a]->_apointers[idx] += frames[a]->_apointers[idx+16];
      frames[a]->__apointers[idx] += frames[a]->__apointers[idx+16];
    }
  }
}

// void calculate_net_sum(Frame * frame, const int m, int p,
// std::vector<float>& sum) {
//   // sum of m elements
//   for(int i = 1; i <= (1<<10); i+=64) {
//     for(int j = 1; j <= (1<<4); j++) {
//       ptrdiff_t offset = (i-1)*(1<<4) + j-1;
//       sum.emplace_back(frame->_pointers[0][offset]);
//     }
//   }
// }

void execute_task_for_sum(Frame * frame, FrameParams * frameParams, float * data, int p, const int frame_no)
{
  #pragma omp single nowait
  {
    for(int i = p; i < frameParams->num_subtasks + p; i++) {
      #pragma omp task
      {
        calculate_vector_sum(frame, frameParams, data, i, frame_no);
      }
    }
  }
}

void execute_section_for_frame(Frame * frame, FrameParams * frameParams, float * data, const int frame_no)
{
  #pragma omp for
  for(int i = 1; i <= frameParams->num_tasks; i++) {
    execute_task_for_sum(frame, frameParams, data, (i-1)*frameParams->num_subtasks, frame_no);
  }
}

void execute_task_wise_frames(vector<Frame*> frames, FrameParams * frameParams, float * data, const int z)
{
  #pragma omp for
  for(int i = (z-1)*frames.size()/(1<<4); i < z*frames.size()/(1<<4); i++) {
    execute_section_for_frame(frames[i], frameParams, data, i);
  }
}

void execute_section_wise_frames(vector<Frame*> frames, FrameParams * frameParams, float * data)
{
  
  #pragma omp parallel 
  {
    #pragma omp for nowait
    for(int z = 1; z <= (1<<4); z++) {
      execute_task_wise_frames(frames, frameParams, data, z);
    }
  }
}