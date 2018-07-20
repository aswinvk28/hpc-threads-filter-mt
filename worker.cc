#include <vector>
#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <omp.h>
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
  vector< vector<vector<float*>> > _pointers;
  vector< vector<vector<float*>> > __pointers;
  vector< vector<vector<float*>> > _apointers;
  vector< vector<vector<float*>> > __apointers;
  vector<vector<float>> sum;
  void setPointers(int num_vectors, int num_codes, int num_pointers, int n_borrow);
  Frame(FrameParams * frameParams);
};

void Frame::setPointers(int num_vectors, int num_codes, int num_pointers, int n_borrow)
{
  _pointers = 
  vector< vector<vector<float*>> > ( n_borrow*num_vectors , 
    vector<vector<float*>>(num_codes, vector<float*>(num_pointers)) );
  __pointers = 
  vector< vector<vector<float*>> > ( n_borrow*num_vectors , 
    vector<vector<float*>>(num_codes, vector<float*>(num_pointers)) );
  _apointers = 
  vector< vector<vector<float*>> > ( n_borrow*num_vectors , 
    vector<vector<float*>>(num_codes, vector<float*>(num_pointers)) );
  __apointers = 
  vector< vector<vector<float*>> > ( n_borrow*num_vectors , 
    vector<vector<float*>>(num_codes, vector<float*>(num_pointers)) );
}

Frame::Frame(FrameParams * frameParams)
{
  setPointers(frameParams->num_vectors, frameParams->num_codes, frameParams->num_pointers, frameParams->n_borrow);
}

void calculate_vector_sum(Frame * frame, FrameParams * frameParams, long p, int frame_no);
void initialise_frames(vector<Frame*>& frames, float * data, FrameParams * frameParams);
void calculate_row_sum(std::vector<Frame*> frames, FrameParams * frameParams);
void calculate_odd_row_sum(std::vector<Frame*> frames, FrameParams * frameParams);
void calculate_next_odd_row_sum(std::vector<Frame*> frames, FrameParams * frameParams);
void calculate_next2_odd_row_sum(std::vector<Frame*> frames, FrameParams * frameParams);
void calculate_next3_odd_row_sum(std::vector<Frame*> frames, FrameParams * frameParams);
void execute_task_for_sum(Frame * frame, FrameParams * frameParams, int p, int frame_no);
void execute_section_for_frame(Frame * frame, FrameParams * frameParams, int frame_no);
void execute_section_wise_frames(vector<Frame*> frames, FrameParams * frameParams);

void filter(const long n, const long m, float *data, const float threshold, std::vector<long> &result_row_ind) {
  
  int num_frames = 1<<5;
  int num_vectors = 1<<5;
  int num_objects = 1<<2;

  int num_codes = 1<<3;
  int num_pointers = 1<<2;
  int num_offset = 1<<5;
  int num_tasks = 1<<4;
  int num_subtasks = 1<<4;

  int n_borrow = 1<<3;

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

  std::vector<float> total0(0); 
  std::vector<float> total1(0); std::vector<float> total2(0); 
  std::vector<float> total3(0); std::vector<float> total4(0);
  std::vector<float> total5(0); std::vector<float> total6(0);
  std::vector<float> total7(0); std::vector<float> total8(0); 
  std::vector<float> total9(0); 
  std::vector<float> total10(0); std::vector<float> total11(0);
  std::vector<float> total12(0); std::vector<float> total13(0);
  std::vector<float> total14(0); std::vector<float> total15(0);

  std::vector<float> total[16] = {
    total0, total1, total2, total3, total4, total5, total6, total7,
    total8, total9, total10, total11, total12, total13, total14, total15};
  
  const double t3 = omp_get_wtime();

  // execution (sum) of 18 members
  // sections created to test
  execute_section_wise_frames(frames, frameParams);

  const double t4 = omp_get_wtime();

  printf("Task Time: %f\t", t4-t3);

  #pragma omp taskwait
  {
    calculate_row_sum(frames, frameParams);
    calculate_odd_row_sum(frames, frameParams);
    calculate_next_odd_row_sum(frames, frameParams);
    calculate_next2_odd_row_sum(frames, frameParams);
    calculate_next3_odd_row_sum(frames, frameParams);
    const double t5 = omp_get_wtime();
    printf("Task Time: %f\t", t5-t4);
  }

  //sort the values stored in the vector
  std::sort(result_row_ind.begin(),
            result_row_ind.end());
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
  long w_index = frameParams->num_division * frameParams->n_borrow;
  #pragma omp parallel for schedule(guided, 4)
  for(int j = 0; j < frames.size(); j++) {
    Frame * frame = new Frame(frameParams);
    frames[j] = frame;
    int idx = -1;
    #pragma omp parallel for firstprivate(frame)
    for(int idx = 0; idx < frameParams->n_borrow*frameParams->num_vectors; idx++) {
      #pragma omp simd
      for(int pp = 0; pp < frameParams->num_codes; pp++) {
        #pragma omp simd
        for(int aa = 0; aa < frameParams->num_pointers; aa++) {
          long index = ((idx*(frameParams->num_codes)+pp)*frameParams->num_pointers + aa);
          frame->_pointers[idx][pp][aa] = &data[index*(frameParams->num_product) + j*(frameParams->num_superindex)];
          frame->__pointers[idx][pp][aa] = &data[frameParams->num_index + index*(frameParams->num_product) + j*(frameParams->num_superindex)];
          frame->_apointers[idx][pp][aa] = &data[2*frameParams->num_index + index*(frameParams->num_product) + j*(frameParams->num_superindex)];
          frame->__apointers[idx][pp][aa] = &data[3*frameParams->num_index + index*(frameParams->num_product) + j*(frameParams->num_superindex)];
        }
      }
    }
  }
}

/**
 * execution (sum) of 18 members
 * breadth-first search
 * low performance tuning parameters as higher time to traverse through parallely
 */
void calculate_vector_sum(Frame * frame, FrameParams * frameParams, const int p, const int frame_no) 
{
  #pragma omp parallel
  {
    const int p_index = p/frameParams->n_divide;
    const int p_offset = p*(frameParams->num_offset);
    #pragma omp for firstprivate(frame)
    for(int ii = 0; ii < frameParams->num_vectors; ii++) {
      float supersum1, supersum2, supersum3, supersum4 = 0.0f;
      int idx = p_index*frameParams->num_vectors + ii;
      #pragma vector nontemporal
      #pragma vector aligned
      #pragma omp simd reduction(+: supersum1, supersum2, supersum3, supersum4)
      for(int pp = 0; pp < frameParams->num_codes; pp++) {
        float sum1, sum2, sum3, sum4 = 0.0f;
        #pragma vector nontemporal
        #pragma vector aligned
        #pragma omp simd reduction(+: sum1, sum2, sum3, sum4)
        for(int aa = 0; aa < frameParams->num_pointers; aa++) {
          float intersum1, intersum2, intersum3, intersum4 = 0.0f;
          #pragma ivdep
          #pragma vector nontemporal
          #pragma vector aligned
          #pragma omp simd reduction(+: intersum1, intersum2, intersum3, intersum4)
          for(int j = 1; j <= frameParams->num_offset; j++) {
            ptrdiff_t offset = p_offset + j-1;
            intersum1 += frame->_pointers[idx][pp][aa][offset];
            intersum2 += frame->__pointers[idx][pp][aa][offset];
            intersum3 += frame->_apointers[idx][pp][aa][offset];
            intersum4 += frame->__apointers[idx][pp][aa][offset];
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
      frame->_pointers[idx][0][0][0] += supersum1;
      frame->__pointers[idx][0][0][0] += supersum2;
      frame->_apointers[idx][0][0][0] += supersum3;
      frame->__apointers[idx][0][0][0] += supersum4;
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
  #pragma omp parallel for
  for(int a = 0; a < frames.size(); a++) {
    #pragma ivdep
    #pragma vector nontemporal
    #pragma vector aligned
    #pragma omp simd
    for(int idx = 0; idx < frameParams->num_vectors * frameParams->n_borrow; idx+=2) {
      frames[a]->_pointers[idx][0][0][0] += frames[a]->_pointers[idx+1][0][0][0];
      frames[a]->__pointers[idx][0][0][0] += frames[a]->__pointers[idx+1][0][0][0];
      frames[a]->_apointers[idx][0][0][0] += frames[a]->_apointers[idx+1][0][0][0];
      frames[a]->__apointers[idx][0][0][0] += frames[a]->__apointers[idx+1][0][0][0];
    }
  }
}

/**
 * depth
 */
void calculate_odd_row_sum(std::vector<Frame*> frames, FrameParams * frameParams) 
{
  #pragma omp parallel for
  for(int a = 0; a < frames.size(); a++) {
    #pragma ivdep
    #pragma vector nontemporal
    #pragma vector aligned
    #pragma omp simd
    for(int idx = 0; idx < frameParams->num_vectors*frameParams->n_borrow; idx+=4) {
      frames[a]->_pointers[idx][0][0][0] += frames[a]->_pointers[idx+2][0][0][0];
      frames[a]->__pointers[idx][0][0][0] += frames[a]->__pointers[idx+2][0][0][0];
      frames[a]->_apointers[idx][0][0][0] += frames[a]->_apointers[idx+2][0][0][0];
      frames[a]->__apointers[idx][0][0][0] += frames[a]->__apointers[idx+2][0][0][0];
    }
  }
}

/**
 * depth
 */
void calculate_next_odd_row_sum(std::vector<Frame*> frames, FrameParams * frameParams) 
{
  #pragma omp parallel for
  for(int a = 0; a < frames.size(); a++) {
    #pragma ivdep
    #pragma vector nontemporal
    #pragma vector aligned
    #pragma omp simd
    for(int idx = 0; idx < frameParams->num_vectors*frameParams->n_borrow; idx+=8) {
      frames[a]->_pointers[idx][0][0][0] += frames[a]->_pointers[idx+4][0][0][0];
      frames[a]->__pointers[idx][0][0][0] += frames[a]->__pointers[idx+4][0][0][0];
      frames[a]->_apointers[idx][0][0][0] += frames[a]->_apointers[idx+4][0][0][0];
      frames[a]->__apointers[idx][0][0][0] += frames[a]->__apointers[idx+4][0][0][0];
    }
  }
}

/**
 * depth
 */
void calculate_next2_odd_row_sum(std::vector<Frame*> frames, FrameParams * frameParams) 
{
  #pragma omp parallel for
  for(int a = 0; a < frames.size(); a++) {
    #pragma ivdep
    #pragma vector nontemporal
    #pragma vector aligned
    #pragma omp simd
    for(int idx = 0; idx < frameParams->num_vectors*frameParams->n_borrow; idx+=16) {
      frames[a]->_pointers[idx][0][0][0] += frames[a]->_pointers[idx+8][0][0][0];
      frames[a]->__pointers[idx][0][0][0] += frames[a]->__pointers[idx+8][0][0][0];
      frames[a]->_apointers[idx][0][0][0] += frames[a]->_apointers[idx+8][0][0][0];
      frames[a]->__apointers[idx][0][0][0] += frames[a]->__apointers[idx+8][0][0][0];
    }
  }
}

/**
 * depth
 */
void calculate_next3_odd_row_sum(std::vector<Frame*> frames, FrameParams * frameParams) 
{
  #pragma omp parallel for
  for(int a = 0; a < frames.size(); a++) {
    #pragma ivdep
    #pragma vector nontemporal
    #pragma vector aligned
    #pragma omp simd
    for(int idx = 0; idx < frameParams->num_vectors*frameParams->n_borrow; idx+=32) {
      frames[a]->_pointers[idx][0][0][0] += frames[a]->_pointers[idx+16][0][0][0];
      frames[a]->__pointers[idx][0][0][0] += frames[a]->__pointers[idx+16][0][0][0];
      frames[a]->_apointers[idx][0][0][0] += frames[a]->_apointers[idx+16][0][0][0];
      frames[a]->__apointers[idx][0][0][0] += frames[a]->__apointers[idx+16][0][0][0];
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

void execute_task_for_sum(Frame * frame, FrameParams * frameParams, int p, const int frame_no)
{
  #pragma omp single nowait
  {
    for(int i = p; i < frameParams->num_subtasks + p; i++) {
      #pragma omp task
      {
        calculate_vector_sum(frame, frameParams, i, frame_no);
      }
    }
  }
}

void execute_section_for_frame(Frame * frame, FrameParams * frameParams, const int frame_no)
{
  #pragma omp single nowait
  {
    for(int i = 1; i <= frameParams->num_tasks; i++) {
      #pragma omp task
      {
        execute_task_for_sum(frame, frameParams, (i-1)*frameParams->num_subtasks, frame_no);
      }
    }
  }
}

void execute_task_wise_frames(vector<Frame*> frames, FrameParams * frameParams, const int z)
{
  #pragma omp single nowait
  {
    for(int i = (z-1)*frames.size()/(1<<4); i < z*frames.size()/(1<<4); i++) {
      #pragma omp task
      {
        execute_section_for_frame(frames[i], frameParams, i);
      }
    }
  }
}

void execute_section_wise_frames(vector<Frame*> frames, FrameParams * frameParams)
{
  #pragma omp parallel
  {
    #pragma omp for
    for(int z = 1; z <= (1<<4); z++) {
      execute_task_wise_frames(frames, frameParams, z);
    }
  }
}