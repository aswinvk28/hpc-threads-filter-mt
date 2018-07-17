#include <vector>
#include <algorithm>
#include <iostream>
#include <omp.h>

using namespace std;

/**
 * breadth
 * 1 section offsets, (1<<3) task offsets, (1<<4) offsets, (1<<4) pointers, (1<<0) sum offsets
 */
struct FrameParams
{
  int N;
	int M;
	FrameParams(int _M, int _N)
	{
		M = _M;
		N = _N;
	}
};

/**
 * depth
 * (1<<4) objects, (1<<3) vectors, (1<<8) frames
 */
struct Frame
{
  vector<vector<float*>> _pointers;
  vector<vector<float*>> __pointers;
  vector<vector<float*>> _apointers;
  vector<vector<float*>> __apointers;
  vector<vector<float*>> _bpointers;
  vector<vector<float*>> __bpointers;
  vector<vector<float*>> _cpointers;
  vector<vector<float*>> __cpointers;
  vector<vector<float*>> _dpointers;
  vector<vector<float*>> __dpointers;
  vector<vector<float*>> _epointers;
  vector<vector<float*>> __epointers;
  vector<vector<float*>> _fpointers;
  vector<vector<float*>> __fpointers;
  vector<vector<float*>> _gpointers;
  vector<vector<float*>> __gpointers;
  vector<vector<float>> sum;
};

//helper function , refer instructions on the lab page
void append_vec(std::vector<long> &v1, std::vector<long> &v2) {
  v1.insert(v1.end(),v2.begin(),v2.end());
}

/**
 * execution (sum) of 18 members
 * breadth-first search
 * low performance tuning parameters as higher time to traverse through parallely
 */
void calculate_vector_sum(Frame * frame, const int m, long p) {
  #pragma ivdep
  #pragma omp simd
  #pragma vector nontemporal
  #pragma vector aligned
  for(int i = 0; i < (1<<3); i++) {
    #pragma omp simd
    for(int aa = 0; aa < (1<<4); aa++) {
      #pragma omp simd
      for(int j = 1; j <= (1<<4); j++) {
        frame->_pointers[i][0][j-1] += frame->_pointers[i][aa][p*(1<<4) + j-1];
        frame->__pointers[i][0][j-1] += frame->__pointers[i][aa][p*(1<<4) + j-1];
        frame->_apointers[i][0][j-1] += frame->_apointers[i][aa][p*(1<<4) + j-1];
        frame->__apointers[i][0][j-1] += frame->__apointers[i][aa][p*(1<<4) + j-1];
        frame->_bpointers[i][0][j-1] += frame->_bpointers[i][aa][p*(1<<4) + j-1];
        frame->__bpointers[i][0][j-1] += frame->__bpointers[i][aa][p*(1<<4) + j-1];
        frame->_cpointers[i][0][j-1] += frame->_cpointers[i][aa][p*(1<<4) + j-1];
        frame->__cpointers[i][0][j-1] += frame->__cpointers[i][aa][p*(1<<4) + j-1];
        frame->_dpointers[i][0][j-1] += frame->_dpointers[i][aa][p*(1<<4) + j-1];
        frame->__dpointers[i][0][j-1] += frame->__dpointers[i][aa][p*(1<<4) + j-1];
        frame->_epointers[i][0][j-1] += frame->_epointers[i][aa][p*(1<<4) + j-1];
        frame->__epointers[i][0][j-1] += frame->__epointers[i][aa][p*(1<<4) + j-1];
        frame->_fpointers[i][0][j-1] += frame->_fpointers[i][aa][p*(1<<4) + j-1];
        frame->__fpointers[i][0][j-1] += frame->__fpointers[i][aa][p*(1<<4) + j-1];
        frame->_gpointers[i][0][j-1] += frame->_gpointers[i][aa][p*(1<<4) + j-1];
        frame->__gpointers[i][0][j-1] += frame->__gpointers[i][aa][p*(1<<4) + j-1];
      }
    }
  }
}

/**
 * trawl for n (nouns)
 * depth-first search
 * high performance tuning parameters as lower time to sum up the dependencies
 */
void calculate_row_sum(Frame * frame, const int m) {
  #pragma ivdep
  #pragma omp simd
  #pragma vector nontemporal
  #pragma vector aligned
  for(int ii = 0; ii < (1<<3); ii++) {
    #pragma omp simd
    for(int i = 1; i <= (1<<1); i++) {
      #pragma ivdep
      #pragma omp simd
      for(long j = 1; j < (1<<3); j+=2) {
        ptrdiff_t offset = (i-1)*(1<<3) + j-1;
        frame->_pointers[ii][0][offset] += frame->_pointers[ii][0][offset+1];
        frame->__pointers[ii][0][offset] += frame->__pointers[ii][0][offset+1];
        frame->_apointers[ii][0][offset] += frame->_apointers[ii][0][offset+1];
        frame->__apointers[ii][0][offset] += frame->__apointers[ii][0][offset+1];
        frame->_bpointers[ii][0][offset] += frame->_bpointers[ii][0][offset+1];
        frame->__bpointers[ii][0][offset] += frame->__bpointers[ii][0][offset+1];
        frame->_cpointers[ii][0][offset] += frame->_cpointers[ii][0][offset+1];
        frame->__cpointers[ii][0][offset] += frame->__cpointers[ii][0][offset+1];
        frame->_dpointers[ii][0][offset] += frame->_dpointers[ii][0][offset+1];
        frame->__dpointers[ii][0][offset] += frame->__dpointers[ii][0][offset+1];
        frame->_epointers[ii][0][offset] += frame->_epointers[ii][0][offset+1];
        frame->__epointers[ii][0][offset] += frame->__epointers[ii][0][offset+1];
        frame->_fpointers[ii][0][offset] += frame->_fpointers[ii][0][offset+1];
        frame->__fpointers[ii][0][offset] += frame->__fpointers[ii][0][offset+1];
        frame->_gpointers[ii][0][offset] += frame->_gpointers[ii][0][offset+1];
        frame->__gpointers[ii][0][offset] += frame->__gpointers[ii][0][offset+1];
      }
    }
  }
}

/**
 * depth
 */
void calculate_odd_row_sum(Frame * frame, const int m) {
  #pragma ivdep
  #pragma omp simd
  #pragma vector nontemporal
  #pragma vector aligned
  for(int ii = 0; ii < (1<<3); ii++) {
    #pragma omp simd
    for(long i = 1; i <= (1<<1); i++) {
      #pragma ivdep
      #pragma omp simd
      for(long j = 1; j < (1<<3); j+=4) {
        ptrdiff_t offset = (i-1)*(1<<3) + j-1;
        frame->_pointers[ii][0][offset] += frame->_pointers[ii][0][offset+2];
        frame->__pointers[ii][0][offset] += frame->__pointers[ii][0][offset+2];
        frame->_apointers[ii][0][offset] += frame->_apointers[ii][0][offset+2];
        frame->__apointers[ii][0][offset] += frame->__apointers[ii][0][offset+2];
        frame->_bpointers[ii][0][offset] += frame->_bpointers[ii][0][offset+2];
        frame->__bpointers[ii][0][offset] += frame->__bpointers[ii][0][offset+2];
        frame->_cpointers[ii][0][offset] += frame->_cpointers[ii][0][offset+2];
        frame->__cpointers[ii][0][offset] += frame->__cpointers[ii][0][offset+2];
        frame->_dpointers[ii][0][offset] += frame->_dpointers[ii][0][offset+2];
        frame->__dpointers[ii][0][offset] += frame->__dpointers[ii][0][offset+2];
        frame->_epointers[ii][0][offset] += frame->_epointers[ii][0][offset+2];
        frame->__epointers[ii][0][offset] += frame->__epointers[ii][0][offset+2];
        frame->_fpointers[ii][0][offset] += frame->_fpointers[ii][0][offset+2];
        frame->__fpointers[ii][0][offset] += frame->__fpointers[ii][0][offset+2];
        frame->_gpointers[ii][0][offset] += frame->_gpointers[ii][0][offset+2];
        frame->__gpointers[ii][0][offset] += frame->__gpointers[ii][0][offset+2];
      }
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

/**
 * breadth
 * (1<<4) section offsets, (1<<4) task offsets, (1<<4) offsets, (1<<6) pointers
 * depth
 * (1<<4) pointers, (1<<3) objects
 */
void load_frame(Frame * frame, float * data, unsigned long index) {
  int ii;
  for(ii = 0; ii < (1<<3); ii++) {
    vector<float*> _pointers;
    for(int i = 0; i < 1<<4; i++) {
      _pointers.emplace_back(&data[(ii*(1<<4)+i)*(1<<3)*(1<<4) + index]);
    }
    frame->_pointers.emplace_back(_pointers);
  }
  for(ii = 0; ii < (1<<3); ii++) {
    vector<float*> __pointers;
    for(int i = 0; i < 1<<4; i++) {
      __pointers.emplace_back(&data[(1<<14) + (ii*(1<<4)+i)*(1<<3)*(1<<4) + index]);
    }
    frame->__pointers.emplace_back(__pointers);
  }
  for(ii = 0; ii < (1<<3); ii++) {
    vector<float*> _apointers;
    for(int i = 0; i < 1<<4; i++) {
      _apointers.emplace_back(&data[2*(1<<14) + (ii*(1<<4)+i)*(1<<3)*(1<<4) + index]);
    }
    frame->_apointers.emplace_back(_apointers);
  }
  for(ii = 0; ii < (1<<3); ii++) {
    vector<float*> __apointers;
    for(int i = 0; i < 1<<4; i++) {
      __apointers.emplace_back(&data[3*(1<<14) + (ii*(1<<4)+i)*(1<<3)*(1<<4) + index]);
    }
    frame->__apointers.emplace_back(__apointers);
  }
  for(ii = 0; ii < (1<<3); ii++) {
    vector<float*> _bpointers;
    for(int i = 0; i < 1<<4; i++) {
      _bpointers.emplace_back(&data[4*(1<<14) + (ii*(1<<4)+i)*(1<<3)*(1<<4) + index]);
    }
    frame->_bpointers.emplace_back(_bpointers);
  }
  for(ii = 0; ii < (1<<3); ii++) {
    vector<float*> __bpointers;
    for(int i = 0; i < 1<<4; i++) {
      __bpointers.emplace_back(&data[5*(1<<14) + (ii*(1<<4)+i)*(1<<3)*(1<<4) + index]);
    }
    frame->__bpointers.emplace_back(__bpointers);
  }
  for(ii = 0; ii < (1<<3); ii++) {
    vector<float*> _cpointers;
    for(int i = 0; i < 1<<4; i++) {
      _cpointers.emplace_back(&data[6*(1<<14) + (ii*(1<<4)+i)*(1<<3)*(1<<4) + index]);
    }
    frame->_cpointers.emplace_back(_cpointers);
  }
  for(ii = 0; ii < (1<<3); ii++) {
    vector<float*> __cpointers;
    for(int i = 0; i < 1<<4; i++) {
      __cpointers.emplace_back(&data[7*(1<<14) + (ii*(1<<4)+i)*(1<<3)*(1<<4) + index]);
    }
    frame->__cpointers.emplace_back(__cpointers);
  }
  for(ii = 0; ii < (1<<3); ii++) {
    vector<float*> _dpointers;
    for(int i = 0; i < (1<<4); i++) {
      _dpointers.emplace_back(&data[8*(1<<14) + (ii*(1<<4)+i)*(1<<3)*(1<<4) + index]);
    }
    frame->_dpointers.emplace_back(_dpointers);
  }
  for(ii = 0; ii < (1<<3); ii++) {
    vector<float*> __dpointers;
    for(int i = 0; i < 1<<4; i++) {
      __dpointers.emplace_back(&data[9*(1<<14) + (ii*(1<<4)+i)*(1<<3)*(1<<4) + index]);
    }
    frame->__dpointers.emplace_back(__dpointers);
  }
  for(ii = 0; ii < (1<<3); ii++) {
    vector<float*> _epointers;
    for(int i = 0; i < 1<<4; i++) {
      _epointers.emplace_back(&data[10*(1<<14) + (ii*(1<<4)+i)*(1<<3)*(1<<4) + index]);
    }
    frame->_epointers.emplace_back(_epointers);
  }
  for(ii = 0; ii < (1<<3); ii++) {
    vector<float*> __epointers;
    for(int i = 0; i < 1<<4; i++) {
      __epointers.emplace_back(&data[11*(1<<14) + (ii*(1<<4)+i)*(1<<3)*(1<<4) + index]);
    }
    frame->__epointers.emplace_back(__epointers);
  }
  for(ii = 0; ii < (1<<3); ii++) {
    vector<float*> _fpointers;
    for(int i = 0; i < 1<<4; i++) {
      _fpointers.emplace_back(&data[12*(1<<14) + (ii*(1<<4)+i)*(1<<3)*(1<<4) + index]);
    }
    frame->_fpointers.emplace_back(_fpointers);
  }
  for(ii = 0; ii < (1<<3); ii++) {
    vector<float*> __fpointers;
    for(int i = 0; i < 1<<4; i++) {
      __fpointers.emplace_back(&data[13*(1<<14) + (ii*(1<<4)+i)*(1<<3)*(1<<4) + index]);
    }
    frame->__fpointers.emplace_back(__fpointers);
  }
  for(ii = 0; ii < (1<<3); ii++) {
    vector<float*> _gpointers;
    for(int i = 0; i < 1<<4; i++) {
      _gpointers.emplace_back(&data[14*(1<<14) + (ii*(1<<4)+i)*(1<<3)*(1<<4) + index]);
    }
    frame->_gpointers.emplace_back(_gpointers);
  }
  for(ii = 0; ii < (1<<3); ii++) {
    vector<float*> __gpointers;
    for(int i = 0; i < 1<<4; i++) {
      __gpointers.emplace_back(&data[15*(1<<14) + (ii*(1<<4)+i)*(1<<3)*(1<<4) + index]);
    }
    frame->__gpointers.emplace_back(__gpointers);
  }
}

void execute_task_for_sum(Frame * frame, const int m, int p)
{
  #pragma omp single nowait
  {
    for(int i = p; i < 8 + p; i++) {
      #pragma omp task
      {
        calculate_vector_sum(frame, m, p/8*(1<<4) + i-p);
      }
    }
  }
}

void execute_section_for_frame(Frame * frame, const int m)
{
  // const double t3 = omp_get_wtime();

  #pragma omp parallel sections
  {
    #pragma omp section
    {
      execute_task_for_sum(frame, m, 0);
    }
  }

  #pragma omp taskwait
  {
    calculate_row_sum(frame, m);
    calculate_odd_row_sum(frame, m);
    // calculate_next_odd_row_sum(frame, m);
    // calculate_next2_odd_row_sum(frame, m);
    // const double t4 = omp_get_wtime();
    // printf("Task Time: %f\t", t4-t3);
  }
}

/**
 * depth
 * (1<<4) objects, (1<<3) vectors, (1<<8) frames
 */
void initialise_frames(vector<Frame*>& frames, float * data, int num_frames)
{
  for(int i = 0; i < num_frames; i++) {
    Frame * frame = new Frame();
    load_frame(frame, data, i*(1UL<<18));
    frames[i] = frame;
  }  
}

void filter(const long n, const long m, float *data, const float threshold, std::vector<long> &result_row_ind) {

  int num_frames = 1<<8;
  vector<Frame*> frames(num_frames);
  initialise_frames(frames, data, num_frames);

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
	
	// n = 1<<15, m = 1<<18, N = 1<<4, M = 1<<10, factor = 1<<4
  FrameParams frameParams (1<<4, 1<<4);

  // execution (sum) of 18 members
  for(int i = 0; i < frames.size(); i++) {
    execute_section_for_frame(frames[i], m);
  }

  //sort the values stored in the vector
  std::sort(result_row_ind.begin(),
            result_row_ind.end());
}