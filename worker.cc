#include <vector>
#include <algorithm>
#include <iostream>
#include <omp.h>

using namespace std;

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

struct Frame
{
  vector<float*> _pointers;
  vector<float*> __pointers;
  vector<float> sum;
};

//helper function , refer instructions on the lab page
void append_vec(std::vector<long> &v1, std::vector<long> &v2) {
  v1.insert(v1.end(),v2.begin(),v2.end());
}

void calculate_vector_sum(Frame * frame, const int m, long p) {
  #pragma ivdep
  #pragma omp simd
  #pragma vector nontemporal
  #pragma vector aligned
  for(int aa = 0; aa < (1<<8); aa++) {
    #pragma omp simd
    for(int j = 1; j <= (1<<16); j++) {
      frame->_pointers[0][p + j-1] += frame->_pointers[aa][p + j-1];
    }
  }
}

void calculate_row_sum(Frame * frame, const int m, int p) {
  #pragma ivdep
  #pragma omp simd
  #pragma vector nontemporal
  #pragma vector aligned
  for(long i = 1; i <= (1<<5); i++) {
    #pragma ivdep
    #pragma omp simd
    for(long j = 1; j < (1<<11); j+=2) {
      ptrdiff_t offset = (i-1)*(1<<11) + j-1 + p;
      frame->_pointers[0][offset] += frame->_pointers[0][offset+1];
    }
  }
}

void calculate_odd_row_sum(Frame * frame, const int m, int p) {
  #pragma ivdep
  #pragma omp simd
  #pragma vector nontemporal
  #pragma vector aligned
  for(long i = 1; i <= (1<<5); i++) {
    #pragma ivdep
    #pragma omp simd
    for(long j = 1; j < (1<<11); j+=4) {
      ptrdiff_t offset = (i-1)*(1<<11) + j-1 + p;
      frame->_pointers[0][offset] += frame->_pointers[0][offset+2];
    }
  }
}

void calculate_next_odd_row_sum(Frame * frame, const int m, int p) {
  #pragma ivdep
  #pragma omp simd
  #pragma vector nontemporal
  #pragma vector aligned
  for(long i = 1; i <= (1<<5); i++) {
    #pragma ivdep
    #pragma omp simd
    for(long j = 1; j < (1<<11); j+=8) {
      ptrdiff_t offset = (i-1)*(1<<11) + j-1 + p;
      frame->_pointers[0][offset] += frame->_pointers[0][offset+4];
    }
  }
}

void calculate_next2_odd_row_sum(Frame * frame, const int m, int p) {
  #pragma ivdep
  #pragma omp simd
  #pragma vector nontemporal
  #pragma vector aligned
  for(long i = 1; i <= (1<<5); i++) {
    #pragma ivdep
    #pragma omp simd
    for(long j = 1; j < (1<<11); j+=16) {
      ptrdiff_t offset = (i-1)*(1<<11) + j-1 + p;
      frame->_pointers[0][offset] += frame->_pointers[0][offset+8];
    }
  }
}

void calculate_next3_odd_row_sum(Frame * frame, const int m, int p) {
  #pragma ivdep
  #pragma omp simd
  #pragma vector nontemporal
  #pragma vector aligned
  for(long i = 1; i <= (1<<5); i++) {
    #pragma ivdep
    #pragma omp simd
    for(long j = 1; j < (1<<11); j+=32) {
      ptrdiff_t offset = (i-1)*(1<<11) + j-1 + p;
      frame->_pointers[0][offset] += frame->_pointers[0][offset+16];
    }
  }
}

void calculate_next4_odd_row_sum(Frame * frame, const int m, int p) {
  #pragma ivdep
  #pragma omp simd
  #pragma vector nontemporal
  #pragma vector aligned
  for(long i = 1; i <= (1<<5); i++) {
    #pragma ivdep
    #pragma omp simd
    for(long j = 1; j < (1<<11); j+=64) {
      ptrdiff_t offset = (i-1)*(1<<11) + j-1 + p;
      frame->_pointers[0][offset] += frame->_pointers[0][offset+32];
    }
  }
}

void calculate_next5_odd_row_sum(Frame * frame, const int m, int p) {
  #pragma ivdep
  #pragma omp simd
  #pragma vector nontemporal
  #pragma vector aligned
  for(long i = 1; i <= (1<<5); i++) {
    #pragma ivdep
    #pragma omp simd
    for(long j = 1; j < (1<<11); j+=128) {
      ptrdiff_t offset = (i-1)*(1<<11) + j-1 + p;
      frame->_pointers[0][offset] += frame->_pointers[0][offset+64];
    }
  }
}

void calculate_next6_odd_row_sum(Frame * frame, const int m, int p) {
  #pragma ivdep
  #pragma omp simd
  #pragma vector nontemporal
  #pragma vector aligned
  for(long i = 1; i <= (1<<5); i++) {
    #pragma ivdep
    #pragma omp simd
    for(long j = 1; j < (1<<11); j+=256) {
      ptrdiff_t offset = (i-1)*(1<<11) + j-1 + p;
      frame->_pointers[0][offset] += frame->_pointers[0][offset+128];
    }
  }
}

void calculate_next7_odd_row_sum(Frame * frame, const int m, int p) {
  #pragma ivdep
  #pragma omp simd
  #pragma vector nontemporal
  #pragma vector aligned
  for(long i = 1; i <= (1<<5); i++) {
    #pragma ivdep
    #pragma omp simd
    for(long j = 1; j < (1<<11); j+=512) {
      ptrdiff_t offset = (i-1)*(1<<11) + j-1 + p;
      frame->_pointers[0][offset] += frame->_pointers[0][offset+256];
    }
  }
}

void calculate_next8_odd_row_sum(Frame * frame, const int m, int p) {
  #pragma ivdep
  #pragma omp simd
  #pragma vector nontemporal
  #pragma vector aligned
  for(long i = 1; i <= (1<<5); i++) {
    #pragma ivdep
    #pragma omp simd
    for(long j = 1; j < (1<<11); j+=1024) {
      ptrdiff_t offset = (i-1)*(1<<11) + j-1 + p;
      frame->_pointers[0][offset] += frame->_pointers[0][offset+512];
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

void load_frame(Frame * frame, float * data) {
  for(int i = 0; i < (1<<8); i++) {
    frame->_pointers.emplace_back(&data[i*(1<<11)*(1<<11)]);
  }
}

void filter(const long n, const long m, float *data, const float threshold, std::vector<long> &result_row_ind) {
  
  Frame * frame = new Frame();

  // const double t0 = omp_get_wtime();
  load_frame(frame, data);
  // const double t1 = omp_get_wtime();
  // printf("Time: %f\n", t1-t0);

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
  
  const double t0 = omp_get_wtime();

  #pragma omp parallel for
  for(int ii = 0; ii < 4; ii++) {
    for(int i = 0; i < 16; i++) {
      #pragma omp task
      {
        calculate_vector_sum(frame, m, (ii*16+i)*(1<<16));
        calculate_row_sum(frame, m, (ii*16+i)*(1<<16));
        calculate_odd_row_sum(frame, m, (ii*16+i)*(1<<16));
        calculate_next_odd_row_sum(frame, m, (ii*16+i)*(1<<16));
        calculate_next2_odd_row_sum(frame, m, (ii*16+i)*(1<<16));
        calculate_next3_odd_row_sum(frame, m, (ii*16+i)*(1<<16));
        calculate_next4_odd_row_sum(frame, m, (ii*16+i)*(1<<16));
        calculate_next5_odd_row_sum(frame, m, (ii*16+i)*(1<<16));
        calculate_next6_odd_row_sum(frame, m, (ii*16+i)*(1<<16));
        calculate_next7_odd_row_sum(frame, m, (ii*16+i)*(1<<16));
        calculate_next8_odd_row_sum(frame, m, (ii*16+i)*(1<<16));
      }
    }
  }

  #pragma omp taskwait
  const double t1 = omp_get_wtime();
  
  printf("Task Time: %f\t", t1-t0);

  //sort the values stored in the vector
  std::sort(result_row_ind.begin(),
            result_row_ind.end());
}