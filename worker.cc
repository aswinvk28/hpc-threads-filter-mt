#include <vector>
#include <algorithm>
#include <iostream>
#include <omp.h>

using namespace std;

/**
 * breadth
 * (1<<8) section offsets, (1<<4) task offsets, (1<<4) offsets, (1<<2) pointers, 4 sum offsets
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
 * (1<<4) objects, (1<<6) vectors, (1<<5) frames = (1<<15) n
 */
struct Frame
{
  vector< vector<float*> > _pointers = vector< vector<float*> > ( 64 , vector<float*>(4) );
  vector< vector<float*> > __pointers = vector< vector<float*> > ( 64, vector<float*>(4) );
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
  #pragma omp parallel for
  for(int i = 0; i < (1<<6); i++) {
    #pragma omp simd
    for(int aa = 0; aa < (1<<2); aa++) {
      #pragma ivdep
      #pragma omp simd
      for(int j = 1; j <= (1<<4); j++) {
        frame->_pointers[i][0][j-1] += frame->_pointers[i][aa][p*(1<<4) + j-1];
        frame->__pointers[i][0][j-1] += frame->__pointers[i][aa][p*(1<<4) + j-1];
      }
    }
  }
}

/**
 * trawl for n (nouns)
 * depth-first search
 * high performance tuning parameters as lower time to sum up the dependencies
 */
void calculate_row_sum(std::vector<Frame*> frames, const int m) {
  #pragma omp parallel for
  for(int a = 0; a < frames.size(); a++) {
    #pragma ivdep
    #pragma omp simd
    #pragma vector nontemporal
    #pragma vector aligned
    for(int ii = 0; ii < (1<<6); ii++) {
      #pragma ivdep
      #pragma omp simd
      for(int j = 1; j < (1<<4); j+=2) {
        frames[a]->_pointers[ii][0][j-1] += frames[a]->_pointers[ii][0][j];
        frames[a]->__pointers[ii][0][j-1] += frames[a]->__pointers[ii][0][j];
      }
    }
  }
}

/**
 * depth
 */
void calculate_odd_row_sum(std::vector<Frame*> frames, const int m) {
  #pragma omp parallel for
  for(int a = 0; a < frames.size(); a++) {
    #pragma ivdep
    #pragma omp simd
    #pragma vector nontemporal
    #pragma vector aligned
    for(int ii = 0; ii < (1<<6); ii++) {
      #pragma ivdep
      #pragma omp simd
      for(int j = 1; j < (1<<4); j+=4) {
        frames[a]->_pointers[ii][0][j-1] += frames[a]->_pointers[ii][0][j+1];
        frames[a]->__pointers[ii][0][j-1] += frames[a]->__pointers[ii][0][j+1];
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

void execute_task_for_sum(Frame * frame, const int m, int p)
{
  for(int i = p; i < 16 + p; i++) {
    #pragma omp task
    {
      calculate_vector_sum(frame, m, i);
    }
  }
}

void execute_section_for_frame(Frame * frame, const int m)
{
  #pragma omp single nowait
  {
    for(int i = 0; i <= 3; i++) {
      #pragma omp task
      {
        execute_task_for_sum(frame, m, i*16);
      }
    }
  }
}

/**
 * breadth
 * (1<<4) section offsets, (1<<4) task offsets, (1<<4) offsets, (1<<6) pointers
 * depth
 * (1<<2) pointers, (1<<3) vectors
 * (1<<3) offset_equator1, (1<<4) offset_equator2
 */
void initialise_frames(vector<Frame*>& frames, float * data, int num_frames)
{
  #pragma omp parallel for
  for(int j = 0; j < num_frames; j++) {
    Frame * frame = new Frame();
    frames[j] = frame;
    #pragma omp parallel sections
    {
      #pragma omp section
      {
        #pragma omp simd
        for(int ii = 0; ii < (1<<6); ii++) {
          #pragma omp simd
          for(int i = 0; i < 1<<2; i++) {
            frames[j]->_pointers[ii][i] = &data[(ii*(1<<2)+i)*(1<<5)*(1<<5) + j*(1UL<<22)];
          }
        }
      }
      #pragma omp section
      {
        #pragma omp simd
        for(int ii = 0; ii < (1<<6); ii++) {
          #pragma omp simd
          for(int i = 0; i < 1<<2; i++) {
            frames[j]->__pointers[ii][i] = &data[(1<<18) + (ii*(1<<2)+i)*(1<<5)*(1<<5) + j*(1UL<<22)];
          }
        }
      }
    }
  }  
}

void filter(const long n, const long m, float *data, const float threshold, std::vector<long> &result_row_ind) {

  int num_frames = 1<<8;
  vector<Frame*> frames(num_frames);
  const double _t = omp_get_wtime();
  initialise_frames(frames, data, num_frames);
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
	
	// n = 1<<15, m = 1<<18, N = 1<<4, M = 1<<10, factor = 1<<4
  FrameParams frameParams (1<<4, 1<<4);

  const double t3 = omp_get_wtime();

  // execution (sum) of 18 members
  // sections created to test
  #pragma omp parallel
  {
    #pragma omp sections nowait
    {
      #pragma omp section
      {
        for(int i = 0; i < frames.size()/(1<<4); i++) {
          #pragma omp task
          {
            execute_section_for_frame(frames[i], m);
          }
        }
      }
      #pragma omp section
      {
        for(int i = frames.size()/(1<<4); i < 2*frames.size()/(1<<4); i++) {
          #pragma omp task
          {
            execute_section_for_frame(frames[i], m);
          }
        }
      }
      #pragma omp section
      {
        for(int i = 2*frames.size()/(1<<4); i < 3*frames.size()/(1<<4); i++) {
          #pragma omp task
          {
            execute_section_for_frame(frames[i], m);
          }
        }
      }
      #pragma omp section
      {
        for(int i = 3*frames.size()/(1<<4); i < 4*frames.size()/(1<<4); i++) {
          #pragma omp task
          {
            execute_section_for_frame(frames[i], m);
          }
        }
      }
      #pragma omp section
      {
        for(int i = 4*frames.size()/(1<<4); i < 5*frames.size()/(1<<4); i++) {
          #pragma omp task
          {
            execute_section_for_frame(frames[i], m);
          }
        }
      }
      #pragma omp section
      {
        for(int i = 5*frames.size()/(1<<4); i < 6*frames.size()/(1<<4); i++) {
          #pragma omp task
          {
            execute_section_for_frame(frames[i], m);
          }
        }
      }
      #pragma omp section
      {
        for(int i = 6*frames.size()/(1<<4); i < 7*frames.size()/(1<<4); i++) {
          #pragma omp task
          {
            execute_section_for_frame(frames[i], m);
          }
        }
      }
      #pragma omp section
      {
        for(int i = 7*frames.size()/(1<<4); i < 8*frames.size()/(1<<4); i++) {
          #pragma omp task
          {
            execute_section_for_frame(frames[i], m);
          }
        }
      }
      #pragma omp section
      {
        for(int i = 8*frames.size()/(1<<4); i < 9*frames.size()/(1<<4); i++) {
          #pragma omp task
          {
            execute_section_for_frame(frames[i], m);
          }
        }
      }
      #pragma omp section
      {
        for(int i = 9*frames.size()/(1<<4); i < 10*frames.size()/(1<<4); i++) {
          #pragma omp task
          {
            execute_section_for_frame(frames[i], m);
          }
        }
      }
      #pragma omp section
      {
        for(int i = 10*frames.size()/(1<<4); i < 11*frames.size()/(1<<4); i++) {
          #pragma omp task
          {
            execute_section_for_frame(frames[i], m);
          }
        }
      }
      #pragma omp section
      {
        for(int i = 11*frames.size()/(1<<4); i < 12*frames.size()/(1<<4); i++) {
          #pragma omp task
          {
            execute_section_for_frame(frames[i], m);
          }
        }
      }
      #pragma omp section
      {
        for(int i = 12*frames.size()/(1<<4); i < 13*frames.size()/(1<<4); i++) {
          #pragma omp task
          {
            execute_section_for_frame(frames[i], m);
          }
        }
      }
      #pragma omp section
      {
        for(int i = 13*frames.size()/(1<<4); i < 14*frames.size()/(1<<4); i++) {
          #pragma omp task
          {
            execute_section_for_frame(frames[i], m);
          }
        }
      }
      #pragma omp section
      {
        for(int i = 14*frames.size()/(1<<4); i < 15*frames.size()/(1<<4); i++) {
          #pragma omp task
          {
            execute_section_for_frame(frames[i], m);
          }
        }
      }
      #pragma omp section
      {
        for(int i = 15*frames.size()/(1<<4); i < frames.size(); i++) {
          #pragma omp task
          {
            execute_section_for_frame(frames[i], m);
          }
        }
      }
    }
  }

  const double t4 = omp_get_wtime();

  printf("Task Time: %f\t", t4-t3);

  #pragma omp taskwait
  {
    calculate_row_sum(frames, m);
    calculate_odd_row_sum(frames, m);
    // calculate_next_odd_row_sum(frame, m);
    // calculate_next2_odd_row_sum(frame, m);
    const double t5 = omp_get_wtime();
    printf("Task Time: %f\t", t5-t4);
  }

  //sort the values stored in the vector
  std::sort(result_row_ind.begin(),
            result_row_ind.end());
}