#include <algorithm>
#include <thread>
#include <functional>
#include <vector>
#include <emmintrin.h>
#include <xmmintrin.h>
#include <math.h>

/// @param[in] nb_elements : size of your for loop
/// @param[in] functor(start, end) :
/// your function processing a sub chunk of the for loop.
/// "start" is the first index to process (included) until the index "end"
/// (excluded)
/// @code
///     for(int i = start; i < end; ++i)
///         computation(i);
/// @endcode
/// @param use_threads : enable / disable threads.
///
///
//https://stackoverflow.com/questions/36246300/parallel-loops-in-c/36246386
static
void parallel_for(unsigned nb_elements,
                  std::function<void (int start, int end)> functor,
                  bool use_threads = true)
{
    // -------
    unsigned nb_threads_hint = std::thread::hardware_concurrency();
    unsigned nb_threads = nb_threads_hint == 0 ? 8 : (nb_threads_hint);

    unsigned batch_size = nb_elements / nb_threads;
    unsigned batch_remainder = nb_elements % nb_threads;

    std::vector< std::thread > my_threads(nb_threads);

    if( use_threads )
    {
        // Multithread execution
        for(unsigned i = 0; i < nb_threads; ++i)
        {
            int start = i * batch_size;
            my_threads[i] = std::thread(functor, start, start+batch_size);
        }
    }
    else
    {
        // Single thread execution (for easy debugging)
        for(unsigned i = 0; i < nb_threads; ++i){
            int start = i * batch_size;
            functor( start, start+batch_size );
        }
    }

    // Deform the elements left
    int start = nb_threads * batch_size;
    functor( start, start+batch_remainder);

    // Wait for the other thread to finish their task
    if( use_threads )
        std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
}


//Optimizing RANSAC with SSE
//http://opticode.ch/blog/ransac-sse/

struct fvec4 {
    __m128 data;

    fvec4() {}
    fvec4(__m128 a) {data = a;}
    fvec4(const fvec4 &a) {data = a.data;}
    fvec4(float a) {data = _mm_set_ps1(a);} //Set all four words with the same value
    //fvec4(float a, float b, float c, float d) { data = _mm_set_ps(a,b,c,d); } //Set four values, address aligned
    fvec4(float a, float b, float c, float d) { data = _mm_set_ps(d,c,b,a); }

    fvec4 operator = (const fvec4 &a) { data = a.data; return *this; }
	fvec4 operator = (const __m128 a) { data = a; return *this; }
	fvec4 operator = (float a) { data = _mm_set_ps1(a); return *this; }

    fvec4 operator += (const fvec4 &a) {  return *this = _mm_add_ps(data,a.data); }
	fvec4 operator -= (const fvec4 &a) {  return *this = _mm_sub_ps(data,a.data); }
	fvec4 operator *= (const fvec4 &a) {  return *this = _mm_mul_ps(data,a.data); }
	fvec4 operator /= (const fvec4 &a) {  return *this = _mm_div_ps(data,a.data); }

    float &operator [] (int i) { return ((float *) &data)[i]; }
	const float &operator [] (int i) const { return ((const float *) &data)[i]; }

    float horizontal_max() const { 
		return std::max( 
			std::max( (*this)[0], (*this)[1] ),
			std::max( (*this)[2], (*this)[3] ));
	}

    float horizontal_sum() const {
		const __m128 t = _mm_add_ps(data, _mm_movehl_ps(data, data));
		float result;
		_mm_store_ss(&result, _mm_add_ss(t, _mm_shuffle_ps(t, t, 1)));
		return result;
	}

	int horizontal_max_index() { 
		int best=0;
		float val = (*this)[0];
		for (int i=1; i<4; i++) {
			if (val < (*this)[i]) {
				val = (*this)[i];
				best = i;
			}
		}
		return best;
	}

};

inline fvec4 operator + (const fvec4 &a, const fvec4 &b) { return fvec4(_mm_add_ps(a.data, b.data)); }
inline fvec4 operator - (const fvec4 &a, const fvec4 &b) { return fvec4(_mm_sub_ps(a.data, b.data)); }
inline fvec4 operator - (const fvec4 &a) { return fvec4(0) - a; }
inline fvec4 operator * (const fvec4 &a, const fvec4 &b) { return fvec4(_mm_mul_ps(a.data, b.data)); }
inline fvec4 operator / (const fvec4 &a, const fvec4 &b) { return fvec4(_mm_div_ps(a.data, b.data)); }
inline fvec4 operator < (const fvec4 &a, const fvec4 &b) { return fvec4(_mm_cmplt_ps(a.data, b.data)); }
inline fvec4 operator > (const fvec4 &a, const fvec4 &b) { return fvec4(_mm_cmpgt_ps(a.data, b.data)); }
inline fvec4 operator <= (const fvec4 &a, const fvec4 &b) { return fvec4(_mm_cmpngt_ps(a.data, b.data)); }
inline fvec4 operator >= (const fvec4 &a, const fvec4 &b) { return fvec4(_mm_cmpnlt_ps(a.data, b.data)); }
inline fvec4 operator == (const fvec4 &a, const fvec4 &b) { return fvec4(_mm_cmpeq_ps(a.data, b.data)); }

inline fvec4 operator & (const fvec4 &a, const fvec4 &b) { return fvec4(_mm_and_ps(a.data, b.data)); }
inline fvec4 operator | (const fvec4 &a, const fvec4 &b) { return fvec4(_mm_or_ps(a.data, b.data)); }
inline fvec4 operator ^ (const fvec4 &a, const fvec4 &b) { return fvec4(_mm_xor_ps(a.data, b.data)); }
inline fvec4 min(const fvec4 &a, const fvec4 &b) { return fvec4(_mm_min_ps(a.data, b.data)); }
inline fvec4 max(const fvec4 &a, const fvec4 &b) { return fvec4(_mm_max_ps(a.data, b.data)); }

inline fvec4 loadu(const float *ptr) {return fvec4(_mm_loadu_ps(ptr));}
inline void storeu(fvec4 *dest, const fvec4 &value) {_mm_storeu_ps((float *) dest, value.data);}

inline fvec4 load(const float *__restrict__ pIn) {return fvec4(_mm_load_ss(pIn));}
inline void store(float *__restrict__ pOut , const fvec4 &pIn) {_mm_store_ss(pOut, pIn.data);}
inline float convert(const fvec4 &pIn) 
{
    float pOut;
    _mm_store_ss(&pOut, pIn.data);
    return pOut;
}

inline fvec4 f_sqrt(const fvec4 &a) {return fvec4(_mm_sqrt_ss(a.data));}
inline fvec4 f_rsqrt(const fvec4 &a) {return fvec4(_mm_rsqrt_ss(a.data));}
inline fvec4 f_RTsqrt(const fvec4 &a) {return fvec4(_mm_mul_ss( a.data, _mm_rsqrt_ss(a.data)));}


inline void sse_rsqrt( float *__restrict__ pOut, float *__restrict__ pIn )
{
   __m128 in = _mm_load_ss(pIn);
   _mm_store_ss( pOut, _mm_rsqrt_ss(in));
   // compiles to movss, movaps, rsqrtss, mulss, movss
}

inline void sse_sqrt( float *__restrict__ pOut, float *__restrict__ pIn )
{
   __m128 in = _mm_load_ss(pIn);
   _mm_store_ss( pOut, _mm_sqrt_ss(in) );
   // compiles to movss, movaps, rsqrtss, mulss, movss
}

inline void SSESqrt_Recip_Times_X( float *__restrict__ pOut, float *__restrict__ pIn )
{
   __m128 in = _mm_load_ss(pIn);
   _mm_store_ss( pOut, _mm_mul_ss( in, _mm_rsqrt_ss(in)));
   // compiles to movss, movaps, rsqrtss, mulss, movss
}

static inline float SqrtFunction( float in )
{  return sqrt(in); }


