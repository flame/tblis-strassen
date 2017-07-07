#ifndef _TCI_H_
#define _TCI_H_

#if defined(__MIC__)
#define TCI_ARCH_MIC 1
#elif defined(__ia64) || defined(__itanium__) || defined(_M_IA64)
#define TCI_ARCH_IA64 1
#elif defined(__x86_64__) || defined(_M_X64)
#define TCI_ARCH_X64 1
#elif defined(__i386) || defined(_M_IX86)
#define TCI_ARCH_X86 1
#elif defined(__aarch64__)
#define TCI_ARCH_ARM64 1
#elif defined(__arm__) || defined(_M_ARM)
#define TCI_ARCH_ARM32 1
#elif defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
#define TCI_ARCH_PPC64 1
#elif defined(__powerpc__) || defined(__ppc__) || defined(__PPC__)
#define TCI_ARCH_PPC32 1
#elif defined(__bgq__)
#define TCI_ARCH_BGQ 1
#elif defined(__sparc)
#define TCI_ARCH_SPARC 1
#elif defined(__mips)
#define TCI_ARCH_MIPS 1
#else
#error "Unknown architecture"
#endif

#define TCI_USE_ATOMIC_SPINLOCK  1
#define TCI_USE_OMP_LOCK         0
#define TCI_USE_OPENMP_THREADS   1
#define TCI_USE_OSX_SPINLOCK     0
#define TCI_USE_PTHREADS_THREADS 0
#define TCI_USE_PTHREAD_BARRIER  0
#define TCI_USE_PTHREAD_MUTEX    0
#define TCI_USE_PTHREAD_SPINLOCK 0
#define TCI_USE_SPIN_BARRIER     1

#define TCI_MIN(x,y) ((y)<(x)?(y):(x))
#define TCI_MAX(x,y) ((x)<(y)?(y):(x))

#include <stdint.h>
#include <errno.h>

#ifdef __cplusplus
#define TCI_INLINE inline
#else
#define TCI_INLINE static inline
#include <stdbool.h>
#endif

#if TCI_ARCH_MIC
#include <immintrin.h>
#endif

#if TCI_ARCH_X86 || TCI_ARCH_X64
#include <xmmintrin.h>
#endif

#if TCI_USE_OSX_SPINLOCK
#include <libkern/OSAtomic.h>
#endif

#if TCI_USE_PTHREAD_SPINLOCK || \
    TCI_USE_PTHREAD_MUTEX || \
    TCI_USE_PTHREAD_BARRIER
#include <pthread.h>
#endif

#if TCI_USE_OMP_LOCK
#include <omp.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if TCI_ARCH_MIC

TCI_INLINE void tci_yield()
{
    _mm_delay(32);
}

#elif TCI_ARCH_X86 || TCI_ARCH_X64

TCI_INLINE void tci_yield()
{
    //_mm_pause();
    __asm__ __volatile__ ("pause");
}

#else

TCI_INLINE void tci_yield() {}

#endif

#if TCI_USE_OSX_SPINLOCK

typedef OSSpinLock tci_mutex_t;

#elif TCI_USE_PTHREAD_SPINLOCK

typedef pthread_spinlock_t tci_mutex_t;

#elif TCI_USE_ATOMIC_SPINLOCK

typedef int tci_mutex_t;

#elif TCI_USE_OMP_LOCK

typedef omp_lock_t tci_mutex_t;

#elif TCI_USE_PTHREAD_MUTEX

typedef pthread_mutex_t tci_mutex_t;

#endif

int tci_mutex_init(tci_mutex_t* mutex);

int tci_mutex_destroy(tci_mutex_t* mutex);

int tci_mutex_lock(tci_mutex_t* mutex);

int tci_mutex_trylock(tci_mutex_t* mutex);

int tci_mutex_unlock(tci_mutex_t* mutex);

#if TCI_USE_PTHREAD_BARRIER

typedef struct tci_barrier_node_s
{
    struct tci_barrier_node_s* parent;
    pthread_barrier_t barrier;
} tci_barrier_node_t;

#elif TCI_USE_SPIN_BARRIER

typedef struct tci_barrier_node_s
{
    struct tci_barrier_node_s* parent;
    unsigned nchildren;
    volatile unsigned step;
    volatile unsigned nwaiting;
} tci_barrier_node_t;

#endif

int tci_barrier_node_init(tci_barrier_node_t* barrier,
                          tci_barrier_node_t* parent,
                          unsigned nchildren);

int tci_barrier_node_destroy(tci_barrier_node_t* barrier);

int tci_barrier_node_wait(tci_barrier_node_t* barrier);

typedef struct
{
    union
    {
        tci_barrier_node_t* array;
        tci_barrier_node_t single;
    } barrier;
    unsigned nthread;
    unsigned group_size;
    int is_tree;
} tci_barrier_t;

int tci_barrier_is_tree(tci_barrier_t* barrier);

int tci_barrier_init(tci_barrier_t* barrier,
                     unsigned nthread, unsigned group_size);

int tci_barrier_destroy(tci_barrier_t* barrier);

int tci_barrier_wait(tci_barrier_t* barrier, unsigned tid);

typedef struct
{
    tci_barrier_t barrier;
    void* buffer;
    volatile unsigned refcount;
} tci_context_t;

int tci_context_init(tci_context_t** context,
                     unsigned nthread, unsigned group_size);

int tci_context_attach(tci_context_t* context);

int tci_context_detach(tci_context_t* context);

int tci_context_barrier(tci_context_t* context, unsigned tid);

int tci_context_send(tci_context_t* context, unsigned tid, void* object);

int tci_context_send_nowait(tci_context_t* context,
                            unsigned tid,void* object);

int tci_context_receive(tci_context_t* context, unsigned tid, void** object);

int tci_context_receive_nowait(tci_context_t* context,
                               unsigned tid, void** object);

typedef struct
{
    tci_context_t* context;
    unsigned nthread;
    unsigned tid;
    unsigned ngang;
    unsigned gid;
} tci_comm_t;

typedef enum
{
    TCI_EVENLY         = (1u<<1),
    TCI_CYCLIC         = (2u<<1),
    TCI_BLOCK_CYCLIC   = (3u<<1),
    TCI_BLOCKED        = (4u<<1),
    TCI_NO_CONTEXT     =    0x1u
} tci_gang_t;

extern const tci_comm_t* const tci_single;

int tci_comm_init_single(tci_comm_t* comm);

int tci_comm_init(tci_comm_t* comm, tci_context_t* context,
                  unsigned nthread, unsigned tid, unsigned ngang, unsigned gid);

int tci_comm_destroy(tci_comm_t* comm);

int tci_comm_is_master(const tci_comm_t* comm);

int tci_comm_barrier(tci_comm_t* comm);

int tci_comm_bcast(tci_comm_t* comm, void** object, unsigned root);

int tci_comm_bcast_nowait(tci_comm_t* comm, void** object, unsigned root);

int tci_comm_gang(tci_comm_t* parent, tci_comm_t* child,
                  unsigned type, unsigned n, unsigned bs);

void tci_distribute(unsigned n, unsigned idx, uint64_t range,
                    uint64_t granularity, uint64_t* first, uint64_t* last,
                    uint64_t* max);

void tci_comm_distribute_over_gangs(tci_comm_t* comm, uint64_t range,
                                    uint64_t granularity, uint64_t* first,
                                    uint64_t* last, uint64_t* max);

void tci_comm_distribute_over_threads(tci_comm_t* comm, uint64_t range,
                                      uint64_t granularity, uint64_t* first,
                                      uint64_t* last, uint64_t* max);

void tci_comm_distribute_over_gangs_2d(tci_comm_t* comm,
    uint64_t range_m, uint64_t range_n,
    uint64_t granularity_m, uint64_t granularity_n,
    uint64_t* first_m, uint64_t* last_m, uint64_t* max_m,
    uint64_t* first_n, uint64_t* last_n, uint64_t* max_n);

void tci_comm_distribute_over_threads_2d(tci_comm_t* comm,
    uint64_t range_m, uint64_t range_n,
    uint64_t granularity_m, uint64_t granularity_n,
    uint64_t* first_m, uint64_t* last_m, uint64_t* max_m,
    uint64_t* first_n, uint64_t* last_n, uint64_t* max_n);

typedef void (*tci_thread_func_t)(tci_comm_t*, void*);

int tci_parallelize(tci_thread_func_t func, void* payload,
                    unsigned nthread, unsigned arity);

typedef struct
{
    unsigned n;
    unsigned sqrt_n;
    unsigned f;
} tci_prime_factors_t;

void tci_prime_factorization(unsigned n, tci_prime_factors_t* factors);

unsigned tci_next_prime_factor(tci_prime_factors_t* factors);

void tci_partition_2x2(unsigned nthread, uint64_t work1, uint64_t work2,
                       unsigned* nt1, unsigned* nt2);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

#include <system_error>
#include <tuple>
#include <utility>

namespace tci
{

class mutex
{
    public:
        mutex()
        {
            int ret = tci_mutex_init(&_lock);
            if (ret != 0) throw std::system_error(ret, std::system_category());
        }

        ~mutex() noexcept(false)
        {
            int ret = tci_mutex_destroy(&_lock);
            if (ret != 0) throw std::system_error(ret, std::system_category());
        }

        mutex(const mutex&) = delete;

        mutex(mutex&&) = default;

        mutex& operator=(const mutex&) = delete;

        mutex& operator=(mutex&&) = default;

        void lock()
        {
            int ret = tci_mutex_lock(&_lock);
            if (ret != 0) throw std::system_error(ret, std::system_category());
        }

        bool try_lock()
        {
            int ret = tci_mutex_trylock(&_lock);
            if (ret == EBUSY) return false;
            if (ret != 0) throw std::system_error(ret, std::system_category());
            return true;
        }

        void unlock()
        {
            int ret = tci_mutex_unlock(&_lock);
            if (ret != 0) throw std::system_error(ret, std::system_category());
        }
        
        operator tci_mutex_t*() { return &_lock; }
        
        operator const tci_mutex_t*() const { return &_lock; }

    protected:
        tci_mutex_t _lock;
};

class barrier
{
    public:
        barrier(unsigned nthread, unsigned group_size=0)
        {
            int ret = tci_barrier_init(&_barrier, nthread, group_size);
            if (ret != 0) throw std::system_error(ret, std::system_category());
        }

        ~barrier() noexcept(false)
        {
            int ret = tci_barrier_destroy(&_barrier);
            if (ret != 0) throw std::system_error(ret, std::system_category());
        }

        barrier(const barrier&) = delete;

        barrier(barrier&) = default;

        barrier& operator=(const barrier&) = delete;

        barrier& operator=(barrier&) = default;

        void wait(unsigned tid)
        {
            int ret = tci_barrier_wait(&_barrier, tid);
            if (ret != 0) throw std::system_error(ret, std::system_category());
        }

        unsigned num_threads() const
        {
            return _barrier.nthread;
        }

        unsigned group_size() const
        {
            return _barrier.group_size;
        }
        
        operator tci_barrier_t*() { return &_barrier; }
        
        operator const tci_barrier_t*() const { return &_barrier; }

    protected:
        tci_barrier_t _barrier;
};

class communicator
{
    public:
        communicator()
        {
            tci_comm_init_single(*this);
        }

        ~communicator()
        {
            tci_comm_destroy(*this);
        }

        communicator(const communicator&) = delete;

        communicator(communicator&& other)
        : _comm(other._comm)
        {
            other._comm.context = nullptr;
        }

        communicator& operator=(const communicator&) = delete;

        communicator& operator=(communicator&& other)
        {
            std::swap(_comm, other._comm);
            return *this;
        }

        bool master() const
        {
            return tci_comm_is_master(*this);
        }

        void barrier() const
        {
            int ret = tci_comm_barrier(*this);
            if (ret != 0) throw std::system_error(ret, std::system_category());
        }

        unsigned num_threads() const
        {
            return _comm.nthread;
        }

        unsigned thread_num() const
        {
            return _comm.tid;
        }

        unsigned num_gangs() const
        {
            return _comm.ngang;
        }

        unsigned gang_num() const
        {
            return _comm.gid;
        }

        template <typename T>
        void broadcast(T*& object, unsigned root=0) const
        {
            tci_comm_bcast(*this, reinterpret_cast<void**>(&object), root);
        }

        template <typename T>
        void broadcast_nowait(T*& object, unsigned root=0) const
        {
            tci_comm_bcast_nowait(*this, reinterpret_cast<void**>(&object), root);
        }

        communicator gang(unsigned type, unsigned n, unsigned bs=0) const
        {
            communicator child;
            int ret = tci_comm_gang(*this, &child._comm, type, n, bs);
            if (ret != 0) throw std::system_error(ret, std::system_category());
            return child;
        }

        std::tuple<uint64_t,uint64_t,uint64_t>
        distribute_over_gangs(uint64_t range, uint64_t granularity=1) const
        {
            uint64_t first, last, max;
            tci_comm_distribute_over_gangs(*this, range, granularity,
                                           &first, &last, &max);
            return std::make_tuple(first, last, max);
        }

        std::tuple<uint64_t,uint64_t,uint64_t>
        distribute_over_threads(uint64_t range, uint64_t granularity=1) const
        {
            uint64_t first, last, max;
            tci_comm_distribute_over_threads(*this, range, granularity,
                                             &first, &last, &max);
            return std::make_tuple(first, last, max);
        }

        std::tuple<uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t>
        distribute_over_gangs_2d(uint64_t range_m, uint64_t range_n,
                                 uint64_t granularity_m=1,
                                 uint64_t granularity_n=1) const
        {
            uint64_t first_m, last_m, max_m, first_n, last_n, max_n;
            tci_comm_distribute_over_gangs_2d(*this, range_m, range_n,
                                              granularity_m, granularity_n,
                                              &first_m, &last_m, &max_m,
                                              &first_n, &last_n, &max_n);
            return std::make_tuple(first_m, last_m, max_m,
                                   first_n, last_n, max_n);
        }

        std::tuple<uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t>
        distribute_over_threads_2d(uint64_t range_m, uint64_t range_n,
                                   uint64_t granularity_m=1,
                                   uint64_t granularity_n=1) const
        {
            uint64_t first_m, last_m, max_m, first_n, last_n, max_n;
            tci_comm_distribute_over_threads_2d(*this, range_m, range_n,
                                                granularity_m, granularity_n,
                                                &first_m, &last_m, &max_m,
                                                &first_n, &last_n, &max_n);
            return std::make_tuple(first_m, last_m, max_m,
                                   first_n, last_n, max_n);
        }
        
        operator tci_comm_t*() const { return const_cast<tci_comm_t*>(&_comm); }

    protected:
        tci_comm_t _comm;
};

namespace detail
{

template <typename Body>
void body_wrapper(tci_comm_t* comm, void* data)
{
    Body& body = *static_cast<Body*>(data);
    body(*reinterpret_cast<communicator*>(comm));
}

}

template <typename Body>
void parallelize(Body&& body, unsigned nthread, unsigned arity=0)
{
    // The first line is necessary to trigger template instantiation
    tci_thread_func_t func = detail::body_wrapper<Body>;
    tci_parallelize(func, const_cast<void*>(static_cast<const void*>(&body)),
                    nthread, arity);
}

class prime_factorization
{
    public:
        prime_factorization(unsigned n)
        {
            tci_prime_factorization(n, &_factors);
        }

        unsigned next()
        {
            return tci_next_prime_factor(&_factors);
        }

    protected:
        tci_prime_factors_t _factors;
};

inline std::pair<unsigned,unsigned>
partition_2x2(unsigned nthreads, uint64_t work1, uint64_t work2)
{
    unsigned nt1, nt2;
    tci_partition_2x2(nthreads, work1, work2, &nt1, &nt2);
    return std::make_pair(nt1, nt2);
}

}

#endif

#endif
