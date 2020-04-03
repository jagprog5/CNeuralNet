#ifndef PRINT_REDUCER_H
#define PRINT_REDUCER_H
#if defined(__CYGWIN__)
#define prt_redu(iter, amount, code) if ((iter) % (amount) == 0) { code }
#else
#define prt_redu(iter, amount, code) code
#endif
#endif