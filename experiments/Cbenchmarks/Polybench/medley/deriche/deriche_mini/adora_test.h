#ifndef _CGRVTEST_H
#define _CGRVTEST_H

// #include "include/ISA.h"

#  ifdef ROCKET_TARGET
// int fprintf(const char* fmt, ...);
int printf(const char* fmt, ...);
void free(void* x);
#  endif

#endif //_CGRVTEST_H