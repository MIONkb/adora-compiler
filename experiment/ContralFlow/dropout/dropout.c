#include <string.h>
#include <stdio.h>

int dropout(int i[1024], int sel_ptr[128]) {//int o[1024], 
  int buf_sel =0, sel =0;
  int value =0;
  for (unsigned int k = 0; k < 1024; ++k) {
    if (!(k % 8))
      buf_sel = sel_ptr[k >> 3];
    sel = buf_sel & 0x01;
    // o[k] = sel ? (i[k] * 2) : 0;
    value += sel ? (i[k] * 2) : 0;
    buf_sel >>= 1;
  }
  return value;
}

